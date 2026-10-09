#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
bool has_role(const CaptureGroup& group, runtime::CheckpointRole role) {
    return std::find(group.roles.begin(), group.roles.end(), role) != group.roles.end();
}
} // namespace

CaptureReservation::~CaptureReservation() {
    if (!owner) { return; }
    if (owner->state_store->valid(state) && !owner->state_store->release(state)) {
        std::terminate();
    }
    for (const auto& [handle, role] : points) { owner->checkpoints[handle.index].reserved = false; }
}

bool ProgramImpl::reclaim_capture_reservation(runtime::ContextResourceUsage shortage) {
    for (auto& request : requests) {
        const auto& ticket = request.capture_reservation;
        if (!ticket) { continue; }
        const bool relevant =
            (shortage.state_slots && state_store->device_resident(ticket->state)) ||
            (shortage.host_bytes && ticket->host) ||
            (shortage.main_kv_pages && ticket->main_tail && ticket->main_tail->pages()) ||
            (shortage.backend_kv_pages && ticket->backend_tail && ticket->backend_tail->pages());
        if (relevant) {
            request.capture_reservation.reset();
            return true;
        }
    }
    return false;
}

bool ProgramImpl::reserve_capture_destination(std::uint32_t lane, std::uint32_t frontier) {
    auto& request = requests[lane];
    if (request.capture_reservation) { return request.capture_reservation->frontier == frontier; }
    if (request.next_capture >= request.capture_groups.size()) { return false; }
    const auto& group = request.capture_groups[request.next_capture];
    if (group.frontier != frontier) { return false; }
    auto ticket      = std::make_unique<CaptureReservation>();
    ticket->owner    = this;
    ticket->frontier = frontier;
    ticket->points.reserve(group.roles.size());
    for (const auto role : group.roles) {
        const auto point = reserve_checkpoint();
        if (!point) {
            if (role == runtime::CheckpointRole::SharedPrefix && !ticket->points.empty()) {
                continue;
            }
            return false;
        }
        ticket->points.emplace_back(*point, role);
    }
    auto destination = state_store->reserve_destination();
    if (!destination && host_context_arena) {
        ticket->host = host_context_arena->allocate(state_images->host_layout().image_bytes);
        if (ticket->host) { destination = state_store->reserve_logical_destination(); }
    }
    if (!destination) { return false; }
    ticket->state = *destination;
    const auto shared =
        std::find_if(ticket->points.begin(), ticket->points.end(), [](const auto& point) {
            return point.second == runtime::CheckpointRole::SharedPrefix;
        });
    if (shared != ticket->points.end()) {
        const auto main_tail_pages = frontier % kPagedKVPageSize ? 1U : 0U;
        const auto backend_frontier =
            speculative_backend == SpeculativeBackend::Mtp ? frontier - 1U : frontier;
        const auto backend_tail_pages =
            backend_kv_pages && backend_frontier % kPagedKVPageSize ? 1U : 0U;
        if (main_tail_pages) {
            ticket->main_tail = text_kv_pages->physical_pool().reserve(main_tail_pages);
        }
        if (backend_tail_pages) {
            ticket->backend_tail = backend_kv_pages->physical_pool().reserve(backend_tail_pages);
        }
        if ((main_tail_pages && !ticket->main_tail) ||
            (backend_tail_pages && !ticket->backend_tail)) {
            // The private recovery point has no independent KV snapshot requirement.
            checkpoints[shared->first.index].reserved = false;
            ticket->points.erase(shared);
            ticket->main_tail.reset();
            ticket->backend_tail.reset();
            if (ticket->points.empty()) { return false; }
        }
    }
    request.capture_reservation = std::move(ticket);
    return true;
}

std::optional<CapturePreparation> ProgramImpl::prepare_capture(SequenceHandle handle) {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto lane = ContractAccess::lane(handle).value;
    auto& request   = requests[lane];
    if (!request.base->capture_backing || !request.permit) { return std::nullopt; }
    if (request.recovery) {
        // Recovery owns enough resources to regain old progress and execute new work.
        // Optional saves cannot add a State/COW requirement to that finite guarantee.
        request.capture_reservation.reset();
        while (request.next_capture < request.capture_groups.size() &&
               request.capture_groups[request.next_capture].frontier <=
                   request.permit->main_frontier) {
            ++request.next_capture;
        }
        return std::nullopt;
    }
    if (request.capture_reservation) {
        return CapturePreparation{.frontier = request.capture_reservation->frontier,
                                  .reserved = true};
    }
    const auto cursor =
        request.lifecycle == Lifecycle::Replaying ? request.replay_cursor : request.prefill->cursor;
    while (request.next_capture < request.capture_groups.size()) {
        const auto frontier = request.capture_groups[request.next_capture].frontier;
        if (frontier <= cursor) {
            ++request.next_capture;
            continue;
        }
        if (frontier > request.permit->main_frontier) { return std::nullopt; }
        if (reserve_capture_destination(lane, frontier)) {
            return CapturePreparation{.frontier = frontier, .reserved = true};
        }
        CapturePreparation result{.frontier = frontier};
        if (state_store->device_occupied() == state_store->device_capacity()) {
            result.shortage.state_slots = 1;
            result.host_bytes           = state_images->host_layout().image_bytes;
        }
        const auto& group = request.capture_groups[request.next_capture];
        const bool private_point =
            std::any_of(group.roles.begin(), group.roles.end(),
                        [](auto role) { return role != runtime::CheckpointRole::SharedPrefix; });
        if (!private_point) {
            result.shortage.main_kv_pages =
                frontier % kPagedKVPageSize && !text_kv_pages->physical_pool().available_pages()
                    ? 1U
                    : 0U;
            const auto backend =
                speculative_backend == SpeculativeBackend::Mtp ? frontier - 1U : frontier;
            if (backend_kv_pages) {
                result.shortage.backend_kv_pages =
                    backend % kPagedKVPageSize &&
                            !backend_kv_pages->physical_pool().available_pages()
                        ? 1U
                        : 0U;
            }
        }
        return result;
    }
    return std::nullopt;
}

void ProgramImpl::prepare_capture_boundary(std::uint32_t lane) {
    while (const auto prepared = prepare_capture(sequence_handle(lane))) {
        if (prepared->reserved) { return; }
        // Direct Native callers can omit optional saving; Engine handles admission before
        // advancing. Unavailable optional splits never change required identity boundaries.
        skip_capture(sequence_handle(lane));
    }
}

void ProgramImpl::skip_capture(SequenceHandle handle) {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    auto& request = requests[ContractAccess::lane(handle).value];
    request.capture_reservation.reset();
    if (!request.capture_pending) {
        if (request.next_capture < request.capture_groups.size()) { ++request.next_capture; }
        return;
    }
    request.capture_pending = false;
    ++request.next_capture;
    if (request.lifecycle != Lifecycle::Replaying && request.prefill &&
        request.prefill->cursor == request.prefill->prompt_tokens) {
        request.prefill.reset();
    }
}

bool ProgramImpl::capture_is_input(SequenceHandle handle) const {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto& request = requests[ContractAccess::lane(handle).value];
    if (request.next_capture >= request.capture_groups.size()) { return false; }
    const auto& group = request.capture_groups[request.next_capture];
    return has_role(group, runtime::CheckpointRole::InputReplay) &&
           (request.capture_pending ||
            (request.permit && group.frontier <= request.permit->main_frontier));
}

bool ProgramImpl::start_capture(SequenceHandle handle) {
    if (context_transaction_ || pending_transaction_) { return false; }
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto lane = ContractAccess::lane(handle).value;
    auto& request   = requests[lane];
    auto& state     = sequences[lane];
    if (!request.capture_pending || request.next_capture >= request.capture_groups.size()) {
        return false;
    }
    if (state.state.fork_pending || state.state.read != state.state.write) { return false; }
    const auto frontier = request.capture_groups[request.next_capture].frontier;
    if (frontier != state.text_kv_valid || !state.tail_hidden_valid) { return false; }
    if (!reserve_capture_destination(lane, frontier)) { return false; }
    auto ticket = std::move(request.capture_reservation);
    ContextTransaction tx;
    tx.kind           = ContextOperationKind::Capture;
    tx.lane           = lane;
    tx.epoch          = lane_epochs[lane];
    tx.capture_points = std::move(ticket->points);
    tx.reuse_frontier = frontier;
    tx.backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp ? frontier - 1U : backend_kv_valid(state);
    tx.reserved_state     = ticket->state;
    const bool host_state = ticket->host.has_value();
    ticket->owner         = nullptr;
    context_transaction_.emplace(std::move(tx));
    auto& operation = *context_transaction_;
    try {
        const bool shared =
            std::any_of(operation.capture_points.begin(), operation.capture_points.end(),
                        [](const auto& point) {
                            return point.second == runtime::CheckpointRole::SharedPrefix;
                        });
        if (shared) {
            auto main = text_kv_addresses->create_inactive();
            if (!main) {
                abort_context();
                return false;
            }
            operation.reserved_kv = SequenceKVBundle{.text = *main};
            if (state.kv->backend) {
                const auto backend = backend_kv_addresses->create_inactive();
                if (!backend) {
                    abort_context();
                    return false;
                }
                operation.reserved_kv->backend = *backend;
            }
            operation.binding_history          = std::make_shared<KVHistory>();
            operation.binding_history->text    = *main;
            operation.binding_history->backend = operation.reserved_kv->backend;
            // Reserve the protected range before exporting an external view. Abort restores
            // the surviving private requirements; the directory and writer remain in place.
            text_kv_addresses->set_checkpoint_requirement(state.kv->text, frontier);
            ticket->main_tail.reset();
            operation.text_view.emplace(
                text_kv_addresses->prepare_active_prefix_view(state.kv->text, *main, frontier));
            if (state.kv->backend && operation.backend_frontier) {
                backend_kv_addresses->set_checkpoint_requirement(*state.kv->backend,
                                                                 operation.backend_frontier);
                ticket->backend_tail.reset();
                operation.backend_view.emplace(backend_kv_addresses->prepare_active_prefix_view(
                    *state.kv->backend, *operation.reserved_kv->backend,
                    operation.backend_frontier));
            }
        }
        state_store->freeze(state.state.read);
        if (host_state) {
            ticket->host.reset();
            auto transfer = state_store->reserve_device_to_host(state.state.read);
            if (!transfer) { throw std::logic_error("capture lost its reserved Host state space"); }
            operation.state_transfer.emplace(std::move(*transfer));
        }
        context_source_ready_.record(device.stream);
        context_source_ready_.wait(device.transfer_stream);
        if (operation.state_transfer) {
            enqueue_state_backup(operation);
        } else if (is_masked_draft_backend(speculative_backend)) {
            copy_local_for_context(operation, state_store->physical_slot(state.state.read),
                                   state_store->physical_slot(*operation.reserved_state));
        }
        const auto copy = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                              const auto& view, runtime::ContextResourceClass resource) {
            if (view && view->needs_tail_copy()) {
                copy_context_tail(operation, pages, addresses.active_prefix_view_tail_source(*view),
                                  addresses.active_prefix_view_tail_destination(*view), resource);
            }
        };
        copy(*text_kv_addresses, *text_kv_pages, operation.text_view,
             runtime::ContextResourceClass::MainKV);
        if (backend_kv_pages) {
            copy(*backend_kv_addresses, *backend_kv_pages, operation.backend_view,
                 runtime::ContextResourceClass::BackendKV);
        }
        enqueue_context_transfers(operation);
        return true;
    } catch (...) {
        abort_context();
        throw;
    }
}

void ProgramImpl::publish_capture(ContextTransaction& tx) {
    auto& state             = sequences[tx.lane];
    auto& request           = requests[tx.lane];
    const auto& group       = request.capture_groups[request.next_capture];
    const auto source_state = state.state.read;
    if (tx.text_view) {
        text_kv_addresses->commit_active_prefix_view(std::move(*tx.text_view));
        if (tx.backend_view) {
            backend_kv_addresses->commit_active_prefix_view(std::move(*tx.backend_view));
        }
        tx.binding_history->owner = this;
        tx.reserved_kv.reset();
    }
    for (const auto& [handle, role] : tx.capture_points) {
        state_store->retain_checkpoint_reference(source_state);
        checkpoints[handle.index].value = CheckpointState{
            .kv    = role == runtime::CheckpointRole::SharedPrefix ? tx.binding_history : state.kv,
            .role  = role,
            .state = source_state,
            .identity          = group.identity,
            .key               = group.key,
            .frontier          = tx.reuse_frontier,
            .backend_frontier  = tx.backend_frontier,
            .rope_delta        = state.rope_delta,
            .tail_hidden_valid = true};
        checkpoints[handle.index].reserved = false;
    }
    if (state_store->residency(*tx.reserved_state) == StateReplicaResidency::None) {
        state_store->split_device_replica_identity(source_state, *tx.reserved_state);
        state.state = {.read = *tx.reserved_state, .write = *tx.reserved_state};
    } else {
        ++tx.operations.state_forks;
        (void)state_store->begin_fork(source_state, *tx.reserved_state);
        state.state = {
            .read         = source_state,
            .write        = *tx.reserved_state,
            .fork_pending = true,
        };
    }
    tx.reserved_state.reset();
    refresh_history_requirements(state.kv);
    refresh_state_views(state);
    skip_capture(sequence_handle(tx.lane));
}

} // namespace ninfer::models::qwen3_5::detail
