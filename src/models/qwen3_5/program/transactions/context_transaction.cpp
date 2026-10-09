#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::context_blocks(SequenceHandle sequence) const noexcept {
    if (!context_transaction_ || !valid_sequence(sequence)) { return false; }
    const auto& tx = *context_transaction_;
    return tx.kind == ContextOperationKind::Bind && (tx.text_view || tx.backend_view) &&
           sequences[ContractAccess::lane(sequence).value].kv == tx.source_history;
}

void ProgramImpl::enqueue_state_backup(ContextTransaction& tx) {
    if (!tx.state_transfer) { return; }
    tx.transfers.push_back(state_transfer_requirement(
        state_images->host_layout(), runtime::ContextTransferDirection::DeviceToHost));
    start_context_transfer_timer(runtime::ContextResourceClass::State);
    state_store->enqueue_device_to_host(*tx.state_transfer, device.transfer_stream);
    stop_context_transfer_timer(runtime::ContextResourceClass::State);
}

void ProgramImpl::copy_local_for_context(ContextTransaction& tx, std::int32_t source,
                                         std::int32_t destination) {
    tx.transfers.push_back(state_transfer_requirement(
        state_images->host_layout(), runtime::ContextTransferDirection::DeviceToDevice, true));
    start_context_transfer_timer(runtime::ContextResourceClass::State);
    state_images->copy_dflash_local(source, destination, device.transfer_stream);
    stop_context_transfer_timer(runtime::ContextResourceClass::State);
}

void ProgramImpl::copy_context_tail(ContextTransaction& tx, LogicalKVPageStore& pages,
                                    DeviceKVPageHandle source, DeviceKVPageHandle destination,
                                    runtime::ContextResourceClass resource) {
    start_context_transfer_timer(resource);
    const auto work = pages.physical_pool().copy_page(source, destination, device.transfer_stream);
    stop_context_transfer_timer(resource);
    tx.transfers.push_back(kv_transfer_requirement(
        resource, runtime::ContextTransferDirection::DeviceToDevice, 1, work));
    ++tx.operations.partial_tail_cow_pages;
}

void ProgramImpl::enqueue_context_transfers(ContextTransaction& tx) {
    for (const auto resource :
         {runtime::ContextResourceClass::MainKV, runtime::ContextResourceClass::BackendKV}) {
        bool started = false;
        bool restore = false;
        TransferWork total;
        std::uint32_t page_count = 0;
        for (auto& transfer : tx.kv_transfers) {
            if (transfer.resource != resource || transfer.logical.empty()) { continue; }
            if (!started) {
                start_context_transfer_timer(resource);
                restore = transfer.restore;
                started = true;
            } else if (restore != transfer.restore) {
                throw std::logic_error("one transfer batch mixes directions within a typed pool");
            }
            TransferWork work;
            if (transfer.restore) {
                std::size_t begin = 0;
                while (begin < transfer.logical.size()) {
                    const auto first = transfer.pages->host_replica(transfer.logical[begin]);
                    std::size_t end  = begin + 1;
                    while (end < transfer.logical.size()) {
                        const auto next = transfer.pages->host_replica(transfer.logical[end]);
                        if (next.extent != first.extent ||
                            next.page_offset != first.page_offset + end - begin) {
                            break;
                        }
                        ++end;
                    }
                    const auto host =
                        host_kv_extents->view(first.extent)
                            .subview(first.page_offset, static_cast<std::uint32_t>(end - begin));
                    const auto destination = std::span<const DeviceKVPageHandle>(transfer.physical)
                                                 .subspan(begin, end - begin);
                    const auto part = transfer.pages->physical_pool().copy_from_host(
                        host, destination, device.transfer_stream);
                    work.payload_bytes += part.payload_bytes;
                    work.copy_operations += part.copy_operations;
                    begin = end;
                }
            } else if (transfer.host_destination) {
                const auto host = host_kv_extents->writable_view(*transfer.host_destination);
                work = transfer.pages->physical_pool().copy_to_host(transfer.physical, host,
                                                                    device.transfer_stream);
                if (tx.kind == ContextOperationKind::Pause ||
                    tx.kind == ContextOperationKind::Demote) {
                    tx.operations.pressure_spill_pages += transfer.physical.size();
                }
            }
            total.payload_bytes += work.payload_bytes;
            total.copy_operations += work.copy_operations;
            page_count += static_cast<std::uint32_t>(transfer.logical.size());
        }
        if (!started) { continue; }
        stop_context_transfer_timer(resource);
        tx.transfers.push_back({.resource   = resource,
                                .direction  = restore
                                                  ? runtime::ContextTransferDirection::HostToDevice
                                                  : runtime::ContextTransferDirection::DeviceToHost,
                                .units      = total.payload_bytes,
                                .page_count = page_count,
                                .work       = total});
    }
    tx.submitted = !tx.transfers.empty();
    if (tx.submitted) { context_completion_.record(device.transfer_stream); }
}

void ProgramImpl::publish_context_transfers(ContextTransaction& tx) {
    if (tx.state_transfer) {
        state_store->publish_transfer(std::move(*tx.state_transfer), tx.preserve_state_device);
        tx.state_transfer.reset();
    }
    for (auto& transfer : tx.kv_transfers) {
        if (transfer.restore) {
            for (const auto page : transfer.logical) {
                transfer.pages->publish_device_replica(page);
            }
            transfer.logical.clear();
        } else if (transfer.host_destination) {
            (void)host_kv_extents->publish(std::move(*transfer.host_destination));
            transfer.host_destination.reset();
        }
        if (transfer.drop_device) {
            for (const auto page : transfer.logical) {
                if (!transfer.pages->drop_device_replica(page)) {
                    throw std::logic_error("published Host copy still has a Device reader");
                }
            }
        }
    }
}

ContextProgress ProgramImpl::poll_context(runtime::CancellationFlagView cancellation) {
    if (!context_transaction_) { throw std::logic_error("there is no context operation"); }
    auto& tx = *context_transaction_;
    ContextProgress out{.kind = tx.kind};
    // Bind has two transfer phases. Advance both in this call when their real events
    // are already complete; an unready event yields without synchronizing or spinning.
    for (std::uint32_t phase = 0; phase < 2; ++phase) {
        if (tx.submitted && !context_completion_.ready()) { return out; }
        for (const auto& transfer : tx.transfers) {
            tx.observations.push_back(context_transfer_observation(
                transfer.resource, transfer.direction, transfer.work, transfer.page_count,
                transfer.resource == runtime::ContextResourceClass::State ? transfer.units : 0));
        }
        tx.transfers.clear();
        if (tx.kind == ContextOperationKind::Pause && tx.paused && tx.paused->impl_) {
            out.request_timings     = tx.paused->impl_->control.timings;
            out.request_speculative = tx.paused->impl_->control.speculative_stats;
        }
        if (cancellation.requested()) {
            out.transfers  = std::move(tx.observations);
            out.operations = tx.operations;
            abort_context();
            out.complete = true;
            return out;
        }
        if (!tx.binding_prepared) {
            publish_context_transfers(tx);
            if (tx.kind == ContextOperationKind::Bind) {
                prepare_binding(tx);
                tx.binding_prepared = true;
                out.advanced        = true;
                continue;
            }
        }
        if (tx.kind == ContextOperationKind::Bind) {
            complete_binding(tx, out);
        } else if (tx.kind == ContextOperationKind::Capture) {
            // Capture publication is implemented with the matching state/KV reservation in
            // capture.cpp.
            out.captured_checkpoints.reserve(tx.capture_points.size());
            publish_capture(tx);
            for (const auto& [handle, role] : tx.capture_points) {
                out.captured_checkpoints.push_back(handle);
            }
        } else if (tx.kind == ContextOperationKind::Pause) {
            out.paused.emplace(complete_pause(tx));
        }
        if (tx.source) { --checkpoints[tx.source->index].pins; }
        for (const auto point : tx.carried_pins) { --checkpoints[point.index].pins; }
        tx.carried_pins.clear();
        out.complete = out.published = true;
        out.transfers                = std::move(tx.observations);
        out.operations               = tx.operations;
        context_transaction_.reset();
        return out;
    }
    return out;
}

void ProgramImpl::abort_context() noexcept {
    if (!context_transaction_) { return; }
    auto& tx = *context_transaction_;
    try {
        // Enqueue can fail before recording the completion event. Drain the transfer stream
        // itself so every submitted reader has retired before releasing its source/destination.
        // CUDA failure is already fatal to the Engine; it must not prevent CPU ownership cleanup.
        (void)cudaStreamSynchronize(device.transfer_stream);
        if (tx.state_transfer) {
            state_store->abort_transfer(std::move(*tx.state_transfer));
            tx.state_transfer.reset();
        }
        for (auto& transfer : tx.kv_transfers) {
            if (transfer.restore) {
                for (const auto page : transfer.logical) {
                    transfer.pages->abort_device_replica(page, *transfer.device_reservation);
                }
            }
            transfer.host_destination.reset();
        }
        tx.text_fork.reset();
        tx.backend_fork.reset();
        tx.text_activation.reset();
        tx.backend_activation.reset();
        tx.text_view.reset();
        tx.backend_view.reset();
        if (tx.kind == ContextOperationKind::Bind && tx.adopted) {
            release_sequence_state(sequences[tx.lane]);
            release_sequence_kv(sequences[tx.lane]);
            tx.reserved_state.reset();
        }
        if (tx.reserved_kv) {
            const auto unwind_address = [&](KVAddressSpaceStore& addresses,
                                            KVAddressSpaceHandle address, bool borrowed,
                                            std::uint32_t frontier) {
                if (!borrowed) {
                    if (!addresses.release_after_deactivate(address)) { std::terminate(); }
                } else if (addresses.active(address)) {
                    addresses.deactivate(address);
                    addresses.set_checkpoint_requirement(address, frontier);
                }
            };
            if (tx.reserved_kv->backend) {
                unwind_address(*backend_kv_addresses, *tx.reserved_kv->backend, tx.borrow_backend,
                               tx.backend_frontier);
            }
            unwind_address(*text_kv_addresses, tx.reserved_kv->text, tx.borrow_text,
                           tx.reuse_frontier);
            tx.reserved_kv.reset();
        }
        if (tx.kind == ContextOperationKind::Bind) {
            // Any already-installed state binding is still private to this unpublished lane.
            if (sequences[tx.lane].state.fork_pending) {
                state_store->abort_fork(sequences[tx.lane].state.read,
                                        sequences[tx.lane].state.write);
            }
            sequences[tx.lane].state = {};
            sequences[tx.lane].kv.reset();
            requests[tx.lane] = {};
            invalidate_lane(tx.lane);
        }
        if (tx.kind == ContextOperationKind::Capture) {
            auto& state = sequences[tx.lane];
            refresh_history_requirements(state.kv);
            if (state_store->role(state.state.read) == StateImageRole::CheckpointImmutable &&
                !state_store->checkpoint_references(state.state.read)) {
                state_store->thaw(state.state.read);
            }
        }
        if (tx.kind == ContextOperationKind::Pause) {
            requests[tx.lane] = {};
            invalidate_lane(tx.lane);
        }
        if (tx.reserved_state && !tx.borrow_state) {
            (void)state_store->release(*tx.reserved_state);
        }
        if (tx.source) { --checkpoints[tx.source->index].pins; }
        for (const auto point : tx.carried_pins) { --checkpoints[point.index].pins; }
        for (const auto& [from, to] : tx.carried_clones) {
            if (valid_checkpoint(to)) {
                (void)release_checkpoint(to);
            } else {
                checkpoints[to.index].reserved = false;
            }
        }
        for (const auto& [handle, role] : tx.capture_points) {
            if (!valid_checkpoint(handle)) { checkpoints[handle.index].reserved = false; }
        }
        context_transaction_.reset();
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::qwen3_5::detail
