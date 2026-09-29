#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/evictable_kv_pool.h"
#include "core/host_kv_clamp.h"
#include "core/startup.h"
#include "core/device.h"
#include <cuda_runtime.h>
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options);

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options) {
    if (!options.max_private_continuations || *options.max_private_continuations == 0) {
        throw std::logic_error("Qwen3.5 context cache private capacity is not normalized");
    }
    return *options.max_private_continuations;
}

// Overlay Vision residency backs the persistent arena with virtual memory so free KV granules can
// fund a Vision window. The lendable prefix ends past the last page-major KV plane.
std::unique_ptr<EvictableKVPool> make_kv_arena(DeviceContext& device,
                                               const execution::Parameters& parameters,
                                               const SequencePlanImpl& plan) {
    if (!plan.features.overlay_vision()) { return nullptr; }
    const EvictableWeightPool* const pool = parameters.model.weight_pool();
    if (pool == nullptr || !pool->mirror_captured()) {
        throw std::logic_error("overlay Vision weight pool has no captured window");
    }
    const std::size_t window      = pool->window_capacity_bytes();
    const std::size_t granularity = EvictableKVPool::device_granularity(device);
    if (window == 0 || granularity == 0 || plan.persistent.lendable_kv_end_bytes == 0) {
        return nullptr;
    }
    // A KV cache smaller than one window can never fund a concurrent encode. The Engine still runs:
    // every window then borrows the weight tail.
    const std::size_t lendable = plan.persistent.lendable_kv_end_bytes / granularity * granularity;
    if (window > lendable) { return nullptr; }
    return std::make_unique<EvictableKVPool>(
        device, EvictableKVPool::Config{
                    .arena_bytes           = plan.persistent.bytes,
                    .lendable_prefix_bytes = plan.persistent.lendable_kv_end_bytes,
                    .window_capacity_bytes = window,
                });
}

// Persistent state for the ranks beyond the first, each allocated while its own device is current
// so it lands in that card's memory. Empty on one device, which leaves that path unchanged.
std::vector<DeviceArena> make_rank_persistent(DeviceContext& device,
                                              const std::vector<std::size_t>& bytes_by_rank) {
    std::vector<DeviceArena> out;
    out.reserve(bytes_by_rank.size());
    for (std::size_t index = 0; index < bytes_by_rank.size(); ++index) {
        RankBinding bind(device, index + 1);
        out.emplace_back(bytes_by_rank[index]);
    }
    return out;
}

// The state shards and boundary links a forward pass across pipeline stages uses, sized for the
// widest pass the Program will run.
std::unique_ptr<execution::StageRuntime>
make_stage_runtime(DeviceContext& device, const execution::Parameters& parameters,
                   qwen3_5::StateImageDevicePool& state, const SequencePlanImpl& plan) {
    const auto& text         = parameters.text;
    const std::size_t stages = text.rank_count;
    auto runtime             = std::make_unique<execution::StageRuntime>();
    for (std::size_t shard = 0; shard < state.shard_count(); ++shard) {
        runtime->state.push_back(&state.linear(shard));
        runtime->state_first_layer.push_back(state.shard(shard).first_layer);
    }

    const std::uint64_t columns = stage_boundary_columns(plan);
    const std::size_t hidden = static_cast<std::size_t>(parameters.model.config().text.hidden_size);
    const bool force_staged  = [] {
        const char* value = std::getenv("NINFER_FORCE_STAGED_LINKS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    const StageLinkOptions residual{.slot_bytes   = columns * hidden * sizeof(std::uint16_t),
                                    .slots        = 2,
                                    .force_staged = force_staged};
    // Six int32 control tensors of at most `columns` entries, plus alignment.
    const StageLinkOptions control{.slot_bytes   = (6 * columns + 16) * sizeof(std::int32_t),
                                   .slots        = 2,
                                   .force_staged = force_staged};
    for (std::size_t stage = 0; stage + 1 < stages; ++stage) {
        runtime->forward.emplace_back(device, stage, stage + 1, residual);
    }
    runtime->back.emplace(device, stages - 1, 0, residual);
    for (std::size_t stage = 1; stage < stages; ++stage) {
        runtime->control.emplace_back(device, 0, stage, control);
    }
    return runtime;
}

// Scratch for the ranks beyond the first. Each is allocated while its own device is current, so the
// arena lands in that card's memory; `work` then borrows a slice of each and switches between them
// as the layer loop walks stages. Empty without a split, which leaves the single-GPU path unchanged.
//
// Only the general region is duplicated: a stage runs its layers and nothing else, so it never needs
// the Vision, causal-score or bridge regions the primary card's capacity also covers.
std::vector<DeviceArena> make_rank_workspaces(DeviceContext& device,
                                              const execution::TextParameters& text,
                                              std::size_t general_capacity_bytes) {
    std::vector<DeviceArena> out;
    if (!text.split_execution()) { return out; }
    if (text.rank_count > device.size()) {
        throw std::invalid_argument("Qwen3.5 pipeline split needs more devices than are attached");
    }
    out.reserve(text.rank_count - 1);
    for (std::size_t rank = 1; rank < text.rank_count; ++rank) {
        ScopedDeviceRank guard(device, rank);
        out.emplace_back(general_capacity_bytes);
    }
    return out;
}

} // namespace

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache),
      continuation_capacity(normalized_private_capacity(plan.context_cache)),
      shared_prefix_capacity(plan.context_cache.max_shared_prefixes.value_or(0)),
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window), lookup_ngram(plan.lookup_ngram),
      speculative_backend(plan.speculative_backend), kv_storage(plan.kv_storage),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), causal_scoring(plan.causal_scoring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      kv_arena(make_kv_arena(device_in, parameters_in, plan)),
      persistent(kv_arena ? DeviceArena(kv_arena->arena()) : DeviceArena(plan.persistent.bytes)),
      persistent_by_rank(make_rank_persistent(device_in, plan.persistent.extra_rank_bytes)),
      workspace_storage(plan.workspace.capacity),
      workspace_storage_by_rank(
          make_rank_workspaces(device_in, parameters_in.text, plan.workspace.general_capacity)),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      continuation_states(continuation_capacity), continuation_slots(continuation_capacity),
      shared_prefix_states(shared_prefix_capacity), shared_prefix_slots(shared_prefix_capacity),
      round_host(plan.causal_scoring ? std::nullopt
                                     : std::make_optional<PinnedHostBuffer>(sizeof(TokenId))),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreTile * sizeof(float))
                                              : std::nullopt),
      ordinary_host(
          !plan.causal_scoring && plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(is_masked_draft_backend(plan.speculative_backend)
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress))
                      : std::nullopt),
      dflash_draft_host(is_masked_draft_backend(plan.speculative_backend)
                            ? std::make_optional<PinnedHostBuffer>(
                                  sizeof(TokenId) * qwen3_5::kDFlashDecodeMaximumDrafts *
                                  kMaximumConcurrency)
                            : std::nullopt),
      dflash_proposal_ready(is_masked_draft_backend(plan.speculative_backend)
                                ? std::make_optional<CudaCompletionEvent>(device_in)
                                : std::nullopt),
      token_mask_host(plan.persistent.token_masks
                          ? std::make_optional<PinnedHostBuffer>(
                                plan.persistent.token_masks->region.bytes)
                          : std::nullopt),
      compute_streams(RankStreams::compute(device_in)),
      transfer_streams(RankStreams::transfer(device_in)), context_source_ready_(device_in),
      context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)} {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    // Hand `work` the extra ranks' storage. From here one arena serves every device: the layer loop
    // switches ranks alongside ScopedDeviceRank and every workspace call site is unchanged.
    for (DeviceArena& rank_storage : workspace_storage_by_rank) {
        work.attach_rank_storage(
            DeviceSpan{rank_storage.base(), workspace_plan.general_capacity});
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        workspace_plan.vision_resident == plan.features.overlay_vision() ||
        (workspace_plan.vision && workspace_plan.vision_resident &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    if (plan.features.overlay_vision()) {
        EvictableWeightPool* const pool = parameters.model.weight_pool();
        if (pool == nullptr || !parameters.model.vision_overlay() || !workspace_plan.vision ||
            workspace_plan.vision_bridge_bytes == 0 ||
            workspace_plan.vision_bridge_offset + workspace_plan.vision_bridge_bytes >
                workspace_storage.capacity()) {
            throw std::invalid_argument("overlay Vision assets are incomplete");
        }
        vision_broker.emplace(device, *pool);
        vision_results.emplace(max_concurrency, workspace_plan.vision->handoff_capacity_bytes);
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    // Every rank's persistent memory, in rank order: rank 0's is `backing`.
    std::vector<DeviceSpan> backings{backing};
    for (std::size_t index = 0; index < persistent_by_rank.size(); ++index) {
        backings.push_back(persistent_by_rank[index].alloc_bytes(
            plan.persistent.extra_rank_bytes[index], 256));
    }
    if (!plan.context_cache.max_private_continuations || !plan.context_cache.max_shared_prefixes) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::uint64_t address_capacity64 =
        static_cast<std::uint64_t>(*plan.context_cache.max_private_continuations) +
        *plan.context_cache.max_shared_prefixes;
    if (address_capacity64 == 0 || address_capacity64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV address-space capacity exceeds uint32");
    }
    // One unpublished descriptor is reserved for the single in-flight active-capture snapshot.
    // Published private/shared address spaces remain bounded by P + S; the transaction slot lets a
    // full shared catalog replace one entry without releasing the old checkpoint before the new
    // snapshot has been prepared.
    if (address_capacity64 == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV transaction address capacity exceeds uint32");
    }
    const auto address_capacity      = static_cast<std::uint32_t>(address_capacity64 + 1U);
    const auto logical_page_capacity = [&](const DeviceKVPagePool& pool) {
        const HostKVPageLayout host_layout = plan_host_kv_page_layout(pool.geometry());
        const std::uint64_t host_pages =
            plan.context_cache.host_kv_capacity_bytes / host_layout.page_stride;
        const std::uint64_t total = static_cast<std::uint64_t>(pool.capacity_pages()) + host_pages;
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 logical KV page capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(total);
    };

    decoder = std::make_unique<qwen3_5::DecoderState>(backings, plan.persistent.decoder);
    text_host_kv_page_stride =
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()).page_stride;
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_page_capacity(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    if (vision_broker && kv_arena) {
        // A loan changes the admission capacity, so it may not race a sealed plan: refuse one while
        // a context transaction or a pressure-planning session is in flight, or while any lane has
        // been provisionally admitted (prefill staged) but hasn't yet bound its own KV entitlement
        // (lifecycle still Empty) -- that lane's future KV demand is real but not yet reflected in
        // the arena's free-page accounting.
        vision_broker->enable_kv_tier(
            *kv_arena, decoder->text_kv.page_pool(),
            [this] {
                if (has_context_transaction() || pressure_planning_active_) { return false; }
                for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
                    if (requests[lane].prefill && requests[lane].lifecycle == Lifecycle::Empty) {
                        return false;
                    }
                }
                return true;
            },
            [this] { advance_resource_revision(); });
    }
    state_images = std::make_unique<qwen3_5::StateImageDevicePool>(backings,
                                                                    plan.persistent.state_images);
    if (plan.context_cache.host_state_slots != 0) {
        const std::uint64_t host_state_bytes =
            static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
            plan.context_cache.host_state_slots;
        StartupPhaseScope host_state_phase(startup_observer, StartupPhase::HostStatePin,
                                           StartupProgressUnit::Bytes, host_state_bytes);
        host_state_images = std::make_unique<qwen3_5::HostStatePool>(
            state_images->host_layout(), plan.context_cache.host_state_slots);
        host_state_phase.complete(host_state_bytes, host_state_bytes);
    }
    const std::uint64_t logical_state_capacity =
        static_cast<std::uint64_t>(state_images->slot_count()) +
        plan.context_cache.host_state_slots;
    if (logical_state_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 logical StateImage capacity exceeds uint32");
    }
    state_store = std::make_unique<StateImageStore>(
        *state_images, host_state_images.get(), static_cast<std::uint32_t>(logical_state_capacity));
    pressure_private_owner_scratch_.resize(continuation_capacity);
    pressure_shared_owner_scratch_.resize(shared_prefix_capacity);
    pressure_private_drop_scratch_.resize(continuation_capacity);
    const std::size_t pressure_checkpoint_capacity =
        2U + context_cache.max_long_anchors_per_continuation.value_or(0U);
    for (auto& dropped : pressure_private_drop_scratch_) {
        dropped.reserve(pressure_checkpoint_capacity);
    }
    pressure_state_scratch_.reserve(static_cast<std::size_t>(logical_state_capacity));
    if (plan.persistent.replay_records) {
        replay_records.emplace(backings[state_images->shard(0).rank],
                               *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear(0).all_layers_view());
        for (std::size_t shard = 1; shard < state_images->shard_count(); ++shard) {
            extra_replay_records.push_back(std::make_unique<GdnReplayRecords>(
                backings[state_images->shard(shard).rank],
                plan.persistent.extra_replay_records.at(shard - 1)));
            extra_replay_fold.push_back(std::make_unique<ops::GdnReplayFoldPlan>(
                *extra_replay_records.back(), state_images->linear(shard).all_layers_view()));
        }
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (parameters.text.split_execution()) {
        stage_runtime = make_stage_runtime(device, parameters, *state_images, plan);
        if (replay_records) {
            stage_runtime->replay.push_back(&*replay_records);
            for (const auto& records : extra_replay_records) {
                stage_runtime->replay.push_back(records.get());
            }
        }
    }
    if (plan.persistent.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) {
            throw std::logic_error("DFlash StateImage has no local fixed state");
        }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    if (dflash.has_value() != plan.features.masked_draft()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    if (qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        backend_host_kv_page_stride =
            plan_host_kv_page_layout(backend->page_pool().geometry()).page_stride;
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_page_capacity(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }
    pressure_text_page_scratch_.resize(text_kv_pages->capacity());
    pressure_text_selected_pages_.reserve(text_kv_pages->capacity());
    if (backend_kv_pages) {
        pressure_backend_page_scratch_.resize(backend_kv_pages->capacity());
        pressure_backend_selected_pages_.reserve(backend_kv_pages->capacity());
    }
    if (plan.context_cache.host_kv_capacity_bytes != 0) {
        std::vector<HostKVPageLayout> layouts;
        layouts.push_back(plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()));
        if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
            HostKVPageLayout backend_layout =
                plan_host_kv_page_layout(backend->page_pool().geometry());
            if (backend_layout != layouts.front()) { layouts.push_back(std::move(backend_layout)); }
        }
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }

        // The pinned host KV buffer competes with the model for *device* memory, and on Windows
        // that is what actually stops it.
        //
        // Measured on an RTX 3090 (24,576 MiB), one process, allocating N MiB on the device and
        // then finding the largest cudaMallocHost that succeeds:
        //
        //     device resident   VRAM free   largest pin
        //         15,360 MiB     7,972 MiB     8,192 MiB
        //         17,408 MiB     5,924 MiB     6,656 MiB
        //         19,456 MiB     3,876 MiB     3,840 MiB
        //         21,504 MiB     1,828 MiB     2,816 MiB
        //         22,528 MiB       804 MiB     1,536 MiB
        //
        // Resident-device plus pinned-host lands within a few hundred MiB of the card's capacity
        // every time: WDDM maps pinned host memory into the GPU's address space and charges it
        // against the same budget. The default 8 GiB therefore cannot be reserved beside any model
        // this project ships, and the server refused to start.
        //
        // **Asking and backing off does not work.** One failed cudaMallocHost poisons every later
        // one in the process (`cudaErrorAlreadyMapped` persists and `cudaGetLastError` does not
        // clear it), so the size has to be chosen before the first attempt.
        //
        // Windows only: the coupling is a WDDM property and has not been measured on Linux, where
        // shrinking the buffer would cost prefix reuse for no reason. Clamping to zero is a
        // supported outcome -- prefix reuse then works from device pages alone.
        std::size_t reserved_bytes = plan.context_cache.host_kv_capacity_bytes;
#if defined(_WIN32)
        // The 1 GiB floor inside the clamp is memory the run still needs after this point and that
        // no plan accounts for (module loads, graph instantiation, launch-local backing); halving
        // what remains is the conservative split, so the cache never takes more than it leaves.
        std::size_t free_device = 0, total_device = 0;
        if (cudaMemGetInfo(&free_device, &total_device) == cudaSuccess) {
            reserved_bytes =
                clamp_host_kv_reservation_bytes(reserved_bytes, free_device, minimum_stride);
        }
#endif
        StartupPhaseScope host_kv_phase(
            startup_observer, StartupPhase::HostKvPin, StartupProgressUnit::Bytes,
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        if (reserved_bytes >= minimum_stride) {
            try {
                host_kv_arena = std::make_unique<HostKVArena>(
                    reserved_bytes,
                    std::span<const HostKVPageLayout>(layouts.data(), layouts.size()));
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("failed to reserve the host KV cache: ") + error.what() +
                    "\nThis is the context cache's pinned host buffer, sized by --host-kv-mib "
                    "(default 8192) and clamped at startup to the device memory still free. On "
                    "Windows the allocation is mapped into the GPU's address space and competes "
                    "with the model for VRAM, so this is a full card rather than short system RAM: "
                    "free VRAM, lower --max-context, or pass --no-prefix-reuse to disable the "
                    "context cache entirely.");
            }
        } else {
            // Nothing was pinned. Prefix reuse still works from device pages; only the host-side
            // spill is gone. `host_kv_arena` stays null, which every user of it already handles.
            reserved_bytes = 0;
        }
        host_kv_phase.complete(
            static_cast<std::uint64_t>(reserved_bytes),
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        const std::size_t extent_capacity = reserved_bytes == 0 ? 0 : reserved_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 Host KV extent capacity exceeds uint32");
        }
        if (extent_capacity != 0) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(
                *host_kv_arena, static_cast<std::uint32_t>(extent_capacity));
        }
    }

    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    if (plan.persistent.token_masks) {
        token_masks      = plan.persistent.token_masks->bind(backing);
        token_mask_words = static_cast<std::uint32_t>(token_masks.ne[0]);
    }
    active_continuations.fill(continuation_capacity);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { lane_epochs[lane] = 1; }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        SequenceState& sequence = continuation_states[index];
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.long_anchors.reserve(context_cache.max_long_anchors_per_continuation.value_or(0));
        // One retained shared resume source can coexist with every fixed per-request candidate
        // and, when a request asks for it, every prefix-grid candidate.
        sequence.shared_prefix_references.reserve(8U + kPrefixGridCandidates);
    }
    materialization_ledger_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_identity_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_prefix_digests_.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    set_device_i32(io.text_kv_table_row, 0);
    if (!causal_scoring) { set_device_i32(io.backend_kv_table_row, 0); }

    host_tokens = round_host ? static_cast<TokenId*>(round_host->data()) : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    }
    device.synchronize();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        prepare_graphs();
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

void ProgramImpl::synchronize_transfer_streams() const {
    for (std::size_t rank = 0; rank < transfer_streams.size(); ++rank) {
        (void)cudaStreamSynchronize(transfer_streams[rank]);
    }
}

ProgramImpl::~ProgramImpl() noexcept {
    for (std::size_t rank = 0; rank < transfer_streams.size(); ++rank) {
        (void)cudaStreamSynchronize(transfer_streams[rank]);
        (void)cudaStreamSynchronize(compute_streams[rank]);
    }
}

std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,
                                             std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t entitlement            = kv_pages_for_frontier(predictor_count);
    if (entitlement == 0) { throw std::logic_error("causal score has no KV entitlement"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;

    try {
        state = state_store->reserve_reset(compute_streams);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(entitlement, 0, compute_streams);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, compute_streams);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {columns});
            Tensor logprobs   = work.alloc(DType::FP32, {columns});
            Tensor hidden     = score_hidden->slice(1, 0, columns);
            execution::project(hidden, parameters.text.output_head, logits, work, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids,
                                              dimension(parameters.model.resources().public_token_count),
                                              logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(0), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head, stage_runtime.get()},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    out.weights = ArenaMemorySummary{weights.device_capacity_bytes, weights.device_capacity_bytes,
                                     weights.device_capacity_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
            .residency              = workspace_plan.vision_resident ? VisionResidency::Resident
                                                                     : VisionResidency::Overlay,
        };
        if (const EvictableWeightPool* const pool = parameters.model.weight_pool();
            !workspace_plan.vision_resident && pool != nullptr) {
            out.vision_workspace->window_capacity_bytes = pool->window_capacity_bytes();
            out.vision_workspace->pinned_weight_bytes   = parameters.model.pinned_weights().size();
            out.vision_workspace->mirror_bytes          = pool->mirror_bytes();
        }
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    if (host_state_images) {
        out.host_state_capacity_slots = host_state_images->capacity();
        out.host_state_occupied_slots = host_state_images->occupied();
    }
    if (host_kv_arena) {
        out.host_kv_capacity_bytes = host_kv_arena->capacity_bytes();
        out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes();
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace ninfer::models::qwen3_5::detail
