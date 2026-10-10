#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include <cuda_runtime.h>
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

// Overlay Vision residency backs the persistent arena with virtual memory so free Main KV granules
// can fund a Vision window. The lendable prefix ends past the last page-major Main KV plane.
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
    // A window for the largest item may exceed the whole KV cache, but smaller items still fit; one
    // that does not borrows the weight tail instead.
    const std::size_t lendable = plan.persistent.lendable_kv_end_bytes / granularity * granularity;
    if (lendable == 0) { return nullptr; }
    return std::make_unique<EvictableKVPool>(
        device, EvictableKVPool::Config{
                    .arena_bytes           = plan.persistent.bytes,
                    .lendable_prefix_bytes = plan.persistent.lendable_kv_end_bytes,
                    .window_capacity_bytes = std::min(window, lendable),
                });
}

DeviceArena make_persistent(EvictableKVPool* arena, std::size_t bytes) {
    return arena != nullptr ? DeviceArena(arena->arena()) : DeviceArena(bytes);
}

// Persistent state for the ranks beyond the first, each allocated while its own device is current
// so it lands in that card's memory. Empty on one device.
std::vector<DeviceArena> make_rank_persistent(const DeviceContext& device,
                                              const std::vector<std::size_t>& bytes_by_rank) {
    if (bytes_by_rank.size() + 1 != device.size()) {
        throw std::invalid_argument("the sequence plan does not cover every attached device");
    }
    std::vector<DeviceArena> out;
    out.reserve(bytes_by_rank.size());
    for (std::size_t index = 0; index < bytes_by_rank.size(); ++index) {
        RankBinding bind(device, index + 1);
        out.emplace_back(bytes_by_rank[index]);
    }
    return out;
}

// Scratch for the ranks beyond the first, allocated on each rank's own device. Only the general
// region is duplicated: a stage runs its layers and nothing else, so it never needs the Vision or
// causal-score regions the primary card's capacity also covers.
std::vector<DeviceArena> make_rank_workspaces(const DeviceContext& device,
                                              std::size_t general_capacity_bytes) {
    std::vector<DeviceArena> out;
    out.reserve(device.size() - 1);
    for (std::size_t rank = 1; rank < device.size(); ++rank) {
        RankBinding bind(device, rank);
        out.emplace_back(general_capacity_bytes);
    }
    return out;
}

bool force_staged_links() {
    const char* value = std::getenv("NINFER_FORCE_STAGED_LINKS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// The state shards, boundary links and fences a forward pass across pipeline stages uses, sized for
// the widest pass the Program will run.
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
    const bool force_staged  = force_staged_links();
    const StageLinkOptions residual{.slot_bytes   = columns * hidden * sizeof(std::uint16_t),
                                    .slots        = 2,
                                    .force_staged = force_staged};
    // Six int32 control tensors, plus alignment. Of the six, only the position pair is ever full
    // width in prefill (cache `columns`, multimodal rope `3 * columns`), so the sum stays inside
    // `6 * columns`; `run_staged` checks the packed size against this slot regardless.
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
    // DFlash reads the hidden state after each of its feature layers in rank 0's buffers; a later
    // stage owning some of them sends them back in one block per pass.
    runtime->features.resize(stages - 1);
    for (std::size_t stage = 1; stage < stages; ++stage) {
        const std::size_t layers = stage_feature_layer_count(plan, stage);
        if (layers == 0) { continue; }
        runtime->features[stage - 1].emplace(
            device, stage, 0,
            StageLinkOptions{.slot_bytes   = layers * columns * hidden * sizeof(std::uint16_t),
                             .slots        = 2,
                             .force_staged = force_staged});
    }
    runtime->rank_fences.reserve(device.size());
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        runtime->rank_fences.emplace_back(device.rank(rank));
    }
    return runtime;
}

} // namespace

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window),
      speculative_backend(plan.speculative_backend),
      kv_storage(plan.kv_storage), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      causal_scoring(plan.causal_scoring), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      kv_arena(make_kv_arena(device_in, parameters_in, plan)),
      persistent(make_persistent(kv_arena.get(), plan.persistent.bytes)),
      persistent_by_rank(make_rank_persistent(device_in, plan.persistent.extra_rank_bytes)),
      workspace_storage(plan.workspace.capacity),
      workspace_storage_by_rank(make_rank_workspaces(device_in, plan.workspace.general_capacity)),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      compute_streams(RankStreams::compute(device_in)),
      transfer_streams(RankStreams::transfer(device_in)),
      round_host(plan.causal_scoring
                     ? std::nullopt
                     : std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::PrefillRoundHost))),
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
                                                             sizeof(qwen3_5::DFlashDecodeEgress) +
                                                             sizeof(qwen3_5::DFlashPrefillIngress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)},
      prefill_gpu_timer_(device_in) {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
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
        // One slot per lane is pinned up front. A resumed lane briefly holds a replay session, a
        // prefill session and the session it replaces; those extra slots are pinned on demand.
        vision_results.emplace(max_concurrency, 3U * max_concurrency,
                               workspace_plan.vision->handoff_capacity_bytes);
    }
    if (parameters.text.rank_count != device.size()) {
        throw std::invalid_argument("the model's pipeline stages do not match the attached devices");
    }
    // From here one workspace arena serves every rank: the layer loop switches it alongside the
    // device, and every workspace call site is unchanged.
    for (DeviceArena& rank_storage : workspace_storage_by_rank) {
        work.attach_rank_storage(DeviceSpan{rank_storage.base(), workspace_plan.general_capacity});
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    // Every rank's persistent memory, in rank order: rank 0's is `backing`.
    std::vector<DeviceSpan> backings{backing};
    for (std::size_t index = 0; index < persistent_by_rank.size(); ++index) {
        backings.push_back(
            persistent_by_rank[index].alloc_bytes(plan.persistent.extra_rank_bytes[index], 256));
    }
    decoder      = std::make_unique<qwen3_5::DecoderState>(backings, plan.persistent.decoder);
    state_images = std::make_unique<StateImageDevicePool>(backings, plan.persistent.state_images);
    if (plan.persistent.extra_replay_records.size() + 1 !=
        (plan.persistent.replay_records ? state_images->shard_count() : 1)) {
        throw std::logic_error("ReplaySSM records do not cover every StateImage shard");
    }
    if (plan.persistent.replay_records) {
        // Each shard records and folds only its own layers, in its own device's memory.
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
        auto* local = state_images->dflash_local();
        if (!local) { throw std::logic_error("DFlash StateImage has no local state"); }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    std::size_t host_bytes = plan.context_cache.host_capacity_bytes.value();
    std::vector<HostKVPageLayout> layouts{
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry())};
    text_host_kv_page_stride = layouts.front().page_stride;
    if (const auto* backend = backend_kv_cache()) {
        auto layout                 = plan_host_kv_page_layout(backend->page_pool().geometry());
        backend_host_kv_page_stride = layout.page_stride;
        if (layout != layouts.front()) { layouts.push_back(std::move(layout)); }
    }
    std::size_t minimum_stride    = state_images->host_layout().image_bytes;
    std::size_t minimum_kv_stride = layouts.front().page_stride;
    for (const auto& layout : layouts) {
        minimum_stride    = std::min(minimum_stride, layout.page_stride);
        minimum_kv_stride = std::min(minimum_kv_stride, layout.page_stride);
    }
    // The Host context arena is pinned at its full size on every platform. This fork used to clamp
    // it on Windows to (free VRAM - 1 GiB) / 2, after a measurement where the largest single
    // cudaMallocHost tracked free VRAM (WDDM charging pinned memory to the card). That coupling
    // is gone on current drivers: on an RTX 3090 with driver 616.64, 22,528 MiB resident on the
    // device and 804 MiB free, cudaHostAlloc pinned 16 GiB in one allocation and 32 GiB in 1 GiB
    // chunks, reported free VRAM fell by a fixed ~0.5 GiB, and a further 512 MiB cudaMalloc still
    // succeeded. The clamp had shrunk the 27B profiles' host tier to a few hundred MiB for nothing.
    const auto checked_count = [](std::uint64_t value) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("context resource descriptor capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(value);
    };
    const auto logical_states = checked_count(state_images->slot_count() +
                                              host_bytes / state_images->host_layout().image_bytes);
    // One immutable image can serve private replay, an explicit anchor, a public view and an
    // exact-hit endpoint. Alias descriptors are bounded separately from physical state slots.
    checkpoints.resize(checked_count(4ULL * logical_states + 2ULL * max_concurrency));
    const auto address_capacity = checked_count(checkpoints.size() + max_concurrency + 2U);
    if (host_bytes) {
        StartupPhaseScope phase(startup_observer, StartupPhase::HostContextPin,
                                StartupProgressUnit::Bytes, host_bytes);
        try {
            host_context_arena = std::make_unique<HostContextArena>(host_bytes, minimum_stride);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("failed to pin the Host context cache: ") + error.what() +
                "\nLower the Host context capacity (--host-context-mib) or free system RAM.");
        }
        host_state_images =
            std::make_unique<HostStatePool>(*host_context_arena, state_images->host_layout());
        host_kv_arena      = std::make_unique<HostKVArena>(*host_context_arena, layouts);
        const auto extents = checked_count(host_bytes / minimum_kv_stride);
        if (extents) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(*host_kv_arena, extents);
        }
        phase.complete(host_bytes, host_bytes);
    }
    state_store =
        std::make_unique<StateImageStore>(*state_images, host_state_images.get(), logical_states);
    const auto logical_pages = [&](const DeviceKVPagePool& pool) {
        const auto stride = plan_host_kv_page_layout(pool.geometry()).page_stride;
        return checked_count(pool.capacity_pages() + host_bytes / stride);
    };
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_pages(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    if (vision_broker && kv_arena) {
        // A window that outlives its unit must not change capacity under a context transaction,
        // which plans its page moves across several polls.
        vision_broker->enable_kv_tier(*kv_arena, decoder->text_kv.page_pool(),
                                      [this] { return !has_context_transaction(); });
    }
    if (auto* backend = backend_kv_cache()) {
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_pages(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
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
    if (plan.persistent.grammar_masks) {
        grammar_masks_device = plan.persistent.grammar_masks->bind(backing);
        grammar_masks_host.emplace(grammar_masks_device.bytes());
    }
    if (is_masked_draft_backend(speculative_backend)) {
        dflash_draft_handoff.emplace(device, draft_window * max_concurrency);
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        lane_epochs[lane]    = 1;
        sequences[lane].lane = lane;
    }
    host_tokens = round_host
                      ? &static_cast<qwen3_5::PrefillRoundHost*>(round_host->data())->sampled_token
                      : nullptr;
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
        *dflash_host_ingress        = {};
        *dflash_host_egress         = {};
        dflash_prefill_host_ingress = reinterpret_cast<qwen3_5::DFlashPrefillIngress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress) + sizeof(qwen3_5::DFlashDecodeEgress));
        *dflash_prefill_host_ingress = {};
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

void ProgramImpl::synchronize_transfer_streams() const noexcept {
    for (std::size_t rank = 0; rank < transfer_streams.size(); ++rank) {
        const cudaStream_t stream = transfer_streams[rank];
        if (stream != nullptr) { (void)cudaStreamSynchronize(stream); }
    }
}

ProgramImpl::~ProgramImpl() noexcept {
    synchronize_transfer_streams();
    for (std::size_t rank = 0; rank < compute_streams.size(); ++rank) {
        const cudaStream_t stream = compute_streams[rank];
        if (stream != nullptr) { (void)cudaStreamSynchronize(stream); }
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
    const std::uint32_t required_pages         = kv_pages_for_frontier(predictor_count);
    if (required_pages == 0) { throw std::logic_error("causal score requires KV pages"); }

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
        address = text_kv_addresses->create_active(required_pages, 0, compute_streams);
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
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
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
    if (host_state_images) { out.host_state_occupied_slots = host_state_images->occupied(); }
    if (host_kv_arena) { out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes(); }
    if (host_context_arena) {
        out.host_context_capacity_bytes = host_context_arena->capacity_bytes();
        out.host_context_occupied_bytes = host_context_arena->occupied_bytes();
        out.host_context_reserved_bytes = host_context_arena->reserved_bytes();
    }
    return out;
}

std::unique_ptr<execution::VisionPrefillSession>
ProgramImpl::make_vision_session(const PreparedPromptData& prompt, const VisionPrefillPlan& plan) {
    if (!workspace_plan.vision) {
        throw std::logic_error("Vision prefill has no startup workspace plan");
    }
    if (vision_broker) {
        return std::make_unique<execution::VisionPrefillSession>(
            device, parameters, *workspace_plan.vision, prompt, plan, *vision_broker,
            vision_results->acquire(),
            DeviceSpan{static_cast<std::byte*>(workspace_storage.base()) +
                           workspace_plan.vision_bridge_offset,
                       workspace_plan.vision_bridge_bytes});
    }
    return std::make_unique<execution::VisionPrefillSession>(
        device, parameters, DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
        *workspace_plan.vision, prompt, plan, vision_handoff, vision_handoff_peak_bytes);
}

runtime::ResourceReservation ProgramImpl::reserve_vision_window(SequenceHandle handle) {
    if (!vision_broker || !vision_broker->kv_tier() || !valid_sequence(handle)) { return {}; }
    RequestControl& request = requests[ContractAccess::lane(handle).value];
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill ||
        !request.prefill->vision || !request.permit ||
        request.permit->kind != ExecutionUnitKind::Prefill || vision_broker->window_open() ||
        !vision_broker->can_submit()) {
        return {};
    }
    auto& staged             = *request.prefill;
    const auto window_bytes  = staged.vision->pending_window_bytes(staged.cursor, prefill_chunk);
    if (!window_bytes) { return {}; }
    const std::uint32_t available = vision_broker->available_kv_pages();
    const std::uint32_t needed    = vision_broker->kv_loan_pages(*window_bytes);
    if (needed != 0 && needed <= available) {
        return {.reserved = staged.vision->submit_item(staged.cursor, prefill_chunk)};
    }
    // The free page runs cannot fund the window now. Report what reclaiming cached content would
    // have to free; fragmentation alone (enough pages, no long enough runs) is not worth evicting
    // anything for, and the unit then encodes synchronously.
    const std::uint32_t minimum =
        needed != 0 ? needed : vision_broker->kv_loan_minimum_pages(*window_bytes);
    if (minimum <= available) { return {}; }
    return {.reserved = false, .shortage = {.main_kv_pages = minimum - available}};
}

bool ProgramImpl::vision_pending(SequenceHandle handle) const noexcept {
    if (!vision_broker || !valid_sequence(handle)) { return false; }
    const RequestControl& request = requests[ContractAccess::lane(handle).value];
    try {
        return request.prefill && request.prefill->vision &&
               request.prefill->vision->vision_pending();
    } catch (...) {
        // A failed completion query surfaces when the prefill unit takes the item.
        return false;
    }
}

bool ProgramImpl::poll_vision() {
    if (!vision_broker) { return false; }
    bool closed = false;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        RequestControl& request = requests[lane];
        if (request.prefill && request.prefill->vision) {
            closed = request.prefill->vision->poll() || closed;
        }
    }
    return closed;
}

bool ProgramImpl::drain_vision_window() { return vision_broker && vision_broker->drain(); }

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
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
