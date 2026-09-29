#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "models/qwen3_5/frontend/graft.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// Free memory on each rank's device, in rank order. Ranks that share a physical device (a test mode
// that exercises the multi-stage path on one card) split what is free between them, since each one's
// budget is spent from the same memory.
std::vector<std::size_t> free_bytes_by_rank(const DeviceContext& device) {
    std::vector<std::size_t> out;
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < device.size(); ++other) {
            if (device.same_physical_device(rank, other)) { ++sharing; }
        }
        DeviceBinding bind(device.rank(rank).device);
        out.push_back(current_free_device_bytes() / sharing);
    }
    return out;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        if (!options.grafts.empty()) {
            throw std::invalid_argument("a CausalScoring Engine takes no prompt grafts");
        }
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    // Injected grafts stay resident for the life of the Engine, each in a StateImage and a
    // shared-prefix slot of its own, so the pools grow by that many beyond what requests use.
    const std::uint32_t direct_grafts = models::qwen3_5::count_direct_grafts(options.grafts);
    if (!cache.enabled && direct_grafts != 0) {
        throw std::invalid_argument(
            "direct grafts are held in the context cache, which is disabled");
    }
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);
        return options;
    }

    cache.device_state_slots = cache.device_state_slots.value_or(concurrency) + direct_grafts;
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes =
        cache.max_shared_prefixes.value_or(
            std::max(concurrency, static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers))) +
        direct_grafts;
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);
    cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .architecture             = model->config().text.architecture,
                               .vision_enabled           = options.enable_vision,
                               .max_context              = options.max_context,
                               .media_cache_bytes        = options.media_cache_bytes,
                               .media_live_bytes         = options.media_live_bytes,
                               .media_preprocess_threads = options.media_preprocess_threads,
                               .vision_max_merged_tokens = options.vision_max_merged_tokens,
                               .grafts                   = models::qwen3_5::load_prompt_grafts(
                                   options.grafts, model->config().text)})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

void ModelInstance::inject_pinned_grafts() {
    for (const auto& graft : frontend.grafts()) {
        if (graft.kind != models::qwen3_5::GraftKind::PrefillKV) { program->inject_graft(graft); }
    }
}

ConstructedModel construct_model(const EngineOptions& requested, DeviceContext& device) {
    validate_options(requested);
    const auto start = Clock::now();
    // Every later stage of startup checks its options against the ones the model was loaded with, so
    // the stage split is decided once, here, and carried in the options from then on.
    EngineOptions options = requested;
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    if (options.devices.size() > 1 && options.stage_layers.empty()) {
        const std::vector<std::size_t> free_now = free_bytes_by_rank(device);
        const std::vector<std::uint64_t> free_bytes(free_now.begin(), free_now.end());
        const models::qwen3_5::StageSizing sizing{
            .kv_storage = options.kv_cache,
            .state_slots =
                options.max_concurrency + options.context_cache.device_state_slots.value_or(0U)};
        options.stage_layers = models::qwen3_5::default_stage_layers(
            reader, models::load_options(options), sizing, free_bytes);
    }
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const std::size_t overlay_window_bytes =
        models::qwen3_5::prepare_vision_overlay(instance->parameters, device, options);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    // Each injected graft keeps its text KV pages for good; the pool grows by that many so a request
    // can still use all the capacity that was asked for.
    std::uint32_t graft_main_pages = 0;
    for (const auto& graft : instance->frontend.grafts()) {
        if (graft.kind != models::qwen3_5::GraftKind::PrefillKV) {
            const auto page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
            graft_main_pages += (graft.n_slots + page_tokens - 1U) / page_tokens;
        }
    }
    auto planner = models::qwen3_5::make_sequence_planner(instance->parameters, device, options,
                                                          graft_main_pages);
    const std::vector<std::size_t> free_by_rank = free_bytes_by_rank(device);
    auto resolution = resolve_kv_capacity(
        options.kv_capacity, planner.capacity_curve(), free_by_rank.front(),
        std::span<const std::size_t>(free_by_rank).subspan(1));
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens ||
        !std::ranges::equal(sequence.extra_rank_reservation_bytes(),
                            resolution.extra_rank_reservation_bytes)) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    instance->kv_capacity_resolution = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    instance->inject_pinned_grafts();
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.cuda_sync_mode = device.sync_mode();
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.pinned_weight_bytes  = stats.pinned_bytes;
    summary.overlay_window_bytes = overlay_window_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
