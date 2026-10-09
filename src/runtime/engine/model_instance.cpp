#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/frontend/graft.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "runtime/engine/host_cache.h"

#include <algorithm>
#include <chrono>
#include <limits>
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

// Fork features that the current context cache and scheduler do not carry yet. Each is refused
// before the device is initialized or anything loads, so a configuration that asks for one never
// silently runs without it.
void reject_unavailable_options(const EngineOptions& options) {
    if (options.devices.size() > 1 || !options.stage_layers.empty()) {
        throw std::invalid_argument(
            "multi-GPU pipeline stages are not available on this build yet; use one device");
    }
    if (options.context_store.enabled()) {
        throw std::invalid_argument("the context store is not available on this build yet");
    }
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    reject_unavailable_options(options);
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
        options.context_cache        = ContextCacheOptions{
                   .enabled = false, .device_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    // Each direct graft is installed as a pinned context-cache checkpoint that holds one StateImage
    // for the Engine's life, so the Device state pool grows by one per graft.
    const auto direct_grafts = static_cast<std::uint32_t>(
        models::qwen3_5::direct_graft_slots(options.grafts).size());
    if (direct_grafts != 0 && !cache.enabled) {
        throw std::invalid_argument(
            "direct prompt grafts are held in the context cache, which is disabled");
    }
    if (cache.host_cache_percent) {
        if (!cache.auto_host_cache) {
            throw std::invalid_argument("host_cache_percent needs auto_host_cache");
        }
        if (*cache.host_cache_percent == 0 || *cache.host_cache_percent > 100) {
            throw std::invalid_argument("host_cache_percent must be in [1,100]");
        }
    }
    if (cache.host_cache_max_bytes && !cache.auto_host_cache) {
        throw std::invalid_argument("host_cache_max_bytes needs auto_host_cache");
    }
    if (cache.auto_host_cache && cache.host_capacity_bytes) {
        throw std::invalid_argument(
            "auto_host_cache sizes the Host context capacity; leave host_capacity_bytes unset");
    }
    cache.device_state_slots = cache.device_state_slots.value_or(concurrency) + direct_grafts;
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(), {.chat_template_path       = options.chat_template_path,
                               .grammar_cache_bytes      = options.grammar_cache_bytes,
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

void ModelInstance::install_external_checkpoints() {
    const auto& grafts = frontend.grafts();
    if (external_checkpoints.empty()) {
        for (const auto& graft : grafts) {
            if (graft.kind != models::qwen3_5::GraftKind::PrefillKV) {
                external_checkpoints.push_back(program->install_external_checkpoint(graft));
            }
        }
        return;
    }
    // Reinstallation keeps each graft's position; only a handle the Program no longer holds is
    // installed again.
    std::size_t index = 0;
    for (const auto& graft : grafts) {
        if (graft.kind == models::qwen3_5::GraftKind::PrefillKV) { continue; }
        auto& handle = external_checkpoints.at(index++);
        if (!program->valid_checkpoint(handle)) {
            handle = program->install_external_checkpoint(graft);
        }
    }
}

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
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
    // Overlay Vision: size the encode window and pin the borrowable weight tail's mirror before
    // anything measures free device memory, so the KV capacity sees the final picture.
    const std::size_t overlay_window_bytes =
        models::qwen3_5::prepare_vision_overlay(instance->parameters, device, options);
    if (options.context_cache.auto_host_cache) {
        // Sized here, after the weights are loaded and pinned, from the host memory still free;
        // planning then treats it exactly like an explicit capacity. Host capacity does not enter
        // the device plan.
        const std::optional<std::uint64_t> available = available_host_memory_bytes();
        if (!available) {
            throw std::runtime_error(
                "cannot determine the available host memory for --auto-host-cache; set "
                "--host-context-mib explicitly");
        }
        HostCacheEnvironment environment;
        if (options.context_cache.host_cache_percent) {
            environment.total_host_bytes = total_host_memory_bytes();
            if (!environment.total_host_bytes) {
                throw std::runtime_error(
                    "cannot determine the total host memory for --host-cache-percent; use "
                    "--host-cache-max-mib instead");
            }
        }
        // Host memory that still grows after this point and is bounded only by options: the
        // decoded media cache and the media in flight. Counted in full so a vision workload
        // cannot outgrow the margin the pinned tier leaves.
        if (options.enable_vision) {
            environment.extra_reserve_bytes =
                options.media_cache_bytes >
                        std::numeric_limits<std::uint64_t>::max() - options.media_live_bytes
                    ? std::numeric_limits<std::uint64_t>::max()
                    : static_cast<std::uint64_t>(options.media_cache_bytes) +
                          options.media_live_bytes;
        }
        options.context_cache.host_capacity_bytes =
            resolve_host_capacity_bytes(options.context_cache, *available, environment);
        options.context_cache.auto_host_cache = false;
    }
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner    = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          current_free_device_bytes());
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
    instance->kv_capacity_resolution          = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    instance->install_external_checkpoints();
    device.synchronize();
    program.complete();
    // What the Program actually pinned.
    options.context_cache.host_capacity_bytes =
        instance->program->physical_usage().capacity.host_bytes;
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
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
