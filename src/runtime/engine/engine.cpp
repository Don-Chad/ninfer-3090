#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    if (options.devices.empty()) {
        DeviceContext device(options.device);
        phase.complete();
        return device;
    }
#ifdef _WIN32
    // Multi-GPU execution is a Linux feature. Repeating one device id still works on Windows, which
    // exercises the whole stage path on a single card.
    for (const int id : options.devices) {
        if (id != options.devices.front()) {
            throw std::invalid_argument(
                "multi-GPU execution is supported on Linux only; repeat one device id "
                "(for example --devices 0,0) to test the pipeline on a single GPU");
        }
    }
#endif
    DeviceContext device{std::span<const int>(options.devices)};
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    resolved.constraint                        = options.constraint;
    resolved.tool_choice                       = std::move(options.tool_choice);
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

// What a stored checkpoint image binds to: the model, its weight formats and quantization, the
// artifact file, the KV and speculative settings its layout was built for, and the image format.
// Images under any other binding are never read, so they age out of the store as misses.
std::string context_store_binding(const EngineOptions& options, const LoadSummary& load) {
    std::string binding = load.architecture + '\n' + load.model_name + '\n';
    for (const std::string& format : load.weight_formats) { binding += format + ','; }
    binding += '\n' + load.prefill_signature + '\n';
    binding += std::to_string(static_cast<unsigned>(options.kv_cache)) + ',' +
               std::to_string(static_cast<unsigned>(options.speculative.backend)) + ',' +
               std::to_string(options.speculative.draft_tokens) + ',' +
               std::to_string(static_cast<unsigned>(options.speculative.proposal_head)) + ',' +
               (options.enable_vision ? "vision" : "text") + '\n';
    std::error_code size_error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, size_error);
    binding += size_error ? std::string("?") : std::to_string(size);
    // A re-converted artifact can have the same size; its modification time tells it apart.
    std::error_code time_error;
    const auto written = std::filesystem::last_write_time(options.artifact_path, time_error);
    binding += '\n';
    binding += time_error ? std::string("?") : std::to_string(written.time_since_epoch().count());
    // Prompt grafts: a stored session that starts from a graft holds that graft's installed K/V and
    // state under its placeholder ids, so a changed graft file must not match it. Any change to
    // the configured grafts makes the store's images misses.
    for (const GraftSource& graft : options.grafts) {
        std::error_code graft_size_error;
        std::error_code graft_time_error;
        const std::uintmax_t graft_size = std::filesystem::file_size(graft.path, graft_size_error);
        const auto graft_written = std::filesystem::last_write_time(graft.path, graft_time_error);
        binding += "\ngraft " + graft.name + ' ' +
                   (graft_size_error ? std::string("?") : std::to_string(graft_size)) + ' ' +
                   (graft_time_error ? std::string("?")
                                     : std::to_string(graft_written.time_since_epoch().count()));
    }
    // The checkpoint image format; stores written by an older engine become misses.
    binding += "\ncheckpoint-image-1";
    return binding;
}

// Images queued for the writer thread. Each holds a whole continuation, several GB for a deep one;
// beyond the bound an idle continuation is simply written later.
constexpr std::size_t kMaximumPendingStoreWrites = 2;

// The store keeps up to this share of the volume's free space when no limit is configured.
constexpr std::uint64_t kAutomaticStoreShareDivisor = 2;
constexpr std::uint64_t kFallbackStoreBytes         = std::uint64_t{64} << 30U;

std::uint64_t directory_bytes(const std::filesystem::path& directory) noexcept {
    std::uint64_t total = 0;
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator it(directory, error), end;
         !error && it != end; it.increment(error)) {
        std::error_code size_error;
        if (it->is_regular_file(size_error)) {
            const auto size = it->file_size(size_error);
            if (!size_error) { total += size; }
        }
    }
    return total;
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         models::qwen3_5::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    models::qwen3_5::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
        virtual std::optional<std::uint32_t> effective_thinking_budget() const noexcept       = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

        std::optional<std::uint32_t> effective_thinking_budget() const noexcept override {
            return submission_.effective_thinking_budget();
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

    [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
        return state_->effective_thinking_budget();
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

std::optional<std::uint32_t> GenerationHandle::effective_thinking_budget() const noexcept {
    return impl_ != nullptr ? impl_->effective_thinking_budget() : std::nullopt;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using GenerationCore = runtime::EngineCore<runtime::ModelInstance>;
    using ScoringCore    = runtime::CausalScoreCore<runtime::ModelInstance>;
    using Core =
        std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>>;

    explicit Impl(EngineOptions engine_options)
        : options(runtime::normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        auto constructed    = runtime::construct_model(options, device);
        active              = std::move(constructed.instance);
        load                = std::move(constructed.load);
        load.cuda_sync_mode = device.sync_mode();
        sampling_defaults   = active->frontend.sampling_defaults();
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        if (options.purpose == EnginePurpose::CausalScoring) {
            core = std::make_unique<ScoringCore>(*active, device);
        } else {
            auto generation = std::make_unique<GenerationCore>(
                *active, device, options, std::move(constructed.context_cost));
            if (options.context_store.enabled()) {
                if (!options.context_cache.enabled) {
                    throw std::invalid_argument(
                        "the context store requires the context cache to be enabled");
                }
                if (options.speculative.backend == SpeculativeBackend::DFlash ||
                    options.speculative.backend == SpeculativeBackend::DFlash2) {
                    // Their draft-side state round trip through a stored image is unverified.
                    throw std::invalid_argument(
                        "the context store does not support the DFlash speculative backends");
                }
                if (active->program->physical_usage().capacity.host_bytes == 0) {
                    throw std::invalid_argument(
                        "the context store restores sessions into the Host context cache; give it "
                        "a nonzero Host budget (--host-context-mib or --auto-host-cache)");
                }
                open_context_store();
                using Hooks = GenerationCore::ContextStoreHooks;
                generation->set_context_store(Hooks{
                    .store   = store.get(),
                    .binding = store_binding,
                    .sink    = [this](CheckpointImage&& image) {
                        return enqueue_store_write(std::move(image));
                    },
                    .ready    = [this] { return store_queue_idle(); },
                    .failures = [this] {
                        return store_write_failures.load(std::memory_order_relaxed);
                    },
                    .idle_persist = std::chrono::duration_cast<std::chrono::milliseconds>(
                        options.context_store.idle_persist),
                    .hydration_budget = std::chrono::duration_cast<std::chrono::milliseconds>(
                        options.context_store.restore_budget),
                });
            }
            core = std::move(generation);
            if (store) { restore_from_store(); }
        }
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        // Queued writes are older than what the final flush writes: drain them first, so none
        // lands after it and replaces a newer image.
        if (store) {
            const auto deadline =
                std::chrono::steady_clock::now() + options.context_store.flush_budget;
            stop_writer(deadline);
            flush_context_store(deadline);
        }
        core.emplace<std::monostate>();
        // Everything written is queued for upload by now; give the uploads their budget.
        if (store) {
            (void)store->flush_remote(std::chrono::steady_clock::now() +
                                          options.context_store.remote_flush_budget,
                                      true);
        }
        try {
            device.synchronize();
        } catch (...) {}
    }

    // Adds the context store's counters to a RuntimeStats snapshot (zero when it is off).
    void add_store_stats(RuntimeStats& out) const {
        if (!store) { return; }
        const runtime::ContextStore::Stats s = store->stats();
        out.context_store_images          = s.images;
        out.context_store_used_bytes      = s.used_bytes;
        out.context_store_writes          = s.puts;
        out.context_store_write_failures  = s.put_failures;
        out.context_store_dropped         = store_dropped.load(std::memory_order_relaxed);
        out.context_store_bytes_written   = s.bytes_written;
        out.context_store_bytes_reused    = s.bytes_reused;
        out.context_store_evicted         = s.evicted_for_space + s.expired + s.superseded;
        out.context_store_corrupt         = s.corrupt_removed;
        out.context_store_restored        = restored_sessions;
        out.context_store_restored_bytes  = restored_bytes;
        out.context_store_restore_seconds = restore_seconds;
        out.context_store_remote_images            = s.remote_images;
        out.context_store_remote_uploads           = s.remote_uploads;
        out.context_store_remote_upload_bytes      = s.remote_upload_bytes;
        out.context_store_remote_upload_failures   = s.remote_upload_failures;
        out.context_store_remote_downloads         = s.remote_downloads;
        out.context_store_remote_download_bytes    = s.remote_download_bytes;
        out.context_store_remote_download_failures = s.remote_download_failures;
    }

    using CheckpointImage = runtime::ModelInstance::ModelContract::CheckpointImage;

    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::ModelInstance> active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    // Declared before `core`: the worker holds hooks that reach the store and the writer queue,
    // so they must outlive it.
    std::unique_ptr<runtime::ContextStore> store;
    std::string store_binding;
    std::uint64_t restored_sessions = 0;
    std::uint64_t restored_bytes    = 0;
    double restore_seconds          = 0.0;
    std::atomic<std::uint64_t> store_dropped{0};
    std::atomic<std::uint64_t> store_write_failures{0};
    std::mutex writer_mutex;
    std::condition_variable writer_cv;
    std::deque<CheckpointImage> pending_writes;
    bool writer_active = false;
    bool writer_stop   = false;
    std::optional<std::chrono::steady_clock::time_point> drain_deadline;
    std::thread writer;
    Core core;

private:
    void open_context_store() {
        const ContextStoreOptions& config = options.context_store;
        std::error_code error;
        std::filesystem::create_directories(config.directory, error);
        if (!std::filesystem::is_directory(config.directory)) {
            throw std::invalid_argument("the context store directory is not usable: " +
                                        config.directory.string());
        }
        std::uint64_t max_bytes = config.max_bytes;
        if (max_bytes == 0) {
            const std::filesystem::space_info space =
                std::filesystem::space(config.directory, error);
            // If the volume cannot be queried, fall back to a bounded default rather than no limit.
            max_bytes = error ? kFallbackStoreBytes
                              : (space.available + directory_bytes(config.directory)) /
                                    kAutomaticStoreShareDivisor;
        }
        runtime::ContextStore::Options store_options;
        store_options.directory = config.directory;
        store_options.max_bytes = max_bytes;
        store_options.ttl       = config.ttl;
        store_options.remote        = config.remote;
        store_options.remote_prefix = config.remote_prefix;
        store         = std::make_unique<runtime::ContextStore>(std::move(store_options));
        if (config.remote) {
            // Images other engines left in the bucket become restorable before start-up restores.
            (void)store->refresh_remote(std::chrono::steady_clock::now() + config.restore_budget);
        }
        store_binding = context_store_binding(options, load);
    }

    // Lowercase hex of the image's endpoint key: an image of the same endpoint replaces it.
    static std::string image_id(const CheckpointImage& image) {
        const auto& key = image.keys.front();
        char text[41];
        std::snprintf(text, sizeof(text), "%016llx%016llx%08x",
                      static_cast<unsigned long long>(key.digests[0]),
                      static_cast<unsigned long long>(key.digests[1]),
                      static_cast<unsigned>(key.frontier));
        return text;
    }

    // Writes one image to the store. On the writer thread, or at shutdown.
    void store_put(const CheckpointImage& image) {
        runtime::ContextStore::Description description;
        description.id      = image_id(image);
        description.binding = store_binding;
        description.tokens  = image.tokens;
        description.checkpoints.reserve(image.keys.size());
        for (const auto& key : image.keys) {
            description.checkpoints.push_back(runtime::ContextStore::CheckpointKey{
                .frontier = key.frontier, .digests = key.digests, .identity_tag = key.identity_tag});
        }
        description.prefix_digests = image.prefix_digests;
        std::vector<runtime::ContextStore::Region> regions;
        regions.reserve(image.regions.size());
        for (const auto& region : image.regions) {
            regions.push_back({.offset = region.offset, .length = region.length});
        }
        (void)store->put(description, image.bytes, regions);
    }

    // Restores the most recently used stored sessions into the empty cache, within the configured
    // time and the free Host capacity. Anything that cannot be restored stays in the store.
    void restore_from_store() noexcept {
        const auto started = std::chrono::steady_clock::now();
        try {
            auto& generation    = *std::get<std::unique_ptr<GenerationCore>>(core);
            const auto deadline = started + options.context_store.restore_budget;
            auto images         = store->list();
            std::sort(images.begin(), images.end(), [](const auto& a, const auto& b) {
                return a.last_used_ms > b.last_used_ms;
            });
            for (const runtime::ContextStore::Info& info : images) {
                if (std::chrono::steady_clock::now() >= deadline) { break; }
                if (info.binding != store_binding || info.tokens > options.max_context) {
                    continue;
                }
                // The payload is nearly all of an image: one larger than the free Host space
                // cannot fit, and reading it would only spend the budget.
                if (info.image_bytes > generation.free_host_bytes()) { continue; }
                std::optional<std::vector<std::uint8_t>> bytes = store->load(info.id, false);
                if (!bytes) { continue; }
                if (generation.restore_image(*bytes)) {
                    store->touch(info.id);
                    ++restored_sessions;
                    restored_bytes += bytes->size();
                }
            }
        } catch (...) {}
        restore_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    // At shutdown: everything the store does not hold in its current state, most recently used
    // first, within the flush budget.
    void flush_context_store(std::chrono::steady_clock::time_point deadline) noexcept {
        try {
            auto* generation = std::get_if<std::unique_ptr<GenerationCore>>(&core);
            if (generation == nullptr || *generation == nullptr) { return; }
            (void)(*generation)->persist_all(
                [this](CheckpointImage&& image) {
                    try {
                        store_put(image);
                        return true;
                    } catch (...) { return false; }
                },
                deadline);
        } catch (...) {}
    }

    [[nodiscard]] bool store_queue_idle() {
        std::scoped_lock lock(writer_mutex);
        return pending_writes.empty() && !writer_active;
    }

    // Queues an image for the writer thread. Never blocks: when the queue is full the image is
    // not written now, and its continuation is tried again later.
    bool enqueue_store_write(CheckpointImage&& image) {
        std::unique_lock lock(writer_mutex);
        if (writer_stop) { return false; } // shutting down: the final flush writes it instead
        if (pending_writes.size() >= kMaximumPendingStoreWrites) {
            store_dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!writer.joinable()) { writer = std::thread([this] { writer_loop(); }); }
        pending_writes.push_back(std::move(image));
        lock.unlock();
        writer_cv.notify_one();
        return true;
    }

    void writer_loop() noexcept {
        for (;;) {
            CheckpointImage image;
            {
                std::unique_lock lock(writer_mutex);
                writer_cv.wait(lock, [&] { return writer_stop || !pending_writes.empty(); });
                if (pending_writes.empty() ||
                    (writer_stop && drain_deadline &&
                     std::chrono::steady_clock::now() >= *drain_deadline)) {
                    return;
                }
                image = std::move(pending_writes.front());
                pending_writes.pop_front();
                writer_active = true;
            }
            ContextStoreWriteEvent event;
            event.id           = image_id(image);
            event.tokens       = image.tokens;
            event.bytes        = image.bytes.size();
            const auto started = std::chrono::steady_clock::now();
            try {
                store_put(image);
            } catch (const std::exception& error) {
                event.error = error.what();
            } catch (...) { event.error = "unknown context store write failure"; }
            if (!event.error.empty()) {
                // The continuation was counted as stored when it was queued; have it written again.
                store_write_failures.fetch_add(1, std::memory_order_relaxed);
            }
            event.seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            notify(event);
            {
                std::scoped_lock lock(writer_mutex);
                writer_active = false;
            }
        }
    }

    // Reports one background write to the configured listener; its exceptions are ignored.
    void notify(const ContextStoreWriteEvent& event) const noexcept {
        if (!options.context_store.listener) { return; }
        try {
            options.context_store.listener(event);
        } catch (...) {}
    }

    // Lets the writer finish the queued images until `deadline`, then stops it.
    void stop_writer(std::chrono::steady_clock::time_point deadline) noexcept {
        {
            std::scoped_lock lock(writer_mutex);
            writer_stop    = true;
            drain_deadline = deadline;
        }
        writer_cv.notify_all();
        if (writer.joinable()) { writer.join(); }
        std::scoped_lock lock(writer_mutex);
        pending_writes.clear();
    }
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared      = impl_->active->frontend.prepare(std::move(input), control);
    PromptSummary info = prepared.summary();
    const SamplingMode sampling_mode =
        info.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(info, preparation, sampling_mode,
                                                                 std::move(prepared)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (token_ids.size() > impl_->active->capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(token_ids.size(), impl_->active->capacity));
    }
    auto prepared =
        impl_->active->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, preparation, SamplingMode::Thinking, std::move(prepared)));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.tokenize_text(text);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                return core->score(std::move(prompt.impl_->value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (const auto external = models::qwen3_5::PreparedPromptAccess::view(prompt.impl_->value)
                                  .external_prefix_tokens;
        external != 0 && (!resolved_options.execution.allow_prefix_reuse ||
                          external >= prompt_summary.prompt_tokens)) {
        // A direct graft exists only as its installed checkpoint, which only reuse can bind.
        throw std::invalid_argument(
            "a request selecting a direct prompt graft must allow prefix reuse and add tokens "
            "after the graft");
    }
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }

            [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
                return result.thinking.effective_budget;
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                    = prompt_summary;
        immediate.result.finish_reason             = FinishReason::OutputLimit;
        immediate.result.thinking.requested_budget = resolved_options.execution.thinking.budget;
        // No output is licensed, so the cap never binds: effective equals requested.
        immediate.result.thinking.effective_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds   = prepare_seconds;
        immediate.result.timings.total_seconds     = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission =
                    core->submit(std::move(prompt.impl_->value), prompt_summary, prepare_seconds,
                                 std::move(resolved_options), consumer_mode, std::move(observation),
                                 pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.media_cache_summary();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    RuntimeStats stats = std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
    impl_->add_store_stats(stats);
    return stats;
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

} // namespace ninfer
