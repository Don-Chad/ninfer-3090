#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/context_store/context_store.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
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
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
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

namespace {

// What a session snapshot binds to: the model, its weight formats and quantization, and the
// artifact size, since the prefill signature describes weight geometry but not weight values.
std::string slot_model_binding(const EngineOptions& options, const LoadSummary& load) {
    std::string binding = load.architecture + '\n' + load.model_name + '\n';
    for (const std::string& format : load.weight_formats) { binding += format + ','; }
    binding += '\n' + load.prefill_signature + '\n';
    // What a snapshot's restore checks beyond the artifact: the KV storage and the speculative
    // configuration its state and KV layout were built for. Part of the binding so an image made
    // under another setting is filtered out before anything is read or evicted for it.
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
    binding += time_error ? std::string("?")
                          : std::to_string(written.time_since_epoch().count());
    return binding;
}

// The writer thread queues at most this many sessions. Each holds a whole session, several GB for a
// deep one; beyond the bound a write is dropped and counted rather than growing host memory.
constexpr std::size_t kMaximumPendingStoreWrites = 2;
constexpr std::chrono::minutes kStoreMaintenanceInterval{10};

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
        auto constructed  = runtime::construct_model(options, device);
        active            = std::move(constructed.instance);
        load              = std::move(constructed.load);
        options.context_cache = std::move(constructed.context_cache);
        sampling_defaults = active->frontend.sampling_defaults();
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
                if (options.speculative.backend == SpeculativeBackend::DFlash) {
                    // Session snapshots do not capture this backend's lane-local state.
                    throw std::invalid_argument(
                        "the context store does not support the DFlash speculative backend");
                }
                open_context_store();
                generation->set_context_store(
                    slot_model_binding(options, load), store.get(),
                    [this](runtime::ModelInstance::ModelContract::SessionSnapshot&& snapshot) {
                        return enqueue_store_write(std::move(snapshot));
                    },
                    [this] { return store_queue_idle(); },
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        options.context_store.idle_persist),
                    [this] { return store_write_failures.load(std::memory_order_relaxed); });
                start_writer();
            }
            core = std::move(generation);
            if (store) { restore_from_store(); }
        }
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        // Queued store writes are older than what the final flush writes: drain them first, so
        // none lands after it and replaces a newer image.
        std::chrono::steady_clock::time_point flush_deadline{};
        if (store) {
            flush_deadline = std::chrono::steady_clock::now() + options.context_store.flush_budget;
            {
                std::scoped_lock lock(writer_mutex);
                drain_deadline = flush_deadline;
            }
            stop_writer();
        }
        flush_context_store(flush_deadline);
        core.emplace<std::monostate>();
        stop_writer();
        // Everything written is queued for upload by now; give the uploads their budget.
        if (store) {
            (void)store->flush_remote(std::chrono::steady_clock::now() +
                                      options.context_store.remote_flush_budget);
        }
        try {
            device.synchronize();
        } catch (...) {}
    }

    [[nodiscard]] GenerationCore& generation_core() {
        auto* generation = std::get_if<std::unique_ptr<GenerationCore>>(&core);
        if (generation == nullptr || *generation == nullptr) {
            throw std::logic_error("session persistence requires a generation Engine");
        }
        return **generation;
    }

    [[nodiscard]] const GenerationCore& generation_core() const {
        const auto* generation = std::get_if<std::unique_ptr<GenerationCore>>(&core);
        if (generation == nullptr || *generation == nullptr) {
            throw std::logic_error("session persistence requires a generation Engine");
        }
        return **generation;
    }

    // Adds the context store's counters to a RuntimeStats snapshot (all zero when it is disabled).
    void add_store_stats(RuntimeStats& out) const {
        if (!store) { return; }
        const runtime::ContextStore::Stats s = store->stats();
        out.context_store_images         = s.images;
        out.context_store_used_bytes     = s.used_bytes;
        out.context_store_writes         = s.puts;
        out.context_store_write_failures = s.put_failures;
        out.context_store_dropped        = store_dropped.load(std::memory_order_relaxed);
        out.context_store_bytes_written  = s.bytes_written;
        out.context_store_bytes_reused   = s.bytes_reused;
        out.context_store_evicted        = s.evicted_for_space + s.expired + s.superseded;
        out.context_store_corrupt        = s.corrupt_removed;
        out.context_store_restored       = restored_sessions.load(std::memory_order_relaxed);
        out.context_store_restored_bytes = restored_bytes.load(std::memory_order_relaxed);
        out.context_store_restore_seconds = restore_seconds;
        out.context_store_remote_images           = s.remote_images;
        out.context_store_remote_uploads          = s.remote_uploads;
        out.context_store_remote_upload_bytes     = s.remote_upload_bytes;
        out.context_store_remote_upload_failures  = s.remote_upload_failures;
        out.context_store_remote_downloads        = s.remote_downloads;
        out.context_store_remote_download_bytes   = s.remote_download_bytes;
        out.context_store_remote_download_failures = s.remote_download_failures;
        if (const auto* generation = std::get_if<std::unique_ptr<GenerationCore>>(&core);
            generation != nullptr && *generation != nullptr) {
            const auto read = (*generation)->store_read_stats();
            out.context_store_hydrations         = read.hydrations;
            out.context_store_hydrated_tokens    = read.hydrated_tokens;
            out.context_store_hydration_failures = read.failures;
            out.context_store_hydration_seconds  = read.seconds;
        }
    }

    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::ModelInstance> active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    // Declared before `core`: the worker holds sinks that reach the store, so the store must
    // outlive it.
    std::unique_ptr<runtime::ContextStore> store;
    std::string store_binding;
    Core core;

private:
    struct PendingWrite {
        runtime::ModelInstance::ModelContract::SessionSnapshot snapshot;
    };

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
            const std::filesystem::space_info space = std::filesystem::space(config.directory, error);
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
        store                   = std::make_unique<runtime::ContextStore>(std::move(store_options));
        store_binding           = slot_model_binding(options, load);
        // Register what other engines left in the bucket before start-up restores sessions; an
        // unreachable bucket costs the connection timeouts, then the store runs on its directory.
        if (config.remote) {
            (void)store->refresh_remote(std::chrono::steady_clock::now() + config.restore_budget);
        }
    }

    // Writes one session to the context store. Called on the writer thread, or at shutdown.
    void store_put(const runtime::ModelInstance::ModelContract::SessionSnapshot& snapshot) {
        runtime::ContextStore::Description description;
        description.id      = snapshot.session_digest;
        description.binding = store_binding;
        description.tokens  = snapshot.tokens;
        description.checkpoints.reserve(snapshot.checkpoints.size());
        for (const auto& key : snapshot.checkpoints) {
            description.checkpoints.push_back(runtime::ContextStore::CheckpointKey{
                .frontier = key.frontier, .digests = key.digests, .identity_tag = key.identity_tag});
        }
        description.prefix_digests = snapshot.prefix_digests;
        std::vector<runtime::ContextStore::Region> regions;
        regions.reserve(snapshot.regions.size());
        for (const auto& region : snapshot.regions) {
            regions.push_back(
                runtime::ContextStore::Region{.offset = region.offset, .length = region.length});
        }
        (void)store->put(description, snapshot.bytes, regions);
    }

    // Restores the most recently used stored sessions into the empty cache, within the configured
    // time and capacity. Anything that cannot be restored is left in the store.
    void restore_from_store() noexcept {
        const auto started = std::chrono::steady_clock::now();
        try {
            const auto deadline = started + options.context_store.restore_budget;
            for (const runtime::ContextStore::Info& info : store->list()) {
                if (std::chrono::steady_clock::now() >= deadline) { break; }
                if (info.binding != store_binding) { continue; }
                const std::optional<std::uint32_t> slot = generation_core().first_vacant_slot();
                if (!slot) { break; }
                std::optional<std::vector<std::uint8_t>> bytes = store->load(info.id, false);
                if (!bytes) { continue; }
                try {
                    generation_core().restore_session(
                        *slot, std::span<const std::uint8_t>(bytes->data(), bytes->size()),
                        store_binding);
                    store->touch(info.id);
                    restored_sessions.fetch_add(1, std::memory_order_relaxed);
                    restored_bytes.fetch_add(bytes->size(), std::memory_order_relaxed);
                } catch (const RequestError&) {
                    // No idle lane or an open transaction: nothing more can be restored now.
                    break;
                } catch (const std::invalid_argument&) {
                    // This session does not fit the free capacity; a smaller, older one still may.
                }
            }
        } catch (...) {}
        restore_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    // At shutdown, before the worker stops: everything the store does not hold in its current
    // state, most recently used first, within the flush budget.
    void flush_context_store(std::chrono::steady_clock::time_point deadline) noexcept {
        if (!store) { return; }
        try {
            auto* generation = std::get_if<std::unique_ptr<GenerationCore>>(&core);
            if (generation == nullptr || *generation == nullptr) { return; }
            (void)(*generation)->persist_all(
                [this](runtime::ModelInstance::ModelContract::SessionSnapshot&& snapshot) {
                    try {
                        store_put(snapshot);
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

    // Queues a session for the writer thread. Never blocks: when the queue is full the session is
    // not written now (an evicted one is lost to the store, an idle one is tried again later).
    bool enqueue_store_write(runtime::ModelInstance::ModelContract::SessionSnapshot&& snapshot) {
        std::unique_lock lock(writer_mutex);
        if (writer_stop) { return false; } // shutting down: the final flush writes it instead
        if (pending_writes.size() >= kMaximumPendingStoreWrites) {
            store_dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!writer.joinable()) { writer = std::thread([this] { writer_loop(); }); }
        pending_writes.push_back(PendingWrite{std::move(snapshot)});
        lock.unlock();
        writer_cv.notify_one();
        return true;
    }

    void notify(const ContextStoreWriteEvent& event) const noexcept {
        if (!options.context_store.listener) { return; }
        try {
            options.context_store.listener(event);
        } catch (...) {}
    }

    void write_pending(PendingWrite& item) {
        ContextStoreWriteEvent event;
        event.id           = item.snapshot.session_digest;
        event.tokens       = item.snapshot.tokens;
        event.bytes        = item.snapshot.bytes.size();
        const auto started = std::chrono::steady_clock::now();
        try {
            try {
                store_put(item.snapshot);
            } catch (...) {
                // The session was counted as stored when it was queued; have it queued again.
                store_write_failures.fetch_add(1, std::memory_order_relaxed);
                throw;
            }
        } catch (const std::exception& error) {
            event.error = error.what();
        } catch (...) { event.error = "unknown context store write failure"; }
        event.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        notify(event);
    }

    void start_writer() {
        std::scoped_lock lock(writer_mutex);
        if (!writer.joinable()) { writer = std::thread([this] { writer_loop(); }); }
    }

    void writer_loop() {
        for (;;) {
            {
                std::unique_lock lock(writer_mutex);
                const auto ready = [this] { return writer_stop || !pending_writes.empty(); };
                if (store) {
                    // A quiet server still ages sessions out: the store only trims after a write.
                    if (!writer_cv.wait_for(lock, kStoreMaintenanceInterval, ready)) {
                        lock.unlock();
                        try {
                            store->maintain();
                        } catch (...) {}
                        continue;
                    }
                } else {
                    writer_cv.wait(lock, ready);
                }
                if (pending_writes.empty()) { return; }
            }
            std::optional<PendingWrite> item;
            {
                std::scoped_lock lock(writer_mutex);
                if (pending_writes.empty()) { continue; }
                item.emplace(std::move(pending_writes.front()));
                pending_writes.pop_front();
                if (drain_deadline &&
                    std::chrono::steady_clock::now() >= *drain_deadline) {
                    // Shutdown budget spent: the write is abandoned, like one that failed.
                    store_dropped.fetch_add(1, std::memory_order_relaxed);
                    store_write_failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                writer_active = true;
            }
            write_pending(*item);
            std::scoped_lock lock(writer_mutex);
            writer_active = false;
        }
    }

    // Pending writes are flushed before the thread exits.
    void stop_writer() noexcept {
        {
            std::scoped_lock lock(writer_mutex);
            writer_stop = true;
        }
        writer_cv.notify_all();
        if (writer.joinable()) {
            try {
                writer.join();
            } catch (...) {}
        }
    }

    std::mutex writer_mutex;
    std::condition_variable writer_cv;
    std::deque<PendingWrite> pending_writes;
    bool writer_stop   = false;
    // Set at shutdown: queued store writes still pending after it are abandoned. A write already
    // under way is not interrupted, so the budget can be exceeded by one write.
    std::optional<std::chrono::steady_clock::time_point> drain_deadline;
    bool writer_active = false;
    std::thread writer;
    std::atomic<std::uint64_t> store_dropped{0};
    std::atomic<std::uint64_t> store_write_failures{0};
    std::atomic<std::uint64_t> restored_sessions{0};
    std::atomic<std::uint64_t> restored_bytes{0};
    double restore_seconds = 0.0;
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
    resolved_options.execution.output_reservation_tokens = impl_->options.output_reservation_tokens;
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
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
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
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
                auto submission = core->submit(std::move(prompt.impl_->value), prompt_summary,
                                               prepare_seconds, std::move(resolved_options),
                                               consumer_mode, observation, pending_deadline);
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

std::uint32_t Engine::concurrent_output_budget(const PreparedPrompt& prompt) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("concurrent_output_budget requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    return impl_->active->program->concurrent_output_budget(prompt.impl_->summary.prompt_tokens);
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

std::vector<SlotState> Engine::slot_states() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->generation_core().slot_states();
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
