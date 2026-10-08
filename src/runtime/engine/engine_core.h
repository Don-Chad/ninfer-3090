#pragma once

// Small fixed-capacity request execution for every backend.

#include "core/device.h"
#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/token_constraint.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/context_store/context_store.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/effective_thinking_budget.h"
#include "runtime/engine/worker_fault.h"
#include "runtime/engine/worker_recovery.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class EngineCore {

public:
    using ModelContract      = typename Instance::ModelContract;
    using Program            = typename ModelContract::Program;
    using BasePlan           = typename ModelContract::RequestBasePlan;
    using Plan               = typename ModelContract::AdmissionCandidate;
    using SequenceHandle     = typename ModelContract::SequenceHandle;
    using CaptureOffer       = typename ModelContract::CaptureOffer;
    using PendingBatch       = typename ModelContract::PendingBatch;
    using PreparedPrompt     = typename ModelContract::PreparedPrompt;
    using OutputSession      = typename ModelContract::OutputSession;
    using PublishedOutput    = typename ModelContract::PublishedOutput;
    using Request            = RequestRecord<ModelContract>;
    using Scheduling         = Scheduler<Request>;
    using FifoSnapshot       = typename Scheduling::FifoSnapshot;
    using RoundMembership    = typename Scheduling::RoundMembership;
    using ControlMembership  = typename Scheduling::ControlMembership;
    using ActiveAdmissionSet = typename Scheduling::ActiveAdmissionSet;
    using ExecutionAction    = typename Scheduling::ExecutionAction;
    using AdmissionGrant     = typename Scheduling::AdmissionGrant;
    using ResourceManagement = ResourceManager<ModelContract>;
    using ResourceInspection = typename ResourceManagement::Inspection;
    using Clock              = std::chrono::steady_clock;

    EngineCore(Instance& instance, DeviceContext& device, const EngineOptions& options,
               ContextMachineCostModel context_cost)
        : instance_(instance), device_(device), max_context_(options.max_context),
          max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          context_cache_enabled_(options.context_cache.enabled),
          fault_listener_(options.fault_listener),
          resources_(max_concurrency_, options.context_cache.max_private_continuations.value(),
                     options.context_cache.max_shared_prefixes.value(),
                     options.context_cache.enabled,
                     options.context_cache.max_long_anchors_per_continuation.value_or(0),
                     std::move(context_cost)) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        if (!options.context_cache.max_private_continuations ||
            !options.context_cache.max_shared_prefixes) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        if (options.max_prefill_lanes == 0 || options.max_prefill_lanes > max_concurrency_ ||
            options.prefill_max_skip == 0 || options.decode_rounds_per_prefill == 0) {
            throw std::invalid_argument("Engine prefill lane bounds are invalid");
        }
        scheduler_.configure_prefill(options.max_prefill_lanes, options.prefill_max_skip,
                                     options.decode_rounds_per_prefill);
        catalog_pinned_grafts();
        slot_session_paths_.resize(resources_.catalog_capacity());
        slot_digest_cache_.resize(resources_.catalog_capacity());
        slot_usage_.resize(resources_.catalog_capacity());
        slot_persisted_.resize(resources_.catalog_capacity());
        resources_.set_eviction_observer(
            [this](std::uint32_t slot, const typename ModelContract::ContinuationHandle& handle) {
                spill_catalog_slot(slot, handle);
            });
        // A slot file binding belongs to the session that saved or restored it; the cell is only
        // where that session lives now. Ending the binding with the catalog entry keeps it from
        // being inherited by the next session in the cell. (The usage record is not cleared here:
        // a request that continues a session in place consumes the entry and publishes into the
        // same cell, and the record has to survive that. Nothing needs to clear it: every session
        // enters a cell through a publication or a restore, and each writes the record afresh.)
        resources_.set_slot_release_observer(
            [this](std::uint32_t slot) { clear_slot_session(slot); });
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~EngineCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)),
              effective_thinking_budget_(other.effective_thinking_budget_) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
                effective_thinking_budget_ = other.effective_thinking_budget_;
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCore* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

        // Fixed when the request was admitted; never reads the session the worker mutates.
        [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
            return effective_thinking_budget_;
        }

    private:
        Submission(EngineCore& owner, std::shared_ptr<Request> request,
                   std::optional<std::uint32_t> effective_thinking_budget) noexcept
            : owner_(&owner), request_(std::move(request)),
              effective_thinking_budget_(effective_thinking_budget) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;
        std::optional<std::uint32_t> effective_thinking_budget_;

        friend class EngineCore;
    };

    Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id        = 0;
        std::uint64_t publication_order = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            if (next_request_id_ == 0 || next_publication_order_ == 0) {
                throw std::overflow_error("request identity space exhausted");
            }
            ++outstanding_;
            request_id        = next_request_id_++;
            publication_order = next_publication_order_++;
        }

        std::shared_ptr<Request> request;
        std::optional<std::uint32_t> effective_budget;
        try {
            apply_effective_thinking_budget(
                options.execution.thinking,
                effective_output_capacity(options.execution.requested_output_tokens, max_context_,
                                          prompt_summary.prompt_tokens),
                instance_.frontend.thinking_control_token_count());
            auto output = instance_.frontend.make_output_session(
                prompt, options.stop, options.output, options.execution.thinking);
            request = std::make_shared<Request>(request_id, publication_order, std::move(prompt),
                                                std::move(output), prompt_summary, prepare_seconds,
                                                std::move(options), consumer_mode, observation,
                                                pending_deadline, submitted);
            effective_budget = request->output.thinking_stats().effective_budget;
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request), effective_budget);
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    // Session persistence. A slot is one private catalog cell. Each entry point takes the
    // execution mutex, so its copies run between units, and refuses (Overloaded) rather than
    // waits when a resource transaction is open or the cell is in use. A non-empty
    // expected_digest is a precondition on the cell's resident session, checked atomically with
    // the operation. A successful save or restore binds the cell to session_path so an
    // involuntary eviction can write the session back there (see spill_catalog_slot).
    // `claim` runs under the execution mutex once the operation has succeeded, so the caller can
    // mark every spill queued before it as superseded and every spill queued after it as newer.
    [[nodiscard]] typename ModelContract::SessionSnapshot
    save_slot(std::uint32_t slot, std::string_view model_binding,
              std::string_view expected_digest, std::string_view session_path,
              const std::function<void()>& claim) {
        std::scoped_lock lock(execution_mutex_);
        device_.bind_to_current_thread();
        require_settled_slot(slot);
        const auto view = resources_.catalog_slot(slot);
        if (view.state != ResourceManagement::CatalogState::Catalogued || view.handle == nullptr) {
            throw std::invalid_argument("slot holds no retained session");
        }
        require_session_digest(view, expected_digest);
        auto snapshot = instance_.program->save_continuation(*view.handle, model_binding);
        bind_slot_session(slot, session_path);
        claim();
        // So `GET /slots` shows the binding now, not at the next unit boundary (an idle engine
        // may not have one for a while).
        publish_runtime_stats();
        return snapshot;
    }

    // A cell that holds no session and is not reserved by a request, for restoring into.
    [[nodiscard]] std::optional<std::uint32_t> first_vacant_slot() const {
        std::scoped_lock lock(execution_mutex_);
        return first_vacant_slot_locked();
    }
    [[nodiscard]] std::optional<std::uint32_t> first_vacant_slot_locked() const {
        for (std::uint32_t slot = 0; slot < resources_.catalog_capacity(); ++slot) {
            if (resources_.catalog_state(slot) == ResourceManagement::CatalogState::Vacant) {
                return slot;
            }
        }
        return std::nullopt;
    }

    // `already_durable` marks a session restored from the context store: it is by definition
    // already stored, so the store is not written again until the session changes.
    [[nodiscard]] std::pair<std::uint32_t, std::string>
    restore_slot(std::uint32_t slot, std::span<const std::uint8_t> snapshot,
                 std::string_view model_binding, std::string_view session_path,
                 const std::function<void()>& claim, bool already_durable = false) {
        std::scoped_lock lock(execution_mutex_);
        return restore_slot_locked(slot, snapshot, model_binding, session_path, claim,
                                   already_durable);
    }

    struct StoreReadStats {
        std::uint64_t hydrations     = 0;
        std::uint64_t hydrated_tokens = 0;
        std::uint64_t failures       = 0;
        double seconds               = 0.0;
    };
    [[nodiscard]] StoreReadStats store_read_stats() const noexcept {
        return {store_hydrations_.load(std::memory_order_relaxed),
                store_hydrated_tokens_.load(std::memory_order_relaxed),
                store_hydration_failures_.load(std::memory_order_relaxed),
                static_cast<double>(store_hydration_ns_.load(std::memory_order_relaxed)) * 1e-9};
    }

    // restore_slot for a caller that already holds the execution mutex (the worker).
    [[nodiscard]] std::pair<std::uint32_t, std::string>
    restore_slot_locked(std::uint32_t slot, std::span<const std::uint8_t> snapshot,
                        std::string_view model_binding, std::string_view session_path,
                        const std::function<void()>& claim, bool already_durable = false) {
        device_.bind_to_current_thread();
        if (!context_cache_enabled_) {
            // Without the cache no finished request can reuse a retained session.
            throw std::invalid_argument("session restore requires the context cache to be enabled");
        }
        require_settled_slot(slot);
        bool idle_lane = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            idle_lane = idle_lane || slots_[lane] == nullptr;
        }
        if (!idle_lane || materializing_) {
            throw RequestError(RequestErrorKind::Overloaded,
                               "session restore requires an idle Engine lane");
        }
        const auto view = resources_.catalog_slot(slot);
        if (view.state == ResourceManagement::CatalogState::Catalogued && view.handle != nullptr) {
            // Involuntary for the session that held the cell: the client asked for a restore,
            // not for that session's destruction. A resident bound to the very file being
            // restored is not written back over it: the client has declared the file's content.
            if (slot < slot_session_paths_.size() && slot_session_paths_[slot] == session_path) {
                clear_slot_session(slot);
            }
            spill_catalog_slot(slot, *view.handle);
            auto evicted = resources_.take_catalogued(slot);
            (void)instance_.program->release_continuation(std::move(evicted));
        }
        clear_slot_session(slot);
        auto restored = instance_.program->restore_continuation(snapshot, model_binding);
        const std::uint32_t tokens = instance_.program->continuation_depth(restored);
        std::string digest         = instance_.program->continuation_digest(restored);
        try {
            const auto summary = instance_.program->continuation_summary(restored);
            resources_.adopt_restored(slot, std::move(restored), summary);
        } catch (...) {
            // adopt_restored throws only before taking the handle.
            (void)instance_.program->release_continuation(std::move(restored));
            throw;
        }
        bind_slot_session(slot, session_path);
        // A restored session starts a new usage record: how it was used before the restart is not known.
        slot_usage_[slot] = SlotUsage{.last_used_unix_ms = unix_time_ms()};
        if (already_durable) {
            const auto restored_view = resources_.catalog_slot(slot);
            slot_persisted_[slot]    = {restored_view.id, restored_view.revision};
        }
        claim();
        publish_runtime_stats();
        return {tokens, std::move(digest)};
    }

    // `claim` receives the file the erased session was bound to, if any, so spills of it still
    // queued are superseded: an explicit erase never writes the slot file.
    std::uint32_t erase_slot(std::uint32_t slot, std::string_view expected_digest,
                             const std::function<void(const std::string&)>& claim) {
        std::scoped_lock lock(execution_mutex_);
        require_settled_slot(slot);
        const auto view = resources_.catalog_slot(slot);
        require_session_digest(view, expected_digest);
        const std::string bound_path =
            slot < slot_session_paths_.size() ? slot_session_paths_[slot] : std::string();
        clear_slot_session(slot);
        if (!bound_path.empty()) { claim(bound_path); }
        if (view.state != ResourceManagement::CatalogState::Catalogued || view.handle == nullptr) {
            return 0;
        }
        const std::uint32_t tokens = instance_.program->continuation_depth(*view.handle);
        auto evicted               = resources_.take_catalogued(slot);
        (void)instance_.program->release_continuation(std::move(evicted));
        publish_runtime_stats();
        return tokens;
    }

    // Served from the snapshot the worker publishes at unit boundaries: the execution mutex is
    // held for most of a running request, so a reader that waited on it would stall behind a
    // long prefill.
    [[nodiscard]] std::vector<SlotState> slot_states() const {
        std::lock_guard lock(stats_mutex_);
        std::vector<SlotState> states = published_slots_;
        states.resize(resources_.catalog_capacity());
        return states;
    }

    // Installs the auto-save sink: the model binding a snapshot carries, and a consumer that
    // takes each spilled session's (path, snapshot) and writes the file off-thread.
    void set_eviction_sink(
        std::string model_binding,
        std::function<void(std::string, typename ModelContract::SessionSnapshot&&)> sink) {
        std::scoped_lock lock(execution_mutex_);
        eviction_model_binding_ = std::move(model_binding);
        eviction_sink_          = std::move(sink);
    }

    // The durable context store. Every session about to be evicted is handed to `sink` whether or
    // not it is bound to a slot file; a retained session unused for `idle` is handed over in the
    // background (only when no request is waiting or prefilling and `ready` agrees, so a slow disk
    // cannot build a backlog or delay admission). `ready` and `sink` are called with the execution
    // mutex held and must not block. `sink` returns whether the snapshot was accepted into the
    // write queue, not whether it reached disk; `failures` is a count of queued writes that
    // failed, and when it grows every session is treated as not stored, so it is queued again.
    void set_context_store(std::string model_binding, runtime::ContextStore* store,
                           std::function<bool(typename ModelContract::SessionSnapshot&&)> sink,
                           std::function<bool()> ready, std::chrono::milliseconds idle,
                           std::function<std::uint64_t()> failures) {
        std::scoped_lock lock(execution_mutex_);
        eviction_model_binding_ = std::move(model_binding);
        store_                  = store;
        store_sink_             = std::move(sink);
        store_ready_            = std::move(ready);
        store_failures_         = std::move(failures);
        store_removals_seen_    = store_removal_count();
        store_idle_ms_.store(idle.count(), std::memory_order_release);
    }

    // Hands every retained session that is not already stored in its current state to `write`,
    // most recently used first, until `deadline`. For shutdown: `write` may block. Returns the
    // number written.
    std::uint32_t persist_all(
        const std::function<bool(typename ModelContract::SessionSnapshot&&)>& write,
        Clock::time_point deadline) {
        std::scoped_lock lock(execution_mutex_);
        device_.bind_to_current_thread();
        forget_failed_store_writes();
        forget_removed_store_images();
        std::vector<std::uint32_t> order;
        for (std::uint32_t slot = 0; slot < resources_.catalog_capacity(); ++slot) {
            order.push_back(slot);
        }
        std::sort(order.begin(), order.end(), [&](std::uint32_t left, std::uint32_t right) {
            return slot_usage_[left].last_used_unix_ms > slot_usage_[right].last_used_unix_ms;
        });
        std::uint32_t written = 0;
        for (const std::uint32_t slot : order) {
            if (Clock::now() >= deadline) { break; }
            if (!persistable_slot(slot)) { continue; }
            const auto view = resources_.catalog_slot(slot);
            try {
                if (write(instance_.program->save_continuation(*view.handle,
                                                               eviction_model_binding_))) {
                    slot_persisted_[slot] = {view.id, view.revision};
                    ++written;
                }
            } catch (...) {}
        }
        return written;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
        } catch (...) {}
    }

private:
    enum class HostWorkClass : std::uint8_t {
        Decode,
        Prefill,
        Control,
    };

    using EngineHostPhase = RequestEngineHostPhase;

    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept {
        switch (phase) {
        case EngineHostPhase::Boundary:
            return nvtx::Name::EngineBoundary;
        case EngineHostPhase::CommitOutput:
            return nvtx::Name::EngineCommitOutput;
        case EngineHostPhase::Maintenance:
            return nvtx::Name::EngineMaintenance;
        }
        return nvtx::Name::EngineBoundary;
    }

    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
    };

    struct ActiveExposureSet {
        std::array<ActiveExposure, kMaximumConcurrency> entries{};
        std::size_t size = 0;
    };

    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ActiveExposureSet exposed;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count();
        return count > 0 ? static_cast<std::uint64_t>(count) : 0;
    }

    [[nodiscard]] ActiveExposureSet active_exposure_set() const {
        ActiveExposureSet result;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            result.entries[result.size++] = ActiveExposure{.request = slots_[lane], .lane = lane};
        }
        return result;
    }

    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const {
        return HostPhaseMeasurement{
            .started          = Clock::now(),
            .accounted_before = worker_accounted_elapsed_ns_,
            .exposed          = active_exposure_set(),
        };
    }

    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {}) noexcept {
        current_host_work_class_   = work_class;
        current_decode_lane_count_ = decode_lanes.size();
        for (std::size_t i = 0; i < decode_lanes.size(); ++i) {
            current_decode_lanes_[i] = decode_lanes[i];
        }
    }

    [[nodiscard]] bool current_decode_contains(std::uint32_t lane) const noexcept {
        return std::find(current_decode_lanes_.begin(),
                         current_decode_lanes_.begin() +
                             static_cast<std::ptrdiff_t>(current_decode_lane_count_),
                         lane) != current_decode_lanes_.begin() +
                                      static_cast<std::ptrdiff_t>(current_decode_lane_count_);
    }

    void add_class_host_time(std::uint64_t host_ns, std::uint64_t device_wait_ns) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        switch (current_host_work_class_) {
        case HostWorkClass::Decode:
            stats.decode_host_ns += host_ns;
            stats.decode_device_wait_ns += device_wait_ns;
            break;
        case HostWorkClass::Prefill:
            stats.prefill_host_ns += host_ns;
            stats.prefill_device_wait_ns += device_wait_ns;
            break;
        case HostWorkClass::Control:
            stats.control_host_ns += host_ns;
            stats.control_device_wait_ns += device_wait_ns;
            break;
        }
    }

    void expose_engine_phase(const ActiveExposureSet& exposed, EngineHostPhase phase,
                             std::uint64_t elapsed) noexcept {
        for (std::size_t i = 0; i < exposed.size; ++i) {
            const ActiveExposure& exposure = exposed.entries[i];
            RequestHostTiming& timing      = exposure.request->host_timing;
            timing.expose_engine(phase, elapsed,
                                 current_host_work_class_ == HostWorkClass::Decode &&
                                     current_decode_contains(exposure.lane));
        }
    }

    void finish_engine_phase(const HostPhaseMeasurement& measurement,
                             EngineHostPhase phase) noexcept {
        const std::uint64_t wall    = elapsed_ns(measurement.started, Clock::now());
        const std::uint64_t nested  = worker_accounted_elapsed_ns_ - measurement.accounted_before;
        const std::uint64_t own     = wall > nested ? wall - nested : 0;
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        switch (phase) {
        case EngineHostPhase::Boundary:
            stats.engine_boundary_ns += own;
            break;
        case EngineHostPhase::CommitOutput:
            stats.engine_commit_output_ns += own;
            break;
        case EngineHostPhase::Maintenance:
            stats.engine_maintenance_ns += own;
            break;
        }
        add_class_host_time(own, 0);
        expose_engine_phase(measurement.exposed, phase, own);
        worker_accounted_elapsed_ns_ += own;
    }

    void record_program_timing(runtime::ExecutionTiming timing,
                               const ActiveExposureSet& exposed) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        stats.program_submit_ns += timing.submit_host_ns;
        stats.program_post_ns += timing.post_host_ns;
        stats.device_wait_ns += timing.device_wait_ns;
        add_class_host_time(timing.host_ns(), timing.device_wait_ns);
        for (std::size_t i = 0; i < exposed.size; ++i) {
            const ActiveExposure& exposure = exposed.entries[i];
            RequestHostTiming& request     = exposure.request->host_timing;
            request.expose_program(timing, current_host_work_class_ == HostWorkClass::Decode &&
                                               current_decode_contains(exposure.lane));
        }
        worker_accounted_elapsed_ns_ += timing.elapsed_ns();
    }

    void finish_program_call(const HostPhaseMeasurement& measurement,
                             runtime::ExecutionTiming timing) noexcept {
        const std::uint64_t wall     = elapsed_ns(measurement.started, Clock::now());
        const std::uint64_t nested   = worker_accounted_elapsed_ns_ - measurement.accounted_before;
        const std::uint64_t observed = timing.elapsed_ns() + nested;
        if (wall > observed) { timing.submit_host_ns += wall - observed; }
        record_program_timing(timing, measurement.exposed);
    }

    void record_detail(std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                       std::uint64_t RuntimeHostWorkStats::*invocation_member,
                       Clock::time_point started) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        stats.*elapsed_member += elapsed_ns(started, Clock::now());
        ++(stats.*invocation_member);
    }

    class DetailScope {
    public:
        DetailScope(EngineCore& owner, std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                    std::uint64_t RuntimeHostWorkStats::*invocation_member,
                    nvtx::Name range_name) noexcept
            : owner_(owner), elapsed_member_(elapsed_member), invocation_member_(invocation_member),
              started_(Clock::now()) {
            range_.emplace(range_name, nvtx::Category::Control);
        }

        ~DetailScope() {
            range_.reset();
            owner_.record_detail(elapsed_member_, invocation_member_, started_);
        }

        DetailScope(const DetailScope&)            = delete;
        DetailScope& operator=(const DetailScope&) = delete;

    private:
        EngineCore& owner_;
        std::uint64_t RuntimeHostWorkStats::*elapsed_member_;
        std::uint64_t RuntimeHostWorkStats::*invocation_member_;
        Clock::time_point started_;
        std::optional<nvtx::ScopedRange> range_;
    };

    class EnginePhaseScope {
    public:
        EnginePhaseScope(EngineCore& owner, EngineHostPhase phase)
            : owner_(owner), phase_(phase), measurement_(owner.begin_host_phase()) {
            range_.emplace(phase_range_name(phase), nvtx::Category::Runtime);
        }

        ~EnginePhaseScope() { finish(); }

        EnginePhaseScope(const EnginePhaseScope&)            = delete;
        EnginePhaseScope& operator=(const EnginePhaseScope&) = delete;

        void pause_range() noexcept { range_.reset(); }

        void resume_range() noexcept {
            if (active_ && !range_) {
                range_.emplace(phase_range_name(phase_), nvtx::Category::Runtime);
            }
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ProgramCallScope {
    public:
        explicit ProgramCallScope(EngineCore& owner)
            : owner_(owner), measurement_(owner.begin_host_phase()) {}

        ~ProgramCallScope() noexcept { finish(failed_timing_); }

        ProgramCallScope(const ProgramCallScope&)            = delete;
        ProgramCallScope& operator=(const ProgramCallScope&) = delete;

        [[nodiscard]] runtime::ExecutionTiming& failed_timing() noexcept { return failed_timing_; }

        void finish(runtime::ExecutionTiming timing) noexcept {
            if (!active_) { return; }
            owner_.finish_program_call(measurement_, timing);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        HostPhaseMeasurement measurement_;
        runtime::ExecutionTiming failed_timing_;
        bool active_ = true;
    };

    void publish_runtime_stats() {
        HostPhaseMeasurement measurement = begin_host_phase();
        std::optional<nvtx::ScopedRange> phase_range;
        phase_range.emplace(nvtx::Name::EngineMaintenance, nvtx::Category::Runtime);
        const Clock::time_point detail_started = Clock::now();
        std::optional<nvtx::ScopedRange> detail_range;
        detail_range.emplace(nvtx::Name::StatsPublication, nvtx::Category::Control);
        RuntimeStats snapshot = cumulative_stats_;
        resources_.populate_runtime_stats(*instance_.program, snapshot);
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (scheduler_.is_prefill_owner(lane) && slots_[lane] != nullptr &&
                !slots_[lane]->capture_pending) {
                ++snapshot.prefilling_requests;
            }
        }
        snapshot.materializing_requests = materializing_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->is_decode_ready()) { ++snapshot.decode_ready_requests; }
            if (slots_[lane]->capture_pending) { ++snapshot.capture_pending_requests; }
            if (slots_[lane]->terminal_reason) { ++snapshot.terminal_pending_requests; }
        }
        std::vector<SlotState> slot_snapshot = collect_slot_states();
        detail_range.reset();
        record_detail(&RuntimeHostWorkStats::stats_publication_ns,
                      &RuntimeHostWorkStats::stats_publication_invocations, detail_started);
        phase_range.reset();
        finish_engine_phase(measurement, EngineHostPhase::Maintenance);
        snapshot.host_work = cumulative_stats_.host_work;
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
        published_slots_ = std::move(slot_snapshot);
    }

    void require_settled_slot(std::uint32_t slot) const {
        if (slot >= resources_.catalog_capacity()) {
            throw std::invalid_argument("slot id is outside the retained-session catalog");
        }
        if (instance_.program->has_context_transaction() ||
            resources_.context_transaction_kind()) {
            throw RequestError(RequestErrorKind::Overloaded,
                               "slot catalog is busy with a resource transaction");
        }
        const auto view = resources_.catalog_slot(slot);
        if (view.state == ResourceManagement::CatalogState::Claimed ||
            view.state == ResourceManagement::CatalogState::ReservedForActive || view.active_edge) {
            throw RequestError(RequestErrorKind::Overloaded, "slot is in use by an active request");
        }
    }

    void require_session_digest(const typename ResourceManagement::CatalogSlotView& view,
                                std::string_view expected_digest) const {
        if (expected_digest.empty()) { return; }
        const std::string digest =
            view.state == ResourceManagement::CatalogState::Catalogued && view.handle != nullptr
                ? instance_.program->continuation_digest(*view.handle)
                : std::string();
        if (digest != expected_digest) {
            throw SlotSessionMismatch("slot session does not match if_digest");
        }
    }

    // Best-effort spill of a retained session about to be destroyed involuntarily. The snapshot
    // runs on the calling thread; the file write runs on the Engine's writer thread through the
    // sink. Only sessions bound to a slot file spill, and a failed spill never blocks the
    // eviction: it costs the client one cold prefill, as without the feature.
    void spill_catalog_slot(std::uint32_t slot,
                            const typename ModelContract::ContinuationHandle& handle) noexcept {
        if (store_sink_) {
            // The durable store takes every session about to be destroyed, bound to a slot file
            // or not, unless it already holds the session in its current state.
            // Both resets first: a write that failed after being accepted, or an image the store
            // has since removed, must not make this session look stored.
            (void)forget_failed_store_writes();
            forget_removed_store_images();
            const auto view = resources_.catalog_slot(slot);
            if (slot < slot_persisted_.size() &&
                slot_persisted_[slot] != PersistedState{view.id, view.revision}) {
                try {
                    if (store_sink_(
                            instance_.program->save_continuation(handle, eviction_model_binding_))) {
                        slot_persisted_[slot] = {view.id, view.revision};
                    }
                } catch (...) {}
            }
            return;
        }
        if (!eviction_sink_ || slot >= slot_session_paths_.size() ||
            slot_session_paths_[slot].empty()) {
            return;
        }
        try {
            auto snapshot = instance_.program->save_continuation(handle, eviction_model_binding_);
            eviction_sink_(slot_session_paths_[slot], std::move(snapshot));
        } catch (...) {}
        // The binding stays: a planned eviction whose transaction aborts leaves the session in the
        // cell, still bound. The release observer clears it when the entry really goes.
    }

    void clear_slot_session(std::uint32_t slot) noexcept {
        if (slot < slot_session_paths_.size()) { slot_session_paths_[slot].clear(); }
    }

    // A retained session in a cell nothing is using, that the context store does not yet hold in
    // its current state, and that can be snapshotted now.
    [[nodiscard]] bool persistable_slot(std::uint32_t slot) const {
        if (slot >= slot_persisted_.size() || instance_.program->has_context_transaction() ||
            resources_.context_transaction_kind()) {
            return false;
        }
        const auto view = resources_.catalog_slot(slot);
        return view.state == ResourceManagement::CatalogState::Catalogued &&
               view.handle != nullptr && !view.active_edge &&
               slot_persisted_[slot] != PersistedState{view.id, view.revision};
    }

    // A write that was accepted into the queue and then failed left its session marked as stored.
    // When the failure count has grown, every session is treated as not stored again.
    bool forget_failed_store_writes() noexcept {
        if (!store_failures_) { return false; }
        const std::uint64_t failed = store_failures_();
        if (failed == store_failures_seen_) { return false; }
        store_failures_seen_ = failed;
        std::fill(slot_persisted_.begin(), slot_persisted_.end(), PersistedState{});
        return true;
    }

    // Images the store has removed on its own (expiry, size limit, damage). A session marked as
    // stored whose image is gone would be skipped when it is evicted and lost, so when the count
    // grows every session is treated as not stored again. The window between this check and the
    // eviction it guards is the store's own removal running in between.
    [[nodiscard]] std::uint64_t store_removal_count() const noexcept {
        if (store_ == nullptr) { return 0; }
        try {
            const runtime::ContextStore::Stats stats = store_->stats();
            return stats.evicted_for_space + stats.expired + stats.corrupt_removed;
        } catch (...) { return store_removals_seen_; }
    }

    void forget_removed_store_images() noexcept {
        const std::uint64_t removed = store_removal_count();
        if (removed == store_removals_seen_) { return; }
        store_removals_seen_ = removed;
        std::fill(slot_persisted_.begin(), slot_persisted_.end(), PersistedState{});
    }

    // Writes at most one idle retained session to the context store. Called by the worker between
    // units; it does nothing while a request is waiting for admission, being admitted or
    // prefilling, so keeping the store current does not delay a request that is already waiting (one
    // that arrives while the snapshot is being taken waits for it), and it stays at least a
    // second apart so a long scan or a slow disk is not a per-unit cost.
    void persist_idle_session() noexcept {
        const std::int64_t idle_limit_ms = store_idle_ms_.load(std::memory_order_acquire);
        if (!store_sink_ || idle_limit_ms <= 0) { return; }
        const auto now = Clock::now();
        if (now - last_persist_scan_ < std::chrono::seconds(1)) { return; }
        last_persist_scan_ = now;
        try {
            if (forget_failed_store_writes()) {
                // A failing disk: do not snapshot deep sessions over and over.
                last_persist_scan_ = now + std::chrono::seconds(30);
                return;
            }
            forget_removed_store_images();
            if (materializing_ || (store_ready_ && !store_ready_())) { return; }
            {
                std::lock_guard lock(queue_mutex_);
                if (!pending_.empty()) { return; }
            }
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr && !slots_[lane]->is_decode_ready()) { return; }
            }
            const std::uint64_t idle_ms  = static_cast<std::uint64_t>(idle_limit_ms);
            const std::uint64_t unix_now = unix_time_ms();
            std::optional<std::uint32_t> chosen;
            std::uint32_t chosen_depth = 0;
            for (std::uint32_t slot = 0; slot < resources_.catalog_capacity(); ++slot) {
                if (!persistable_slot(slot)) { continue; }
                const std::uint64_t last_used = slot_usage_[slot].last_used_unix_ms;
                if (unix_now < last_used + idle_ms) { continue; }
                // The deepest first: it is the most expensive to lose.
                const std::uint32_t depth = instance_.program->continuation_depth(
                    *resources_.catalog_slot(slot).handle);
                if (chosen && depth <= chosen_depth) { continue; }
                chosen       = slot;
                chosen_depth = depth;
            }
            if (!chosen) { return; }
            const auto view = resources_.catalog_slot(*chosen);
            if (store_sink_(
                    instance_.program->save_continuation(*view.handle, eviction_model_binding_))) {
                slot_persisted_[*chosen] = {view.id, view.revision};
            } else {
                // The write queue is full or the write failed: try again later, not on every unit.
                last_persist_scan_ = now + std::chrono::seconds(4);
            }
        } catch (...) {
            last_persist_scan_ = now + std::chrono::seconds(4);
        }
    }

    // Prompt tokens a stored session must add beyond what the cache already offers before reading it
    // back is worth the stall: the read and the upload run on the worker, and a few thousand
    // tokens prefill faster than a deep image comes off a slow disk.
    static constexpr std::uint32_t kMinimumHydrationGain = 4096;

    // The retained session nothing is using that was used longest ago: the one to give up when a
    // stored session needs the room. It is written to the store first when the store does not hold
    // its current state, as for any eviction.
    [[nodiscard]] bool evict_least_recently_used_session() noexcept {
        try {
            std::optional<std::uint32_t> victim;
            for (std::uint32_t slot = 0; slot < resources_.catalog_capacity(); ++slot) {
                const auto view = resources_.catalog_slot(slot);
                if (view.state != ResourceManagement::CatalogState::Catalogued ||
                    view.handle == nullptr || view.active_edge) {
                    continue;
                }
                if (!victim ||
                    slot_usage_[slot].last_used_unix_ms < slot_usage_[*victim].last_used_unix_ms) {
                    victim = slot;
                }
            }
            if (!victim) { return false; }
            const auto view = resources_.catalog_slot(*victim);
            spill_catalog_slot(*victim, *view.handle);
            clear_slot_session(*victim);
            auto evicted = resources_.take_catalogued(*victim);
            (void)instance_.program->release_continuation(std::move(evicted));
            return true;
        } catch (...) { return false; }
    }

    // A stored session read back for a request, until the request is planned again against it.
    struct HydrationAttempt {
        bool restored = false;
        std::uint32_t expected_frontier = 0; // the stored checkpoint the request should resume from
        std::string stored_id;               // the stored image, touched once the plan uses it
        Clock::time_point started;
    };

    // Before a request is admitted: when the context store holds a checkpoint of this very prompt
    // deeper than anything the cache offers (`resident_reuse` is the deepest it can offer, selected
    // or not), read that session back so the request resumes from it instead of prefilling the
    // difference. Called by the worker with the execution mutex held. `restored` tells the caller
    // the cache changed and the request must be planned again. Every failure is a miss: the
    // request is prefilled as it would have been without the store, and the time spent trying is
    // counted.
    [[nodiscard]] HydrationAttempt hydrate_from_store(const std::shared_ptr<Request>& request,
                                                      std::uint32_t resident_reuse) noexcept {
        HydrationAttempt attempt;
        if (store_ == nullptr || request->store_probed || !context_cache_enabled_ ||
            !request->options.execution.allow_prefix_reuse || !request->base_plan ||
            materializing_ || instance_.program->has_context_transaction() ||
            resources_.context_transaction_kind()) {
            return attempt;
        }
        bool idle_lane = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            idle_lane = idle_lane || slots_[lane] == nullptr;
        }
        if (!idle_lane) { return attempt; }
        request->store_probed = true;
        attempt.started       = Clock::now();
        const auto charge     = [&] {
            store_hydration_ns_.fetch_add(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                         attempt.started)
                        .count()),
                std::memory_order_relaxed);
        };
        try {
            std::string best_id;
            bool best_local             = true;
            std::uint32_t best_frontier = resident_reuse + kMinimumHydrationGain - 1U;
            const std::uint32_t resolved_tokens = instance_.kv_capacity_resolution.resolved_tokens;
            const std::uint32_t hydration_token_limit =
                resolved_tokens != 0 ? std::min(max_context_, resolved_tokens) : max_context_;
            for (const runtime::ContextStore::Info& info : store_->list()) {
                if (info.binding != eviction_model_binding_) { continue; }
                // The binding does not include the context length or the KV capacity (the latter
                // varies run to run when sized automatically), and an image deeper than either can
                // never be restored: reading it would only evict sessions for a miss.
                if (info.tokens > hydration_token_limit) { continue; }
                for (const runtime::ContextStore::CheckpointKey& key : info.checkpoints) {
                    if (key.frontier <= best_frontier) { continue; }
                    const auto mine = request->base_plan->prefix_shortlist_key(key.frontier);
                    if (mine && mine->digests == key.digests &&
                        mine->identity_tag == key.identity_tag) {
                        best_id       = info.id;
                        best_frontier = key.frontier;
                        best_local    = info.local;
                    }
                }
            }
            if (best_id.empty()) { return attempt; } // nothing worth reading: not an attempt
            if (!best_local) {
                // Only the remote holds all of it: fetching a deep session over the network on the
                // worker would stall every running request for minutes, so it is fetched in the
                // background for the next request that continues it, and this one is prefilled.
                (void)store_->prefetch(best_id);
                return attempt;
            }
            std::optional<std::vector<std::uint8_t>> bytes = store_->load(best_id, false);
            if (!bytes) {
                store_hydration_failures_.fetch_add(1, std::memory_order_relaxed);
                charge();
                return attempt;
            }
            bool restored = false;
            // Each refusal for room gives up one more retained session; a bounded few, so a
            // request never empties the cache for one image.
            constexpr std::uint32_t kMaximumRestoreTries = 8;
            for (std::uint32_t tries = 0; tries < kMaximumRestoreTries && !restored; ++tries) {
                std::optional<std::uint32_t> slot = first_vacant_slot_locked();
                if (!slot) {
                    if (!evict_least_recently_used_session()) { break; }
                    slot = first_vacant_slot_locked();
                    if (!slot) { break; }
                }
                try {
                    // Not marked durable: see settle_hydration.
                    (void)restore_slot_locked(
                        *slot, std::span<const std::uint8_t>(bytes->data(), bytes->size()),
                        eviction_model_binding_, std::string_view(), [] {});
                    restored = true;
                } catch (const std::invalid_argument& error) {
                    if (std::string_view(error.what()).find("evict other sessions first") ==
                        std::string_view::npos) {
                        break;
                    }
                    if (tries + 1 == kMaximumRestoreTries || !evict_least_recently_used_session()) {
                        break; // the last try cannot retry, so it gives up no further session
                    }
                }
            }
            if (!restored) {
                store_hydration_failures_.fetch_add(1, std::memory_order_relaxed);
                charge();
                return attempt;
            }
            attempt.restored          = true;
            attempt.expected_frontier = best_frontier;
            attempt.stored_id         = std::move(best_id);
            return attempt; // the caller settles the counters once the request is planned again
        } catch (...) {
            store_hydration_failures_.fetch_add(1, std::memory_order_relaxed);
            charge();
            return HydrationAttempt{};
        }
    }

    // Settles a restored attempt once the request has been planned again: it counts as a hydration
    // only if the plan now resumes from (at least) the stored checkpoint, and the tokens it counts
    // are the ones the plan actually gained over `reuse_before`.
    void settle_hydration(const HydrationAttempt& attempt, std::uint32_t reuse_before,
                          std::uint32_t reuse_after) noexcept {
        store_hydration_ns_.fetch_add(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - attempt.started)
                    .count()),
            std::memory_order_relaxed);
        if (reuse_after >= attempt.expected_frontier && reuse_after > reuse_before) {
            store_hydrations_.fetch_add(1, std::memory_order_relaxed);
            store_hydrated_tokens_.fetch_add(reuse_after - reuse_before, std::memory_order_relaxed);
            // The image was used: only now does reading it count against its age. The restored
            // entry is deliberately not marked durable: the store can drop the image at any moment
            // (maintenance, a size limit), no check-then-mark can be atomic with that, and an entry
            // wrongly marked would never be written again. Unmarked, its eviction writes it, which
            // costs only the chunks the store no longer holds.
            try {
                store_->touch(attempt.stored_id);
            } catch (...) {}
        } else {
            store_hydration_failures_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // After an admissible inspection of `request`: reads a deeper stored session back into the cache
    // when there is one and plans the request again against it, leaving the new inspection in
    // `inspected`. False when planning again failed, which removed the request.
    [[nodiscard]] bool hydrate_and_replan(const std::shared_ptr<Request>& request,
                                          std::optional<ResourceInspection>& inspected,
                                          PlanningAllowance allowance) {
        if (store_ == nullptr || !inspected || !inspected->choice ||
            (inspected->readiness != Readiness::Ready &&
             inspected->readiness != Readiness::NeedsTransfer)) {
            return true;
        }
        const std::uint32_t chosen = inspected->choice->summary().reusable_prompt_tokens;
        const std::uint32_t resident =
            std::max(chosen, resources_.max_resident_prefix_frontier(*request->base_plan));
        const HydrationAttempt attempt = hydrate_from_store(request, resident);
        if (!attempt.restored) { return true; }
        inspected.reset();
        std::optional<ResourceInspection> replanned = inspect_admission_or_fail(request, allowance);
        if (!replanned) {
            settle_hydration(attempt, chosen, 0);
            return false;
        }
        inspected.emplace(std::move(*replanned));
        settle_hydration(attempt, chosen,
                         inspected->choice ? inspected->choice->summary().reusable_prompt_tokens : 0U);
        return true;
    }

    [[nodiscard]] static std::uint64_t unix_time_ms() noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    // A slot file binds to at most one cell. A conversation's live continuation can move to a
    // new cell turn to turn (an anchor or rewrite restore retains its source), leaving an older
    // copy in another cell bound to the same file; if that copy were evicted later it would
    // overwrite the newer save. Whoever saved or restored a path last owns it, and every other
    // cell holding it is unbound. Called under execution_mutex_.
    void bind_slot_session(std::uint32_t slot, std::string_view session_path) {
        if (session_path.empty() || slot >= slot_session_paths_.size()) { return; }
        for (std::size_t other = 0; other < slot_session_paths_.size(); ++other) {
            if (other != slot && slot_session_paths_[other] == session_path) {
                slot_session_paths_[other].clear();
            }
        }
        slot_session_paths_[slot] = std::string(session_path);
    }

    // Per-cell occupancy for slot readers. Digests come from a cache keyed by the catalog
    // entry's (id, revision), so an unchanged session is not rehashed every unit.
    [[nodiscard]] std::vector<SlotState> collect_slot_states() {
        std::vector<SlotState> out(resources_.catalog_capacity());
        for (std::uint32_t slot = 0; slot < out.size(); ++slot) {
            const auto view = resources_.catalog_slot(slot);
            if (view.state != ResourceManagement::CatalogState::Catalogued ||
                view.handle == nullptr) {
                continue;
            }
            SlotDigestCacheEntry& cache = slot_digest_cache_[slot];
            if (cache.id != view.id || cache.revision != view.revision) {
                cache.id          = view.id;
                cache.revision    = view.revision;
                cache.depth       = instance_.program->continuation_depth(*view.handle);
                cache.digest      = instance_.program->continuation_digest(*view.handle);
                cache.checkpoints = instance_.program->continuation_checkpoints(*view.handle);
            }
            SlotState& state     = out[slot];
            state.retained       = true;
            state.prompt_tokens  = cache.depth;
            state.cached_tokens  = cache.depth;
            state.session_digest = cache.digest;
            state.checkpoints    = cache.checkpoints;
            if (slot < slot_session_paths_.size()) {
                state.snapshot_file =
                    std::filesystem::path(slot_session_paths_[slot]).filename().string();
            }
            if (slot < slot_usage_.size()) {
                state.last_used_unix_ms = slot_usage_[slot].last_used_unix_ms;
                state.reuse_count       = slot_usage_[slot].reuse_count;
                state.reused_tokens     = slot_usage_[slot].reused_tokens;
            }
        }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            const std::optional<std::uint32_t> publication =
                resources_.lane_publication_slot(LaneId{lane});
            if (!publication || *publication >= out.size()) { continue; }
            SlotState& state    = out[*publication];
            state.processing    = true;
            state.prompt_tokens = slots_[lane]->prompt_summary.prompt_tokens;
            state.cached_tokens =
                slots_[lane]->begin ? slots_[lane]->begin->reused_prompt_tokens : 0U;
        }
        return out;
    }

    void record_prefix_selection(const RequestPlanSummary& summary) noexcept {
        switch (summary.prefix_reuse_path) {
        case PrefixReusePath::Root:
            ++cumulative_stats_.root_selections;
            break;
        case PrefixReusePath::PrivateEndpoint:
            ++cumulative_stats_.private_endpoint_selections;
            break;
        case PrefixReusePath::PrivateTurnClosure:
            ++cumulative_stats_.private_turn_closure_selections;
            break;
        case PrefixReusePath::PrivateResponseReplay:
            ++cumulative_stats_.private_response_replay_selections;
            break;
        case PrefixReusePath::PrivateLongAnchor:
            ++cumulative_stats_.private_long_anchor_selections;
            break;
        case PrefixReusePath::SharedStablePrefix:
            ++cumulative_stats_.shared_stable_prefix_selections;
            break;
        }
        cumulative_stats_.reused_prompt_tokens += summary.reusable_prompt_tokens;
        cumulative_stats_.last_selected_frontier_tokens = summary.reusable_prompt_tokens;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            EngineCore* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<typename Request::StreamEvent> events;
        for (;;) {
            start.reset();
            progress.reset();
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->response_done || request->stream_start.has_value() ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                start = std::move(request->stream_start);
                request->stream_start.reset();
                progress = std::move(request->stream_progress);
                request->stream_progress.reset();
                events.swap(request->events);
                done = request->response_done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (start) { sink->start(std::move(*start)); }
                    if (progress) { sink->progress(std::move(*progress)); }
                    for (auto& event : events) {
                        if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                            sink->timing(std::move(*timing));
                        } else {
                            sink->publish(std::move(std::get<OutputDelta>(event)));
                        }
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    request_admission_check();
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        request_admission_check();
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    request_admission_check();
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
    };

    // Coalesces admission-visible queue/resource changes. Ordinary prefill/decode progress does not
    // re-arm a temporarily blocked inspection.
    void request_admission_check() noexcept {
        admission_check_pending_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool consume_admission_check() noexcept {
        return admission_check_pending_.exchange(false, std::memory_order_acq_rel);
    }

    struct MaterializingRequest {
        std::shared_ptr<Request> request;
        LaneId destination;
        GenerationBudget budget;
        RequestPlanSummary summary;
        BackfillClass backfill_class   = BackfillClass::None;
        std::uint64_t protection_epoch = 0;
        Clock::time_point started;
    };

    [[nodiscard]] std::optional<GenerationTimingObservation>
    record_committed_output(const std::shared_ptr<Request>& request,
                            std::uint32_t accepted_tokens) {
        if (accepted_tokens == 0) { return std::nullopt; }
        const bool observe_wall =
            request->observation.phase_timings || request->observation.live_timings;
        const bool need_now         = !request->first_token || observe_wall;
        const Clock::time_point now = need_now ? Clock::now() : Clock::time_point{};
        if (!request->first_token) { request->first_token = now; }
        if (!observe_wall) { return std::nullopt; }
        if (!request->admitted_at || !request->first_token) {
            throw std::logic_error("committed output has no observed admission boundary");
        }
        request->last_token = now;
        if (!request->observation.live_timings) { return std::nullopt; }
        if (request->generated.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("generated token count exceeds observation domain");
        }
        return GenerationTimingObservation{
            .generated_tokens      = static_cast<std::uint32_t>(request->generated.size()),
            .prompt_elapsed_ns     = elapsed_ns(*request->admitted_at, *request->first_token),
            .generation_elapsed_ns = elapsed_ns(*request->first_token, *request->last_token),
        };
    }

    void append_output(const std::shared_ptr<Request>& request, PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt) {
        if (output.empty() && !timing) { return; }
        const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
        {
            std::lock_guard lock(request->mutex);
            if (streaming && timing) { request->events.emplace_back(std::move(*timing)); }
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                if (streaming) { request->events.emplace_back(std::move(delta)); }
            }
        }
        if (streaming) { request->cv.notify_one(); }
    }

    void publish_prompt_progress(const std::shared_ptr<Request>& request) {
        if (!request->observation.prompt_progress) { return; }
        if (!request->admitted_begin || !request->admitted_at) {
            throw std::logic_error("prompt progress has no admitted request boundary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens ||
            request->computed_prompt_tokens > begin.prompt_tokens - begin.reused_prompt_tokens) {
            throw std::logic_error("prompt progress exceeds the admitted prompt frontier");
        }
        const PromptProgress progress{
            .total_prompt_tokens     = begin.prompt_tokens,
            .reused_prompt_tokens    = begin.reused_prompt_tokens,
            .processed_prompt_tokens = begin.reused_prompt_tokens + request->computed_prompt_tokens,
            .elapsed_ns              = elapsed_ns(*request->admitted_at, Clock::now()),
        };
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->stream_progress = progress;
        }
        request->cv.notify_one();
    }

    void publish_generation_start(const std::shared_ptr<Request>& request, BeginSummary begin) {
        if (request->admitted_begin) {
            throw std::logic_error("request admission published generation start twice");
        }
        request->admitted_begin = begin;
        if (request->observation.phase_timings || request->observation.live_timings ||
            request->observation.prompt_progress) {
            request->admitted_at = Clock::now();
        }
        if (request->consumer_mode != OutputConsumerMode::Streaming) { return; }
        {
            std::lock_guard lock(request->mutex);
            if (request->stream_start || request->response_done) {
                throw std::logic_error("streaming request has an invalid generation-start state");
            }
            request->stream_start = GenerationStart{
                .prompt               = request->prompt_summary,
                .reused_prompt_tokens = begin.reused_prompt_tokens,
            };
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->response_done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        request_admission_check();
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->error         = std::move(error);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        HostPhaseMeasurement completion = begin_host_phase();
        double prompt_wall_seconds      = 0.0;
        double generation_wall_seconds  = 0.0;
        if (request->observation.phase_timings && request->first_token) {
            if (!request->admitted_at || !request->last_token) {
                throw std::logic_error("observed request completed without stable timing bounds");
            }
            prompt_wall_seconds =
                std::chrono::duration<double>(*request->first_token - *request->admitted_at)
                    .count();
            generation_wall_seconds =
                std::chrono::duration<double>(*request->last_token - *request->first_token).count();
        }
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        if (!request->queue_wait_recorded) {
            request->host_timing.queue_wait_ns = elapsed_ns(request->submitted, Clock::now());
            request->queue_wait_recorded       = true;
        }
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.tool_calls              = request->output.take_tool_calls();
        result.tool_call_parse         = request->output.tool_call_parse_diagnostics();
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.matched_stop_string     = request->output.matched_stop_string();
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        result.timings                 = request->generation_timings;
        result.timings.prepare_seconds = request->prepare_seconds;
        result.speculative             = std::move(request->speculative_stats);
        result.thinking                = request->output.thinking_stats();
        result.materialization         = request->materialization_diagnostics;
        result.slot                    = request->retained_slot;
        result.session_digest          = request->retained_session_digest;
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.prompt_wall_seconds     = prompt_wall_seconds;
        result.timings.generation_wall_seconds = generation_wall_seconds;
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        // An exhaustion counts only if the request really ended at its reserved output; one that
        // stopped naturally in the tokens it still had was not cut short.
        if (request->output_reservation_exhausted && reason == FinishReason::OutputLimit) {
            ++cumulative_stats_.output_reservation_exhaustions;
        }
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        finish_engine_phase(completion, EngineHostPhase::CommitOutput);
        result.engine_timing = request->host_timing.public_snapshot();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->result        = std::move(result);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
        // Only a published, uncancelled result shows the Engine serving again; a cancellation,
        // or a completion that threw before publication, leaves the recovery streak standing.
        if (reason != FinishReason::Cancelled) { recovery_streak_.record_success(); }
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    void complete_detached_cancelled(const std::shared_ptr<Request>& request) {
        try {
            complete_cancelled(request);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            complete_error(request, error);
            throw;
        }
    }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        request_admission_check();
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        // An already-issued active unit may finish while another row owns the global resource
        // transaction.  Its cancellation cannot release topology until that transaction reaches
        // a stable terminal state.
        if (instance_.program->has_context_transaction()) { return cancelled; }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    bool settle_terminal_requests(HostPhaseMeasurement& boundary) {
        const bool manager_transaction = resources_.context_transaction_kind().has_value();
        const bool program_transaction = instance_.program->has_context_transaction();
        if (manager_transaction != program_transaction) {
            throw std::logic_error("Engine and Program disagree before terminal settlement");
        }
        if (program_transaction) { return false; }

        bool changed = false;
        for (;;) {
            std::optional<std::uint32_t> selected;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request == nullptr || !request->terminal_reason) { continue; }
                if (!selected ||
                    request->publication_order < slots_[*selected]->publication_order) {
                    selected = lane;
                }
            }
            if (!selected) { break; }

            const std::uint32_t lane = *selected;
            const auto request       = slots_[lane];
            if (!request->is_model_finished() || request->capture_pending || !request->sequence ||
                !request->lane || request->lane->value != lane ||
                resources_.lane_state(LaneId{lane}) != LogicalLaneState::TerminalPending) {
                throw std::logic_error("terminal-pending request has invalid ownership");
            }
            const FinishReason reason = *request->terminal_reason;
            const CatalogContext catalog = capture_catalog_context(*request);
            auto finished =
                resources_.finish(*instance_.program, *request->lane, *request->sequence);
            request->generation_timings = finished.timings;
            request->speculative_stats  = std::move(finished.speculative);
            record_catalogued_publication(request, finished.disposition, catalog, true);
            request->terminal_reason.reset();

            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_success(request, reason);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
        return changed;
    }

    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return; }
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            if (request->capture_pending) { continue; }
            if (!request->sequence || !request->lane || request->lane->value != lane) {
                throw std::logic_error("active cancellation has no sequence binding");
            }
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            if (scheduler_.is_prefill_owner(lane)) {
                const CatalogContext catalog = capture_catalog_context(*request);
                // The checkpoints a cancelled prompt has already captured outlive it, so the
                // client's retry resumes from them rather than prefilling the prompt again. The
                // Program declines when nothing can be kept and `finish` then discards the lane
                // exactly as an abort would.
                resources_.mark_terminal_pending(*request->lane);
                auto finished =
                    resources_.finish(*instance_.program, *request->lane, *request->sequence);
                request->generation_timings = finished.timings;
                request->speculative_stats  = std::move(finished.speculative);
                record_catalogued_publication(request, finished.disposition, catalog, false);
                ++cumulative_stats_.cancelled_prefills;
                cumulative_stats_.cancelled_prefill_computed_tokens += request->computed_prompt_tokens;
                if (request->retained_slot >= 0) {
                    const auto kept = resources_.catalog_slot(static_cast<std::uint32_t>(request->retained_slot));
                    if (kept.state == ResourceManagement::CatalogState::Catalogued &&
                        kept.handle != nullptr) {
                        ++cumulative_stats_.cancelled_prefills_retained;
                        cumulative_stats_.cancelled_prefill_retained_tokens +=
                            instance_.program->continuation_depth(*kept.handle);
                    }
                }
                scheduler_.clear_prefill_lane(lane);
            } else {
                auto aborted =
                    resources_.abort(*instance_.program, *request->lane, *request->sequence);
                request->generation_timings = aborted.timings;
                request->speculative_stats  = std::move(aborted.speculative);
            }
            append_output(request, request->output.commit_preview());
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_success(request, FinishReason::Cancelled);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        for (const auto& request : cancelled) {
            on_waiting_removed(request, WaitingRemoval::Cancelled);
        }
        for (const auto& request : expired) {
            on_waiting_removed(request, WaitingRemoval::Expired);
        }
        try {
            for (const auto& request : cancelled) { complete_detached_cancelled(request); }
            for (const auto& request : expired) {
                complete_error(request,
                               std::make_exception_ptr(RequestError(
                                   RequestErrorKind::QueueTimeout,
                                   "inference request expired while waiting for admission")));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (const auto& request : cancelled) { complete_error(request, error); }
            for (const auto& request : expired) { complete_error(request, error); }
            throw;
        }
        if (!cancelled.empty() || !expired.empty()) {
            request_admission_check();
            publish_runtime_stats();
        }
        return have_pending;
    }

    void commit_pending(PendingBatch&& pending, std::span<const std::uint32_t> lane_indices,
                        bool decode_round,
                        const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        const std::size_t row_count = lane_indices.size();
        if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
            (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
            pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            std::array<LaneId, kMaximumConcurrency> invalid_lanes{};
            for (std::size_t row = 0; row < row_count; ++row) {
                invalid_lanes[row] = LaneId{lane_indices[row]};
            }
            resources_.apply_discard(std::span<const LaneId>(invalid_lanes.data(), row_count),
                                     discarded);
            throw std::logic_error("pending batch returned an invalid ragged layout");
        }

        std::array<LaneId, kMaximumConcurrency> lanes{};
        std::array<CommitDecision, kMaximumConcurrency> decisions{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        std::array<ContinuationAction, kMaximumConcurrency> continuations{};
        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<bool, kMaximumConcurrency> cancelled{};
        bool generated_staged = false;
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> terminal_requests{};
        std::array<std::uint32_t, kMaximumConcurrency> terminal_lanes{};
        std::array<FinishReason, kMaximumConcurrency> terminal_reasons{};
        std::size_t terminal_count = 0;
        for (std::size_t row = 0; row < row_count; ++row) {
            lanes[row] = LaneId{lane_indices[row]};
        }
        const auto rollback_generated = [&]() noexcept {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < row_count; ++row) {
                const auto& request = slots_[lane_indices[row]];
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };
        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane = lane_indices[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->sequence || !request->lane ||
                    request->lane->value != lane || !request->budget) {
                    throw std::logic_error("pending row has no active Engine request");
                }
                if (decode_round) { ++request->host_timing.decode_rounds; }
                cancelled[row] = cancelled_at_unit_start[lane];
                const std::int32_t raw_count =
                    pending.row_counts().empty() ? 1 : pending.row_counts()[row];
                if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                    throw std::logic_error("pending row has an invalid licensed extent");
                }
                const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
                const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                     static_cast<std::size_t>(count));
                generated_sizes[row]      = request->generated.size();
                if (cancelled[row]) {
                    (void)request->output.preview_terminal(FinishReason::Cancelled);
                    decisions[row] = CommitDecision{
                        .accepted_tokens = 0,
                        .terminal        = true,
                        .cancelled       = true,
                    };
                    finish_reasons[row] = FinishReason::Cancelled;
                    continue;
                }
                const OutputDecision decision = request->output.preview_model(
                    row_tokens, request->budget->remaining(), request->budget->limit_reason());
                if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                    (!decision.finished() && decision.accepted_tokens != count) ||
                    (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("output policy returned an invalid licensed prefix");
                }
                decisions[row] = CommitDecision{
                    .accepted_tokens              = decision.accepted_tokens,
                    .terminal                     = decision.finished(),
                    .cancelled                    = false,
                    .prefix_execution_split_after = decision.prefix_execution_split_after,
                };
                finish_reasons[row] = decision.finish_reason;
                continuations[row]  = decision.continuation;
            }
            generated_staged = true;
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (accepted == 0) { continue; }
                const auto& request = slots_[lane_indices[row]];
                if (request->generated.size() > request->generated.capacity() ||
                    accepted > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error("admission did not reserve generated-token capacity");
                }
                const auto first = pending.tokens().begin() +
                                   static_cast<std::ptrdiff_t>(row * pending.row_stride());
                request->generated.insert(request->generated.end(), first,
                                          first + static_cast<std::ptrdiff_t>(accepted));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            rollback_generated();
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            if (discarded.status == ConsumeStatus::Consumed) {
                resources_.apply_discard(std::span<const LaneId>(lanes.data(), row_count),
                                         discarded);
            } else if (!instance_.program->has_context_transaction()) {
                throw std::logic_error("Program could not abort a failed pending batch");
            }
            std::rethrow_exception(error);
        }

        std::optional<typename ModelContract::CommitResult> committed_storage;
        try {
            phase.pause_range();
            ProgramCallScope program_call(*this);
            auto committed = instance_.program->commit(
                std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count),
                CommitObservation::ReleasedRowsOnly, &program_call.failed_timing());
            program_call.finish(committed.timing);
            // The commit samples and settles the unit's output, so its time belongs to the unit:
            // a decode round, or the prefill unit that emitted the first token.
            (decode_round ? cumulative_stats_.decode_seconds_total
                          : cumulative_stats_.prefill_seconds_total) +=
                static_cast<double>(committed.timing.elapsed_ns()) * 1e-9;
            committed_storage.emplace(std::move(committed));
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            if (!instance_.program->has_context_transaction()) {
                resources_.release_failed_commit(std::span<const LaneId>(lanes.data(), row_count));
            }
            throw;
        }
        generated_staged = false;
        auto& committed  = *committed_storage;
        if (committed.row_count != row_count) {
            throw std::logic_error("Runtime commit result is not row aligned");
        }
        for (std::size_t row = 0; row < row_count; ++row) {
            const CommitDisposition expected = cancelled[row] ? CommitDisposition::CancelledReleased
                                               : decisions[row].terminal
                                                   ? CommitDisposition::Finishable
                                                   : CommitDisposition::Active;
            if (committed.rows[row].disposition != expected) {
                throw std::logic_error("Runtime commit row disposition is invalid");
            }
            if (committed.captures[row].has_value() &&
                (decode_round || expected != CommitDisposition::Active)) {
                throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
            }
        }
        resources_.apply_commit(std::span<const LaneId>(lanes.data(), row_count), committed);
        const bool terminal_in_batch = std::any_of(
            decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
            [](const CommitDecision& decision) { return decision.terminal; });

        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request = slots_[lane_indices[row]];
            if (cancelled[row]) {
                request->generation_timings = committed.rows[row].timings;
                request->speculative_stats  = std::move(committed.rows[row].speculative);
            }
        }

        if (decode_round) {
            ++cumulative_stats_.decode_rounds;
            cumulative_stats_.decode_row_rounds += row_count;
            for (std::size_t row = 0; row < row_count; ++row) {
                if (!cancelled[row]) {
                    cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                }
            }
        }

        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane     = lane_indices[row];
                const auto& request          = slots_[lane];
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (!cancelled[row]) {
                    request->budget->commit(accepted);
                    if (decode_round) { Scheduling::consume_service_work(*request, accepted); }
                }
                auto published = request->output.commit_preview();
                auto timing    = record_committed_output(request, accepted);
                append_output(request, std::move(published), std::move(timing));
                if (decisions[row].terminal) {
                    if (cancelled[row]) {
                        terminal_requests[terminal_count] = request;
                        terminal_lanes[terminal_count]    = lane;
                        terminal_reasons[terminal_count]  = finish_reasons[row];
                        ++terminal_count;
                    } else {
                        request->model_state     = EngineRequestState::ModelFinished;
                        request->terminal_reason = finish_reasons[row];
                    }
                } else if (committed.captures[row]) {
                    if (!request->is_prefilling()) {
                        throw std::logic_error("prompt-frontier capture lost its prefill owner");
                    }
                    const EngineRequestState post_capture_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                    if (terminal_in_batch) {
                        instance_.program->skip_capture(std::move(*committed.captures[row]));
                        request->model_state = post_capture_state;
                    } else {
                        reserve_active_capture(request, std::move(*committed.captures[row]),
                                               post_capture_state);
                    }
                    committed.captures[row].reset();
                    if (!request->capture_pending) {
                        request->model_state =
                            continuations[row] == ContinuationAction::ApplyTargetControl
                                ? EngineRequestState::ControlReady
                                : EngineRequestState::DecodeReady;
                    }
                } else {
                    request->model_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                }
            }
        } catch (...) {
            phase.finish();
            for (std::size_t index = 0; index < terminal_count; ++index) {
                complete_success(terminal_requests[index], terminal_reasons[index]);
                remove_completed_slot(terminal_lanes[index]);
            }
            throw;
        }
        phase.finish();
        for (std::size_t index = 0; index < terminal_count; ++index) {
            complete_success(terminal_requests[index], terminal_reasons[index]);
            remove_completed_slot(terminal_lanes[index]);
        }
    }

    [[nodiscard]] std::shared_ptr<Request> active_capture_owner() const {
        std::shared_ptr<Request> request;
        for (std::uint32_t candidate = 0; candidate < max_concurrency_; ++candidate) {
            if (slots_[candidate] == nullptr || !slots_[candidate]->capture_pending) { continue; }
            if (request != nullptr) {
                throw std::logic_error("multiple requests own one active-capture transaction");
            }
            request = slots_[candidate];
            if (!request->lane || request->lane->value != candidate || !request->sequence) {
                throw std::logic_error("capture-pending request has no active sequence binding");
            }
        }
        return request;
    }

    void reserve_active_capture(const std::shared_ptr<Request>& request, CaptureOffer&& offer,
                                EngineRequestState post_capture_state) {
        if (!request->lane || !request->sequence || request->capture_pending ||
            post_capture_state == EngineRequestState::Materializing ||
            post_capture_state == EngineRequestState::Waiting ||
            post_capture_state == EngineRequestState::ModelFinished) {
            throw std::logic_error("committed capture offer has invalid Engine ownership");
        }
        std::uint64_t blocked = 0;
        {
            std::lock_guard lock(queue_mutex_);
            blocked = pending_.size();
        }
        for (const auto& active : slots_) {
            if (active != nullptr && active != request && !active->terminal_reason) { ++blocked; }
        }
        const std::uint32_t blocked_runnable_requests =
            blocked > std::numeric_limits<std::uint32_t>::max()
                ? std::numeric_limits<std::uint32_t>::max()
                : static_cast<std::uint32_t>(blocked);
        const auto reserved = resources_.reserve_active_capture(
            *instance_.program, *request->lane, std::move(offer), blocked_runnable_requests,
            CancellationFlagView{&request->cancelled});
        if (reserved == ResourceManagement::ActiveCaptureReserveResult::Skipped) { return; }
        request->capture_pending    = true;
        request->post_capture_state = post_capture_state;
        (void)progress_context_transaction(false);
    }

    void
    resolve_prefill_progress(const std::shared_ptr<Request>& request,
                             typename ModelContract::PrefillProgress&& progress,
                             const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        ++cumulative_stats_.host_work.prefill_units;
        ++request->host_timing.prefill_units;
        cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
        Scheduling::consume_service_work(*request, 1);
        if (!request->admitted_begin) {
            throw std::logic_error("prefill progress has no committed admission summary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens) {
            throw std::logic_error("admitted prefix exceeds its prompt");
        }
        const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
        if (request->computed_prompt_tokens > suffix_tokens ||
            progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
            throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
        }
        request->computed_prompt_tokens += progress.processed_prompt_tokens;
        if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
            throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
        }
        if (progress.processed_prompt_tokens != 0) { publish_prompt_progress(request); }
        // The lane's first unit has settled its reused state once it wrote tokens or finished the
        // prompt. A zero-progress unit (a capture offer at the reused frontier) has not, so the
        // lane stays fresh and admission stays closed until a later unit does.
        if ((progress.processed_prompt_tokens != 0 || progress.complete) && request->lane &&
            scheduler_.is_fresh_prefill_lane(request->lane->value)) {
            scheduler_.mark_prefill_settled(request->lane->value);
            request_admission_check();
        }
        if (progress.capture) {
            if (progress.complete || progress.pending) {
                throw std::logic_error("prefill capture offer overlaps prompt completion");
            }
            reserve_active_capture(request, std::move(*progress.capture),
                                   EngineRequestState::Prefill);
            progress.capture.reset();
            return;
        }
        if (!progress.complete) { return; }
        if (!request->lane || !progress.pending) {
            throw std::logic_error("completed prefill has no lane or pending token");
        }
        if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
            throw std::logic_error("runtime Begin summary differs from committed admission");
        }
        const std::uint32_t lane = request->lane->value;
        if (scheduler_.is_prefill_owner(lane)) {
            scheduler_.clear_prefill_lane(lane);
            request_admission_check();
        }
        request->begin = progress.summary;
        const std::array<std::uint32_t, 1> lanes{lane};
        phase.finish();
        commit_pending(std::move(*progress.pending), lanes, false, cancelled_at_unit_start);
        progress.pending.reset();
    }

    // Prefill owners that can run a unit now. A context transaction (materialization or capture)
    // owns the Program's reuse state, so no lane prefills while one is open; decode continues.
    [[nodiscard]] std::size_t
    collect_prefill_candidates(std::array<typename Scheduling::PrefillCandidate,
                                          kMaximumConcurrency>& candidates) const {
        std::size_t count = 0;
        if (instance_.program->has_context_transaction()) { return count; }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (!scheduler_.is_prefill_owner(lane)) { continue; }
            const auto& request = slots_[lane];
            if (request == nullptr || !request->is_prefilling()) {
                throw std::logic_error("prefill owner has no active Engine request");
            }
            // A lane whose media item still encodes in a concurrent overlay window yields its
            // prefill unit; the other lanes run meanwhile.
            if (request->capture_pending || !request->sequence || !request->admitted_begin ||
                instance_.program->vision_pending(*request->sequence)) {
                continue;
            }
            const BeginSummary& begin = *request->admitted_begin;
            const std::uint32_t suffix = begin.prompt_tokens - begin.reused_prompt_tokens;
            candidates[count++]        = {
                lane, suffix > request->computed_prompt_tokens
                             ? suffix - request->computed_prompt_tokens
                             : 0U};
        }
        return count;
    }

    void run_prefill_step(std::uint32_t lane,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange prefill_range(nvtx::Name::Prefill, nvtx::Category::Prefill);
        EnginePhaseScope setup(*this, EngineHostPhase::CommitOutput);
        if (!scheduler_.is_prefill_owner(lane)) {
            throw std::logic_error("no request owns staged prefill");
        }
        const auto request = slots_[lane];
        if (request == nullptr || !request->is_prefilling() || request->capture_pending) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        if (!request->sequence) {
            throw std::logic_error("prefill request has no sequence handle");
        }
        setup.finish();
        ProgramCallScope program_call(*this);
        auto progress = instance_.program->advance_prefill(
            *request->sequence, request->output.token_constraint(), &program_call.failed_timing());
        program_call.finish(progress.timing);
        cumulative_stats_.prefill_seconds_total +=
            static_cast<double>(progress.timing.elapsed_ns()) * 1e-9;
        consume_armed_worker_failure();
        resolve_prefill_progress(request, std::move(progress), cancelled_at_unit_start);
        publish_runtime_stats();
    }

    [[nodiscard]] FifoSnapshot pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return Scheduling::fifo_snapshot(pending_);
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    // Why a request left the waiting queue. The caller states it from the decision that removed
    // the request: re-reading the cancellation flag or the clock afterwards could attribute the
    // removal to a cancellation or expiry that happened only after the request was already gone.
    enum class WaitingRemoval { Cancelled, Expired, Failed };

    void on_waiting_removed(const std::shared_ptr<Request>& request,
                            WaitingRemoval reason) noexcept {
        if (reason != WaitingRemoval::Failed) {
            if (reason == WaitingRemoval::Cancelled) {
                ++cumulative_stats_.waiting_cancelled_requests;
            } else {
                ++cumulative_stats_.waiting_expired_requests;
            }
            cumulative_stats_.waiting_abandoned_seconds +=
                std::chrono::duration<double>(Clock::now() - request->submitted).count();
        }
        scheduler_.on_waiting_removed(request->id);
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    [[nodiscard]] ResourceInspection inspect_admission(const std::shared_ptr<Request>& request,
                                                       PlanningAllowance allowance) {
        allowance.cancellation = &request->cancelled;
        allowance.control_deadline_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           request->deadline.time_since_epoch())
                                           .count());
        return resources_.inspect(*instance_.program, request->prompt, *request->base_plan,
                                  request->publication_order, allowance);
    }

    // Inspection plans one waiting request against the context cache and the Program. It opens no
    // transaction and reserves nothing: a throw leaves no state behind to clean up, and the running
    // lanes took no part in it. Failing the Engine for it would deliver the error to every running
    // request instead of the one that cannot be planned, so fail that request alone.
    [[nodiscard]] std::optional<ResourceInspection>
    inspect_admission_or_fail(const std::shared_ptr<Request>& request, PlanningAllowance allowance) {
        try {
            consume_armed_planning_failure();
            return inspect_admission(request, allowance);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            EngineFaultEvent fault;
            try {
                fault.message   = exception_text(error);
                fault.unit      = "admission";
                fault.contained = true;
                fault.request_ids.push_back(request->id);
            } catch (...) {}
            (void)remove_pending_error(request, error);
            notify_fault(fault);
            return std::nullopt;
        }
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error,
                                                         WaitingRemoval reason =
                                                             WaitingRemoval::Failed) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        on_waiting_removed(request, reason);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress progress_context_transaction(bool yield_requested) {
        const std::optional<ContextTransactionKind> kind = resources_.context_transaction_kind();
        if (kind.has_value() != instance_.program->has_context_transaction()) {
            throw std::logic_error("Engine and Program disagree on context-transaction ownership");
        }
        if (!kind) { return AdmissionProgress::None; }
        DetailScope detail(*this, &RuntimeHostWorkStats::context_progress_ns,
                           &RuntimeHostWorkStats::context_progress_invocations,
                           nvtx::Name::ContextProgress);
        ++cumulative_stats_.host_work.control_units;

        const std::shared_ptr<Request> capture = active_capture_owner();
        std::atomic<bool> yield{yield_requested};
        CancellationFlagView cancellation{&yield};
        switch (*kind) {
        case ContextTransactionKind::Materialization: {
            if (!materializing_ || capture) {
                throw std::logic_error("materialization has conflicting Engine ownership");
            }
            const MaterializingRequest& control = *materializing_;
            const std::uint32_t lane            = control.destination.value;
            const auto& request                 = control.request;
            if (request == nullptr || !request->is_materializing() || request->lane ||
                request->sequence || request->budget || lane >= max_concurrency_ ||
                slots_[lane] != nullptr) {
                throw std::logic_error("materializing request has invalid Engine ownership");
            }
            cancellation = CancellationFlagView{&request->cancelled};
            break;
        }
        case ContextTransactionKind::ActiveCapture:
            if (materializing_ || !capture) {
                throw std::logic_error("active capture has conflicting Engine ownership");
            }
            cancellation = CancellationFlagView{&capture->cancelled};
            break;
        }

        auto outcome = resources_.progress_context_transaction(*instance_.program, cancellation);
        return std::visit(
            [&](auto&& terminal) -> AdmissionProgress {
                using Outcome = std::decay_t<decltype(terminal)>;
                if constexpr (std::is_same_v<Outcome, ContextTransactionInProgress>) {
                    return AdmissionProgress::ControlProgress;
                } else if constexpr (std::is_same_v<
                                         Outcome,
                                         typename ResourceManagement::MaterializationOutcome>) {
                    if (*kind != ContextTransactionKind::Materialization || !materializing_) {
                        throw std::logic_error(
                            "materialization outcome has no Engine control record");
                    }
                    MaterializingRequest& control = *materializing_;
                    const std::uint32_t lane      = control.destination.value;
                    const auto request            = control.request;
                    if (terminal.status == ContextTransactionStatus::Aborted) {
                        if (terminal.activation) {
                            throw std::logic_error(
                                "aborted materialization retained an activation");
                        }
                        materializing_.reset();
                        complete_detached_cancelled(request);
                        request_admission_check();
                        publish_runtime_stats();
                        return AdmissionProgress::ControlProgress;
                    }
                    if (terminal.status != ContextTransactionStatus::Published ||
                        !terminal.activation) {
                        throw std::logic_error("published materialization has no adoption token");
                    }
                    auto activation = std::move(*terminal.activation);
                    terminal.activation.reset();
                    const SequenceHandle sequence = activation.sequence();
                    resources_.adopt(*instance_.program, std::move(activation));
                    request->sequence.emplace(sequence);
                    request->budget.emplace(std::move(control.budget));
                    request->granted_output_tokens = control.summary.reserved_output_tokens;
                    request->deferred_output_tokens =
                        control.summary.effective_output_tokens -
                        control.summary.reserved_output_tokens;
                    request->deferred_limit_reason = control.summary.effective_limit_reason;
                    request->lane.emplace(control.destination);
                    request->remaining_service_work      = control.summary.service_work_quanta;
                    request->backfill_epoch              = control.protection_epoch;
                    request->backfill_class              = control.backfill_class;
                    request->materialization_diagnostics = terminal.diagnostics;
                    request->model_state                 = EngineRequestState::Prefill;
                    request->host_timing.queue_wait_ns =
                        elapsed_ns(request->submitted, Clock::now());
                    request->queue_wait_recorded = true;
                    slots_[lane]                 = request;
                    record_prefix_selection(control.summary);
                    materializing_.reset();
                    scheduler_.set_prefill_lane(lane);
                    request_admission_check();
                    publish_runtime_stats();
                    return AdmissionProgress::ControlProgress;
                } else if constexpr (std::is_same_v<
                                         Outcome,
                                         typename ResourceManagement::ActiveCaptureOutcome>) {
                    if (*kind != ContextTransactionKind::ActiveCapture || !capture) {
                        throw std::logic_error("active-capture outcome has no Engine owner");
                    }
                    if (terminal.status == ContextTransactionStatus::Published) {
                        ++cumulative_stats_.active_captures_completed;
                    } else if (terminal.status == ContextTransactionStatus::Aborted) {
                        ++cumulative_stats_.active_captures_aborted;
                    } else {
                        throw std::logic_error("active capture returned an invalid terminal state");
                    }
                    capture->capture_pending = false;
                    capture->model_state     = capture->post_capture_state;
                    request_admission_check();
                    publish_runtime_stats();
                    return AdmissionProgress::ControlProgress;
                }
                throw std::logic_error("unknown resource transaction outcome");
            },
            std::move(outcome));
    }

    [[nodiscard]] AdmissionProgress
    admit_planned_request(const std::shared_ptr<Request>& request,
                          typename ResourceManagement::Choice&& choice, AdmissionGrant grant) {
        if (Clock::now() >= request->deadline) {
            const AdmissionProgress progress = remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")),
                            WaitingRemoval::Expired);
            if (progress == AdmissionProgress::ControlProgress) { request_admission_check(); }
            return progress;
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            on_waiting_removed(request, WaitingRemoval::Cancelled);
            complete_detached_cancelled(request);
            request_admission_check();
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const LaneId destination         = choice.destination();
        const std::uint32_t lane         = destination.value;
        const RequestPlanSummary summary = choice.summary();
        if (grant.request_id() != request->id ||
            grant.service_work_quanta() != summary.service_work_quanta ||
            !scheduler_.validate_grant(grant)) {
            throw std::logic_error("admission choice lost its Scheduler grant");
        }
        // Until the rest of the output is reserved, running out of the reserved part is a length stop.
        GenerationBudget prepared_budget(summary.reserved_output_tokens,
                                         summary.reserved_output_tokens < summary.effective_output_tokens
                                             ? FinishReason::OutputLimit
                                             : summary.effective_limit_reason);
        try {
            request->generated.reserve(summary.effective_output_tokens);
        } catch (...) {
            const AdmissionProgress progress =
                remove_pending_error(request, std::current_exception());
            if (progress == AdmissionProgress::ControlProgress) { request_admission_check(); }
            return progress;
        }
        MaterializingRequest control{
            .request          = request,
            .destination      = destination,
            .budget           = std::move(prepared_budget),
            .summary          = summary,
            .backfill_class   = grant.backfill_class(),
            .protection_epoch = grant.protection_epoch(),
            .started          = Clock::now(),
        };

        const auto reserved = resources_.reserve_materialization(
            *instance_.program, std::move(choice), std::move(request->prompt),
            CancellationFlagView{&request->cancelled});
        if (reserved == ResourceManagement::MaterializationReserveResult::Stale) {
            request_admission_check();
            return AdmissionProgress::ControlProgress;
        }
        if (reserved == ResourceManagement::MaterializationReserveResult::Aborted) {
            if (!erase_pending(request)) {
                throw std::logic_error("aborted materialization lost its waiting request");
            }
            on_waiting_removed(request, WaitingRemoval::Cancelled);
            complete_detached_cancelled(request);
            request_admission_check();
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }
        if (!erase_pending(request)) {
            throw std::logic_error("admitted request disappeared from the FIFO queue");
        }
        release_planning_state(request);
        if (materializing_ || slots_[lane] != nullptr) {
            throw std::logic_error("reserved materialization destination is not empty");
        }
        request->model_state = EngineRequestState::Materializing;
        materializing_.emplace(std::move(control));
        scheduler_.commit_admission(std::move(grant));
        publish_generation_start(
            request, BeginSummary{.prompt_tokens        = summary.prompt_tokens,
                                  .reused_prompt_tokens = summary.reusable_prompt_tokens,
                                  .prefix_reuse_path    = summary.prefix_reuse_path});

        publish_runtime_stats();
        return progress_context_transaction(false);
    }

    AdmissionProgress try_admit_one() {
        const auto other_runnable = static_cast<std::uint32_t>(
            std::count_if(slots_.begin(), slots_.end(), [](const auto& request) {
                return request && !request->capture_pending &&
                       (request->is_decode_ready() || request->is_prefilling());
            }));
        const PlanningAllowance allowance = PlanningAllowance::boundary(other_runnable);
        DetailScope detail(*this, &RuntimeHostWorkStats::admission_policy_ns,
                           &RuntimeHostWorkStats::admission_policy_invocations,
                           nvtx::Name::AdmissionPolicy);
        bool control_progress = false;
        for (;;) {
            const FifoSnapshot queued = pending_snapshot();
            if (queued.empty()) {
                scheduler_.observe_fifo_head(std::nullopt);
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.head();
            scheduler_.observe_fifo_head(head->id);
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    on_waiting_removed(head, WaitingRemoval::Cancelled);
                    complete_detached_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")),
                             WaitingRemoval::Expired);
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            std::optional<ResourceInspection> head_inspected =
                inspect_admission_or_fail(head, allowance);
            if (!head_inspected) {
                control_progress = true;
                continue;
            }
            if (!hydrate_and_replan(head, head_inspected, allowance)) {
                control_progress = true;
                continue;
            }
            auto head_inspection = std::move(*head_inspected);
            if (head_inspection.readiness == Readiness::PermanentlyInfeasible) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }
            if (head_inspection.readiness == Readiness::Ready ||
                head_inspection.readiness == Readiness::NeedsTransfer) {
                if (!head_inspection.choice) {
                    throw std::logic_error("ready resource inspection has no admission choice");
                }
                AdmissionGrant grant = scheduler_.grant_head(
                    head->id, head_inspection.choice->summary().service_work_quanta);
                return admit_planned_request(head, std::move(*head_inspection.choice),
                                             std::move(grant));
            }

            const ActiveAdmissionSet active =
                scheduler_.active_admission_set(slots_, max_concurrency_);
            if (active.size == 0) {
                throw std::logic_error("isolated-feasible request is blocked in an idle Engine");
            }
            if (!scheduler_.protect_blocked_head(head->id, active.span(),
                                                 instance_.program->resource_revision())) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::optional<std::uint64_t> protection_epoch = scheduler_.protection_epoch();
            if (!protection_epoch) {
                throw std::logic_error("blocked FIFO head has no protection epoch");
            }

            std::array<SequenceHandle, kMaximumConcurrency> persistent_borrowers{};
            std::size_t persistent_borrower_count = 0;
            for (const auto& active_request : slots_) {
                if (active_request == nullptr ||
                    active_request->backfill_class != BackfillClass::Persistent ||
                    active_request->backfill_epoch != *protection_epoch) {
                    continue;
                }
                if (!active_request->sequence) {
                    throw std::logic_error("persistent borrower has no sequence reservation");
                }
                persistent_borrowers[persistent_borrower_count++] = *active_request->sequence;
            }

            for (const std::shared_ptr<Request>& candidate : queued.backfill_candidates()) {
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        on_waiting_removed(candidate, WaitingRemoval::Cancelled);
                        complete_detached_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")),
                                      WaitingRemoval::Expired);
                    control_progress = true;
                    continue;
                }
                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                std::optional<ResourceInspection> candidate_inspected =
                    inspect_admission_or_fail(candidate, allowance);
                if (!candidate_inspected) {
                    control_progress = true;
                    continue;
                }
                // A backfill candidate resumes from a stored session too. What it reads back is
                // ordinary cache, which planning for the blocked head may evict again, so the
                // persistent-backfill proof below still decides whether it may run. The proof is
                // checked first too, so a candidate that cannot backfill never pays for a read or
                // evicts anything; hydration changes the cache, so the proof is repeated after it.
                if (candidate_inspected->choice &&
                    (candidate_inspected->readiness == Readiness::Ready ||
                     candidate_inspected->readiness == Readiness::NeedsTransfer) &&
                    !resources_.prove_persistent_backfill(
                        *instance_.program, *head->base_plan, *candidate_inspected->choice,
                        std::span<const SequenceHandle>(persistent_borrowers.data(),
                                                        persistent_borrower_count))) {
                    continue;
                }
                if (!hydrate_and_replan(candidate, candidate_inspected, allowance)) {
                    control_progress = true;
                    continue;
                }
                auto candidate_inspection = std::move(*candidate_inspected);
                if (candidate_inspection.readiness == Readiness::PermanentlyInfeasible) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }
                if ((candidate_inspection.readiness != Readiness::Ready &&
                     candidate_inspection.readiness != Readiness::NeedsTransfer) ||
                    !candidate_inspection.choice) {
                    continue;
                }
                const auto proof = resources_.prove_persistent_backfill(
                    *instance_.program, *head->base_plan, *candidate_inspection.choice,
                    std::span<const SequenceHandle>(persistent_borrowers.data(),
                                                    persistent_borrower_count));
                if (!proof) { continue; }
                const RequestPlanSummary& candidate_plan = candidate_inspection.choice->summary();
                auto grant =
                    scheduler_.qualify_backfill(candidate->id, candidate_plan.service_work_quanta,
                                                active.span(), proof->resource_revision());
                if (grant) {
                    return admit_planned_request(candidate, std::move(*candidate_inspection.choice),
                                                 std::move(*grant));
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    void run_decode_round(const RoundMembership& membership,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange decode_range(nvtx::Name::Decode, nvtx::Category::Decode,
                                       static_cast<std::uint64_t>(membership.size));
        std::array<TokenMaskSource*, kMaximumConcurrency> constraints{};
        for (std::size_t row = 0; row < membership.size; ++row) {
            constraints[row] = slots_[membership.lanes[row]]->output.token_constraint();
        }
        ProgramCallScope program_call(*this);
        auto pending = instance_.program->decode(
            membership.sequence_span(), membership.budget_span(),
            std::span<TokenMaskSource* const>(constraints.data(), membership.size),
            &program_call.failed_timing());
        program_call.finish(pending.execution_timing());
        cumulative_stats_.decode_seconds_total +=
            static_cast<double>(pending.execution_timing().elapsed_ns()) * 1e-9;
        commit_pending(std::move(pending), membership.lane_span(), true, cancelled_at_unit_start);
        publish_runtime_stats();
    }

    void run_control_batch(const ControlMembership& membership) {
        nvtx::ScopedRange control_range(nvtx::Name::ControlBatch, nvtx::Category::Control,
                                        static_cast<std::uint64_t>(membership.size));
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        if (membership.empty() || membership.row_stride == 0 ||
            membership.tokens.size() !=
                static_cast<std::size_t>(membership.row_stride) * membership.size) {
            throw std::logic_error("thinking control membership is invalid");
        }

        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        bool generated_staged         = false;
        const auto rollback_generated = [&]() noexcept {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < membership.size; ++row) {
                const auto& request = slots_[membership.lanes[row]];
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };

        for (std::size_t row = 0; row < membership.size; ++row) {
            const auto& request = slots_[membership.lanes[row]];
            if (request == nullptr) {
                throw std::logic_error("thinking control membership lost its request");
            }
            generated_sizes[row] = request->generated.size();
        }
        generated_staged = true;
        try {
            for (std::size_t row = 0; row < membership.size; ++row) {
                const std::uint32_t lane = membership.lanes[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->is_control_ready() ||
                    request->capture_pending || !request->budget || !request->sequence ||
                    !request->lane || request->lane->value != lane) {
                    throw std::logic_error("thinking control row lost its active request");
                }
                const std::span<const TokenId> tokens =
                    std::span<const TokenId>(membership.tokens)
                        .subspan(row * membership.row_stride, membership.row_stride);
                const OutputDecision decision =
                    request->output.preview_control(tokens, request->budget->remaining());
                if (decision.accepted_tokens != membership.row_stride || decision.finished() ||
                    decision.continuation != ContinuationAction::Decode ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("target control preview returned an invalid decision");
                }
                prefix_execution_splits[row] = decision.prefix_execution_split_after;
                if (request->generated.size() > request->generated.capacity() ||
                    tokens.size() > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error(
                        "admission did not reserve thinking-control token capacity");
                }
                request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
            }
            phase.pause_range();
            ProgramCallScope program_call(*this);
            const runtime::ExecutionTiming timing = instance_.program->append_forced_tokens(
                membership.sequence_span(), membership.tokens, membership.row_stride,
                std::span<const std::optional<std::uint32_t>>(prefix_execution_splits.data(),
                                                              membership.size),
                &program_call.failed_timing());
            program_call.finish(timing);
            // Forced control tokens are counted as committed decode tokens below, so their
            // execution time belongs to the decode total as well.
            cumulative_stats_.decode_seconds_total +=
                static_cast<double>(timing.elapsed_ns()) * 1e-9;
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            throw;
        }
        generated_staged = false;

        ++cumulative_stats_.host_work.control_units;
        for (const std::uint32_t lane : membership.lane_span()) {
            ++slots_[lane]->host_timing.control_units;
        }

        for (std::size_t row = 0; row < membership.size; ++row) {
            const std::uint32_t lane = membership.lanes[row];
            const auto& request      = slots_[lane];
            request->budget->commit(membership.row_stride);
            Scheduling::consume_service_work(*request, membership.row_stride);
            cumulative_stats_.committed_decode_tokens += membership.row_stride;
            auto timing = record_committed_output(request, membership.row_stride);
            append_output(request, request->output.commit_preview(), std::move(timing));
            request->model_state = EngineRequestState::DecodeReady;
        }
        publish_runtime_stats();
    }

    // The worker holds execution_mutex_ across the failing operation and this cleanup, so no
    // Program introspection can observe a partially cleared physical state.
    // With a `fault`, the ids of the queued requests being failed are recorded into it from the
    // very set swapped out under the lock, so a request enqueued concurrently is either in the
    // event or was rejected by failed_ -- never failed but unreported.
    void fail_all_locked(std::exception_ptr error, EngineFaultEvent* fault = nullptr) noexcept {
        std::deque<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.swap(pending_);
        }
        if (fault != nullptr) {
            try {
                for (const auto& request : pending) {
                    fault->queued_request_ids.push_back(request->id);
                }
            } catch (...) {}
        }
        scheduler_.reset();
        const std::shared_ptr<Request> materializing_request =
            materializing_ ? materializing_->request : nullptr;
        materializing_.reset();
        instance_.program->fail_all_cleanup();
        resources_.clear_after_program_cleanup();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        if (materializing_request != nullptr) { complete_error(materializing_request, error); }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    // A host-side failure in the worker (an invariant or capacity error from the context cache, a
    // planner, or a unit's host bookkeeping; CUDA errors abort the process instead) used to latch
    // the Engine: every request failed and it served nothing more until a restart. Recover
    // instead: fail only the requests the failure could have corrupted -- the running lanes and
    // the one materializing -- clear the Program and the context cache as the latch does, and
    // keep the queue. Recovery is refused, and the latch taken, when the device cannot be
    // synchronized, when the cleanup leaves physical resources behind (the Program's state is
    // then not known to be sound), or when failures repeat with no request completing and less
    // than kRecoveryHealthyWindow between them, so a persistent fault cannot spin.
    // The caller synchronizes the device first: the cleanup frees pages and StateImages that work
    // issued before the failure may still reference.
    // Catalogs each graft the Program holds pinned as an external shared prefix, so requests that
    // select it by name plan against it.
    void catalog_pinned_grafts() {
        for (auto& entry : instance_.program->graft_catalog_entries()) {
            const std::uint32_t rm_slot = resources_.register_external_shared_prefix(
                std::move(entry.handle), std::move(entry.summary));
            instance_.program->set_graft_rm_slot(entry.name, rm_slot);
        }
    }

    // On refusal `refusal` names why, for the operator log.
    // Assigning a refusal allocates, and recover_locked is noexcept: a failed allocation must
    // leave the refusal empty and recovery refused, not terminate the process before the latch
    // is published.
    static void set_refusal(std::string& refusal, std::string_view reason,
                            std::string_view detail = {}) noexcept {
        try {
            refusal.assign(reason);
            if (!detail.empty()) {
                refusal += ": ";
                refusal += detail;
            }
        } catch (...) {
            refusal.clear();
        }
    }

    [[nodiscard]] bool recover_locked(std::exception_ptr error, std::string& refusal) noexcept {
        const auto failed_at = Clock::now();
        // The failure being handled counts toward the streak, so the third consecutive one
        // latches rather than the fourth.
        if (!recovery_streak_.permits_recovery(failed_at)) {
            set_refusal(refusal, "failures repeated with no request completing between them");
            return false;
        }
        scheduler_.reset();
        const std::shared_ptr<Request> materializing_request =
            materializing_ ? materializing_->request : nullptr;
        materializing_.reset();
        instance_.program->fail_all_cleanup();
        resources_.clear_after_program_cleanup();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        if (materializing_request != nullptr) { complete_error(materializing_request, error); }
        const auto usage = instance_.program->physical_usage();
        if (instance_.program->has_context_transaction() || usage.device_state_slots != 0 ||
            usage.host_state_slots != 0 || usage.device_main_kv_pages != 0 ||
            usage.device_backend_kv_pages != 0 || usage.host_kv_bytes != 0) {
            set_refusal(refusal, "cleanup left physical resources behind");
            return false;
        }
        // The cleanup released the startup-pinned grafts with everything else. With the empty
        // baseline verified above, reinstall them exactly as startup did; an Engine that cannot is
        // not serving what it was configured with.
        try {
            instance_.inject_pinned_grafts();
            device_.synchronize();
            catalog_pinned_grafts();
        } catch (...) {
            set_refusal(refusal, "pinned grafts could not be reinstalled",
                        exception_text(std::current_exception()));
            return false;
        }
        recovery_streak_.record_recovery(failed_at);
        ++cumulative_stats_.engine_recoveries;
        request_admission_check();
        return true;
    }

    // Worker-only, before recovery clears them: the requests a failure is about to be delivered to.
    void collect_fault_requests(EngineFaultEvent& event) const {
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                event.request_ids.push_back(slots_[lane]->id);
                event.lanes.push_back(lane);
            }
        }
        if (materializing_ && materializing_->request != nullptr) {
            event.request_ids.push_back(materializing_->request->id);
        }
    }

    void notify_fault(const EngineFaultEvent& event) const noexcept {
        if (!fault_listener_) { return; }
        try {
            fault_listener_(event);
        } catch (...) {}
    }

    void worker_loop() noexcept {
        // Decode and control units executed since the last prefill unit; saturates.
        std::uint32_t decode_run = 0;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = materializing_.has_value();
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        const auto ready = [&] { return stopping_ || !pending_.empty(); };
                        if (store_idle_ms_.load(std::memory_order_acquire) > 0) {
                            // An idle Engine still has retained sessions to keep current in the
                            // store, so wake once a second to look for them.
                            queue_cv_.wait_for(lock, std::chrono::seconds(1), ready);
                        } else {
                            queue_cv_.wait(lock, ready);
                        }
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    const auto error = std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down"));
                    std::scoped_lock execution_lock(execution_mutex_);
                    fail_all_locked(error);
                    return;
                }
            }

            std::unique_lock execution_lock(execution_mutex_);
            const char* unit = "boundary";
            try {
                set_host_work_class(HostWorkClass::Control);
                HostPhaseMeasurement boundary = begin_host_phase();
                const bool have_pending       = expire_pending_requests();
                (void)progress_context_transaction(have_pending);
                (void)settle_terminal_requests(boundary);
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary, boundary);
                RoundMembership membership =
                    scheduler_.build_round_membership(slots_, max_concurrency_);
                const bool admission_check_pending =
                    admission_check_pending_.load(std::memory_order_acquire);
                if (scheduler_.should_attempt_admission(
                        have_pending, admission_check_pending, !membership.empty(),
                        decode_run, instance_.program->has_context_transaction()) &&
                    consume_admission_check()) {
                    unit = "admission";
                    (void)try_admit_one();
                    unit = "boundary";
                    membership = scheduler_.build_round_membership(slots_, max_concurrency_);
                }

                persist_idle_session();

                // Cancellation is sampled once for the execution unit. A request arriving while
                // the GPU unit is in flight is observed at the next worker boundary; commit does
                // not reinterpret an already-issued unit with a later atomic read.
                const auto cancelled_at_unit_start = snapshot_cancellations();
                cancel_active_requests(cancelled_at_unit_start, boundary);
                extend_output_reservations();
                const ControlMembership control_membership =
                    scheduler_.build_control_membership(slots_, max_concurrency_);
                if (!control_membership.empty()) {
                    set_host_work_class(HostWorkClass::Control);
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    unit = "control";
                    run_control_batch(control_membership);
                    decode_run += decode_run != std::numeric_limits<std::uint32_t>::max();
                    continue;
                }
                membership = scheduler_.build_round_membership(slots_, max_concurrency_);

                std::array<typename Scheduling::PrefillCandidate, kMaximumConcurrency>
                    prefill_candidates{};
                const std::size_t prefill_candidate_count =
                    collect_prefill_candidates(prefill_candidates);
                const std::span<const typename Scheduling::PrefillCandidate> candidate_span{
                    prefill_candidates.data(), prefill_candidate_count};
                const std::optional<std::uint32_t> prefill_lane =
                    scheduler_.select_prefill_lane(candidate_span);
                const ExecutionAction action = scheduler_.choose_execution(
                    !membership.empty(), prefill_lane.has_value(), decode_run);
                if (action == ExecutionAction::Prefill) {
                    set_host_work_class(HostWorkClass::Prefill);
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    unit = "prefill";
                    scheduler_.record_prefill_unit(*prefill_lane, candidate_span);
                    run_prefill_step(*prefill_lane, cancelled_at_unit_start);
                    decode_run = 0;
                    continue;
                }
                if (action == ExecutionAction::Decode) {
                    set_host_work_class(HostWorkClass::Decode, membership.lane_span());
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    unit = "decode";
                    run_decode_round(membership, cancelled_at_unit_start);
                    decode_run += decode_run != std::numeric_limits<std::uint32_t>::max();
                    continue;
                }
                set_host_work_class(HostWorkClass::Control);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
            } catch (...) {
                const std::exception_ptr error = std::current_exception();
                EngineFaultEvent fault;
                try {
                    fault.message = exception_text(error);
                    fault.unit    = unit;
                    collect_fault_requests(fault);
                } catch (...) {}
                HostPhaseMeasurement cleanup = begin_host_phase();
                // Both recovery and the latch free physical state, so both wait for the device.
                bool fenced = true;
                try {
                    device_.synchronize();
                } catch (...) { fenced = false; }
                std::string refusal;
                bool recovered = false;
                if (fenced) {
                    recovered = recover_locked(error, refusal);
                } else {
                    set_refusal(refusal, "device synchronize failed");
                }
                if (!recovered) { fail_all_locked(error, &fault); }
                try {
                    fault.latched      = !recovered;
                    fault.latch_reason = refusal;
                    fault.consecutive_failures =
                        recovery_streak_.count() + (recovered ? 0U : 1U);
                    fault.maximum_consecutive_failures = recovery_streak_.maximum();
                } catch (...) {}
                // After the latch took effect, so /health already reports unavailable.
                notify_fault(fault);
                finish_engine_phase(cleanup, EngineHostPhase::Maintenance);
                try {
                    publish_runtime_stats();
                } catch (...) {}
                if (!recovered) { return; }
                continue;
            }
            execution_lock.unlock();
            std::unique_lock wait_lock(queue_mutex_);
            queue_cv_.wait_for(wait_lock, std::chrono::milliseconds(1));
        }
    }

    Instance& instance_;
    DeviceContext& device_;
    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const bool context_cache_enabled_;
    const std::function<void(const EngineFaultEvent&)> fault_listener_;
    ResourceManagement resources_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_              = 0;
    std::uint64_t next_request_id_        = 1;
    std::uint64_t next_publication_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<MaterializingRequest> materializing_;
    Scheduling scheduler_;
    std::atomic<bool> admission_check_pending_{false};
    // Worker-only: recoveries since the last request completed successfully.
    // Failures closer together than the window count toward the latch; see RecoveryStreak.
    static constexpr std::uint32_t kMaximumConsecutiveRecoveries = 3;
    static constexpr std::chrono::seconds kRecoveryHealthyWindow{30};
    RecoveryStreak recovery_streak_{kMaximumConsecutiveRecoveries, kRecoveryHealthyWindow};
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::array<std::uint32_t, kMaximumConcurrency> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    // Guarded by stats_mutex_, republished with the runtime stats.
    std::vector<SlotState> published_slots_;
    // Session persistence, guarded by execution_mutex_: the file each cell is bound to, the
    // auto-save sink, and the per-cell digest cache.
    struct SlotDigestCacheEntry {
        std::uint64_t id       = 0;
        std::uint64_t revision = 0;
        std::uint32_t depth    = 0;
        std::string digest;
        std::vector<SlotCheckpoint> checkpoints;
    };
    // How a retained session has been used, for readers deciding which are worth keeping. It
    // belongs to the session, not the cell: when a conversation moves to a new cell it moves with it.
    struct SlotUsage {
        std::uint64_t last_used_unix_ms = 0;
        std::uint32_t reuse_count       = 0;
        std::uint64_t reused_tokens     = 0;
    };

    struct CatalogContext {
        std::optional<std::uint32_t> publication;
        std::optional<std::uint32_t> retained_source;
        bool continued = false;
        std::uint32_t reused_prompt_tokens = 0;
        SlotUsage source_usage;
    };

    // What `finish` needs to know about the conversation a lane belongs to, read before it can
    // release the cell the usage record is in.
    [[nodiscard]] CatalogContext capture_catalog_context(const Request& request) const {
        CatalogContext context;
        context.publication     = resources_.lane_publication_slot(*request.lane);
        context.retained_source = resources_.lane_retained_private_source_slot(*request.lane);
        // A turn continues a conversation when it reused a private session's prefix: from a
        // cell it left retained (a rewrite or anchor restore) or, when the session's endpoint
        // was consumed, from the cell it publishes into. A turn that started from the root or a
        // shared prefix starts a conversation.
        const std::optional<BeginSummary>& begin = request.begin ? request.begin : request.admitted_begin;
        context.continued =
            context.retained_source.has_value() ||
            (begin && begin->reused_prompt_tokens > 0 &&
             begin->prefix_reuse_path != PrefixReusePath::Root &&
             begin->prefix_reuse_path != PrefixReusePath::SharedStablePrefix);
        context.reused_prompt_tokens = begin ? begin->reused_prompt_tokens : 0U;
        // The usage record travels with the conversation.
        const std::optional<std::uint32_t> usage_source =
            context.retained_source ? context.retained_source
                                    : (context.continued ? context.publication : std::nullopt);
        if (usage_source && *usage_source < slot_usage_.size()) {
            context.source_usage = slot_usage_[*usage_source];
        }
        return context;
    }

    // Output tokens left in a request's budget at which the next chunk of its output is reserved,
    // and the size of that chunk. The headroom covers the widest round (draft window included).
    static constexpr std::uint32_t kReservationHeadroom = 128;
    static constexpr std::uint32_t kReservationChunk    = 1024;

    // A request admitted with only part of its output reserved (output_reservation_tokens) gets the
    // next chunk reserved before its budget runs out, from pages nothing else holds. If none is
    // free it keeps what it has and stops, with the length finish reason, when that is spent.
    void extend_output_reservations() noexcept {
        // Growth takes free pages and advances the resource revision, which an open context
        // transaction's sealed plan is bound to; the next boundary without one retries.
        if (instance_.program->has_context_transaction() ||
            resources_.context_transaction_kind().has_value()) {
            return;
        }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || request->deferred_output_tokens == 0 || !request->budget ||
                !request->sequence || request->capture_pending ||
                !(request->is_decode_ready() || request->is_control_ready()) ||
                request->budget->remaining() > kReservationHeadroom) {
                continue;
            }
            // The whole chunk if it is free, else the largest half of it that is.
            std::uint32_t grown = 0;
            for (std::uint32_t chunk = std::min(request->deferred_output_tokens, kReservationChunk);
                 chunk != 0; chunk /= 2U) {
                if (instance_.program->grow_output_reservation(
                        *request->sequence, request->granted_output_tokens + chunk)) {
                    grown = chunk;
                    break;
                }
            }
            if (grown != 0) {
                request->granted_output_tokens += grown;
                request->deferred_output_tokens -= grown;
                request->budget->extend(grown);
                if (request->deferred_output_tokens == 0) {
                    request->budget->set_limit_reason(request->deferred_limit_reason);
                }
                ++cumulative_stats_.output_reservation_growths;
            } else {
                request->deferred_output_tokens       = 0;
                request->output_reservation_exhausted = true;
            }
        }
    }

    // Binds the retained slot's session file, usage record and digest once `finish` has catalogued
    // a continuation, for both a completed request and a cancelled prefill that kept its anchors.
    void record_catalogued_publication(const std::shared_ptr<Request>& request,
                                       FinishDisposition disposition,
                                       const CatalogContext& catalog, bool has_endpoint) {
        const auto& [publication, retained_source, continued, reused_prompt_tokens, source_usage] =
            catalog;
        if (disposition == FinishDisposition::Catalogued && publication) {
            const auto view = resources_.catalog_slot(*publication);
            if (view.state == ResourceManagement::CatalogState::Catalogued &&
                view.handle != nullptr) {
                request->retained_slot = static_cast<std::int32_t>(*publication);
                request->retained_session_digest =
                    instance_.program->continuation_digest(*view.handle);
                // A conversation continued from a retained source lives on in the new cell;
                // its slot file follows it there, leaving the older copy unbound. A cancelled
                // prefill's continuation has no endpoint and cannot be saved as a session, so it
                // must not take the binding from a cell that still can.
                if (has_endpoint && retained_source && *retained_source != *publication &&
                    *retained_source < slot_session_paths_.size() &&
                    !slot_session_paths_[*retained_source].empty()) {
                    const std::string path = slot_session_paths_[*retained_source];
                    bind_slot_session(*publication, path);
                }
                // The cell now holds the endpoint-less continuation, so a slot file bound to the
                // session it replaced (an in-place reuse) no longer describes what is resident.
                if (!has_endpoint) { clear_slot_session(*publication); }
                SlotUsage usage         = source_usage;
                usage.last_used_unix_ms = unix_time_ms();
                if (continued) {
                    ++usage.reuse_count;
                    usage.reused_tokens += reused_prompt_tokens;
                }
                if (*publication < slot_usage_.size()) { slot_usage_[*publication] = usage; }
            }
        }
    }

    // The (entry id, revision) of the session in each cell as the context store last received it. A
    // cell whose current entry matches is already durable; any change to the session (a new turn
    // consumes the entry and publishes a new one) makes it differ.
    struct PersistedState {
        std::uint64_t id       = 0;
        std::uint64_t revision = 0;

        [[nodiscard]] friend constexpr bool operator==(const PersistedState&,
                                                       const PersistedState&) noexcept = default;
    };

    std::vector<std::string> slot_session_paths_;
    std::vector<SlotDigestCacheEntry> slot_digest_cache_;
    std::vector<SlotUsage> slot_usage_;
    std::vector<PersistedState> slot_persisted_;
    std::string eviction_model_binding_;
    std::function<void(std::string, typename ModelContract::SessionSnapshot&&)> eviction_sink_;
    std::function<bool(typename ModelContract::SessionSnapshot&&)> store_sink_;
    runtime::ContextStore* store_ = nullptr; // owned by the Engine, which outlives the worker
    std::atomic<std::uint64_t> store_hydrations_{0};
    std::atomic<std::uint64_t> store_hydrated_tokens_{0};
    std::atomic<std::uint64_t> store_hydration_failures_{0};
    std::atomic<std::uint64_t> store_hydration_ns_{0};
    std::function<bool()> store_ready_;
    std::function<std::uint64_t()> store_failures_;
    std::uint64_t store_failures_seen_ = 0;
    std::uint64_t store_removals_seen_ = 0;
    // Read by the worker before it takes the execution mutex, hence atomic.
    std::atomic<std::int64_t> store_idle_ms_{0};
    Clock::time_point last_persist_scan_{};
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
