// Worker failure recovery against the real model (engine-architecture §6, "Worker 失败恢复").
//
// Failures are injected through runtime/engine/worker_fault.h, after the Program has executed the
// unit, so every recovery has live KV pages, State and (for decode) an open pending batch to
// release. Scenarios, selected with NINFER_RECOVERY_REAL_SCENARIO (default all):
//   queue      one failure fails only the in-flight request, the queued one is served, the Engine
//              returns to its startup physical baseline and then reproduces a fresh Engine's
//              greedy output; three consecutive failures latch it.
//   paused     a decode failure while a request is paused fails the resident one; the paused one
//              keeps its place, gives up its snapshot and completes by replay.
//   admission  a planning failure fails only the request being admitted, without a recovery.
//   device     a device fault latches on the first failure.
//   stages     the queue scenario with the model split into two pipeline stages on device 0.

#include "guarded_main.h"
#include "ninfer/engine.h"
#include "runtime/engine/worker_fault.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string_view setting(const char* name, std::string_view fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

ninfer::SpeculativeBackend backend(std::string_view name) {
    if (name == "none") { return ninfer::SpeculativeBackend::None; }
    if (name == "mtp") { return ninfer::SpeculativeBackend::Mtp; }
    throw std::invalid_argument("NINFER_TEST_BACKEND must be none or mtp");
}

void disarm() {
    ninfer::runtime::arm_worker_failures(0);
    ninfer::runtime::arm_decode_failures(0);
    ninfer::runtime::arm_planning_failures(0);
    ninfer::runtime::arm_device_faults(0);
}

// Fault events in delivery order; the listener runs on the worker thread.
struct FaultLog {
    std::mutex mutex;
    std::vector<ninfer::EngineFaultEvent> events;

    void install(ninfer::EngineOptions& options) {
        options.fault_listener = [this](const ninfer::EngineFaultEvent& event) {
            std::lock_guard lock(mutex);
            events.push_back(event);
        };
    }

    std::vector<ninfer::EngineFaultEvent> snapshot() {
        std::lock_guard lock(mutex);
        return events;
    }
};

// Pipeline stages for the scenario being run; empty is one device.
std::vector<int> g_devices;

ninfer::EngineOptions base_options(const std::filesystem::path& artifact,
                                   ninfer::SpeculativeBackend selected) {
    ninfer::EngineOptions options;
    options.artifact_path       = artifact;
    options.devices             = g_devices;
    options.speculative.backend = selected;
    if (selected != ninfer::SpeculativeBackend::None) {
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

ninfer::PromptInput chat(std::string text) {
    ninfer::PromptInput prompt;
    prompt.messages.push_back(ninfer::ChatMessage{
        .role  = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}}}});
    prompt.options.enable_thinking = false;
    return prompt;
}

// A prompt long enough to need several 256-token prefill units.
std::string long_text(int lineage) {
    std::string text = "Lineage " + std::to_string(lineage) +
                       ". Review this engineering checklist and summarise it in one line.";
    for (int line = 0; line < 40; ++line) {
        text += " Item " + std::to_string(line) +
                ": prefer measured figures to recalled ones, and say which you used.";
    }
    return text;
}

enum class Outcome { Served, Failed, Unavailable };

const char* name(Outcome outcome) {
    switch (outcome) {
    case Outcome::Served: return "served";
    case Outcome::Failed: return "failed";
    case Outcome::Unavailable: return "unavailable";
    }
    return "?";
}

Outcome wait_for(ninfer::GenerationHandle& handle, std::string* error_text = nullptr) {
    try {
        (void)handle.wait();
        return Outcome::Served;
    } catch (const ninfer::RequestError& error) {
        if (error_text) { *error_text = error.what(); }
        return error.kind() == ninfer::RequestErrorKind::Unavailable ? Outcome::Unavailable
                                                                     : Outcome::Failed;
    } catch (const std::exception& error) {
        if (error_text) { *error_text = error.what(); }
        return Outcome::Failed;
    }
}

// The Program's physical occupancy as the Engine publishes it.
struct Occupancy {
    std::uint32_t device_state_slots = 0, main_kv_pages = 0, backend_kv_pages = 0,
                  host_state_slots   = 0;
    std::size_t host_kv_bytes = 0, host_context_bytes = 0;

    static Occupancy of(const ninfer::RuntimeStats& stats) {
        return {stats.device_state_occupied_slots, stats.device_main_kv_occupied_pages,
                stats.device_backend_kv_occupied_pages, stats.host_state_occupied_slots,
                stats.host_kv_occupied_bytes, stats.host_context_occupied_bytes};
    }

    bool operator==(const Occupancy&) const = default;

    std::string text() const {
        return "state=" + std::to_string(device_state_slots) +
               " main_kv=" + std::to_string(main_kv_pages) +
               " backend_kv=" + std::to_string(backend_kv_pages) +
               " host_state=" + std::to_string(host_state_slots) +
               " host_kv=" + std::to_string(host_kv_bytes) +
               " host_context=" + std::to_string(host_context_bytes);
    }
};

// The published stats trail the response: wait() returns once the failed request is completed,
// and the worker publishes after it finishes the recovery.
ninfer::RuntimeStats await_recoveries(const ninfer::Engine& engine, std::uint64_t recoveries) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        auto stats = engine.runtime_stats();
        if (stats.engine_recoveries >= recoveries) { return stats; }
        require(std::chrono::steady_clock::now() < deadline,
                "recovery was never published: engine_recoveries=" +
                    std::to_string(stats.engine_recoveries));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void await_events(FaultLog& log, std::size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (log.snapshot().size() < count) {
        require(std::chrono::steady_clock::now() < deadline, "fault event was never delivered");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void exercise_queue(const std::filesystem::path& artifact, ninfer::SpeculativeBackend selected) {
    FaultLog log;
    ninfer::EngineOptions options             = base_options(artifact, selected);
    options.max_context                       = 1024;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                     = 256;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 2;
    options.context_cache.host_capacity_bytes = std::size_t{64} << 20U;
    log.install(options);
    ninfer::Engine engine(std::move(options));

    const Occupancy baseline = Occupancy::of(engine.runtime_stats());

    // The first request of a fresh Engine is the reference the recovered Engine must reproduce.
    const std::string probe_text = "Name three properties of a good unit test.";
    const ninfer::GenerationResult reference =
        engine.generate(engine.prepare(chat(probe_text)), greedy(24));
    require(reference.generated_token_ids.size() == 24, "reference request did not complete");
    // Cache it once more so the first recovery has a populated context cache to clear.
    (void)engine.generate(engine.prepare(chat(probe_text)), greedy(24));
    require(!(Occupancy::of(engine.runtime_stats()) == baseline),
            "the context cache held nothing before the first failure; the baseline check would "
            "prove nothing");

    // One failure per round, with a second request queued behind it. Three rounds reach the latch
    // count only because each served request ends the streak.
    constexpr std::uint32_t kRounds = 3;
    for (std::uint32_t round = 0; round < kRounds; ++round) {
        ninfer::runtime::arm_worker_failures(1);
        ninfer::GenerationHandle failing =
            engine.submit(engine.prepare(chat(long_text(static_cast<int>(round)))), greedy(1));
        ninfer::GenerationHandle queued = engine.submit(
            engine.prepare(chat("Queued request " + std::to_string(round) + ": say hello.")),
            greedy(1));
        std::string failure;
        const Outcome failed = wait_for(failing, &failure);
        const Outcome served = wait_for(queued);
        disarm();
        require(failed == Outcome::Failed && failure == "injected worker failure" &&
                    served == Outcome::Served && engine.is_available(),
                "round " + std::to_string(round) + ": in-flight=" + name(failed) + " (" + failure +
                    ") queued=" + name(served));
        await_recoveries(engine, round + 1);
        await_events(log, round + 1);
        const auto event = log.snapshot().back();
        require(!event.latched && !event.contained && event.unit == "prefill" &&
                    event.request_ids.size() == 1 && event.lanes.size() == 1 &&
                    event.queued_request_ids.empty() && event.consecutive_failures == 1 &&
                    event.maximum_consecutive_failures == 3 &&
                    event.message == "injected worker failure",
                "round " + std::to_string(round) + " fault event is wrong: unit=" + event.unit +
                    " latched=" + std::to_string(event.latched) +
                    " consecutive=" + std::to_string(event.consecutive_failures));
    }

    // A failure with nothing else queued: the cache the queued requests filled is cleared too, so
    // the Program is back at its startup occupancy.
    ninfer::runtime::arm_worker_failures(1);
    ninfer::GenerationHandle alone = engine.submit(engine.prepare(chat(long_text(9))), greedy(1));
    require(wait_for(alone) == Outcome::Failed, "lone injected failure did not fail its request");
    disarm();
    const auto recovered = await_recoveries(engine, kRounds + 1);
    const Occupancy after = Occupancy::of(recovered);
    require(after == baseline, "recovery did not return physical usage to the startup baseline: " +
                                   after.text() + " vs " + baseline.text());
    require(recovered.running_requests == 0 && recovered.waiting_requests == 0 &&
                recovered.paused_requests == 0 && recovered.materializing_requests == 0,
            "recovery left scheduling membership behind");

    // The recovered Engine computes the reference again from scratch and agrees with a fresh one.
    const ninfer::GenerationResult again =
        engine.generate(engine.prepare(chat(probe_text)), greedy(24));
    require(again.reused_prompt_tokens == 0,
            "a checkpoint survived recovery: reused=" + std::to_string(again.reused_prompt_tokens));
    require(again.generated_token_ids == reference.generated_token_ids,
            "greedy output after recovery differs from the fresh Engine's");

    // Three consecutive failures with no request served between them: two recover, the third
    // latches and the Engine refuses work from then on.
    const std::uint64_t before_streak = engine.runtime_stats().engine_recoveries;
    const std::size_t events_before   = log.snapshot().size();
    ninfer::runtime::arm_worker_failures(3);
    std::vector<Outcome> streak;
    for (int index = 0; index < 3; ++index) {
        ninfer::GenerationHandle handle =
            engine.submit(engine.prepare(chat(long_text(20 + index))), greedy(1));
        streak.push_back(wait_for(handle));
    }
    disarm();
    await_events(log, events_before + 3);
    const auto events = log.snapshot();
    const std::uint64_t streak_recoveries =
        engine.runtime_stats().engine_recoveries - before_streak;
    require(streak[0] == Outcome::Failed && streak[1] == Outcome::Failed &&
                streak[2] == Outcome::Failed && streak_recoveries == 2 && !engine.is_available(),
            std::string("consecutive failures did not latch on the third: ") + name(streak[0]) +
                "," + name(streak[1]) + "," + name(streak[2]) +
                " recoveries=" + std::to_string(streak_recoveries) +
                " available=" + std::to_string(engine.is_available()));
    for (std::size_t index = 0; index < 3; ++index) {
        const auto& event = events[events_before + index];
        require(event.latched == (index == 2) && event.consecutive_failures == index + 1,
                "streak event " + std::to_string(index) +
                    " latched=" + std::to_string(event.latched) +
                    " consecutive=" + std::to_string(event.consecutive_failures));
    }
    require(events.back().latch_reason.starts_with("failures repeated"),
            "latch reason is wrong: " + events.back().latch_reason);
    bool refused = false;
    try {
        (void)engine.submit(engine.prepare(chat("After the latch.")), greedy(1));
    } catch (const ninfer::RequestError& error) {
        refused = error.kind() == ninfer::RequestErrorKind::Unavailable;
    }
    require(refused, "a latched Engine accepted a request");
    std::cout << "queue: recoveries=" << kRounds + 1 << " queued_served=" << kRounds
              << " baseline=" << baseline.text() << " latched_after=3\n";
}

void exercise_paused(const std::filesystem::path& artifact, ninfer::SpeculativeBackend selected) {
    // The preemption fixture: two 192-token prompts fit together, but their continued growth to
    // 448 tokens does not, so the younger request is paused (with a Host snapshot) while the older
    // decodes.
    constexpr std::uint32_t kPrompt = 192, kOutput = 256, kCapacity = 512;
    FaultLog log;
    ninfer::EngineOptions options             = base_options(artifact, selected);
    options.max_context                       = kCapacity;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk                     = 128;
    options.max_concurrency                   = 2;
    options.max_pending_requests              = 2;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = std::size_t{512} << 20U;
    log.install(options);
    ninfer::Engine engine(std::move(options));
    const Occupancy baseline = Occupancy::of(engine.runtime_stats());

    std::vector<ninfer::TokenId> first_prompt(kPrompt, 198), second_prompt(kPrompt, 198);
    first_prompt.front()  = 1000;
    second_prompt.front() = 1001;
    ninfer::RequestOptions request                 = greedy(kOutput);
    request.execution.allow_prefix_reuse           = false;
    ninfer::GenerationHandle first  = engine.submit(engine.prepare_tokens(first_prompt), request);
    ninfer::GenerationHandle second = engine.submit(engine.prepare_tokens(second_prompt), request);

    // Arm the decode failure from the paused request's own consumer as soon as it is paused, so
    // the failing round is the resident request's.
    bool armed = false;
    const ninfer::CancellationView observer([&] {
        if (!armed && engine.runtime_stats().paused_requests == 1) {
            ninfer::runtime::arm_decode_failures(1);
            armed = true;
        }
        return false;
    });
    std::string first_error;
    std::optional<ninfer::GenerationResult> paused_result;
    std::thread first_consumer([&] { (void)wait_for(first, &first_error); });
    try {
        paused_result = second.wait(nullptr, observer);
    } catch (...) {
        first_consumer.join();
        disarm();
        throw;
    }
    first_consumer.join();
    disarm();
    require(armed, "growth pressure never paused a request");
    require(first_error == "injected decode failure",
            "the resident request did not receive the decode failure: " + first_error);
    require(paused_result->generated_token_ids.size() == kOutput &&
                paused_result->finish_reason == ninfer::FinishReason::OutputLimit,
            "the paused request did not complete its budget after recovery");
    require(paused_result->scheduling.preemptions >= 1 &&
                paused_result->scheduling.replay_restores >= 1,
            "the paused request did not restore by replay after recovery: preemptions=" +
                std::to_string(paused_result->scheduling.preemptions) +
                " replays=" + std::to_string(paused_result->scheduling.replay_restores) +
                " snapshots=" + std::to_string(paused_result->scheduling.snapshot_restores));
    await_events(log, 1);
    const auto event = log.snapshot().front();
    require(!event.latched && event.unit == "decode" && event.request_ids.size() == 1 &&
                event.queued_request_ids.empty(),
            "paused-scenario fault event is wrong: unit=" + event.unit +
                " requests=" + std::to_string(event.request_ids.size()));
    const auto stats = await_recoveries(engine, 1);
    require(stats.engine_recoveries == 1 && engine.is_available(), "recovery count is wrong");
    // Wait for the worker to settle the completed request before reading occupancy.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    Occupancy settled = Occupancy::of(engine.runtime_stats());
    while (!(settled == baseline) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        settled = Occupancy::of(engine.runtime_stats());
    }
    require(settled == baseline, "physical usage did not settle at the baseline: " +
                                     settled.text() + " vs " + baseline.text());
    const auto probe = engine.generate(engine.prepare_tokens({198, 1002, 198}), greedy(4));
    require(probe.generated_token_ids.size() == 4, "Engine did not serve after the recovery");
    std::cout << "paused: resident failed, paused request completed by replay (preemptions="
              << paused_result->scheduling.preemptions
              << " replays=" << paused_result->scheduling.replay_restores << ")\n";
}

void exercise_admission(const std::filesystem::path& artifact,
                        ninfer::SpeculativeBackend selected) {
    FaultLog log;
    ninfer::EngineOptions options             = base_options(artifact, selected);
    options.max_context                       = 1024;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk                     = 256;
    options.max_concurrency                   = 2;
    options.max_pending_requests              = 2;
    options.context_cache.host_capacity_bytes = std::size_t{64} << 20U;
    log.install(options);
    ninfer::Engine engine(std::move(options));

    ninfer::GenerationHandle running =
        engine.submit(engine.prepare(chat("Write a long story about a lighthouse keeper.")),
                      greedy(200));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (engine.runtime_stats().running_requests == 0) {
        require(std::chrono::steady_clock::now() < deadline, "the running request never started");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    ninfer::runtime::arm_planning_failures(1);
    ninfer::GenerationHandle poisoned = engine.submit(
        engine.prepare(chat("Summarise the trade-offs between paged and contiguous KV.")),
        greedy(1));
    std::string failure;
    const Outcome poisoned_outcome = wait_for(poisoned, &failure);
    disarm();
    std::size_t running_tokens = 0;
    try {
        running_tokens = running.wait().generated_token_ids.size();
    } catch (const std::exception&) {}
    const bool served_after =
        engine.generate(engine.prepare(chat("Say hello.")), greedy(1)).generated_token_ids.size() ==
        1;
    await_events(log, 1);
    const auto events = log.snapshot();
    require(poisoned_outcome == Outcome::Failed && failure == "injected planning failure" &&
                running_tokens == 200 && served_after &&
                engine.runtime_stats().engine_recoveries == 0 && engine.is_available(),
            "admission failure was not contained: poisoned=" + std::string(name(poisoned_outcome)) +
                " running_tokens=" + std::to_string(running_tokens));
    require(events.size() == 1 && events[0].contained && !events[0].latched &&
                events[0].unit == "admission" && events[0].request_ids.size() == 1,
            "contained admission fault event is wrong");
    std::cout << "admission: contained, running_tokens=" << running_tokens << " recoveries=0\n";
}

void exercise_device(const std::filesystem::path& artifact, ninfer::SpeculativeBackend selected) {
    FaultLog log;
    ninfer::EngineOptions options = base_options(artifact, selected);
    options.max_context           = 1024;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk         = 256;
    log.install(options);
    ninfer::Engine engine(std::move(options));
    ninfer::runtime::arm_device_faults(1);
    ninfer::runtime::arm_worker_failures(1);
    ninfer::GenerationHandle handle = engine.submit(engine.prepare(chat(long_text(0))), greedy(1));
    const Outcome outcome           = wait_for(handle);
    disarm();
    await_events(log, 1);
    const auto event = log.snapshot().front();
    require(outcome == Outcome::Failed && !engine.is_available() &&
                engine.runtime_stats().engine_recoveries == 0 && event.latched &&
                event.consecutive_failures == 1 && event.latch_reason.starts_with("device fault"),
            "a device fault did not latch on the first failure: latched=" +
                std::to_string(event.latched) + " reason=" + event.latch_reason);
    std::cout << "device: latched on the first failure (" << event.latch_reason << ")\n";
}

int run() {
    const auto configured = setting("NINFER_TEST_ARTIFACT", "");
    if (configured.empty()) {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const std::filesystem::path artifact(configured);
    require(std::filesystem::is_regular_file(artifact), "NINFER_TEST_ARTIFACT is not a file");
    const auto selected = backend(setting("NINFER_TEST_BACKEND", "mtp"));
    const auto scenario = setting("NINFER_RECOVERY_REAL_SCENARIO", "all");
    require(scenario == "all" || scenario == "queue" || scenario == "paused" ||
                scenario == "admission" || scenario == "device" || scenario == "stages",
            "NINFER_RECOVERY_REAL_SCENARIO must be all, queue, paused, admission, device or stages");
    try {
        if (scenario == "all" || scenario == "queue") { exercise_queue(artifact, selected); }
        if (scenario == "all" || scenario == "paused") { exercise_paused(artifact, selected); }
        if (scenario == "all" || scenario == "admission") {
            exercise_admission(artifact, selected);
        }
        if (scenario == "all" || scenario == "device") { exercise_device(artifact, selected); }
        if (scenario == "all" || scenario == "stages") {
            // The queue scenario with the model split into two stages on one device: recovery drains
            // and fences every rank, and the startup baseline covers both ranks' pages and state.
            g_devices = {0, 0};
            std::cout << "stages: ";
            exercise_queue(artifact, selected);
            g_devices.clear();
        }
    } catch (...) {
        disarm();
        throw;
    }
    std::cout << "OK engine_recovery_real\n";
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
