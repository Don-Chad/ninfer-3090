#include "ninfer/engine.h"
#include "product/object_store/s3_object_store.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

// The durable context store on the real artifact: a retained conversation survives an Engine
// restart and a crash, and a session evicted from the cache is read back for its next turn, each
// producing exactly the greedy output of a control Engine whose session never left the device.
//
//   restart    shutdown writes the session; a fresh Engine restores it at start-up
//   crash      the idle write lands while the Engine runs; the process then dies without shutdown
//   hydration  two deeper unrelated sessions evict the first from Device and Host; its next turn
//              reads it back from the store instead of prefilling
//   damaged    a store whose chunk was damaged is not trusted
namespace {

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

constexpr std::string_view kCrashChild = "--crash-child";

ninfer::EngineOptions store_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 16384;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(16384);
    options.prefill_chunk             = 1024;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.max_concurrency           = 1;
    options.max_pending_requests      = 2;
    options.context_cache.device_state_slots = 2;
    // Room for one deep session (its KV and two StateImages, about 0.85 GB) and not two: later
    // sessions must push the first one out of the Host tier as well.
    options.context_cache.host_capacity_bytes = std::size_t{1280} << 20U;
    return options;
}

ninfer::EngineOptions with_store(ninfer::EngineOptions options, const fs::path& directory,
                                 std::chrono::seconds idle) {
    options.context_store.directory      = directory;
    options.context_store.idle_persist   = idle;
    options.context_store.restore_budget = std::chrono::seconds(120);
    options.context_store.flush_budget   = std::chrono::seconds(120);
    return options;
}

ninfer::PromptInput conversation(const std::vector<std::string>& turns) {
    ninfer::PromptInput input;
    for (std::size_t index = 0; index < turns.size(); ++index) {
        ninfer::ChatMessage message;
        message.role = index % 2 == 0 ? ninfer::ChatRole::User : ninfer::ChatRole::Assistant;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = turns[index], .media = {}});
        input.messages.push_back(std::move(message));
    }
    input.options.enable_thinking = false;
    // Only the conversation's own recovery points: a public structural prefix would survive the
    // eviction below and serve the continuation in place of the stored session.
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t tokens = 12) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

std::string pump_log(int lines, int variant) {
    std::string log = "Read the following maintenance log and answer the question at the end.";
    for (int line = 0; line < lines; ++line) {
        log += " Entry " + std::to_string(line) + ": the pump on line " +
               std::to_string((line + variant) % 17) + " reported " +
               std::to_string((line * 37 + variant * 11) % 1013) +
               " kPa and the operator noted nothing unusual.";
    }
    return log;
}

const std::vector<std::string> kShortFirst{
    "List three uses for a lathe in a small workshop, one line each."};
constexpr std::string_view kShortQuestion = "Which of those needs the most care with tool speed?";
constexpr std::string_view kDeepQuestion  = "Which line reported the highest pressure?";

std::vector<std::string> next_turn(std::vector<std::string> turns, const std::string& reply,
                                   std::string_view question) {
    turns.push_back(reply);
    turns.emplace_back(question);
    return turns;
}

bool wait_for(const std::function<bool()>& condition, std::chrono::seconds limit) {
    const auto deadline = Clock::now() + limit;
    while (Clock::now() < deadline) {
        if (condition()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return condition();
}

struct Control {
    std::string reply;
    std::vector<ninfer::TokenId> tokens;
    std::uint32_t reused = 0;
    double first_token_seconds = 0.0;
};

// A conversation whose session never leaves the device.
Control control_turns(ninfer::Engine& engine, const std::vector<std::string>& first,
                      std::string_view question) {
    Control control;
    control.reply = engine.generate(engine.prepare(conversation(first)), greedy()).content;
    const auto next = engine.generate(
        engine.prepare(conversation(next_turn(first, control.reply, question))), greedy());
    control.tokens              = next.generated_token_ids;
    control.reused              = next.reused_prompt_tokens;
    control.first_token_seconds = next.timings.first_token_seconds;
    return control;
}

int check_continuation(const ninfer::GenerationResult& next, const Control& control,
                       std::string_view label) {
    if (next.reused_prompt_tokens != control.reused || next.reused_prompt_tokens == 0) {
        std::cerr << label << ": the continuation reused " << next.reused_prompt_tokens
                  << " tokens against " << control.reused << " for the warm control\n";
        return 1;
    }
    if (next.generated_token_ids != control.tokens) {
        std::cerr << label << ": output differs from the warm control\n";
        return 1;
    }
    return 0;
}

std::string self_command(const char* program) {
#if defined(_WIN32)
    char path[MAX_PATH * 4] = {};
    const DWORD length      = GetModuleFileNameA(nullptr, path, sizeof(path));
    const std::string executable(path, length);
#else
    char path[4096]    = {};
    const auto length  = readlink("/proc/self/exe", path, sizeof(path) - 1);
    const std::string executable(path, length > 0 ? static_cast<std::size_t>(length) : 0U);
#endif
    std::string command = "\"" + executable + "\"";
    // In a bundle the program name is the first argument; standalone it is the executable.
    if (fs::path(executable).stem() != fs::path(program).stem()) {
        command += std::string(" ") + program;
    }
    return command;
}

// The crash child: one turn, wait for the idle write, then die without shutting down.
int crash_child(const char* artifact, const fs::path& directory) {
    ninfer::Engine engine(
        with_store(store_engine_options(artifact), directory, std::chrono::seconds(1)));
    (void)engine.generate(engine.prepare(conversation(kShortFirst)), greedy());
    if (!wait_for([&] { return engine.runtime_stats().context_store_writes >= 1; },
                  std::chrono::seconds(60))) {
        std::cerr << "crash child: the idle session was not written in the background\n";
        std::_Exit(2);
    }
    std::cout.flush();
    std::_Exit(0);
}

int exercise_restart(const char* artifact, const fs::path& root, const Control& control) {
    const fs::path directory = root / "restart";
    {
        ninfer::Engine engine(
            with_store(store_engine_options(artifact), directory, std::chrono::seconds(0)));
        const auto reply = engine.generate(engine.prepare(conversation(kShortFirst)), greedy());
        if (reply.content != control.reply) {
            std::cerr << "restart: the first turn differs from the control\n";
            return 1;
        }
        const auto stats = engine.runtime_stats();
        if (stats.context_store_writes != 0 || stats.context_store_restored != 0) {
            std::cerr << "restart: the store was written before shutdown with idle writes off\n";
            return 1;
        }
    }
    ninfer::Engine engine(
        with_store(store_engine_options(artifact), directory, std::chrono::seconds(0)));
    const auto stats = engine.runtime_stats();
    if (stats.context_store_restored != 1 || stats.context_store_images != 1) {
        std::cerr << "restart: shutdown did not persist the session or start-up did not restore "
                     "it: restored="
                  << stats.context_store_restored << " images=" << stats.context_store_images
                  << '\n';
        return 1;
    }
    const auto next = engine.generate(
        engine.prepare(conversation(next_turn(kShortFirst, control.reply, kShortQuestion))),
        greedy());
    if (check_continuation(next, control, "restart") != 0) { return 1; }
    std::cout << "restart: restored " << stats.context_store_restored_bytes << " bytes in "
              << stats.context_store_restore_seconds << " s; continuation reused "
              << next.reused_prompt_tokens << " tokens, first token "
              << next.timings.first_token_seconds << " s (warm control "
              << control.first_token_seconds << " s)\n";
    return 0;
}

// The same restart with a deep session: an image of many chunks, restored at start-up and resumed.
int exercise_deep_restart(const char* artifact, const fs::path& root, const Control& control,
                          const std::vector<std::string>& first) {
    const fs::path directory = root / "deep-restart";
    {
        ninfer::Engine engine(
            with_store(store_engine_options(artifact), directory, std::chrono::seconds(0)));
        (void)engine.generate(engine.prepare(conversation(first)), greedy());
    }
    ninfer::Engine engine(
        with_store(store_engine_options(artifact), directory, std::chrono::seconds(0)));
    const auto stats = engine.runtime_stats();
    if (stats.context_store_restored != 1) {
        std::cerr << "deep restart: the session was not restored: restored="
                  << stats.context_store_restored << '\n';
        return 1;
    }
    const auto next = engine.generate(
        engine.prepare(conversation(next_turn(first, control.reply, kDeepQuestion))), greedy());
    if (check_continuation(next, control, "deep restart") != 0) { return 1; }
    std::cout << "deep restart: restored " << stats.context_store_restored_bytes << " bytes ("
              << stats.context_store_used_bytes << " on disk) in "
              << stats.context_store_restore_seconds << " s; continuation reused "
              << next.reused_prompt_tokens << " tokens, first token "
              << next.timings.first_token_seconds << " s\n";
    return 0;
}

int exercise_crash(const char* artifact, const char* program, const fs::path& root,
                   const Control& control) {
    const fs::path directory = root / "crash";
    const std::string command =
        self_command(program) + " " + std::string(kCrashChild) + " \"" + directory.string() + "\"";
#if defined(_WIN32)
    const int status = std::system(("\"" + command + "\"").c_str());
#else
    const int status = std::system(command.c_str());
#endif
    if (status != 0) {
        std::cerr << "crash: the child Engine failed (status " << status << ")\n";
        return 1;
    }
    ninfer::Engine engine(
        with_store(store_engine_options(artifact), directory, std::chrono::seconds(0)));
    const auto stats = engine.runtime_stats();
    if (stats.context_store_restored != 1) {
        std::cerr << "crash: the idle write did not survive the crash: restored="
                  << stats.context_store_restored << " images=" << stats.context_store_images
                  << '\n';
        return 1;
    }
    const auto next = engine.generate(
        engine.prepare(conversation(next_turn(kShortFirst, control.reply, kShortQuestion))),
        greedy());
    if (check_continuation(next, control, "crash") != 0) { return 1; }
    std::cout << "crash: restored " << stats.context_store_restored_bytes << " bytes in "
              << stats.context_store_restore_seconds << " s\n";
    return 0;
}

int exercise_hydration(const char* artifact, const fs::path& root, const Control& control,
                       const std::vector<std::string>& first) {
    const fs::path directory = root / "hydration";
    ninfer::Engine engine(
        with_store(store_engine_options(artifact), directory, std::chrono::seconds(1)));
    const auto reply = engine.generate(engine.prepare(conversation(first)), greedy());
    if (reply.content != control.reply) {
        std::cerr << "hydration: the first turn differs from the control\n";
        return 1;
    }
    if (!wait_for([&] { return engine.runtime_stats().context_store_writes >= 1; },
                  std::chrono::seconds(60))) {
        std::cerr << "hydration: the idle session was not written\n";
        return 1;
    }
    // Unrelated sessions as deep as the first, twice what Device and Host hold together, push the
    // oldest (the first) out of both.
    for (int variant = 1; variant <= 4; ++variant) {
        const std::vector<std::string> other{pump_log(300, variant)};
        (void)engine.generate(engine.prepare(conversation(other)), greedy(4));
    }
    const auto before = engine.runtime_stats();
    std::cout << "hydration: store holds " << before.context_store_images << " images, "
              << before.context_store_used_bytes << " bytes; Host "
              << before.host_context_occupied_bytes << " bytes occupied\n";
    const auto next =
        engine.generate(engine.prepare(conversation(next_turn(first, reply.content, kDeepQuestion))),
                        greedy());
    const auto stats = engine.runtime_stats();
    const auto hydrations = stats.context_store_hydrations - before.context_store_hydrations;
    const auto tokens = stats.context_store_hydrated_tokens - before.context_store_hydrated_tokens;
    if (hydrations != 1 || tokens < 4096 || stats.context_store_hydration_failures != 0) {
        std::cerr << "hydration: the continuation did not read the stored session back: "
                     "hydrations="
                  << hydrations << " tokens=" << tokens
                  << " failures=" << stats.context_store_hydration_failures
                  << " reused=" << next.reused_prompt_tokens << '\n';
        return 1;
    }
    if (check_continuation(next, control, "hydration") != 0) { return 1; }
    std::cout << "hydration: read back " << tokens << " tokens in "
              << stats.context_store_hydration_seconds - before.context_store_hydration_seconds
              << " s; first token " << next.timings.first_token_seconds
              << " s (warm control " << control.first_token_seconds << " s, prompt "
              << next.prompt.prompt_tokens << " tokens)\n";
    return 0;
}

// With an S3-compatible bucket (NINFER_TEST_S3_ENDPOINT, NINFER_TEST_S3_BUCKET; credentials from
// AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY): a second Engine with an empty directory starts warm
// from what the first uploaded at shutdown.
int exercise_bucket(const char* artifact, const fs::path& root, const Control& control) {
    const char* endpoint = std::getenv("NINFER_TEST_S3_ENDPOINT");
    const char* bucket   = std::getenv("NINFER_TEST_S3_BUCKET");
    if (endpoint == nullptr || bucket == nullptr) {
        std::cout << "bucket: skipped (NINFER_TEST_S3_ENDPOINT / NINFER_TEST_S3_BUCKET not set)\n";
        return 0;
    }
    const auto remote = [&] {
        ninfer::product::S3Config config;
        config.endpoint   = endpoint;
        config.bucket     = bucket;
        config.access_key = std::getenv("AWS_ACCESS_KEY_ID") ? std::getenv("AWS_ACCESS_KEY_ID") : "";
        config.secret_key =
            std::getenv("AWS_SECRET_ACCESS_KEY") ? std::getenv("AWS_SECRET_ACCESS_KEY") : "";
        return ninfer::product::make_s3_object_store(std::move(config));
    };
    const std::string prefix =
        "ninfer-real-" +
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "/";
    const auto options = [&](const fs::path& directory) {
        auto out = with_store(store_engine_options(artifact), directory, std::chrono::seconds(0));
        out.context_store.remote        = remote();
        out.context_store.remote_prefix = prefix;
        return out;
    };
    std::uint64_t uploaded = 0;
    {
        ninfer::Engine engine(options(root / "bucket-writer"));
        (void)engine.generate(engine.prepare(conversation(kShortFirst)), greedy());
        uploaded = engine.runtime_stats().context_store_remote_uploads;
    }
    ninfer::Engine engine(options(root / "bucket-reader"));
    const auto stats = engine.runtime_stats();
    if (stats.context_store_restored != 1 || stats.context_store_remote_downloads == 0) {
        std::cerr << "bucket: a fresh directory did not restore from the bucket: restored="
                  << stats.context_store_restored
                  << " downloads=" << stats.context_store_remote_downloads
                  << " failures=" << stats.context_store_remote_download_failures << '\n';
        return 1;
    }
    const auto next = engine.generate(
        engine.prepare(conversation(next_turn(kShortFirst, control.reply, kShortQuestion))),
        greedy());
    if (check_continuation(next, control, "bucket") != 0) { return 1; }
    std::cout << "bucket: restored " << stats.context_store_restored_bytes << " bytes ("
              << stats.context_store_remote_download_bytes << " downloaded) in "
              << stats.context_store_restore_seconds << " s; writer uploads before shutdown "
              << uploaded << '\n';
    return 0;
}

int exercise_damaged(const char* artifact, const fs::path& root) {
    const fs::path directory = root / "restart";
    // Every chunk: the images of both turns share their prefix chunks, and neither may restore.
    std::size_t damaged = 0;
    for (const auto& item : fs::recursive_directory_iterator(directory / "chunks")) {
        if (!item.is_regular_file()) { continue; }
        std::fstream file(item.path(), std::ios::in | std::ios::out | std::ios::binary);
        char byte = 0;
        file.read(&byte, 1);
        file.seekp(0);
        byte ^= 0x5a;
        file.write(&byte, 1);
        ++damaged;
    }
    if (damaged == 0) {
        std::cerr << "damaged: the store holds no chunks to damage\n";
        return 1;
    }
    ninfer::Engine engine(
        with_store(store_engine_options(artifact), directory, std::chrono::seconds(0)));
    const auto stats = engine.runtime_stats();
    if (stats.context_store_restored != 0 || stats.context_store_corrupt == 0) {
        std::cerr << "damaged: a damaged store was trusted: restored="
                  << stats.context_store_restored << " corrupt=" << stats.context_store_corrupt
                  << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        if (argc == 3 && argv[1] == kCrashChild) { return crash_child(artifact, argv[2]); }
        const fs::path root = fs::temp_directory_path() / "ninfer-context-store-real";
        fs::remove_all(root);
        fs::create_directories(root);
        struct Cleanup {
            fs::path path;
            ~Cleanup() {
                std::error_code ignored;
                fs::remove_all(path, ignored);
            }
        } const cleanup{root};

        const std::vector<std::string> deep_first{pump_log(300, 0)};
        Control short_control, deep_control;
        {
            ninfer::Engine control(store_engine_options(artifact));
            short_control = control_turns(control, kShortFirst, kShortQuestion);
            deep_control  = control_turns(control, deep_first, kDeepQuestion);
            const auto cold = control.generate(
                control.prepare(conversation(next_turn(deep_first, deep_control.reply,
                                                       kDeepQuestion))),
                [] {
                    auto options                         = greedy();
                    options.execution.allow_prefix_reuse = false;
                    return options;
                }());
            std::cout << "control: deep continuation first token " << deep_control.first_token_seconds
                      << " s warm, " << cold.timings.first_token_seconds << " s cold ("
                      << cold.prompt.prompt_tokens << " tokens)\n";
            if (deep_control.reused < 4096) {
                std::cerr << "the deep prompt is too short to exercise a hydration: reused "
                          << deep_control.reused << '\n';
                return 1;
            }
        }
        // NINFER_STORE_REAL_SCENARIO=hydration runs only that scenario (after the controls).
        const char* selected       = std::getenv("NINFER_STORE_REAL_SCENARIO");
        const std::string scenario = selected ? selected : "all";
        if (scenario == "all") {
            if (exercise_restart(artifact, root, short_control) != 0) { return 1; }
            if (exercise_crash(artifact, argv[0], root, short_control) != 0) { return 1; }
        }
        if (scenario == "all" || scenario == "deep-restart") {
            if (exercise_deep_restart(artifact, root, deep_control, deep_first) != 0) { return 1; }
        }
        if (scenario == "all" || scenario == "hydration") {
            if (exercise_hydration(artifact, root, deep_control, deep_first) != 0) { return 1; }
        }
        if (scenario == "all" || scenario == "bucket") {
            if (exercise_bucket(artifact, root, short_control) != 0) { return 1; }
        }
        if (scenario == "all" && exercise_damaged(artifact, root) != 0) { return 1; }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "context store real test failed: " << error.what() << '\n';
        return 1;
    }
}
