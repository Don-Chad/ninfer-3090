// Overlay Vision residency (`--vision-residency overlay`) on a real artifact.
//
// Greedy output is an exact oracle throughout: overlay encodes with the same kernels on the same
// weight bytes as the resident tower, and only moves where those bytes and the encode scratch live.
//
//  - equivalence: one image request, then a repeat served from the prefix cache, produce exactly
//    what resident residency produces, once with a window funded by free KV pages and once with a
//    KV cache too small for one, which takes the exclusive weight-tail window;
//  - concurrent lanes: text requests decoding while an image request runs its overlay windows
//    produce exactly what they produce without the image request, the image produces exactly its
//    single-lane output, the window ran ahead of its prefill unit on free KV pages, and the Engine
//    paused nobody (RuntimeStats::preemptions does not move);
//  - interleaved: two image requests prefilling beside a long and a short text request, each equal
//    to its serial output (the case where one lane's window has to give way to the other's).
//
// NINFER_OVERLAY_MEASURE=1 instead measures the DFlash2 rk4v4 launcher profile for both residencies:
// per-image window time, the image request's time to first token, and the device memory the tower
// costs when resident.
//
// NINFER_OVERLAY_SCENARIO selects one of equivalence, concurrent, interleaved (default: all three).

#include "guarded_main.h"
#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kOutputTokens = 32;

const char* residency_name(ninfer::VisionResidency residency) {
    return residency == ninfer::VisionResidency::Overlay ? "overlay" : "resident";
}

ninfer::EngineOptions engine_options(const char* artifact, ninfer::VisionResidency residency,
                                     std::uint32_t lanes, std::uint32_t context = 8192) {
    ninfer::EngineOptions options;
    options.artifact_path    = artifact;
    options.max_context      = context;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(context * lanes);
    options.max_concurrency  = lanes;
    options.prefill_chunk    = 1024;
    options.kv_cache         = ninfer::KvCacheStorage::Int8Group64;
    options.enable_vision    = true;
    options.vision_residency = residency;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t tokens, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<std::uint8_t> gradient_ppm(int width, int height) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput image_prompt(int width, int height, std::string_view question) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm(width, height);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "overlay.ppm";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::string(question), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

// Token ids inside the vocabulary and not special.
std::vector<ninfer::TokenId> text_prompt(std::uint32_t tokens, std::uint32_t salt) {
    std::vector<ninfer::TokenId> prompt{248045};
    for (std::uint32_t index = 0; index < tokens; ++index) {
        prompt.push_back(static_cast<ninfer::TokenId>(1000 + (index * 37U + salt * 101U) % 5000U));
    }
    return prompt;
}

std::vector<ninfer::TokenId> short_prompt() { return {248045, 846, 198, 5834, 248046, 198}; }

void print_ids(const char* label, const std::vector<ninfer::TokenId>& ids) {
    std::cerr << "  " << label << ':';
    for (const ninfer::TokenId id : ids) { std::cerr << ' ' << id; }
    std::cerr << '\n';
}

bool engine_settles(const ninfer::Engine& engine) {
    for (int attempt = 0; attempt < 200; ++attempt) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.running_requests == 0 && stats.prefilling_requests == 0 &&
            stats.materializing_requests == 0 && stats.capture_pending_requests == 0 &&
            stats.waiting_requests == 0 && stats.terminal_pending_requests == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

void print_overlay(const char* label, const ninfer::GenerationTimings& timings) {
    std::cout << "  " << label << ": vision " << timings.vision_seconds * 1000.0 << " ms, windows "
              << timings.overlay_windows << " (ahead " << timings.overlay_ahead_windows
              << ", exclusive " << timings.overlay_exclusive_windows << "), window "
              << timings.overlay_window_seconds * 1000.0 << " ms, borrowed "
              << timings.overlay_evicted_bytes / (1024.0 * 1024.0) << " MiB, staged "
              << timings.overlay_staged_bytes / (1024.0 * 1024.0) << " MiB\n";
}

struct ImageRun {
    std::vector<ninfer::TokenId> first;
    std::vector<ninfer::TokenId> repeat;
    std::uint32_t repeat_reused = 0;
    bool had_media              = false;
    ninfer::GenerationTimings timings;
};

ImageRun run_image(const char* artifact, ninfer::VisionResidency residency,
                   std::uint32_t context) {
    ninfer::Engine engine(engine_options(artifact, residency, 1, context));
    ImageRun out;
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(image_prompt(512, 384, "Describe the picture.")),
                        greedy(kOutputTokens, true));
    out.first     = first.generated_token_ids;
    out.had_media = first.prompt.has_media && first.timings.vision_seconds > 0.0;
    out.timings   = first.timings;
    const ninfer::GenerationResult repeat =
        engine.generate(engine.prepare(image_prompt(512, 384, "Describe the picture.")),
                        greedy(kOutputTokens, true));
    out.repeat        = repeat.generated_token_ids;
    out.repeat_reused = repeat.reused_prompt_tokens;
    return out;
}

// `context` 8192 leaves free KV pages enough for the window; 1024 does not (the whole Main KV cache
// is smaller than one window), which forces the exclusive weight-tail window.
int exercise_equivalence(const char* artifact, std::uint32_t context, bool exclusive) {
    const ImageRun resident = run_image(artifact, ninfer::VisionResidency::Resident, context);
    const ImageRun overlay  = run_image(artifact, ninfer::VisionResidency::Overlay, context);
    print_overlay("overlay image", overlay.timings);
    if (resident.first.size() != kOutputTokens || !resident.had_media ||
        resident.repeat_reused == 0) {
        std::cerr << "equivalence: the resident reference did not encode or reuse its image\n";
        return 1;
    }
    if (overlay.first != resident.first || overlay.repeat != resident.repeat ||
        overlay.repeat_reused != resident.repeat_reused || !overlay.had_media) {
        std::cerr << "equivalence: overlay output differs from resident\n";
        print_ids("resident", resident.first);
        print_ids("overlay ", overlay.first);
        print_ids("resident repeat", resident.repeat);
        print_ids("overlay repeat ", overlay.repeat);
        return 1;
    }
    if (overlay.timings.overlay_windows != 1 ||
        overlay.timings.overlay_exclusive_windows != (exclusive ? 1U : 0U)) {
        std::cerr << "equivalence: the overlay image opened " << overlay.timings.overlay_windows
                  << " windows (" << overlay.timings.overlay_exclusive_windows
                  << " exclusive), expected one " << (exclusive ? "weight-tail" : "KV-funded")
                  << " window\n";
        return 1;
    }
    std::cout << "equivalence (" << (exclusive ? "weight-tail" : "KV-funded")
              << " window): overlay matches resident, first turn and prefix-reused repeat\n";
    return 0;
}

// Text lanes decode while an image request runs its overlay window. The text lanes are compared
// with the same two requests on the same Engine configuration without the image request.
int exercise_concurrent(const char* artifact, ninfer::VisionResidency residency) {
    const char* label        = residency_name(residency);
    constexpr std::uint32_t kTextOutput = 160;
    // Decode on this Engine is not batch-invariant: a third row in the batch can flip a near-tie
    // in greedy output. The image request therefore stops at its first token, which prefill
    // produces, so it never joins a decode batch: the text lanes decode in the same composition as
    // in the reference run, and any difference in their output can only come from the image
    // request's prefill and its window.
    constexpr std::uint32_t kImageTokens = 1;
    const auto long_text     = text_prompt(1500, 1);
    const auto short_text    = short_prompt();
    const auto image_request = [] { return image_prompt(768, 512, "What is in this image?"); };

    std::vector<ninfer::TokenId> image_reference;
    {
        ninfer::Engine engine(engine_options(artifact, residency, 1));
        image_reference =
            engine.generate(engine.prepare(image_request()), greedy(kImageTokens))
                .generated_token_ids;
    }

    ninfer::Engine engine(engine_options(artifact, residency, 3));
    std::vector<ninfer::TokenId> long_reference;
    std::vector<ninfer::TokenId> short_reference;
    {
        auto a = engine.submit(engine.prepare_tokens(long_text), greedy(kTextOutput));
        auto b = engine.submit(engine.prepare_tokens(short_text), greedy(kTextOutput));
        long_reference  = a.wait().generated_token_ids;
        short_reference = b.wait().generated_token_ids;
    }
    if (long_reference.size() != kTextOutput || short_reference.size() != kTextOutput ||
        image_reference.size() != kImageTokens) {
        std::cerr << label << " concurrent: a reference run did not produce its output\n";
        return 1;
    }
    if (!engine_settles(engine)) {
        std::cerr << label << " concurrent: reference requests did not settle\n";
        return 1;
    }

    const ninfer::RuntimeStats before = engine.runtime_stats();
    auto a = engine.submit(engine.prepare_tokens(long_text), greedy(kTextOutput));
    auto b = engine.submit(engine.prepare_tokens(short_text), greedy(kTextOutput));
    // Let both text lanes reach decode before the image arrives, so its window opens beside them.
    for (int attempt = 0; attempt < 400; ++attempt) {
        const ninfer::RuntimeStats now = engine.runtime_stats();
        if (now.committed_decode_tokens >= before.committed_decode_tokens + 16 &&
            now.prefilling_requests == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    auto image                             = engine.submit(engine.prepare(image_request()),
                                                           greedy(kImageTokens));
    const ninfer::GenerationResult image_r = image.wait();
    const ninfer::GenerationResult long_r  = a.wait();
    const ninfer::GenerationResult short_r = b.wait();
    if (!engine_settles(engine)) {
        std::cerr << label << " concurrent: requests did not settle\n";
        return 1;
    }
    const ninfer::RuntimeStats after = engine.runtime_stats();
    const std::uint64_t preemptions  = after.preemptions - before.preemptions;
    std::cout << label << " concurrent: image TTFT " << image_r.timings.first_token_seconds * 1000.0
              << " ms, preemptions +" << preemptions << ", decode rounds +"
              << after.decode_rounds - before.decode_rounds << '\n';
    print_overlay("image", image_r.timings);

    int failures = 0;
    if (long_r.generated_token_ids != long_reference ||
        short_r.generated_token_ids != short_reference) {
        std::cerr << label << " concurrent: a text lane changed while an image ran beside it\n";
        print_ids("reference long ", long_reference);
        print_ids("got       long ", long_r.generated_token_ids);
        print_ids("reference short", short_reference);
        print_ids("got       short", short_r.generated_token_ids);
        ++failures;
    }
    if (image_r.generated_token_ids != image_reference || !image_r.prompt.has_media) {
        std::cerr << label << " concurrent: the image output changed beside decoding lanes\n";
        print_ids("reference image", image_reference);
        print_ids("got       image", image_r.generated_token_ids);
        ++failures;
    }
    if (preemptions != 0) {
        std::cerr << label << " concurrent: the image request caused " << preemptions
                  << " preemptions\n";
        ++failures;
    }
    if (residency == ninfer::VisionResidency::Overlay &&
        (image_r.timings.overlay_windows != 1 || image_r.timings.overlay_exclusive_windows != 0 ||
         image_r.timings.overlay_ahead_windows != 1)) {
        std::cerr << label
                  << " concurrent: expected one KV-funded window encoded ahead of its unit\n";
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

// Two images prefill beside a long and a short text prompt, each equal to its serial output.
int exercise_interleaved(const char* artifact) {
    const auto long_text  = text_prompt(3000, 0);
    const auto short_text = short_prompt();
    const auto image_a    = [] { return image_prompt(1024, 1024, "What is visible?"); };
    const auto image_b    = [] { return image_prompt(512, 768, "Describe the colors."); };
    const auto residency  = ninfer::VisionResidency::Overlay;

    std::vector<ninfer::TokenId> ref_long, ref_short, ref_a, ref_b;
    {
        ninfer::Engine engine(engine_options(artifact, residency, 1));
        const auto run = [&](ninfer::PreparedPrompt prompt) {
            return engine.generate(std::move(prompt), greedy(kOutputTokens)).generated_token_ids;
        };
        ref_long  = run(engine.prepare_tokens(long_text));
        ref_short = run(engine.prepare_tokens(short_text));
        ref_a     = run(engine.prepare(image_a()));
        ref_b     = run(engine.prepare(image_b()));
    }
    ninfer::Engine engine(engine_options(artifact, residency, 3));
    const ninfer::RuntimeStats before = engine.runtime_stats();
    const auto submit = [&](ninfer::PreparedPrompt prompt) {
        return engine.submit(std::move(prompt), greedy(kOutputTokens));
    };
    auto long_h  = submit(engine.prepare_tokens(long_text));
    auto a_h     = submit(engine.prepare(image_a()));
    auto b_h     = submit(engine.prepare(image_b()));
    auto short_h = submit(engine.prepare_tokens(short_text));
    const auto short_r = short_h.wait();
    const auto long_r  = long_h.wait();
    const auto a_r     = a_h.wait();
    const auto b_r     = b_h.wait();
    if (!engine_settles(engine)) {
        std::cerr << "interleaved: requests did not settle\n";
        return 1;
    }
    std::cout << "interleaved overlay: preemptions +"
              << engine.runtime_stats().preemptions - before.preemptions << '\n';
    print_overlay("image A", a_r.timings);
    print_overlay("image B", b_r.timings);
    if (long_r.generated_token_ids != ref_long || short_r.generated_token_ids != ref_short ||
        a_r.generated_token_ids != ref_a || b_r.generated_token_ids != ref_b) {
        std::cerr << "interleaved: interleaving overlay vision prefill changed an output\n";
        print_ids("serial image A", ref_a);
        print_ids("got    image A", a_r.generated_token_ids);
        print_ids("serial image B", ref_b);
        print_ids("got    image B", b_r.generated_token_ids);
        print_ids("serial long   ", ref_long);
        print_ids("got    long   ", long_r.generated_token_ids);
        print_ids("serial short  ", ref_short);
        print_ids("got    short  ", short_r.generated_token_ids);
        return 1;
    }
    return 0;
}

// The DFlash2 launcher profile (rk4v4 KV) for one residency.
ninfer::EngineOptions profile_options(const char* artifact, ninfer::VisionResidency residency) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 65536;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(65536);
    options.max_concurrency                  = 1;
    options.prefill_chunk                    = 4096;
    options.kv_cache                         = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.speculative.backend              = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens         = 7;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.prefill_cublas                   = true;
    options.gdn_state_fp16                   = true;
    options.enable_vision                    = true;
    options.vision_residency                 = residency;
    return options;
}

int measure(const char* artifact) {
    struct Row {
        double ttft_ms    = 0.0;
        double vision_ms  = 0.0;
        double window_ms  = 0.0;
        std::size_t weights   = 0;
        std::size_t workspace = 0;
        std::size_t sequence  = 0;
        std::size_t window    = 0;
        std::size_t pinned    = 0;
        std::vector<ninfer::TokenId> output;
    };
    const auto run = [&](ninfer::VisionResidency residency) {
        ninfer::Engine engine(profile_options(artifact, residency));
        Row row;
        const ninfer::MemorySummary memory = engine.memory_summary();
        const ninfer::LoadSummary load     = engine.load_summary();
        row.weights   = memory.weights.capacity_bytes;
        row.workspace = memory.workspace.capacity_bytes;
        row.sequence  = memory.sequence.capacity_bytes;
        row.window    = load.overlay_window_bytes;
        row.pinned    = load.pinned_weight_bytes;
        // Warm-up: first-use kernels, cuBLAS handles and graphs, on a different image.
        (void)engine.generate(engine.prepare(image_prompt(640, 480, "Warm up.")), greedy(4));
        constexpr int kRuns = 5;
        for (int index = 0; index < kRuns; ++index) {
            // A fresh question each time defeats the prefix cache, so every run encodes.
            const ninfer::GenerationResult result = engine.generate(
                engine.prepare(image_prompt(1280, 960, "Run " + std::to_string(index) +
                                                           ": what is in this image?")),
                greedy(kOutputTokens));
            row.ttft_ms += result.timings.first_token_seconds * 1000.0 / kRuns;
            row.vision_ms += result.timings.vision_seconds * 1000.0 / kRuns;
            row.window_ms += result.timings.overlay_window_seconds * 1000.0 / kRuns;
            std::cout << "  " << residency_name(residency) << " run " << index << ": TTFT "
                      << result.timings.first_token_seconds * 1000.0 << " ms, vision "
                      << result.timings.vision_seconds * 1000.0 << " ms, windows "
                      << result.timings.overlay_windows << " (exclusive "
                      << result.timings.overlay_exclusive_windows << ", ahead "
                      << result.timings.overlay_ahead_windows << "), borrowed "
                      << result.timings.overlay_evicted_bytes / (1024.0 * 1024.0) << " MiB\n";
            if (index == 0) { row.output = result.generated_token_ids; }
        }
        return row;
    };
    const Row resident = run(ninfer::VisionResidency::Resident);
    const Row overlay  = run(ninfer::VisionResidency::Overlay);
    const auto mib     = [](std::size_t bytes) { return static_cast<double>(bytes) / 1048576.0; };
    std::cout << std::fixed << std::setprecision(1);
    for (const auto& [label, row] : {std::pair{"resident", &resident}, std::pair{"overlay", &overlay}}) {
        std::cout << label << ": TTFT " << row->ttft_ms << " ms, vision " << row->vision_ms
                  << " ms, window " << row->window_ms << " ms; device weights "
                  << mib(row->weights) << " MiB, workspace " << mib(row->workspace)
                  << " MiB, sequence " << mib(row->sequence) << " MiB; window "
                  << mib(row->window) << " MiB, pinned tower " << mib(row->pinned) << " MiB\n";
    }
    const auto device = [](const Row& row) { return row.weights + row.workspace + row.sequence; };
    std::cout << "device reservation saved by overlay: "
              << (static_cast<double>(device(resident)) - static_cast<double>(device(overlay))) /
                     1048576.0
              << " MiB\n";
    if (resident.output != overlay.output) {
        std::cerr << "measure: overlay output differs from resident on the DFlash2 profile\n";
        return 1;
    }
    return 0;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    if (const char* value = std::getenv("NINFER_OVERLAY_MEASURE");
        value != nullptr && value[0] != '\0' && value[0] != '0') {
        return measure(artifact);
    }
    const char* selected = std::getenv("NINFER_OVERLAY_SCENARIO");
    const std::string scenario = selected != nullptr ? selected : "all";
    int failures               = 0;
    if (scenario == "all" || scenario == "equivalence") {
        failures += exercise_equivalence(artifact, 8192, false);
        failures += exercise_equivalence(artifact, 1024, true);
    }
    if (scenario == "all" || scenario == "concurrent") {
        failures += exercise_concurrent(artifact, ninfer::VisionResidency::Overlay);
        // Control: the same schedule with the tower resident separates what concurrent decode
        // batching does to greedy output from what the overlay window does.
        if (failures != 0) {
            (void)exercise_concurrent(artifact, ninfer::VisionResidency::Resident);
        }
    }
    if (scenario == "all" || scenario == "interleaved") { failures += exercise_interleaved(artifact); }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
