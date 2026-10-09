// Overlay Vision residency (`--vision-residency overlay`) on a real artifact.
//
// Greedy output is an exact oracle throughout: overlay encodes with the same kernels on the same
// weight bytes as the resident tower, and only moves where those bytes and the encode scratch live.
//
//  - equivalence: one image request, then a repeat served from the prefix cache, produce exactly
//    what resident residency produces, once with a window funded by free KV pages and once with a
//    KV cache too small for one, which takes the exclusive weight-tail window;
//  - concurrent: a text lane decoding while four image requests run overlay windows beside it (one
//    alone, then three at once) produces exactly its alone output; every image's first token and
//    cached prefix match a run without concurrency; the windows ran ahead of their prefill units on
//    free KV pages; the Engine paused nobody (RuntimeStats::preemptions does not move);
//  - interleaved: two image requests prefilling beside a long and a short text request (two
//    windows competing), every first token and cached prefix equal to a serial run.
//
// Decode is not batch-invariant on this Engine, so no comparison depends on which lanes share a
// decode round; see the comment above exercise_batch_variance.
//
// NINFER_OVERLAY_MEASURE=1 instead measures the DFlash2 rk4v4 launcher profile for both residencies:
// per-image window time, the image request's time to first token, and the device memory the tower
// costs when resident.
//
// NINFER_OVERLAY_SCENARIO selects equivalence, concurrent or interleaved (default: all three), or a
// diagnostic: batch-variance (no Vision) or reuse-control. NINFER_OVERLAY_REPEAT=N repeats the
// concurrent and interleaved scenarios (batch-variance: N concurrent runs).

#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <array>
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

// Decode on this Engine is not batch-invariant: the rows that share a decode round change the
// arithmetic of each row (split and tile choices follow the batch size), so a greedy near-tie can
// flip when the round composition changes. Which rounds two lanes share depends on host timing
// (when the worker admits each request relative to the other's prefill units), so comparing
// concurrent lanes with serial runs, or with another concurrent run, is not an oracle. Every
// comparison below keeps the round composition identical by construction:
//  - a text lane that decodes is the only lane that ever decodes, in the reference and in the test;
//  - image requests stop at their first token, which prefill produces, so they never join a
//    decode round; prefill is per lane and has no batch dependence;
//  - the KV and state a prefill wrote under concurrent windows are then checked by continuing each
//    request alone from its cached prompt prefix, against the same continuation from a prefix the
//    reference wrote with no concurrency. A continuation from a cached prefix is not always
//    bit-identical to a run from scratch on this Engine, in either residency (see the
//    reuse-control diagnostic), so the reference takes the same route.

// Diagnostic: the same short prompt alone and beside a long one, without any Vision. Reports how
// often the concurrent output differs from the alone output and from itself.
int exercise_batch_variance(const char* artifact, int repeats) {
    constexpr std::uint32_t kTokens = 160;
    const auto long_text            = text_prompt(1500, 1);
    const auto short_text           = short_prompt();
    ninfer::EngineOptions options   = engine_options(artifact, ninfer::VisionResidency::Resident, 3);
    options.enable_vision           = false;
    ninfer::Engine engine(options);
    const auto alone =
        engine.generate(engine.prepare_tokens(short_text), greedy(kTokens)).generated_token_ids;
    std::vector<std::vector<ninfer::TokenId>> outputs;
    for (int index = 0; index < repeats; ++index) {
        auto a = engine.submit(engine.prepare_tokens(long_text), greedy(kTokens));
        auto b = engine.submit(engine.prepare_tokens(short_text), greedy(kTokens));
        outputs.push_back(b.wait().generated_token_ids);
        (void)a.wait();
        (void)engine_settles(engine);
    }
    int differ_alone = 0;
    std::vector<std::vector<ninfer::TokenId>> distinct;
    for (const auto& output : outputs) {
        if (output != alone) {
            ++differ_alone;
            std::size_t first = 0;
            while (first < output.size() && output[first] == alone[first]) { ++first; }
            std::cout << "  batch-variance: concurrent short prompt diverges from alone at token "
                      << first << '\n';
        }
        if (std::find(distinct.begin(), distinct.end(), output) == distinct.end()) {
            distinct.push_back(output);
        }
    }
    const auto window = [](const std::vector<ninfer::TokenId>& ids) {
        std::string text;
        for (std::size_t index = 20; index < std::min<std::size_t>(ids.size(), 30); ++index) {
            text += ' ' + std::to_string(ids[index]);
        }
        return text;
    };
    std::cout << "  alone tokens 20-29:" << window(alone) << '\n';
    for (const auto& output : distinct) {
        std::cout << "  concurrent tokens 20-29:" << window(output) << '\n';
    }
    std::cout << "batch-variance (no Vision): " << differ_alone << "/" << repeats
              << " concurrent runs differ from the alone run, " << distinct.size()
              << " distinct concurrent outputs\n";
    return 0;
}

// Diagnostic: on one lane with no concurrency at all, does continuing an image request from the
// prefix its first-token request cached reproduce a run from scratch?
int exercise_reuse_control(const char* artifact) {
    for (const auto residency : {ninfer::VisionResidency::Resident,
                                 ninfer::VisionResidency::Overlay}) {
        ninfer::Engine engine(engine_options(artifact, residency, 1));
        for (const auto& [width, height] : {std::pair{512, 768}, std::pair{640, 480}}) {
            const auto prompt = [&] { return image_prompt(width, height, "Describe the colors."); };
            const auto fresh =
                engine.generate(engine.prepare(prompt()), greedy(kOutputTokens)).generated_token_ids;
            (void)engine.generate(engine.prepare(prompt()), greedy(1, true));
            const auto continued =
                engine.generate(engine.prepare(prompt()), greedy(kOutputTokens, true));
            std::size_t first = 0;
            while (first < fresh.size() && first < continued.generated_token_ids.size() &&
                   fresh[first] == continued.generated_token_ids[first]) {
                ++first;
            }
            std::cout << "reuse-control " << residency_name(residency) << ' ' << width << 'x'
                      << height << ": reused " << continued.reused_prompt_tokens << ", "
                      << (continued.generated_token_ids == fresh
                              ? std::string("identical")
                              : "diverges at token " + std::to_string(first))
                      << '\n';
        }
    }
    return 0;
}

// One text lane decodes while image requests prefill beside it in overlay windows. The text lane
// decodes alone in every round in both runs, so its output must equal its alone run exactly.
int exercise_concurrent(const char* artifact) {
    constexpr std::uint32_t kTextOutput = 400;
    const auto long_text                = text_prompt(1500, 1);
    const std::vector<std::pair<int, int>> sizes{{768, 512}, {1024, 1024}, {512, 768}, {640, 480}};
    const auto image_request = [&](std::size_t index) {
        return image_prompt(sizes[index].first, sizes[index].second,
                            "Image " + std::to_string(index) + ": what is in it?");
    };
    const auto residency = ninfer::VisionResidency::Overlay;

    std::vector<std::vector<ninfer::TokenId>> image_first;
    std::vector<std::vector<ninfer::TokenId>> image_serial;
    {
        ninfer::Engine engine(engine_options(artifact, residency, 1));
        for (std::size_t index = 0; index < sizes.size(); ++index) {
            image_first.push_back(
                engine.generate(engine.prepare(image_request(index)), greedy(1, true))
                    .generated_token_ids);
            image_serial.push_back(engine
                                       .generate(engine.prepare(image_request(index)),
                                                 greedy(kOutputTokens, true))
                                       .generated_token_ids);
        }
    }

    ninfer::Engine engine(engine_options(artifact, residency, 3));
    const auto text_reference =
        engine.generate(engine.prepare_tokens(long_text), greedy(kTextOutput)).generated_token_ids;
    if (text_reference.size() != kTextOutput || !engine_settles(engine)) {
        std::cerr << "concurrent: the text reference did not produce its output\n";
        return 1;
    }

    const ninfer::RuntimeStats before = engine.runtime_stats();
    auto text = engine.submit(engine.prepare_tokens(long_text), greedy(kTextOutput));
    for (int attempt = 0; attempt < 400; ++attempt) {
        if (engine.runtime_stats().committed_decode_tokens >= before.committed_decode_tokens + 16) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // Two waves: one image alone, then three at once, all while the text lane keeps decoding.
    std::vector<ninfer::GenerationResult> images;
    images.push_back(engine.submit(engine.prepare(image_request(0)), greedy(1, true)).wait());
    {
        std::vector<ninfer::GenerationHandle> handles;
        for (std::size_t index = 1; index < sizes.size(); ++index) {
            handles.push_back(engine.submit(engine.prepare(image_request(index)), greedy(1, true)));
        }
        for (auto& handle : handles) { images.push_back(handle.wait()); }
    }
    const ninfer::RuntimeStats mid      = engine.runtime_stats();
    const ninfer::GenerationResult text_r = text.wait();
    if (!engine_settles(engine)) {
        std::cerr << "concurrent: requests did not settle\n";
        return 1;
    }
    const ninfer::RuntimeStats after = engine.runtime_stats();
    const std::uint64_t preemptions  = after.preemptions - before.preemptions;
    std::uint32_t ahead = 0, exclusive = 0;
    for (const auto& image : images) {
        ahead += image.timings.overlay_ahead_windows;
        exclusive += image.timings.overlay_exclusive_windows;
    }
    std::cout << "concurrent: " << images.size() << " images, windows ahead " << ahead
              << ", exclusive " << exclusive << ", text decode tokens during the images "
              << mid.committed_decode_tokens - before.committed_decode_tokens
              << ", preemptions +" << preemptions << '\n';

    int failures = 0;
    if (text_r.generated_token_ids != text_reference) {
        std::cerr << "concurrent: the text lane changed while images ran overlay windows\n";
        print_ids("reference text", text_reference);
        print_ids("got       text", text_r.generated_token_ids);
        ++failures;
    }
    for (std::size_t index = 0; index < images.size(); ++index) {
        if (images[index].generated_token_ids != image_first[index]) {
            std::cerr << "concurrent: image " << index << "'s first token changed\n";
            ++failures;
        }
    }
    if (preemptions != 0) {
        std::cerr << "concurrent: the images caused " << preemptions << " preemptions\n";
        ++failures;
    }
    if (ahead == 0 || exclusive != 0) {
        std::cerr << "concurrent: expected KV-funded windows encoded ahead of their units\n";
        ++failures;
    }
    // The prefixes written beside the decoding lane, continued alone.
    for (std::size_t index = 0; index < sizes.size(); ++index) {
        const auto continued =
            engine.generate(engine.prepare(image_request(index)), greedy(kOutputTokens, true));
        if (continued.generated_token_ids != image_serial[index]) {
            std::cerr << "concurrent: image " << index << " continued from its concurrent prefix ("
                      << continued.reused_prompt_tokens << " reused tokens) differs\n";
            print_ids("serial   ", image_serial[index]);
            print_ids("continued", continued.generated_token_ids);
            ++failures;
        } else if (continued.reused_prompt_tokens == 0) {
            std::cerr << "concurrent: image " << index << " found no cached prefix\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}

// Two images prefill beside a long and a short text prompt, all four at once, so two windows
// compete and text prefill units run between encodes. Every request stops at its first token
// (no decode rounds); each is then continued alone from its cached prefix and compared with a
// serial run computed from scratch.
int exercise_interleaved(const char* artifact) {
    const auto long_text  = text_prompt(3000, 0);
    const auto short_text = short_prompt();
    const auto image_a    = [] { return image_prompt(1024, 1024, "What is visible?"); };
    const auto image_b    = [] { return image_prompt(512, 768, "Describe the colors."); };
    const auto residency  = ninfer::VisionResidency::Overlay;
    const auto prepare    = [&](const ninfer::Engine& engine, int which) {
        switch (which) {
        case 0: return engine.prepare_tokens(long_text);
        case 1: return engine.prepare_tokens(short_text);
        case 2: return engine.prepare(image_a());
        default: return engine.prepare(image_b());
        }
    };
    const char* labels[] = {"long   ", "short  ", "image A", "image B"};

    std::vector<std::vector<ninfer::TokenId>> first(4), serial(4);
    {
        ninfer::Engine engine(engine_options(artifact, residency, 1));
        for (int which = 0; which < 4; ++which) {
            first[which] =
                engine.generate(prepare(engine, which), greedy(1, true)).generated_token_ids;
            serial[which] = engine.generate(prepare(engine, which), greedy(kOutputTokens, true))
                                .generated_token_ids;
        }
    }
    ninfer::Engine engine(engine_options(artifact, residency, 3));
    const ninfer::RuntimeStats before = engine.runtime_stats();
    std::vector<ninfer::GenerationHandle> handles;
    for (const int which : {0, 2, 3, 1}) {
        handles.push_back(engine.submit(prepare(engine, which), greedy(1, true)));
    }
    std::vector<ninfer::GenerationResult> results(4);
    for (std::size_t slot = 0; slot < handles.size(); ++slot) {
        results[std::array{0, 2, 3, 1}[slot]] = handles[slot].wait();
    }
    if (!engine_settles(engine)) {
        std::cerr << "interleaved: requests did not settle\n";
        return 1;
    }
    std::cout << "interleaved overlay: preemptions +"
              << engine.runtime_stats().preemptions - before.preemptions << '\n';
    print_overlay("image A", results[2].timings);
    print_overlay("image B", results[3].timings);
    int failures = 0;
    for (int which = 0; which < 4; ++which) {
        if (results[which].generated_token_ids != first[which]) {
            std::cerr << "interleaved: " << labels[which] << " first token changed\n";
            ++failures;
        }
    }
    for (int which = 0; which < 4; ++which) {
        const auto continued = engine.generate(prepare(engine, which), greedy(kOutputTokens, true));
        if (continued.generated_token_ids != serial[which]) {
            std::cerr << "interleaved: " << labels[which] << " continued from its concurrent prefix ("
                      << continued.reused_prompt_tokens << " reused tokens) differs\n";
            print_ids("serial   ", serial[which]);
            print_ids("continued", continued.generated_token_ids);
            ++failures;
        } else if (continued.reused_prompt_tokens == 0) {
            std::cerr << "interleaved: " << labels[which] << " found no cached prefix\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
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
    const char* repeat_text    = std::getenv("NINFER_OVERLAY_REPEAT");
    const int repeats          = repeat_text != nullptr ? std::max(1, std::atoi(repeat_text)) : 1;
    if (scenario == "batch-variance") { return exercise_batch_variance(artifact, repeats); }
    if (scenario == "reuse-control") { return exercise_reuse_control(artifact); }
    int failures = 0;
    if (scenario == "all" || scenario == "equivalence") {
        failures += exercise_equivalence(artifact, 8192, false);
        failures += exercise_equivalence(artifact, 1024, true);
    }
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (repeats > 1) { std::cout << "repeat " << repeat + 1 << "/" << repeats << '\n'; }
        if (scenario == "all" || scenario == "concurrent") {
            failures += exercise_concurrent(artifact);
        }
        if (scenario == "all" || scenario == "interleaved") {
            failures += exercise_interleaved(artifact);
        }
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
