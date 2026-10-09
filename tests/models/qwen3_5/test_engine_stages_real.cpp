// A model split into pipeline stages must generate exactly what the same model does on one device.
//
// The layers run the same kernels on the same data whichever device holds them, so greedy output is
// an exact oracle: any difference is a wrong-stage weight, a stale control tensor, a KV plane read
// from the wrong copy of the block table, a state shard copied on the wrong stream, a boundary
// transfer that lost or reordered bytes, or a forward pass that raced its own staging. Each
// configuration below is compared token for token with the same configuration on one device.
//
// By default every stage shares device 0, so a pointer into "another stage's" memory still works and
// that cannot see a wrong-device access. What it does cover is everything else about the stage path,
// including the pinned-host protocol (forced, since same-device stages would otherwise take a
// device-to-device shortcut), CUDA graph capture across stages, prefill that spans several chunks,
// the context cache's checkpoints (a continuation and an exact repeat served from them), MTP and
// DFlash2 speculative decoding (DFlash2's feature layers cross the stage boundary), rk4v4 KV, and two
// concurrent requests under KV pressure, where one is paused and replayed.
//
// On a machine with several GPUs, NINFER_TEST_DEVICE_IDS=0,1 puts stage i on the i-th listed device
// (wrapping around), which is the check the aliased run cannot make. Two settings adapt it to a
// pair of cards that cannot hold the model alone:
//   NINFER_TEST_SPLIT_INVARIANCE=1  compare every row with a two-stage run instead of one device
//   NINFER_TEST_MAX_STAGES=2        skip rows with more stages than that
//
// Rows whose feature the artifact lacks (MTP weights, a DFlash2 draft, a Vision tower) are skipped
// with a message. NINFER_TEST_GRAFT=<container.bin> also compares a grafted chat request.

#include "guarded_main.h"
#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Configuration {
    std::string label;
    std::vector<int> devices;
    std::vector<std::uint32_t> stage_layers;
    bool cuda_graph   = true;
    bool force_staged = false;
    // Speculative decoding changes the arithmetic (verification evaluates several columns at once),
    // so a speculative row is compared with the same backend on one device.
    ninfer::SpeculativeBackend backend = ninfer::SpeculativeBackend::None;
    ninfer::KvCacheStorage kv_cache    = ninfer::KvCacheStorage::Int8Group64;
    // Vision tower on rank 0 with this residency; unset leaves vision off.
    std::optional<ninfer::VisionResidency> vision;
};

// Stage i of a configuration sits on the i-th id of NINFER_TEST_DEVICE_IDS, wrapping around; without
// the variable every stage is on device 0.
std::vector<int> stage_devices(const std::vector<int>& requested) {
    const char* ids = std::getenv("NINFER_TEST_DEVICE_IDS");
    if (ids == nullptr || *ids == '\0' || requested.empty()) { return requested; }
    std::vector<int> available;
    std::istringstream list{std::string(ids)};
    for (std::string item; std::getline(list, item, ',');) { available.push_back(std::stoi(item)); }
    if (available.empty()) { return requested; }
    std::vector<int> out;
    for (std::size_t stage = 0; stage < requested.size(); ++stage) {
        out.push_back(available[stage % available.size()]);
    }
    return out;
}

bool flag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void set_force_staged(bool enabled) {
#ifdef _WIN32
    _putenv_s("NINFER_FORCE_STAGED_LINKS", enabled ? "1" : "0");
#else
    setenv("NINFER_FORCE_STAGED_LINKS", enabled ? "1" : "0", 1);
#endif
}

ninfer::EngineOptions engine_options(const char* artifact, const Configuration& configuration) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 2048;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk  = 512;
    options.kv_cache       = configuration.kv_cache;
    options.use_cuda_graph = configuration.cuda_graph;
    if (configuration.vision) {
        options.enable_vision    = true;
        options.vision_residency = *configuration.vision;
    }
    switch (configuration.backend) {
    case ninfer::SpeculativeBackend::Mtp:
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        break;
    case ninfer::SpeculativeBackend::DFlash2:
        options.speculative.backend       = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens  = 7;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        break;
    default:
        break;
    }
    options.devices      = stage_devices(configuration.devices);
    options.stage_layers = configuration.stage_layers;
    if (const char* graft = std::getenv("NINFER_TEST_GRAFT"); graft != nullptr && *graft != '\0') {
        options.grafts.push_back(ninfer::GraftSource{.name = "g", .path = graft});
    }
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

// Token ids inside the vocabulary and not special. 1,300 of them is three prefill chunks of 512
// with a partial last one, so the stage loop runs for every chunk shape and the KV grows across them.
std::vector<ninfer::TokenId> long_prompt() {
    std::vector<ninfer::TokenId> prompt;
    prompt.push_back(248045);
    for (std::uint32_t index = 0; index < 1300; ++index) {
        prompt.push_back(static_cast<ninfer::TokenId>(1000 + (index * 37U) % 5000U));
    }
    return prompt;
}

std::vector<ninfer::TokenId> short_prompt() { return {248045, 846, 198, 5834, 248046, 198}; }

struct Outputs {
    std::vector<ninfer::TokenId> short_run;
    std::vector<ninfer::TokenId> long_run;
    // A continuation of the long prompt, which the context cache serves in part from the first run:
    // its state and KV come back from a checkpoint, on whichever devices hold them.
    std::vector<ninfer::TokenId> continued_run;
    std::uint32_t reused_tokens = 0;
    // The long prompt again: an exact hit on the checkpoint its first run left.
    std::vector<ninfer::TokenId> repeat_run;
    std::uint32_t repeat_reused_tokens = 0;
    // A chat request selecting NINFER_TEST_GRAFT; empty without it.
    std::vector<ninfer::TokenId> graft_run;
    // A chat request with an image, then the same request again served from the prefix cache; empty
    // unless the configuration enables vision.
    std::vector<ninfer::TokenId> image_run;
    std::vector<ninfer::TokenId> image_repeat_run;
    std::uint32_t image_reused_tokens = 0;
    bool image_had_media              = false;
};

ninfer::PromptInput image_request() {
    constexpr int side = 64;
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < side * side; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = ninfer::MediaKind::Image;
    media.media.bytes       = std::move(ppm);
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "inline.ppm";

    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(std::move(media));
    user.parts.push_back(ninfer::MessagePart{.text = "Describe the visible pattern briefly."});

    ninfer::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

Outputs generate(const char* artifact, const Configuration& configuration) {
    set_force_staged(configuration.force_staged);
    std::cout << "running: " << configuration.label << std::endl;
    ninfer::Engine engine(engine_options(artifact, configuration));
    Outputs out;
    out.short_run =
        engine.generate(engine.prepare_tokens(short_prompt()), greedy(24)).generated_token_ids;
    out.long_run =
        engine.generate(engine.prepare_tokens(long_prompt()), greedy(24, true)).generated_token_ids;

    std::vector<ninfer::TokenId> continuation = long_prompt();
    continuation.insert(continuation.end(), out.long_run.begin(), out.long_run.end());
    continuation.push_back(198);
    const ninfer::GenerationResult continued =
        engine.generate(engine.prepare_tokens(continuation), greedy(8, true));
    out.continued_run = continued.generated_token_ids;
    out.reused_tokens = continued.reused_prompt_tokens;

    // The continuation and all but its last generated token: exactly the frontier the continuation's
    // checkpoint holds, so the whole prompt is served from it.
    std::vector<ninfer::TokenId> frontier = continuation;
    if (!out.continued_run.empty()) {
        frontier.insert(frontier.end(), out.continued_run.begin(), out.continued_run.end() - 1);
    }
    const ninfer::GenerationResult repeat =
        engine.generate(engine.prepare_tokens(std::move(frontier)), greedy(8, true));
    out.repeat_run           = repeat.generated_token_ids;
    out.repeat_reused_tokens = repeat.reused_prompt_tokens;

    const char* graft = std::getenv("NINFER_TEST_GRAFT");
    if (graft != nullptr && *graft != '\0') {
        ninfer::PromptInput input;
        input.messages.push_back(ninfer::ChatMessage{
            .role = ninfer::ChatRole::User, .parts = {ninfer::MessagePart{.text = "Who are you?"}}});
        input.options.graft = "g";
        out.graft_run =
            engine.generate(engine.prepare(std::move(input)), greedy(24)).generated_token_ids;
    }
    if (configuration.vision) {
        const ninfer::GenerationResult image =
            engine.generate(engine.prepare(image_request()), greedy(24, true));
        out.image_run       = image.generated_token_ids;
        out.image_had_media = image.prompt.has_media && image.timings.vision_seconds > 0.0;
        const ninfer::GenerationResult image_repeat =
            engine.generate(engine.prepare(image_request()), greedy(24, true));
        out.image_repeat_run    = image_repeat.generated_token_ids;
        out.image_reused_tokens = image_repeat.reused_prompt_tokens;
    }
    return out;
}

std::size_t first_difference(const std::vector<ninfer::TokenId>& want,
                             const std::vector<ninfer::TokenId>& got) {
    std::size_t first = 0;
    while (first < want.size() && first < got.size() && want[first] == got[first]) { ++first; }
    return first;
}

// Compares one row with its reference, printing each part's verdict. Returns true when identical.
bool compare(const std::string& label, const Outputs& expected, const Outputs& outputs,
             bool vision) {
    struct Part {
        const char* name;
        bool ok;
        const std::vector<ninfer::TokenId>* want;
        const std::vector<ninfer::TokenId>* got;
    };
    std::vector<Part> parts = {
        {"short", outputs.short_run == expected.short_run, &expected.short_run, &outputs.short_run},
        {"long", outputs.long_run == expected.long_run, &expected.long_run, &outputs.long_run},
        {"continuation", outputs.continued_run == expected.continued_run &&
                             outputs.reused_tokens == expected.reused_tokens,
         &expected.continued_run, &outputs.continued_run},
        {"exact repeat", outputs.repeat_run == expected.repeat_run &&
                             outputs.repeat_reused_tokens == expected.repeat_reused_tokens,
         &expected.repeat_run, &outputs.repeat_run},
    };
    if (!expected.graft_run.empty()) {
        parts.push_back({"graft", outputs.graft_run == expected.graft_run, &expected.graft_run,
                         &outputs.graft_run});
    }
    if (vision) {
        parts.push_back({"image",
                         outputs.image_run == expected.image_run &&
                             outputs.image_repeat_run == expected.image_repeat_run &&
                             outputs.image_reused_tokens == expected.image_reused_tokens &&
                             outputs.image_had_media == expected.image_had_media,
                         &expected.image_run, &outputs.image_run});
    }
    bool all = true;
    std::cout << label << ":";
    for (const Part& part : parts) {
        std::cout << ' ' << part.name << ' ' << (part.ok ? "identical" : "DIFFERS");
        all = all && part.ok;
    }
    std::cout << " (continuation reused " << outputs.reused_tokens << ", repeat reused "
              << outputs.repeat_reused_tokens << " tokens)\n";
    for (const Part& part : parts) {
        if (!part.ok) {
            std::cerr << "  " << label << ": " << part.name << " first difference at token "
                      << first_difference(*part.want, *part.got) << '\n';
        }
    }
    return all;
}

// Two requests decoding at once on a cache that cannot hold both to completion: the scheduler
// pauses the younger one, lets the older finish, and replays the paused one from its tokens. A split
// model must pause, replay and resume to exactly the single-device tokens.
struct PressureOutputs {
    std::vector<ninfer::TokenId> first;
    std::vector<ninfer::TokenId> second;
    std::uint64_t preemptions     = 0;
    std::uint64_t replay_restores = 0;
};

PressureOutputs pressure(const char* artifact, const std::vector<int>& devices,
                         const std::string& label) {
    set_force_staged(false);
    std::cout << "running: " << label << std::endl;
    constexpr std::uint32_t kCapacity = 512;
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = kCapacity;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk        = 128;
    options.max_concurrency      = 2;
    options.max_pending_requests = 2;
    options.kv_cache             = ninfer::KvCacheStorage::Int8Group64;
    // History is disabled so the paused request recovers by replaying its own tokens.
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    options.devices                           = stage_devices(devices);
    ninfer::Engine engine(options);

    // Main KV pages hold 64 tokens. Both 192-token prompts fit together in six of eight pages and
    // both begin decoding, but their growth to 448 tokens cannot remain resident together.
    std::vector<ninfer::TokenId> first_prompt(192, 198);
    std::vector<ninfer::TokenId> second_prompt(192, 198);
    first_prompt.front()  = 1000;
    second_prompt.front() = 1001;
    auto first  = engine.submit(engine.prepare_tokens(std::move(first_prompt)), greedy(256));
    auto second = engine.submit(engine.prepare_tokens(std::move(second_prompt)), greedy(256));
    const ninfer::GenerationResult first_result  = first.wait();
    const ninfer::GenerationResult second_result = second.wait();
    PressureOutputs out;
    out.first           = first_result.generated_token_ids;
    out.second          = second_result.generated_token_ids;
    out.preemptions     = first_result.scheduling.preemptions + second_result.scheduling.preemptions;
    out.replay_restores =
        first_result.scheduling.replay_restores + second_result.scheduling.replay_restores;
    std::cout << label << ": " << out.first.size() << " + " << out.second.size()
              << " tokens, preemptions " << out.preemptions << ", replay restores "
              << out.replay_restores << '\n';
    return out;
}

// A reference run that needs a feature the artifact may lack. Returns nullopt, after saying why,
// when the reference cannot be built.
std::optional<Outputs> optional_reference(const char* artifact, const Configuration& configuration,
                                          const char* rows) {
    try {
        return generate(artifact, configuration);
    } catch (const std::exception& error) {
        std::cout << rows << " rows skipped: " << error.what() << '\n';
        return std::nullopt;
    }
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    const bool split_invariance = flag("NINFER_TEST_SPLIT_INVARIANCE");
    const char* max_stages_text = std::getenv("NINFER_TEST_MAX_STAGES");
    const std::size_t max_stages =
        max_stages_text != nullptr ? static_cast<std::size_t>(std::atoi(max_stages_text)) : 64;
    // One card cannot always hold the model. Then a two-stage run is the reference, and what is
    // checked is that where the layers are cut does not change the output.
    const std::vector<int> reference_devices =
        split_invariance ? std::vector<int>{0, 0} : std::vector<int>{};
    const auto reference_of = [&](Configuration configuration) {
        configuration.label   = (split_invariance ? "two stages reference, " : "single device, ") +
                              configuration.label;
        configuration.devices = reference_devices;
        configuration.stage_layers.clear();
        configuration.cuda_graph   = true;
        configuration.force_staged = false;
        return configuration;
    };

    const Outputs plain = generate(artifact, reference_of({.label = "plain"}));
    if (plain.short_run.size() != 24 || plain.long_run.size() != 24 ||
        plain.continued_run.size() != 8 || plain.reused_tokens == 0 ||
        plain.repeat_run.size() != 8 ||
        plain.repeat_reused_tokens !=
            long_prompt().size() + plain.long_run.size() + 1 + plain.continued_run.size() - 1) {
        std::cerr << "the reference did not generate its tokens or reuse its checkpoints: short "
                  << plain.short_run.size() << ", long " << plain.long_run.size()
                  << ", continuation " << plain.continued_run.size() << " reusing "
                  << plain.reused_tokens << ", repeat " << plain.repeat_run.size()
                  << " reusing " << plain.repeat_reused_tokens << '\n';
        return 1;
    }
    const char* graft = std::getenv("NINFER_TEST_GRAFT");
    if (graft != nullptr && *graft != '\0' && plain.graft_run.size() != 24) {
        std::cerr << "the reference did not generate the grafted request's tokens\n";
        return 1;
    }
    const auto mtp = optional_reference(
        artifact, reference_of({.label = "MTP", .backend = ninfer::SpeculativeBackend::Mtp}),
        "MTP");
    const auto dflash2 = optional_reference(
        artifact,
        reference_of({.label = "DFlash2", .backend = ninfer::SpeculativeBackend::DFlash2}),
        "DFlash2");
    const auto rk4v4 = optional_reference(
        artifact,
        reference_of({.label    = "rk4v4",
                      .kv_cache = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value}),
        "rk4v4");
    const auto vision = optional_reference(
        artifact, reference_of({.label = "vision resident", .vision = ninfer::VisionResidency::Resident}),
        "vision");
    if (vision && (vision->image_run.size() != 24 || !vision->image_had_media)) {
        std::cerr << "the vision reference did not encode its image\n";
        return 1;
    }
    if (vision) {
        std::cout << "vision reference: the repeated image request reused "
                  << vision->image_reused_tokens << " tokens\n";
    }

    struct Row {
        Configuration configuration;
        const std::optional<Outputs>* reference;
    };
    const std::optional<Outputs> plain_reference = plain;
    const std::vector<Row> rows = {
        {{.label = "two stages, graphs", .devices = {0, 0}}, &plain_reference},
        {{.label = "two stages, eager", .devices = {0, 0}, .cuda_graph = false}, &plain_reference},
        {{.label = "two stages, staged transport", .devices = {0, 0}, .force_staged = true},
         &plain_reference},
        {{.label = "three stages", .devices = {0, 0, 0}}, &plain_reference},
        // Uneven counts: the split must not change what the model computes.
        {{.label = "two stages, uneven layers", .devices = {0, 0}, .stage_layers = {20, 44}},
         &plain_reference},
        {{.label   = "two stages, MTP",
          .devices = {0, 0},
          .backend = ninfer::SpeculativeBackend::Mtp},
         &mtp},
        {{.label      = "three stages, MTP, eager",
          .devices    = {0, 0, 0},
          .cuda_graph = false,
          .backend    = ninfer::SpeculativeBackend::Mtp},
         &mtp},
        {{.label   = "two stages, DFlash2",
          .devices = {0, 0},
          .backend = ninfer::SpeculativeBackend::DFlash2},
         &dflash2},
        {{.label      = "two stages, DFlash2, eager",
          .devices    = {0, 0},
          .cuda_graph = false,
          .backend    = ninfer::SpeculativeBackend::DFlash2},
         &dflash2},
        // Most feature layers past the first stage, crossing two boundaries, through the host.
        {{.label        = "three stages, DFlash2, staged transport, uneven layers",
          .devices      = {0, 0, 0},
          .stage_layers = {10, 20, 34},
          .force_staged = true,
          .backend      = ninfer::SpeculativeBackend::DFlash2},
         &dflash2},
        {{.label    = "two stages, rk4v4",
          .devices  = {0, 0},
          .kv_cache = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value},
         &rk4v4},
        {{.label   = "two stages, vision resident",
          .devices = {0, 0},
          .vision  = ninfer::VisionResidency::Resident},
         &vision},
    };

    int failures = 0;
    for (const Row& row : rows) {
        if (!row.reference->has_value()) { continue; }
        if (row.configuration.devices.size() > max_stages) { continue; }
        const Outputs outputs = generate(artifact, row.configuration);
        if (!compare(row.configuration.label, **row.reference, outputs,
                     row.configuration.vision.has_value())) {
            ++failures;
        }
    }

    // Pause and replay under KV pressure, two lanes decoding together.
    const PressureOutputs pressure_reference = pressure(
        artifact, reference_devices,
        split_invariance ? "two stages reference, pressure" : "single device, pressure");
    if (pressure_reference.first.size() != 256 || pressure_reference.second.size() != 256 ||
        pressure_reference.preemptions == 0 || pressure_reference.replay_restores == 0) {
        std::cerr << "the pressure reference did not pause and replay a request to completion\n";
        ++failures;
    } else if (max_stages >= 2) {
        const PressureOutputs split = pressure(artifact, {0, 0}, "two stages, pressure");
        const bool first_ok  = split.first == pressure_reference.first;
        const bool second_ok = split.second == pressure_reference.second;
        const bool recovered = split.preemptions != 0 && split.replay_restores != 0;
        std::cout << "two stages, pressure: first " << (first_ok ? "identical" : "DIFFERS")
                  << ", second " << (second_ok ? "identical" : "DIFFERS") << ", "
                  << (recovered ? "paused and replayed" : "NO PAUSE/REPLAY") << '\n';
        if (!first_ok) {
            std::cerr << "  pressure first request: first difference at token "
                      << first_difference(pressure_reference.first, split.first) << '\n';
        }
        if (!second_ok) {
            std::cerr << "  pressure second request: first difference at token "
                      << first_difference(pressure_reference.second, split.second) << '\n';
        }
        if (!first_ok || !second_ok || !recovered) { ++failures; }
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
