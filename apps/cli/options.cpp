#include "options.h"
#include "product/reasoning_loop_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ninfer::cli {
namespace {

std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const char* text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

// Same shape as serve's --devices: the ordered devices the model's pipeline stages run on.
// Repeating an id puts several stages on one card, which exercises the split path without a
// second GPU. Existence is checked at engine startup; this only parses the shape.
constexpr std::size_t kMaximumDevices = 8;
std::vector<int> parse_device_list(std::string_view text) {
    std::vector<int> devices;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece =
            text.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                               : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--devices entries must not be empty"); }
        const std::string entry(piece);
        const std::uint64_t raw = parse_u64(entry.c_str(), "devices");
        if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("invalid devices: " + entry);
        }
        devices.push_back(static_cast<int>(raw));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    if (devices.empty() || devices.size() > kMaximumDevices) {
        throw std::invalid_argument("--devices takes between one and " +
                                    std::to_string(kMaximumDevices) + " CUDA device ids");
    }
    // Repeated ids are deliberately permitted: they put several stages on one card, which saves no
    // memory but exercises the whole split path on a single-GPU machine.
    return devices;
}

// "30,34": the layers each pipeline stage owns, one count per device in --devices. Whether they add
// up to the model's layers is checked when the model loads; this only parses the shape.
std::vector<std::uint32_t> parse_stage_layers(std::string_view text) {
    std::vector<std::uint32_t> counts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece =
            text.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                               : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--stage-layers entries must not be empty"); }
        const std::string entry(piece);
        counts.push_back(parse_u32(entry.c_str(), "stage-layers", false));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return counts;
}

int parse_device(const char* text) {
    const std::uint64_t value = parse_u64(text, "device");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid device: ") + text);
    }
    return static_cast<int>(value);
}

float parse_float(const char* text, std::string_view label, float minimum, float maximum) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<float>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane. Opt-in.
    if (text == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    // rk4v4: rotated 4-bit Lloyd-Max keys with rk8v4's packed int4 value plane. Opt-in.
    if (text == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

ReasoningEffort parse_reasoning_effort(std::string_view text) {
    if (text == "none") { return ReasoningEffort::None; }
    if (text == "minimal") { return ReasoningEffort::Minimal; }
    if (text == "high") { return ReasoningEffort::High; }
    if (text == "max") { return ReasoningEffort::Max; }
    if (text == "low") { return ReasoningEffort::Low; }
    if (text == "medium") { return ReasoningEffort::Medium; }
    if (text == "xhigh") { return ReasoningEffort::XHigh; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--device N] [--devices N,M,...] [--stage-layers A,B,...]\n"
           "       [--kv-dtype bf16|int8|fp8|rk8v4|rk4v4|nvfp4|k8v4] [--spec mtp|dflash|dflash2 --draft-tokens N]\n"
           "       [--lm-head-draft] [--lm-head-q4|--lm-head-q6] [--embedding-q4|--embedding-q6] [--mtp-experts-q4]\n"
           "       [--gdn-state-fp16] [--mlp-a8-decode] [--no-prefill-a8]\n"
           "       [--prefill-cublas [--no-prefill-cublas-projections]]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--chat-template FILE]\n"
           "       [--grammar-file FILE | --json-object | --json-schema-file FILE |\n"
           "        --regex PATTERN | --choice TEXT ...]\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-loop off|stop|conclude]\n"
           "       [--reasoning-effort none|minimal|low|medium|high|xhigh|max] [--vision]\n"
           "       [--vision-residency resident|overlay] [--vision-max-merged N]\n"
           "       [--no-cuda-graph]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "\n"
           "Streams answer content to stdout and reasoning plus diagnostics to stderr.\n"
           "Structured message content accepts text, image/image_url, and video/video_url parts;\n"
           "media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n"
           "--vision enables image/video input and loads the fixed Vision GPU allocations.\n"
           "--vision-residency overlay keeps the Vision tower in host memory and borrows device "
           "memory per image; --vision-max-merged bounds one item's merged tokens (default 16384).\n"
           "--thinking-budget caps model-origin thinking tokens; inserted control tokens count "
           "toward --max-new.\n"
           "--reasoning-loop off (default) | stop | conclude: when the open thinking keeps repeating "
           "whole passages (checked every 512 thinking tokens), end the reply (stop) or close the "
           "thinking and let the model answer (conclude).\n"
           "--devices N,M,... splits the model's layers into one pipeline stage per GPU, each owning "
           "its layers' weights, KV cache and state; the first GPU also holds the embedding, head "
           "and round state. --stage-layers A,B,... sets the layers per stage (default: chosen "
           "from each GPU's free memory). Multi-GPU execution is Linux only; repeat one id "
           "(--devices 0,0) to exercise the path on a single GPU.\n"
           "--lm-head-q4, --lm-head-q6, --embedding-q4, --embedding-q6, --mtp-experts-q4, "
           "--gdn-state-fp16 and --mlp-a8-decode are "
           "speed- or memory-for-quality trades, off by "
           "default; see "
           "docs/maintainer/quality-trade-experiments.md for the measured cost of each.\n"
           "--no-prefill-a8 returns full prefill tiles to their A16 routes, which is how the "
           "integer routes are measured on a whole request.\n"
           "--prefill-cublas hands wide prefill GEMMs to cuBLAS: a large prefill speedup for a small "
           "perplexity cost (docs/performance.md), off by default, and it wants a larger "
           "--prefill-chunk to pay. --no-prefill-cublas-projections keeps the attention and GDN "
           "input projections off that route.\n"
           "--kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom.\n"
           "Sampling defaults come from the loaded model and thinking mode; flags override "
           "individual fields.\n"
           "--version prints the build version and exits.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument(".ninfer model path is required"); }
    options.artifact_path     = argv[1];
    bool kv_capacity_explicit = false;
    bool device_explicit      = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--chat-template") {
            options.chat_template_path = value(arg);
        } else if (arg == "--grammar-file") {
            options.grammar_path = value(arg);
        } else if (arg == "--json-object") {
            options.json_object = true;
        } else if (arg == "--json-schema-file") {
            options.json_schema_path = value(arg);
        } else if (arg == "--regex") {
            options.regex = value(arg);
        } else if (arg == "--choice") {
            options.choices.emplace_back(value(arg));
        } else if (arg == "--messages") {
            options.messages_path = value(arg);
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device  = parse_device(value(arg));
            device_explicit = true;
        } else if (arg == "--devices") {
            options.devices = parse_device_list(value(arg));
        } else if (arg == "--stage-layers") {
            options.stage_layers = parse_stage_layers(value(arg));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--lm-head-q4") {
            options.lm_head_q4 = true;
        } else if (arg == "--lm-head-q6") {
            options.lm_head_q6 = true;
        } else if (arg == "--embedding-q4") {
            options.embedding_q4 = true;
        } else if (arg == "--embedding-q6") {
            options.embedding_q6 = true;
        } else if (arg == "--mtp-experts-q4") {
            options.mtp_experts_q4 = true;
        } else if (arg == "--gdn-state-fp16") {
            options.gdn_state_fp16 = true;
        } else if (arg == "--mlp-a8-decode") {
            options.mlp_a8_decode = true;
        } else if (arg == "--no-prefill-a8") {
            options.prefill_a8 = false;
        } else if (arg == "--prefill-cublas") {
            options.prefill_cublas = true;
        } else if (arg == "--no-prefill-cublas-projections") {
            options.prefill_cublas_projections = false;
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value(arg), "thinking-budget");
        } else if (arg == "--reasoning-loop") {
            options.reasoning_loop = ninfer::product::parse_reasoning_loop_action(value(arg));
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-residency") {
            const std::string_view mode = value(arg);
            if (mode == "resident") {
                options.vision_residency = VisionResidency::Resident;
            } else if (mode == "overlay") {
                options.vision_residency = VisionResidency::Overlay;
            } else {
                throw std::invalid_argument("--vision-residency must be resident or overlay");
            }
        } else if (arg == "--vision-max-merged") {
            options.vision_max_merged_tokens = parse_u32(value(arg), "vision-max-merged");
            if (options.vision_max_merged_tokens < 64 || options.vision_max_merged_tokens > 16384) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 16384]");
            }
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_float(value(arg), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_float(value(arg), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            const std::uint32_t top_k = parse_u32(value(arg), "top-k", true);
            if (top_k > 20) { throw std::invalid_argument("--top-k must be in [0,20]"); }
            options.sampling.top_k = static_cast<std::int32_t>(top_k);
        } else if (arg == "--min-p") {
            options.sampling.min_p = parse_float(value(arg), "min-p", 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            options.sampling.presence_penalty =
                parse_float(value(arg), "presence-penalty", -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            options.sampling.frequency_penalty =
                parse_float(value(arg), "frequency-penalty", -2.0F, 2.0F);
        } else if (arg == "--seed") {
            options.sampling.seed = parse_u64(value(arg), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(value(arg));
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (int(!options.grammar_path.empty()) + int(!options.json_schema_path.empty()) +
            int(options.json_object) + int(options.regex.has_value()) +
            int(!options.choices.empty()) >
        1)
        throw std::invalid_argument("select only one of --grammar-file, --json-object, "
                                    "--json-schema-file, --regex or --choice");
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (!options.devices.empty() && device_explicit) {
        throw std::invalid_argument("--device and --devices are mutually exclusive");
    }
    if (!options.stage_layers.empty() && options.devices.size() < 2) {
        throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    product::apply_speculative_defaults(options.speculative);
    product::validate_speculative_cli_options(options.speculative);
    if (options.vision_residency == VisionResidency::Overlay && !options.enable_vision) {
        throw std::invalid_argument("--vision-residency overlay requires --vision");
    }
    if (options.enable_thinking == false && options.reasoning_effort &&
        *options.reasoning_effort != ReasoningEffort::None) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (options.reasoning_effort == ReasoningEffort::None) options.enable_thinking = false;
    if (options.enable_thinking == false && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.enable_thinking == false &&
        options.reasoning_loop != ninfer::ReasoningLoopAction::Off) {
        throw std::invalid_argument("--reasoning-loop cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }
    return options;
}

} // namespace ninfer::cli
