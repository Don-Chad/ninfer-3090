#include "serve/serve_options.h"

#include <algorithm>
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane. Opt-in.
    if (value == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    // rk4v4: rotated 4-bit Lloyd-Max keys with rk8v4's packed int4 value plane. Opt-in.
    if (value == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [--host H] [--port N] [--api-key KEY] "
           "[--model-id ID] [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--prefill-chunk N] [--max-prefill-lanes N] [--prefill-max-skip N] "
           "[--decode-rounds-per-prefill N] [--log-stats-interval-ms N] [--device N] "
           "[--context-cost-presets FILE] "
           "[--max-request-mib N] [--media-cache-mib N] [--media-live-mib N] "
           "[--media-preprocess-threads N] "
           "[--device-state-slots N] [--host-state-slots N] [--host-kv-mib N] "
           "[--max-private-continuations N] [--max-shared-prefixes N] [--auto-host-cache [--host-cache-reserve-mib N] [--host-cache-max-mib N] [--host-cache-percent N]] "
           "[--max-long-anchors-per-continuation N] [--auto-long-anchors N] "
           "[--progress-anchor-tokens N] "
           "[--max-cache-markers-per-request N] "
           "[--request-log-jsonl FILE] [--slot-save-path DIR] [--auto-save-evicted] [--no-exit-on-engine-failure] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype bf16|int8|fp8|rk8v4|rk4v4|nvfp4|k8v4] "
           "[--spec mtp|dflash|dflash2 --draft-tokens N] "
           "[--default-max-tokens N] [--default-thinking-budget N] "
           "[--vision] [--vision-residency resident|overlay] [--vision-max-merged N] "
           "[--no-cuda-graph] [--no-prefix-reuse] [--auto-prefix-grid] [--devices N,M,...] [--stage-layers A,B,...] "
           "[--chat-template FILE] "
           "[--lm-head-draft] [--lm-head-q4|--lm-head-q6] [--embedding-q4|--embedding-q6] [--mtp-experts-q4] "
           "[--gdn-state-fp16] "
           "[--mlp-a8-decode] [--no-prefill-a8] "
           "[--prefill-cublas [--no-prefill-cublas-projections]] [--lookup-ngram N] "
           "[--no-thinking] [--preserve-thinking] [--graft NAME=PATH]... [--default-graft NAME] "
           "[--reasoning-effort minimal|low|medium|high|xhigh|max] [--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --default-max-tokens fixes the output budget of requests that omit a limit; "
           "unset, such a request gets the largest budget that still lets every lane be admitted "
           "at once (the remaining context with one lane)\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --model-id overrides the artifact metadata.name reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --vision-residency overlay keeps the Vision tower in host memory and borrows "
           "device memory per image\n"
           "       --vision-max-merged bounds the merged tokens of one media item (default 16384); "
           "larger media is downscaled\n"
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom\n"
           "       --max-prefill-lanes N (default 1, at most --max-concurrency) lets that many requests "
           "prefill at once: each prefill unit goes to the lane with the shortest remaining prompt "
           "suffix, so a short or prefix-cached prompt is not stuck behind a very long one; the KV "
           "capacity must hold the long prompt and the short ones together. A lane passed over "
           "--prefill-max-skip units (default 8) is served before any shorter one\n"
           "       --decode-rounds-per-prefill N runs that many decode rounds after each prefill unit "
           "while other requests are generating (default 0 = --prefill-chunk / 64, so 16 at the "
           "default chunk of 1024); 1 alternates strictly, which leaves a decoding stream a few "
           "tokens per second while a long prompt prefills; a larger N keeps streams responsive and "
           "makes the prefill finish later\n"
           "       --prefill-cublas hands wide prefill GEMMs to cuBLAS: a large prefill speedup for a "
           "small perplexity cost (docs/performance.md), off by default, and it wants a larger "
           "--prefill-chunk to pay; --no-prefill-cublas-projections keeps the attention and GDN "
           "input projections off that route\n"
           "       --lookup-ngram N adds context-lookup drafting alongside --spec: the last N tokens "
           "are matched against the sequence so far and what followed is proposed; it is exact, and "
           "0 (the default) disables it\n"
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       --auto-prefix-grid offers shared candidates on a token grid so unrelated "
           "callers whose prompts start alike share a cached prefix without any client hint; a grid "
           "frontier is only published once two callers have both asked for it\n"
           "       --auto-long-anchors N proposes a private long anchor at each of the last N "
           "message boundaries of every prompt, so a client that rewrites recent history "
           "restores at the anchor below the edit instead of re-prefilling from zero; it "
           "defaults to and is clamped to --max-long-anchors-per-continuation (raise that and "
           "--host-state-slots for deeper edits); 0 disables\n"
           "       --progress-anchor-tokens N proposes a private long anchor at every multiple of N "
           "tokens of a prompt (default 16384, minimum 256) and keeps the anchors a cancelled prefill already "
           "holds, so a client that times out on a very long prompt and retries resumes from the "
           "last anchor instead of prefilling from zero; it shares the "
           "--max-long-anchors-per-continuation budget with --auto-long-anchors; 0 disables\n"
           "       --slot-save-path DIR enables POST /slots/{id}?action=save|restore|erase, which "
           "writes a retained session to a file in DIR or restores one from it\n"
           "       --auto-save-evicted writes a retained session back to the slot file it was last "
           "saved to or restored from before an involuntary eviction destroys it (requires "
           "--slot-save-path; erase never saves)\n"
           "       --no-exit-on-engine-failure keeps the process alive when the Engine latches "
           "unavailable after repeated worker failures; by default it logs FATAL and exits with "
           "status 3 after a short grace period so a supervisor can restart it\n"
           "       context cache defaults: device-state=max-concurrency, private=2x concurrency, "
           "shared=max(max-concurrency,4), anchors=2; Host state=8 slots, Host KV=8192 MiB\n"
           "       --device-state-slots is extra checkpoint capacity beyond active lanes; "
           "--host-kv-mib uses MiB\n"
           "       --auto-host-cache sizes Host state slots, Host KV and both catalogs from the "
           "host memory free after the model loads (all but --host-cache-reserve-mib, default 3072, "
           "an eighth of it for state slots); it replaces the four explicit Host capacity "
           "options\n"
           "       --host-cache-max-mib N caps what --auto-host-cache pins, for machines whose "
           "memory other tenants share (default: no cap)\n"
           "       --host-cache-percent N caps it at N% (1-100) of the machine's total memory (or "
           "its container's limit) whatever is free at startup; the smallest of this, the cap and "
           "the free memory less the reserve applies. With vision on, the reserve also covers "
           "--media-cache-mib and --media-live-mib\n"
           "       --default-thinking-budget caps model-origin thinking for enabled requests; "
           "control tokens count toward the request output limit\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       --graft NAME=PATH loads a phantom-kv prefill graft (a safetensors container with a "
           ".json sidecar beside it); a request selecting it with \"graft\": \"NAME\" runs as if the "
           "graft's hidden turn preceded its own messages. Repeatable\n"
           "       --default-graft NAME applies a loaded graft to every request that states none; "
           "a request opts out with \"graft\": \"\"\n"
           "       --reasoning-effort is the effort of thinking-enabled requests that state none; "
           "request values override it; like request values, minimal runs as low and high/max as "
           "xhigh\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n";
}

// "1,2,3" selects the ordered devices the model's pipeline stages run on; the first also holds the
// embedding, head and round state. One entry is accepted and is equivalent to --device. The engine
// validates matching compute capability at startup; this only parses the shape.
constexpr std::size_t kMaximumDevices = 8;
std::vector<int> parse_device_list(std::string_view value) {
    std::vector<int> devices;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = value.find(',', start);
        const std::string_view piece =
            value.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                                : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--devices entries must not be empty"); }
        const std::string entry(piece);
        devices.push_back(parse_nonnegative_int(entry.c_str(), "devices"));
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
        const int count = parse_nonnegative_int(entry.c_str(), "stage-layers");
        if (count == 0) { throw std::invalid_argument("--stage-layers counts must be positive"); }
        counts.push_back(static_cast<std::uint32_t>(count));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return counts;
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool kv_capacity_explicit        = false;
    bool device_explicit             = false;
    bool context_capacity_explicit   = false;
    bool host_sizing_explicit        = false;
    bool host_reserve_explicit       = false;
    bool host_max_explicit           = false;
    bool host_percent_explicit       = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--max-prefill-lanes") {
            options.max_prefill_lanes = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-prefill-lanes"), "max-prefill-lanes"));
        } else if (arg == "--prefill-max-skip") {
            options.prefill_max_skip = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-max-skip"), "prefill-max-skip"));
        } else if (arg == "--decode-rounds-per-prefill") {
            options.decode_rounds_per_prefill = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--decode-rounds-per-prefill"), "decode-rounds-per-prefill"));
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-state-slots") {
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
            host_sizing_explicit      = true;
        } else if (arg == "--host-kv-mib") {
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
            host_sizing_explicit                         = true;
        } else if (arg == "--max-private-continuations") {
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
            host_sizing_explicit      = true;
        } else if (arg == "--max-shared-prefixes") {
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
            host_sizing_explicit      = true;
        } else if (arg == "--auto-host-cache") {
            options.context_cache.auto_host_cache = true;
            context_capacity_explicit             = true;
        } else if (arg == "--host-cache-reserve-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--host-cache-reserve-mib"), "host-cache-reserve-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-cache-reserve-mib is out of range");
            }
            options.context_cache.host_cache_reserve_bytes = static_cast<std::size_t>(mib << 20);
            host_reserve_explicit                          = true;
        } else if (arg == "--host-cache-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--host-cache-max-mib"), "host-cache-max-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-cache-max-mib is out of range");
            }
            options.context_cache.host_cache_max_bytes = static_cast<std::size_t>(mib << 20);
            host_max_explicit                          = true;
        } else if (arg == "--host-cache-percent") {
            const std::uint64_t percent =
                parse_u64(require_value("--host-cache-percent"), "host-cache-percent");
            if (percent < 1 || percent > 100) {
                throw std::invalid_argument("--host-cache-percent must be in [1,100]");
            }
            options.context_cache.host_cache_percent = static_cast<std::uint32_t>(percent);
            host_percent_explicit                    = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
        } else if (arg == "--auto-long-anchors") {
            options.auto_long_anchors = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--auto-long-anchors"), "auto-long-anchors"));
        } else if (arg == "--progress-anchor-tokens") {
            options.progress_anchor_tokens = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--progress-anchor-tokens"), "progress-anchor-tokens"));
            if (*options.progress_anchor_tokens != 0 &&
                *options.progress_anchor_tokens < kMinimumProgressAnchorTokens) {
                throw std::invalid_argument("--progress-anchor-tokens must be 0 or at least " +
                                            std::to_string(kMinimumProgressAnchorTokens));
            }
        } else if (arg == "--max-cache-markers-per-request") {
            options.context_cache.max_cache_markers_per_request = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-cache-markers-per-request"),
                                      "max-cache-markers-per-request"));
            context_capacity_explicit = true;
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
            device_explicit = true;
        } else if (arg == "--devices") {
            options.devices = parse_device_list(require_value("--devices"));
        } else if (arg == "--stage-layers") {
            options.stage_layers = parse_stage_layers(require_value("--stage-layers"));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-residency") {
            const std::string_view mode = require_value("--vision-residency");
            if (mode == "resident") {
                options.vision_residency = VisionResidency::Resident;
            } else if (mode == "overlay") {
                options.vision_residency = VisionResidency::Overlay;
            } else {
                throw std::invalid_argument("--vision-residency must be resident or overlay");
            }
        } else if (arg == "--vision-max-merged") {
            const std::uint64_t merged =
                parse_u64(require_value("--vision-max-merged"), "vision-max-merged");
            if (merged < 64 || merged > 16384) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 16384]");
            }
            options.vision_max_merged_tokens = static_cast<std::uint32_t>(merged);
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--slot-save-path") {
            options.slot_save_path = require_value("--slot-save-path");
            if (options.slot_save_path.empty()) {
                throw std::invalid_argument("--slot-save-path must not be empty");
            }
        } else if (arg == "--auto-save-evicted") {
            options.auto_save_evicted = true;
        } else if (arg == "--no-exit-on-engine-failure") {
            options.exit_on_engine_failure = false;
        } else if (arg == "--auto-prefix-grid") {
            options.auto_prefix_grid = true;
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
        } else if (arg == "--lookup-ngram") {
            options.speculative.lookup_ngram = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--lookup-ngram"), "lookup-ngram"));
        } else if (arg == "--prefill-cublas") {
            options.prefill_cublas = true;
        } else if (arg == "--no-prefill-cublas-projections") {
            options.prefill_cublas_projections = false;
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--graft") {
            const std::string_view spec = require_value("--graft");
            const std::size_t equals    = spec.find('=');
            if (equals == 0 || equals == std::string_view::npos || equals + 1 == spec.size()) {
                throw std::invalid_argument("--graft must be NAME=PATH");
            }
            options.grafts.push_back(GraftSource{.name = std::string(spec.substr(0, equals)),
                                                 .path = std::string(spec.substr(equals + 1))});
        } else if (arg == "--default-graft") {
            options.default_graft = require_value("--default-graft");
            if (options.default_graft.empty()) {
                throw std::invalid_argument("--default-graft needs a graft name");
            }
        } else if (arg == "--reasoning-effort") {
            const std::string value = require_value("--reasoning-effort");
            const std::optional<RequestedReasoningEffort> effort =
                parse_requested_reasoning_effort(value);
            if (!effort) { throw std::invalid_argument("invalid reasoning-effort: " + value); }
            if (*effort == RequestedReasoningEffort::None) {
                throw std::invalid_argument(
                    "--reasoning-effort none is not a default effort; use --no-thinking");
            }
            options.default_reasoning_effort = *effort;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!options.default_graft.empty() &&
        std::none_of(options.grafts.begin(), options.grafts.end(), [&](const GraftSource& source) {
            return source.name == options.default_graft;
        })) {
        throw std::invalid_argument("--default-graft '" + options.default_graft +
                                    "' does not name a --graft");
    }
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (host_reserve_explicit && !options.context_cache.auto_host_cache) {
        throw std::invalid_argument("--host-cache-reserve-mib requires --auto-host-cache");
    }
    if (host_max_explicit && !options.context_cache.auto_host_cache) {
        throw std::invalid_argument("--host-cache-max-mib requires --auto-host-cache");
    }
    if (host_percent_explicit && !options.context_cache.auto_host_cache) {
        throw std::invalid_argument("--host-cache-percent requires --auto-host-cache");
    }
    if (options.context_cache.auto_host_cache && host_sizing_explicit) {
        throw std::invalid_argument(
            "--auto-host-cache sizes --host-state-slots, --host-kv-mib, "
            "--max-private-continuations and --max-shared-prefixes; do not set them as well");
    }
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        if (options.auto_prefix_grid) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --auto-prefix-grid");
        }
        if (options.auto_long_anchors) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --auto-long-anchors");
        }
        if (options.progress_anchor_tokens) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --progress-anchor-tokens");
        }
        if (!options.slot_save_path.empty()) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --slot-save-path");
        }
        options.context_cache.enabled                = false;
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    }
    if (!options.devices.empty() && device_explicit) {
        throw std::invalid_argument("--device and --devices are mutually exclusive");
    }
    if (!options.stage_layers.empty() && options.devices.size() < 2) {
        throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
    }
    if (options.auto_save_evicted && options.slot_save_path.empty()) {
        throw std::invalid_argument("--auto-save-evicted requires --slot-save-path");
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (options.max_prefill_lanes == 0 || options.max_prefill_lanes > options.max_concurrency) {
        throw std::invalid_argument("--max-prefill-lanes must be in [1,--max-concurrency]");
    }
    if (options.prefill_max_skip == 0) {
        throw std::invalid_argument("--prefill-max-skip must be positive");
    }
    if (options.decode_rounds_per_prefill > 4096) {
        throw std::invalid_argument("--decode-rounds-per-prefill must be in [0,4096]");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.vision_residency == VisionResidency::Overlay && !options.enable_vision) {
        throw std::invalid_argument("--vision-residency overlay requires --vision");
    }
    if (options.default_max_tokens && *options.default_max_tokens <= 0) {
        throw std::invalid_argument("--default-max-tokens must be positive");
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

std::uint32_t resolve_automatic_private_anchors(const ServeOptions& options,
                                                const ContextCacheOptions& resolved) {
    if (!resolved.enabled || !options.allow_prefix_reuse) { return 0; }
    const std::uint32_t cap = resolved.max_long_anchors_per_continuation.value_or(0U);
    return std::min(options.auto_long_anchors.value_or(cap), cap);
}

std::uint32_t resolve_progress_anchor_stride(const ServeOptions& options,
                                             const ContextCacheOptions& resolved) {
    if (!resolved.enabled || !options.allow_prefix_reuse ||
        resolved.max_long_anchors_per_continuation.value_or(0U) == 0) {
        return 0;
    }
    return options.progress_anchor_tokens.value_or(kDefaultProgressAnchorTokens);
}

} // namespace ninfer::serve
