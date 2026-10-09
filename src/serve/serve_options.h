#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::filesystem::path chat_template_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    // Admission waits behind the active set, so the deadline has to cover the generation
    // time of the requests ahead in the FIFO. One 1K-token response already runs past 15 s
    // at C1 on an RTX 3090, and a 6.5K-token one past 100 s; a 30 s deadline expired those
    // callers before they were ever admitted.
    std::uint32_t pending_timeout_ms   = 600000;
    std::uint32_t prefill_chunk        = 1024;
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    // Ordered CUDA devices, one pipeline stage each: primary first, also holding the embedding,
    // head and round state. Empty keeps the single-device route selected by `device`. Mutually
    // exclusive with --device.
    std::vector<int> devices;
    // Layers per stage, one count per entry of `devices`. Empty lets the engine choose.
    std::vector<std::uint32_t> stage_layers;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    ContextCacheOptions context_cache;
    bool enable_vision      = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    std::uint32_t vision_max_merged_tokens = 16384;
    bool use_cuda_graph     = true;
    bool lm_head_q4         = false;
    bool lm_head_q6         = false;
    bool embedding_q4       = false;
    bool embedding_q6       = false;
    bool mtp_experts_q4     = false;
    bool gdn_state_fp16     = false;
    bool mlp_a8_decode      = false;
    bool prefill_a8         = true;
    bool prefill_cublas     = false;
    bool prefill_cublas_projections = true;
    bool allow_prefix_reuse = true;
    // --context-store DIR: keep retained sessions on disk so a restart or crash does not lose the
    // context cache. Empty disables it.
    std::filesystem::path context_store_path;
    // --context-store-max-gib N: unset lets the Engine take half the volume's free space.
    std::optional<std::uint64_t> context_store_max_gib;
    std::uint32_t context_store_ttl_hours      = 24 * 7;
    std::uint32_t context_store_idle_seconds   = 30;
    std::uint32_t context_store_restore_seconds = 120;
    std::uint32_t context_store_flush_seconds   = 60;
    // --context-store-s3-endpoint URL / --context-store-s3-bucket NAME: keep a copy of the store in
    // an S3-compatible bucket (credentials come from the environment, never the command line).
    std::string context_store_s3_endpoint;
    std::string context_store_s3_bucket;
    std::string context_store_s3_prefix; // "" or ending in '/'
    std::string context_store_s3_region = "us-east-1";
    // Exit non-zero shortly after the Engine latches unavailable after a worker failure, so a
    // supervisor restarts the process instead of leaving it holding VRAM and answering 503.
    bool exit_on_engine_failure = true;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    // --graft NAME=PATH, repeatable: phantom-kv grafts a request may select with "graft": NAME.
    std::vector<GraftSource> grafts;
    // --default-graft NAME: graft applied to a request that states none ("graft": "" opts out).
    // Empty means no default; otherwise it names an entry of `grafts`.
    std::string default_graft;
    std::optional<std::uint32_t> default_thinking_budget;
    // --reasoning-loop off|stop|conclude: the reasoning-loop guard for every thinking request,
    // constrained ones included (off by default).
    ninfer::ReasoningLoopAction reasoning_loop = ninfer::ReasoningLoopAction::Off;
    // Output limit for a request that omits one. Unset means the request's remaining context: see
    // request_limits().
    std::optional<int> default_max_tokens;
    // Upper bound on the output budget of every request, stated or derived; see docs/serving.md.
    std::optional<int> max_output_tokens;
    // Reasoning effort for a thinking-enabled request that states none. Never None: disabling
    // thinking by default is --no-thinking.
    std::optional<RequestedReasoningEffort> default_reasoning_effort;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

// Parse-time limits. A request that omits max_tokens / max_completion_tokens / max_output_tokens
// gets --default-max-tokens when set; otherwise GenerationService gives it the remaining context
// once its prompt is prepared. The Engine reserves KV per execution unit and pauses a younger
// request when concurrent growth exhausts the pool, so a large derived budget no longer holds pages
// it never writes.
[[nodiscard]] inline RequestLimits request_limits(const ServeOptions& options) noexcept {
    return RequestLimits{.default_max_tokens = options.default_max_tokens,
                         .max_context        = static_cast<int>(options.max_context)};
}

// The output budget actually submitted: the request's own (stated or derived) budget, bounded by
// --max-output-tokens when set.
[[nodiscard]] inline std::uint32_t
bounded_output_budget(std::uint32_t requested, const std::optional<int>& max_output_tokens) noexcept {
    return max_output_tokens ? std::min(requested, static_cast<std::uint32_t>(*max_output_tokens))
                             : requested;
}

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
