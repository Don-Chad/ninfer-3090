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
    // Offer shared-prefix candidates on a content-independent token grid so unrelated callers whose
    // prompts merely start alike converge on the same frontier. Off by default: it adds host-side
    // candidate work to every request and only pays for itself on a multi-tenant preamble.
    bool auto_prefix_grid = false;
    // --auto-long-anchors N: propose a private long anchor at each of the last N message
    // boundaries of every prompt. Unset resolves to the retained-anchor cap once the Engine has
    // normalized it; 0 disables. See resolve_automatic_private_anchors.
    std::optional<std::uint32_t> auto_long_anchors;
    // Directory for /slots session files; empty disables slot save/restore.
    std::filesystem::path slot_save_path;
    // Spill an involuntarily evicted session back to the slot file it was last saved to or
    // restored from. Requires slot_save_path.
    bool auto_save_evicted = false;
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
    // Output limit for a request that omits one. Unset means the Engine's concurrent lane budget:
    // see request_limits().
    std::optional<int> default_max_tokens;
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
// gets --default-max-tokens when set; otherwise GenerationService asks the Engine for the largest
// budget that still lets every configured lane be admitted at once (the remaining context with one
// lane).
[[nodiscard]] inline RequestLimits request_limits(const ServeOptions& options) noexcept {
    return RequestLimits{.default_max_tokens = options.default_max_tokens,
                         .max_context        = static_cast<int>(options.max_context)};
}

ServeOptions parse_serve_options(int argc, char** argv);
// The per-request ContextCacheHints::automatic_private_anchors for this server: the explicit
// --auto-long-anchors when given, else the resolved anchor cap, never more than that cap. Zero
// when the context cache is disabled. `resolved` must be the Engine's normalized options, whose
// optional capacities are filled in.
std::uint32_t resolve_automatic_private_anchors(const ServeOptions& options,
                                                const ContextCacheOptions& resolved);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
