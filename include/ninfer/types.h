#pragma once

#include "ninfer/object_store.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes  = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes   = 2ULL << 30;
// Host memory `auto_host_cache` leaves unpinned for everything that runs after it is sized. The
// 27B server measured ~2 GiB of non-pinned growth after sizing (CUDA/cuBLAS host state, tokenizer,
// HTTP and Program setup; two requests added ~30 MB); this keeps a margin over that.
inline constexpr std::size_t kDefaultHostCacheReserveBytes = 3ULL << 30;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane over a group-32
    // scale. Ported onto the kv_cache_append Op that owns KV quantization, and opt-in through
    // --kv-dtype rk8v4. See docs/rtx-3090-windows.md.
    RotatedInt8KeyInt4ValueGroup64,
    // NVFP4 e2m1, group-16 scale, both K and V planes packed two codes per byte.
    Nvfp4Group16,
    // K8V4: FP8 E4M3 key plane (same coding as Fp8E4M3Row256) paired with an NVFP4 value plane.
    Fp8KeyNvfp4Value,
    // rk4v4 (this fork only): rotated keys as 4-bit indices into a Lloyd-Max codebook with a
    // group-64 FP16 scale, paired with rk8v4's packed signed int4 group-32 value plane. Opt-in
    // through --kv-dtype rk4v4.
    RotatedLloyd4KeyInt4Value,
};

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: 1..15 for MTP, DFlash and DFlash2 (query width K+1).
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostContextPin,
    CudaGraphPrepare,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

// One session written to the context store, reported from the writer thread.
struct ContextStoreWriteEvent {
    std::string id;
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    // Empty on success.
    std::string error;
};

// A durable store for retained sessions, so the context cache survives a restart or a crash.
// Disabled unless a directory is given; requires the context cache and a nonzero Host context
// budget. A retained session is written there when it has been idle for `idle_persist` and (all of
// them) when the Engine shuts down; at start-up the most recently used sessions are restored into
// the Host tier, and a request whose session was evicted reads it back. The store holds only changed
// pieces of each session (see ContextStore), so keeping a long conversation current is cheap.
struct ContextStoreOptions {
    std::filesystem::path directory;
    // Bytes the store may hold; the least recently used sessions are removed beyond it. 0 chooses
    // half of the free space on the volume when the Engine starts.
    std::uint64_t max_bytes = 0;
    // A session unused for this long is removed. Zero keeps sessions until space is needed.
    std::chrono::seconds ttl{std::chrono::hours(24 * 7)};
    // A retained session unused for this long, and not yet written in its current state, is written
    // in the background so a crash loses at most this much of a conversation. Zero disables it
    // (sessions are then written only at shutdown).
    std::chrono::seconds idle_persist{30};
    // Upper bound on the time spent restoring sessions into the cache at start-up, and on how long
    // a request waits for its stored session to be read back.
    std::chrono::seconds restore_budget{120};
    // Upper bound on the time spent writing sessions at shutdown.
    std::chrono::seconds flush_budget{60};
    // A remote copy of the store (an S3-compatible bucket). The directory then caches it: sessions
    // are uploaded in the background, sessions other engines wrote appear in the store, and one
    // evicted from the directory stays available remotely. Null keeps the store local. Expiry
    // there is the bucket's lifecycle rule; `remote_prefix` is prepended to every key.
    std::shared_ptr<ObjectStore> remote;
    std::string remote_prefix;
    // How long shutdown waits for the uploads still queued.
    std::chrono::seconds remote_flush_budget{120};
    // Called on the writer thread after each session is written. Exceptions are ignored.
    std::function<void(const ContextStoreWriteEvent& event)> listener;

    [[nodiscard]] bool enabled() const noexcept { return !directory.empty(); }
};

// A host-side failure of the Engine worker, reported from the worker thread after the Engine has
// recovered from it or latched. `message` is the exception text, which is operator diagnostics
// and must not be forwarded to API clients.
struct EngineFaultEvent {
    std::string message;
    // The execution unit that threw: "boundary", "admission", "control", "prefill" or "decode".
    std::string unit;
    // The requests the failure was delivered to: the running lanes and the one materializing.
    std::vector<std::uint64_t> request_ids;
    std::vector<std::uint32_t> lanes;
    // Requests still queued that the latch fails as well; empty when the Engine recovered, since
    // recovery keeps the queue.
    std::vector<std::uint64_t> queued_request_ids;
    // True: the failure was confined to the one request being admitted (planned, or its binding
    // started), which alone received it. No other request was affected and the Engine neither
    // recovered nor counted the failure toward the latch.
    bool contained = false;
    // False: the Engine recovered and keeps serving. True: it is now permanently unavailable.
    bool latched = false;
    // Why recovery was refused; set only when latched.
    std::string latch_reason;
    // Failures in the current streak, this one included, and the count that latches.
    std::uint32_t consecutive_failures         = 0;
    std::uint32_t maximum_consecutive_failures = 0;
};

struct ContextCacheOptions {
    // Controls cross-request history reads and writes. Request pause/replay resources remain
    // available when history is disabled.
    bool enabled = true;
    // Extra Device StateImage slots beyond max_concurrency. Defaults to max_concurrency.
    std::optional<std::uint32_t> device_state_slots;
    // Shared Host quota for StateImages, KV, pause snapshots and in-flight destinations. Native
    // startup defaults to 8 GiB plus eight Host StateImages using the model's actual layout.
    // This does not bound total process RAM.
    // Engine::options() returns both resolved capacities after construction.
    std::optional<std::size_t> host_capacity_bytes;
    // Sizes host_capacity_bytes from the host memory available once the model is loaded, and
    // refuses an explicit host_capacity_bytes. Engine::options() returns the resolved capacity
    // with this flag cleared.
    bool auto_host_cache = false;
    // Memory left unpinned beneath whatever is available when auto_host_cache sizes the tier.
    std::size_t host_cache_reserve_bytes = kDefaultHostCacheReserveBytes;
    // An upper bound on what auto_host_cache pins, applied after the reserve. Empty: no bound, so a
    // machine with a lot of free memory pins nearly all of it. Set it where memory is shared with
    // other tenants (a rented GPU box): the reserve only protects the memory still to be used, not
    // what a neighbour may want of the host's.
    std::optional<std::size_t> host_cache_max_bytes;
    // An upper bound on what auto_host_cache pins as a share (1-100) of the machine's total memory
    // (the smaller of physical memory and the container's cgroup limit). Unlike the reserve and the
    // cap it does not depend on what happens to be free at startup, so it is the predictable way to
    // share a box. It only lowers the budget: the reserve and the available memory still apply.
    std::optional<std::uint32_t> host_cache_percent;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

enum class VisionResidency : std::uint8_t {
    Resident, // fixed Vision GPU allocations for the process lifetime
    Overlay,  // tower host-pinned; each image borrows device memory inside a bounded window
};

// A phantom-kv prompt graft: a hidden conversation prefix (e.g. a system turn plus an assistant
// acknowledgement) that sits in front of every request selecting it. `path` names the safetensors
// container; its metadata sidecar is the same path with a .json extension. The graft's tokens occupy
// positions [0, n) and the request's own rendered prompt starts at position n, exactly as if the
// hidden turn had been sent as text. They count toward max_context and the reported prompt tokens.
struct GraftSource {
    std::string name;
    std::filesystem::path path;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::size_t grammar_cache_bytes    = 256ULL * 1024 * 1024;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    // Empty or one entry keeps the single-device route and `device` selects it. Several entries
    // split the model's layers into that many pipeline stages, one per device, in the given order:
    // the first holds the embedding, head and round state. Matching compute capability is required
    // at construction; peer access is only probed and recorded as a capability -- boundary
    // transfers stage through pinned host memory when it is unavailable.
    std::vector<int> devices;
    // Layers per stage, one count per entry of `devices`. Empty lets the engine choose from each
    // device's free memory.
    std::vector<std::uint32_t> stage_layers;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    // Speed-for-quality trades, opt-in and off by default. Measured in
    // docs/maintainer/quality-trade-experiments.md: lm_head_q4 costs +0.69% perplexity for a
    // C8 decode gain (~3%, real chat prompts); gdn_state_fp16 is free within measurement noise
    // (bit-identical greedy output, unchanged perplexity) for a ~2% C8 gain and a halved
    // per-slot host state image.
    bool lm_head_q4                        = false;
    bool gdn_state_fp16                    = false;
    // Device-memory trades that store a W8 vocabulary matrix at a narrower group-64 width while
    // the weights load (the artifact is unchanged). lm_head_q6 saves ~26% of the output head for
    // +0.01% perplexity; embedding_q4 halves the token embedding with no measurable perplexity
    // change. lm_head_q4 and lm_head_q6 are mutually exclusive.
    bool lm_head_q6                        = false;
    bool embedding_q4                      = false;
    // Q6G64 token embedding: the narrower-model option where Q4 costs measurable perplexity
    // (the 35B-A3B's 2048-wide table). Mutually exclusive with embedding_q4.
    bool embedding_q6                      = false;
    // Qwen3.6-35B-A3B only: the MTP draft layer's routed experts are stored W8 in the artifact;
    // transcode them to the text layers' formats (Q4 gate_up, Q6 down) at load. Drafts are
    // verified exactly, so this can change acceptance but never the output distribution.
    bool mtp_experts_q4                    = false;
    // Integer-activation MLP gate_up at decode and verify widths. Measured 4-6% off that Op from
    // sixteen columns up; see docs/maintainer/quality-trade-experiments.md.
    bool mlp_a8_decode                     = false;
    // Integer-activation routes at full prefill tiles, on by default where a route is registered.
    // Clearing it returns prefill to the A16 routes, which is the only way to measure what the
    // integer routes are worth on a whole request rather than per Op.
    bool prefill_a8                        = true;
    // Hand wide prefill GEMMs to cuBLAS instead of this fork's integer mainloop: the weight is
    // materialised as int8 with one scale per row and the activations quantised per token, which
    // runs about twice as fast (ops/linear_swiglu/q4cublas/w4_cublas_prefill.h) and is a further
    // quality trade on top of prefill_a8 -- hence off by default. Its dequantise pass is
    // weight-sized, so it wants a large prefill_chunk to amortise; the two belong together.
    bool prefill_cublas                    = false;
    // Extends prefill_cublas to the attention and GDN input projections, which hold about a fifth
    // of the linear parameters. Worth +13% prefill for +0.071% perplexity on top of what the route
    // already costs, the GDN half carrying nearly all of both. Separable because that is a
    // different trade from the MLP one and an owner may want only the cheaper half.
    bool prefill_cublas_projections        = true;
    // Largest merged-token count one media item may occupy; larger media is downscaled at
    // preprocessing. Also bounds the overlay window.
    std::uint32_t vision_max_merged_tokens = 16384;
    bool use_cuda_graph                    = true;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    // Prompt grafts a request may select by name through PromptOptions::graft. Each is loaded and
    // validated against the resident model at construction.
    std::vector<GraftSource> grafts;
    StartupObserver startup_observer;
    ContextStoreOptions context_store;
    // Called on the worker thread for each host-side worker failure, after recovery or latch.
    // Must be quick; exceptions are ignored.
    std::function<void(const EngineFaultEvent& event)> fault_listener;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
    // Effective thinking budget derived from capacity and control requirements.
    // When omitted, defaults to budget.
    std::optional<std::uint32_t> effective_budget;
    // Whether canonical early-close guidance and control tokens can be inserted.
    bool early_close_available = true;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

enum class OutputConstraintKind : std::uint8_t { Grammar, JsonObject, JsonSchema, Choice, Regex };

// Constrains generated content; Chat reasoning retains the model's framing. Source is owning
// GBNF, JSON Schema or regex text. Choice owns literal alternatives; JsonObject has no payload.
struct OutputConstraint {
    OutputConstraintKind kind = OutputConstraintKind::Grammar;
    std::string source;
    std::vector<std::string> choices;

    [[nodiscard]] static OutputConstraint grammar(std::string source) {
        return {OutputConstraintKind::Grammar, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint json_object() {
        return {OutputConstraintKind::JsonObject, {}, {}};
    }

    [[nodiscard]] static OutputConstraint json_schema(std::string source) {
        return {OutputConstraintKind::JsonSchema, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint choice(std::vector<std::string> values) {
        return {OutputConstraintKind::Choice, {}, std::move(values)};
    }

    [[nodiscard]] static OutputConstraint regex(std::string pattern) {
        return {OutputConstraintKind::Regex, std::move(pattern), {}};
    }

    bool operator==(const OutputConstraint&) const = default;
};

enum class ToolChoiceMode : std::uint8_t { Auto, None, Required };
// Basic protects every tool call. Automatic applies constraints only for strict tools or an
// explicit selection/count policy.
enum class ToolConstraintMode : std::uint8_t { Automatic, Basic };

// Declarations belong to PromptOptions. Selection and cardinality affect this generation only.
struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::optional<std::vector<std::string>> allowed_names;
    bool parallel                  = true;
    ToolConstraintMode constraints = ToolConstraintMode::Basic;
};

struct RequestOptions {
    std::optional<OutputConstraint> constraint;
    ToolChoice tool_choice;
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Diagnostics for model-origin tool-call markup. Free generation uses schema-guided value
// normalization and can return malformed markup as text. Constrained calls use the model's
// parameter encoding contract and publish only complete calls.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                            = false;
    std::uint32_t structured_call_count         = 0;
    std::uint32_t empty_arguments_omitted       = 0;
    std::uint32_t schema_mismatch_arguments     = 0;
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    // Name of an EngineOptions::grafts entry to place in front of the rendered prompt; empty for
    // none. The rendered prompt should carry no system turn of its own: the graft already holds one.
    std::string graft;
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's structural
    // candidates. Exact reads from already-published shared prefixes remain enabled.
    bool allow_engine_automatic_shared_prefixes = true;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    InvalidToolConstraint,
    InvalidGrammar,
    InvalidChoice,
    InvalidRegex,
    InvalidJsonSchema,
    UnsupportedJsonSchema,
    UnsatisfiableJsonSchema,
    ConstraintDeadEnd,
    ContextLengthExceeded,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

enum class RequestErrorSource : std::uint8_t { OutputConstraint, Tools };

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message, std::string pointer = {},
                 RequestErrorSource source = RequestErrorSource::OutputConstraint)
        : std::invalid_argument(std::move(message)), kind_(kind), pointer_(std::move(pointer)),
          source_(source) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

    [[nodiscard]] RequestErrorSource source() const noexcept { return source_; }

    [[nodiscard]] const std::string& pointer() const noexcept { return pointer_; }

private:
    RequestErrorKind kind_;
    std::string pointer_;
    RequestErrorSource source_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

enum class GenerationSchedulingTransition : std::uint8_t {
    PauseStarted,
    Paused,
    RestoreStarted,
    Restored,
    ReplayComplete,
    RecoveryComplete,
    SnapshotRevoked,
    Terminal,
};

enum class GenerationRecoveryRoute : std::uint8_t { None, Snapshot, Replay };

// Sparse lifecycle observations captured by the Engine worker and delivered by wait() on the
// consumer thread. Counter deltas between boundaries, less the request's own deltas, measure
// other requests' completed work during recovery. Decode includes forced control tokens.
struct GenerationSchedulingObservation {
    GenerationSchedulingTransition transition = GenerationSchedulingTransition::PauseStarted;
    GenerationRecoveryRoute route             = GenerationRecoveryRoute::None;
    std::uint64_t engine_request_id           = 0;
    std::uint64_t steady_ns                   = 0;
    std::uint64_t elapsed_ns                  = 0;
    std::uint64_t preemption_index            = 0;
    std::uint64_t global_prefill_tokens       = 0;
    std::uint64_t global_decode_tokens        = 0;
    std::uint64_t global_replayed_tokens      = 0;
    std::uint64_t request_prefill_tokens      = 0;
    std::uint64_t request_decode_tokens       = 0;
    std::uint64_t request_replayed_tokens     = 0;
};

using GenerationSchedulingObserver = std::function<void(const GenerationSchedulingObservation&)>;

// Emitted once on the first accepted model/control token, delivered by wait() in either consumer
// mode. Submission elapsed includes queueing and binding; preparation is reported separately.
struct GenerationFirstTokenObservation {
    double prepare_seconds              = 0.0;
    double elapsed_since_submit_seconds = 0.0;
    double queue_wait_seconds           = 0.0;
};

using GenerationFirstTokenObserver = std::function<void(const GenerationFirstTokenObservation&)>;

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live output timings and prompt progress require a
// Streaming consumer; phase timings and scheduling observations also support Aggregate consumers.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
    // Optional for either consumer mode. No scheduling events are retained when unset.
    GenerationSchedulingObserver scheduling;
    GenerationFirstTokenObserver first_token;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    // Overlay vision residency: per-request sums over the image windows.
    std::uint32_t overlay_windows      = 0;
    double overlay_window_seconds      = 0.0;
    double overlay_evict_seconds       = 0.0;
    double overlay_restore_seconds     = 0.0;
    std::size_t overlay_evicted_bytes  = 0;
    std::size_t overlay_staged_bytes   = 0;
    // Windows that had to borrow the text weights, which stalls every other lane.
    std::uint32_t overlay_exclusive_windows = 0;
    // KV-funded windows whose encode started ahead of the request's prefill unit, beside other
    // lanes' work; the rest ran inside the unit.
    std::uint32_t overlay_ahead_windows = 0;
    double decode_seconds      = 0.0;
    // Prompt wall time begins at the successful initial binding attempt and ends at the first
    // accepted output token. Generation spans the first through last accepted output token and
    // therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                    = 0.0;
    double engine_boundary_exposed_seconds       = 0.0;
    double program_submit_exposed_seconds        = 0.0;
    double program_post_exposed_seconds          = 0.0;
    double engine_commit_output_exposed_seconds  = 0.0;
    double engine_maintenance_exposed_seconds    = 0.0;
    double device_wait_exposed_seconds           = 0.0;
    double constraint_draft_wait_exposed_seconds = 0.0;
    double decode_host_exposed_seconds           = 0.0;
    double decode_device_wait_exposed_seconds    = 0.0;
    std::uint64_t prefill_units                  = 0;
    std::uint64_t decode_rounds                  = 0;
    std::uint64_t control_units                  = 0;
};

// Request-owned scheduling observations. Restore counters count completed restorations;
// replayed_tokens counts tokens actually recomputed, including previously generated output;
// these tokens are separate from initial prompt prefill and delivered output. paused_ns covers
// pause preparation, waiting and binding restoration, excluding replay computation.
// Transfer bytes count request-owned payload submitted for binding, capture,
// pause and restore; background cache maintenance remains in the Engine-wide counters.
struct GenerationSchedulingStats {
    std::uint64_t preemptions          = 0;
    std::uint64_t snapshot_restores    = 0;
    std::uint64_t replay_restores      = 0;
    std::uint64_t replayed_tokens      = 0;
    std::uint64_t paused_ns            = 0;
    std::uint64_t device_to_host_bytes = 0;
    std::uint64_t host_to_device_bytes = 0;
};

// Request-owned execution work. GPU stream intervals overlap Host submission and completion
// waits; they are evidence about Device work, not an additional wall-time component.
struct GenerationWorkTiming {
    double submit_seconds = 0.0;
    double wait_seconds   = 0.0;
    double post_seconds   = 0.0;
    double gpu_seconds    = 0.0;
};

struct GenerationTransferTiming {
    std::uint64_t bytes = 0;
    double seconds      = 0.0;
};

// Frozen immediately before the first nonempty public output delta is published. This boundary
// differs from the first accepted model token and from the client's first HTTP output. Elapsed
// starts at Engine submission, excluding Frontend preparation. Initial queue, initial binding,
// paused time and the remaining resident interval partition this Engine wall time.
struct GenerationFirstOutputTiming {
    double elapsed_seconds         = 0.0;
    double initial_binding_seconds = 0.0;
    GenerationEngineTiming engine;
    GenerationWorkTiming prefill;
    GenerationWorkTiming replay;
    GenerationSchedulingStats scheduling;
    std::uint32_t computed_prefill_tokens = 0;
    // Resources: State, Main KV, Backend KV. Directions: D2H, H2D, D2D.
    // Completed request-owned transfers only; background reclamation remains Engine-wide.
    std::array<std::array<GenerationTransferTiming, 3>, 3> context_transfers{};
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> requested_budget;
    std::optional<std::uint32_t> effective_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

enum class ConstraintCacheAccess : std::uint8_t { Hit, Built, Waited };
enum class ConstraintOutputBranch : std::uint8_t { Undecided, Content, Tools };

// State describes the committed output language, including an assistant continuation prefix.
// Work includes speculative lookahead that was subsequently rolled back. Times are subintervals
// of existing request timings, not extra latency to add to them.
struct ConstraintObservation {
    ConstraintOutputBranch branch   = ConstraintOutputBranch::Undecided;
    bool complete                   = false;
    bool terminated                 = false;
    ConstraintCacheAccess cache     = ConstraintCacheAccess::Hit;
    bool timings_collected          = false;
    double prepare_seconds          = 0.0;
    double mask_seconds             = 0.0;
    double matcher_seconds          = 0.0;
    std::uint64_t mask_positions    = 0;
    std::uint64_t mask_upload_bytes = 0;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    Checkpoint,
};

enum class AdmissionFallbackReason : std::uint8_t {
    None,
    SourceInvalid,
    SourceRevoked,
    CostChanged,
    CapacityLimit,
    IsolatedCapacity,
};

struct GenerationAdmissionStats {
    std::uint32_t preferred_reused_tokens = 0;
    // Subset of initial queue wait, not additional TTFT. Includes waiting for a lane
    // after a useful source has been selected.
    double source_wait_seconds              = 0.0;
    std::uint32_t revoked_checkpoints       = 0;
    AdmissionFallbackReason fallback_reason = AdmissionFallbackReason::None;
};

struct GenerationResult {
    // Unique within this Engine instance; diagnostic correlation only.
    std::uint64_t engine_request_id = 0;
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    // Completed initial prefill, excluding cached tokens and separately counted replay work.
    std::uint32_t computed_prefill_tokens = 0;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    GenerationSchedulingStats scheduling;
    GenerationAdmissionStats admission;
    std::optional<GenerationFirstOutputTiming> first_output_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
    std::optional<ConstraintObservation> constraint;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
    VisionResidency residency             = VisionResidency::Resident;
    std::size_t window_capacity_bytes     = 0; // overlay: device bytes one window borrows
    std::size_t pinned_weight_bytes       = 0; // overlay: host-pinned tower bytes
    std::size_t mirror_bytes              = 0; // overlay: pinned mirror of the borrowable tail
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    // Pipeline stages (`devices`): the runtime reservation of each further stage's device (stage 1
    // first; the fields above are stage 0's), and the stage whose device bounded the resolved KV
    // capacity. Empty and 0 on one device.
    std::vector<std::size_t> stage_runtime_reservation_bytes;
    std::size_t kv_capacity_binding_stage = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t kv_payload_bytes                  = 0;
    // One shared physical Host context backing. Reserved bytes are included in occupied bytes;
    // State/KV occupancy below is a breakdown and must not be added to this ledger again.
    std::size_t host_context_capacity_bytes = 0;
    std::size_t host_context_occupied_bytes = 0;
    std::size_t host_context_reserved_bytes = 0;
    std::uint32_t host_state_occupied_slots = 0;
    std::size_t host_kv_occupied_bytes      = 0;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns       = 0;
    std::uint64_t program_submit_ns        = 0;
    std::uint64_t program_post_ns          = 0;
    std::uint64_t engine_commit_output_ns  = 0;
    std::uint64_t engine_maintenance_ns    = 0;
    std::uint64_t device_wait_ns           = 0;
    std::uint64_t constraint_draft_wait_ns = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t stats_publication_invocations = 0;

    // Host-active time: every engine and program phase, device wait excluded.
    [[nodiscard]] std::uint64_t active_ns() const noexcept {
        return engine_boundary_ns + program_submit_ns + program_post_ns + engine_commit_output_ns +
               engine_maintenance_ns;
    }
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Full prompt counted once when initial binding succeeds; includes reused tokens.
    std::uint64_t prompt_tokens = 0;
    // All committed output tokens, including the first token and Engine-injected control tokens.
    std::uint64_t generated_tokens = 0;
    // Native verification work, using the same acceptance semantics as SpeculativeStats.
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    // Initial prompt tokens evaluated by prefill. Reused checkpoint-prefix tokens and replay
    // recomputation are excluded; replayed_tokens separately counts that additional model work.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Execution time of prefill units and decode rounds (host submission, device wait and host
    // post-processing). Advances per unit, so a scraper sees rates move during a long request.
    double prefill_seconds_total = 0.0;
    double decode_seconds_total  = 0.0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t paused_requests           = 0;
    std::uint32_t replaying_requests        = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;
    std::uint64_t preemptions               = 0;
    std::uint64_t snapshot_restores         = 0;
    std::uint64_t replay_restores           = 0;
    std::uint64_t replayed_tokens           = 0;

    // Requests that left the queue without being admitted, and the time they had waited: the
    // client gave up (cancelled) or the pending timeout fired (expired).
    std::uint64_t waiting_cancelled_requests = 0;
    std::uint64_t waiting_expired_requests   = 0;
    double waiting_abandoned_seconds         = 0.0;
    // Requests cancelled while their prompt was prefilling and the prompt tokens they had computed.
    std::uint64_t cancelled_prefills                = 0;
    std::uint64_t cancelled_prefill_computed_tokens = 0;

    std::uint64_t root_selections               = 0;
    std::uint64_t checkpoint_selections         = 0;
    std::uint64_t reused_prompt_tokens          = 0;
    std::uint32_t last_selected_frontier_tokens = 0;

    std::uint64_t state_moves                 = 0;
    std::uint64_t state_forks                 = 0;
    std::uint64_t materialization_state_forks = 0;
    std::uint64_t state_restores              = 0;
    std::uint64_t state_d2h_count             = 0;
    std::uint64_t state_h2d_count             = 0;
    std::uint64_t state_d2d_count             = 0;
    std::uint64_t state_d2h_bytes             = 0;
    std::uint64_t state_h2d_bytes             = 0;
    std::uint64_t state_d2d_bytes             = 0;
    double state_d2h_seconds                  = 0.0;
    double state_h2d_seconds                  = 0.0;
    double state_d2d_seconds                  = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages             = 0;
    std::uint64_t partial_tail_cow_pages           = 0;
    std::uint32_t device_state_occupied_slots      = 0;
    std::uint32_t host_state_occupied_slots        = 0;
    std::uint32_t device_main_kv_occupied_pages    = 0;
    std::uint32_t device_backend_kv_occupied_pages = 0;
    std::size_t host_kv_occupied_bytes             = 0;
    // Unified physical Host context occupancy. Reserved destinations are included in occupied.
    std::size_t host_context_occupied_bytes = 0;
    std::size_t host_context_reserved_bytes = 0;
    // Allocator lifetime high-water mark, including reserved transfer destinations.
    std::size_t host_context_peak_occupied_bytes = 0;
    double actual_context_transfer_seconds       = 0.0;
    // Host-side failures the worker survived by failing the in-flight requests and clearing the
    // context cache instead of latching the Engine unavailable.
    std::uint64_t engine_recoveries = 0;

    // The durable context store (zero throughout when it is disabled).
    std::uint64_t context_store_images         = 0;
    std::uint64_t context_store_used_bytes     = 0;
    std::uint64_t context_store_writes         = 0; // sessions written
    std::uint64_t context_store_write_failures = 0;
    std::uint64_t context_store_dropped        = 0; // sessions not written: write queue full
    std::uint64_t context_store_bytes_written  = 0; // new chunk bytes written to disk
    std::uint64_t context_store_bytes_reused   = 0; // chunk bytes already held, not rewritten
    std::uint64_t context_store_evicted        = 0; // removed for space, age or supersession
    std::uint64_t context_store_corrupt        = 0; // removed because they could not be read
    std::uint64_t context_store_restored       = 0; // sessions restored into the cache at start-up
    std::uint64_t context_store_restored_bytes = 0;
    double context_store_restore_seconds       = 0.0;
    // Stored sessions brought back into the cache for a request that would otherwise have been
    // prefilled from further back, and the prompt tokens that bought.
    std::uint64_t context_store_hydrations         = 0;
    std::uint64_t context_store_hydrated_tokens    = 0;
    std::uint64_t context_store_hydration_failures = 0;
    double context_store_hydration_seconds         = 0.0;
    // With a remote: images only the remote holds, objects and bytes moved, and failures.
    std::uint64_t context_store_remote_images            = 0;
    std::uint64_t context_store_remote_uploads           = 0;
    std::uint64_t context_store_remote_upload_bytes      = 0;
    std::uint64_t context_store_remote_upload_failures   = 0;
    std::uint64_t context_store_remote_downloads         = 0;
    std::uint64_t context_store_remote_download_bytes    = 0;
    std::uint64_t context_store_remote_download_failures = 0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::uint64_t pinned_weight_bytes  = 0; // host-pinned weights (overlay vision tower)
    std::uint64_t overlay_window_bytes = 0; // device bytes one overlay vision window borrows
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;
};

} // namespace ninfer
