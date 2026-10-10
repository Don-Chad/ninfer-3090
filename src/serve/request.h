#pragma once

#include "product/media_acquire/source.h"

#include <ninfer/types.h>

// Internal, wire-format-independent representation of a generation request.
//
// OpenAI and Anthropic schemas map into this wire-independent value.
// translate.cpp then produces the public PromptInput and RequestOptions consumed
// by Engine; media sources remain unresolved until the product service acquires
// owning bytes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    // --default-max-tokens: the fixed budget of a request that omits its limit. Unset, such a
    // request receives its remaining context once its prompt is prepared.
    std::optional<int> default_max_tokens;
    int max_context = 8192; // --max-context, the upper bound of any derived budget
};

enum class ContentKind {
    Text,
    Image,
    Video,
};

struct CacheBoundary {
    enum class Ttl : std::uint8_t {
        Default,
        FiveMinutes,
        OneHour,
    };

    ninfer::PromptCacheMarkerKind kind       = ninfer::PromptCacheMarkerKind::SharedStablePrefix;
    ninfer::SharedCandidateEvidence evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary;
    Ttl ttl                                  = Ttl::Default;

    [[nodiscard]] friend constexpr bool operator==(CacheBoundary, CacheBoundary) noexcept = default;
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original wire "type" string for diagnostics
    ninfer::product::media_acquire::Source source;
    ninfer::ImageResizePolicy image_resize_policy = ninfer::ImageResizePolicy::Downsize;
    std::optional<CacheBoundary> cache_boundary_after;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string input_schema_json;
    std::string schema_param; // Original protocol path; never included in the model prompt.
    bool strict = false;
    std::optional<std::string> input_examples_json;
    std::optional<CacheBoundary> cache_boundary_after;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
    // Serve-only wire kind (Responses `custom_tool_call` vs `function_call`); the Engine ignores it.
    // Kept so call-graph validation can reject a result of the other kind, including across
    // previous_response_id replay.
    bool custom = false;
};

using ToolChoiceMode = ninfer::ToolChoiceMode;
using ToolChoice     = ninfer::ToolChoice;

struct ChatTurn {
    ChatRole role = ChatRole::User;
    std::vector<ContentPart> content; // ordered parts; may be empty when wire content is empty
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id; // populated for role=tool
    // Optional protocol assertion for a tool result. Call-graph normalization verifies it against
    // the function identified by tool_call_id before the Engine sees the history.
    std::optional<std::string> tool_result_name;
    // Optional wire-kind assertion (true: custom_tool_call_output, false: function_call_output),
    // verified against ToolCall::custom of the call identified by tool_call_id. Unset when the
    // protocol has no such distinction.
    std::optional<bool> tool_result_custom;
    bool tool_result_is_error = false;
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
    std::optional<CacheBoundary> cache_boundary_after;
};

// Sampling overrides that have an executable Engine meaning. Protocol-only
// fields are normalized or rejected before this value is constructed.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<double> min_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Protocol-level effort vocabulary. Each wire adapter accepts the values from
// its external contract; translation passes explicit values to the selected template.
enum class RequestedReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::optional<RequestedReasoningEffort>
parse_requested_reasoning_effort(std::string_view value) noexcept {
    if (value == "none") { return RequestedReasoningEffort::None; }
    if (value == "minimal") { return RequestedReasoningEffort::Minimal; }
    if (value == "low") { return RequestedReasoningEffort::Low; }
    if (value == "medium") { return RequestedReasoningEffort::Medium; }
    if (value == "high") { return RequestedReasoningEffort::High; }
    if (value == "xhigh") { return RequestedReasoningEffort::XHigh; }
    if (value == "max") { return RequestedReasoningEffort::Max; }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view
requested_reasoning_effort_name(RequestedReasoningEffort effort) noexcept {
    switch (effort) {
    case RequestedReasoningEffort::None:
        return "none";
    case RequestedReasoningEffort::Minimal:
        return "minimal";
    case RequestedReasoningEffort::Low:
        return "low";
    case RequestedReasoningEffort::Medium:
        return "medium";
    case RequestedReasoningEffort::High:
        return "high";
    case RequestedReasoningEffort::XHigh:
        return "xhigh";
    case RequestedReasoningEffort::Max:
        return "max";
    }
    return {};
}

struct GenerationRequest {
    std::optional<OutputConstraint> constraint;
    std::string constraint_param;
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = 64;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    bool stop_strings_apply_to_reasoning = false;
    // Benchmark/serving extension shared with vLLM, SGLang and llama.cpp: suppress the
    // checkpoint's default stop tokens so generation runs to the requested token budget.
    // Caller-supplied stop tokens and stop strings still apply.
    bool ignore_eos = false;
    int max_tokens                       = 0; // resolved budget; zero means immediate output limit
    // The request omitted its limit and the server has no fixed default: GenerationService replaces
    // max_tokens (then the --max-context upper bound) with the prompt's remaining context.
    bool derive_output_budget = false;
    std::optional<bool> enable_thinking;      // unset => use the server default
    std::optional<std::uint32_t> thinking_budget;
    std::optional<RequestedReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    // NInfer extension: name of a phantom-kv graft loaded with --graft. Unset => use the server's
    // --default-graft (if any); an empty name explicitly selects no graft.
    std::optional<std::string> graft;
    ninfer::PromptContinuationMode continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    bool private_cache_boundary_at_prompt_end   = false;
    bool allow_engine_automatic_shared_prefixes = true;
    SamplingParams sampling;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None &&
               (!tool_choice.allowed_names || !tool_choice.allowed_names->empty());
    }

    [[nodiscard]] bool constrains_tools() const noexcept {
        if (tools.empty() || constraint) return false;
        if (!uses_tools()) return true;
        if (tool_choice.mode == ToolChoiceMode::Required || !tool_choice.parallel ||
            tool_choice.allowed_names || tool_choice.constraints == ToolConstraintMode::Basic)
            return true;
        return std::any_of(tools.begin(), tools.end(),
                           [](const auto& tool) { return tool.strict; });
    }

    [[nodiscard]] std::size_t media_item_count() const noexcept {
        std::size_t count = 0;
        for (const ChatTurn& message : messages) {
            for (const ContentPart& part : message.content) {
                if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
            }
        }
        return count;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == ChatRole::Tool) { return true; }
        }
        return false;
    }
};

// The output limit of a request that omitted one.
inline void apply_default_output_limit(GenerationRequest& request, const RequestLimits& limits) {
    request.derive_output_budget = !limits.default_max_tokens.has_value();
    request.max_tokens           = limits.default_max_tokens.value_or(limits.max_context);
}

} // namespace ninfer::serve
