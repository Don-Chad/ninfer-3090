#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const RequestJson& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

// True for OpenAI tool types the *server* would have executed - hosted search, hosted code
// execution, hosted file search and the like. NInfer has no executor for any of them, and a client
// never waits on one itself, so declaring one is silently dropped rather than failing the request:
// the model is simply never told the tool exists, which is the same outcome as not declaring it.
//
// Client-executed types stay rejected. Dropping one of those would leave the caller waiting for a
// call that can never arrive, which is worse than a clear error.
[[nodiscard]] bool is_hosted_openai_tool_type(std::string_view type) noexcept;

// What /v1/models advertises about the one resident model.
struct ModelDescription {
    std::string id;
    std::uint32_t max_model_len = 0; // --max-context, each sequence's ceiling
    bool vision                 = false;
};

std::string make_models_list(const ModelDescription& model, std::int64_t created);
std::string make_model_object(const ModelDescription& model, std::int64_t created);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

void validate_openai_model(std::string_view requested, std::string_view available);

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
