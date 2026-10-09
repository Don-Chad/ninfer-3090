#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::text {
struct ParsedJsonNumbers;
}

namespace ninfer::serve {
void validate_schema_number_input(const text::ParsedJsonNumbers& parsed);

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);
enum class JsonFormatProtocol { Chat, Responses, Anthropic };
void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol);

void parse_structured_outputs(const RequestJson& body, GenerationRequest& request);

// The NInfer `graft` extension field: a graft name, "" for explicitly none, or absent/null (nullopt)
// to take the server's default graft. Whether the name is loaded is checked against the server's
// grafts during translation.
std::optional<std::string> parse_graft_field(const RequestJson& body);

// The NInfer `thinking_budget` extension field: a positive cap on model-origin thinking tokens for
// this request, or absent/null (nullopt) to take the server default. Ignored when the request does
// not think. Anthropic Messages carries the same cap as `thinking.budget_tokens`.
std::optional<std::uint32_t> parse_thinking_budget_field(const RequestJson& body);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

} // namespace ninfer::serve
