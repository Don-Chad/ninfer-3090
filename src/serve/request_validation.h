#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

// The NInfer `graft` extension field: a graft name, or absent/null/"" for none. Whether the name is
// loaded is checked against the server's grafts during translation.
std::string parse_graft_field(const RequestJson& body);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// Structured output. `schema` must be a JSON Schema object; `param` names the protocol field for
// errors. The schema is kept serialized in its original member order, which is the order the
// Engine emits properties in.
[[nodiscard]] ninfer::OutputFormat json_schema_output_format(const RequestJson& schema, bool strict,
                                                             const std::string& param);
// Rejects combinations that a structured-output request cannot honor: active tools (tool calls are
// not JSON values) and ignore_eos (a completed value can only be followed by a stop token).
void validate_output_format_compatibility(const GenerationRequest& request,
                                          const std::string& param);

} // namespace ninfer::serve
