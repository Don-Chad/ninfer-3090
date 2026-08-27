#pragma once

#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

// The Qwen tool syntax carries each top-level argument as untyped text between
// <parameter> tags. This request-owned contract retains only whether an explicit
// JSON Schema type admits a string or requires JSON decoding.
struct ToolArgumentTypeContracts {
    enum class Encoding : std::uint8_t {
        Json,
        String,
    };

    struct Parameter {
        std::string name;
        Encoding encoding = Encoding::Json;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
};

ToolArgumentTypeContracts build_tool_argument_type_contracts(const GenerationRequest& request);

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolArgumentTypeContracts& contracts);

// Incrementally publishes text that is provably outside a possible Qwen
// <tool_call> suffix. At terminal time, a valid tool response discards the
// buffered tool region; malformed/non-tool output flushes it verbatim.
class ToolCallStreamFilter {
public:
    std::string feed(std::string_view text);
    std::string finish(bool is_tool_call_response);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    std::string trailing_whitespace_;
    std::string tool_region_;
    std::size_t marker_prefix_bytes_ = 0;
    std::size_t emitted_bytes_       = 0;
    bool saw_tool_marker_            = false;
    bool finished_                   = false;
};

} // namespace ninfer::serve
