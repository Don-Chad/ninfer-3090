#include "serve/responses_schema.h"

#include "serve/generation_service.h"
#include "serve/openai_schema.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kCompactionPrompt =
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another "
    "LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work.";

constexpr std::string_view kCompactionSummaryPrefix =
    "Another language model started to solve this problem and produced a summary of its thinking "
    "process. You also have access to the state of the tools that were used by that language "
    "model. Use this to build on the work that has already been done and avoid duplicating work. "
    "Here is the summary produced by the other language model, use the information in this summary "
    "to assist with your own analysis:";

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {}) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

const Json& require_object(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    return body;
}

bool optional_bool(const Json& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

std::optional<double> optional_number(const Json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

std::optional<int> optional_int(const Json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number_integer()) {
        bad_request(std::string(key) + " must be an integer", key);
    }
    if (object.at(key).is_number_unsigned()) {
        const std::uint64_t value = object.at(key).get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(value);
    }
    const std::int64_t value = object.at(key).get<std::int64_t>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(value);
}

bool valid_function_name(const std::string& name) {
    if (name.empty() || name.size() > 64) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string require_function_name(const Json& object, const char* param) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    std::string name = object.at("name").get<std::string>();
    if (!valid_function_name(name)) {
        bad_request("function name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

std::string optional_tool_namespace(const Json& object, const char* param) {
    if (!object.contains("namespace") || object.at("namespace").is_null()) { return {}; }
    if (!object.at("namespace").is_string() || object.at("namespace").get<std::string>().empty()) {
        bad_request("tool namespace must be a non-empty string", param);
    }
    const std::string value = object.at("namespace").get<std::string>();
    if (value.size() > 64) { bad_request("tool namespace must be at most 64 characters", param); }
    return value;
}

std::string model_tool_name(const std::string& tool_namespace, const std::string& wire_name) {
    if (tool_namespace.empty() || tool_namespace == "functions") { return wire_name; }
    std::string encoded;
    encoded.reserve(tool_namespace.size() + 2 + wire_name.size());
    for (const unsigned char c : tool_namespace) {
        encoded.push_back(std::isalnum(c) != 0 || c == '_' || c == '-' ? static_cast<char>(c) : '_');
    }
    encoded += "__";
    encoded += wire_name;
    return encoded;
}

std::string item_id(const Json& item, const char* prefix, const char* param) {
    if (!item.contains("id") || item.at("id").is_null()) { return new_response_item_id(prefix); }
    if (!item.at("id").is_string() || item.at("id").get<std::string>().empty()) {
        bad_request("input Item id must be a non-empty string", param);
    }
    return item.at("id").get<std::string>();
}

ninfer::product::media_acquire::Source parse_image_source(const Json& part) {
    if (part.contains("file_id") && !part.at("file_id").is_null()) {
        bad_request("input_image.file_id is not supported; use image_url", "input",
                    "file_inputs_not_supported");
    }
    if (!part.contains("image_url") || !part.at("image_url").is_string() ||
        part.at("image_url").get<std::string>().empty()) {
        bad_request("input_image must contain a non-empty image_url", "input");
    }
    if (part.contains("detail") && !part.at("detail").is_null()) {
        if (!part.at("detail").is_string()) {
            bad_request("input_image.detail must be a string", "input");
        }
        const std::string detail = part.at("detail").get<std::string>();
        if (detail != "auto" && detail != "low" && detail != "high" &&
            detail != "original") {
            bad_request("input_image detail must be 'auto', 'low', 'high', or 'original'", "input",
                        "image_detail_not_supported");
        }
    }

    ninfer::product::media_acquire::Source source;
    source.value = part.at("image_url").get<std::string>();
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request("input_image.image_url must use HTTP(S) or a data URI", "input");
    }
    return source;
}

ninfer::product::media_acquire::Source parse_video_source(const Json& part) {
    if (!part.contains("video_url") || !part.at("video_url").is_string() ||
        part.at("video_url").get<std::string>().empty()) {
        bad_request("input_video must contain a non-empty video_url", "input");
    }
    ninfer::product::media_acquire::Source source;
    source.value = part.at("video_url").get<std::string>();
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request("input_video.video_url must use HTTP(S) or a data URI", "input");
    }
    return source;
}

struct ParsedMessage {
    ChatTurn turn;
    Json canonical;
};

ParsedMessage parse_message_item(const Json& item, std::size_t index) {
    if (!item.contains("role") || !item.at("role").is_string()) {
        bad_request("input message " + std::to_string(index) + " must contain a string role",
                    "input");
    }
    const std::string role = item.at("role").get<std::string>();
    ChatRole parsed_role;
    if (role == "user") {
        parsed_role = ChatRole::User;
    } else if (role == "assistant") {
        parsed_role = ChatRole::Assistant;
    } else if (role == "system") {
        parsed_role = ChatRole::System;
    } else if (role == "developer") {
        parsed_role = ChatRole::Developer;
    } else {
        bad_request("unsupported input message role: " + role, "input", "unsupported_role");
    }
    if (item.contains("phase") && !item.at("phase").is_null() &&
        !item.at("phase").is_string()) {
        bad_request("message phase must be a string or null", "input");
    }
    if (item.contains("status") && !item.at("status").is_null()) {
        if (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed") {
            bad_request("input message status must be 'completed'", "input");
        }
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("input message " + std::to_string(index) + " must contain content", "input");
    }

    ParsedMessage parsed;
    parsed.turn.role       = parsed_role;
    Json content           = Json::array();
    const auto append_text = [&](const std::string& text, const std::string& wire_type) {
        ContentPart part;
        part.kind     = ContentKind::Text;
        part.text     = text;
        part.type_raw = wire_type;
        parsed.turn.content.push_back(std::move(part));
        Json canonical = {{"type", wire_type}, {"text", text}};
        if (wire_type == "output_text") { canonical["annotations"] = Json::array(); }
        content.push_back(std::move(canonical));
    };

    if (item.at("content").is_string()) {
        append_text(item.at("content").get<std::string>(),
                    parsed_role == ChatRole::Assistant ? "output_text" : "input_text");
    } else if (item.at("content").is_array()) {
        for (const Json& value : item.at("content")) {
            if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
                bad_request("input message content parts must have a string type", "input");
            }
            const std::string type = value.at("type").get<std::string>();
            if (type == "input_text" || type == "output_text") {
                if (!value.contains("text") || !value.at("text").is_string()) {
                    bad_request(type + " must contain a string text", "input");
                }
                if (type == "output_text" && parsed_role != ChatRole::Assistant) {
                    bad_request("output_text is only valid on assistant messages", "input");
                }
                append_text(value.at("text").get<std::string>(), type);
            } else if (type == "input_image") {
                if (parsed_role != ChatRole::User) {
                    bad_request("input_image is only supported on user messages", "input");
                }
                ContentPart part;
                part.kind     = ContentKind::Image;
                part.type_raw = type;
                part.source   = parse_image_source(value);
                parsed.turn.content.push_back(std::move(part));
                content.push_back(Json{{"type", "input_image"},
                                       {"image_url", value.at("image_url")},
                                       {"detail", "auto"}});
            } else if (type == "input_video") {
                if (parsed_role != ChatRole::User) {
                    bad_request("input_video is only supported on user messages", "input");
                }
                ContentPart part;
                part.kind     = ContentKind::Video;
                part.type_raw = type;
                part.source   = parse_video_source(value);
                parsed.turn.content.push_back(std::move(part));
                content.push_back(
                    Json{{"type", "input_video"}, {"video_url", value.at("video_url")}});
            } else if (type == "input_file") {
                bad_request("input_file is not supported", "input", "file_inputs_not_supported");
            } else if (type == "input_audio") {
                bad_request("input_audio is not supported", "input", "audio_inputs_not_supported");
            } else {
                bad_request("unsupported message content type: " + type, "input",
                            "modality_not_supported");
            }
        }
    } else {
        bad_request("input message content must be a string or array", "input");
    }
    if (parsed.turn.content.empty()) {
        bad_request("input message content must not be empty", "input");
    }

    parsed.canonical = {{"id", item_id(item, "msg", "input")},
                        {"type", "message"},
                        {"role", role},
                        {"content", std::move(content)}};
    return parsed;
}

std::string parse_reasoning_item(const Json& item, Json& canonical) {
    if (item.contains("encrypted_content") && !item.at("encrypted_content").is_null() &&
        !item.at("encrypted_content").is_string()) {
        bad_request("encrypted reasoning content must be a string or null", "input");
    }
    if (item.contains("summary") && !item.at("summary").is_null()) {
        if (!item.at("summary").is_array()) {
            bad_request("reasoning summary must be an array", "input");
        }
    }
    if ((!item.contains("content") || item.at("content").is_null()) &&
        item.contains("encrypted_content") && !item.at("encrypted_content").is_null()) {
        canonical = { {"id", item_id(item, "rs", "input")},
                      {"type", "reasoning"},
                      {"summary", Json::array()},
                      {"content", Json::array()} };
        return {};
    }
    if (!item.contains("content") || !item.at("content").is_array()) {
        bad_request("reasoning Item must contain a content array", "input");
    }
    std::string text;
    Json content = Json::array();
    for (const Json& part : item.at("content")) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string() ||
            part.at("type").get<std::string>() != "reasoning_text" || !part.contains("text") ||
            !part.at("text").is_string()) {
            bad_request("reasoning content only supports reasoning_text parts", "input");
        }
        text += part.at("text").get<std::string>();
        content.push_back(Json{{"type", "reasoning_text"}, {"text", part.at("text")}});
    }
    canonical = {{"id", item_id(item, "rs", "input")},
                 {"type", "reasoning"},
                 {"summary", Json::array()},
                 {"content", std::move(content)}};
    return text;
}

ToolCall parse_function_call_item(const Json& item, Json& canonical) {
    ToolCall call;
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get<std::string>().empty()) {
        bad_request("function_call must contain a non-empty call_id", "input");
    }
    call.id                       = item.at("call_id").get<std::string>();
    const std::string wire_name   = require_function_name(item, "input");
    call.namespace_name           = optional_tool_namespace(item, "input");
    call.name                     = model_tool_name(call.namespace_name, wire_name);
    if (!item.contains("arguments") || !item.at("arguments").is_string()) {
        bad_request("function_call arguments must be a JSON string", "input");
    }
    call.arguments_json  = item.at("arguments").get<std::string>();
    const Json arguments = Json::parse(call.arguments_json, nullptr, false);
    if (arguments.is_discarded() || !arguments.is_object()) {
        bad_request("function_call arguments must encode a JSON object", "input");
    }
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("function_call status must be 'completed'", "input");
    }
    canonical = {{"id", item_id(item, "fc", "input")},
                 {"type", "function_call"},
                 {"status", "completed"},
                 {"call_id", call.id},
                 {"name", wire_name},
                 {"arguments", call.arguments_json}};
    if (!call.namespace_name.empty()) { canonical["namespace"] = call.namespace_name; }
    return call;
}

ToolCall parse_custom_tool_call_item(const Json& item, Json& canonical) {
    ToolCall call;
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get<std::string>().empty()) {
        bad_request("custom_tool_call must contain a non-empty call_id", "input");
    }
    call.id                     = item.at("call_id").get<std::string>();
    const std::string wire_name = require_function_name(item, "input");
    call.namespace_name         = optional_tool_namespace(item, "input");
    call.name                   = model_tool_name(call.namespace_name, wire_name);
    call.custom                 = true;
    if (!item.contains("input") || !item.at("input").is_string()) {
        bad_request("custom_tool_call input must be a string", "input");
    }
    call.custom_input   = item.at("input").get<std::string>();
    call.arguments_json = Json{{"input", call.custom_input}}.dump();
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("custom_tool_call status must be 'completed'", "input");
    }
    canonical = {{"id", item_id(item, "ctc", "input")},
                 {"type", "custom_tool_call"},
                 {"status", "completed"},
                 {"call_id", call.id},
                 {"name", wire_name},
                 {"input", call.custom_input}};
    if (!call.namespace_name.empty()) { canonical["namespace"] = call.namespace_name; }
    return call;
}

ToolCall parse_tool_search_call_item(const Json& item, Json& canonical) {
    ToolCall call;
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get<std::string>().empty()) {
        bad_request("tool_search_call must contain a non-empty call_id", "input");
    }
    if (!item.contains("execution") || !item.at("execution").is_string() ||
        item.at("execution").get<std::string>() != "client") {
        bad_request("tool_search_call execution must be 'client'", "input");
    }
    if (!item.contains("arguments") || !item.at("arguments").is_object()) {
        bad_request("tool_search_call arguments must be an object", "input");
    }
    call.id                    = item.at("call_id").get<std::string>();
    call.name                  = "tool_search";
    call.arguments_json        = item.at("arguments").dump();
    call.tool_search           = true;
    call.tool_search_execution = "client";
    canonical = {{"id", item_id(item, "ts", "input")},
                 {"type", "tool_search_call"},
                 {"status", "completed"},
                 {"call_id", call.id},
                 {"execution", "client"},
                 {"arguments", item.at("arguments")}};
    return call;
}

ChatTurn parse_function_call_output_item(const Json& item, Json& canonical) {
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get<std::string>().empty()) {
        bad_request("function_call_output must contain a non-empty call_id", "input");
    }
    if (!item.contains("output") ||
        (!item.at("output").is_string() && !item.at("output").is_array())) {
        bad_request("function_call_output output must be a string or content array", "input");
    }
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("function_call_output status must be 'completed'", "input");
    }
    ChatTurn turn;
    turn.role         = ChatRole::Tool;
    turn.tool_call_id = item.at("call_id").get<std::string>();
    const auto append_text = [&](const std::string& text) {
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.type_raw = "input_text";
        content.text     = text;
        turn.content.push_back(std::move(content));
    };
    if (item.at("output").is_string()) {
        append_text(item.at("output").get<std::string>());
    } else {
        if (item.at("output").empty()) {
            bad_request("function_call_output content array must not be empty", "input");
        }
        for (const Json& value : item.at("output")) {
            if (!value.is_object() || !value.contains("type") ||
                !value.at("type").is_string()) {
                bad_request("function_call_output content items require a string type", "input");
            }
            const std::string type = value.at("type").get<std::string>();
            if (type == "input_text") {
                if (!value.contains("text") || !value.at("text").is_string()) {
                    bad_request("function_call_output input_text requires text", "input");
                }
                append_text(value.at("text").get<std::string>());
            } else if (type == "input_image") {
                ContentPart content;
                content.kind     = ContentKind::Image;
                content.type_raw = type;
                content.source   = parse_image_source(value);
                turn.content.push_back(std::move(content));
            } else if (type == "input_audio") {
                bad_request("function_call_output input_audio is not supported", "input",
                            "audio_inputs_not_supported");
            } else if (type == "encrypted_content") {
                if (!value.contains("encrypted_content") ||
                    !value.at("encrypted_content").is_string()) {
                    bad_request("encrypted_content requires encrypted_content", "input");
                }
            } else {
                bad_request("unsupported function_call_output content type: " + type, "input",
                            "modality_not_supported");
            }
        }
        if (turn.content.empty()) {
            bad_request("function_call_output contains no locally usable content", "input",
                        "modality_not_supported");
        }
    }
    canonical = {{"id", item_id(item, "fco", "input")},
                 {"type", "function_call_output"},
                 {"status", "completed"},
                 {"call_id", turn.tool_call_id},
                 {"output", item.at("output")}};
    return turn;
}

ChatTurn parse_custom_tool_call_output_item(const Json& item, Json& canonical) {
    ChatTurn turn     = parse_function_call_output_item(item, canonical);
    canonical["type"] = "custom_tool_call_output";
    return turn;
}

ChatTurn parse_tool_search_output_item(const Json& item, Json& canonical) {
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get<std::string>().empty() || !item.contains("tools") ||
        !item.at("tools").is_array()) {
        bad_request("tool_search_output requires call_id and a tools array", "input");
    }
    if (!item.contains("execution") || !item.at("execution").is_string() ||
        item.at("execution").get<std::string>() != "client") {
        bad_request("tool_search_output execution must be 'client'", "input");
    }
    ChatTurn turn;
    turn.role         = ChatRole::Tool;
    turn.tool_call_id = item.at("call_id").get<std::string>();
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.type_raw = "input_text";
    content.text     = item.at("tools").dump();
    turn.content.push_back(std::move(content));
    canonical = {{"id", item_id(item, "tso", "input")},
                 {"type", "tool_search_output"},
                 {"status", "completed"},
                 {"call_id", turn.tool_call_id},
                 {"execution", "client"},
                 {"tools", item.at("tools")}};
    return turn;
}

void parse_input(const Json& input, ResponsesRequest& out) {
    Json values;
    if (input.is_string()) {
        values = Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", input}}});
    } else if (input.is_array()) {
        values = input;
    } else {
        bad_request("input must be a string or an array of Items", "input");
    }
    if (values.empty()) { bad_request("input must not be empty", "input"); }

    std::string pending_reasoning;
    bool pending_reasoning_present = false;
    bool can_group_function_calls  = false;
    std::unordered_set<std::string> ids;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const Json& item = values.at(index);
        if (!item.is_object()) {
            bad_request("input Item " + std::to_string(index) + " must be an object", "input");
        }
        std::string type;
        if (item.contains("type") && !item.at("type").is_null()) {
            if (!item.at("type").is_string()) {
                bad_request("input Item type must be a string", "input");
            }
            type = item.at("type").get<std::string>();
        } else if (item.contains("role")) {
            type = "message";
        } else {
            bad_request("input Item must contain type", "input");
        }

        Json canonical;
        if (type == "message") {
            ParsedMessage message = parse_message_item(item, index);
            if (pending_reasoning_present) {
                if (message.turn.role != ChatRole::Assistant) {
                    bad_request("a reasoning Item must be followed by an assistant output Item",
                                "input");
                }
                message.turn.reasoning_content = std::move(pending_reasoning);
                pending_reasoning.clear();
                pending_reasoning_present = false;
            }
            out.input_turns.push_back(std::move(message.turn));
            canonical                = std::move(message.canonical);
            can_group_function_calls = false;
        } else if (type == "reasoning") {
            if (pending_reasoning_present) {
                bad_request("adjacent reasoning Items are not supported", "input");
            }
            pending_reasoning         = parse_reasoning_item(item, canonical);
            pending_reasoning_present = true;
            can_group_function_calls  = false;
        } else if (type == "function_call" || type == "custom_tool_call" ||
                   type == "tool_search_call") {
            ToolCall call = type == "function_call"    ? parse_function_call_item(item, canonical)
                            : type == "custom_tool_call" ? parse_custom_tool_call_item(item, canonical)
                                                          : parse_tool_search_call_item(item, canonical);
            if (can_group_function_calls && !pending_reasoning_present &&
                !out.input_turns.empty() && out.input_turns.back().role == ChatRole::Assistant &&
                out.input_turns.back().content.empty() &&
                !out.input_turns.back().tool_calls.empty()) {
                out.input_turns.back().tool_calls.push_back(std::move(call));
            } else {
                ChatTurn turn;
                turn.role              = ChatRole::Assistant;
                turn.reasoning_content = std::move(pending_reasoning);
                pending_reasoning.clear();
                pending_reasoning_present = false;
                turn.tool_calls.push_back(std::move(call));
                out.input_turns.push_back(std::move(turn));
            }
            can_group_function_calls = true;
        } else if (type == "function_call_output" || type == "custom_tool_call_output" ||
                   type == "tool_search_output") {
            if (pending_reasoning_present) {
                bad_request("a reasoning Item must be followed by an assistant output Item",
                            "input");
            }
            out.input_turns.push_back(
                type == "function_call_output"
                    ? parse_function_call_output_item(item, canonical)
                : type == "custom_tool_call_output"
                    ? parse_custom_tool_call_output_item(item, canonical)
                    : parse_tool_search_output_item(item, canonical));
            can_group_function_calls = false;
        } else if (type == "input_file") {
            bad_request("input_file is not supported", "input", "file_inputs_not_supported");
        } else {
            bad_request("unsupported input Item type: " + type, "input", "item_type_not_supported");
        }

        const std::string id = canonical.at("id").get<std::string>();
        if (!ids.insert(id).second) { bad_request("duplicate input Item id: " + id, "input"); }
        out.input_items.push_back(std::move(canonical));
    }
    if (pending_reasoning_present) {
        bad_request("a reasoning Item must be followed by an assistant output Item", "input");
    }
}

void parse_tools(const Json& body, ResponsesRequest& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    if (!body.at("tools").is_array()) { bad_request("tools must be an array", "tools"); }
    std::unordered_set<std::string> names;
    std::unordered_set<std::string> loaded_names;
    auto add_tool = [&](const Json& item, const std::string& tool_namespace, bool is_tool_search,
                        const std::string& tool_search_execution, bool discovered = false) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must be objects with a string type", "tools");
        }
        const std::string type = item.at("type").get<std::string>();
        if (!is_tool_search && type != "function" && type != "custom") {
            bad_request("only function and custom tools are supported inside namespaces; got " +
                            type,
                        "tools",
                        "tool_type_not_supported");
        }
        ToolDefinition tool;
        tool.wire_name      = is_tool_search ? "tool_search" : require_function_name(item, "tools");
        tool.wire_namespace = tool_namespace;
        tool.name           = model_tool_name(tool_namespace, tool.wire_name);
        tool.custom         = !is_tool_search && type == "custom";
        tool.tool_search    = is_tool_search;
        tool.tool_search_execution = tool_search_execution;
        bool defer_loading  = false;
        if (item.contains("defer_loading") && !item.at("defer_loading").is_null()) {
            if (!item.at("defer_loading").is_boolean()) {
                bad_request("tool defer_loading must be a boolean", "tools");
            }
            defer_loading = item.at("defer_loading").get<bool>();
        }
        const bool already_declared = !names.insert(tool.name).second;
        if (!discovered && already_declared) {
            bad_request("duplicate or colliding model tool name: " + tool.name, "tools");
        }
        if (item.contains("description") && !item.at("description").is_null()) {
            if (!item.at("description").is_string()) {
                bad_request("tool description must be a string", "tools");
            }
            tool.description = item.at("description").get<std::string>();
        }
        if (tool.custom && item.contains("format") && !item.at("format").is_null()) {
            const Json& format = item.at("format");
            if (!format.is_object() || !format.contains("type") ||
                !format.at("type").is_string()) {
                bad_request("custom tool format must contain a string type", "tools");
            }
            if (format.at("type").get<std::string>() == "grammar") {
                if (!format.contains("syntax") || !format.at("syntax").is_string() ||
                    !format.contains("definition") || !format.at("definition").is_string()) {
                    bad_request("custom grammar format requires syntax and definition", "tools");
                }
                tool.description += "\n\nThe raw input must follow this " +
                                    format.at("syntax").get<std::string>() +
                                    " grammar exactly:\n" +
                                    format.at("definition").get<std::string>();
            } else if (format.at("type").get<std::string>() != "text") {
                bad_request("unsupported custom tool format type", "tools",
                            "tool_type_not_supported");
            }
        }
        if (tool.custom && tool.wire_name == "apply_patch") {
            tool.description +=
                "\n\nUse workspace-relative file paths in patch headers. Do not use absolute "
                "Windows drive paths or duplicate the workspace prefix.";
        }
        Json parameters = tool.custom
                              ? Json{{"type", "object"},
                                     {"properties", Json{{"input", Json{{"type", "string"}}}}},
                                     {"required", Json::array({"input"})},
                                     {"additionalProperties", false}}
                              : Json{{"type", "object"}, {"properties", Json::object()}};
        if (!tool.custom && item.contains("parameters") && !item.at("parameters").is_null()) {
            if (!item.at("parameters").is_object()) {
                bad_request("function parameters must be a JSON object", "tools");
            }
            parameters = item.at("parameters");
        }
        if (!tool.custom && item.contains("strict") && !item.at("strict").is_null()) {
            if (!item.at("strict").is_boolean()) {
                bad_request("function strict must be a boolean", "tools");
            }
        }
        tool.strict          = false;
        tool.parameters_json = parameters.dump();
        Json nested = {
            {"type", "function"},
            {"function", Json{{"name", tool.name}, {"parameters", parameters}, {"strict", false}}}};
        if (!tool.description.empty()) { nested["function"]["description"] = tool.description; }
        tool.definition_json = nested.dump();
        // Deferred tools are registry metadata for Codex tool_search. They must not be expanded
        // into the model prompt until Codex returns them in a tool_search_output.
        if (defer_loading && !discovered) { return; }
        if (!loaded_names.insert(tool.name).second) { return; }
        out.generation.tool_name_max_length =
            std::max(out.generation.tool_name_max_length, tool.name.size());
        out.generation.tools.push_back(std::move(tool));
    };

    for (const Json& item : body.at("tools")) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must be objects with a string type", "tools");
        }
        const std::string type = item.at("type").get<std::string>();
        if (type == "namespace") {
            const std::string tool_namespace = require_function_name(item, "tools");
            if (!item.contains("tools") || !item.at("tools").is_array() || item.at("tools").empty()) {
                bad_request("namespace tools must contain a non-empty tools array", "tools");
            }
            for (const Json& child : item.at("tools")) {
                add_tool(child, tool_namespace, false, {}, false);
            }
            out.tools.push_back(item);
        } else if (type == "function" || type == "custom") {
            add_tool(item, {}, false, {}, false);
            out.tools.push_back(item);
        } else if (type == "tool_search") {
            if (!item.contains("execution") || !item.at("execution").is_string() ||
                item.at("execution").get<std::string>() != "client") {
                bad_request("tool_search execution must be 'client'", "tools");
            }
            add_tool(item, {}, true, item.at("execution").get<std::string>(), false);
            out.tools.push_back(item);
        } else {
            bad_request("unsupported tool type: " + type, "tools", "tool_type_not_supported");
        }
    }

    for (const Json& input_item : out.input_items) {
        if (!input_item.is_object() || input_item.value("type", "") != "tool_search_output" ||
            !input_item.contains("tools") || !input_item.at("tools").is_array()) {
            continue;
        }
        for (const Json& discovered : input_item.at("tools")) {
            if (!discovered.is_object() || !discovered.contains("type") ||
                !discovered.at("type").is_string()) {
                bad_request("tool_search_output tools require a string type", "input");
            }
            const std::string type = discovered.at("type").get<std::string>();
            if (type == "namespace") {
                const std::string tool_namespace = require_function_name(discovered, "input");
                if (!discovered.contains("tools") || !discovered.at("tools").is_array()) {
                    bad_request("discovered namespace requires a tools array", "input");
                }
                for (const Json& child : discovered.at("tools")) {
                    add_tool(child, tool_namespace, false, {}, true);
                }
            } else if (type == "function" || type == "custom") {
                add_tool(discovered, {}, false, {}, true);
            } else {
                bad_request("unsupported discovered tool type: " + type, "input",
                            "tool_type_not_supported");
            }
        }
    }
}

void parse_tool_choice(const Json& body, ResponsesRequest& out) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) {
        out.tool_choice = "auto";
        return;
    }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "auto") {
            out.generation.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "none") {
            out.generation.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "required") {
            out.generation.tool_choice.mode = ToolChoiceMode::Required;
        } else {
            bad_request("tool_choice must be 'auto', 'none', or 'required'", "tool_choice");
        }
        out.tool_choice = value;
    } else if (choice.is_object()) {
        if (!choice.contains("type") || !choice.at("type").is_string() ||
            (choice.at("type").get<std::string>() != "function" &&
             choice.at("type").get<std::string>() != "custom")) {
            bad_request("only function and custom tool_choice objects are supported", "tool_choice",
                        "tool_type_not_supported");
        }
        const Json* function = &choice;
        if (choice.contains("function")) {
            if (!choice.at("function").is_object()) {
                bad_request("tool_choice.function must be an object", "tool_choice");
            }
            function = &choice.at("function");
        }
        out.generation.tool_choice.mode = ToolChoiceMode::Named;
        out.generation.tool_choice.name = model_tool_name(
            optional_tool_namespace(*function, "tool_choice"),
            require_function_name(*function, "tool_choice"));
        out.tool_choice                 = choice;
    } else {
        bad_request("tool_choice must be a string or object", "tool_choice");
    }
    if ((out.generation.tool_choice.mode == ToolChoiceMode::Required ||
         out.generation.tool_choice.mode == ToolChoiceMode::Named) &&
        out.generation.tools.empty()) {
        bad_request("tool_choice requires tools", "tool_choice");
    }
    if (out.generation.tool_choice.mode == ToolChoiceMode::Named) {
        const bool found = std::ranges::any_of(
            out.generation.tools, [&](const ToolDefinition& tool) {
                return tool.name == out.generation.tool_choice.name;
            });
        if (!found) {
            bad_request("tool_choice references unknown function: " +
                            out.generation.tool_choice.name,
                        "tool_choice");
        }
    }
}

void parse_reasoning(const Json& body, ResponsesRequest& out) {
    if (!body.contains("reasoning") || body.at("reasoning").is_null()) { return; }
    const Json& reasoning = body.at("reasoning");
    if (!reasoning.is_object()) { bad_request("reasoning must be an object", "reasoning"); }
    for (auto it = reasoning.begin(); it != reasoning.end(); ++it) {
        if (it.key() != "effort" && it.key() != "summary" && it.key() != "context" &&
            !it.value().is_null()) {
            bad_request("reasoning." + it.key() + " is not supported", "reasoning",
                        "reasoning_option_not_supported");
        }
    }
    for (const char* key : {"summary", "context"}) {
        if (reasoning.contains(key) && !reasoning.at(key).is_null() &&
            !reasoning.at(key).is_string()) {
            bad_request(std::string("reasoning.") + key + " must be a string or null",
                        "reasoning");
        }
    }
    if (!reasoning.contains("effort") || reasoning.at("effort").is_null()) { return; }
    if (!reasoning.at("effort").is_string()) {
        bad_request("reasoning.effort must be a string", "reasoning");
    }
    const std::string value = reasoning.at("effort").get<std::string>();
    const std::optional<RequestedReasoningEffort> effort = parse_requested_reasoning_effort(value);
    if (!effort) {
        bad_request("reasoning.effort must be one of none, minimal, low, medium, high, xhigh, or "
                    "max",
                    "reasoning");
    }
    out.generation.reasoning_effort       = *effort;
    out.generation.reasoning_effort_param = "reasoning.effort";
}

void validate_metadata(const Json& body, ResponsesRequest& out) {
    if (!body.contains("metadata") || body.at("metadata").is_null()) { return; }
    if (!body.at("metadata").is_object()) { bad_request("metadata must be an object", "metadata"); }
    if (body.at("metadata").size() > 16) {
        bad_request("metadata supports at most 16 entries", "metadata");
    }
    for (auto it = body.at("metadata").begin(); it != body.at("metadata").end(); ++it) {
        if (it.key().size() > 64 || !it.value().is_string() ||
            it.value().get_ref<const std::string&>().size() > 512) {
            bad_request("metadata keys must be at most 64 characters and string values at most "
                        "512 characters",
                        "metadata");
        }
    }
    out.metadata = body.at("metadata");
}

void reject_unknown_top_level(const Json& body) {
    static const std::unordered_set<std::string> allowed = {
        "background",
        "chat_template_kwargs",
        "client_metadata",
        "context_management",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "moderation",
        "parallel_tool_calls",
        "previous_response_id",
        "preserve_thinking",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    };
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (!allowed.contains(it.key())) {
            bad_request("unknown parameter: " + it.key(), it.key(), "unknown_parameter");
        }
    }
}

void reject_server_managed_features(const Json& body) {
    for (const char* key : {"context_management", "conversation", "max_tool_calls", "moderation",
                            "prompt", "prompt_cache_options",
                            "prompt_cache_retention", "safety_identifier", "user"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            bad_request(std::string(key) + " is not supported", key, "parameter_not_supported");
        }
    }
    if (body.contains("background") && !body.at("background").is_null()) {
        if (!body.at("background").is_boolean()) {
            bad_request("background must be a boolean", "background");
        }
        if (body.at("background").get<bool>()) {
            bad_request("background responses are not supported", "background",
                        "background_not_supported");
        }
    }
    if (body.contains("client_metadata") && !body.at("client_metadata").is_null() &&
        !body.at("client_metadata").is_object()) {
        bad_request("client_metadata must be an object or null", "client_metadata");
    }
    if (body.contains("prompt_cache_key") && !body.at("prompt_cache_key").is_null() &&
        !body.at("prompt_cache_key").is_string()) {
        bad_request("prompt_cache_key must be a string or null", "prompt_cache_key");
    }
    if (body.contains("include") && !body.at("include").is_null()) {
        if (!body.at("include").is_array()) { bad_request("include must be an array", "include"); }
        for (const Json& value : body.at("include")) {
            if (!value.is_string()) { bad_request("include entries must be strings", "include"); }
        }
    }
    if (body.contains("parallel_tool_calls") && !body.at("parallel_tool_calls").is_null()) {
        if (!body.at("parallel_tool_calls").is_boolean()) {
            bad_request("parallel_tool_calls must be a boolean", "parallel_tool_calls");
        }
    }
    if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        const std::optional<int> value = optional_int(body, "top_logprobs");
        if (!value || *value != 0) {
            bad_request("top_logprobs is not supported", "top_logprobs", "logprobs_not_supported");
        }
    }
    if (body.contains("truncation") && !body.at("truncation").is_null()) {
        if (!body.at("truncation").is_string() ||
            body.at("truncation").get<std::string>() != "disabled") {
            bad_request("only truncation 'disabled' is supported", "truncation",
                        "truncation_not_supported");
        }
    }
    if (body.contains("service_tier") && !body.at("service_tier").is_null()) {
        if (!body.at("service_tier").is_string()) {
            bad_request("service_tier must be a string", "service_tier");
        }
        const std::string tier = body.at("service_tier").get<std::string>();
        if (tier != "auto" && tier != "default") {
            bad_request("only service_tier 'auto' or 'default' is supported", "service_tier",
                        "service_tier_not_supported");
        }
    }
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        if (!body.at("stream_options").is_object()) {
            bad_request("stream_options must be an object", "stream_options");
        }
        for (auto it = body.at("stream_options").begin(); it != body.at("stream_options").end();
             ++it) {
            if (it.key() != "include_obfuscation" || !it.value().is_boolean() ||
                it.value().get<bool>()) {
                bad_request("stream_options only supports include_obfuscation=false",
                            "stream_options", "stream_option_not_supported");
            }
        }
    }
    if (body.contains("text") && !body.at("text").is_null()) {
        if (!body.at("text").is_object()) { bad_request("text must be an object", "text"); }
        for (auto it = body.at("text").begin(); it != body.at("text").end(); ++it) {
            if (it.key() != "format" && it.key() != "verbosity" && !it.value().is_null()) {
                bad_request("text." + it.key() + " is not supported", "text",
                            "text_option_not_supported");
            }
        }
        if (body.at("text").contains("verbosity") &&
            !body.at("text").at("verbosity").is_null() &&
            !body.at("text").at("verbosity").is_string()) {
            bad_request("text.verbosity must be a string or null", "text");
        }
        if (body.at("text").contains("format") && !body.at("text").at("format").is_null()) {
            const Json& format = body.at("text").at("format");
            if (!format.is_object() || !format.contains("type") || !format.at("type").is_string() ||
                format.at("type").get<std::string>() != "text") {
                bad_request("only text.format {type:'text'} is supported", "text",
                            "structured_outputs_not_supported");
            }
            for (auto it = format.begin(); it != format.end(); ++it) {
                if (it.key() != "type") {
                    bad_request("only text.format {type:'text'} is supported", "text",
                                "structured_outputs_not_supported");
                }
            }
        }
    }
}

ResponsesRequest parse_request_impl(const Json& body, const RequestLimits& limits) {
    require_object(body);
    reject_unknown_top_level(body);
    reject_server_managed_features(body);

    ResponsesRequest out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.generation.model = body.at("model").get<std::string>();
    if (!body.contains("input")) { bad_request("missing required field: input", "input"); }
    parse_input(body.at("input"), out);

    if (body.contains("instructions") && !body.at("instructions").is_null()) {
        if (!body.at("instructions").is_string()) {
            bad_request("instructions must be a string", "instructions");
        }
        out.instructions = body.at("instructions").get<std::string>();
    }
    if (body.contains("previous_response_id") && !body.at("previous_response_id").is_null()) {
        if (!body.at("previous_response_id").is_string() ||
            body.at("previous_response_id").get<std::string>().empty()) {
            bad_request("previous_response_id must be a non-empty string", "previous_response_id");
        }
        out.previous_response_id = body.at("previous_response_id").get<std::string>();
    }

    out.store             = optional_bool(body, "store", true);
    out.stream            = optional_bool(body, "stream", false);
    out.generation.stream = out.stream;
    out.generation.parallel_tool_calls = optional_bool(body, "parallel_tool_calls", true);
    validate_metadata(body, out);
    parse_tools(body, out);
    parse_tool_choice(body, out);
    parse_reasoning(body, out);
    out.generation.preserve_thinking = parse_openai_preserve_thinking(body);

    if (const std::optional<double> temperature = optional_number(body, "temperature")) {
        if (*temperature < 0.0 || *temperature > 2.0) {
            bad_request("temperature must be in [0,2]", "temperature");
        }
        out.generation.sampling.temperature = *temperature;
    }
    if (const std::optional<double> top_p = optional_number(body, "top_p")) {
        if (*top_p < 0.0 || *top_p > 1.0) { bad_request("top_p must be in [0,1]", "top_p"); }
        out.generation.sampling.top_p = *top_p;
    }

    if (const std::optional<int> max_output = optional_int(body, "max_output_tokens")) {
        if (*max_output < 16) {
            bad_request("max_output_tokens must be at least 16", "max_output_tokens",
                        "invalid_value");
        }
        out.generation.max_tokens     = *max_output;
        out.generation.max_tokens_set = true;
    } else {
        out.generation.max_tokens     = limits.default_max_tokens;
        out.generation.max_tokens_set = false;
    }
    out.generation.messages = out.input_turns;
    return out;
}

std::string random_id(const char* prefix) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 48> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx%016llx",
                  static_cast<unsigned long long>(distribution(rng)),
                  static_cast<unsigned long long>(distribution(rng)));
    return std::string(prefix) + "_" + buffer.data();
}

std::int64_t completion_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string response_status(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "incomplete";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
        return "completed";
    }
    return "failed";
}

struct ItemIds {
    std::string reasoning;
    std::string message;
    std::vector<std::string> function_calls;
};

Json response_common(const std::string& id, std::int64_t created_at,
                     const ResponsesRequest& request, const ResponsesRuntimeValues& runtime) {
    const Json reasoning = {
        {"effort", request.generation.reasoning_effort
                       ? Json(requested_reasoning_effort_name(*request.generation.reasoning_effort))
                       : Json(nullptr)},
        {"summary", nullptr}};
    return Json{
        {"id", id},
        {"object", "response"},
        {"created_at", created_at},
        {"background", false},
        {"instructions", request.instructions ? Json(*request.instructions) : Json(nullptr)},
        {"max_output_tokens", request.generation.max_tokens},
        {"max_tool_calls", nullptr},
        {"metadata", request.metadata},
        {"model", request.generation.model},
        {"parallel_tool_calls", request.generation.parallel_tool_calls},
        {"previous_response_id",
         request.previous_response_id ? Json(*request.previous_response_id) : Json(nullptr)},
        {"reasoning", reasoning},
        {"service_tier", "default"},
        {"store", request.store},
        {"temperature", runtime.temperature},
        {"text", Json{{"format", Json{{"type", "text"}}}}},
        {"tool_choice", request.tool_choice},
        {"tools", request.tools},
        {"top_logprobs", 0},
        {"top_p", runtime.top_p},
        {"truncation", "disabled"}};
}

BuiltResponse build_response(const std::string& id, std::int64_t created_at,
                             const ResponsesRequest& request, const ResponsesRuntimeValues& runtime,
                             const GenerationOutcome& outcome, ItemIds ids) {
    BuiltResponse built;
    const std::string status      = response_status(outcome.finish_reason);
    const std::string item_status = status == "completed" ? "completed" : "incomplete";

    if (!outcome.text.empty() || outcome.tool_calls.empty()) {
        if (ids.message.empty()) { ids.message = new_response_item_id("msg"); }
        built.output_items.push_back(
            Json{{"id", ids.message},
                 {"type", "message"},
                 {"status", item_status},
                 {"role", "assistant"},
                 {"content", Json::array({Json{{"type", "output_text"},
                                               {"annotations", Json::array()},
                                               {"text", outcome.text}}})}});
    }

    ids.function_calls.resize(outcome.tool_calls.size());
    for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
        if (ids.function_calls[index].empty()) {
            ids.function_calls[index] = new_response_item_id("fc");
        }
        const ToolCall& call = outcome.tool_calls[index];
        const char* call_type = call.tool_search ? "tool_search_call"
                                : call.custom    ? "custom_tool_call"
                                                 : "function_call";
        Json item = {{"id", ids.function_calls[index]},
                     {"type", call_type},
                     {"status", "completed"},
                     {"call_id", call.id}};
        if (call.tool_search) {
            Json arguments = Json::parse(call.arguments_json, nullptr, false);
            if (arguments.is_discarded()) { arguments = Json::object(); }
            item["execution"] = call.tool_search_execution.empty() ? "client"
                                                                    : call.tool_search_execution;
            item["arguments"] = std::move(arguments);
        } else if (call.custom) {
            item["name"]  = call.name;
            item["input"] = call.custom_input;
        } else {
            item["name"]      = call.name;
            item["arguments"] = call.arguments_json;
        }
        if (!call.namespace_name.empty()) { item["namespace"] = call.namespace_name; }
        built.output_items.push_back(std::move(item));
    }

    ChatTurn history;
    history.role              = ChatRole::Assistant;
    history.reasoning_content = outcome.reasoning;
    history.tool_calls        = outcome.tool_calls;
    if (!outcome.text.empty()) {
        ContentPart part;
        part.kind     = ContentKind::Text;
        part.type_raw = "output_text";
        part.text     = outcome.text;
        history.content.push_back(std::move(part));
    }
    built.output_history.push_back(std::move(history));

    Json response            = response_common(id, created_at, request, runtime);
    response["status"]       = status;
    response["completed_at"] = completion_time_now();
    response["error"]        = nullptr;
    response["output"]       = built.output_items;
    response["incomplete_details"] =
        status == "incomplete" ? Json{{"reason", "max_output_tokens"}} : Json(nullptr);
    const int observed_cached = std::max(runtime.cached_input_tokens,
                                         static_cast<int>(outcome.metrics.prefix_cache_hit_tokens));
    const int cached_tokens   = std::clamp(observed_cached, 0, outcome.prompt_tokens);
    response["usage"] =
        Json{{"input_tokens", outcome.prompt_tokens},
             {"input_tokens_details", Json{{"cached_tokens", cached_tokens}}},
             {"output_tokens", outcome.completion_tokens},
             {"output_tokens_details", Json{{"reasoning_tokens", outcome.reasoning_tokens}}},
             {"total_tokens", outcome.prompt_tokens + outcome.completion_tokens}};
    built.body = std::move(response);
    return built;
}

std::string sse(const Json& event) {
    return "event: " + event.at("type").get<std::string>() + "\n" + "data: " + event.dump() +
           "\n\n";
}

Json in_progress_response(const std::string& id, std::int64_t created_at,
                          const ResponsesRequest& request, const ResponsesRuntimeValues& runtime) {
    Json response                  = response_common(id, created_at, request, runtime);
    response["status"]             = "in_progress";
    response["completed_at"]       = nullptr;
    response["error"]              = nullptr;
    response["incomplete_details"] = nullptr;
    response["output"]             = Json::array();
    response["usage"]              = nullptr;
    return response;
}

} // namespace

ResponsesRequest parse_responses_request(const Json& body, const RequestLimits& limits) {
    return parse_request_impl(body, limits);
}

ResponsesRequest parse_response_input_tokens_request(const Json& body,
                                                     const RequestLimits& limits) {
    require_object(body);
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (it.key() != "model" && it.key() != "input" && it.key() != "chat_template_kwargs" &&
            it.key() != "preserve_thinking") {
            bad_request("unknown parameter: " + it.key(), it.key(), "unknown_parameter");
        }
    }
    ResponsesRequest parsed  = parse_request_impl(body, limits);
    parsed.store             = false;
    parsed.stream            = false;
    parsed.generation.stream = false;
    return parsed;
}

ResponsesCompactRequest parse_responses_compact_request(const Json& body,
                                                        const RequestLimits& limits) {
    require_object(body);
    static const std::unordered_set<std::string> allowed = {
        "input",          "instructions",       "model",     "parallel_tool_calls",
        "prompt_cache_key", "reasoning",          "service_tier", "text",
        "tools",
    };
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (!allowed.contains(it.key())) {
            bad_request("unknown parameter: " + it.key(), it.key(), "unknown_parameter");
        }
    }
    if (!body.contains("input") || !body.at("input").is_array() ||
        body.at("input").empty()) {
        bad_request("input must be a non-empty array", "input");
    }
    if (!body.contains("parallel_tool_calls") ||
        !body.at("parallel_tool_calls").is_boolean()) {
        bad_request("parallel_tool_calls must be a boolean", "parallel_tool_calls");
    }
    if (body.contains("tools") && !body.at("tools").is_null() &&
        !body.at("tools").is_array()) {
        bad_request("tools must be an array", "tools");
    }

    // Compact requests carry the current model-visible tool registry so the remote service can
    // understand history. Local summarization needs the history, but must never call those tools.
    Json generation_body = body;
    generation_body.erase("tools");
    ResponsesRequest parsed = parse_request_impl(generation_body, limits);
    compose_responses_generation_messages(parsed, {});

    ChatTurn prompt;
    prompt.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.type_raw = "input_text";
    content.text     = std::string(kCompactionPrompt);
    prompt.content.push_back(std::move(content));
    parsed.generation.messages.push_back(std::move(prompt));

    ResponsesCompactRequest compact;
    compact.generation = std::move(parsed.generation);
    compact.generation.stream = false;
    compact.generation.tools.clear();
    compact.generation.tool_choice.mode = ToolChoiceMode::None;
    compact.generation.parallel_tool_calls = false;
    return compact;
}

std::string make_responses_compact_body(const GenerationOutcome& outcome) {
    if ((outcome.finish_reason != ninfer::FinishReason::StopToken &&
         outcome.finish_reason != ninfer::FinishReason::StopString) ||
        outcome.text.empty() || !outcome.tool_calls.empty()) {
        ApiError error;
        error.status  = 500;
        error.type    = "server_error";
        error.code    = "compaction_generation_failed";
        error.message = "compaction generation did not produce a complete plaintext summary";
        throw ApiException(std::move(error));
    }
    const std::string summary = std::string(kCompactionSummaryPrefix) + "\n" + outcome.text;
    const Json item = {{"type", "message"},
                       {"role", "user"},
                       {"content", Json::array({Json{{"type", "input_text"},
                                                       {"text", summary}}})}};
    return Json{{"output", Json::array({item})}}.dump();
}

void inherit_responses_preserve_thinking(ResponsesRequest& request, bool parent_value) {
    if (request.generation.preserve_thinking) {
        request.generation.preserve_thinking_semantic_change =
            *request.generation.preserve_thinking != parent_value;
        return;
    }
    request.generation.preserve_thinking = parent_value;
}

void compose_responses_generation_messages(ResponsesRequest& request,
                                           const std::vector<ChatTurn>& previous_context) {
    std::vector<ChatTurn> messages;
    messages.reserve((request.instructions ? 1U : 0U) + previous_context.size() +
                     request.input_turns.size());
    if (request.instructions) {
        ChatTurn instructions;
        instructions.role = ChatRole::Developer;
        ContentPart part;
        part.kind     = ContentKind::Text;
        part.type_raw = "input_text";
        part.text     = *request.instructions;
        instructions.content.push_back(std::move(part));
        messages.push_back(std::move(instructions));
    }
    messages.insert(messages.end(), previous_context.begin(), previous_context.end());
    messages.insert(messages.end(), request.input_turns.begin(), request.input_turns.end());
    request.generation.messages = std::move(messages);
}

BuiltResponse make_response_object(const std::string& id, std::int64_t created_at,
                                   const ResponsesRequest& request,
                                   const ResponsesRuntimeValues& runtime,
                                   const GenerationOutcome& outcome) {
    return build_response(id, created_at, request, runtime, outcome, {});
}

std::string make_response_input_tokens_body(int input_tokens) {
    return Json{{"object", "response.input_tokens"}, {"input_tokens", input_tokens}}.dump();
}

class ResponsesEventStream::Impl {
public:
    Impl(std::string response_id, std::int64_t created_at_, ResponsesRequest request_,
         ResponsesRuntimeValues runtime_)
        : id(std::move(response_id)), created_at(created_at_), request(std::move(request_)),
          runtime(runtime_) {}

    Json event(std::string type, Json fields = Json::object()) {
        fields["type"]            = std::move(type);
        fields["sequence_number"] = sequence++;
        return fields;
    }

    std::vector<std::string> ensure_reasoning() {
        if (reasoning_started) { return {}; }
        reasoning_started = true;
        ids.reasoning     = new_response_item_id("rs");
        reasoning_index   = next_output_index++;
        const Json item   = {{"id", ids.reasoning},
                             {"type", "reasoning"},
                             {"status", "in_progress"},
                             {"summary", Json::array()},
                             {"content", Json::array()}};
        const Json part   = {{"type", "reasoning_text"}, {"text", ""}};
        return {sse(event("response.output_item.added",
                          Json{{"output_index", reasoning_index}, {"item", item}})),
                sse(event("response.content_part.added", Json{{"item_id", ids.reasoning},
                                                              {"output_index", reasoning_index},
                                                              {"content_index", 0},
                                                              {"part", part}}))};
    }

    std::vector<std::string> close_reasoning(const std::string& final_text,
                                             const char* item_status = "completed") {
        if (!reasoning_started || reasoning_done) { return {}; }
        reasoning_done  = true;
        reasoning_text  = final_text;
        const Json part = {{"type", "reasoning_text"}, {"text", reasoning_text}};
        const Json item = {{"id", ids.reasoning},
                           {"type", "reasoning"},
                           {"status", item_status},
                           {"summary", Json::array()},
                           {"content", Json::array({part})}};
        return {sse(event("response.reasoning_text.done", Json{{"item_id", ids.reasoning},
                                                               {"output_index", reasoning_index},
                                                               {"content_index", 0},
                                                               {"text", reasoning_text}})),
                sse(event("response.content_part.done", Json{{"item_id", ids.reasoning},
                                                             {"output_index", reasoning_index},
                                                             {"content_index", 0},
                                                             {"part", part}})),
                sse(event("response.output_item.done",
                          Json{{"output_index", reasoning_index}, {"item", item}}))};
    }

    std::vector<std::string> ensure_message() {
        if (message_started) { return {}; }
        message_started = true;
        ids.message     = new_response_item_id("msg");
        message_index   = next_output_index++;
        const Json item = {{"id", ids.message},
                           {"type", "message"},
                           {"status", "in_progress"},
                           {"role", "assistant"},
                           {"content", Json::array()}};
        const Json part = {{"type", "output_text"}, {"annotations", Json::array()}, {"text", ""}};
        return {sse(event("response.output_item.added",
                          Json{{"output_index", message_index}, {"item", item}})),
                sse(event("response.content_part.added", Json{{"item_id", ids.message},
                                                              {"output_index", message_index},
                                                              {"content_index", 0},
                                                              {"part", part}}))};
    }

    std::vector<std::string> close_message(const std::string& final_text,
                                           const char* item_status = "completed") {
        if (!message_started || message_done) { return {}; }
        message_done    = true;
        content_text    = final_text;
        const Json part = {
            {"type", "output_text"}, {"annotations", Json::array()}, {"text", content_text}};
        const Json item = {{"id", ids.message},
                           {"type", "message"},
                           {"status", item_status},
                           {"role", "assistant"},
                           {"content", Json::array({part})}};
        return {sse(event("response.output_text.done", Json{{"item_id", ids.message},
                                                            {"output_index", message_index},
                                                            {"content_index", 0},
                                                            {"text", content_text},
                                                            {"logprobs", Json::array()}})),
                sse(event("response.content_part.done", Json{{"item_id", ids.message},
                                                             {"output_index", message_index},
                                                             {"content_index", 0},
                                                             {"part", part}})),
                sse(event("response.output_item.done",
                          Json{{"output_index", message_index}, {"item", item}}))};
    }

    std::string id;
    std::int64_t created_at = 0;
    ResponsesRequest request;
    ResponsesRuntimeValues runtime;
    std::uint64_t sequence = 0;
    int next_output_index  = 0;
    int reasoning_index    = -1;
    int message_index      = -1;
    bool started           = false;
    bool reasoning_started = false;
    bool reasoning_done    = false;
    bool message_started   = false;
    bool message_done      = false;
    bool finish_built      = false;
    bool terminal_emitted  = false;
    std::string reasoning_text;
    std::string content_text;
    ItemIds ids;
};

ResponsesEventStream::ResponsesEventStream(std::string response_id, std::int64_t created_at,
                                           ResponsesRequest request, ResponsesRuntimeValues runtime)
    : impl_(std::make_unique<Impl>(std::move(response_id), created_at, std::move(request),
                                   runtime)) {}

ResponsesEventStream::~ResponsesEventStream()                                          = default;
ResponsesEventStream::ResponsesEventStream(ResponsesEventStream&&) noexcept            = default;
ResponsesEventStream& ResponsesEventStream::operator=(ResponsesEventStream&&) noexcept = default;

std::vector<std::string> ResponsesEventStream::start() {
    if (impl_->started) { throw std::logic_error("Responses event stream already started"); }
    impl_->started = true;
    const Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    return {sse(impl_->event("response.created", Json{{"response", response}})),
            sse(impl_->event("response.in_progress", Json{{"response", response}}))};
}

std::vector<std::string> ResponsesEventStream::reasoning_delta(const std::string& text) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid reasoning delta event state");
    }
    if (text.empty()) { return {}; }
    impl_->reasoning_text += text;
    return {};
}

std::vector<std::string> ResponsesEventStream::content_delta(const std::string& text) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid content delta event state");
    }
    if (text.empty()) { return {}; }
    std::vector<std::string> events;
    std::vector<std::string> added = impl_->ensure_message();
    events.insert(events.end(), std::make_move_iterator(added.begin()),
                  std::make_move_iterator(added.end()));
    impl_->content_text += text;
    events.push_back(
        sse(impl_->event("response.output_text.delta", Json{{"item_id", impl_->ids.message},
                                                            {"output_index", impl_->message_index},
                                                            {"content_index", 0},
                                                            {"delta", text},
                                                            {"logprobs", Json::array()}})));
    return events;
}

ResponsesStreamFinish ResponsesEventStream::finish(const GenerationOutcome& outcome) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid Responses stream finish state");
    }
    impl_->finish_built = true;
    ResponsesStreamFinish finished;
    const std::string status = response_status(outcome.finish_reason);
    const char* item_status  = status == "completed" ? "completed" : "incomplete";

    auto append = [&](std::vector<std::string> events) {
        finished.events_before_terminal.insert(finished.events_before_terminal.end(),
                                               std::make_move_iterator(events.begin()),
                                               std::make_move_iterator(events.end()));
    };
    const bool needs_message = !outcome.text.empty() || outcome.tool_calls.empty();
    if (needs_message) {
        append(impl_->ensure_message());
        if (outcome.text != impl_->content_text) {
            if (!outcome.text.starts_with(impl_->content_text)) {
                throw std::logic_error("streamed content does not match terminal content");
            }
            const std::string suffix = outcome.text.substr(impl_->content_text.size());
            if (!suffix.empty()) {
                impl_->content_text += suffix;
                finished.events_before_terminal.push_back(sse(impl_->event(
                    "response.output_text.delta", Json{{"item_id", impl_->ids.message},
                                                       {"output_index", impl_->message_index},
                                                       {"content_index", 0},
                                                       {"delta", suffix},
                                                       {"logprobs", Json::array()}})));
            }
        }
        append(impl_->close_message(outcome.text, item_status));
    }

    impl_->ids.function_calls.reserve(outcome.tool_calls.size());
    for (const ToolCall& call : outcome.tool_calls) {
        const std::string item_id =
            new_response_item_id(call.tool_search ? "ts" : call.custom ? "ctc" : "fc");
        impl_->ids.function_calls.push_back(item_id);
        const int output_index = impl_->next_output_index++;
        const char* call_type = call.tool_search ? "tool_search_call"
                                : call.custom    ? "custom_tool_call"
                                                 : "function_call";
        Json added_item = {{"id", item_id},
                           {"type", call_type},
                           {"status", "in_progress"},
                           {"call_id", call.id}};
        if (call.tool_search) {
            added_item["execution"] = call.tool_search_execution.empty()
                                           ? "client"
                                           : call.tool_search_execution;
            added_item["arguments"] = Json::object();
        } else {
            added_item["name"] = call.name;
            added_item[call.custom ? "input" : "arguments"] = "";
        }
        if (!call.namespace_name.empty()) { added_item["namespace"] = call.namespace_name; }
        finished.events_before_terminal.push_back(
            sse(impl_->event("response.output_item.added",
                             Json{{"output_index", output_index}, {"item", added_item}})));
        if (!call.custom && !call.tool_search && !call.arguments_json.empty()) {
            finished.events_before_terminal.push_back(sse(impl_->event(
                "response.function_call_arguments.delta", Json{{"item_id", item_id},
                                                               {"output_index", output_index},
                                                               {"delta", call.arguments_json}})));
        }
        if (!call.custom && !call.tool_search) {
            Json done = {{"item_id", item_id},
                         {"output_index", output_index},
                         {"name", call.name},
                         {"arguments", call.arguments_json}};
            if (!call.namespace_name.empty()) { done["namespace"] = call.namespace_name; }
            finished.events_before_terminal.push_back(
                sse(impl_->event("response.function_call_arguments.done", std::move(done))));
        }
        Json search_arguments = Json::parse(call.arguments_json, nullptr, false);
        if (search_arguments.is_discarded()) { search_arguments = Json::object(); }
        Json done_item = {{"id", item_id},
                          {"type", call_type},
                          {"status", "completed"},
                          {"call_id", call.id}};
        if (call.tool_search) {
            done_item["execution"] = call.tool_search_execution.empty()
                                          ? "client"
                                          : call.tool_search_execution;
            done_item["arguments"] = std::move(search_arguments);
        } else {
            done_item["name"] = call.name;
            done_item[call.custom ? "input" : "arguments"] =
                call.custom ? call.custom_input : call.arguments_json;
        }
        if (!call.namespace_name.empty()) { done_item["namespace"] = call.namespace_name; }
        finished.events_before_terminal.push_back(
            sse(impl_->event("response.output_item.done",
                             Json{{"output_index", output_index}, {"item", done_item}})));
    }

    finished.response = build_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime,
                                       outcome, impl_->ids);
    return finished;
}

std::string ResponsesEventStream::terminal(const BuiltResponse& response) {
    if (!impl_->finish_built || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses terminal event state");
    }
    impl_->terminal_emitted  = true;
    const std::string status = response.body.at("status").get<std::string>();
    const std::string type   = status == "completed"    ? "response.completed"
                               : status == "incomplete" ? "response.incomplete"
                               : status == "cancelled"  ? "response.cancelled"
                                                        : "response.failed";
    return sse(impl_->event(type, Json{{"response", response.body}}));
}

std::string ResponsesEventStream::failed(const ApiError& error) {
    if (!impl_->started || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses failed event state");
    }
    impl_->terminal_emitted = true;
    Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    response["status"]       = "failed";
    response["completed_at"] = completion_time_now();
    response["error"]        = Json{{"code", error.code.empty() ? Json(nullptr) : Json(error.code)},
                                    {"message", error.message}};
    return sse(impl_->event("response.failed", Json{{"response", response}}));
}

std::string new_response_id() { return random_id("resp"); }

std::string new_response_item_id(const char* prefix) { return random_id(prefix); }

} // namespace ninfer::serve
