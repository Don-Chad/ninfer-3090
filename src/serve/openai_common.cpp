#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <string_view>
#include <utility>

namespace ninfer::serve {

namespace {

std::string chat_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx",
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + buffer.data();
}

std::string responses_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 48> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx%016llx",
                  static_cast<unsigned long long>(distribution(random)),
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + "_" + buffer.data();
}

} // namespace

using Json = nlohmann::json;

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len) {
    // vLLM/llama.cpp-compatible discovery metadata for the configured per-request context limit.
    const Json payload = {{"object", "list"},
                          {"data", Json::array({Json{{"id", model_id},
                                                     {"object", "model"},
                                                     {"created", created},
                                                     {"owned_by", "ninfer"},
                                                     {"max_model_len", max_model_len}}})}};
    return payload.dump();
}

std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len) {
    // vLLM/llama.cpp-compatible discovery metadata for the configured per-request context limit.
    const Json payload = {{"id", model_id},
                          {"object", "model"},
                          {"created", created},
                          {"owned_by", "ninfer"},
                          {"max_model_len", max_model_len}};
    return payload.dump();
}

std::string make_error_body(const ApiError& error) {
    Json rendered     = {{"message", error.message}, {"type", error.type}};
    rendered["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    rendered["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", std::move(rendered)}}.dump();
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void validate_openai_model(std::string_view requested, std::string_view available) {
    if (requested == available) { return; }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "model";
    error.code    = "model_not_found";
    error.message = "model '" + std::string(requested) + "' not found";
    throw ApiException(std::move(error));
}

std::string new_openai_chat_completion_id() { return chat_identifier("chatcmpl-"); }

std::string new_openai_chat_tool_call_id() { return chat_identifier("call_"); }

std::string new_openai_request_id() { return responses_identifier("req"); }

std::string new_openai_response_id() { return responses_identifier("resp"); }

std::string new_openai_response_item_id(std::string_view prefix) {
    return responses_identifier(prefix);
}

} // namespace ninfer::serve
