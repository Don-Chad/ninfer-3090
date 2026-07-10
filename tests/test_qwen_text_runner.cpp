#include "qus/text/text_runner.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check(bool condition, const char* message) { return condition ? 0 : fail(message); }

std::string minimal_tokenizer_json() {
    return R"({"model":{"type":"BPE","vocab":{"a":0,"b":1}},"added_tokens":[]})";
}

qus::text::QwenTokenizer make_tokenizer_with_stop_ids(const std::vector<int>& ids) {
    std::string config = R"({"eos_token_id":[)";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) { config += ","; }
        config += std::to_string(ids[i]);
    }
    config += "]}";
    qus::Q5090TokenizerBundle bundle;
    bundle.tokenizer_json         = minimal_tokenizer_json();
    bundle.merges_txt             = "#version: 0.2\n";
    bundle.generation_config_json = std::move(config);
    return qus::text::QwenTokenizer(std::move(bundle));
}

bool throws_invalid_argument_for_negative_override(const qus::text::QwenTokenizer& tokenizer) {
    try {
        (void)qus::text::resolve_stop_token_ids(tokenizer, {4, -1});
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int test_resolve_stop_token_ids() {
    const qus::text::QwenTokenizer tokenizer = make_tokenizer_with_stop_ids({248046, 248044});

    int failures = 0;
    failures +=
        check(qus::text::resolve_stop_token_ids(tokenizer, {}) == std::vector<int>{248046, 248044},
              "empty stop id overrides did not use tokenizer defaults");
    failures +=
        check(qus::text::resolve_stop_token_ids(tokenizer, {9, 9, 8}) == std::vector<int>{9, 8},
              "stop id overrides were not deduplicated in first-occurrence order");
    failures += check(throws_invalid_argument_for_negative_override(tokenizer),
                      "negative stop id override did not throw invalid_argument");
    return failures;
}

int test_decode_stop_trimming_modes() {
    const qus::text::QwenTokenizer tokenizer = make_tokenizer_with_stop_ids({1});
    const std::vector<int> generated{0, 1};

    int failures = 0;
    failures += check(tokenizer.decode(generated, qus::text::DecodeOptions{false, {}}) == "ab",
                      "raw decode did not preserve terminal stop id");
    failures += check(tokenizer.decode(generated, qus::text::DecodeOptions{true, {1}}) == "a",
                      "clean decode did not trim terminal stop id");
    return failures;
}

int test_stream_callback_api_can_capture_chunks() {
    qus::text::TextGenerationOptions options;
    std::vector<std::string> pieces;
    std::vector<qus::text::TextChannel> channels;
    options.stream_callback = [&](const qus::text::TextStreamChunk& chunk) {
        pieces.push_back(chunk.text);
        channels.push_back(chunk.channel);
    };

    options.stream_callback(
        qus::text::TextStreamChunk{.text = "why", .channel = qus::text::TextChannel::Reasoning});
    options.stream_callback(
        qus::text::TextStreamChunk{.text = "hi", .channel = qus::text::TextChannel::Content});

    int failures = 0;
    failures += check(pieces == std::vector<std::string>{"why", "hi"},
                      "stream callback did not capture text chunks");
    failures +=
        check(channels == std::vector<qus::text::TextChannel>{qus::text::TextChannel::Reasoning,
                                                              qus::text::TextChannel::Content},
              "stream callback did not capture channels");

    qus::text::TextGenerationResult result;
    result.timings.render_tokenize_seconds = 0.25;
    result.timings.prefill_seconds         = 1.0;
    result.timings.decode_seconds          = 2.0;
    result.timings.total_seconds           = 3.25;
    failures += check(result.timings.total_seconds == 3.25, "timing result API mismatch");
    return failures;
}

} // namespace

int main() {
    try {
        int failures = 0;
        failures += test_resolve_stop_token_ids();
        failures += test_decode_stop_trimming_modes();
        failures += test_stream_callback_api_can_capture_chunks();
        return failures == 0 ? 0 : fail("qwen text runner test failed");
    } catch (const std::exception& ex) {
        std::cerr << "qwen text runner test failed: " << ex.what() << '\n';
        return 1;
    }
}
