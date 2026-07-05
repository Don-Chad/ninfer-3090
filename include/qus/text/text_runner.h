#pragma once

#include "qus/runtime/engine.h"
#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <functional>
#include <string>
#include <vector>

namespace qus::text {

struct TextStreamChunk {
    int token_id = 0;
    std::string text;
};

using TextStreamCallback = std::function<void(const TextStreamChunk&)>;

struct TextGenerationTimings {
    double render_tokenize_seconds = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    double total_seconds = 0.0;
};

struct TextGenerationOptions {
    int max_new_tokens = 128;
    bool raw_output = false;
    bool enable_thinking = false;
    std::vector<int> stop_token_ids;
    TextStreamCallback stream_callback;
};

struct TextGenerationResult {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string text;
    TextGenerationTimings timings;
};

class TextGenerationRunner {
public:
    TextGenerationRunner(QwenTokenizer& tokenizer, qus::Engine& engine);

    TextGenerationResult generate(const std::vector<ChatMessage>& messages,
                                  const TextGenerationOptions& options);

private:
    QwenTokenizer& tokenizer_;
    qus::Engine& engine_;
};

std::vector<int> resolve_stop_token_ids(const QwenTokenizer& tokenizer,
                                        const std::vector<int>& overrides);

} // namespace qus::text
