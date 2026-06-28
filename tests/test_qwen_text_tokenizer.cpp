#include "qus/text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path repo_file(const char* relative) {
    return std::filesystem::path(QUS_SOURCE_DIR) / relative;
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check(bool condition, const char* message) { return condition ? 0 : fail(message); }

int test_load_real_tokenizer_metadata() {
    (void)repo_file("tests/fixtures/text/qwen36_text_golden.json");

    const std::filesystem::path tokenizer_dir =
        "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16";
    if (!std::filesystem::exists(tokenizer_dir / "tokenizer.json")) {
        std::cout << "SKIP qwen tokenizer metadata test: missing "
                  << (tokenizer_dir / "tokenizer.json") << '\n';
        return 0;
    }

    const qus::text::QwenTokenizer tokenizer(tokenizer_dir);

    int failures = 0;
    failures += check(tokenizer.default_stop_token_ids() == std::vector<int>{248046, 248044},
                      "default stop token ids mismatch");

    bool found_im_start = false;
    bool found_think    = false;
    for (const qus::text::AddedToken& token : tokenizer.added_tokens()) {
        if (token.id == 248045) {
            found_im_start = token.content == "<|im_start|>" && token.special;
        }
        if (token.id == 248068) { found_think = token.content == "<think>" && !token.special; }
    }
    failures += check(found_im_start, "missing <|im_start|> added token metadata");
    failures += check(found_think, "missing <think> added token metadata");
    return failures;
}

} // namespace

int main() {
    try {
        return test_load_real_tokenizer_metadata() == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "qwen tokenizer metadata test failed: " << ex.what() << '\n';
        return 1;
    }
}
