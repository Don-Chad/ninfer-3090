#include "runtime/speculative/prompt_lookup.h"

#include <iostream>
#include <vector>

namespace {

int expect(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    {
        const std::vector<ninfer::TokenId> history{10, 11, 12, 20, 21, 10, 11, 12};
        const auto match = ninfer::runtime::find_prompt_lookup(history, 5, 3);
        failures += expect(match.matched_tokens == 3, "three-token suffix match");
        failures += expect(match.source_offset == 0, "source offset");
        failures += expect(match.draft == std::vector<ninfer::TokenId>({20, 21, 10, 11, 12}),
                           "continuation tokens");
    }
    {
        const std::vector<ninfer::TokenId> history{1, 2, 7, 1, 2, 8, 1, 2};
        const auto match = ninfer::runtime::find_prompt_lookup(history, 2, 2);
        failures += expect(match.source_offset == 3, "ties prefer recent occurrence");
        failures += expect(match.draft == std::vector<ninfer::TokenId>({8, 1}),
                           "recent continuation");
    }
    {
        const std::vector<ninfer::TokenId> history{1, 2, 3, 4, 5};
        failures += expect(!ninfer::runtime::find_prompt_lookup(history, 4, 2),
                           "no repeated suffix");
        failures += expect(!ninfer::runtime::find_prompt_lookup(history, 0, 2),
                           "zero draft window disables lookup");
    }
    {
        const std::vector<ninfer::TokenId> history{4, 5, 4, 5};
        const auto match = ninfer::runtime::find_prompt_lookup(history, 8, 2);
        failures += expect(match.draft == std::vector<ninfer::TokenId>({4, 5}),
                           "draft is bounded by available history");
    }
    return failures == 0 ? 0 : 1;
}
