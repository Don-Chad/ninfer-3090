#include "runtime/speculative/prompt_lookup.h"

#include <chrono>
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
        failures +=
            expect(match.draft == std::vector<ninfer::TokenId>({8, 1}), "recent continuation");
    }
    {
        const std::vector<ninfer::TokenId> history{1, 2, 3, 4, 5};
        failures +=
            expect(!ninfer::runtime::find_prompt_lookup(history, 4, 2), "no repeated suffix");
        failures += expect(!ninfer::runtime::find_prompt_lookup(history, 0, 2),
                           "zero draft window disables lookup");
    }
    {
        const std::vector<ninfer::TokenId> history{4, 5, 4, 5};
        const auto match = ninfer::runtime::find_prompt_lookup(history, 8, 2);
        failures += expect(!match, "incomplete continuation does not license a lookup round");
    }
    {
        const std::vector<ninfer::TokenId> history{9, 1, 2, 3, 7, 8, 1, 2, 3};
        const auto match = ninfer::runtime::find_prompt_lookup(history, 2, 2);
        failures += expect(match.matched_tokens == 3, "single scan extends the suffix backward");
        failures += expect(match.source_offset == 1, "extended suffix source offset");
        failures += expect(match.draft == std::vector<ninfer::TokenId>({7, 8}),
                           "extended suffix continuation");
    }
    {
        const std::vector<ninfer::TokenId> history{1, 2, 3, 4, 1, 2, 3, 4};
        failures += expect(ninfer::runtime::prompt_repetition_coverage(history, 4) == 0.2,
                           "rolling repetition coverage");
        failures += expect(ninfer::runtime::prompt_repetition_coverage(history, 9) == 0.0,
                           "short history repetition coverage");
    }
    {
        std::vector<ninfer::TokenId> history(1500);
        for (std::size_t i = 0; i < history.size(); ++i) {
            history[i] = static_cast<ninfer::TokenId>((i * 2654435761ULL) % 4093);
        }
        constexpr int iterations = 1000;
        double checksum          = 0.0;
        const auto start         = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            checksum += ninfer::runtime::prompt_repetition_coverage(history, 15);
        }
        const double microseconds =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                .count() /
            iterations;
        std::cout << "prompt repetition scan 1500 tokens: " << microseconds << " us\n";
        failures += expect(checksum >= 0.0, "repetition benchmark checksum");
    }
    return failures == 0 ? 0 : 1;
}
