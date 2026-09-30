#pragma once

#include <cstdint>

namespace ninfer::runtime {

struct EffectiveThinkingBudget {
    std::uint32_t effective_budget = 0;
    bool early_close_available     = true;
};

// Computes the effective thinking budget and early-close availability flag given:
// B: requested thinking budget
// C: remaining output capacity (min(requested_output_tokens, max_context - prompt_tokens + 1))
// R: required tokens for early close (thinking_control_tokens.size() + 1)
//
// Cases:
// 1. C <= B or C - B >= R: B, available (unchanged behavior).
// 2. C > R (and C - B < R): C - R, available. This always lowers B.
// 3. C <= R (and C > B): B unchanged, not available, so thinking runs to the output limit
//    with no control insertion.
[[nodiscard]] constexpr EffectiveThinkingBudget
effective_thinking_budget(std::uint32_t B, std::uint32_t C, std::uint32_t R) noexcept {
    if (C <= B || C - B >= R) {
        return {.effective_budget = B, .early_close_available = true};
    }
    if (C > R) {
        return {.effective_budget = C - R, .early_close_available = true};
    }
    return {.effective_budget = B, .early_close_available = false};
}

} // namespace ninfer::runtime
