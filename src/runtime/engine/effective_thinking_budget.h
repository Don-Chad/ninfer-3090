#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::runtime {

struct EffectiveThinkingBudget {
    std::uint32_t effective_budget = 0;
    bool early_close_available     = true;
};

// Computes the effective thinking budget and early-close availability flag given:
// B: requested thinking budget
// C: remaining output capacity (see effective_output_capacity)
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

// C: the output a request can still produce, bounded by its limit and by the context window left
// after the prompt (the last context position is writable, hence the +1).
[[nodiscard]] constexpr std::uint32_t
effective_output_capacity(std::uint32_t requested_output_tokens, std::uint32_t max_context,
                          std::uint32_t prompt_tokens) noexcept {
    return std::min(requested_output_tokens, max_context - prompt_tokens + std::uint32_t{1});
}

// Resolves the request's thinking cap against its output capacity. A request without a cap is left
// untouched. `control_token_count` is the length of the early-close control suffix.
constexpr void apply_effective_thinking_budget(ThinkingControlOptions& thinking,
                                               std::uint32_t capacity,
                                               std::uint32_t control_token_count) noexcept {
    if (!thinking.budget.has_value()) { return; }
    const EffectiveThinkingBudget effective =
        effective_thinking_budget(*thinking.budget, capacity, control_token_count + 1U);
    thinking.effective_budget      = effective.effective_budget;
    thinking.early_close_available = effective.early_close_available;
}

} // namespace ninfer::runtime
