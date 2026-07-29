#pragma once

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime {

struct PromptLookupMatch {
    std::size_t source_offset  = 0;
    std::size_t matched_tokens = 0;
    std::vector<TokenId> draft;

    [[nodiscard]] explicit operator bool() const noexcept { return !draft.empty(); }
};

// Find the longest suffix of history that occurred earlier with a complete max_draft_tokens
// continuation. Ties prefer the most recent occurrence because local code structure is generally
// a stronger continuation signal than distant boilerplate.
[[nodiscard]] PromptLookupMatch find_prompt_lookup(std::span<const TokenId> history,
                                                   std::uint32_t max_draft_tokens,
                                                   std::uint32_t min_match_tokens,
                                                   std::uint32_t max_match_tokens = 64);

} // namespace ninfer::runtime
