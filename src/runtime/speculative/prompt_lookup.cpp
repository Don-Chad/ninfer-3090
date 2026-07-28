#include "runtime/speculative/prompt_lookup.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {

PromptLookupMatch find_prompt_lookup(std::span<const TokenId> history,
                                     std::uint32_t max_draft_tokens,
                                     std::uint32_t min_match_tokens,
                                     std::uint32_t max_match_tokens) {
    if (max_draft_tokens == 0) { return {}; }
    if (min_match_tokens == 0) {
        throw std::invalid_argument("prompt lookup minimum match must be nonzero");
    }
    if (max_match_tokens < min_match_tokens) {
        throw std::invalid_argument("prompt lookup maximum match is below its minimum");
    }
    if (history.size() <= min_match_tokens) { return {}; }

    const std::size_t largest =
        std::min<std::size_t>({max_match_tokens, history.size() - 1, history.size() / 2});
    for (std::size_t length = largest; length >= min_match_tokens; --length) {
        const auto suffix = history.last(length);
        const std::size_t latest_start = history.size() - length;
        for (std::size_t start = latest_start; start-- > 0;) {
            const std::size_t continuation = start + length;
            if (continuation >= history.size()) { continue; }
            if (!std::equal(suffix.begin(), suffix.end(), history.begin() + start)) { continue; }

            const std::size_t available = history.size() - continuation;
            const std::size_t count = std::min<std::size_t>(max_draft_tokens, available);
            PromptLookupMatch result;
            result.source_offset = start;
            result.matched_tokens = length;
            result.draft.assign(history.begin() + continuation,
                                history.begin() + continuation + count);
            return result;
        }
        if (length == min_match_tokens) { break; }
    }
    return {};
}

} // namespace ninfer::runtime
