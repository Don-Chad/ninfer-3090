#include "runtime/speculative/prompt_lookup.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {

PromptLookupMatch find_prompt_lookup(std::span<const TokenId> history,
                                     std::uint32_t max_draft_tokens, std::uint32_t min_match_tokens,
                                     std::uint32_t max_match_tokens) {
    if (max_draft_tokens == 0) { return {}; }
    if (min_match_tokens == 0) {
        throw std::invalid_argument("prompt lookup minimum match must be nonzero");
    }
    if (max_match_tokens < min_match_tokens) {
        throw std::invalid_argument("prompt lookup maximum match is below its minimum");
    }
    if (history.size() <= min_match_tokens) { return {}; }

    const std::size_t suffix_start = history.size() - min_match_tokens;
    const auto suffix              = history.subspan(suffix_start, min_match_tokens);
    const std::size_t largest =
        std::min<std::size_t>({max_match_tokens, history.size() - 1, history.size() / 2});

    std::size_t best_start        = 0;
    std::size_t best_continuation = 0;
    std::size_t best_length       = 0;
    for (std::size_t candidate = suffix_start; candidate-- > 0;) {
        const std::size_t continuation = candidate + min_match_tokens;
        if (history.size() - continuation < max_draft_tokens) { continue; }
        if (!std::equal(suffix.begin(), suffix.end(), history.begin() + candidate)) { continue; }

        std::size_t length = min_match_tokens;
        while (length < largest && candidate >= length - min_match_tokens + 1 &&
               history[candidate - (length - min_match_tokens) - 1] ==
                   history[suffix_start - (length - min_match_tokens) - 1]) {
            ++length;
        }
        const std::size_t start = candidate - (length - min_match_tokens);
        if (length > best_length) {
            best_start        = start;
            best_continuation = continuation;
            best_length       = length;
            if (best_length == largest) { break; }
        }
    }
    if (best_length == 0) { return {}; }

    PromptLookupMatch result;
    result.source_offset  = best_start;
    result.matched_tokens = best_length;
    result.draft.assign(history.begin() + best_continuation,
                        history.begin() + best_continuation + max_draft_tokens);
    return result;
}

} // namespace ninfer::runtime
