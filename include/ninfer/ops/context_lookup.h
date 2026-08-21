#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace ninfer::ops {

// The lookup is deliberately bounded and entirely host-side. It searches only the committed
// ledger, never retains a per-round cursor, and emits no more than the existing MTP width.
inline constexpr std::uint32_t kContextLookupMinimumMatchTokens = 6;
inline constexpr std::uint32_t kContextLookupStrongMatchTokens  = 12;

struct ContextLookupResult {
    std::uint32_t match_tokens    = 0;
    std::uint32_t proposal_tokens = 0;
};

struct ContextLookupFusion {
    std::uint32_t lookup_tokens = 0;
    bool strong_override         = false;
};

// Finds the longest suffix with an earlier occurrence in ledger. Ties select the most recent
// occurrence. Earlier and suffix ranges may overlap; continuation reads stay within ledger.
inline ContextLookupResult context_lookup(std::span<const TokenId> ledger,
                                          std::span<TokenId> proposals) {
    if (proposals.empty() || ledger.size() < kContextLookupMinimumMatchTokens + 1) { return {}; }

    const std::uint32_t maximum = static_cast<std::uint32_t>(
        std::min<std::size_t>(kContextLookupStrongMatchTokens, ledger.size() - 1));
    for (std::uint32_t length = maximum; length >= kContextLookupMinimumMatchTokens; --length) {
        const std::size_t suffix = ledger.size() - length;
        // start + length < ledger.size() guarantees at least one continuation token.
        for (std::size_t start = ledger.size() - length; start-- > 0;) {
            if (start + length >= ledger.size()) { continue; }
            if (!std::equal(ledger.begin() + static_cast<std::ptrdiff_t>(start),
                            ledger.begin() + static_cast<std::ptrdiff_t>(start + length),
                            ledger.begin() + static_cast<std::ptrdiff_t>(suffix))) {
                continue;
            }
            const std::size_t available = ledger.size() - (start + length);
            const std::size_t count     = std::min(available, proposals.size());
            std::copy_n(ledger.begin() + static_cast<std::ptrdiff_t>(start + length), count,
                        proposals.begin());
            return {.match_tokens = length, .proposal_tokens = static_cast<std::uint32_t>(count)};
        }
        if (length == kContextLookupMinimumMatchTokens) { break; }
    }
    return {};
}

// Starts from the MTP proposal unconditionally. A 12-token suffix match is strong enough to
// replace available positions; shorter matches may contribute only if their first token agrees
// with MTP. This keeps ordinary MTP behavior whenever lookup confidence is unavailable.
inline ContextLookupFusion fuse_context_lookup(std::span<const TokenId> mtp,
                                                std::span<const TokenId> lookup,
                                                std::uint32_t match_tokens,
                                                std::span<TokenId> fused) {
    if (mtp.size() != fused.size() || lookup.size() > fused.size()) { return {}; }
    std::copy(mtp.begin(), mtp.end(), fused.begin());
    if (lookup.empty() || match_tokens < kContextLookupMinimumMatchTokens) { return {}; }
    const bool strong = match_tokens >= kContextLookupStrongMatchTokens;
    if (!strong && lookup.front() != mtp.front()) { return {}; }
    std::copy(lookup.begin(), lookup.end(), fused.begin());
    return {.lookup_tokens = static_cast<std::uint32_t>(lookup.size()), .strong_override = strong};
}

} // namespace ninfer::ops
