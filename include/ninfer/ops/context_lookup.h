#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace ninfer::ops {

// The lookup is deliberately bounded and entirely host-side. It searches only the committed
// ledger, never retains a per-round cursor, and emits no more than the caller's fixed verify span.
inline constexpr std::uint32_t kContextLookupStrongMatchTokens  = 12;

struct ContextLookupResult {
    std::uint32_t match_tokens    = 0;
    std::uint32_t proposal_tokens = 0;
};

struct ContextLookupTail {
    std::uint32_t verify_tokens = 0;
    std::uint32_t tail_tokens   = 0;
};

// Host round gate used before lookup. A long traversal is considered only when the complete MTP
// head and the complete configured V window fit; otherwise the caller must execute its ordinary
// P-wide target shape. This deliberately has no partial-width outcome.
inline bool context_lookup_long_round_ready(std::uint32_t proposal_depth,
                                            std::uint32_t long_verify_tokens,
                                            std::uint32_t available_verify_tokens,
                                            std::uint32_t mtp_extent,
                                            bool controller_ready) noexcept {
    return controller_ready && long_verify_tokens > proposal_depth &&
           mtp_extent == proposal_depth && available_verify_tokens >= long_verify_tokens;
}

// Finds the longest suffix with an earlier occurrence in ledger. Ties select the most recent
// occurrence. Earlier and suffix ranges may overlap; continuation reads stay within ledger.
inline ContextLookupResult context_lookup(std::span<const TokenId> ledger,
                                          std::span<TokenId> proposals) {
    if (proposals.empty() || ledger.size() < kContextLookupStrongMatchTokens + 1) { return {}; }

    const std::uint32_t maximum = static_cast<std::uint32_t>(
        std::min<std::size_t>(kContextLookupStrongMatchTokens, ledger.size() - 1));
    for (std::uint32_t length = maximum; length >= kContextLookupStrongMatchTokens; --length) {
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
        if (length == kContextLookupStrongMatchTokens) { break; }
    }
    return {};
}

// Preserves every MTP proposal and appends only the lookup continuation beyond it. A tail is
// licensed only after a full 12-token match, caller-owned controller gating, and exact agreement
// between lookup and MTP across the complete model proposal. The caller passes either the ordinary
// P-wide target span or the complete long V-wide target span. A short continuation never creates
// an intermediate target width: the result is exactly P or exactly output.size().
inline ContextLookupTail append_context_lookup_tail(std::span<const TokenId> mtp,
                                                    std::span<const TokenId> lookup,
                                                    std::uint32_t match_tokens,
                                                    bool controller_ready,
                                                    std::span<TokenId> output) {
    if (mtp.empty() || mtp.size() > output.size()) { return {}; }
    std::copy(mtp.begin(), mtp.end(), output.begin());
    ContextLookupTail result{.verify_tokens = static_cast<std::uint32_t>(mtp.size())};
    if (!controller_ready || match_tokens < kContextLookupStrongMatchTokens ||
        output.size() == mtp.size() || lookup.size() < output.size() ||
        !std::equal(mtp.begin(), mtp.end(), lookup.begin())) {
        return result;
    }
    const std::size_t tail = output.size() - mtp.size();
    std::copy_n(lookup.begin() + static_cast<std::ptrdiff_t>(mtp.size()), tail,
                output.begin() + static_cast<std::ptrdiff_t>(mtp.size()));
    result.verify_tokens += static_cast<std::uint32_t>(tail);
    result.tail_tokens = static_cast<std::uint32_t>(tail);
    return result;
}

} // namespace ninfer::ops
