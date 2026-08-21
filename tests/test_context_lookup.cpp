#include "ninfer/ops/context_lookup.h"

#include <array>
#include <iostream>
#include <span>
#include <vector>

namespace {

using ninfer::TokenId;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool equals(std::span<const TokenId> actual, std::initializer_list<TokenId> expected) {
    return actual.size() == expected.size() &&
           std::equal(actual.begin(), actual.end(), expected.begin());
}

} // namespace

int main() {
    using namespace ninfer::ops;
    int failures = 0;

    std::array<TokenId, 4> output{};
    const std::vector<TokenId> no_match{1, 2, 3, 4, 5, 6, 7};
    const ContextLookupResult none = context_lookup(no_match, output);
    failures += check(none.match_tokens == 0 && none.proposal_tokens == 0,
                      "lookup found a nonexistent suffix match");

    const std::vector<TokenId> longest{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99,
                                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const ContextLookupResult longest_result = context_lookup(longest, output);
    failures += check(longest_result.match_tokens == 12 && longest_result.proposal_tokens == 4 &&
                          equals(std::span<const TokenId>(output.data(), 4), {99, 1, 2, 3}),
                      "lookup did not prefer the longest suffix match");

    const std::vector<TokenId> recent_tie{1, 2, 3, 4, 5, 6, 70, 1, 2, 3, 4, 5, 6,
                                          80, 1, 2, 3, 4, 5, 6};
    const ContextLookupResult tie_result = context_lookup(recent_tie, output);
    failures += check(tie_result.match_tokens == 6 && tie_result.proposal_tokens == 4 &&
                          equals(std::span<const TokenId>(output.data(), 4), {80, 1, 2, 3}),
                      "lookup did not prefer the most recent suffix tie");

    // Matching is deliberately capped at 12 tokens. A more recent 12-token match therefore wins
    // even when an older occurrence shares a longer prefix with the current suffix.
    const std::vector<TokenId> bounded_tie{
        90, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 70,
        91, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 80,
        90, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const ContextLookupResult bounded_result = context_lookup(bounded_tie, output);
    failures += check(bounded_result.match_tokens == 12 && bounded_result.proposal_tokens == 4 &&
                          equals(std::span<const TokenId>(output.data(), 4), {80, 90, 1, 2}),
                      "lookup did not apply the bounded most-recent tie contract");

    const std::vector<TokenId> overlap{1, 2, 1, 2, 1, 2, 1, 2};
    const ContextLookupResult overlap_result = context_lookup(overlap, output);
    failures += check(overlap_result.match_tokens == 6 && overlap_result.proposal_tokens == 2 &&
                          equals(std::span<const TokenId>(output.data(), 2), {1, 2}),
                      "lookup rejected an overlapping suffix match");

    const std::vector<TokenId> partial(7, 42);
    const ContextLookupResult partial_result = context_lookup(partial, output);
    failures += check(partial_result.match_tokens == 6 && partial_result.proposal_tokens == 1 &&
                          equals(std::span<const TokenId>(output.data(), 1), {42}),
                      "lookup read an incorrect partial continuation");

    const std::vector<TokenId> repeated(14, 5);
    const ContextLookupResult repeated_result = context_lookup(repeated, output);
    failures += check(repeated_result.match_tokens == 12 && repeated_result.proposal_tokens == 1 &&
                          equals(std::span<const TokenId>(output.data(), 1), {5}),
                      "lookup mishandled repeated tokens");

    std::array<TokenId, 2> limited{};
    const std::vector<TokenId> capacity{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 51,
                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const ContextLookupResult capacity_result = context_lookup(capacity, limited);
    failures += check(capacity_result.match_tokens == 12 && capacity_result.proposal_tokens == 2 &&
                          equals(limited, {50, 51}),
                      "lookup exceeded or failed to fill the proposal capacity");

    const std::array<TokenId, 3> mtp{10, 11, 12};
    std::array<TokenId, 3> fused{};
    const ContextLookupFusion fallback = fuse_context_lookup(mtp, {}, 0, fused);
    failures += check(fallback.lookup_tokens == 0 && equals(fused, {10, 11, 12}),
                      "empty lookup did not preserve MTP proposals");

    const std::array<TokenId, 2> disagreement{20, 21};
    const ContextLookupFusion rejected =
        fuse_context_lookup(mtp, disagreement, kContextLookupMinimumMatchTokens, fused);
    failures += check(rejected.lookup_tokens == 0 && equals(fused, {10, 11, 12}),
                      "weak lookup without MTP agreement was accepted");

    const std::array<TokenId, 2> agreement{10, 77};
    const ContextLookupFusion agreed =
        fuse_context_lookup(mtp, agreement, kContextLookupMinimumMatchTokens, fused);
    failures += check(agreed.lookup_tokens == 2 && !agreed.strong_override &&
                          equals(fused, {10, 77, 12}),
                      "weak agreeing lookup did not preserve the MTP tail");

    const std::array<TokenId, 3> strong{30, 31, 32};
    const ContextLookupFusion override =
        fuse_context_lookup(mtp, strong, kContextLookupStrongMatchTokens, fused);
    failures += check(override.lookup_tokens == 3 && override.strong_override &&
                          equals(fused, {30, 31, 32}),
                      "strong lookup did not override MTP proposals");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
