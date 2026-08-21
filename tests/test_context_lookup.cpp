#include "ninfer/ops/context_lookup.h"

#include <algorithm>
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
    std::array<TokenId, 5> lookup_tokens{};
    const std::vector<TokenId> no_match{1, 2, 3, 4, 5, 6, 7};
    const auto none = context_lookup(no_match, lookup_tokens);
    failures += check(none.match_tokens == 0 && none.proposal_tokens == 0,
                      "lookup found a nonexistent suffix match");

    const std::vector<TokenId> repeated{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 30, 31, 32, 33, 34,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const auto found = context_lookup(repeated, lookup_tokens);
    failures += check(found.match_tokens == 12 && found.proposal_tokens == 5 &&
                          equals(lookup_tokens, {30, 31, 32, 33, 34}),
                      "12-token lookup continuation is incorrect");

    std::array<TokenId, 2> limited{};
    const auto clamped = context_lookup(repeated, limited);
    failures += check(clamped.proposal_tokens == 2 && equals(limited, {30, 31}),
                      "lookup did not clamp to the caller-owned row extent");

    const std::array<TokenId, 3> mtp{30, 31, 32};
    std::array<TokenId, 5> verify{};
    const auto controller_blocked = append_context_lookup_tail(
        mtp, lookup_tokens, kContextLookupStrongMatchTokens, false, verify);
    failures += check(controller_blocked.verify_tokens == 3 &&
                          controller_blocked.tail_tokens == 0 &&
                          equals(std::span<const TokenId>(verify.data(), 3), {30, 31, 32}),
                      "controller gate did not preserve the MTP head exactly");

    verify.fill(0);
    const auto full = append_context_lookup_tail(
        mtp, lookup_tokens, kContextLookupStrongMatchTokens, true, verify);
    failures += check(full.verify_tokens == 5 && full.tail_tokens == 2 &&
                          equals(verify, {30, 31, 32, 33, 34}),
                      "lookup tail was not appended after the complete MTP proposal");

    verify.fill(0);
    const std::array<TokenId, 5> disagreement{30, 99, 32, 33, 34};
    const auto rejected = append_context_lookup_tail(
        mtp, disagreement, kContextLookupStrongMatchTokens, true, verify);
    failures += check(rejected.verify_tokens == 3 && rejected.tail_tokens == 0 &&
                          equals(std::span<const TokenId>(verify.data(), 3), {30, 31, 32}),
                      "lookup disagreement overrode an MTP head token");

    std::array<TokenId, 4> partial_verify{};
    const auto partial = append_context_lookup_tail(
        mtp, lookup_tokens, kContextLookupStrongMatchTokens, true, partial_verify);
    failures += check(partial.verify_tokens == 4 && partial.tail_tokens == 1 &&
                          equals(partial_verify, {30, 31, 32, 33}),
                      "budget/context-sized verification span did not clamp the tail");

    std::array<TokenId, 3> no_tail_verify{};
    const auto no_tail = append_context_lookup_tail(
        mtp, lookup_tokens, kContextLookupStrongMatchTokens, true, no_tail_verify);
    failures += check(no_tail.verify_tokens == 3 && no_tail.tail_tokens == 0 &&
                          equals(no_tail_verify, {30, 31, 32}),
                      "V=P changed the MTP proposal");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
