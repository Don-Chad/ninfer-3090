#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>

namespace ninfer::qwen3_5 {

// Context-lookup drafting: propose the continuation that followed the last time this n-gram
// appeared.
//
// A draft model guesses what comes next from weights; this guesses it from the text already in
// front of it, which costs no device work at all. It is worthless on free-form prose and close to
// perfect wherever the output repeats the input -- summarising, editing, answering from a retrieved
// passage, or emitting code that reuses names and structure already on screen. Those are the cases
// a draft head is weakest on, which is why the two are complements rather than alternatives.
//
// It is exact. The proposal is one-hot, the verify step accepts the longest correct prefix and
// commits the target model's own argmax at the first mismatch, so a wrong guess costs throughput
// and never changes a token.
//
// MEASURED, 2026-09-19, Qwen3.8-27B on one 3090, `-pg 4096,256`, MTP with the optimised draft head,
// lookup preferred over the head's proposal whenever it matches:
//
//   n-gram    2      3      4      6      8     16     off
//   decode  135.6  148.9  143.5  145.2  148.7  148.5  144.9 tok/s
//   accept   0.87   1.00   0.96   0.97   1.00   1.00   0.964
//
// **Short n-grams lose.** At 2 the proposal is worse than the draft head's and decode drops 7%: a
// two-token context matches all over the place and the continuations disagree. From 8 up every
// match that fires is correct -- acceptance is exactly 1.0 -- and decode gains ~3%.
//
// **The gain is small because there was little to win.** The draft head already accepts 95-96% on
// this model, so a perfect drafter is worth +5.5% at K=3 and +13% at K=5 and no more; this takes
// about half of that (K=5: 182.7 -> 191.0 tok/s). Anyone expecting the large numbers reported for
// lookup drafting elsewhere should check the baseline's acceptance first -- against a weak drafter
// the same technique is transformative, and against this one it is a trim.
//
// **Where the headroom was:** at 100% acceptance tokens per round is set by the draft window, which
// was capped at five (K=3/4/5: 148.9/169.3/191.0 tok/s). The cap is now kMtpDecodeMaximumDrafts =
// 15, and the lookup proposes up to the same window. MEASURED, 2026-09-28, same card and model,
// rk4v4, 512 tokens of a "rename an identifier, return the whole file" answer: K=15 decodes at
// 262.6 tok/s without lookup and 262.4 with --lookup-ngram 8, at the same 12.46 tokens per round.
// On output that copies its input the draft head copies too, so the wider window, not the lookup,
// is what pays there; the lookup's remaining value is the case the head misses.
//
// Returns the number of drafts written, and zero when the n-gram has not been seen before -- which
// the round already treats as a plain decode step.
[[nodiscard]] inline std::uint32_t lookup_draft(std::span<const TokenId> ledger,
                                                std::uint32_t ngram, std::uint32_t max_drafts,
                                                TokenId* out) {
    if (ngram == 0 || max_drafts == 0 || out == nullptr) { return 0; }
    const std::size_t size = ledger.size();
    if (size <= ngram) { return 0; }

    // The n-gram to match is the tail of the ledger, whose last token is the one just committed.
    const TokenId* const tail = ledger.data() + (size - ngram);

    // Backwards, so the most recent occurrence wins: in repetitive text the nearest copy is the one
    // most likely to continue the same way.
    for (std::size_t start = size - ngram; start-- > 0;) {
        bool match = true;
        for (std::uint32_t j = 0; j < ngram; ++j) {
            if (ledger[start + j] != tail[j]) {
                match = false;
                break;
            }
        }
        if (!match) { continue; }
        const std::size_t follow = start + ngram;
        if (follow >= size) { continue; }
        const std::size_t available = size - follow;
        const auto count =
            static_cast<std::uint32_t>(available < max_drafts ? available : max_drafts);
        for (std::uint32_t j = 0; j < count; ++j) { out[j] = ledger[follow + j]; }
        return count;
    }
    return 0;
}

} // namespace ninfer::qwen3_5
