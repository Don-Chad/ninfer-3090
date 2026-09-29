#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

// Request-level restriction of the generated-token language, such as a JSON grammar. The
// Frontend implements it from its committed output state; Program consults it immediately before
// a sampling decision and hands the result to the sampling Op as a token mask.
//
// A mask is a bitset over the public token domain: token v is licensed when bit (v & 31) of word
// (v >> 5) is set. The source never changes its committed state here: commitment still happens
// only through the OutputSession preview/commit transaction, after Program commits the model
// state.
class TokenMaskSource {
public:
    virtual ~TokenMaskSource() = default;

    // Fills masks[c * words_per_mask, (c + 1) * words_per_mask) for every column c < columns.
    // Column c is the mask for the token after the committed output followed by
    // speculative[0, c); columns <= speculative.size() are required. A column that can never be
    // consumed, because an earlier speculative token is itself unlicensed or already ends the
    // output, licenses every token. Returns false when no column restricts the vocabulary, in
    // which case the caller may skip applying a mask for this decision.
    [[nodiscard]] virtual bool fill_token_masks(std::span<const TokenId> speculative,
                                                std::uint32_t columns,
                                                std::span<std::uint32_t> masks,
                                                std::size_t words_per_mask) = 0;
};

} // namespace ninfer::runtime
