#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

// N-gram copy proposals: when the tail of the committed text already occurred earlier in the
// request (its prompt, a tool result, or its own output), propose what followed that occurrence.
// A proposal is only a draft; the target verifies every token, so a wrong guess costs throughput
// and never changes output.
//
// Ported from Infernix (Apache-2.0, src/models/qwen3_5/program/ngram_proposer.h, built on remesis's
// work for Neroued/ninfer#234 and Infernix 4f891f5c for bulk prompt indexing). The lookup and its
// round admission rule are unchanged in behaviour; this fork packs bucket entries into 8 bytes,
// sizes the index from the context capacity instead of a fixed 36 MiB, and adds the per-request
// lazy ledger catch-up.
namespace ninfer::models::qwen3_5::detail {

// The ledger tail a lookup may match backwards through.
inline constexpr std::size_t kNgramMatchHistoryTokens = 64;

struct NgramMatch {
    std::vector<TokenId> tokens;
    // Ledger-tail tokens equal to the tokens before the copied span (at least the hashed width).
    std::uint32_t matched = 0;
};

// A bounded, position-indexed proposal corpus. It never owns target state. Absolute positions make
// overwritten ring entries rejectable before a read.
class NgramProposer {
public:
    using Token = TokenId;
    using Match = NgramMatch;

    explicit NgramProposer(std::size_t token_capacity = std::size_t{1} << 20,
                           std::size_t bucket_count   = std::size_t{1} << 19)
        : tokens_(power_of_two(token_capacity) && token_capacity >= 64 ? token_capacity : 0, -1),
          buckets_(power_of_two(bucket_count) ? bucket_count : 0),
          token_mask_(token_capacity - 1U) {
        if (tokens_.empty() || buckets_.empty()) {
            throw std::invalid_argument("invalid ngram corpus capacity");
        }
    }

    void boundary() { append(-1); }

    // Tokens that end a copyable span wherever they occur (the tokenizer's special tokens).
    void set_boundaries(std::vector<Token> tokens) {
        boundaries_ = std::move(tokens);
        std::sort(boundaries_.begin(), boundaries_.end());
    }

    void append(Token token) {
        if (!store(token)) { return; }
        for (const auto n : kWidths) {
            if (insertable(n)) { insert(hash_at(end_ - n, n), n); }
        }
    }

    // Indexes a whole span between boundaries. Each window is hashed once from the input, a few
    // tokens before its insert, so its bucket line is already on its way; an inserted window
    // never spans a boundary, so the input holds exactly the tokens the ring then holds, and the
    // index is the one token-by-token append() builds.
    void ingest(std::span<const Token> tokens) {
        boundary();
        constexpr std::size_t kLookahead = 8;
        std::array<std::array<std::uint64_t, kWidths.size()>, kLookahead> ahead{};
        const auto hash_windows = [&](std::size_t end) {
            auto& hashes = ahead[end % kLookahead];
            for (std::size_t w = 0; w < kWidths.size(); ++w) {
                if (end < kWidths[w]) { continue; }
                hashes[w] = hash(tokens.subspan(end - kWidths[w], kWidths[w]), kWidths[w]);
                prefetch(&buckets_[hashes[w] & (buckets_.size() - 1)]);
            }
        };
        for (std::size_t end = 1; end <= std::min(kLookahead, tokens.size()); ++end) {
            hash_windows(end);
        }
        for (std::size_t end = 1; end <= tokens.size(); ++end) {
            const std::array<std::uint64_t, kWidths.size()> hashes = ahead[end % kLookahead];
            if (end + kLookahead <= tokens.size()) { hash_windows(end + kLookahead); }
            if (!store(tokens[end - 1])) { continue; }
            for (std::size_t w = 0; w < kWidths.size(); ++w) {
                if (insertable(kWidths[w])) { insert(hashes[w], kWidths[w]); }
            }
        }
        boundary();
    }

    [[nodiscard]] Match propose_for_round(std::span<const Token> history, std::uint32_t maximum,
                                          std::uint32_t neural_drafts,
                                          std::uint32_t minimum_match = 12) const {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        return finish_round(propose(history, maximum + 2U, minimum_match), maximum, neural_drafts);
    }

    // Sizes a lookup of maximum + 2 tokens for a round of at most `maximum` drafts. The target
    // emits a bonus after the drafts; one source token is left beyond it, because a source's ending
    // may need different surrounding text. A copy shorter than the window that offers no more
    // drafts than the neural proposal is dropped, so a copy only takes over a round when it is
    // longer than the neural proposal or fills the whole window.
    [[nodiscard]] static Match finish_round(Match match, std::uint32_t maximum,
                                            std::uint32_t neural_drafts) {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        const auto available = static_cast<std::uint32_t>(match.tokens.size());
        const auto drafts    = std::min(maximum, available > 2U ? available - 2U : 0U);
        if (available < maximum + 2U && drafts <= neural_drafts) { return {}; }
        match.tokens.resize(drafts);
        return match;
    }

    // The longest backward match over the hashed widths, then the longest copy, of at most
    // `maximum` tokens. Hash collisions and overwritten ring entries are rejected by exact compare.
    [[nodiscard]] Match propose(std::span<const Token> history, std::uint32_t maximum,
                                std::uint32_t minimum_match = 12) const {
        Match best;
        if (maximum == 0) { return best; }
        for (const auto n : kWidths) {
            if (history.size() < n) { continue; }
            const auto tail = history.last(n);
            if (std::find_if(tail.begin(), tail.end(), [](Token t) { return t < 0; }) !=
                tail.end()) {
                continue;
            }
            const auto& bucket = buckets_[hash(tail, n) & (buckets_.size() - 1)];
            for (const auto entry : bucket) {
                if (entry.width != n || entry.end < n || entry.end - n < live_begin() ||
                    entry.end >= end_) {
                    continue;
                }
                bool equal = true;
                for (std::uint32_t j = 0; j < n; ++j) {
                    if (at(entry.end - n + j) != tail[j]) {
                        equal = false;
                        break;
                    }
                }
                if (!equal) { continue; }
                std::uint32_t matched = n;
                while (matched < history.size() && entry.end - matched > live_begin()) {
                    const auto token = at(entry.end - matched - 1);
                    if (token < 0 || token != history[history.size() - matched - 1]) { break; }
                    ++matched;
                }
                if (matched < minimum_match || matched < best.matched) { continue; }
                std::vector<Token> draft;
                for (std::uint64_t p = entry.end; p < end_ && draft.size() < maximum; ++p) {
                    const auto token = at(p);
                    if (token < 0) { break; }
                    draft.push_back(token);
                }
                if (!draft.empty() &&
                    (matched > best.matched || draft.size() > best.tokens.size())) {
                    best = {std::move(draft), matched};
                }
            }
        }
        return best;
    }

    // Host bytes this index holds.
    [[nodiscard]] std::size_t bytes() const noexcept {
        return tokens_.capacity() * sizeof(Token) + buckets_.capacity() * sizeof(Bucket) +
               boundaries_.capacity() * sizeof(Token);
    }

private:
    // Eight bytes, so a context-sized table can afford a bucket per ring token. A request ingests
    // at most a few times its context, far below 2^32 positions; insert() stops past that.
    struct Entry {
        std::uint32_t end   = 0;
        std::uint32_t width = 0;
    };
    using Bucket = std::array<Entry, 4>;

    static constexpr std::array<std::uint32_t, 3> kWidths{16, 8, 4};
    std::vector<Token> tokens_;
    std::vector<Token> boundaries_;
    std::vector<Bucket> buckets_;
    std::uint64_t token_mask_    = 0;
    std::uint64_t end_           = 0;
    std::uint64_t segment_start_ = 0;

    [[nodiscard]] static bool power_of_two(std::size_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }

    static void prefetch(const void* address) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(address, 1);
#else
        (void)address;
#endif
    }

    // Writes the next ring token; false when it is a boundary, which starts a new segment.
    bool store(Token token) {
        if (std::binary_search(boundaries_.begin(), boundaries_.end(), token)) { token = -1; }
        tokens_[end_ & token_mask_] = token;
        ++end_;
        if (token < 0) {
            segment_start_ = end_;
            return false;
        }
        return true;
    }

    // Whether the width-n window ending at the ring end lies in one live segment.
    [[nodiscard]] bool insertable(std::uint32_t n) const {
        return end_ - segment_start_ >= n && end_ - live_begin() >= n;
    }

    void insert(std::uint64_t window_hash, std::uint32_t n) {
        if (end_ > std::numeric_limits<std::uint32_t>::max()) { return; }
        auto& bucket = buckets_[window_hash & (buckets_.size() - 1)];
        for (std::size_t i = bucket.size() - 1; i > 0; --i) { bucket[i] = bucket[i - 1]; }
        bucket[0] = {static_cast<std::uint32_t>(end_), n};
    }

    [[nodiscard]] std::uint64_t live_begin() const {
        return end_ > tokens_.size() ? end_ - tokens_.size() : 0;
    }

    [[nodiscard]] Token at(std::uint64_t p) const { return tokens_[p & token_mask_]; }

    static std::uint64_t mix(std::uint64_t h, Token token) {
        return (h ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
    }

    static std::uint64_t finalize_hash(std::uint64_t value) {
        // Buckets mask low bits; avalanche patterned IDs before selecting a bucket.
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    static std::uint64_t hash(std::span<const Token> tokens, std::uint32_t n) {
        std::uint64_t h = 1469598103934665603ULL ^ n;
        for (auto token : tokens) { h = mix(h, token); }
        return finalize_hash(h);
    }

    [[nodiscard]] std::uint64_t hash_at(std::uint64_t p, std::uint32_t n) const {
        std::uint64_t h = 1469598103934665603ULL ^ n;
        for (std::uint32_t j = 0; j < n; ++j) { h = mix(h, at(p + j)); }
        return finalize_hash(h);
    }
};

struct NgramIndexCapacity {
    std::size_t tokens  = 0;
    std::size_t buckets = 0;
};

// A request's index is sized from the context capacity: the ring holds every position a sequence
// can reach, so its own prompt and output are never displaced by each other (proposal-only tool
// sources, ingested after the prompt, can still push its oldest tokens out). Each token inserts
// three windows into four-entry buckets, so one bucket per ring token keeps a full context at about
// 0.55-0.75 load and its oldest windows findable. Infernix's ratio of half as many buckets found
// 917 of 2,000 early-prompt copies after a 190K-token prompt, this sizing 1,878. At 188K or 196K
// tokens that is a 2^18 ring and 2^18 buckets, 9 MiB (Infernix: 36 MiB fixed); the upper bound is a
// 2^20 ring.
[[nodiscard]] inline NgramIndexCapacity ngram_index_capacity(std::uint32_t max_context) {
    constexpr std::size_t kMinimum = std::size_t{1} << 12;
    constexpr std::size_t kMaximum = std::size_t{1} << 20;
    const std::size_t tokens =
        std::bit_ceil(std::clamp<std::size_t>(max_context, kMinimum, kMaximum));
    return {.tokens = tokens, .buckets = tokens};
}

// One request's copy index: its prepared prompt and proposal-only sources, then its committed
// ledger, appended lazily when a round asks for a proposal. The frontend builds it on the preparing
// thread; the Program moves it into the request's control at binding, so it follows the request
// through pause and replay together with the ledger it indexes.
class NgramRequestIndex {
public:
    NgramRequestIndex(NgramIndexCapacity capacity, std::vector<TokenId> boundaries)
        : proposer_(capacity.tokens, capacity.buckets) {
        proposer_.set_boundaries(std::move(boundaries));
    }

    // Indexes the prompt (its leading `skip` positions are installed placeholders, not text) and
    // each proposal-only source as its own segment. The ledger starts as exactly this prompt.
    void index_prompt(std::span<const TokenId> prompt, std::size_t skip,
                      std::span<const std::vector<TokenId>> sources) {
        proposer_.ingest(prompt.subspan(std::min(skip, prompt.size())));
        for (const auto& source : sources) { proposer_.ingest(source); }
        indexed_ = prompt.size();
    }

    // Appends the ledger tokens committed since the last call, then proposes a round's copy from
    // the ledger tail.
    [[nodiscard]] NgramMatch propose_for_round(std::span<const TokenId> ledger,
                                               std::uint32_t maximum, std::uint32_t neural_drafts,
                                               std::uint32_t minimum_match) {
        catch_up(ledger);
        const auto history = ledger.last(std::min(kNgramMatchHistoryTokens, ledger.size()));
        return proposer_.propose_for_round(history, maximum, neural_drafts, minimum_match);
    }

    // Only committed ledger tokens are indexed; rejected or pending draft columns never are. A
    // ledger shorter than what was indexed cannot be un-indexed: a boundary keeps new windows from
    // spanning the discontinuity, and the stale tokens remain only as candidates the target rejects.
    void catch_up(std::span<const TokenId> ledger) {
        if (ledger.size() < indexed_) {
            proposer_.boundary();
            indexed_ = ledger.size();
        }
        while (indexed_ < ledger.size()) { proposer_.append(ledger[indexed_++]); }
    }

    [[nodiscard]] std::size_t indexed() const noexcept { return indexed_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return sizeof(*this) + proposer_.bytes(); }

private:
    NgramProposer proposer_;
    std::size_t indexed_ = 0;
};

} // namespace ninfer::models::qwen3_5::detail
