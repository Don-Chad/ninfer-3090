#pragma once

// Repeated-passage detection over a reasoning stream, for the opt-in reasoning-loop guard
// (ThinkingControlOptions::loop). A model stuck in its thinking rewrites the same passages: a
// verification pass, a recap, a list of cases, over and over. Token penalties do not see it (each
// pass varies a little), so the measure works on words and long passages: the share of the most
// recent words that sit inside a 12-word passage seen at least three times in the recent history.
// That is Strata's measure (its #728 recovery), which also fixes the thresholds below; the
// literature's looping criterion (a 30-gram repeated 20 times) only fires far later.
// Ported from Infernix (github.com/wallawalla47/infernix, Apache-2.0).

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kLoopCheckTokens = 512;   // reasoning tokens between two checks
inline constexpr std::size_t kLoopPassageWords  = 12;    // words in one passage
inline constexpr std::uint32_t kLoopRepeats     = 3;     // occurrences that make a passage repeated
inline constexpr std::size_t kLoopWindowWords   = 2000;  // the recent words whose coverage is measured
inline constexpr std::size_t kLoopHistoryWords  = 30000; // how far back passages are counted
inline constexpr double kLoopCoverage           = 0.25;  // coverage at which the guard fires

// Appends the words of UTF-8 `bytes` to `words` as 64-bit hashes. A word is a run of ASCII letters
// and digits, '_' and non-ASCII bytes (ASCII letters lower-cased); every other non-space ASCII
// character is a word of its own; whitespace separates. `partial` holds a word a token boundary
// cut, completed by the next call.
void append_words(std::string_view bytes, std::string& partial, std::vector<std::uint64_t>& words);

// The share of the last kLoopWindowWords of `words` covered by kLoopPassageWords-word passages that
// occur at least kLoopRepeats times in `words` (the caller passes the last kLoopHistoryWords at
// most). Zero while fewer than kLoopWindowWords words exist.
[[nodiscard]] double repeated_passage_coverage(std::span<const std::uint64_t> words);

} // namespace ninfer::models::qwen3_5
