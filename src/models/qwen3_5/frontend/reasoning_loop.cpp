#include "models/qwen3_5/frontend/reasoning_loop.h"

#include <algorithm>
#include <unordered_map>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint64_t fnv1a(std::string_view text) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

bool word_byte(unsigned char c) {
    return c >= 0x80 || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') || c == '_';
}

bool space_byte(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

} // namespace

void append_words(std::string_view bytes, std::string& partial, std::vector<std::uint64_t>& words) {
    for (const char raw : bytes) {
        const auto c = static_cast<unsigned char>(raw);
        if (word_byte(c)) {
            partial.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : raw);
            continue;
        }
        if (!partial.empty()) {
            words.push_back(fnv1a(partial));
            partial.clear();
        }
        if (!space_byte(c)) { words.push_back(fnv1a(std::string_view(&raw, 1))); }
    }
}

double repeated_passage_coverage(std::span<const std::uint64_t> words) {
    const std::size_t n = words.size();
    if (n < kLoopWindowWords || n < kLoopPassageWords) { return 0.0; }
    // Polynomial hash of every passage, rolled along the words.
    constexpr std::uint64_t kBase = 0x9E3779B97F4A7C15ULL;
    std::uint64_t top             = 1; // kBase^(kLoopPassageWords - 1)
    for (std::size_t i = 1; i < kLoopPassageWords; ++i) { top *= kBase; }
    const std::size_t passages = n - kLoopPassageWords + 1;
    std::vector<std::uint64_t> hashes(passages);
    std::uint64_t hash = 0;
    for (std::size_t i = 0; i < kLoopPassageWords; ++i) { hash = hash * kBase + words[i]; }
    hashes[0] = hash;
    for (std::size_t i = 1; i < passages; ++i) {
        hash      = (hash - words[i - 1] * top) * kBase + words[i + kLoopPassageWords - 1];
        hashes[i] = hash;
    }
    std::unordered_map<std::uint64_t, std::uint32_t> counts;
    counts.reserve(passages);
    for (const std::uint64_t h : hashes) { ++counts[h]; }

    // Mark the window's words that lie inside a repeated passage (a difference array over starts).
    const std::size_t window_start = n - kLoopWindowWords;
    std::vector<int> delta(kLoopWindowWords + 1, 0);
    const std::size_t first =
        window_start >= kLoopPassageWords - 1 ? window_start - (kLoopPassageWords - 1) : 0;
    for (std::size_t i = first; i < passages; ++i) {
        if (counts[hashes[i]] < kLoopRepeats) { continue; }
        const std::size_t begin = std::max(i, window_start) - window_start;
        const std::size_t end   = std::min(i + kLoopPassageWords, n) - window_start;
        ++delta[begin];
        --delta[end];
    }
    std::size_t covered = 0;
    int depth           = 0;
    for (std::size_t p = 0; p < kLoopWindowWords; ++p) {
        depth += delta[p];
        covered += depth > 0 ? 1U : 0U;
    }
    return static_cast<double>(covered) / static_cast<double>(kLoopWindowWords);
}

} // namespace ninfer::models::qwen3_5
