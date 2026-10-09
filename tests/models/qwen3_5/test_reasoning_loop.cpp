// The reasoning-loop measure: words split across token boundaries join; a reasoning that keeps
// rewriting the same verification passes reaches the threshold; varied work, and a short text, stay
// below it; a passage repeated twice is not a loop.
#include "models/qwen3_5/frontend/reasoning_loop.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL " << label << '\n';
        ++failures;
    }
}

std::vector<std::uint64_t> words_of(const std::string& text, std::size_t piece = 3) {
    std::vector<std::uint64_t> words;
    std::string partial;
    for (std::size_t at = 0; at < text.size(); at += piece) {
        append_words(std::string_view(text).substr(at, piece), partial, words);
    }
    append_words(" ", partial, words);
    return words;
}

// Deterministic varied prose: a new combination of words in every sentence.
std::string varied(int sentences) {
    static const char* subjects[] = {"the parser", "this branch", "the cache", "our bound",
                                     "the loop",   "a tile",      "the queue", "that edge",
                                     "the schema", "each lane",   "the budget"};
    static const char* verbs[]    = {"handles", "skips", "rounds", "checks", "splits",
                                     "merges",  "drops", "keeps",  "orders", "limits",
                                     "scales",  "tracks", "maps"};
    static const char* objects[]  = {"empty input",      "the last token", "two columns",
                                     "a ragged width",   "every retry",    "the null case",
                                     "large values",     "unicode names",  "negative offsets",
                                     "the first page",   "stale entries",  "short reads"};
    std::string text;
    unsigned state = 12345;
    for (int s = 0; s < sentences; ++s) {
        state = state * 1103515245U + 12345U;
        text += std::string(subjects[(state >> 8) % 11]) + " " + verbs[(state >> 12) % 13] + " " +
                objects[(state >> 16) % 12] + " when step " + std::to_string(s) +
                " runs with value " + std::to_string((state >> 4) % 997) + ". ";
    }
    return text;
}

} // namespace

int main() {
    // Splitting: case folding, punctuation as words, a word cut by a token boundary.
    {
        std::vector<std::uint64_t> a, b;
        std::string pa, pb;
        append_words("Hello, World", pa, a);
        append_words("!", pa, a);
        append_words("hel", pb, b);
        append_words("lo , world!", pb, b);
        expect(a == b && a.size() == 4, "split and fold");
    }
    // The split does not depend on where token boundaries fall.
    {
        const std::string text = varied(50);
        expect(words_of(text, 1) == words_of(text, 7) && words_of(text, 7) == words_of(text, 64),
               "split depends on token boundaries");
    }
    // Below a full window nothing is measured.
    expect(repeated_passage_coverage(words_of(varied(20))) == 0.0, "short text");
    // Varied work stays low.
    const double novel = repeated_passage_coverage(words_of(varied(400)));
    expect(novel < 0.05, "varied reasoning coverage " + std::to_string(novel));
    // The same verification pass over and over, after some real work.
    const std::string pass =
        "Wait, let me double-check the edge case where the input list is empty. If the list is "
        "empty, the function returns zero, which matches the expected output. So that case is "
        "fine. ";
    std::string looping = varied(200);
    for (int i = 0; i < 40; ++i) { looping += pass; }
    const double loop = repeated_passage_coverage(words_of(looping));
    expect(loop >= kLoopCoverage, "looping coverage " + std::to_string(loop));
    // A passage seen twice is not repeated.
    std::string twice  = varied(300) + pass + varied(10) + pass + varied(150);
    const double recap = repeated_passage_coverage(words_of(twice));
    expect(recap < kLoopCoverage, "recap coverage " + std::to_string(recap));
    if (failures != 0) { return 1; }
    std::cout << "OK reasoning loop (varied " << novel << ", looping " << loop << ", recap "
              << recap << ")\n";
    return 0;
}
