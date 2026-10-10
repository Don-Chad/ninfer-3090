// N-gram copy proposer and tool-result de-numbering (CPU only).
//
// The proposer and de-numbering cases are ported from Infernix (Apache-2.0,
// tests/models/qwen3_5/test_ngram_proposer.cpp); the capacity, per-request index and lazy ledger
// cases are this fork's.
#include "models/qwen3_5/frontend/ngram_sources.h"
#include "models/qwen3_5/program/speculative/ngram_proposer.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ninfer::models::qwen3_5::detail::NgramProposer;
using ninfer::models::qwen3_5::detail::NgramRequestIndex;
using ninfer::models::qwen3_5::detail::ngram_index_capacity;
using ninfer::models::qwen3_5::frontend::ngram_numbered_sources;
using Token = ninfer::TokenId;

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void test_proposer() {
    std::vector<Token> source(128);
    std::iota(source.begin(), source.end(), 100);
    NgramProposer proposer(256, 1024);
    proposer.ingest(source);
    require(proposer.propose(source, 0).tokens.empty(), "zero budget cannot propose tokens");
    for (const auto [capacity, buckets] :
         {std::pair<std::size_t, std::size_t>{63, 8}, {96, 8}, {64, 0}, {64, 3}}) {
        bool rejected = false;
        try {
            NgramProposer invalid(capacity, buckets);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid index dimensions must be rejected");
    }
    for (unsigned n : {4, 8, 12, 16, 32, 64}) {
        const auto match = proposer.propose(std::span<const Token>(source).first(n), 15, 4);
        require(match.matched == n, "backward match extension");
        require(match.tokens == std::vector<Token>(source.begin() + n, source.begin() + n + 15),
                "contiguous source proposal");
    }
    auto match = proposer.propose(std::span<const Token>(source).first(126), 15, 12);
    require(match.tokens.size() == 2, "stop at source boundary");
    require(proposer.propose(source, 15, 12).tokens.empty(), "no future source tokens");
    require(proposer.propose(std::span<const Token>(source).first(8), 15, 12).tokens.empty(),
            "admission threshold");

    // finish_round: lookahead +2, the window cap, and a short copy that offers no more drafts than
    // the neural proposal is dropped.
    std::uint32_t boundary_checks = 0;
    for (std::uint32_t available = 0; available <= 65; ++available) {
        std::vector<Token> bounded_source(64 + available);
        std::iota(bounded_source.begin(), bounded_source.end(), 1000);
        NgramProposer bounded(256, 1024);
        bounded.ingest(bounded_source);
        const auto history = std::span<const Token>(bounded_source).first(64);
        require(bounded.propose_for_round(history, 0, 5).tokens.empty(),
                "zero round budget cannot admit a draft");
        for (std::uint32_t maximum = 1; maximum <= 63; ++maximum) {
            for (std::uint32_t neural = 1; neural <= 15; ++neural) {
                const auto draft = bounded.propose_for_round(history, maximum, neural);
                const auto count = draft.tokens.size();
                require(count <= maximum && count <= available, "round admission budget");
                if (available >= maximum + 2) {
                    require(count == maximum, "full copy window unnecessarily shortened");
                } else if (count != 0) {
                    require(count > neural && count + 1 < available,
                            "target bonus reached the source boundary");
                } else {
                    require(available <= neural + 2, "useful copy tail discarded");
                }
                require(std::equal(draft.tokens.begin(), draft.tokens.end(),
                                   bounded_source.begin() + 64),
                        "admission changed source tokens");
                ++boundary_checks;
            }
        }
    }
    for (const auto maximum : {std::numeric_limits<std::uint32_t>::max() - 1U,
                               std::numeric_limits<std::uint32_t>::max()}) {
        bool rejected = false;
        try {
            (void)proposer.propose_for_round(source, maximum, 5);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "round lookahead overflow must be rejected");
    }
    std::cout << "round boundary checks=" << boundary_checks << '\n';

    NgramProposer special(256, 1024);
    special.set_boundaries({120});
    special.ingest(source);
    require(special.propose(std::span<const Token>(source).first(18), 15, 12).tokens.size() == 2,
            "stop before control token");

    // Pressure raw and derived sources with both general and patterned token IDs.
    for (const unsigned shift : {0U, 9U}) {
        std::vector<Token> old_source(2048), pressure(240000);
        std::mt19937 pressure_random(91026);
        const auto next_token = [&] {
            return static_cast<Token>((1U + pressure_random() % (150000U >> shift)) << shift);
        };
        for (auto& token : old_source) { token = next_token(); }
        for (auto& token : pressure) { token = next_token(); }
        for (bool derived : {false, true}) {
            NgramProposer retained;
            if (!derived) { retained.ingest(old_source); }
            retained.ingest(pressure);
            if (derived) { retained.ingest(old_source); }
            retained.ingest(pressure);
            unsigned hits = 0;
            for (std::size_t end = 64; end + 15 <= old_source.size(); ++end) {
                const auto tail  = std::span<const Token>(old_source).subspan(end - 64, 64);
                const auto draft = retained.propose(tail, 15);
                if (draft.tokens.empty()) { continue; }
                require(std::equal(draft.tokens.begin(), draft.tokens.end(),
                                   old_source.begin() + end),
                        "pressure changed the exact copied span");
                ++hits;
            }
            std::cout << "pressure shift=" << shift << " derived=" << derived << " hits=" << hits
                      << '\n';
            require(hits >= 1750, "default index lost old spans under long-context pressure");
        }
    }

    // Bulk ingest builds exactly the index token-by-token appends build: same proposals for every
    // history, across boundary tokens, ring wrap-around and bucket collisions.
    for (const auto [capacity, buckets] :
         {std::pair<std::size_t, std::size_t>{1U << 20, 1U << 19}, {256, 16}}) {
        std::mt19937 corpus_random(5501);
        std::vector<Token> corpus(20000);
        for (std::size_t i = 0; i < corpus.size(); ++i) {
            corpus[i] = i > 64 && corpus_random() % 3 == 0
                            ? corpus[i - 17 - corpus_random() % 40]
                            : static_cast<Token>(corpus_random() % 50);
        }
        const std::vector<Token> tool(corpus.begin() + 3000, corpus.begin() + 3400);
        NgramProposer bulk(capacity, buckets), stepwise(capacity, buckets);
        bulk.set_boundaries({7});
        stepwise.set_boundaries({7});
        bulk.ingest(corpus);
        bulk.ingest(tool);
        for (const auto& part : {std::span<const Token>(corpus), std::span<const Token>(tool)}) {
            stepwise.boundary();
            for (const Token token : part) { stepwise.append(token); }
            stepwise.boundary();
        }
        std::size_t proposed = 0;
        const auto compare_at = [&](std::size_t end) {
            const auto history =
                std::span<const Token>(corpus).subspan(end - std::min<std::size_t>(end, 64),
                                                       std::min<std::size_t>(end, 64));
            const auto a = bulk.propose(history, 15, 4);
            const auto b = stepwise.propose(history, 15, 4);
            require(a.tokens == b.tokens && a.matched == b.matched,
                    "bulk ingest built a different index than appends");
            proposed += a.tokens.empty() ? 0 : 1;
        };
        for (std::size_t end = 16; end <= corpus.size(); end += 7) { compare_at(end); }
        // The small ring still holds the tool span, so its histories must propose there.
        for (std::size_t end = 3016; end <= 3400; ++end) { compare_at(end); }
        std::cout << "bulk ingest ring=" << capacity << " proposals=" << proposed << '\n';
        // The small ring keeps 64 bucket entries, so collisions leave it few live windows.
        require(proposed >= (capacity > 256 ? 1000U : 10U),
                "bulk ingest comparison must exercise proposals");
    }

    // Independent live-ring oracle with forced hash collisions and repeated wrap-around.
    NgramProposer ring(128, 8);
    std::vector<Token> ledger;
    std::mt19937 random(72309);
    std::size_t proposals = 0;
    for (int step = 0; step < 100000; ++step) {
        const Token token = random() % 37 == 0 ? -1 : static_cast<Token>(random() % 9);
        ring.append(token);
        ledger.push_back(token);
        if (ledger.size() < 32) { continue; }
        const auto begin   = ledger.size() > 128 ? ledger.size() - 128 : 0;
        const auto end     = begin + 16 + random() % (ledger.size() - begin - 16);
        const auto history = std::span<const Token>(ledger).subspan(end - 16, 16);
        const auto draft   = ring.propose(history, 15, 4);
        if (draft.tokens.empty()) { continue; }
        ++proposals;
        bool found = false;
        for (auto pos = begin + draft.matched; pos + draft.tokens.size() <= ledger.size(); ++pos) {
            const auto tail = history.last(draft.matched);
            if (std::equal(tail.begin(), tail.end(), ledger.begin() + pos - draft.matched) &&
                std::equal(draft.tokens.begin(), draft.tokens.end(), ledger.begin() + pos)) {
                found = true;
                break;
            }
        }
        require(found, "collision/wrapped position proposed a nonexistent span");
        require(std::find(draft.tokens.begin(), draft.tokens.end(), -1) == draft.tokens.end(),
                "proposal crossed a boundary");
    }
    require(proposals > 100, "oracle must exercise actual proposals");
    std::cout << "oracle proposals=" << proposals << '\n';
}

void test_capacity() {
    // Sized from the context capacity: a power-of-two ring covering it, half as many buckets,
    // bounded below and by Infernix's fixed 2^20 ring.
    require(ngram_index_capacity(2'048).tokens == 4'096, "small contexts get the minimum ring");
    require(ngram_index_capacity(188'416).tokens == (1U << 18) &&
                ngram_index_capacity(188'416).buckets == (1U << 18),
            "188K context is a 2^18 ring with 2^18 buckets");
    require(ngram_index_capacity(196'608).tokens == (1U << 18), "196K context fits a 2^18 ring");
    require(ngram_index_capacity(262'144).tokens == (1U << 18), "an exact power of two is kept");
    require(ngram_index_capacity(262'145).tokens == (1U << 19), "one past rounds up");
    require(ngram_index_capacity(std::numeric_limits<std::uint32_t>::max()).tokens == (1U << 20),
            "the ring is bounded at 2^20");

    const NgramRequestIndex sized(ngram_index_capacity(188'416), {});
    const std::size_t ring_and_buckets = (std::size_t{1} << 18) * sizeof(Token) +
                                         (std::size_t{1} << 18) * 4 * 8;
    require(sized.bytes() >= ring_and_buckets && sized.bytes() < ring_and_buckets + 4096,
            "a 188K-context index is its ring and buckets (9 MiB)");
    std::cout << "index bytes at 188416 context=" << sized.bytes() << '\n';

    // A sequence that fills its context keeps its oldest spans proposable in the sized index.
    std::mt19937 random(1234);
    std::vector<Token> sequence(188'000);
    for (auto& token : sequence) { token = static_cast<Token>(1 + random() % 150000); }
    const auto capacity = ngram_index_capacity(188'416);
    NgramProposer filled(capacity.tokens, capacity.buckets);
    filled.ingest(sequence);
    unsigned hits = 0, probes = 0;
    for (std::size_t end = 64; end + 15 <= 4096; ++end, ++probes) {
        const auto history = std::span<const Token>(sequence).subspan(end - 64, 64);
        const auto draft   = filled.propose(history, 15);
        if (draft.tokens.empty()) { continue; }
        require(std::equal(draft.tokens.begin(), draft.tokens.end(), sequence.begin() + end),
                "the sized index proposed a span that is not in the sequence");
        ++hits;
    }
    std::cout << "sized index oldest-span hits=" << hits << '/' << probes << '\n';
    require(hits * 100 >= probes * 95, "the sized index lost the start of a full-context prompt");
}

void test_request_index() {
    std::mt19937 random(4242);
    std::vector<Token> prompt(6'000);
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = i > 64 && random() % 3 == 0 ? prompt[i - 17 - random() % 40]
                                                : static_cast<Token>(random() % 60);
    }
    const std::vector<Token> boundaries{3};
    const std::vector<std::vector<Token>> sources{
        std::vector<Token>(prompt.begin() + 100, prompt.begin() + 900)};
    // Output that alternates copied prompt spans with new text.
    std::vector<Token> generated;
    while (generated.size() < 3'000) {
        const std::size_t from = random() % (prompt.size() - 64);
        generated.insert(generated.end(), prompt.begin() + from, prompt.begin() + from + 48);
        for (int i = 0; i < 12; ++i) { generated.push_back(static_cast<Token>(random() % 60)); }
    }

    // Lazy catch-up at proposal time builds the index token-by-token appends build.
    const auto capacity = ngram_index_capacity(16'384);
    NgramRequestIndex lazy(capacity, boundaries);
    lazy.index_prompt(prompt, 0, sources);
    require(lazy.indexed() == prompt.size(), "the prompt is the indexed ledger prefix");
    NgramProposer eager(capacity.tokens, capacity.buckets);
    eager.set_boundaries(boundaries);
    eager.ingest(prompt);
    eager.ingest(sources[0]);
    std::vector<Token> ledger = prompt;
    std::size_t proposed      = 0;
    for (std::size_t i = 0; i < generated.size(); ++i) {
        ledger.push_back(generated[i]);
        eager.append(generated[i]);
        if (i % 3 != 0) { continue; } // rounds commit several tokens between proposals
        const auto a = lazy.propose_for_round(ledger, 15, 7, 8);
        const auto history =
            std::span<const Token>(ledger).last(std::min<std::size_t>(64, ledger.size()));
        const auto b = eager.propose_for_round(history, 15, 7, 8);
        require(a.tokens == b.tokens && a.matched == b.matched,
                "lazy ledger catch-up proposed differently from eager appends");
        require(lazy.indexed() == ledger.size(), "catch-up left committed tokens unindexed");
        proposed += a.tokens.empty() ? 0 : 1;
    }
    std::cout << "lazy catch-up proposals=" << proposed << '\n';
    require(proposed > 50, "the lazy catch-up comparison must exercise proposals");

    // A proposal-only source is proposable although the ledger never contained it in this form.
    std::vector<Token> numbered_prompt(512);
    std::iota(numbered_prompt.begin(), numbered_prompt.end(), 10'000);
    std::vector<Token> denumbered(64);
    std::iota(denumbered.begin(), denumbered.end(), 20'000);
    const std::vector<std::vector<Token>> tool_sources{denumbered};
    NgramRequestIndex with_source(ngram_index_capacity(4'096), {});
    with_source.index_prompt(numbered_prompt, 0, tool_sources);
    std::vector<Token> echo = numbered_prompt;
    echo.insert(echo.end(), denumbered.begin(), denumbered.begin() + 20);
    const auto copy = with_source.propose_for_round(echo, 15, 7, 12);
    require(copy.tokens == std::vector<Token>(denumbered.begin() + 20, denumbered.begin() + 35),
            "a proposal-only source did not propose its continuation");

    // Installed placeholders before `skip` are not text and are never proposed.
    std::vector<Token> grafted(256);
    std::iota(grafted.begin(), grafted.end(), 30'000);
    const auto grafted_index = [&] {
        NgramRequestIndex index(ngram_index_capacity(4'096), {});
        index.index_prompt(grafted, 128, {});
        return index;
    };
    auto placeholders = grafted_index();
    require(placeholders.indexed() == grafted.size(), "placeholders still count as ledger positions");
    std::vector<Token> repeat_placeholders = grafted;
    repeat_placeholders.insert(repeat_placeholders.end(), grafted.begin(), grafted.begin() + 20);
    require(placeholders.propose_for_round(repeat_placeholders, 15, 7, 12).tokens.empty(),
            "a graft placeholder was proposed as text");
    auto text = grafted_index();
    std::vector<Token> repeat_text = grafted;
    repeat_text.insert(repeat_text.end(), grafted.begin() + 128, grafted.begin() + 148);
    require(text.propose_for_round(repeat_text, 15, 7, 12).tokens ==
                std::vector<Token>(grafted.begin() + 148, grafted.begin() + 163),
            "text after the placeholders was not proposable");

    // A shorter ledger cannot be un-indexed; the index resynchronizes to it.
    std::vector<Token> shorter(grafted.begin(), grafted.begin() + 200);
    (void)text.propose_for_round(shorter, 15, 7, 12);
    require(text.indexed() == shorter.size(), "a shorter ledger did not resynchronize");
}

void test_numbered_sources() {
    for (const auto& delimiter : {std::string(": "), std::string("\t"),
                                  std::string("\xe2\x86\x92"), std::string(" | "),
                                  std::string("| ")}) {
        const auto inputs = "  10" + delimiter + "def f():\n  11" + delimiter + "    x = 1\n  12" +
                            delimiter + "    return x\n";
        const auto sources = ngram_numbered_sources(inputs);
        require(sources == std::vector<std::string>{"def f():\n    x = 1\n    return x\n"},
                "numbered tool text must preserve code indentation");
    }
    // Claude Code's Read output (`cat -n` with an arrow) and nl/`L`-prefixed styles.
    require(ngram_numbered_sources("     1\xe2\x86\x92#include <a>\n     2\xe2\x86\x92\n"
                                   "     3\xe2\x86\x92int x;\n") ==
                std::vector<std::string>{"#include <a>\n\nint x;\n"},
            "Read-style arrow numbering");
    require(ngram_numbered_sources("L7: a\nL8: b\nL9: c\n") == std::vector<std::string>{"a\nb\nc\n"},
            "L-prefixed numbering");
    require(ngram_numbered_sources("1: a\n2: b\n").empty(), "fewer than three lines give no source");
    require(ngram_numbered_sources("1: a\n2: b\n4: d\n").empty(), "truncated run");
    require(ngram_numbered_sources("a\nb\nc\n").empty(), "plain text stays raw");
    require(ngram_numbered_sources("1: a\n2: b\n3: c")[0] == "a\nb\nc",
            "no invented terminal newline");
    require(ngram_numbered_sources("L1: a\r\nL2: \r\nL3:     b\r\n") ==
                std::vector<std::string>{"a\r\n\r\n    b\r\n"},
            "CRLF and empty lines preserved");
    require(ngram_numbered_sources("1: a\n2: b\n3: c\n9: d\n10: e\n11: f\n") ==
                std::vector<std::string>{"a\nb\nc\n", "d\ne\nf\n"},
            "discontinuous runs remain separate");
    require(ngram_numbered_sources("1: a\n2\tb\n3: c\n").empty(), "mixed display styles rejected");
    require(ngram_numbered_sources("1: a\n2: b\n3: c\n4\td\n5\te\n6\tf\n") ==
                std::vector<std::string>{"a\nb\nc\n", "d\ne\nf\n"},
            "a style change splits the run");
    require(ngram_numbered_sources("18446744073709551615: a\n0: b\n1: c\n").empty(),
            "line numbering cannot wrap uint64");
    require(ngram_numbered_sources("18446744073709551616: a\n0: b\n1: c\n").empty(),
            "overflowing line number rejected");
    require(ngram_numbered_sources("\n    \nL\n: x\n").empty(),
            "empty and incomplete prefixes rejected");
    require(ngram_numbered_sources("1: caf\xc3\xa9\n2: cafe\xcc\x81\n3: \tvalue\n") ==
                std::vector<std::string>{"caf\xc3\xa9\ncafe\xcc\x81\n\tvalue\n"},
            "Unicode normalization and indentation bytes must not change");
    require(ngram_numbered_sources(std::string("1: a") + '\0' + "b\n2: c\n3: d") ==
                std::vector<std::string>{std::string("a") + '\0' + "b\nc\nd"},
            "embedded NUL must not truncate a proposal source");
}

} // namespace

int main() {
    try {
        test_proposer();
        test_capacity();
        test_request_index();
        test_numbered_sources();
        std::cout << "ngram proposer tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
