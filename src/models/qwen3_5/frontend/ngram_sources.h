#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// De-numbers line-numbered tool output (`cat -n`, Claude Code's Read `  12→code`, `nl`, `L12: `)
// into additional n-gram proposal sources. Without it a tool result never matches the code the
// model later writes, because every line carries a number prefix.
//
// The frontend calls this only for tool-result text. The sources are proposal-only: they never
// rewrite a message, and the target prompt, positions, prefix identity and context-store identity
// are unchanged. A line is optional leading spaces, an optional `L`, a decimal number and exactly
// one separator (": ", tab, "→", " | " or "| "); a run is three or more consecutive numbers in one
// separator style. Each run's bodies are kept byte for byte after the separator, so indentation,
// CRLF and the final newline (or its absence) survive.
//
// Ported unchanged in behaviour from Infernix (Apache-2.0,
// src/models/qwen3_5/frontend/ngram_sources.h).
//
// `for_each_ngram_numbered_source` hands each run to `visit(std::string&&)` as soon as it is
// complete and stops scanning when `visit` returns false, so a caller with a token budget never
// materializes more than one run beyond what it keeps. `ngram_numbered_sources` collects them all.
template <typename Visitor>
inline void for_each_ngram_numbered_source(std::string_view text, Visitor&& visit) {
    struct Line {
        std::uint64_t number;
        std::string_view body;
        std::string_view style;
    };

    const auto parse = [](std::string_view line) -> std::optional<Line> {
        while (!line.empty() && line.front() == ' ') { line.remove_prefix(1); }
        if (!line.empty() && line.front() == 'L') { line.remove_prefix(1); }
        std::uint64_t number = 0;
        const auto parsed    = std::from_chars(line.data(), line.data() + line.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr == line.data()) { return std::nullopt; }
        auto rest = line.substr(static_cast<std::size_t>(parsed.ptr - line.data()));
        for (std::string_view delimiter : {": ", "\t", "\xe2\x86\x92", " | ", "| "}) {
            if (rest.starts_with(delimiter)) {
                return Line{number, rest.substr(delimiter.size()), delimiter};
            }
        }
        return std::nullopt;
    };
    bool stopped = false;
    std::string run;
    std::uint64_t previous = 0;
    std::string_view style;
    std::size_t count = 0;
    const auto flush  = [&] {
        if (count >= 3 && !visit(std::move(run))) { stopped = true; }
        run.clear();
        count = 0;
    };
    while (!text.empty() && !stopped) {
        auto length        = text.find('\n');
        const bool newline = length != std::string_view::npos;
        if (!newline) { length = text.size(); }
        const auto parsed = parse(text.substr(0, length));
        if (!parsed) {
            flush();
        } else {
            if (count != 0 && (previous == std::numeric_limits<std::uint64_t>::max() ||
                               parsed->number != previous + 1 || parsed->style != style)) {
                flush();
            }
            run.append(parsed->body);
            if (newline) { run.push_back('\n'); }
            previous = parsed->number;
            style    = parsed->style;
            ++count;
        }
        text.remove_prefix(length + (newline ? 1 : 0));
    }
    if (!stopped) { flush(); }
}

inline std::vector<std::string> ngram_numbered_sources(std::string_view text) {
    std::vector<std::string> sources;
    for_each_ngram_numbered_source(text, [&](std::string&& source) {
        sources.push_back(std::move(source));
        return true;
    });
    return sources;
}

} // namespace ninfer::models::qwen3_5::frontend
