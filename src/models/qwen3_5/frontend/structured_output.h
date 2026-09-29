#pragma once

#include "ninfer/types.h"
#include "runtime/contract/token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xgrammar {
class GrammarMatcher;
}

namespace ninfer::models::qwen3_5::frontend {

class Tokenizer;
struct CompiledOutputFormat;

// Compiles OutputFormat values against this model's exact tokenizer. One instance is shared by a
// Frontend and its copies; it is safe to use from several threads. The tokenizer index is built
// on the first structured request, so an Engine that never receives one pays nothing for it.
// Compiled grammars are cached by their grammar text, so a repeated schema compiles once.
class StructuredOutputCompiler {
public:
    explicit StructuredOutputCompiler(std::shared_ptr<const Tokenizer> tokenizer);
    ~StructuredOutputCompiler();

    StructuredOutputCompiler(const StructuredOutputCompiler&)            = delete;
    StructuredOutputCompiler& operator=(const StructuredOutputCompiler&) = delete;

    // Throws RequestError(InvalidOutputFormat) for a schema XGrammar cannot represent. `format`
    // must not be Text.
    [[nodiscard]] std::shared_ptr<const CompiledOutputFormat> compile(const OutputFormat& format) const;

private:
    struct State;
    [[nodiscard]] State& state() const;

    std::shared_ptr<const Tokenizer> tokenizer_;
    mutable std::unique_ptr<State> state_;
    mutable std::once_flag state_once_;
};

// One request's structured-output state. The committed state mirrors the OutputSession's
// committed output; a preview is advanced token by token and then either committed with the
// session or dropped. The grammar applies after the reasoning section closes (immediately when the
// prompt opened none): the reasoning tokens themselves are unconstrained, as is the token that
// closes the section.
class StructuredOutputConstraint final : public runtime::TokenMaskSource {
public:
    // What a caller stop string does to one token that follows the speculative continuation.
    // `ends` is true when a stop string ends the output inside the token, whatever the channel or
    // byte mapping. `published` is the number of leading bytes of its decoded text that are still
    // published; nullopt when the byte mapping is not exact (a UTF-8 sequence split or repaired,
    // leading whitespace stripped, reasoning text), in which case the whole token is published.
    struct StopCut {
        bool ends = false;
        std::optional<std::size_t> published;
    };

    // For every candidate token, its StopCut when the output continues with `speculative` and then
    // that candidate. `cuts.size() == candidates.size()`. The first `always_count` candidates are
    // probed unconditionally; the rest only complete a stop string begun earlier, so they are
    // probed only while the decoder holds back a partial stop match and are left default otherwise.
    using StopPrefixProbe = std::function<void(std::span<const TokenId> speculative,
                                               std::span<const TokenId> candidates,
                                               std::size_t always_count,
                                               std::span<StopCut> cuts)>;

    // `content_stops` are the caller's stop strings that apply to the content channel, the only
    // channel the grammar constrains.
    StructuredOutputConstraint(std::shared_ptr<const Tokenizer> tokenizer,
                               std::shared_ptr<const CompiledOutputFormat> format,
                               std::vector<TokenId> stop_tokens, bool starts_in_reasoning,
                               std::span<const std::string> content_stops);
    ~StructuredOutputConstraint() override;

    StructuredOutputConstraint(const StructuredOutputConstraint&)            = delete;
    StructuredOutputConstraint& operator=(const StructuredOutputConstraint&) = delete;

    void begin_preview();
    // Advances the preview by one generated token. Returns false when the token is outside the
    // licensed language; the preview is then unusable. `stop_prefix_bytes`, when set, is the
    // number of leading bytes of this token's decoded text that a caller stop string still lets
    // through (0 when none of it will be published); only that prefix is validated instead of the
    // full token. Leave unset when no stop match was found in this token.
    [[nodiscard]] bool preview_token(TokenId token,
                                     std::optional<std::size_t> stop_prefix_bytes = std::nullopt);
    void commit_preview() noexcept;

    // Installs the decoder-side view of the stop strings. Mask construction uses it to license
    // tokens that a stop string cuts, on the same terms `preview_token` applies. Without a probe
    // (or without stop strings) masks judge every token by its full decoded text.
    void set_stop_prefix_probe(StopPrefixProbe probe) { stop_probe_ = std::move(probe); }

    [[nodiscard]] bool fill_token_masks(std::span<const TokenId> speculative,
                                        std::uint32_t columns, std::span<std::uint32_t> masks,
                                        std::size_t words_per_mask) override;

private:
    struct State;

    // True when a stop string ends the output inside `token` after `speculative`.
    [[nodiscard]] bool stop_cuts(std::span<const TokenId> speculative, TokenId token) const;
    void license_stop_cut_tokens(xgrammar::GrammarMatcher& matcher,
                                 std::span<const TokenId> speculative, std::span<std::uint32_t> mask,
                                 std::vector<TokenId>& probed, std::vector<StopCut>& cuts) const;

    std::shared_ptr<const Tokenizer> tokenizer_;
    std::unique_ptr<State> committed_;
    std::unique_ptr<State> preview_;
    // Vocabulary tokens whose decoded text can complete one of the stop strings. Only these can be
    // judged differently by a published prefix than by the full token. The first
    // `stop_contained_count_` contain a stop string outright; the rest only begin with the
    // remainder of a stop string whose first part was already emitted.
    std::vector<TokenId> stop_candidates_;
    std::size_t stop_contained_count_ = 0;
    StopPrefixProbe stop_probe_;
};

} // namespace ninfer::models::qwen3_5::frontend
