#pragma once

#include "ninfer/types.h"
#include "runtime/contract/token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

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
    StructuredOutputConstraint(std::shared_ptr<const Tokenizer> tokenizer,
                               std::shared_ptr<const CompiledOutputFormat> format,
                               std::vector<TokenId> stop_tokens, bool starts_in_reasoning);
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

    [[nodiscard]] bool fill_token_masks(std::span<const TokenId> speculative,
                                        std::uint32_t columns, std::span<std::uint32_t> masks,
                                        std::size_t words_per_mask) override;

private:
    struct State;
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::unique_ptr<State> committed_;
    std::unique_ptr<State> preview_;
};

} // namespace ninfer::models::qwen3_5::frontend
