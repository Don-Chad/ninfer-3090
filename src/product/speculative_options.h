#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full ||
            options.context_lookup || options.context_lookup_verify_tokens != 0) {
            throw std::invalid_argument(
                "--draft-tokens, --lm-head-draft, and --context-lookup require --spec mtp|dflash");
        }
        return;
    case SpeculativeBackend::Mtp: {
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        if (!options.context_lookup && options.context_lookup_verify_tokens != 0) {
            throw std::invalid_argument(
                "--context-lookup-verify-tokens requires --context-lookup");
        }
        const std::uint32_t verify_tokens = options.context_lookup_verify_tokens == 0
                                                ? options.draft_tokens
                                                : options.context_lookup_verify_tokens;
        if (options.context_lookup &&
            (verify_tokens < options.draft_tokens || verify_tokens > 5)) {
            throw std::invalid_argument(
                "--context-lookup requires a verification window in [draft-tokens,5]");
        }
        return;
    }
    case SpeculativeBackend::DFlash:
        if (options.context_lookup || options.context_lookup_verify_tokens != 0) {
            throw std::invalid_argument("--context-lookup requires --spec mtp");
        }
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
