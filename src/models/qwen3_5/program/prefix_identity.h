#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(ResidentPrefixIdentity& other) noexcept;
    void append_generated(std::size_t count, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;
    [[nodiscard]] bool equals(const ResidentPrefixIdentity& other) const;
    [[nodiscard]] bool prefix_equals(const ResidentPrefixIdentity& other, std::size_t count) const;

    // Session snapshots export the identity verbatim and restore it into a fresh sequence. The
    // restored identity must be internally consistent; it is not rederived from a prompt.
    [[nodiscard]] const std::vector<std::uint8_t>& token_types() const noexcept {
        return token_types_;
    }
    [[nodiscard]] const std::vector<std::int32_t>& position_axis(std::size_t axis) const {
        return positions_.at(axis);
    }
    [[nodiscard]] const std::vector<VisionItem>& vision_items() const noexcept {
        return vision_items_;
    }
    [[nodiscard]] const std::vector<std::uint32_t>& rewrite_execution_frontiers() const noexcept {
        return rewrite_execution_frontiers_;
    }
    void restore(std::vector<std::uint8_t> token_types,
                 std::array<std::vector<std::int32_t>, 3> positions,
                 std::vector<VisionItem> vision_items,
                 std::vector<std::uint32_t> rewrite_execution_frontiers);

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
    std::vector<std::uint32_t> rewrite_execution_frontiers_;
};

// One rolling digest per token frontier. This is only a content shortlist: exact token and
// ResidentPrefixIdentity comparison remains authoritative for reuse. Keeping it separate from the
// exact identity avoids retaining hash-only state in immutable capture backings.
class PrefixShortlistDigests {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(PrefixShortlistDigests& other) noexcept;
    void append_generated(std::span<const TokenId> tokens, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept {
        return digests_.empty() ? 0 : digests_.size() - 1U;
    }

    [[nodiscard]] std::array<std::uint64_t, 2> at(std::size_t frontier) const;

    // The full digest image (seed plus one entry per token) for session snapshots, so a restored
    // continuation shortlists under bit-identical keys.
    [[nodiscard]] const std::vector<std::array<std::uint64_t, 2>>& image() const noexcept {
        return digests_;
    }
    void restore(std::vector<std::array<std::uint64_t, 2>> image);

private:
    std::vector<std::array<std::uint64_t, 2>> digests_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  std::span<const TokenId> resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);

} // namespace ninfer::models::qwen3_5::detail
