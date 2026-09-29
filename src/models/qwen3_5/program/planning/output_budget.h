#pragma once

// Admission's Device KV page entitlement for one request, and the output budget that keeps every
// configured lane admissible at once. Pure arithmetic over fixed startup capacities, so both the
// request planner and the concurrent-budget query use exactly one formula.

#include "core/paged_kv_cache.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

struct KVPageEntitlement {
    std::uint32_t main_pages    = 0;
    std::uint32_t backend_pages = 0;
};

struct KVEntitlementShape {
    std::uint32_t capacity           = 0; // --max-context, each sequence's ceiling
    std::uint32_t draft_window       = 0;
    SpeculativeBackend backend       = SpeculativeBackend::None;
};

[[nodiscard]] constexpr std::uint32_t kv_pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// Pages reserved through completion for `prompt_tokens` plus an effective output of
// `output_tokens`: the last output token is never written to KV, MTP additionally reserves its
// draft window (bounded by the context), and DFlash mirrors the Main reservation.
[[nodiscard]] constexpr KVPageEntitlement
kv_page_entitlement(const KVEntitlementShape& shape, std::uint32_t prompt_tokens,
                    std::uint32_t output_tokens) noexcept {
    const std::uint32_t reserved = prompt_tokens + (output_tokens == 0 ? 0U : output_tokens - 1U);
    KVPageEntitlement out{.main_pages = kv_pages_for_tokens(reserved)};
    if (shape.backend == SpeculativeBackend::Mtp) {
        out.backend_pages = kv_pages_for_tokens(static_cast<std::uint32_t>(std::min<std::uint64_t>(
            shape.capacity, static_cast<std::uint64_t>(reserved) + shape.draft_window - 1ULL)));
    } else if (shape.backend == SpeculativeBackend::DFlash) {
        out.backend_pages = kv_pages_for_tokens(reserved);
    }
    return out;
}

// The largest output whose entitlement fits one lane's share of each pool (`*_share_pages` =
// pool pages / lanes), clamped to the remaining context `capacity - prompt + 1`. A prompt that alone
// overruns a lane's share can never run beside full-share lanes; it keeps the remaining context
// rather than an arbitrary cut. Zero only for an empty or over-long prompt.
[[nodiscard]] constexpr std::uint32_t
concurrent_output_budget(const KVEntitlementShape& shape, std::uint32_t main_share_pages,
                         std::uint32_t backend_share_pages, std::uint32_t prompt_tokens) noexcept {
    if (prompt_tokens == 0 || prompt_tokens > shape.capacity) { return 0; }
    const std::uint32_t remaining = shape.capacity - prompt_tokens + 1U;
    const auto fits = [&](std::uint32_t output) {
        const KVPageEntitlement need = kv_page_entitlement(shape, prompt_tokens, output);
        return need.main_pages <= main_share_pages && need.backend_pages <= backend_share_pages;
    };
    if (!fits(1U)) { return remaining; }
    // The entitlement is monotonic in the output, so the largest fitting budget is a bisection.
    std::uint32_t low  = 1U;
    std::uint32_t high = remaining;
    while (low < high) {
        const std::uint32_t mid = low + (high - low + 1U) / 2U;
        if (fits(mid)) {
            low = mid;
        } else {
            high = mid - 1U;
        }
    }
    return low;
}

} // namespace ninfer::models::qwen3_5::detail
