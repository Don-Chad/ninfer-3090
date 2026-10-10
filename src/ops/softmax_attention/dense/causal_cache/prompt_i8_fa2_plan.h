#pragma once

// Host plan of the FA2-style INT8-G64 prompt kernel (prompt_i8_fa2.cuh): its CTA shape and key
// splits, and the FP32 partials a split launch publishes. Shared by the launcher, the workspace
// capacity query and the launch-shape query so all three see the same route facts.
//
// The plan's structure follows Infernix's fast_tiled_plan.h / fast_prompt_plan.h
// (github.com/wallawalla47/infernix, commit dbd1f374, Apache-2.0); its cost terms and constants were
// re-measured on the RTX 3090 (see kCausalPromptFa2* below) and its SM count comes from the
// executing device's DeviceExecutionView rather than a process-wide cudaGetDevice.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// A prompt launch: warps per CTA (16 query rows each) and key splits. More than one split divides
// every row block's key pages among CTAs and merges their FP32 partial rows, so a launch whose row
// blocks alone would leave SMs idle still fills them.
struct CausalPromptFa2Plan {
    std::int32_t warps  = 8;
    std::int32_t splits = 1;

    friend bool operator==(const CausalPromptFa2Plan&, const CausalPromptFa2Plan&) = default;
};

// Splits only for widths that no graph-captured verify call can request (W <= 16 stays one
// two-kernel topology for every envelope).
inline constexpr std::int32_t kCausalPromptFa2MinSplitWidth = 17;
inline constexpr std::int32_t kCausalPromptFa2MaxSplits     = 8;
// Every split keeps at least this many 64-key pages.
inline constexpr std::int32_t kCausalPromptFa2MinPagesPerSplit = 8;
// Internal ceiling on the FP32 partials of one split launch.
inline constexpr std::size_t kCausalPromptFa2SplitBudgetBytes = std::size_t{64} << 20;

// Cost model, in units of one eight-warp CTA's sweep over one visible key; a launch takes
// waves x (keys / splits) x rate.
//   * A CTA whose rows occupy at most four warps -- every four-warp CTA, and an eight-warp CTA of
//     a launch narrower than 65 columns -- sweeps a key at kCausalPromptFa2NarrowSweep.
//   * A split sweeps its keys at kCausalPromptFa2SplitSweep and also writes and merges its FP32 row,
//     priced per (column, split, query head) at kCausalPromptFa2SplitPerRowHead.
//   * Waves blend the whole-wave count with the fractional one (kCausalPromptFa2WholeWaves).
// Re-derived on the RTX 3090 (82 SMs, 2026-10-10) from every CTA shape and split count 1-8 forced
// at 17-2048 columns over 4K/32K/128K cached keys, both geometries, two passes (cell pass-to-pass
// median 1.1 %). Against the fastest forced candidate the plan is 1.5 % slower on average and
// never slower than the best unsplit launch; its worst cells are 21 % (H16, 2048 columns over
// 128K keys, where the faster split counts exceed the split budget) and 12-14 % (H24 192-320
// columns and H16 192 columns over 128K keys, whose light partial last row block the model does
// not price). Infernix's 5090 constants (0.8, 0.5/24, whole waves) scored 3.8 % / 33 % with 17
// cells slower than unsplit on the same data.
inline constexpr double kCausalPromptFa2NarrowSweep     = 0.65;
inline constexpr double kCausalPromptFa2SplitSweep      = 1.05;
inline constexpr double kCausalPromptFa2SplitPerRowHead = 1.0 / 24.0;
inline constexpr double kCausalPromptFa2WholeWaves      = 0.85;
// At or below this many visible keys the INT8-G64 cache keeps the tiled prompt kernel
// (prompt_i8.cuh): there a launch is one or a few key tiles and the FA2 CTA's per-warp Q encoding
// (16 rows per warp against the tiled kernel's 4) dominates, 1.1-1.5x slower at 7-128 columns.
inline constexpr std::uint32_t kCausalPromptFa2MinVisibleKeys = 256;

struct CausalPromptFa2Partials {
    Tensor rows;
    Tensor stats;
};

// The normalized FP32 D256 row of every (query head, column, split) and its (max, sum) pair.
template <class Allocator>
CausalPromptFa2Partials allocate_causal_prompt_fa2_partials(Allocator& workspace,
                                                            std::int32_t q_heads,
                                                            std::int32_t width,
                                                            std::int32_t splits) {
    return {workspace.alloc(DType::FP32, {256, q_heads, width, splits}),
            workspace.alloc(DType::FP32, {2, q_heads, width, splits})};
}

inline std::size_t causal_prompt_fa2_split_bytes(std::int32_t q_heads, std::int32_t width,
                                                 std::int32_t splits) {
    if (splits <= 1) return 0;
    WorkspaceLayoutBuilder layout;
    (void)allocate_causal_prompt_fa2_partials(layout, q_heads, width, splits);
    return layout.peak_bytes(1);
}

inline std::int32_t causal_prompt_fa2_pages(std::uint32_t visible_keys) {
    return static_cast<std::int32_t>(
        (static_cast<std::uint64_t>(visible_keys) + kPagedKVPageSize - 1) / kPagedKVPageSize);
}

// Plan for one launch of `width` columns whose last row sweeps `visible_keys` keys. Every candidate
// cost is linear in the key count, with a constant term (the split merge) that grows with the split
// count, so fewer keys never select more splits; the plan at an envelope's maximum therefore bounds
// every launch in it.
inline CausalPromptFa2Plan causal_prompt_fa2_plan(std::int32_t q_heads, std::int32_t width,
                                                  std::uint32_t visible_keys,
                                                  std::int32_t multiprocessor_count) {
    const auto waves = [&](std::int64_t ctas) {
        const double whole =
            static_cast<double>((ctas + multiprocessor_count - 1) / multiprocessor_count);
        const double fraction = static_cast<double>(ctas) / multiprocessor_count;
        return kCausalPromptFa2WholeWaves * whole + (1.0 - kCausalPromptFa2WholeWaves) * fraction;
    };
    const double keys        = static_cast<double>(visible_keys);
    const std::int32_t pages = causal_prompt_fa2_pages(visible_keys);

    const std::int64_t narrow_ctas = static_cast<std::int64_t>((width + 63) / 64) * q_heads;
    CausalPromptFa2Plan best{4, 1};
    double best_cost = waves(narrow_ctas) * kCausalPromptFa2NarrowSweep * keys;
    const std::int64_t ctas = static_cast<std::int64_t>((width + 127) / 128) * q_heads;
    const double rate       = width <= 64 ? kCausalPromptFa2NarrowSweep : 1.0;
    const std::int32_t max_splits =
        width >= kCausalPromptFa2MinSplitWidth ? kCausalPromptFa2MaxSplits : 1;
    for (std::int32_t splits = 1; splits <= max_splits; ++splits) {
        if (splits > 1 &&
            (pages < splits * kCausalPromptFa2MinPagesPerSplit ||
             causal_prompt_fa2_split_bytes(q_heads, width, splits) >
                 kCausalPromptFa2SplitBudgetBytes)) {
            break;
        }
        double cost = waves(ctas * splits) * rate * keys / splits;
        if (splits > 1) {
            cost = cost * kCausalPromptFa2SplitSweep +
                   kCausalPromptFa2SplitPerRowHead * width * splits * q_heads;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best      = {8, splits};
        }
    }
    return best;
}

} // namespace ninfer::ops::detail
