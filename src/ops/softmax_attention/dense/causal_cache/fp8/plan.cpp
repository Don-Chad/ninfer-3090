#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 80;
} // namespace

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope,
                                        int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("FP8 attention: invalid plan inputs");
    constexpr int grouped_limit = Fp8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? Fp8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Fp8KvFamily::ParallelGrouped
                                                                     : Fp8KvFamily::Tiled;
    if (family == Fp8KvFamily::Tiled)
        return {
            family,
            heads,
            width,
            batch,
            envelope,
            mxfp8_tiled_partition(heads, width, envelope.max_visible_keys, multiprocessor_count)};
    const int tiles =
        family == Fp8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    // Decode budgets two CTAs per SM; wider queries start with one and double the
    // budget if complete query-tile groups would leave over 10% of the SMs idle.
    const std::int64_t sms = multiprocessor_count;
    const auto budget =
        width == 1 || causal_query_tiles_underfill_sms(independent_tiles, multiprocessor_count)
            ? 2 * sms
            : sms;
    CausalKvPartition partition{1, causal_partition_target(budget, independent_tiles)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, envelope, partition};
}

std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope,
                                   int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            make_fp8_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == Fp8KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return std::max(maximum, mxfp8_tiled_workspace_bytes(
                                 heads, std::max(min_width, kGroupedPrefillMaxWidth + 1), max_width,
                                 envelope.max_visible_keys, multiprocessor_count));
}

} // namespace ninfer::ops::detail
