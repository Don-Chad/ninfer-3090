#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {

Bf16KvCausalPlan make_bf16_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope) {
    if ((heads != 24 && heads != 16) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("BF16 attention: invalid plan inputs");
    const int kv_heads = heads == 24 ? 4 : 2, group = heads / kv_heads;
    const auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const int visible   = envelope.max_visible_keys;
    if (width > 128 / group) {
        const auto instance =
            heads == 16 && width >= 1024 && width % 128 == 0 && visible - width >= 8192
                ? Bf16KvInstance::Tiled128
                : Bf16KvInstance::Tiled64;
        return {instance, {}, heads, width, batch, envelope};
    }
    const int rows         = width * group;
    const auto instance    = width == 1   ? Bf16KvInstance::GroupedDecode
                             : rows <= 32 ? Bf16KvInstance::Grouped32
                                          : Bf16KvInstance::Grouped64;
    const auto description = bf16_kv_instance_description(instance);
    const int tiles        = ceil_div(rows, description.query_rows);
    // Two SM waves suffice for short partitions. For long contexts, preserve
    // per-head parallelism until the whole grid reaches six waves on RTX 5090.
    const int independent_tiles     = batch * kv_heads * tiles;
    const bool many_memory_tiles    = description.query_rows == 32 && independent_tiles >= 16;
    const bool multiple_query_tiles = tiles > 1;
    const int long_ctas             = many_memory_tiles || multiple_query_tiles
                                          ? std::clamp(85 * independent_tiles, 340, 1020)
                                          : 340;
    Bf16KvPartition partition{1, std::clamp(340 / independent_tiles, 1, 256),
                              std::clamp(long_ctas / independent_tiles, 1, 256),
                              description.key_rows};
    // The envelope bounds the largest live row. Other batch rows may be shorter.
    const int low      = batch == 1 ? static_cast<int>(envelope.min_visible_keys) : 1;
    partition.capacity = std::max(
        partition.interval_capacity(low, std::min(visible, 32767), partition.normal_target),
        partition.interval_capacity(std::max(low, 32768), visible, partition.long_target));
    return {instance, partition, heads, width, batch, envelope};
}

std::size_t bf16_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    const int group     = heads == 24 ? 6 : 8;
    for (int width = min_width; width <= std::min(max_width, 128 / group); ++width) {
        const auto plan = make_bf16_kv_causal_plan(heads, width, batch, envelope);
        WorkspaceLayoutBuilder layout;
        (void)bf16_kv_allocate_partials(layout, heads, width, plan.partition.capacity, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
