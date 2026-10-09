#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Nvfp4KvFamily { Grouped, ParallelGrouped, Tiled };

struct Nvfp4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Nvfp4KvFamily family;
    int query_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count);
std::size_t nvfp4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                     CausalAttentionExecutionEnvelope envelope,
                                     int multiprocessor_count);

} // namespace ninfer::ops::detail
