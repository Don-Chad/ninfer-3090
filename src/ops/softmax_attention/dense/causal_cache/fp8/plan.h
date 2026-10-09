#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Fp8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Fp8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Fp8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope,
                                        int multiprocessor_count);
std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope,
                                   int multiprocessor_count);

} // namespace ninfer::ops::detail
