#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count);
std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

} // namespace ninfer::ops::detail
