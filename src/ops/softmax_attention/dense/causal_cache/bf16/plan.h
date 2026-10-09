#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/instances.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"

namespace ninfer::ops::detail {

struct Bf16KvCausalPlan {
    Bf16KvInstance instance;
    Bf16KvPartition partition;
    int query_heads;
    int width;
    int batch;
    CausalAttentionExecutionEnvelope envelope;

    bool grouped() const { return bf16_kv_instance_description(instance).grouped; }
};

Bf16KvCausalPlan make_bf16_kv_causal_plan(int query_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count);
std::size_t bf16_kv_workspace_bytes(int query_heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

} // namespace ninfer::ops::detail
