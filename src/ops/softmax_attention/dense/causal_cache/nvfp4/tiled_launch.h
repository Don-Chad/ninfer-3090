#pragma once
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands&, Nvfp4KvReadView, cudaStream_t);
} // namespace ninfer::ops::detail
