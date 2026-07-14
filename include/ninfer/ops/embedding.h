#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Gathers one embedding row per token:
 *
 *   out[d,t] = BF16(dequantize(table)[ids[t],d]).
 *
 * `ids` is contiguous I32 [T], `out` is contiguous BF16 [D,T], and every id is in
 * [0,vocab). `table` has logical shape [vocab,D] and is either contiguous BF16_CTRL or
 * Q6G64_F16S RowSplit with group size 64 and FP16 scales. Dense BF16 values are copied exactly;
 * Q6 values are dequantized and rounded to BF16. `out` must not overlap `ids` or any table
 * plane. There is no workspace or persistent state side effect.
 */
void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
