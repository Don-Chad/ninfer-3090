#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Adds a four-corner interpolated position embedding. For d,p:
 *
 *   position[d,p] = BF16(sum_{c=0..3} float(table[d,indices[c,p]]) * weights[c,p])
 *   x'[d,p]       = BF16(float(x[d,p]) + float(position[d,p])).
 *
 * table is contiguous BF16 [D,R], indices is contiguous I32 [4,P] with values in [0,R), weights
 * is contiguous FP32 [4,P], and x is contiguous BF16 [D,P]. The intermediate BF16 round before
 * the residual add is part of the numeric contract. Inputs must not overlap x. The Op updates all
 * of x in place and uses no workspace or other persistent state.
 */
void vision_pos_embed_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                          Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
