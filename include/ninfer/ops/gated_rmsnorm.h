#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies RMS normalization over ne[0] and an elementwise SiLU gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d float(x[d,r])^2 + eps)
 *   out[d,r] = BF16(float(x[d,r]) * inv_r * float(weight[d]) * silu(float(z[d,r]))).
 *
 * `x`, `z`, and `out` are same-shaped contiguous BF16 tensors, `weight` is contiguous BF16 [D],
 * and eps is positive and finite. This form does not apply a unit offset to weight. Inputs and
 * output must not overlap. Computation is FP32 with one BF16 output round. There is no workspace
 * or persistent state side effect.
 */
void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream);

} // namespace ninfer::ops
