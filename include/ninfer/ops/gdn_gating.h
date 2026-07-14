#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Prepares Gated DeltaNet decay and update gates:
 *
 *   g[h,t]    = -exp(A_log[h]) * softplus(float(a[h,t]) + dt_bias[h])
 *   beta[h,t] = sigmoid(float(b[h,t])).
 *
 * `a` and `b` are contiguous BF16 [48,T], `A_log` and `dt_bias` are contiguous FP32 [48], and
 * `g` and `beta` are contiguous FP32 [48,T]. Transcendentals and outputs are FP32. Inputs and the
 * two outputs must be mutually non-overlapping. There is no workspace or persistent state side
 * effect.
 */
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops
