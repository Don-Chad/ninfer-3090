#pragma once

// qus::kernels - gdn_gating: Gated DeltaNet gate preparation.
// g = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h]), beta = sigmoid(b[h,t]).

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

// a/b: BF16 [48,T], A_log/dt_bias: FP32 [48], g/beta: FP32 [48,T], contiguous.
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace qus::kernels
