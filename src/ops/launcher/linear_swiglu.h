#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void linear_rowsplit_q4_gate_up_silu_gemm_mma_launch(const Tensor& x, const Weight& weight,
                                                     Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
