#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void linear_rowsplit_gemv_mlp_gate_up_silu_17408_q4_launch(const Tensor& x, const Weight& w,
                                                           Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
