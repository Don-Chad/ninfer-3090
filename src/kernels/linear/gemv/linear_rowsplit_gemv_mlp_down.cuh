#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void linear_rowsplit_gemv_mlp_down_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);

} // namespace qus::kernels::detail
