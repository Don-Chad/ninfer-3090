#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void linear_rowsplit_gemv_attn_kv_1024_q4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 cudaStream_t stream);
void linear_rowsplit_gemv_attn_kv_1024_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 cudaStream_t stream);

} // namespace qus::kernels::detail
