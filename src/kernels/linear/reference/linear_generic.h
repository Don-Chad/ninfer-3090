#pragma once

#include "kernels/linear/plan/linear_plan.h"   // LinearFormat
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

// Low-bit (Q4/Q5/Q6): launcher selects the codec by fmt. w carries payload/qdata/qhigh + padded_shape.
// GEMV is the T==1 (decode) generic path; the multi-step GEMM is the T>1 (prefill)
// path shared by every low-bit shape (dequant once per K-group, reuse across T).
void linear_generic_lowbit_gemv_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       LinearFormat fmt, cudaStream_t stream);
void linear_rowsplit_gemm_multistep_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           LinearFormat fmt, cudaStream_t stream);

// Dense (BF16/FP32): wrapper passes as_dense(w) as the weight Tensor.
void linear_generic_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);
void linear_generic_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);

} // namespace qus::kernels::detail
