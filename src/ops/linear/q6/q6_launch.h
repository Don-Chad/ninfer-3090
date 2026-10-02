#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q6Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// One warp per row, one or two tokens per weight read (q6_rowsplit_gemv.cu).
void launch_q6_gemv_t1(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_gemv_t2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// One m16 row tile per CTA, eight warps splitting K; up to 8 / 16 tokens (q6_small_t_mma.cu).
void launch_q6_small_t_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_small_t_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c16_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c24_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c32_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c40_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c48_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c56_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c64_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c72_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
