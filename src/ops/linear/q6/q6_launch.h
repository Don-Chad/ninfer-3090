#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
using Q6Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
// sm_86 vocabulary-head routes. One warp per row, one or two tokens per weight read
// (q6_rowsplit_gemv.cu).
void launch_q6_gemv_t1(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_gemv_t2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
// One m16 row tile per CTA, eight warps splitting K; up to 8 / 16 / 32 tokens (q6_small_t_mma.cu).
void launch_q6_small_t_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_small_t_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_small_t_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_a16_simt_r8_t4(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream);
void launch_q6_a16_gemv_r4_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream);
void launch_q6_a16_sliced_r16_t8_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_q6_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r32_t32_w4_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r32_t64_w2_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r16_t24_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r32_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_mma_r64_t40_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t48_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t56_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t64_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t72_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t80(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q6_a16_mma_r64_t96(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q6_a16_mma_r64_t128(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
void launch_q6_a16_mma_r64_t112(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
} // namespace ninfer::ops::detail
