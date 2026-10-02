#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q4LinearAddLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);

// The individual routes, named so a route-boundary sweep can time the ones the table does not
// currently select (bench/ops/dense_linear_add_schedule_bench.cu).
void q4_linear_add_gemv_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_small_t_c8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_small_t_c16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_small_t_c32_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit4_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit32_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r32_c32_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r32_c64_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c48_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c64_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c80_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c96_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c112_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
