#pragma once

#include "core/tensor.h"
#include "ops/linear/q6/q6_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q6_rowsplit_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream);

void q6_rowsplit_mma_r64_c64_launch(Q6KernelVariant variant, const Tensor& x, const Weight& w,
                                    Tensor& out, cudaStream_t stream);
void q6_rowsplit_mma_r64_c128_launch(Q6KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
