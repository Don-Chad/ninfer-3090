#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace qus::kernels::detail {

__global__ void mask_invalid_token_logits_kernel(__nv_bfloat16* logits, int row_stride,
                                                 int valid_vocab, int cols) {
    const int invalid_rows = row_stride - valid_vocab;
    const int index        = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total        = invalid_rows * cols;
    if (index >= total) { return; }
    const int col                                          = index / invalid_rows;
    const int row                                          = valid_vocab + index % invalid_rows;
    logits[static_cast<long long>(col) * row_stride + row] = __float2bfloat16_rn(-CUDART_INF_F);
}

} // namespace qus::kernels::detail
