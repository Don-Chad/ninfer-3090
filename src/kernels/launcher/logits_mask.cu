#include "kernels/launcher/logits_mask.h"

#include "kernels/kernel/logits_mask.cuh"
#include "qus/core/device.h"

namespace qus::kernels::detail {

void mask_invalid_token_logits_launch(Tensor& logits, std::int32_t valid_vocab,
                                      cudaStream_t stream) {
    const int invalid_rows = logits.ne[0] - valid_vocab;
    if (invalid_rows == 0 || logits.ne[1] == 0) { return; }
    constexpr int block = 256;
    const int total     = invalid_rows * logits.ne[1];
    const int grid      = (total + block - 1) / block;
    mask_invalid_token_logits_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data), logits.ne[0], valid_vocab, logits.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
