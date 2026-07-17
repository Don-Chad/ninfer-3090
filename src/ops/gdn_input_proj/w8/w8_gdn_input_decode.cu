#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_k2048_decode.cuh"

namespace ninfer::ops::detail {

void w8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    constexpr int kRows       = 12288;
    constexpr int kRowsPerCta = 8;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    using Output = W8SplitOutput2<8192, 4096>;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    w8_k2048_decode_kernel<kRows, kRowsPerCta>
        <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
