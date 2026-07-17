#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows       = 12288;
constexpr int kHidden     = 2048;
constexpr int kRowsPerCta = 8;
constexpr int kStages     = 2;
constexpr int kCols       = 4;
using Output              = W8SplitOutput2<8192, 4096>;

template <W8KernelVariant Variant>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    const dim3 grid(kRows / kRowsPerCta, static_cast<unsigned>(div_up(x.ne[1], kCols)), 1u);
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, kCols, kRowsPerCta, kStages, Variant,
                                 W8Epilogue::Store, Output><<<grid, kRowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, kRows, kHidden, x.ne[1], kHidden,
        kHidden / 1024);
}

} // namespace

void w8_gdn_input_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                    Tensor& qkv, Tensor& z, cudaStream_t stream) {
    if (variant == W8KernelVariant::Full) {
        launch_variant<W8KernelVariant::Full>(x, weight, qkv, z, stream);
    } else if (variant == W8KernelVariant::Predicated) {
        launch_variant<W8KernelVariant::Predicated>(x, weight, qkv, z, stream);
    } else {
        throw std::invalid_argument("W8 GDN input SIMT requires a tiled variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
