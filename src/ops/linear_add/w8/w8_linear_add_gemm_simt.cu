#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;

template <int ColsPerTile, W8KernelVariant Variant>
void launch_tt(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
               __nv_bfloat16* residual_out, std::int32_t rows, std::int32_t k, std::int32_t cols,
               std::int32_t padded_k, std::int32_t full_slabs, cudaStream_t stream) {
    constexpr int kThreads = kRowsPerBlock * 32;
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    const W8ContiguousOutput output{residual_out, rows};
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages,
                                 Variant, W8Epilogue::Residual><<<grid, kThreads, 0, stream>>>(
        x, codes, scales, output, rows, k, cols, padded_k, full_slabs);
}

template <int ColsPerTile>
void launch_variant(W8KernelVariant variant, const Tensor& x, const Weight& w, Tensor& residual_out,
                    cudaStream_t stream) {
    const auto* xp       = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (x.ne[0] % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? x.ne[0] / 1024 : 0;
    const auto* codes             = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales            = static_cast<const std::uint8_t*>(w.scales);
    auto* out                     = static_cast<__nv_bfloat16*>(residual_out.data);

    switch (variant) {
    case W8KernelVariant::Full:
        launch_tt<ColsPerTile, W8KernelVariant::Full>(xp, codes, scales, out, residual_out.ne[0],
                                                      x.ne[0], x.ne[1], w.padded_shape[1],
                                                      full_slabs, stream);
        break;
    case W8KernelVariant::Predicated:
        launch_tt<ColsPerTile, W8KernelVariant::Predicated>(xp, codes, scales, out,
                                                            residual_out.ne[0], x.ne[0], x.ne[1],
                                                            w.padded_shape[1], full_slabs, stream);
        break;
    case W8KernelVariant::None:
        throw std::invalid_argument("w8 linear_add SIMT requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_linear_add_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream) {
    launch_variant<4>(variant, x, w, residual_out, stream);
}

void w8_linear_add_simt_r8_c8_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream) {
    launch_variant<8>(variant, x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
