#include "ops/linear/q6/q6_rowsplit_kernels.h"

#include "ops/common/math.h"
#include "ops/linear/gemv/linear_rowsplit_gemm_smallt.cuh"
#include "ops/linear/reference/linear_generic.h"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kColsPerTile  = 4;
constexpr int kStages       = 2;

} // namespace

void q6_rowsplit_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (k % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? k / 1024 : 0;
    constexpr int kBlockThreads   = kRowsPerBlock * 32;
    const dim3 grid(static_cast<unsigned>(div_up(n, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(t, kColsPerTile)), 1u);
    linear_rowsplit_gemm_smallt_kernel<Q6Smallt, kColsPerTile, kRowsPerBlock, kStages>
        <<<grid, kBlockThreads, 0, stream>>>(
            xp, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_k, full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
