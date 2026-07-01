// Launcher for the LargeT bf16 tensor-core GEMM (Q4/Q5/Q6 row-split), selected by
// fmt. Routed from the LargeT regime; the multi-step GEMV remains the SmallT path
// and the universal fallback.
#include "kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh"

#include "kernels/linear/reference/linear_generic.h" // launch declaration
#include "qus/core/device.h"                          // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kColTiles      = 8;               // tokens per block tile = 8 * kColTiles
constexpr int kBlockThreads  = kWarpsPerBlock * 32;
constexpr int kBM            = kWarpsPerBlock * 16; // output rows per block
constexpr int kBT            = kColTiles * 8;        // output tokens per block

int ceil_div(int a, int b) { return (a + b - 1) / b; }

} // namespace

void linear_rowsplit_gemm_mma_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     LinearFormat fmt, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const auto*        codes    = static_cast<const std::uint8_t*>(w.qdata);
    const auto*        high     = static_cast<const std::uint8_t*>(w.qhigh);
    const auto*        scales   = static_cast<const std::uint8_t*>(w.scales);
    const auto*        xp       = static_cast<const __nv_bfloat16*>(x.data);
    auto*              outp     = static_cast<__nv_bfloat16*>(out.data);

    const dim3 grid(static_cast<unsigned>(ceil_div(n, kBM)),
                    static_cast<unsigned>(ceil_div(t, kBT)), 1u);

    switch (fmt) {
    case LinearFormat::Q4G64_RowSplit:
        linear_rowsplit_gemm_mma_kernel<Q4Codec, kColTiles>
            <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    case LinearFormat::Q5G64_RowSplit:
        linear_rowsplit_gemm_mma_kernel<Q5Codec, kColTiles>
            <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    case LinearFormat::Q6G64_RowSplit:
        linear_rowsplit_gemm_mma_kernel<Q6Codec, kColTiles>
            <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    default:
        throw std::invalid_argument("linear: mma GEMM requires a Q4/Q5/Q6 row-split format");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
