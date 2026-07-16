#include "ops/linear/q6/q6_rowsplit_kernels.h"

#include "ops/common/math.h"
#include "ops/linear/gemm/linear_rowsplit_gemm_mma.cuh"
#include "ops/linear/reference/linear_generic.h"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using MmaR64C64Cfg  = GemmCfg<64, 64, 64, 32, 32, 2, 3>;
using MmaR64C128Cfg = GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>;

template <class Cfg>
void launch_variant(Q6KernelVariant variant, const Tensor& x, const Weight& w, Tensor& out,
                    cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    auto* outp                  = static_cast<__nv_bfloat16*>(out.data);
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(n, Cfg::BM)),
                    static_cast<unsigned>(div_up(t, Cfg::BN)), 1u);

    switch (variant) {
    case Q6KernelVariant::Full:
        linear_rowsplit_gemm_mma_kernel<Q6Codec, Cfg, true, false>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, high, scales, nullptr, outp, n, k, t,
                                                padded_k);
        break;
    case Q6KernelVariant::Predicated:
        linear_rowsplit_gemm_mma_kernel<Q6Codec, Cfg, false, false>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, high, scales, nullptr, outp, n, k, t,
                                                padded_k);
        break;
    case Q6KernelVariant::None:
        throw std::invalid_argument("q6 MMA launch requires Full or Predicated variant");
    default:
        throw std::invalid_argument("q6 MMA launch received an unknown variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q6_rowsplit_mma_r64_c64_launch(Q6KernelVariant variant, const Tensor& x, const Weight& w,
                                    Tensor& out, cudaStream_t stream) {
    launch_variant<MmaR64C64Cfg>(variant, x, w, out, stream);
}

void q6_rowsplit_mma_r64_c128_launch(Q6KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& out, cudaStream_t stream) {
    launch_variant<MmaR64C128Cfg>(variant, x, w, out, stream);
}

} // namespace ninfer::ops::detail
