#include "ops/linear/q6/q6_rowsplit_kernels.h"

#include "ops/common/math.h"
#include "ops/linear/q6/q6_rowsplit_gemm_simt.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using SimtR8C4Schedule = Q6RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
// The 35B-A3B head has K=2048 and benefits from an extra cp.async stage, L2-only
// code loads, and an explicit two-block launch bound on GA102. Keep the K=5120
// dense-model schedule separate: the same settings regress its multi-column path.
using SimtR8C4K2048Sm86Schedule = Q6RowSplitSimtGemmSchedule<8, 4, 16, 3, Cache::cg, 2>;
using SimtR8C8Schedule = Q6RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <class Schedule>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(n, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(t, Schedule::kColsPerTile)), 1u);
    q6_rowsplit_gemm_simt_kernel<Schedule, Q6KernelVariant::Predicated>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q6_rowsplit_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    if (x.ne[0] == 2048) {
        launch_schedule<SimtR8C4K2048Sm86Schedule>(x, w, out, stream);
        return;
    }
    launch_schedule<SimtR8C4Schedule>(x, w, out, stream);
}

void q6_rowsplit_simt_r8_c8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_schedule<SimtR8C8Schedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
