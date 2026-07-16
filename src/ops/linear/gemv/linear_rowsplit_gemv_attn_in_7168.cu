#include "ops/linear/gemv/linear_rowsplit_gemv_attn_in_7168.cuh"

#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/gemv/linear_rowsplit_gemv_q5_core.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kN = 7168;
constexpr int kK = 5120;

} // namespace

// Q5 attention-in: every one of the 7168 rows is identical work (K=5120, 5 tiles),
// so a single uniform one-row-per-warp pass over the shared cp.async-staged core.
void linear_rowsplit_gemv_attn_in_7168_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                 WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: Attention gate/v Q5 fused GEMV requires 7168x5120");
    }

    launch_q5_rowsplit_gemv<kN, kK, /*kRowsPerBlock=*/16, /*kStages=*/2, /*kStageX=*/true>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
