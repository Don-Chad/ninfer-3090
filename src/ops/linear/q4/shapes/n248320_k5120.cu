#include "ops/linear/q4/q4_shapes.h"

#include "core/device.h"
#include "ops/common/small_t_split_store.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"

#include <stdexcept>

// The vocabulary head transcoded from Q8 at load (artifact/transcode.h). Routes measured on sm_86.
namespace ninfer::ops::detail {
namespace {

using Geometry = Q4LinearGeometry<248320, 5120>;

// Rows carried by the store rather than the geometry; the 16-32-column tiles share staged
// activation slabs as gate_up does (q4_linear_swiglu_gemv.cu has that layout sweep).
template <int TileTokens, int KWarps, int Stages, int TilesPerWarp>
void launch_rows(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int rows = weight.n;
    const SmallTSplitStore store{static_cast<__nv_bfloat16*>(out.data),
                                 static_cast<__nv_bfloat16*>(out.data),
                                 rows,
                                 rows,
                                 rows,
                                 x.ne[1]};
    q4_ksplit_mma_launch<Geometry, TileTokens, TileTokens, SmallTSplitStore, Q4KSplitIdentityRows,
                         true, KWarps, Stages, TilesPerWarp>(
        rows / SmallTLayout<KWarps, TilesPerWarp>::kRowsPerCta, stream,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        store, Q4KSplitIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// Any K = 5120 matrix whose rows are a whole number of 128-row CTAs.
void launch_q4_small_t_rows(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    const int t = x.ne[1];
    if (weight.k != 5120 || weight.padded_shape[1] != weight.k || weight.n % 128 != 0 || t < 2 ||
        t > 32) {
        throw std::invalid_argument("Q4 Linear small-T rows: unsupported problem");
    }
    if (t <= 8) {
        launch_rows<8, 8, 1, 1>(x, weight, out, stream);
    } else if (t <= 16) {
        launch_rows<16, 4, 1, 1>(x, weight, out, stream);
    } else if (t <= 24) {
        launch_rows<24, 4, 2, 2>(x, weight, out, stream);
    } else {
        launch_rows<32, 2, 2, 1>(x, weight, out, stream);
    }
}

Q4Launch select_q4_n248320_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 32) return launch_q4_small_t_rows;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
