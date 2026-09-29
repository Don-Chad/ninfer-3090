#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/small_t_split_store.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kValueZRows = kValueRows + kZRows;
constexpr std::int32_t kHidden     = 5120;

std::int32_t leading_dimension(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

struct GdnQkGeometry {
    static constexpr int kInputRows    = kHidden;
    static constexpr int kGroupsPerRow = kHidden / 64;
};

template <int XCols, int Stages, int KWarps = 8>
void launch_value_z(const Tensor& x, const Weight& w, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    // value_z rows [0, 6144) are the GDN value projection and [6144, 12288) its z gate.
    const SmallTSplitStore epilogue{static_cast<__nv_bfloat16*>(value.data),
                                    static_cast<__nv_bfloat16*>(z.data),
                                    leading_dimension(value),
                                    leading_dimension(z),
                                    kValueRows,
                                    x.ne[1]};
    q5_small_t_mma_launch<kValueZRows, kHidden, XCols, Stages, SmallTSplitStore, KWarps>(
        stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), epilogue, x.ne[1]);
}

template <int TileCols, int KWarps = 8, int Stages = 1>
void launch_qk(const Tensor& x, const Weight& w, Tensor& qk, cudaStream_t stream) {
    const SmallTSplitStore epilogue{static_cast<__nv_bfloat16*>(qk.data),
                                    static_cast<__nv_bfloat16*>(qk.data),
                                    leading_dimension(qk),
                                    leading_dimension(qk),
                                    kQkRows,
                                    x.ne[1]};
    q4_ksplit_mma_launch<GdnQkGeometry, TileCols, TileCols, SmallTSplitStore,
                         Q4KSplitIdentityRows, true, KWarps, Stages>(
        kQkRows / SmallTLayout<KWarps>::kRowsPerCta, stream,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(qk.data),
        epilogue, Q4KSplitIdentityRows{}, x.ne[1]);
}

} // namespace

void q4_q5_gdn_input_small_t_launch(const Tensor& x, const Weight& qk_weight,
                                    const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                    Tensor& z, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4/Q5 GDN input small-T MMA requires T in [1,32]");
    }
    if (x.ne[0] != kHidden || qk_weight.n != kQkRows || value_z_weight.n != kValueZRows ||
        qk_weight.padded_shape[1] != kHidden || value_z_weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 GDN input small-T MMA: unsupported shape");
    }
    // 16 and 32 columns share each staged activation slab between tiles (KWarps 4), which at 32
    // columns cuts what a CTA pulls through L2 by half. One kernel, cold L2, median of 21, us:
    //
    //                      T   kwarps 8          kwarps 4
    //   value_z [12288]   16   79.9              72.7 (s3)
    //   value_z [12288]   32  156.7             117.8
    //   query_key [4096]  16   47.1 / 42.0 (s2)  25.6 (s2)
    //   query_key [4096]  32   75.8             38.9 (s2)
    const int columns = x.ne[1];
    if (columns <= 4) {
        launch_value_z<4, 2>(x, value_z_weight, value, z, stream);
    } else if (columns <= 8) {
        launch_value_z<8, 1>(x, value_z_weight, value, z, stream);
    } else if (columns <= 16) {
        launch_value_z<16, 3, 4>(x, value_z_weight, value, z, stream);
    } else {
        launch_value_z<32, 1, 4>(x, value_z_weight, value, z, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    if (columns <= 8) {
        launch_qk<8>(x, qk_weight, qk, stream);
    } else if (columns <= 16) {
        launch_qk<16, 4, 2>(x, qk_weight, qk, stream);
    } else {
        launch_qk<32, 4, 2>(x, qk_weight, qk, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
