#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t weight_row_offset,
                               std::int32_t rows, Tensor& out, std::int32_t output_row_offset) {
    const std::int64_t groups = weight.padded_shape[1] / weight.group;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(weight_row_offset) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5_G64_FP16
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(weight_row_offset) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(weight_row_offset) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,
        high,
        scales,
        static_cast<__nv_bfloat16*>(out.data),
        rows,
        out.ne[0],
        output_row_offset,
        weight.qtype == QType::Q5_G64_FP16,
    };
}

// A block tile pads every column group to its full width, so the issued MMA work is set by
// BN, not by the live token count. On sm_86 the 128-wide tile is issue-bound well before it
// is bandwidth-bound, and a decode round never fills it: C8 with MTP3 is 32 columns. The
// narrow tiles cut the padded work proportionally over the extents a decode round uses,
// while the 128-wide tile stays the throughput anchor for prefill chunks.
// C8 and C16 continue that logic below 32, which is where every decode extent actually lives: a
// verification round is k+1 wide (6 at the recommended four draft tokens) and a C8 serving cohort
// is 8. At BN=8 the tile is one warp wide in N -- WARPS_N = BN/WN = 1, so 4 warps and 128 threads
// against C32's 16 warps and 512 -- which is the point: the padded MMA work falls with BN, and a
// tile that is 32 wide to serve 8 live columns issues four times the work it needs.
using GdnInputC8Schedule   = GemmCfg<64, 8, 64, 16, 8, 2, 2, false, true, true>;
using GdnInputC16Schedule  = GemmCfg<64, 16, 64, 16, 8, 2, 2, false, true, true>;
using GdnInputC32Schedule  = GemmCfg<64, 32, 64, 16, 8, 2, 2, false, true, true>;
using GdnInputC64Schedule  = GemmCfg<64, 64, 64, 16, 16, 2, 2, false, true, true>;
using GdnInputC128Schedule = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;

template <class Schedule>
void launch_slice(bool full, const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kValueRows = 6144;
    const RowSplitGroupedMmaJob qk    = make_job(qk_weight, 0, qk_weight.n, qkv, 0);
    const RowSplitGroupedMmaJob value = make_job(value_z_weight, 0, kValueRows, qkv, qk_weight.n);
    const RowSplitGroupedMmaJob output_gate =
        make_job(value_z_weight, kValueRows, kValueRows, z, 0);
    RowSplitGroupedMmaJob empty{};
    const int tiles = div_up(qk.n, Schedule::BM) + div_up(value.n, Schedule::BM) +
                      div_up(output_gate.n, Schedule::BM);
    const int cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, RowSplitGroupedMmaCodec::Mixed, 4>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     value, output_gate, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_grouped(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kTileCols = Schedule::BN;
    const bool full                  = (x.ne[1] % kTileCols) == 0;
    for_each_token_slice(x.ne[1], kTileCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor qkv_slice     = qkv.slice(1, offset, count);
        Tensor z_slice       = z.slice(1, offset, count);
        launch_slice<Schedule>(full, x_slice, qk_weight, value_z_weight, qkv_slice, z_slice,
                               stream);
    });
}

} // namespace

void q4_q5_gdn_input_grouped_mma_c8_launch(const Tensor& x, const Weight& qk_weight,
                                           const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                           cudaStream_t stream) {
    launch_grouped<GdnInputC8Schedule>(x, qk_weight, value_z_weight, qkv, z, stream);
}

void q4_q5_gdn_input_grouped_mma_c16_launch(const Tensor& x, const Weight& qk_weight,
                                            const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                            cudaStream_t stream) {
    launch_grouped<GdnInputC16Schedule>(x, qk_weight, value_z_weight, qkv, z, stream);
}

void q4_q5_gdn_input_grouped_mma_c32_launch(const Tensor& x, const Weight& qk_weight,
                                            const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                            cudaStream_t stream) {
    launch_grouped<GdnInputC32Schedule>(x, qk_weight, value_z_weight, qkv, z, stream);
}

void q4_q5_gdn_input_grouped_mma_c64_launch(const Tensor& x, const Weight& qk_weight,
                                            const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                            cudaStream_t stream) {
    launch_grouped<GdnInputC64Schedule>(x, qk_weight, value_z_weight, qkv, z, stream);
}

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        Q4Q5GdnInputScheduleId schedule, cudaStream_t stream) {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C8:
        q4_q5_gdn_input_grouped_mma_c8_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C16:
        q4_q5_gdn_input_grouped_mma_c16_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C32:
        q4_q5_gdn_input_grouped_mma_c32_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C64:
        q4_q5_gdn_input_grouped_mma_c64_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    // GdnInputC128Schedule is upstream's R64C128S2 configuration; the two ids share one kernel.
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128S2:
        launch_grouped<GdnInputC128Schedule>(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2:
        launch_grouped<GemmCfg<32, 32, 64, 16, 16, 2, 1, false, true, true>>(
            x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4:
        launch_grouped<GemmCfg<32, 64, 64, 16, 16, 4, 1, false, true, true>>(
            x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
    case Q4Q5GdnInputScheduleId::SmallTMma:
        break;
    }
    throw std::logic_error("Q4/Q5 GDN input: grouped MMA schedule is unknown");
}

} // namespace ninfer::ops::detail
