#include "ops/linear_add/q4/q4_linear_add_dispatch.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct ResidualEpilogue {
    __device__ __forceinline__ void operator()(__nv_bfloat16* destination, float value) const {
        *destination = __float2bfloat16_rn(__bfloat162float(*destination) + value);
    }
};

struct GemvResidualEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16*, int row,
                                               float value) const {
        static_assert(!SplitOutput);
        ResidualEpilogue{}(out + row, value);
    }
};

// The two registered weights share the 5120-row residual: the attention and GDN output projections
// (K=6144) and the MLP down projection (K=17408).
constexpr std::int32_t kRows      = 5120;
constexpr std::int32_t kMixerCols = 6144;
constexpr std::int32_t kDownCols  = 17408;

struct KSplitResidualEpilogue {
    __nv_bfloat16* residual;
    std::int32_t tokens;

    template <int Capacity>
    __device__ __forceinline__ void store(int row, int col, float4 value) const {
        if (col < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row, value.x);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row + 8, value.z);
        }
        if (col + 1 < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row, value.y);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row + 8,
                               value.w);
        }
    }
};

// K=6144 gives each of the eight warps 12 static groups; K=17408 would give 34, beyond the
// 16-group warp tile, so it uses the dynamic group loop (StaticGroupsPerRow 0).
template <std::int32_t Cols>
using GemvR1W8 =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, (Cols / 64 / 8 <= 16 ? Cols / 64 : 0), 1>;
using MmaR32C32  = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64  = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C128 = Q4RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
// The 64-row tiles the plain `linear` sweep found for this same geometry. Named rather than
// anonymous so bench/ops/dense_linear_add_schedule_bench.cu can time them against the routed
// choice without reimplementing the residual epilogue.
using MmaR64C48 = Q4RowSplitMmaGemmSchedule<64, 48, 64, 16, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C64 = Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C80 = Q4RowSplitMmaGemmSchedule<64, 80, 64, 16, 40, 2, 1, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C96 = Q4RowSplitMmaGemmSchedule<64, 96, 64, 32, 16, 2, 1, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C112 = Q4RowSplitMmaGemmSchedule<64, 112, 64, 32, 16, 2, 1,
                                             Q4FragmentPipeline::Serial, Cache::cg, Cache::cg,
                                             Q4ScaleLoad::Pair32>;

template <std::int32_t Cols, int Capacity>
void launch_ksplit_cols(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<kRows, Cols>;
    auto* output   = static_cast<__nv_bfloat16*>(residual.data);
    q4_ksplit_mma_kernel<Geometry, (Capacity + 7) / 8 * 8, Capacity, KSplitResidualEpilogue,
                         Q4KSplitIdentityRows, true>
        <<<kRows / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output,
            KSplitResidualEpilogue{output, x.ne[1]}, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int Capacity>
void launch_ksplit(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    if (w.k == kDownCols) {
        launch_ksplit_cols<kDownCols, Capacity>(x, w, residual, stream);
    } else {
        launch_ksplit_cols<kMixerCols, Capacity>(x, w, residual, stream);
    }
}

// Small-T MMA with direct loads: one m16 row tile per CTA, eight warps splitting K one group at a
// time, one or two eight-column MMA tiles. The k order and decode are the K-split kernel's
// (q4_ksplit_mma.cuh): lane lid takes k 16*lid .. 16*lid + 15 of its group, so its A operand is
// one 64-bit code load per row and its B operand sixteen contiguous activations. Each lane loads
// its codes and scales straight from global memory, the next group's loads issued before the
// current group's MMAs, so the weight stream does not depend on how many columns there are.
template <std::int32_t Cols, int Tiles>
__global__ void __launch_bounds__(256)
    q4_small_t_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                      const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ residual,
                      int tokens) {
    constexpr int kGroups = Cols / 64;
    __shared__ float partial[8][Tiles][32][4];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * 16;
    const std::int64_t ra = row0 + gid;
    const std::int64_t rb = row0 + gid + 8;

    struct Load {
        uint2 top, bot;
        std::uint16_t top_scale, bot_scale;
    };
    const auto load = [&](int g) {
        Load l;
        l.top       = load_ldg<uint2>(codes + ra * (Cols / 2) + g * 32 + 8 * lid);
        l.bot       = load_ldg<uint2>(codes + rb * (Cols / 2) + g * 32 + 8 * lid);
        l.top_scale = load_ldg<std::uint16_t>(scales + (ra * kGroups + g) * 2);
        l.bot_scale = load_ldg<std::uint16_t>(scales + (rb * kGroups + g) * 2);
        return l;
    };

    float acc[Tiles][4] = {};
    int g               = warp;
    Load next           = load(g);
#pragma unroll 1
    for (; g < kGroups; g += 8) {
        const Load cur = next;
        if (g + 8 < kGroups) { next = load(g + 8); }
        float group_acc[Tiles][4] = {};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            unsigned a_top[4], a_bot[4];
            Q4MmaDecodeAtom::decode_eight(half == 0 ? cur.top.x : cur.top.y, a_top);
            Q4MmaDecodeAtom::decode_eight(half == 0 ? cur.bot.x : cur.bot.y, a_bot);
#pragma unroll
            for (int nt = 0; nt < Tiles; ++nt) {
                const int col = nt * 8 + gid;
                uint4 b       = make_uint4(0u, 0u, 0u, 0u);
                if (col < tokens) {
                    b = load_ldg<uint4>(x + static_cast<std::int64_t>(col) * Cols + g * 64 +
                                        16 * lid + 8 * half);
                }
                float(&q)[4] = group_acc[nt];
                mma_bf16(q[0], q[1], q[2], q[3], a_top[0], a_bot[0], a_top[1], a_bot[1], b.x, b.y);
                mma_bf16(q[0], q[1], q[2], q[3], a_top[2], a_bot[2], a_top[3], a_bot[3], b.z, b.w);
            }
        }
        const float st = __half2float(__ushort_as_half(cur.top_scale));
        const float sb = __half2float(__ushort_as_half(cur.bot_scale));
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], st, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], st, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], sb, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], sb, acc[nt][3]);
        }
    }
#pragma unroll
    for (int nt = 0; nt < Tiles; ++nt) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { partial[warp][nt][lane][c] = acc[nt][c]; }
    }
    __syncthreads();
    if (warp < Tiles) {
        float v[4] = {};
#pragma unroll
        for (int w = 0; w < 8; ++w) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { v[c] += partial[w][warp][lane][c]; }
        }
        const int col = warp * 8 + 2 * lid;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int token = col + (c & 1);
            const int row   = row0 + gid + 8 * (c >> 1);
            if (token < tokens) {
                ResidualEpilogue{}(residual + static_cast<std::int64_t>(token) * kRows + row, v[c]);
            }
        }
    }
}

template <int Tiles>
void launch_small_t(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    const int tokens = x.ne[1];
    const auto* xp   = static_cast<const __nv_bfloat16*>(x.data);
    const auto* c    = static_cast<const std::uint8_t*>(w.qdata);
    const auto* sc   = static_cast<const std::uint8_t*>(w.scales);
    auto* out        = static_cast<__nv_bfloat16*>(residual.data);
    if (tokens < 1 || tokens > Tiles * 8) {
        throw std::invalid_argument("q4 linear_add small-T: unsupported token count");
    }
    if (w.k == kDownCols) {
        q4_small_t_kernel<kDownCols, Tiles><<<kRows / 16, 256, 0, stream>>>(xp, c, sc, out, tokens);
    } else {
        q4_small_t_kernel<kMixerCols, Tiles><<<kRows / 16, 256, 0, stream>>>(xp, c, sc, out, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_add_small_t_c8_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_small_t<1>(x, w, r, s);
}
void q4_linear_add_small_t_c16_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_small_t<2>(x, w, r, s);
}

void q4_linear_add_gemv_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    if (w.k == kDownCols) {
        launch_q4_gemv<GemvR1W8<kDownCols>, GemvResidualEpilogue>(x, w, r, s);
    } else {
        launch_q4_gemv<GemvR1W8<kMixerCols>, GemvResidualEpilogue>(x, w, r, s);
    }
}
void q4_linear_add_ksplit4_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<4>(x, w, r, s);
}
void q4_linear_add_ksplit8_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<8>(x, w, r, s);
}
void q4_linear_add_ksplit16_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<16>(x, w, r, s);
}
void q4_linear_add_ksplit24_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<24>(x, w, r, s);
}
void q4_linear_add_ksplit32_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<32>(x, w, r, s);
}
void q4_linear_add_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR32C32, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR32C64, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c48_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C48, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C64, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c80_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C80, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C96, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c112_launch(const Tensor& x, const Weight& w, Tensor& r,
                                       cudaStream_t s) {
    launch_q4_mma<MmaR64C112, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& r,
                                       cudaStream_t s) {
    launch_q4_mma<MmaR64C128, ResidualEpilogue>(x, w, r, s);
}

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (rows != kRows || (k != kMixerCols && k != kDownCols) || tokens <= 0) {
        throw std::invalid_argument("q4 linear_add: unsupported shape or token extent");
    }
    // K=17408 (MLP down) shares the table below. Swept 2026-10-02 on sm_86 with
    // bench/ops/dense_linear_add_schedule_bench.cu --q4 (cold, median of 20, a weight conversion
    // sharing the GPU): the routed schedule is the fastest at every measured T in 1..512 except
    // T=32, where ksplit_c32 leads mma_r32_c32 by 2%, below the 10% bar. T=1 runs at 65.5 us,
    // 722 GB/s of the 47 MB weight.
    // Re-measured on sm_86 2026-09-17 with bench/ops/dense_linear_add_schedule_bench.cu (--q4),
    // cold, median of 11. The narrow end and 33..64 are upstream's and hold here; the 65..192 band
    // repeats what the plain `linear` sweep found at this same geometry, that the 32-row tiles run
    // more than a wave behind the 64-row ones as soon as the extent needs a second column tile
    // (us, new vs shipped):
    //
    //   T=1   ksplit4 28.7 vs gemv 33.8 (+18%)
    //   T=12  ksplit24 50.2 vs ksplit16 71.7 (+43%)   T=16 57.3 vs 63.5 (+11%)
    //   T=80  r64_c80 144.4 vs r32_c32 205.8 (+43%)   T=72 153.6 vs 205.8 (+34%)
    //   T=96  r64_c96 160.8 vs r32_c32 194.6 (+21%)
    //   T=128 r64_c64 182.3 vs r32_c64 321.5 (+76%)   T=112 188.4 vs 322.6 (+71%)
    //   T=160 r64_c80 218.1 vs r32_c64 330.8 (+52%)   T=192 r64_c96 241.7 vs 329.7 (+36%)
    //
    // 25..32 moves to the 32x32 tile for 9%, which is below the 10% bar the rest of this sweep
    // used; it is taken because it merges a band rather than adding one.
    //
    // 9..16 (a two-lane MTP verify past K=3, three or four lanes at K=3) take the small-T MMA,
    // whose weight stream does not grow with the column count. Same bench, 2026-10-02, median of
    // 21, us, small_t_c16 vs ksplit24:
    //   K=6144   T=9 38.9 vs 46.1   T=12 41.0 vs 51.2   T=16 43.0 vs 58.4
    //   K=17408  T=9 89.1 vs 112.6  T=12 94.2 vs 124.9  T=16 112.6 vs 146.4
    // Through T=8 small_t_c8 ties the K-split kernels (66.6-67.6 vs 64.5-66.6 us at K=17408), so
    // they keep 1..8.
    if (tokens <= 4) return q4_linear_add_ksplit4_launch;
    if (tokens <= 8) return q4_linear_add_ksplit8_launch;
    if (tokens <= 16) return q4_linear_add_small_t_c16_launch;
    if (tokens <= 24) return q4_linear_add_ksplit24_launch;
    if (tokens <= 64) return q4_linear_add_mma_r32_c32_launch;
    if (tokens <= 80) return q4_linear_add_mma_r64_c80_launch;
    if (tokens <= 96) return q4_linear_add_mma_r64_c96_launch;
    if (tokens <= 128) return q4_linear_add_mma_r64_c64_launch;
    if (tokens <= 160) return q4_linear_add_mma_r64_c80_launch;
    if (tokens <= 192) return q4_linear_add_mma_r64_c96_launch;
    return q4_linear_add_mma_r64_c128_launch;
}

} // namespace ninfer::ops::detail
