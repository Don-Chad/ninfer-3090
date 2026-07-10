#pragma once

// SM120 dense GDN prefill GEMM for the fixed Qwen3.6-27B shape:
//
//   a = Wa[48,5120] @ x[5120,T]
//   b = Wb[48,5120] @ x[5120,T]
//
// A CTA computes the same 16 output rows from both weights over 128 tokens.
// Split-K routes use a tuned eight- or sixteen-warp specialization and an
// in-kernel cooperative grid reduction; the unsplit long-context route uses
// eight warps for more independent MMA accumulators. Both preserve a single
// kernel launch.

#include "kernels/kernel/gdn_common.cuh"
#include "kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>
#include <cooperative_groups.h>

#include <cstdint>

namespace qus::kernels::detail {

inline constexpr int kDenseGdnHeads       = 48;
inline constexpr int kDenseGdnHidden      = 5120;
inline constexpr int kDenseGdnLogicalRows = 2 * kDenseGdnHeads;
inline constexpr int kDenseGdnBlockM      = 16;
inline constexpr int kDenseGdnBlockN      = 128;
inline constexpr int kDenseGdnBlockK      = 64;
inline constexpr int kDenseGdnWarps       = 16;
inline constexpr int kDenseGdnStages      = 2;
inline constexpr int kDenseGdnKTiles      = kDenseGdnHidden / kDenseGdnBlockK;
inline constexpr int kDenseGdnMFragments  = kDenseGdnBlockM / 16;
inline constexpr int kDenseGdnSmemElements =
    kDenseGdnStages * (kDenseGdnBlockN * kDenseGdnBlockK + 2 * kDenseGdnBlockM * kDenseGdnBlockK);
inline constexpr int kDenseGdnSmemBytes = kDenseGdnSmemElements * sizeof(__nv_bfloat16);

static_assert(kDenseGdnHidden % kDenseGdnBlockK == 0);

__device__ __forceinline__ int dense_gdn_swizzle(int row, int col) {
    return (col & ~63) + gemm_swz64(row, col & 63);
}

__device__ __forceinline__ void dense_gdn_async_copy_cg(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src));
#else
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
#endif
}

__device__ __forceinline__ void dense_gdn_async_copy_cg_zero(void* dst, const void* src,
                                                             int src_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src),
                   "r"(src_bytes));
#else
    if (src_bytes == 16) {
        *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
    } else {
        *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
    }
#endif
}

__device__ __forceinline__ float dense_gdn_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float dense_gdn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

template <int SplitK, bool FullTokens, int Warps>
__global__ __launch_bounds__(Warps * 32, 1) void linear_dense_gdn_in_ab_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ partial, float* __restrict__ g,
    float* __restrict__ beta, std::int32_t t) {
    static_assert(SplitK >= 1 && kDenseGdnKTiles % SplitK == 0,
                  "dense GDN split-K must divide the 80 K tiles");
    static_assert(Warps == 8 || Warps == 16);
    constexpr int kTilesPerSplit = kDenseGdnKTiles / SplitK;
    constexpr int kKSubtiles     = kDenseGdnBlockK / 16;
    constexpr int kThreads       = Warps * 32;
    constexpr int kWarpN         = kDenseGdnBlockN / Warps;
    constexpr int kNFragments    = kWarpN / 8;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* xs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* aws = xs + kDenseGdnStages * kDenseGdnBlockN * kDenseGdnBlockK;
    auto* bws = aws + kDenseGdnStages * kDenseGdnBlockM * kDenseGdnBlockK;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int head0    = static_cast<int>(blockIdx.y) * kDenseGdnBlockM;
    const int token0   = static_cast<int>(blockIdx.x) * kDenseGdnBlockN;
    const int split    = static_cast<int>(blockIdx.z);
    const int kt_begin = split * kTilesPerSplit;

    float a_acc[kDenseGdnMFragments][kNFragments][4] = {};
    float b_acc[kDenseGdnMFragments][kNFragments][4] = {};

    auto stage_load = [&](int stage, int kt) {
        const int k0 = kt * kDenseGdnBlockK;

        // x is shared by all 16 head rows and by both projections.
        for (int vec = tid; vec < kDenseGdnBlockN * (kDenseGdnBlockK / 8); vec += kThreads) {
            const int token_local = vec / (kDenseGdnBlockK / 8);
            const int k_vec       = vec - token_local * (kDenseGdnBlockK / 8);
            const int kk          = k_vec * 8;
            const int token       = token0 + token_local;
            __nv_bfloat16* dst =
                &xs[stage * kDenseGdnBlockN * kDenseGdnBlockK + token_local * kDenseGdnBlockK +
                    dense_gdn_swizzle(token_local, kk)];
            if constexpr (FullTokens) {
                dense_gdn_async_copy_cg(
                    dst, &x[static_cast<std::int64_t>(token) * kDenseGdnHidden + k0 + kk]);
            } else {
                const bool valid     = token < t;
                const int safe_token = valid ? token : 0;
                dense_gdn_async_copy_cg_zero(
                    dst, &x[static_cast<std::int64_t>(safe_token) * kDenseGdnHidden + k0 + kk],
                    valid ? 16 : 0);
            }
        }

        // Each thread moves one 16-byte vector: 128 vectors from Wa and 128
        // from Wb.  Both dense weights remain BF16 all the way into ldmatrix.
        const int weight_vecs = kDenseGdnBlockM * (kDenseGdnBlockK / 8);
        for (int all_vec = tid; all_vec < 2 * weight_vecs; all_vec += kThreads) {
            const bool is_b    = all_vec >= weight_vecs;
            const int vec      = is_b ? all_vec - weight_vecs : all_vec;
            const int row      = vec / (kDenseGdnBlockK / 8);
            const int k_vec    = vec - row * (kDenseGdnBlockK / 8);
            const int kk       = k_vec * 8;
            __nv_bfloat16* dst = (is_b ? bws : aws) + stage * kDenseGdnBlockM * kDenseGdnBlockK +
                                 row * kDenseGdnBlockK + dense_gdn_swizzle(row, kk);
            const __nv_bfloat16* weight = is_b ? b_weight : a_weight;
            dense_gdn_async_copy_cg(
                dst, &weight[static_cast<std::int64_t>(head0 + row) * kDenseGdnHidden + k0 + kk]);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kDenseGdnStages; ++stage) {
        if (stage < kTilesPerSplit) { stage_load(stage, kt_begin + stage); }
        qus::kernels::async_copy_commit();
    }

    // ldmatrix fragment lane offsets. A is [M,K], while the token-major x tile
    // is exactly the column-major B representation expected by mma.
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int x_rin    = lane & 7;
    const int x_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int it = 0; it < kTilesPerSplit; ++it) {
        const int stage = it & 1;
        qus::kernels::async_copy_wait<kDenseGdnStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kKSubtiles; ++ki) {
            unsigned xf[kNFragments][2];
#pragma unroll
            for (int ni = 0; ni < kNFragments; ++ni) {
                const int xrow = warp * kWarpN + ni * 8 + x_rin;
                const int xcol = ki * 16 + x_koff;
                gemm_ldmatrix_x2(
                    xf[ni][0], xf[ni][1],
                    gemm_smem_addr(&xs[stage * kDenseGdnBlockN * kDenseGdnBlockK +
                                       xrow * kDenseGdnBlockK + dense_gdn_swizzle(xrow, xcol)]));
            }
#pragma unroll
            for (int mi = 0; mi < kDenseGdnMFragments; ++mi) {
                unsigned af[4], bf[4];
                const int arow         = mi * 16 + a_rowoff;
                const int acol         = ki * 16 + a_coloff;
                const int weight_stage = stage * kDenseGdnBlockM * kDenseGdnBlockK;
                gemm_ldmatrix_x4(af[0], af[1], af[2], af[3],
                                 gemm_smem_addr(&aws[weight_stage + arow * kDenseGdnBlockK +
                                                     dense_gdn_swizzle(arow, acol)]));
                gemm_ldmatrix_x4(bf[0], bf[1], bf[2], bf[3],
                                 gemm_smem_addr(&bws[weight_stage + arow * kDenseGdnBlockK +
                                                     dense_gdn_swizzle(arow, acol)]));
#pragma unroll
                for (int ni = 0; ni < kNFragments; ++ni) {
                    gemm_mma_m16n8k16_bf16(a_acc[mi][ni][0], a_acc[mi][ni][1], a_acc[mi][ni][2],
                                           a_acc[mi][ni][3], af[0], af[1], af[2], af[3], xf[ni][0],
                                           xf[ni][1]);
                    gemm_mma_m16n8k16_bf16(b_acc[mi][ni][0], b_acc[mi][ni][1], b_acc[mi][ni][2],
                                           b_acc[mi][ni][3], bf[0], bf[1], bf[2], bf[3], xf[ni][0],
                                           xf[ni][1]);
                }
            }
        }

        __syncthreads();
        const int next = it + kDenseGdnStages;
        if (next < kTilesPerSplit) { stage_load(stage, kt_begin + next); }
        qus::kernels::async_copy_commit();
    }

#pragma unroll
    for (int mi = 0; mi < kDenseGdnMFragments; ++mi) {
        const int row0 = head0 + mi * 16 + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int ni = 0; ni < kNFragments; ++ni) {
            const int col0 = token0 + warp * kWarpN + ni * 8 + 2 * lid;
            const int col1 = col0 + 1;
            auto store     = [&](int token, int row, float av, float bv) {
                if constexpr (SplitK == 1) {
                    const float a = __bfloat162float(__float2bfloat16_rn(av));
                    const float b = __bfloat162float(__float2bfloat16_rn(bv));
                    const std::int64_t out_i =
                        static_cast<std::int64_t>(token) * kDenseGdnHeads + row;
                    g[out_i]    = -expf(A_log[row]) * dense_gdn_softplus(a + dt_bias[row]);
                    beta[out_i] = dense_gdn_sigmoid(b);
                } else {
                    const std::int64_t base =
                        (static_cast<std::int64_t>(split) * t + token) * kDenseGdnLogicalRows;
                    partial[base + row]                  = av;
                    partial[base + kDenseGdnHeads + row] = bv;
                }
            };
            if constexpr (FullTokens) {
                store(col0, row0, a_acc[mi][ni][0], b_acc[mi][ni][0]);
                store(col1, row0, a_acc[mi][ni][1], b_acc[mi][ni][1]);
                store(col0, row1, a_acc[mi][ni][2], b_acc[mi][ni][2]);
                store(col1, row1, a_acc[mi][ni][3], b_acc[mi][ni][3]);
            } else {
                if (col0 < t) {
                    store(col0, row0, a_acc[mi][ni][0], b_acc[mi][ni][0]);
                    store(col0, row1, a_acc[mi][ni][2], b_acc[mi][ni][2]);
                }
                if (col1 < t) {
                    store(col1, row0, a_acc[mi][ni][1], b_acc[mi][ni][1]);
                    store(col1, row1, a_acc[mi][ni][3], b_acc[mi][ni][3]);
                }
            }
        }
    }

    if constexpr (SplitK > 1) {
        cooperative_groups::this_grid().sync();
        const int block_linear = (static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.y) +
                                  static_cast<int>(blockIdx.y)) *
                                     static_cast<int>(gridDim.x) +
                                 static_cast<int>(blockIdx.x);
        const int grid_threads = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y) *
                                 static_cast<int>(gridDim.z) * kThreads;
        const int elems = kDenseGdnHeads * t;
        for (int i = block_linear * kThreads + tid; i < elems; i += grid_threads) {
            const int row   = i % kDenseGdnHeads;
            const int token = i / kDenseGdnHeads;
            float av        = 0.0f;
            float bv        = 0.0f;
#pragma unroll
            for (int s = 0; s < SplitK; ++s) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(s) * t + token) * kDenseGdnLogicalRows;
                av += partial[base + row];
                bv += partial[base + kDenseGdnHeads + row];
            }
            const float a = __bfloat162float(__float2bfloat16_rn(av));
            const float b = __bfloat162float(__float2bfloat16_rn(bv));
            g[i]          = -expf(A_log[row]) * dense_gdn_softplus(a + dt_bias[row]);
            beta[i]       = dense_gdn_sigmoid(b);
        }
    }
}

} // namespace qus::kernels::detail
