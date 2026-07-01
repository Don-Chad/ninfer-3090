#pragma once

// LargeT tensor-core GEMM: out[N,T] = W[N,K] . x[K,T], W is Q4/Q5/Q6 row-split,
// x/out bf16. bf16 mma.sync (m16n8k16, fp32 accumulate) with the low-bit weights
// dequantized on-chip from the existing row-split layout (single VRAM copy; no
// repack). Dequant math is identical to Codec::load_group / the reference.
//
// Numerics: bf16 operands (A16-native, and 2x the tf32 throughput). This rounds
// the dequantized weight to bf16; the fp64 golden keeps it in fp32, so the bf16
// path shows a slightly larger per-element error on near-zero (cancellation)
// outputs. That is intrinsic to a low-precision GEMM, not a bug: the normwise
// error ||a-b||2/||b||2 stays ~2e-3 (identical to the fp32 multi-step path). The
// tensor-core linear parity is therefore judged by a normwise criterion
// (Tolerance::linear_tc), per docs/l1-op-test-standard.md.
//
// Correctness-first version (P2T1): one warp owns a 16(rows) x (8*kColTiles)(tokens)
// output tile, contracting K in steps of 16. The A operand (dequantized W, 16x16)
// is assembled once per K-step and reused across the kColTiles token subtiles; the
// B operand (x, 16x8) is assembled per subtile. Fragments are assembled directly
// per the documented m16n8k16 lane layout (no ldmatrix/shared) to keep the first
// version obviously correct; the roofline pushdown (shared dequant tiles, ldmatrix,
// cp.async, larger reuse) is P2T3.

#include "kernels/linear/codec/linear_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

__device__ __forceinline__ unsigned gemm_mma_pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 v;
    v.x = __float2bfloat16(lo);
    v.y = __float2bfloat16(hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ void gemm_mma_m16n8k16_bf16(float& c0, float& c1, float& c2, float& c3,
                                                       unsigned a0, unsigned a1, unsigned a2,
                                                       unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Dequantize one weight element W[row][k]; math identical to Codec::load_group.
template <class Codec>
__device__ __forceinline__ float gemm_dequant_w(const std::uint8_t* codes,
                                                const std::uint8_t* high,
                                                const std::uint8_t* scales, int row, int k,
                                                int kg) {
    const std::int64_t gi = static_cast<std::int64_t>(row) * kg + (k >> 6); // k / 64
    const int          j  = k & 63;
    float              w0 = 0.0f;
    float              w1 = 0.0f;
    Codec::load_pair(codes, high, scales, gi, j >> 1, w0, w1);
    return (j & 1) ? w1 : w0;
}

template <class Codec, int kColTiles>
__global__ void linear_rowsplit_gemm_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                                const std::uint8_t* __restrict__ codes,
                                                const std::uint8_t* __restrict__ high,
                                                const std::uint8_t* __restrict__ scales,
                                                __nv_bfloat16* __restrict__ out, std::int32_t n,
                                                std::int32_t k, std::int32_t t,
                                                std::int32_t padded_k) {
    const int kg              = padded_k >> 6;
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    const int warp            = static_cast<int>(threadIdx.x) >> 5;
    const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
    const int gid             = lane >> 2; // 0..7  -> mma M-row / B N-col
    const int lid             = lane & 3;  // 0..3  -> mma K / C N-col

    const int m0 = (static_cast<int>(blockIdx.x) * warps_per_block + warp) * 16;
    const int t0 = static_cast<int>(blockIdx.y) * (8 * kColTiles);
    if (m0 >= n) { return; }

    float acc[kColTiles][4];
#pragma unroll
    for (int s = 0; s < kColTiles; ++s) {
        acc[s][0] = 0.0f;
        acc[s][1] = 0.0f;
        acc[s][2] = 0.0f;
        acc[s][3] = 0.0f;
    }

    const int rowA0 = m0 + gid;     // A rows this lane owns: gid and gid+8
    const int rowA1 = m0 + gid + 8;

    for (int k0 = 0; k0 < k; k0 += 16) {
        const int kc0 = k0 + 2 * lid;     // a0/a1 K columns
        const int kc1 = k0 + 2 * lid + 8; // a2/a3 K columns
        auto      wv  = [&](int row, int kk) -> float {
            return (row < n && kk < k) ? gemm_dequant_w<Codec>(codes, high, scales, row, kk, kg)
                                             : 0.0f;
        };
        const unsigned a0 = gemm_mma_pack_bf16x2(wv(rowA0, kc0), wv(rowA0, kc0 + 1));
        const unsigned a1 = gemm_mma_pack_bf16x2(wv(rowA1, kc0), wv(rowA1, kc0 + 1));
        const unsigned a2 = gemm_mma_pack_bf16x2(wv(rowA0, kc1), wv(rowA0, kc1 + 1));
        const unsigned a3 = gemm_mma_pack_bf16x2(wv(rowA1, kc1), wv(rowA1, kc1 + 1));

#pragma unroll
        for (int s = 0; s < kColTiles; ++s) {
            const int col = t0 + s * 8 + gid; // B N-column
            auto      xv  = [&](int kk) -> float {
                return (col < t && kk < k)
                                ? __bfloat162float(x[static_cast<std::int64_t>(col) * k + kk])
                                : 0.0f;
            };
            const unsigned b0 = gemm_mma_pack_bf16x2(xv(kc0), xv(kc0 + 1));
            const unsigned b1 = gemm_mma_pack_bf16x2(xv(kc1), xv(kc1 + 1));
            gemm_mma_m16n8k16_bf16(acc[s][0], acc[s][1], acc[s][2], acc[s][3], a0, a1, a2, a3, b0,
                                   b1);
        }
    }

    // Epilogue: C lane layout c0={gid,2lid} c1={gid,2lid+1} c2={gid+8,2lid} c3={gid+8,2lid+1}.
    const int rowC0 = m0 + gid;
    const int rowC1 = m0 + gid + 8;
#pragma unroll
    for (int s = 0; s < kColTiles; ++s) {
        const int colc0 = t0 + s * 8 + 2 * lid;
        const int colc1 = colc0 + 1;
        if (rowC0 < n) {
            if (colc0 < t) {
                out[static_cast<std::int64_t>(colc0) * n + rowC0] = __float2bfloat16(acc[s][0]);
            }
            if (colc1 < t) {
                out[static_cast<std::int64_t>(colc1) * n + rowC0] = __float2bfloat16(acc[s][1]);
            }
        }
        if (rowC1 < n) {
            if (colc0 < t) {
                out[static_cast<std::int64_t>(colc0) * n + rowC1] = __float2bfloat16(acc[s][2]);
            }
            if (colc1 < t) {
                out[static_cast<std::int64_t>(colc1) * n + rowC1] = __float2bfloat16(acc[s][3]);
            }
        }
    }
}

} // namespace qus::kernels::detail
