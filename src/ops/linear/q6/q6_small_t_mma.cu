// Q6 row-split small-T MMA: out[T, N] = x[T, K] * W[N, K]^T for T <= 16 tokens.
//
// The vocabulary head is a 993 MB Q6 matrix that a speculative verify reads at T = K+1 tokens
// (MTP3: 4, DFlash2 K=7: 8). The per-row GEMV (q6_rowsplit_gemv.cu) grows with T because every
// token is its own FMA stream, and the 64-row MMA tile (q6_rowsplit_gemm_mma.cuh) decodes through
// shared memory and ran at ~424 GB/s from T=5. This kernel follows the Q5 small-T design
// (ops/linear/q5/q5_small_t_mma.cuh): a CTA owns one m16 row tile and its eight warps split K one
// 64-weight group at a time, so an m16n8k16 MMA costs the same from T=1 to T=8.
//
// Each lane loads its eight contiguous code bytes (chunks 2t and 2t+1 of the group) and their four
// high-plane bytes straight from global memory and decodes them exactly into bf16 codes: a Q6 code
// is v - 32 for the six-bit v = nibble | (high ^ 2) << 4, and placing v in the mantissa of bf16
// 128.0 reads 128 + v, so subtracting 160 leaves the code with no conversion. A chunk decodes into
// the pairs (8c + p, 8c + p + 4); the MMA's k order is permuted so that pair p of chunks 2t and
// 2t+1 is what lane t needs for k16 block p, and the activations are packed in the same order. The
// group's fp16 scale is applied per row to an fp32 group accumulator, and the eight K partials are
// reduced through shared memory.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroupK = Q6RowSplitStorage::kGroupK;
constexpr int kWarps  = 8;
constexpr int kRows   = 16;

struct GroupLoad {
    uint2 code[2];         // rows g and g+8: chunks 2t, 2t+1
    unsigned high[2];      // their two high-plane half-words
    std::uint16_t scale[2];
};

__device__ __forceinline__ GroupLoad load_group(const std::uint8_t* __restrict__ codes,
                                                const std::uint8_t* __restrict__ high,
                                                const std::uint8_t* __restrict__ scales,
                                                std::int64_t row_a, std::int64_t row_b,
                                                int groups, int group, int t) {
    GroupLoad g;
    const std::int64_t ga = row_a * groups + group;
    const std::int64_t gb = row_b * groups + group;
    g.code[0]  = load_ldg<uint2>(codes + ga * Q6RowSplitStorage::kCodeBytesPerGroup + t * 8);
    g.code[1]  = load_ldg<uint2>(codes + gb * Q6RowSplitStorage::kCodeBytesPerGroup + t * 8);
    g.high[0]  = load_ldg<unsigned>(high + ga * Q6RowSplitStorage::kHighBytesPerGroup + t * 4);
    g.high[1]  = load_ldg<unsigned>(high + gb * Q6RowSplitStorage::kHighBytesPerGroup + t * 4);
    g.scale[0] = load_ldg<std::uint16_t>(scales + ga * Q6RowSplitStorage::kScaleBytesPerGroup);
    g.scale[1] = load_ldg<std::uint16_t>(scales + gb * Q6RowSplitStorage::kScaleBytesPerGroup);
    return g;
}

// Pair p of one chunk -> bf16 codes (8c + p, 8c + p + 4), low half first. `high` is the chunk's
// high half-word already xor-ed with 0xaaaa.
__device__ __forceinline__ unsigned decode_pair(unsigned word, unsigned high, int p) {
    unsigned bits = ((word >> (4 * p)) & 0x000f000fu) | 0x43004300u;
    bits |= (((high >> (2 * p)) & 3u) << 4) | (((high >> (2 * p + 8)) & 3u) << 20);
    const __nv_bfloat162 value =
        __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits),
                __nv_bfloat162(__ushort_as_bfloat16(0x4320), __ushort_as_bfloat16(0x4320)));
    return *reinterpret_cast<const unsigned*>(&value);
}

template <int Tiles>
__global__ void __launch_bounds__(kWarps * 32)
    q6_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                          const std::uint8_t* __restrict__ codes,
                          const std::uint8_t* __restrict__ high_bits,
                          const std::uint8_t* __restrict__ scales,
                          __nv_bfloat16* __restrict__ out, int n, int k, int padded_groups,
                          int tokens) {
    __shared__ float partial[kWarps][Tiles][32][4];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int gid   = lane >> 2;
    const int t     = lane & 3;
    const int row0  = static_cast<int>(blockIdx.x) * kRows;
    const std::int64_t row_a = row0 + gid;
    const std::int64_t row_b = row0 + gid + 8;
    const int groups = k / kGroupK;

    float acc[Tiles][4];
#pragma unroll
    for (int j = 0; j < Tiles; ++j) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { acc[j][c] = 0.0f; }
    }

    int group = warp;
    GroupLoad next;
    if (group < groups) {
        next = load_group(codes, high_bits, scales, row_a, row_b, padded_groups, group, t);
    }
#pragma unroll 1
    for (; group < groups; group += kWarps) {
        const GroupLoad cur = next;
        if (group + kWarps < groups) {
            next = load_group(codes, high_bits, scales, row_a, row_b, padded_groups,
                              group + kWarps, t);
        }

        // B fragments: token gid of each tile, the sixteen activations of chunks 2t and 2t+1.
        unsigned b[Tiles][4][2];
#pragma unroll
        for (int j = 0; j < Tiles; ++j) {
            const int token = j * 8 + gid;
            uint4 lo = make_uint4(0u, 0u, 0u, 0u);
            uint4 hi = make_uint4(0u, 0u, 0u, 0u);
            if (token < tokens) {
                const __nv_bfloat16* src =
                    x + static_cast<std::int64_t>(token) * k + group * kGroupK + t * 16;
                lo = load_ldg<uint4>(src);
                hi = load_ldg<uint4>(src + 8);
            }
            // (v[p], v[p+4]) from v[0..7] in lo and (v[8+p], v[12+p]) from hi.
            b[j][0][0] = __byte_perm(lo.x, lo.z, 0x5410);
            b[j][1][0] = __byte_perm(lo.x, lo.z, 0x7632);
            b[j][2][0] = __byte_perm(lo.y, lo.w, 0x5410);
            b[j][3][0] = __byte_perm(lo.y, lo.w, 0x7632);
            b[j][0][1] = __byte_perm(hi.x, hi.z, 0x5410);
            b[j][1][1] = __byte_perm(hi.x, hi.z, 0x7632);
            b[j][2][1] = __byte_perm(hi.y, hi.w, 0x5410);
            b[j][3][1] = __byte_perm(hi.y, hi.w, 0x7632);
        }

        const unsigned ha0 = (cur.high[0] & 0xffffu) ^ 0xaaaau;
        const unsigned ha1 = (cur.high[0] >> 16) ^ 0xaaaau;
        const unsigned hb0 = (cur.high[1] & 0xffffu) ^ 0xaaaau;
        const unsigned hb1 = (cur.high[1] >> 16) ^ 0xaaaau;
        float group_acc[Tiles][4];
#pragma unroll
        for (int j = 0; j < Tiles; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { group_acc[j][c] = 0.0f; }
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const unsigned a0 = decode_pair(cur.code[0].x, ha0, p); // row g,   chunk 2t
            const unsigned a1 = decode_pair(cur.code[1].x, hb0, p); // row g+8, chunk 2t
            const unsigned a2 = decode_pair(cur.code[0].y, ha1, p); // row g,   chunk 2t+1
            const unsigned a3 = decode_pair(cur.code[1].y, hb1, p); // row g+8, chunk 2t+1
#pragma unroll
            for (int j = 0; j < Tiles; ++j) {
                mma_bf16(group_acc[j][0], group_acc[j][1], group_acc[j][2], group_acc[j][3], a0,
                         a1, a2, a3, b[j][p][0], b[j][p][1]);
            }
        }
        const float sa = __half2float(__ushort_as_half(cur.scale[0]));
        const float sb = __half2float(__ushort_as_half(cur.scale[1]));
#pragma unroll
        for (int j = 0; j < Tiles; ++j) {
            acc[j][0] = fmaf(group_acc[j][0], sa, acc[j][0]);
            acc[j][1] = fmaf(group_acc[j][1], sa, acc[j][1]);
            acc[j][2] = fmaf(group_acc[j][2], sb, acc[j][2]);
            acc[j][3] = fmaf(group_acc[j][3], sb, acc[j][3]);
        }
    }

#pragma unroll
    for (int j = 0; j < Tiles; ++j) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { partial[warp][j][lane][c] = acc[j][c]; }
    }
    __syncthreads();
    // Thread e sums element e of the Tiles * 32 * 4 fragment over the eight warps. C fragment c of
    // lane l is row (l >> 2) + 8 * (c >> 1), token 2 * (l & 3) + (c & 1) of its tile.
    for (int e = static_cast<int>(threadIdx.x); e < Tiles * 128; e += kWarps * 32) {
        const int j = e / 128;
        const int l = (e / 4) % 32;
        const int c = e % 4;
        float sum   = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { sum += partial[w][j][l][c]; }
        const int token = j * 8 + 2 * (l & 3) + (c & 1);
        const int row   = row0 + (l >> 2) + 8 * (c >> 1);
        if (token < tokens && row < n) {
            out[static_cast<std::int64_t>(token) * n + row] = __float2bfloat16_rn(sum);
        }
    }
}

template <int Tiles>
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    if (tokens < 1 || tokens > Tiles * 8 || n % kRows != 0 || k % kGroupK != 0 ||
        w.padded_shape[1] % kGroupK != 0) {
        throw std::invalid_argument("q6 small-T mma: unsupported token count, N or K");
    }
    q6_small_t_mma_kernel<Tiles><<<n / kRows, kWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), n, k,
        static_cast<int>(w.padded_shape[1] / kGroupK), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q6_small_t_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<1>(x, w, out, stream);
}
void launch_q6_small_t_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<2>(x, w, out, stream);
}
void launch_q6_small_t_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<4>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
