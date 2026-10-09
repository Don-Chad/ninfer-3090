// Q6 row-split decode GEMV for one or two tokens.
//
// The 6-bit routes otherwise start at the 8-row SIMT tiles, which measured 2-5% slower C1 decode
// than the Q8 head they would replace (docs/maintainer/quality-trade-experiments.md), so a 6-bit
// vocabulary head read fewer bytes yet decoded no faster. This kernel follows the Q5 GEMV
// (now q5_a16_gemv.cuh): one warp owns one output row and streams its 16-group tiles through a
// cp.async ring, each lane decodes eight weights per step and the group scale is applied once per
// group. Up to two tokens share each decoded weight; from three, q6_small_t_mma.cu is faster.
//
// A tile is 16 groups: 512 code bytes (32 uint4, one per lane), 256 high-plane bytes (16 uint4)
// and 32 scale bytes (2 uint4). A lane's eight weights are code-byte chunk `pos` of its group and
// the two high-plane bytes of that chunk (Q6RowSplitStorage::kHighBytesPerChunk); the decode is
// Q6SimtDecodeAtom's exact half2 trick with the scale hoisted.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroupK        = Q6RowSplitStorage::kGroupK;
constexpr int kGroupsPerTile = 16;
constexpr int kRowsPerBlock  = 8;
constexpr int kStages        = 3;

__device__ __forceinline__ void issue_tile(uint4* s_code, uint4* s_high, uint4* s_scale,
                                           const std::uint8_t* code_row,
                                           const std::uint8_t* high_row,
                                           const std::uint8_t* scale_row, int tile, int lane) {
    const int g0 = tile * kGroupsPerTile;
    pipe_copy<16>(&s_code[lane], reinterpret_cast<const uint4*>(
                                     code_row + g0 * Q6RowSplitStorage::kCodeBytesPerGroup) +
                                     lane);
    if (lane < 16) {
        pipe_copy<16>(&s_high[lane], reinterpret_cast<const uint4*>(
                                         high_row + g0 * Q6RowSplitStorage::kHighBytesPerGroup) +
                                         lane);
    }
    if (lane < 2) {
        pipe_copy<16>(&s_scale[lane], reinterpret_cast<const uint4*>(
                                          scale_row + g0 * Q6RowSplitStorage::kScaleBytesPerGroup) +
                                          lane);
    }
    pipe_commit();
}

template <int kTokens>
__device__ __forceinline__ void consume_tile(const __nv_bfloat16* __restrict__ x, int k,
                                             const uint4* s_code, const uint4* s_high,
                                             const uint4* s_scale, int tile, int lane,
                                             float (&acc)[kTokens]) {
    constexpr int kWeightsPerLane = 8;
    constexpr int kLanesPerGroup  = kGroupK / kWeightsPerLane; // 8
    constexpr int kGroupsPerStep  = 32 / kLanesPerGroup;       // 4
    const auto* code32            = reinterpret_cast<const std::uint32_t*>(s_code);
    const auto* high16            = reinterpret_cast<const std::uint16_t*>(s_high);
    const auto* scale16           = reinterpret_cast<const std::uint16_t*>(s_scale);
    const int sub                 = lane / kLanesPerGroup;
    const int pos                 = lane % kLanesPerGroup;
    const __half2 bias            = __half2half2(__ushort_as_half(0x6420)); // 1056.0
#pragma unroll
    for (int step = 0; step < kGroupsPerTile / kGroupsPerStep; ++step) {
        const int tg             = step * kGroupsPerStep + sub;
        const float scale        = __half2float(__ushort_as_half(scale16[tg]));
        const std::uint32_t word = code32[tg * (Q6RowSplitStorage::kCodeBytesPerGroup / 4) + pos];
        const std::uint32_t high =
            static_cast<std::uint32_t>(
                high16[tg * (Q6RowSplitStorage::kHighBytesPerGroup / 2) + pos]) ^
            0xaaaau;
        float q[kWeightsPerLane];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((high >> (2 * pair)) & 3u) << 4) | (((high >> (2 * pair + 8)) & 3u) << 20);
            const float2 values = __half22float2(__hsub2(half2_from_bits(bits), bias));
            q[pair]             = values.x;
            q[pair + 4]         = values.y;
        }
        const int k0 = ((tile * kGroupsPerTile) + tg) * kGroupK + pos * kWeightsPerLane;
#pragma unroll
        for (int token = 0; token < kTokens; ++token) {
            const uint4 xv = load_vec<uint4>(
                reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * k + k0));
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            float group_acc = 0.0f;
            group_acc       = fmaf(q[0], f0.x, group_acc);
            group_acc       = fmaf(q[1], f0.y, group_acc);
            group_acc       = fmaf(q[2], f1.x, group_acc);
            group_acc       = fmaf(q[3], f1.y, group_acc);
            group_acc       = fmaf(q[4], f2.x, group_acc);
            group_acc       = fmaf(q[5], f2.y, group_acc);
            group_acc       = fmaf(q[6], f3.x, group_acc);
            group_acc       = fmaf(q[7], f3.y, group_acc);
            acc[token]      = fmaf(group_acc, scale, acc[token]);
        }
    }
}

template <int kTokens>
__global__ void __launch_bounds__(kRowsPerBlock * 32)
    q6_rowsplit_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                            const std::uint8_t* __restrict__ codes,
                            const std::uint8_t* __restrict__ high_bits,
                            const std::uint8_t* __restrict__ scales,
                            __nv_bfloat16* __restrict__ out, int n, int k, int padded_k) {
    __shared__ uint4 s_code[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_high[kRowsPerBlock][kStages][16];
    __shared__ uint4 s_scale[kRowsPerBlock][kStages][2];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    if (row >= n) { return; }

    const int groups = padded_k / kGroupK;
    const int tiles  = k / (kGroupK * kGroupsPerTile);
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * groups * Q6RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits +
        static_cast<std::int64_t>(row) * groups * Q6RowSplitStorage::kHighBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row) * groups * Q6RowSplitStorage::kScaleBytesPerGroup;

    constexpr int kPrefetch = kStages - 1;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < tiles) {
            issue_tile(s_code[warp][p], s_high[warp][p], s_scale[warp][p], code_row, high_row,
                       scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

    float acc[kTokens];
#pragma unroll
    for (int token = 0; token < kTokens; ++token) { acc[token] = 0.0f; }
#pragma unroll 1
    for (int tile = 0; tile < tiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < tiles) {
            const int slot = fetch % kStages;
            issue_tile(s_code[warp][slot], s_high[warp][slot], s_scale[warp][slot], code_row,
                       high_row, scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();
        const int slot = tile % kStages;
        consume_tile<kTokens>(x, k, s_code[warp][slot], s_high[warp][slot], s_scale[warp][slot],
                              tile, lane, acc);
        __syncwarp();
    }

#pragma unroll
    for (int token = 0; token < kTokens; ++token) {
        const float value = warp_reduce_sum(acc[token]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(token) * n + row] = __float2bfloat16_rn(value);
        }
    }
}

template <int kTokens>
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    if (x.ne[1] != kTokens || k % (kGroupK * kGroupsPerTile) != 0 ||
        w.padded_shape[1] % kGroupK != 0) {
        throw std::invalid_argument("q6 gemv: unsupported token count or K");
    }
    const unsigned grid = static_cast<unsigned>((n + kRowsPerBlock - 1) / kRowsPerBlock);
    q6_rowsplit_gemv_kernel<kTokens><<<grid, kRowsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), n, k, static_cast<int>(w.padded_shape[1]));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q6_gemv_t1(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<1>(x, w, out, stream);
}
void launch_q6_gemv_t2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<2>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
