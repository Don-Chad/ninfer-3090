#pragma once

#include "core/pdl.cuh"
#include "ops/common/math.cuh" // bf16x2_bits_to_float2, half2_from_bits
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q5/q5_schedule.cuh"

namespace ninfer::ops::detail {

// Byte-pair decoding with staged code/high/scale planes. Activation staging is
// independent of the number of warps sharing each row.
template <class Schedule, bool WideScales, class Output, class Epilogue, bool TriggerPdl = false,
          bool JoinPdl = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q5_a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int padded_k) {
    constexpr int R  = Schedule::kBlockRows;
    constexpr int W  = Schedule::kWarpsPerRow;
    constexpr int G  = Schedule::kGroupsPerWarpTile;
    constexpr int S  = Schedule::kStages;
    constexpr int NW = Schedule::kWarps;
    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) pdl::trigger_dependents();
    }

    struct alignas(16) SharedStorage {
        uint4 codes[NW][S][G * 2];
        uint4 high[NW][S][G / 2];
        std::uint32_t scales[NW][S][G / 2];
        float partial[R][W];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    __shared__ SharedStorage shared;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* shared_x      = reinterpret_cast<__nv_bfloat16*>(dynamic_shared);
    const int logical_k = Schedule::kStaticK ? Schedule::kStaticK : k;
    if constexpr (Schedule::kStageX) {
        for (int item = int(threadIdx.x); item < logical_k / 8; item += Schedule::kThreads)
            store_vec(shared_x + item * 8, load_vec<uint4>(x + item * 8));
        __syncthreads();
    }
    const auto* activation    = Schedule::kStageX ? shared_x : x;
    const int warp            = int(threadIdx.x) >> 5;
    const int lane            = int(threadIdx.x) & 31;
    const int local_row       = warp / W;
    const int split           = warp % W;
    const int row             = int(blockIdx.x) * R + local_row;
    const int pairs           = logical_k / 128;
    const int group_begin     = (pairs * split / W) * 2;
    const int groups          = (pairs * (split + 1) / W) * 2 - group_begin;
    const int tiles           = (groups + G - 1) / G;
    const int row_groups      = (Schedule::kStaticK ? Schedule::kStaticK : padded_k) / 64;
    const auto row0           = std::int64_t(row < rows ? row : 0) * row_groups + group_begin;
    constexpr bool kFullTiles = Schedule::kStaticK > 0 && (Schedule::kStaticK / 128) % W == 0 &&
                                (Schedule::kStaticK / 64 / W) % G == 0;
    const auto stage = [&](int slot, int tile) {
        const int active  = kFullTiles ? G : min(G, groups - tile * G);
        const auto group0 = row0 + tile * G;
        for (int item = lane; item < G * 2; item += 32)
            cp_async_zfill<16, Schedule::kWeightCache>(&shared.codes[warp][slot][item],
                                                       codes + group0 * 32 + item * 16,
                                                       item < active * 2 ? 16 : 0);
        for (int item = lane; item < G / 2; item += 32)
            cp_async_zfill<16, Schedule::kWeightCache>(&shared.high[warp][slot][item],
                                                       high + group0 * 8 + item * 16,
                                                       item < active / 2 ? 16 : 0);
        if constexpr (WideScales) {
            for (int item = lane; item < G / 8; item += 32)
                cp_async_zfill<16>(reinterpret_cast<uint4*>(shared.scales[warp][slot]) + item,
                                   scales + group0 * 2 + item * 16, item < active / 8 ? 16 : 0);
        } else {
            for (int item = lane; item < G / 2; item += 32)
                cp_async_zfill<4>(&shared.scales[warp][slot][item], scales + group0 * 2 + item * 4,
                                  item < active / 2 ? 4 : 0);
        }
        cp_commit();
    };
#pragma unroll
    for (int prefetch = 0; prefetch < S - 1; ++prefetch) {
        if (prefetch < tiles)
            stage(prefetch, prefetch);
        else
            cp_commit();
    }
    float acc = 0;
#pragma unroll 1
    for (int tile = 0; tile < tiles; ++tile) {
        if (tile + S - 1 < tiles)
            stage((tile + S - 1) % S, tile + S - 1);
        else
            cp_commit();
        cp_wait<S - 1>();
        __syncwarp();
        const int slot = tile % S;
        if constexpr (G % 4 == 0) {
            // sm_86: eight weights per lane, four groups per step (b1ca4515, 29f19f03). The
            // two-weights-per-lane loop below spends four memory-pipe instructions per two weights
            // (`ncu` on the 27B decode: L1/TEX 86.6%, DRAM 50.1%); here a lane reads one 4-byte
            // code word, one high byte, one scale and one 16-byte activation per four groups, a
            // 4x cut in shared/global load instructions (+3.90%). The five-bit codes then decode
            // two at a time through the exact half2 bit-trick -- half 0x6400 is 1024.0, so OR-ing a
            // nibble into the mantissa adds it, and subtracting 1040.0 with the inverted high bit
            // yields `nibble - 16 * high` -- and the group scale is applied once to the group's
            // partial sum rather than to each weight (+2.67%). Code byte b holds weights 2b (low
            // nibble) and 2b + 1; high byte h bit j is bit 4 of weight 8h + j, so bit-pair `pair`
            // of a code word carries weights `pair` and `pair + 4`.
            const auto* tile_codes  = reinterpret_cast<const std::uint32_t*>(shared.codes[warp][slot]);
            const auto* tile_high   = reinterpret_cast<const std::uint8_t*>(shared.high[warp][slot]);
            const auto* tile_scales = reinterpret_cast<const std::uint16_t*>(shared.scales[warp][slot]);
            const int sub           = lane >> 3; // group within the step
            const int pos           = lane & 7;  // eight-weight span within the group
            const __half2 bias      = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
            for (int step = 0; step < G / 4; ++step) {
                const int group = step * 4 + sub;
                if (kFullTiles || group < groups - tile * G) {
                    const float scale        = __half2float(__ushort_as_half(tile_scales[group]));
                    const std::uint32_t word = tile_codes[group * 8 + pos];
                    const std::uint32_t high_inv =
                        static_cast<std::uint32_t>(tile_high[group * 8 + pos]) ^ 0xffu;
                    const auto kk   = (group_begin + tile * G + group) * 64 + pos * 8;
                    const uint4 xv  = load_vec<uint4>(activation + kk);
                    const float2 f0 = bf16x2_bits_to_float2(xv.x);
                    const float2 f1 = bf16x2_bits_to_float2(xv.y);
                    const float2 f2 = bf16x2_bits_to_float2(xv.z);
                    const float2 f3 = bf16x2_bits_to_float2(xv.w);
                    const float xs[8]{f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};
                    float group_acc = 0.0f;
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
                        bits |= (((high_inv >> pair) & 1u) << 4) |
                                (((high_inv >> (pair + 4)) & 1u) << 20);
                        const float2 q = __half22float2(__hsub2(half2_from_bits(bits), bias));
                        group_acc      = fmaf(q.x, xs[pair], group_acc);
                        group_acc      = fmaf(q.y, xs[pair + 4], group_acc);
                    }
                    acc = fmaf(group_acc, scale, acc);
                }
            }
        } else {
#pragma unroll
            for (int group = 0; group < G; ++group) {
                if (kFullTiles || group < groups - tile * G) {
                    float w0, w1;
                    Q5ScalarDecodeAtom::decode_pair(
                        reinterpret_cast<const std::uint8_t*>(shared.codes[warp][slot]),
                        reinterpret_cast<const std::uint8_t*>(shared.high[warp][slot]),
                        reinterpret_cast<const std::uint8_t*>(shared.scales[warp][slot]), group,
                        lane, w0, w1);
                    const auto kk = (group_begin + tile * G + group) * 64 + lane * 2;
                    const float2 value =
                        __bfloat1622float2(load_vec<__nv_bfloat162>(activation + kk));
                    acc = fmaf(w0, value.x, acc);
                    acc = fmaf(w1, value.y, acc);
                }
            }
        }
        __syncwarp();
    }
    acc = warp_reduce_sum(acc);
    if constexpr (W > 1) {
        if (lane == 0) shared.partial[local_row][split] = acc;
        __syncthreads();
        if (split == 0) {
            acc = shared.partial[local_row][0];
#pragma unroll
            for (int part = 1; part < W; ++part) acc += shared.partial[local_row][part];
        }
    }
    if (split == 0 && lane == 0 && row < rows) {
        const float values[1]{acc};
        linear_finish_row(output, epilogue, row, 0, values, 1);
    }
    if constexpr (JoinPdl) pdl::wait_for_dependencies();
}

} // namespace ninfer::ops::detail
