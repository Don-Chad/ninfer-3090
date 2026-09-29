#include "core/weight.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden           = 2048;
constexpr int kExperts          = 256;
constexpr int kRouterRows       = kExperts + 1;
constexpr int kTopK             = 8;
constexpr int kIntermediate     = 512;
constexpr int kD1Warps          = 8;
constexpr int kAdaptiveD3Blocks = 5 * kIntermediate;
constexpr int kAdaptiveD4Blocks = 5 * (kHidden / 4);

__device__ __forceinline__ float dot_bf16_eight(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 av  = load_vec<uint4>(a);
    const uint4 bv  = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x);
    const float2 a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z);
    const float2 a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x);
    const float2 b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z);
    const float2 b3 = bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

__device__ __forceinline__ float router_row_dot(const __nv_bfloat16* x, const __nv_bfloat16* row) {
    constexpr int kSlice = kHidden / kD1Warps;
    constexpr int kVecs  = kSlice / (32 * 8);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    float sum            = 0.0f;
#pragma unroll
    for (int vector = 0; vector < kVecs; ++vector) {
        const int k = warp * kSlice + vector * 32 * 8 + lane * 8;
        sum += dot_bf16_eight(row + k, x + k);
    }
    return warp_reduce_sum(sum);
}

__global__ void sparse_moe_d1_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ router,
                                     float* __restrict__ scores,
                                     const char* __restrict__ shared_down_codes,
                                     unsigned long long shared_down_bytes) {
    __shared__ float partial[kD1Warps];
    const int row   = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const float dot = router_row_dot(x, router + static_cast<std::int64_t>(row) * kHidden);
    if (lane == 0) { partial[warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < kD1Warps ? partial[lane] : 0.0f;
        value       = warp_reduce_sum<kD1Warps>(value);
        if (lane == 0) { scores[row] = value; }
    }
    if (shared_down_codes != nullptr) {
        // Warm L2 for the shared-expert down codes that D4's shared warp
        // (the block's long pole) will stream at t~14.3us. Pure cache hint.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < shared_down_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(shared_down_codes + offset));
        }
    }
}

__global__ void sparse_moe_d2_warp_kernel(const float* __restrict__ scores, int* __restrict__ ids,
                                          float* __restrict__ alpha,
                                          float* __restrict__ shared_scale) {
    __shared__ float selected_logits[kTopK];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, selected_logits);
}

struct Q4Codec {
    static constexpr int kGroupK         = 64;
    static constexpr bool kD3PackedWord8 = true;
};

struct Q5Codec {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint8_t high_bits =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup);
        Q5SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q6Codec {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q6RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint16_t high_bits = *reinterpret_cast<const std::uint16_t*>(
            high + group_index * Q6RowSplitStorage::kHighBytesPerGroup + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q6RowSplitStorage::kScaleBytesPerGroup);
        Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q8Codec {
    static constexpr int kGroupK         = 32;
    // Was kD3SingleValuePerLane, which is no longer what the branch does -- it selects the Q8
    // shared plane in d3, and that plane now takes eight values per lane like everything else.
    static constexpr bool kD3Q8Plane     = true;
    static constexpr bool kD3PackedWord8 = false;
    // The shared expert takes the same eight-per-lane consume loop as the routed ones. It used to
    // take `load_pair`, and in d4 that is not merely slower in isolation -- d4 reduces nine warps
    // through one `__syncthreads()`, so warp 8 running a 4x narrower loop on half a warp
    // (`lane < kGroupK / 2` is 16 of 32 lanes at kGroupK=32) made it the straggler the other eight
    // waited for. See the d4 stall table in TODO section 2c.
    static constexpr bool kPackedWord8          = true;

    // Eight contiguous int8 codes and the group's single FP16 scale. The codes are plain and
    // contiguous (kCodeBytesPerGroup == kGroupK), so the eight bytes are one 64-bit load rather
    // than the four two-byte loads `load_pair` issued for the same span.
    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* /*high*/, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        const std::uint8_t* packed = codes +
                                     group_index * Q8RowSplitStorage::kCodeBytesPerGroup +
                                     static_cast<std::int64_t>(lane_in_group) * 8;
        // kCodeBytesPerGroup is 32 and lane_in_group * 8 is a multiple of 8, so this is 8-byte
        // aligned whenever the plane is, which the loader guarantees. One LDG.64 for eight codes.
        const uint2 raw = *reinterpret_cast<const uint2*>(packed);
        const std::uint32_t words[2] = {raw.x, raw.y};
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const std::int8_t code =
                static_cast<std::int8_t>((words[item >> 2] >> ((item & 3) * 8)) & 0xFFu);
            weights[item] = static_cast<float>(code) * scale;
        }
    }
};

template <class Codec, int K>
__device__ __forceinline__ void dot_two_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                             const std::uint8_t* scales, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    constexpr int kGroups = K / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    const int first_group = k_begin / Codec::kGroupK;
    const int last_group  = k_end / Codec::kGroupK;
    if constexpr (Codec::kD3PackedWord8) {
        // Four adjacent Q4 groups form one 128-byte warp transaction. Each lane owns eight
        // consecutive K values, so one mantissa decode feeds eight FP32 FMAs instead of issuing
        // four scalar code-pair/decode iterations.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group           = group_base + lane_group;
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const std::uint32_t packed0 =
                *reinterpret_cast<const std::uint32_t*>(codes + index0 * 32 + lane_in_group * 4);
            const std::uint32_t packed1 =
                *reinterpret_cast<const std::uint32_t*>(codes + index1 * 32 + lane_in_group * 4);
            const auto scale0 = *reinterpret_cast<const std::uint16_t*>(scales + index0 * 2);
            const auto scale1 = *reinterpret_cast<const std::uint16_t*>(scales + index1 * 2);
            float weights0[8];
            float weights1[8];
            Q4SimtDecodeAtom::decode_eight(packed0, scale0, weights0);
            Q4SimtDecodeAtom::decode_eight(packed1, scale1, weights1);
            const uint4 input     = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0       = bf16x2_bits_to_float2(input.x);
            const float2 x1       = bf16x2_bits_to_float2(input.y);
            const float2 x2       = bf16x2_bits_to_float2(input.z);
            const float2 x3       = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if constexpr (Codec::kD3Q8Plane) {
        // The shared expert's Q8 plane, on the same eight-per-lane ownership as the Q4 branch
        // above. It used to take one value per lane per group: identical arithmetic in 8x the
        // loop trips, each carrying its own dependent load. d3's dominant stall is exactly that
        // latency (long_scoreboard 7.54 of a ~9.3 budget), and eight values per lane is the same
        // fix that was worth +3.90% on the 27B's q5 GEMV.
        constexpr int kLanesPerGroup = Codec::kGroupK / 8;
        constexpr int kGroupsPerIter = 32 / kLanesPerGroup;
        static_assert(kGroups % kGroupsPerIter == 0, "the consume loop must not leave a tail");
        const int lane_group    = lane / kLanesPerGroup;
        const int lane_in_group = lane % kLanesPerGroup;
        for (int group_base = first_group; group_base < last_group;
             group_base += kGroupsPerIter) {
            const int group           = group_base + lane_group;
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            float weights0[8];
            float weights1[8];
            Codec::load_eight(codes, nullptr, scales, index0, lane_in_group, weights0);
            Codec::load_eight(codes, nullptr, scales, index1, lane_in_group, weights1);
            const uint4 input     = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0       = bf16x2_bits_to_float2(input.x);
            const float2 x1       = bf16x2_bits_to_float2(input.y);
            const float2 x2       = bf16x2_bits_to_float2(input.z);
            const float2 x3       = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            float w00, w01, w10, w11;
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row0) * kGroups + group,
                             lane, w00, w01);
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row1) * kGroups + group,
                             lane, w10, w11);
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(load_vec<__nv_bfloat162>(x + k));
            acc0            = fmaf(w00, xv.x, acc0);
            acc0            = fmaf(w01, xv.y, acc0);
            acc1            = fmaf(w10, xv.x, acc1);
            acc1            = fmaf(w11, xv.y, acc1);
        }
    }
    result0 = warp_reduce_sum(acc0);
    result1 = warp_reduce_sum(acc1);
}

// Nine warps: 0..7 take one routed expert each through RoutedCodec, warp 8 takes the shared expert
// through Q8Codec -- kTopK + 1, so eight warps would have nowhere to put the shared path.
//
// The block cannot retire until all nine warps finish, and the shared Q8 path is heavier than a
// routed Q4 one, so eight completed routed warps hold warp slots while warp 8 drains. ncu sees that
// as no warp eligible to issue on 43-48% of cycles, and sparse_moe_d3_path_tiled_kernel below
// exists to spread the same nine paths over three-warp blocks for exactly this reason.
//
// SPLITTING THE BLOCK WAS TRIED AT ONE TOKEN AND IT IS WORSE. Measured 2026-09-09, 35B / int8 /
// decode, clocks locked at 1,500 MHz, ncu kernel replay, medians of 3 launches. The path-tiled
// kernel at tokens == 1 writes an identical activation layout, so this was a pure geometry A/B:
//
//                            nine_warp (512x1, 288 thr)   paths3 (512x3, 96 thr)
//   duration                          29.63 us                  30.75 us
//   block limit registers / warps        5 / 5                    16 / 16
//   theoretical warps per SM                45                        48
//   theoretical occupancy                93.75%                      100%
//   ACHIEVED occupancy                   67.65%                    52.60%
//   no eligible warp                      43.2%                     53.4%
//
// So the split delivers the higher *theoretical* occupancy it was designed for and a materially
// lower *achieved* one, and the scheduler stall it was meant to remove gets worse. The cause is the
// wave tail: three-warp blocks need 1,536 blocks against a 16-per-SM capacity of 1,312, so two
// waves at 58.5% efficiency, against 512 blocks over 410 slots at 62.4%. Removing a serialisation
// inside the block does not pay for spreading a kernel that is only ~1.17 machine-fulls of work
// over more waves. Do not retry this without first making the kernel bigger than the machine.
//
// One measurement note worth keeping: d4 was captured alongside as an unchanged control and moved
// 6.1% between the two arms (DRAM 46.9% -> 52.4%), which is larger than the d3 effect. A downstream
// kernel in the same fused pipeline is NOT a clean control -- d3's different access pattern leaves
// different L2 state behind for d4. The occupancy counters above are what carry this result.

// EIGHT WARPS WAS ALSO TRIED, IN THE OPPOSITE DIRECTION, AND IT WINS THE KERNEL AND LOSES THE
// MODEL. Measured 2026-09-13. The split above failed by making *more, smaller* blocks and paying a
// worse wave tail, so this went the other way: fold the shared expert's K range across the eight
// routed warps (kHidden/8 = 256 each, one trip of the widened Q8 loop) and reduce it once, so the
// block is 8 warps instead of 9. That is a strictly better launch on every counter --
//
//                          nine_warp          eight_warp
//   warps per SM             45 of 48          48 of 48
//   blocks per SM                   5                 6   (410 -> 492 slots)
//   waves per SM                 1.25              1.04
//   ACHIEVED occupancy         77.13%            87.08%
//   eligible warps               1.58              1.85
//   duration                  23.87 us          21.82 us   (-8.6%)
//
// -- and it is worth **nothing** end to end: -0.03% at C1 (paired median of 6, 3/6 positive) and
// +0.11% with MTP3 (4/6), both crossing zero. Under ncu replay the four MoE kernels total 57.05 ->
// 54.97 us, so the kernel time is genuinely there; it just does not convert. These kernels run in
// a PDL pipeline and overlap, so d3 finishing 0.8 us earlier does not let d4 start 0.8 us earlier.
//
// The lesson is the sharper version of the one above: **a faster kernel in isolation is not a
// faster model, and ncu replay cannot tell you which you have.** Only the end-to-end paired A/B
// can. Reverted -- it also reordered the shared expert's FP32 accumulation, which is a real
// numerical change to carry for zero gain. The occupancy lever on d3 is now spent: it works
// exactly as designed and the design does not matter.
//
// What was left on this Op was not d3 but `sparse_moe_d2_warp_kernel`, on <<<1, 32>>> with d3 unable
// to start until it has the ids. This comment used to call it "a parallel-selection problem, not a
// memory one". Half right: not memory, but not parallelism either. It was branch resolution on a
// single warp -- the merge's bitonic restores and the comparator were branches -- and making them
// selects took d2 from 11.04 to 5.73 us for +2.39% end to end, with no extra warps. See the d2
// entry in TODO and `sparse_moe_merge_ranked_runs` in sparse_moe_route.cuh.
template <class RoutedCodec>
__global__ void sparse_moe_d3_nine_warp_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ act) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    if (tid < 256) { store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8)); }
    __syncthreads();

    const int j = static_cast<int>(blockIdx.x);
    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        pdl::wait_for_dependencies();
        const int expert   = ids[warp];
        const int row_base = expert * 1024;
        dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales, row_base + j,
                                           row_base + kIntermediate + j, x_shared, 0, kHidden, gate,
                                           up);
    } else {
        dot_two_rows<Q8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                       x_shared, 0, kHidden, gate, up);
    }
    if (lane == 0) { act[static_cast<std::int64_t>(warp) * kIntermediate + j] = silu(gate) * up; }
}

template <class RoutedCodec, int PathsPerBlock, bool Adaptive>
__global__ void sparse_moe_d3_path_tiled_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ token_ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ token_activations,
    int tokens, const int* __restrict__ adaptive_route_jobs) {
    // Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
    // heavier shared Q8 path from holding eight completed routed warps resident.
    static_assert(PathsPerBlock > 0 && (kTopK + 1) % PathsPerBlock == 0);
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    const int total_work = tokens * kPathBlocks * kIntermediate;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kIntermediate + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token_path = work / kIntermediate;
        const int j          = work - token_path * kIntermediate;
        const int token      = token_path / kPathBlocks;
        const int path_block = token_path - token * kPathBlocks;
        const int path       = path_block * PathsPerBlock + warp;
        const auto* input    = x + static_cast<std::int64_t>(token) * kHidden;
        for (int vector = tid; vector < kHidden / 8; vector += static_cast<int>(blockDim.x)) {
            store_vec(x_shared + vector * 8, load_vec<uint4>(input + vector * 8));
        }
        __syncthreads();

        constexpr int kRouterPartitions = 4;
        const int activation_begin      = (token * (kTopK + 1) + path) * kIntermediate;
        // Routed paths need S2's ids. A shared path is independent only when its output lies
        // beyond the partial-score prefix that S2 may still read from the lifetime-unioned scratch.
        const bool must_wait_for_s2 =
            path < kTopK || activation_begin < tokens * kRouterRows * kRouterPartitions;
        if constexpr (!Adaptive) {
            if (must_wait_for_s2) { pdl::wait_for_dependencies(); }
        }
        float gate = 0.0f;
        float up   = 0.0f;
        if (path < kTopK) {
            const int expert   = token_ids[token * kTopK + path];
            const int row_base = expert * (2 * kIntermediate);
            dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales,
                                               row_base + j, row_base + kIntermediate + j, x_shared,
                                               0, kHidden, gate, up);
        } else {
            dot_two_rows<Q8Codec, kHidden>(shared_codes, nullptr, shared_scales, j,
                                           kIntermediate + j, x_shared, 0, kHidden, gate, up);
        }
        if (lane == 0) {
            token_activations[(static_cast<std::int64_t>(token) * (kTopK + 1) + path) *
                                  kIntermediate +
                              j] = silu(gate) * up;
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

template <class Codec, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                              const std::uint8_t* scales, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    constexpr int kGroups = kIntermediate / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // Q5/Q6 use the same eight-value lane ownership as D3. The high plane and FP16 scale are
        // decoded exactly from their registered row-split codec before FP32 accumulation.
        //
        // Eight values per lane fixes the lane-to-group mapping: a group of kGroupK values needs
        // kGroupK/8 lanes, so a 32-lane warp covers 32/(kGroupK/8) groups per iteration. Q5/Q6 at
        // kGroupK=64 give 8 lanes per group and 4 groups per pass, which is what this loop was
        // written as literally; Q8 at kGroupK=32 gives 4 and 8. Both divide kGroups exactly at
        // kIntermediate 512, and the static_assert says so rather than leaving a silent tail.
        constexpr int kLanesPerGroup = Codec::kGroupK / 8;
        constexpr int kGroupsPerIter = 32 / kLanesPerGroup;
        static_assert(Codec::kGroupK % 8 == 0, "eight values per lane needs kGroupK % 8 == 0");
        static_assert(kGroups % kGroupsPerIter == 0, "the consume loop must not leave a tail");
        const int lane_group    = lane / kLanesPerGroup;
        const int lane_in_group = lane % kLanesPerGroup;
        for (int group_base = first_group; group_base < last_group;
             group_base += kGroupsPerIter) {
            const int group = group_base + lane_group;
            const float4 x0 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float4 x1 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8 + 4);
            const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row_base + row) * kGroups + group;
                float weights[8];
                Codec::load_eight(codes, high, scales, group_index, lane_in_group, weights);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    acc[row] = fmaf(weights[item], values[item], acc[row]);
                }
            }
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = load_vec<float2>(x + k);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                float w0, w1;
                Codec::load_pair(codes, high, scales,
                                 static_cast<std::int64_t>(row_base + row) * kGroups + group, lane,
                                 w0, w1);
                acc[row] = fmaf(w0, xv.x, acc[row]);
                acc[row] = fmaf(w1, xv.y, acc[row]);
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { result[row] = warp_reduce_sum(acc[row]); }
}

template <class RoutedCodec, int Rows>
__global__ void sparse_moe_d4_nine_warp_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ act,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    const char* __restrict__ prefetch_data, unsigned long long prefetch_bytes) {
    __shared__ float paths[kTopK + 1][Rows];
    pdl::wait_for_dependencies();
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;
    if (warp < kTopK) {
        const int expert = ids[warp];
        float dot[Rows];
        dot_fp32_rows<RoutedCodec, Rows>(routed_codes, routed_high, routed_scales,
                                         expert * kHidden + row_base,
                                         act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
                                         kIntermediate / RoutedCodec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[warp][row] = alpha[warp] * dot[row]; }
        }
    } else {
        float dot[Rows];
        dot_fp32_rows<Q8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                     act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                     kIntermediate / Q8Codec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[kTopK][row] = *shared_scale * dot[row]; }
        }
    }
    if (prefetch_data != nullptr) {
        // Fire-and-forget L2 warmup of the next consumer's weight codes,
        // issued BEFORE the block barrier so it hides behind the slowest warp instead
        // of extending the grid tail. Pure cache hint; no values, no addition order.
        const unsigned long long offset =
            (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 128ull;
        if (offset < prefetch_bytes) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(prefetch_data + offset));
        }
    }
    __syncthreads();
    if (warp == 0 && lane < Rows) {
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
        destination[row_base + lane] = __float2bfloat16_rn(value);
    }
}

template <class RoutedCodec, int Rows, bool Adaptive>
__global__ void sparse_moe_d4_token_kernel(
    const int* __restrict__ token_ids, const float* __restrict__ token_alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ token_activations,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    int tokens, const int* __restrict__ adaptive_route_jobs) {
    // Token is a grid dimension rather than an in-CTA serial loop. Rows lets one routed-weight
    // stream serve adjacent outputs while retaining the deterministic rank-order FP32 epilogue.
    __shared__ float paths[kTopK + 1][Rows];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    constexpr int kRowBlocks = kHidden / Rows;
    const int total_work     = tokens * kRowBlocks;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kRowBlocks + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token     = work / kRowBlocks;
        const int row_block = work - token * kRowBlocks;
        const int row_base  = row_block * Rows;
        const float* act =
            token_activations + static_cast<std::int64_t>(token) * (kTopK + 1) * kIntermediate;
        if (warp < kTopK) {
            const int expert = token_ids[token * kTopK + warp];
            float dot[Rows];
            dot_fp32_rows<RoutedCodec, Rows>(routed_codes, routed_high, routed_scales,
                                             expert * kHidden + row_base,
                                             act + static_cast<std::int64_t>(warp) * kIntermediate,
                                             0, kIntermediate / RoutedCodec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[warp][row] = token_alpha[token * kTopK + warp] * dot[row];
                }
            }
        } else {
            float dot[Rows];
            dot_fp32_rows<Q8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                         act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                         kIntermediate / Q8Codec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[kTopK][row] = shared_scale[token] * dot[row];
                }
            }
        }
        __syncthreads();
        if (warp == 0 && lane < Rows) {
            __nv_bfloat16* output =
                destination + static_cast<std::int64_t>(token) * kHidden + row_base + lane;
            float value = __bfloat162float(*output);
#pragma unroll
            for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
            *output = __float2bfloat16_rn(value);
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

void launch_d1(const Tensor& x, const SparseMoeWeights& weights,
               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_d1_kernel<<<kRouterRows, kD1Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata),
        static_cast<float*>(workspace.scratch.data),
        static_cast<const char*>(weights.shared_down.qdata),
        static_cast<unsigned long long>(kHidden) * kIntermediate);
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec>
void launch_d3_dependent_codec(const Tensor& x, const SparseMoeWeights& weights,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kIntermediate), dim3(9 * 32), 0, stream}, sparse_moe_d3_nine_warp_kernel<Codec>,
        input, ids, routed_codes, routed_high, routed_scales, shared_codes, shared_scales, act));
}

void launch_d2_d3(const Tensor& x, const SparseMoeWeights& weights,
                  const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids          = static_cast<int*>(workspace.ids.data);
    auto* alpha        = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    sparse_moe_d2_warp_kernel<<<1, 32, 0, stream>>>(scores, ids, alpha, shared_scale);
    CUDA_CHECK(cudaGetLastError());

    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        launch_d3_dependent_codec<Q4Codec>(x, weights, workspace, stream);
        return;
    case QType::Q8_G32_FP16:
        launch_d3_dependent_codec<Q8Codec>(x, weights, workspace, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D3 codec");
    }
}

template <class Codec>
void launch_d4_dependent_codec(const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                               const void* prefetch_data, std::size_t prefetch_bytes) {
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    const auto* alpha         = static_cast<const float*>(workspace.alpha.data);
    const auto* shared_scale  = static_cast<const float*>(workspace.shared_scale.data);
    const auto* act           = static_cast<const float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);
    auto* output              = static_cast<__nv_bfloat16*>(destination.data);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kHidden), dim3(9 * 32), 0, stream}, sparse_moe_d4_nine_warp_kernel<Codec, 1>, ids,
        alpha, shared_scale, act, routed_codes, routed_high, routed_scales, shared_codes,
        shared_scales, output, static_cast<const char*>(prefetch_data),
        static_cast<unsigned long long>(prefetch_bytes)));
}

void launch_d4_dependent(const SparseMoeWeights& weights, Tensor& destination,
                         const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                         const void* prefetch_data, std::size_t prefetch_bytes) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        launch_d4_dependent_codec<Q5Codec>(weights, destination, workspace, stream, prefetch_data,
                                           prefetch_bytes);
        return;
    case QType::Q6_G64_FP16:
        launch_d4_dependent_codec<Q6Codec>(weights, destination, workspace, stream, prefetch_data,
                                           prefetch_bytes);
        return;
    case QType::Q8_G32_FP16:
        launch_d4_dependent_codec<Q8Codec>(weights, destination, workspace, stream, prefetch_data,
                                           prefetch_bytes);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D4 codec");
    }
}

template <class Codec, int PathsPerBlock, bool Adaptive>
void launch_d3_small_t_paths(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    if constexpr (Adaptive) {
        sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, true>
            <<<kAdaptiveD3Blocks, PathsPerBlock * 32, 0, stream>>>(
                input, token_ids, routed_codes, routed_high, routed_scales, shared_codes,
                shared_scales, token_activations, tokens, adaptive_route_jobs);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kIntermediate, tokens * kPathBlocks), dim3(PathsPerBlock * 32), 0, stream},
            sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, false>, input, token_ids,
            routed_codes, routed_high, routed_scales, shared_codes, shared_scales,
            token_activations, tokens, nullptr));
    }
}

template <class Codec, bool Adaptive>
void launch_d3_small_t_codec(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens,
                             SparseMoeSmallTD3Schedule schedule, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD3Schedule::Paths1:
        launch_d3_small_t_paths<Codec, 1, Adaptive>(x, weights, token_ids, token_activations,
                                                    tokens, stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD3Schedule::Paths3:
        launch_d3_small_t_paths<Codec, 3, Adaptive>(x, weights, token_ids, token_activations,
                                                    tokens, stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD3Schedule::Paths9:
        launch_d3_small_t_paths<Codec, 9, Adaptive>(x, weights, token_ids, token_activations,
                                                    tokens, stream, adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D3 schedule");
}

template <class Codec, int Rows, bool Adaptive>
void launch_d4_small_t_rows(const SparseMoeWeights& weights, Tensor& destination,
                            const int* token_ids, const float* token_alpha,
                            const float* shared_scale, const float* token_activations,
                            std::int32_t tokens, cudaStream_t stream,
                            const int* adaptive_route_jobs) {
    const dim3 grid = Adaptive ? dim3(kAdaptiveD4Blocks) : dim3(kHidden / Rows, tokens);
    sparse_moe_d4_token_kernel<Codec, Rows, Adaptive><<<grid, 9 * 32, 0, stream>>>(
        token_ids, token_alpha, shared_scale, token_activations,
        static_cast<const std::uint8_t*>(weights.routed_down.qdata),
        static_cast<const std::uint8_t*>(weights.routed_down.qhigh),
        static_cast<const std::uint8_t*>(weights.routed_down.scales),
        static_cast<const std::uint8_t*>(weights.shared_down.qdata),
        static_cast<const std::uint8_t*>(weights.shared_down.scales),
        static_cast<__nv_bfloat16*>(destination.data), tokens, adaptive_route_jobs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec, bool Adaptive>
void launch_d4_small_t_codec(const SparseMoeWeights& weights, Tensor& destination,
                             const int* token_ids, const float* token_alpha,
                             const float* shared_scale, const float* token_activations,
                             std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                             cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD4Schedule::Rows1:
        launch_d4_small_t_rows<Codec, 1, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows2:
        launch_d4_small_t_rows<Codec, 2, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows4:
        launch_d4_small_t_rows<Codec, 4, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D4 schedule");
}

} // namespace

void sparse_moe_decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                                         const int* token_ids, float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD3Schedule schedule,
                                         cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d3_small_t_codec<Q4Codec, false>(x, weights, token_ids, token_activations,
                                                    tokens, schedule, stream, nullptr);
        } else {
            launch_d3_small_t_codec<Q4Codec, true>(x, weights, token_ids, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_d3_small_t_codec<Q8Codec, false>(x, weights, token_ids, token_activations, tokens,
                                                schedule, stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D3 codec");
    }
}

void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                         cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_down.qtype) {
    case QType::Q5_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q5Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q5Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q6_G64_FP16:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q6Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q6Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q8_G32_FP16:
        launch_d4_small_t_codec<Q8Codec, false>(weights, destination, token_ids, token_alpha,
                                                shared_scale, token_activations, tokens, schedule,
                                                stream, nullptr);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D4 codec");
    }
}

void sparse_moe_decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodePlan& plan,
                              const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    launch_d1(x, weights, workspace, stream);
    launch_d2_d3(x, weights, workspace, stream);
    launch_d4_dependent(weights, destination, workspace, stream, plan.next_weight_prefetch,
                        plan.next_weight_prefetch_bytes);
}

} // namespace ninfer::ops::detail
