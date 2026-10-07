#pragma once
//
// Integer-activation prefill GEMM over groupwise-int RowSplit weights, on sm_86.
//
//   D[n, t] = sum_k W[n, k] * X[k, t]
//
// The weight is read in the artifact's own RowSplit layout: for Q4 the code plane is row-major
// [n][k/2] with two codes per byte, for Q5 an 8-byte-per-group high plane carries each code's fifth
// bit, and both carry one FP16 scale per (row, group of 64). Activations are quantised to s8 with
// one scale per (token, group of 64), so an outlier channel can only spoil its own group -- a
// per-token absmax is set by whichever channel is largest and starves every other one.
//
// **The token tile is the whole performance story here, and it is not what the kernel's own counters
// suggest.** Ablations on `tools/w4a8_rowsplit_probe.cu` (gate_up shape, T=512) decompose the
// original 128x128 schedule as: 2,068 us complete, 1,423 us with the shared reads removed, and
// **1,378 us with the MMAs removed as well** -- two thirds of the time is the global-to-shared
// streaming path, with DRAM at ~28% of peak, so it is bound by cp.async latency rather than by
// bandwidth, occupancy or the tensor cores. A block that covers BN tokens re-streams the entire
// weight matrix once per column block, so at the production chunk of 1,024 tokens a 128-token tile
// streams it eight times. Widening the tile is what removes those passes:
//
//   tile (rows x tokens)   us at T=1024   TOP/s
//   128 x 128                    4,265     85.6
//   128 x 256                    3,702     98.6
//    64 x 512                    3,060    119.3
//
// A grid swizzle to encourage L2 reuse of the repeated passes measured within 0.5% of nothing, so
// the passes have to be removed rather than cached.
//
// Everything else the probes tried is recorded in TODO.md: occupancy is worth 6%, and permuting the
// weights into MMA-fragment order -- which needs a repack AGENTS.md forbids and ~9.7 GB a 24 GB
// card does not have -- about 17%.
//
// Staging: W arrives as packed nibbles in its natural order (one row's group is 32 contiguous
// bytes), X arrives already in MMA-fragment order because the quantiser below owns that buffer and
// writes it that way, and the weight scales arrive eight groups at a time in a ring, because for
// one row they are contiguous across groups while for one group they are strided by the row.
// Nothing in the prefetch path waits on a synchronous global read.

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::rowsplit_a8 {

constexpr int kThreads = 512;
constexpr int kWarpsM  = 4;
constexpr int kWarpsN  = 4;
constexpr int kGroup   = 64; // k per iteration: exactly one scale group
constexpr int kWRow    = 48; // padded shared stride for a row's 32 packed bytes, 16-byte aligned
constexpr int kStages  = 2;
constexpr int kRingGroups = 8;  // 16 bytes of one row's scales, the widest cp.async
constexpr int kRingBufs   = 2;

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(addr));
    return r;
}

__device__ __forceinline__ unsigned lds16(const void* p) {
    unsigned r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(r) : "r"(addr));
    return r;
}

__device__ __forceinline__ unsigned lds8(const void* p) {
    unsigned r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(r) : "r"(addr));
    return r;
}

template <int kBytes>
__device__ __forceinline__ void cp_async(void* smem, const void* gmem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    if constexpr (kBytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr), "l"(gmem));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(addr), "l"(gmem),
                     "n"(kBytes));
    }
}

// Two packed bytes hold four consecutive k as (low,high) nibble pairs; this spreads them to one
// raw nibble per byte, in k order.
__device__ __forceinline__ unsigned expand_nibbles(unsigned pair) {
    const unsigned lo = pair & 0x00ffu;
    const unsigned hi = (pair >> 8) & 0x00ffu;
    return (lo & 0xfu) | ((lo & 0xf0u) << 4) | ((hi & 0xfu) << 16) | ((hi & 0xf0u) << 20);
}

// Spread the four low bits of `bits` into the low bit of four bytes: byte j becomes bit j.
__device__ __forceinline__ unsigned spread4(unsigned bits) {
    const unsigned t = bits & 0xfu;
    return ((t) | (t << 7) | (t << 14) | (t << 21)) & 0x01010101u;
}

// Codes are two's complement in their own width, so (v ^ half) - half per byte centres them: Q4
// reproduces the (n^8)-8 the A16 decode uses, Q5 the same over [-16,15] once the fifth bit is in.
struct Q4Codec {
    static constexpr bool kHasHigh = false;
    __device__ static unsigned decode(unsigned pair, unsigned /*high_nibble*/) {
        return __vsub4(expand_nibbles(pair) ^ 0x08080808u, 0x08080808u);
    }
};

struct Q5Codec {
    static constexpr bool kHasHigh = true;
    __device__ static unsigned decode(unsigned pair, unsigned high_nibble) {
        const unsigned v = expand_nibbles(pair) | (spread4(high_nibble) << 4);
        return __vsub4(v ^ 0x10101010u, 0x10101010u);
    }
};

// Staged rows are one contiguous range of the weight, offset by row_begin.
template <int MT>
struct ContiguousRows {
    std::int32_t row_begin;
    static constexpr int kStagedRows    = kWarpsM * MT * 16;
    static constexpr int kRowsPerBlock  = kStagedRows;
    __device__ std::int32_t weight_row(std::int32_t block, int staged) const {
        return row_begin + block * kRowsPerBlock + staged;
    }
    __device__ static int staged_row(int warp_m, int m, int gid) {
        return (warp_m * MT + m) * 16 + gid;
    }
    __device__ static int output_row(int warp_m, int m, int gid) { return staged_row(warp_m, m, gid); }
};

// Writes each result to one destination matrix, offsetting the row. A weight whose rows feed two
// destinations is launched once per contiguous row range.
struct StoreEpilogue {
    static constexpr bool kPaired = false;
    __nv_bfloat16* dst;
    std::int32_t dst_rows;
    std::int32_t dst_row_offset;
    __device__ void operator()(std::int32_t row, std::int32_t token, float value) const {
        dst[static_cast<std::size_t>(token) * dst_rows + (row + dst_row_offset)] =
            __float2bfloat16(value);
    }
};

// residual += W @ x. Each output element belongs to exactly one thread, so the read-add-write needs
// no ordering.
struct ResidualAddEpilogue {
    static constexpr bool kPaired = false;
    __nv_bfloat16* residual;
    std::int32_t rows;
    __device__ void operator()(std::int32_t row, std::int32_t token, float value) const {
        const std::size_t i = static_cast<std::size_t>(token) * rows + row;
        residual[i]         = __float2bfloat16(__bfloat162float(residual[i]) + value);
    }
};

// Activation layout. For column block cb and group g the kernel wants one contiguous run per
// n-tile: 32 lanes x 16 bytes, lane l holding token (nt*8 + l/4) and, for each (ks, hi), the four
// k at g*64 + ks*32 + hi*16 + (l%4)*4. The quantiser owns this buffer, so it writes that order
// directly and the GEMM needs one 128-bit shared read per n-tile.
template <int kCols, int BN>
__global__ void quantize_activations(const __nv_bfloat16* __restrict__ x, std::int32_t tokens,
                                     std::int8_t* __restrict__ codes, __half* __restrict__ scales) {
    constexpr std::int32_t kGroups = kCols / kGroup;
    constexpr int kNTiles          = BN / 8;
    const std::int32_t token       = blockIdx.x;
    if (token >= tokens) { return; }
    const std::int32_t cb     = token / BN;
    const std::int32_t tok_in = token % BN;
    const int nt              = tok_in / 8;
    const int gid             = tok_in % 8;

    for (std::int32_t g = threadIdx.x; g < kGroups; g += blockDim.x) {
        const __nv_bfloat16* src = x + static_cast<std::size_t>(token) * kCols + g * kGroup;
        float amax               = 0.0F;
        for (int j = 0; j < kGroup; ++j) {
            amax = fmaxf(amax, fabsf(__bfloat162float(src[j])));
        }
        amax = fmaxf(amax, 1.0e-20F);
        scales[(static_cast<std::size_t>(cb) * kGroups + g) * BN + tok_in] =
            __float2half(amax / 127.0F);
        const float inv = 127.0F / amax;

        std::int8_t* tile = codes + ((static_cast<std::size_t>(cb) * kGroups + g) * kNTiles + nt) *
                                        (32 * 16);
        // Four codes at a time: j is contiguous in the destination, so this is a 32-bit store.
        for (int ks = 0; ks < 2; ++ks) {
            for (int hi = 0; hi < 2; ++hi) {
                for (int tig = 0; tig < 4; ++tig) {
                    std::int8_t quad[4];
                    for (int j = 0; j < 4; ++j) {
                        const float v =
                            __bfloat162float(src[ks * 32 + hi * 16 + tig * 4 + j]) * inv;
                        quad[j] =
                            static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v))));
                    }
                    std::int8_t* dst = tile + (gid * 4 + tig) * 16 + ks * 8 + hi * 4;
                    *reinterpret_cast<unsigned*>(dst) =
                        *reinterpret_cast<const unsigned*>(quad);
                }
            }
        }
    }
}

// MT m-tiles of 16 rows and NT n-tiles of 8 tokens per warp, over a 4x4 warp grid: the block tile
// is (kWarpsM * MT * 16) rows by (kWarpsN * NT * 8) tokens.
template <class Codec, int kCols, int MT, int NT, class RowMap, class Epilogue>
__global__ __launch_bounds__(kThreads) void a8_mma_kernel(
    const std::uint8_t* __restrict__ w_codes, const std::uint8_t* __restrict__ w_high,
    const __half* __restrict__ w_scales, const std::int8_t* __restrict__ x_codes,
    const __half* __restrict__ x_scales, std::int32_t tokens, RowMap rows, Epilogue epilogue,
    int panel_shift) {
    constexpr int kGroups  = kCols / kGroup;
    constexpr int BM       = RowMap::kStagedRows;
    constexpr int BN       = kWarpsN * NT * 8;
    constexpr int kNTiles  = BN / 8;
    constexpr int kWBytes  = BM * kWRow;
    constexpr int kHBytes  = Codec::kHasHigh ? BM * 8 : 0;
    constexpr int kXBytes  = kNTiles * 32 * 16;
    constexpr int kXsBytes = BN * 2;
    constexpr int kStage   = kWBytes + kHBytes + kXBytes + kXsBytes;
    constexpr int kRingBytes = BM * kRingGroups * 2;

    extern __shared__ char smem[];
    char* const s_base = smem;
    char* const s_ring = smem + kStages * kStage;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2; // 0..7, the row inside a 16-row tile
    const int tig    = lane & 3;  // 0..3, the k quarter
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;

    const std::int32_t row_block = blockIdx.x;
    const std::int32_t col_block = blockIdx.y;

    const char* const x_blk =
        reinterpret_cast<const char*>(x_codes) + static_cast<std::size_t>(col_block) * kGroups * kXBytes;
    const char* const xs_blk = reinterpret_cast<const char*>(x_scales) +
                               static_cast<std::size_t>(col_block) * kGroups * kXsBytes;

    // Under the panel layout the code records of (1 << panel_shift) consecutive rows are contiguous
    // within a k-group, so this block's whole row tile for one group arrives in whole cache lines
    // rather than one scattered 32-byte record per row. A shift of zero makes the same expression
    // collapse to `row * (kCols / 2) + g * 32`, which is the row-major address exactly -- so both
    // layouts share one code path and there is no second one to keep in step.
    //
    // The addresses are resolved once here rather than inside `issue`. They are loop-invariant, and
    // leaving them in the lambda costs about 30% (measured in tools/w4a8_marlin_probe.cu): the row
    // map and the panel arithmetic compete with the accumulators for registers on every group.
    const std::size_t panel_mask = (std::size_t{1} << panel_shift) - 1;
    constexpr int kWIter         = (BM * 2 + kThreads - 1) / kThreads;
    constexpr int kHIter         = (BM + kThreads - 1) / kThreads;
    const std::uint8_t* w_lane[kWIter];
    int w_dst[kWIter];
#pragma unroll
    for (int i = 0; i < kWIter; ++i) {
        // kThreads can exceed the work, so surplus lanes address row 0 and then stay idle.
        const int c           = tid + i * kThreads;
        const int safe        = c < BM * 2 ? c : 0;
        const int staged      = safe >> 1;
        const int half        = safe & 1;
        const std::size_t row = static_cast<std::size_t>(rows.weight_row(row_block, staged));
        w_lane[i] = w_codes + (row & ~panel_mask) * (kCols / 2) +
                    (row & panel_mask) * (kGroup / 2) + half * 16;
        w_dst[i] = c < BM * 2 ? (staged * kWRow + half * 16) : -1;
    }
    const std::uint8_t* h_lane[kHIter];
    int h_dst[kHIter];
    if constexpr (Codec::kHasHigh) {
#pragma unroll
        for (int i = 0; i < kHIter; ++i) {
            const int staged      = tid + i * kThreads;
            const int safe        = staged < BM ? staged : 0;
            const std::size_t row = static_cast<std::size_t>(rows.weight_row(row_block, safe));
            h_lane[i] = w_high + (row & ~panel_mask) * (static_cast<std::size_t>(kGroups) * 8) +
                        (row & panel_mask) * 8;
            h_dst[i] = staged < BM ? (kWBytes + staged * 8) : -1;
        }
    }
    const int w_group_stride = (kGroup / 2) << panel_shift;
    const int h_group_stride = 8 << panel_shift;

    auto issue = [&](int g, int buf) {
        char* const dst = s_base + buf * kStage;
        // W: one row's group is 32 contiguous bytes, so two 16-byte copies per row.
#pragma unroll
        for (int i = 0; i < kWIter; ++i) {
            if (w_dst[i] >= 0) {
                cp_async<16>(dst + w_dst[i],
                             w_lane[i] + static_cast<std::size_t>(g) * w_group_stride);
            }
        }
        if constexpr (Codec::kHasHigh) {
#pragma unroll
            for (int i = 0; i < kHIter; ++i) {
                if (h_dst[i] >= 0) {
                    cp_async<8>(dst + h_dst[i],
                                h_lane[i] + static_cast<std::size_t>(g) * h_group_stride);
                }
            }
        }
#pragma unroll
        for (int c = tid; c < kXBytes / 16; c += kThreads) {
            cp_async<16>(dst + kWBytes + kHBytes + c * 16,
                         x_blk + static_cast<std::size_t>(g) * kXBytes + c * 16);
        }
#pragma unroll
        for (int c = tid; c < kXsBytes / 16; c += kThreads) {
            cp_async<16>(dst + kWBytes + kHBytes + kXBytes + c * 16,
                         xs_blk + static_cast<std::size_t>(g) * kXsBytes + c * 16);
        }
        // The scale ring rides the commit group of the stage that first reads it.
        if (g % kRingGroups == 0) {
#pragma unroll
            for (int staged = tid; staged < BM; staged += kThreads) {
                const std::size_t row = static_cast<std::size_t>(rows.weight_row(row_block, staged));
                cp_async<16>(s_ring + ((g / kRingGroups) % kRingBufs) * kRingBytes +
                                 staged * kRingGroups * 2,
                             reinterpret_cast<const char*>(w_scales) +
                                 (row * kGroups + g) * sizeof(__half));
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0F;

#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (i < kGroups) { issue(i, i); }
    }

    for (int g = 0; g < kGroups; ++g) {
        const int buf = g % kStages;
        if (g + kStages - 1 < kGroups) { issue(g + kStages - 1, (g + kStages - 1) % kStages); }
        const int issued  = (g + kStages < kGroups) ? (g + kStages) : kGroups;
        const int allowed = issued - (g + 1);
        if (allowed >= 1) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const char* const sa    = s_base + buf * kStage;
        const char* const sh    = sa + kWBytes;
        const char* const sb    = sh + kHBytes;
        const __half* const sxs = reinterpret_cast<const __half*>(sb + kXBytes);
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / kRingGroups) % kRingBufs) * kRingBytes);

        unsigned af[MT][2][4];
        unsigned bf[NT][2][2];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0 = RowMap::staged_row(warp_m, m, gid);
            const int r1 = r0 + 8;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int off = ks * 16 + tig * 2;
                unsigned h0 = 0, h1 = 0, h2 = 0, h3 = 0;
                if constexpr (Codec::kHasHigh) {
                    // The four codes at ks*32 + tig*4 take one nibble of the high byte covering
                    // codes 8i..8i+7, and the +16 chunk takes a nibble two bytes further on.
                    const int byte = ks * 4 + (tig >> 1);
                    const int shift = (tig & 1) * 4;
                    h0 = lds8(sh + r0 * 8 + byte) >> shift;
                    h1 = lds8(sh + r1 * 8 + byte) >> shift;
                    h2 = lds8(sh + r0 * 8 + byte + 2) >> shift;
                    h3 = lds8(sh + r1 * 8 + byte + 2) >> shift;
                }
                af[m][ks][0] = Codec::decode(lds16(sa + r0 * kWRow + off), h0);
                af[m][ks][1] = Codec::decode(lds16(sa + r1 * kWRow + off), h1);
                af[m][ks][2] = Codec::decode(lds16(sa + r0 * kWRow + off + 8), h2);
                af[m][ks][3] = Codec::decode(lds16(sa + r1 * kWRow + off + 8), h3);
            }
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const uint4 b = lds128(sb + ((warp_n * NT + n) * 32 + lane) * 16);
            bf[n][0][0]   = b.x;
            bf[n][0][1]   = b.y;
            bf[n][1][0]   = b.z;
            bf[n][1][1]   = b.w;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int sr    = RowMap::staged_row(warp_m, m, gid);
            const float ws0 = __half2float(ring[sr * kRingGroups + (g % kRingGroups)]);
            const float ws1 = __half2float(ring[(sr + 8) * kRingGroups + (g % kRingGroups)]);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
                const int c     = (warp_n * NT + n) * 8 + tig * 2;
                const float xa0 = __half2float(sxs[c]);
                const float xa1 = __half2float(sxs[c + 1]);
                acc[m][n][0]    = fmaf(static_cast<float>(s[0]), ws0 * xa0, acc[m][n][0]);
                acc[m][n][1]    = fmaf(static_cast<float>(s[1]), ws0 * xa1, acc[m][n][1]);
                acc[m][n][2]    = fmaf(static_cast<float>(s[2]), ws1 * xa0, acc[m][n][2]);
                acc[m][n][3]    = fmaf(static_cast<float>(s[3]), ws1 * xa1, acc[m][n][3]);
            }
        }
        // The next iteration's issue writes the buffer this one just read.
        __syncthreads();
    }

    const std::int32_t out_base = row_block * RowMap::kRowsPerBlock;
    const std::int32_t col_base = col_block * BN;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c0 = col_base + (warp_n * NT + n) * 8 + tig * 2;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            if constexpr (Epilogue::kPaired) {
                const int row = out_base + RowMap::output_row(warp_m, 0, gid) + half * 8;
                epilogue(row, c0, acc[0][n][half * 2], acc[1][n][half * 2]);
                epilogue(row, c0 + 1, acc[0][n][half * 2 + 1], acc[1][n][half * 2 + 1]);
            } else {
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const int row = out_base + RowMap::output_row(warp_m, m, gid) + half * 8;
                    epilogue(row, c0, acc[m][n][half * 2]);
                    epilogue(row, c0 + 1, acc[m][n][half * 2 + 1]);
                }
            }
        }
    }
}

template <class Codec, int MT, int NT, class RowMap>
[[nodiscard]] constexpr std::size_t shared_bytes(int input_rows) {
    constexpr int BM      = RowMap::kStagedRows;
    constexpr int BN      = kWarpsN * NT * 8;
    constexpr int kWBytes = BM * kWRow;
    constexpr int kHBytes = Codec::kHasHigh ? BM * 8 : 0;
    constexpr int kXBytes = (BN / 8) * 32 * 16;
    (void)input_rows;
    return static_cast<std::size_t>(kStages) * (kWBytes + kHBytes + kXBytes + BN * 2) +
           static_cast<std::size_t>(kRingBufs) * BM * kRingGroups * 2;
}

// Transient bytes the activation planes need for T in [min,max].
[[nodiscard]] inline std::size_t activation_workspace_bytes(std::int32_t input_rows,
                                                            std::int32_t max_tokens) {
    const std::size_t t      = static_cast<std::size_t>(max_tokens);
    const std::size_t groups = static_cast<std::size_t>(input_rows) / kGroup;
    return ((t * static_cast<std::size_t>(input_rows) + 255) / 256) * 256 +
           ((t * groups * sizeof(__half) + 255) / 256) * 256;
}

// The widest token tile that divides T. A wider tile is strictly better -- it removes whole passes
// over the weight matrix -- so this only ever falls back for the narrow chunks the scheduler hands
// a ragged prompt tail.
[[nodiscard]] inline int token_tile(std::int32_t tokens, int widest) {
    for (int bn = widest; bn >= 128; bn >>= 1) {
        if (tokens % bn == 0) { return bn; }
    }
    return 0;
}

[[nodiscard]] inline bool tokens_supported(std::int32_t tokens) {
    return tokens >= 128 && tokens % 128 == 0;
}

} // namespace ninfer::ops::detail::rowsplit_a8
