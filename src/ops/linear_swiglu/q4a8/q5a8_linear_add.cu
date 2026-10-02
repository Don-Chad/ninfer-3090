// Integer-activation route for the 27B mlp/down LinearAdd, companion to the gate_up route.
//
// mlp/down is Q5G64_F16S [5120,17408] in the same artifact: a 4-bit low plane laid out exactly as
// Q4's, an 8-byte-per-group high plane carrying each code's fifth bit, and an FP16 scale per group.
// A code decodes as ((low4 | hbit << 4) ^ 0x10) - 0x10, two's complement over [-16,15], which int8
// represents exactly -- so the fifth bit costs some unpack arithmetic and nothing else. The high
// bit for code i is bit i of the high byte covering codes 8i..8i+7.
//
// The high plane is consumed while unpacking into shared, so it never occupies shared memory.

#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kRows   = 5120;
constexpr std::int32_t kCols   = 17408;
constexpr std::int32_t kGroup  = 64;
constexpr std::int32_t kGroups = kCols / kGroup; // 272

constexpr int kBM      = 128;
constexpr int kBN      = 128;
constexpr int kBK      = 64;
constexpr int kSRow    = kBK + 16;
constexpr int kThreads = 512;

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

// Spread the four low bits of `bits` into the low bit of four bytes: byte j becomes bit j.
__device__ __forceinline__ unsigned spread4(unsigned bits) {
    const unsigned t = bits & 0xfu;
    return ((t) | (t << 7) | (t << 14) | (t << 21)) & 0x01010101u;
}

// Four packed bytes plus the matching high byte -> two s8 words of four codes, in k order.
__device__ __forceinline__ void unpack_q5(unsigned packed, unsigned high, unsigned& w0,
                                          unsigned& w1) {
    const unsigned mask = 0x0f0f0f0fu;
    const unsigned even = packed & mask;
    const unsigned odd  = (packed >> 4) & mask;
    // k order: byte j of the packed word holds code 2j (low nibble) and 2j+1 (high nibble).
    unsigned lo0 = __byte_perm(even, odd, 0x5140); // codes 0..3
    unsigned lo1 = __byte_perm(even, odd, 0x7362); // codes 4..7
    // Fifth bit: code i takes bit i of the high byte.
    lo0 |= spread4(high) << 4;
    lo1 |= spread4(high >> 4) << 4;
    // Five-bit two's complement: (v ^ 16) - 16, per byte.
    w0 = __vsub4(lo0 ^ 0x10101010u, 0x10101010u);
    w1 = __vsub4(lo1 ^ 0x10101010u, 0x10101010u);
}

__global__ void quantize_down_activations(const __nv_bfloat16* __restrict__ x, std::int32_t tokens,
                                          std::int8_t* __restrict__ codes,
                                          __half* __restrict__ scales) {
    const std::int32_t token = blockIdx.x;
    if (token >= tokens) { return; }
    for (std::int32_t g = threadIdx.x; g < kGroups; g += blockDim.x) {
        const __nv_bfloat16* src = x + static_cast<std::size_t>(token) * kCols + g * kGroup;
        float amax               = 0.0F;
        for (int j = 0; j < kGroup; ++j) {
            amax = fmaxf(amax, fabsf(__bfloat162float(src[j])));
        }
        amax                                                  = fmaxf(amax, 1.0e-20F);
        scales[static_cast<std::size_t>(token) * kGroups + g] = __float2half(amax / 127.0F);
        const float inv  = 127.0F / amax;
        std::int8_t* dst = codes + static_cast<std::size_t>(token) * kCols +
                           static_cast<std::size_t>(g) * kGroup;
        for (int j = 0; j < kGroup; ++j) {
            const float v = __bfloat162float(src[j]) * inv;
            dst[j]        = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v))));
        }
    }
}

__global__ __launch_bounds__(kThreads) void q5a8_add_kernel(
    const std::uint8_t* __restrict__ w_codes, const std::uint8_t* __restrict__ w_high,
    const __half* __restrict__ w_scales, const std::int8_t* __restrict__ x_codes,
    const __half* __restrict__ x_scales, __nv_bfloat16* __restrict__ residual,
    std::int32_t tokens) {
    extern __shared__ char smem[];
    std::int8_t* const sa = reinterpret_cast<std::int8_t*>(smem);
    std::int8_t* const sb = sa + kBM * kSRow;
    __half* const sws     = reinterpret_cast<__half*>(sb + kBN * kSRow);
    __half* const sxs     = sws + kBM;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;

    const int row_block = blockIdx.x * kBM;
    const int col_block = blockIdx.y * kBN;

    const int ld_row = tid >> 2;
    const int ld_q   = tid & 3;
    const int w_row  = row_block + ld_row;
    const int x_tok  = col_block + ld_row;

    struct Stage {
        uint2 w;
        unsigned high; // two high bytes, one per packed word
        uint4 x;
        __half ws;
        __half xs;
    };
    auto load_stage = [&](int g, Stage& s) {
        s.w = *reinterpret_cast<const uint2*>(w_codes +
                                              static_cast<std::size_t>(w_row) * (kCols / 2) +
                                              static_cast<std::size_t>(g) * (kBK / 2) + ld_q * 8);
        s.high = *reinterpret_cast<const std::uint16_t*>(
            w_high + static_cast<std::size_t>(w_row) * (static_cast<std::size_t>(kGroups) * 8) +
            static_cast<std::size_t>(g) * 8 + ld_q * 2);
        s.x = *reinterpret_cast<const uint4*>(x_codes + static_cast<std::size_t>(x_tok) * kCols +
                                              static_cast<std::size_t>(g) * kBK + ld_q * 16);
        if (tid < kBM) {
            s.ws = w_scales[static_cast<std::size_t>(row_block + tid) * kGroups + g];
        }
        if (tid < kBN) {
            s.xs = x_scales[static_cast<std::size_t>(col_block + tid) * kGroups + g];
        }
    };
    auto store_stage = [&](const Stage& s) {
        unsigned q[4];
        unpack_q5(s.w.x, s.high & 0xffu, q[0], q[1]);
        unpack_q5(s.w.y, (s.high >> 8) & 0xffu, q[2], q[3]);
        std::int8_t* wrow = sa + ld_row * kSRow + ld_q * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<unsigned*>(wrow + i * 16) = q[i];
        }
        const unsigned xw[4] = {s.x.x, s.x.y, s.x.z, s.x.w};
        std::int8_t* xrow    = sb + ld_row * kSRow + ld_q * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<unsigned*>(xrow + i * 16) = xw[i];
        }
        if (tid < kBM) { sws[tid] = s.ws; }
        if (tid < kBN) { sxs[tid] = s.xs; }
    };

    float acc[2][4][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0F;

    Stage cur;
    load_stage(0, cur);
    store_stage(cur);
    __syncthreads();

    for (int g = 0; g < kGroups; ++g) {
        Stage next;
        if (g + 1 < kGroups) { load_stage(g + 1, next); }

        unsigned af[2][2][4];
        unsigned bf[4][2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int r0   = warp_m * 32 + m * 16 + gid;
            const uint4 lo = lds128(sa + r0 * kSRow + tig * 16);
            const uint4 hi = lds128(sa + (r0 + 8) * kSRow + tig * 16);
            af[m][0][0] = lo.x; af[m][0][1] = hi.x; af[m][0][2] = lo.y; af[m][0][3] = hi.y;
            af[m][1][0] = lo.z; af[m][1][1] = hi.z; af[m][1][2] = lo.w; af[m][1][3] = hi.w;
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const uint4 b = lds128(sb + ((warp_n * 4 + n) * 8 + gid) * kSRow + tig * 16);
            bf[n][0][0] = b.x; bf[n][0][1] = b.y; bf[n][1][0] = b.z; bf[n][1][1] = b.w;
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int sr    = warp_m * 32 + m * 16 + gid;
            const float ws0 = __half2float(sws[sr]);
            const float ws1 = __half2float(sws[sr + 8]);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
                const int c     = (warp_n * 4 + n) * 8 + tig * 2;
                const float xa0 = __half2float(sxs[c]);
                const float xa1 = __half2float(sxs[c + 1]);
                acc[m][n][0]    = fmaf(static_cast<float>(s[0]), ws0 * xa0, acc[m][n][0]);
                acc[m][n][1]    = fmaf(static_cast<float>(s[1]), ws0 * xa1, acc[m][n][1]);
                acc[m][n][2]    = fmaf(static_cast<float>(s[2]), ws1 * xa0, acc[m][n][2]);
                acc[m][n][3]    = fmaf(static_cast<float>(s[3]), ws1 * xa1, acc[m][n][3]);
            }
        }
        __syncthreads();
        if (g + 1 < kGroups) {
            store_stage(next);
            __syncthreads();
        }
    }

    // residual is contiguous [kRows, tokens] and is the only observable mutation: each output
    // element belongs to exactly one thread, so a plain read-add-write is safe.
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c0 = col_block + (warp_n * 4 + n) * 8 + tig * 2;
            const int r0 = row_block + warp_m * 32 + m * 16 + gid;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row     = r0 + half * 8;
                const std::size_t i0 = static_cast<std::size_t>(c0) * kRows + row;
                const std::size_t i1 = static_cast<std::size_t>(c0 + 1) * kRows + row;
                residual[i0] = __float2bfloat16(__bfloat162float(residual[i0]) +
                                                acc[m][n][half * 2]);
                residual[i1] = __float2bfloat16(__bfloat162float(residual[i1]) +
                                                acc[m][n][half * 2 + 1]);
            }
        }
    }
}

} // namespace

bool q5a8_tokens_supported(std::int32_t tokens) { return tokens >= kBN && tokens % kBN == 0; }

bool q5a8_add_supported(const Weight& down, std::int32_t tokens) {
    return down.qtype == QType::Q5G64_F16S && down.layout == QuantLayout::RowSplit &&
           down.n == kRows && down.k == kCols && down.group == kGroup && down.qdata != nullptr &&
           down.qhigh != nullptr && down.scales != nullptr && tokens >= kBN && tokens % kBN == 0;
}

std::size_t q5a8_add_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q5a8 add workspace: invalid token interval");
    }
    const std::size_t t = static_cast<std::size_t>(max_tokens);
    return ((t * kCols + 255) / 256) * 256 + ((t * kGroups * sizeof(__half) + 255) / 256) * 256;
}

void q5a8_add_launch(const Tensor& x, const Weight& down, Tensor& residual,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!q5a8_add_supported(down, tokens)) {
        throw std::invalid_argument("q5a8 add: unsupported profile");
    }

    auto scope             = workspace.scope();
    const DeviceSpan codes = workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kCols);
    const DeviceSpan scales =
        workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kGroups * sizeof(__half));

    quantize_down_activations<<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens,
        reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<__half*>(scales.data));

    const dim3 grid(kRows / kBM, tokens / kBN);
    const std::size_t smem = static_cast<std::size_t>(kBM) * kSRow +
                             static_cast<std::size_t>(kBN) * kSRow +
                             (kBM + kBN) * sizeof(__half);
    q5a8_add_kernel<<<grid, kThreads, smem, stream>>>(
        static_cast<const std::uint8_t*>(down.qdata), static_cast<const std::uint8_t*>(down.qhigh),
        static_cast<const __half*>(down.scales),
        reinterpret_cast<const std::int8_t*>(codes.data),
        reinterpret_cast<const __half*>(scales.data),
        reinterpret_cast<__nv_bfloat16*>(residual.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
