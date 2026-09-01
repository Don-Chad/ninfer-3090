#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The registered 27B gate_up profile this route serves.
constexpr std::int32_t kRows   = 34816;
constexpr std::int32_t kOut    = kRows / 2; // gate rows [0,17408) then up rows [17408,34816)
constexpr std::int32_t kCols   = 5120;
constexpr std::int32_t kGroup  = 64;
constexpr std::int32_t kGroups = kCols / kGroup;

constexpr int kBM  = 128; // 64 gate rows plus their 64 up partners
constexpr int kBN  = 128; // tokens
constexpr int kBK  = 64;  // exactly one scale group, so the rescale needs no partial bookkeeping
constexpr int kSRow = kBK + 16; // padded: 16-byte aligned for vector access, 20 words apart so the
                                // eight rows a warp touches land in eight distinct banks
constexpr int kThreads = 512;

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A uint4 read through a char* is emitted as four 32-bit loads unless the compiler can prove the
// alignment, which it cannot through this pointer arithmetic.
__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(addr));
    return r;
}

// Four packed bytes hold eight codes as (low,high) nibble pairs. Codes are two's complement, so
// flipping bit 3 of every nibble and subtracting 8 reproduces the (n^8)-8 the A16 decode uses.
// Returns the s8 words for the pairs (b0,b1) and (b2,b3).
__device__ __forceinline__ void unpack_q4(unsigned packed, unsigned& w0, unsigned& w1) {
    packed ^= 0x88888888u;
    const unsigned mask = 0x0f0f0f0fu;
    const unsigned even = __vsub4(packed & mask, 0x08080808u);
    const unsigned odd  = __vsub4((packed >> 4) & mask, 0x08080808u);
    w0                  = __byte_perm(even, odd, 0x5140);
    w1                  = __byte_perm(even, odd, 0x7362);
}

// One scale per (token, group of 64). A per-token absmax over all 5120 channels is set by whichever
// channel is largest and starves every other one; a group of 64 confines an outlier to its own
// group. It is also free where it is used, because the kernel already rescales once per group.
__global__ void quantize_activations(const __nv_bfloat16* __restrict__ x, std::int32_t tokens,
                                     std::int8_t* __restrict__ codes, __half* __restrict__ scales) {
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
        const float inv = 127.0F / amax;
        std::int8_t* dst =
            codes + static_cast<std::size_t>(token) * kCols + static_cast<std::size_t>(g) * kGroup;
        for (int j = 0; j < kGroup; ++j) {
            const float v = __bfloat162float(src[j]) * inv;
            dst[j]        = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v))));
        }
    }
}

__global__ __launch_bounds__(kThreads) void q4a8_swiglu_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x_codes, const __half* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ out, std::int32_t tokens) {
    extern __shared__ char smem[];
    std::int8_t* const sa  = reinterpret_cast<std::int8_t*>(smem);         // [kBM][kSRow]
    std::int8_t* const sb  = sa + kBM * kSRow;                             // [kBN][kSRow]
    __half* const sws      = reinterpret_cast<__half*>(sb + kBN * kSRow);  // [kBM]
    __half* const sxs      = sws + kBM;                                    // [kBN]

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2; // 0..7, selects the row inside a 16-row tile
    const int tig    = lane & 3;  // 0..3, selects the k quarter
    const int warp_m = warp >> 2; // 0..3, one gate tile and its up partner
    const int warp_n = warp & 3;  // 0..3, four n-tiles of 8 tokens

    const int row_block = blockIdx.x * 64;   // 64 output rows per block
    const int col_block = blockIdx.y * kBN;

    // Shared rows 0..63 are the gate half, 64..127 the up half, so tile gt pairs with tile gt+4
    // and one warp holds both -- the SwiGLU epilogue then needs no shared exchange.
    const int ld_row = tid >> 2; // 0..127
    const int ld_q   = tid & 3;  // 0..3, sixteen k each
    const int w_abs_row =
        (ld_row < 64) ? (row_block + ld_row) : (kOut + row_block + (ld_row - 64));
    const int x_token = col_block + ld_row;

    // Register-staged prefetch: the reads for group g+1 issue before the MMAs for group g, so
    // their latency is covered by compute rather than stalling on the barrier.
    struct Stage {
        uint2 w;
        uint4 x;
        __half ws;
        __half xs;
    };
    auto load_stage = [&](int g, Stage& s) {
        s.w = *reinterpret_cast<const uint2*>(
            w_codes + static_cast<std::size_t>(w_abs_row) * (kCols / 2) +
            static_cast<std::size_t>(g) * (kBK / 2) + ld_q * 8);
        s.x = *reinterpret_cast<const uint4*>(
            x_codes + static_cast<std::size_t>(x_token) * kCols +
            static_cast<std::size_t>(g) * kBK + ld_q * 16);
        if (tid < kBM) {
            const int r = (tid < 64) ? (row_block + tid) : (kOut + row_block + (tid - 64));
            s.ws        = w_scales[static_cast<std::size_t>(r) * kGroups + g];
        }
        if (tid < kBN) {
            s.xs = x_scales[static_cast<std::size_t>(col_block + tid) * kGroups + g];
        }
    };
    // Shared rows hold k permuted: a thread's four MMA operand chunks for one row are the k
    // quarters [t*4, t*4+16, t*4+32, t*4+48), so storing them adjacent turns fragment assembly
    // into one 128-bit load per row. k = t*4 + c*16 + j maps to position t*16 + c*4 + j, which
    // puts this thread's four k-ordered words at byte offsets ld_q*4 + {0,16,32,48}.
    auto store_stage = [&](const Stage& s) {
        unsigned q[4];
        unpack_q4(s.w.x, q[0], q[1]);
        unpack_q4(s.w.y, q[2], q[3]);
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

    float acc[2][4][4]; // m=0 gate tile, m=1 its up partner
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
            const int r0   = warp_m * 16 + m * 64 + gid; // gate tile, then +64 for the up partner
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
            const int sr    = warp_m * 16 + m * 64 + gid;
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

    // Fused SwiGLU: acc[0] is a gate row, acc[1] its up partner, in the same thread's registers.
    // out is contiguous [kOut, tokens], so element (row, col) sits at col * kOut + row.
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int c0 = col_block + (warp_n * 4 + n) * 8 + tig * 2;
        const int r0 = row_block + warp_m * 16 + gid;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row  = r0 + half * 8;
            const float g0 = acc[0][n][half * 2];
            const float u0 = acc[1][n][half * 2];
            const float g1 = acc[0][n][half * 2 + 1];
            const float u1 = acc[1][n][half * 2 + 1];
            out[static_cast<std::size_t>(c0) * kOut + row] =
                __float2bfloat16(g0 / (1.0F + __expf(-g0)) * u0);
            out[static_cast<std::size_t>(c0 + 1) * kOut + row] =
                __float2bfloat16(g1 / (1.0F + __expf(-g1)) * u1);
        }
    }
}

} // namespace

bool q4a8_swiglu_supported(const Weight& gate_up, std::int32_t tokens) {
    return gate_up.qtype == QType::Q4G64_F16S && gate_up.layout == QuantLayout::RowSplit &&
           gate_up.n == kRows && gate_up.k == kCols && gate_up.group == kGroup &&
           gate_up.qdata != nullptr && gate_up.scales != nullptr && tokens >= kBN &&
           tokens % kBN == 0;
}

std::size_t q4a8_swiglu_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q4a8 swiglu workspace: invalid token interval");
    }
    const std::size_t t = static_cast<std::size_t>(max_tokens);
    // s8 codes plus one FP16 scale per (token, group), each 256-aligned by the arena.
    return ((t * kCols + 255) / 256) * 256 + ((t * kGroups * sizeof(__half) + 255) / 256) * 256;
}

void q4a8_swiglu_launch(const Tensor& x, const Weight& gate_up, Tensor& out,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!q4a8_swiglu_supported(gate_up, tokens)) {
        throw std::invalid_argument("q4a8 swiglu: unsupported profile");
    }

    auto scope             = workspace.scope();
    const DeviceSpan codes = workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kCols);
    const DeviceSpan scales =
        workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kGroups * sizeof(__half));

    quantize_activations<<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens,
        reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<__half*>(scales.data));

    const dim3 grid(kOut / 64, tokens / kBN);
    const std::size_t smem = static_cast<std::size_t>(kBM) * kSRow +
                             static_cast<std::size_t>(kBN) * kSRow +
                             (kBM + kBN) * sizeof(__half);
    q4a8_swiglu_kernel<<<grid, kThreads, smem, stream>>>(
        static_cast<const std::uint8_t*>(gate_up.qdata),
        static_cast<const __half*>(gate_up.scales),
        reinterpret_cast<const std::int8_t*>(codes.data),
        reinterpret_cast<const __half*>(scales.data),
        reinterpret_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
