// W4A8 prefill GEMM for the Qwen3.8-27B mlp/gate_up shape, on sm_86.
//
//   C[N,T] = W[N,K] (Q4, group-64 FP16 scales) x X[K,T] (per-token s8)
//   N = 34816, K = 5120, T = 512
//
// Measured against the A16 route this would replace -- q4_rowsplit_gemm_mma at the same shape,
// 3357.7us / 54.36 TFLOP/s, itself already at 91% of the BF16 f32-accumulate tensor ceiling:
//
//   structure                                          us      TOP/s   vs A16
//   ---------------------------------------------------------------------------
//   1: shared holds unpacked bytes, register staging
//      first working version                       2254.8      80.95    1.49x
//      + word stores for unpack, hoisted B frags   2183.2      83.61    1.54x
//      + global loads pipelined one group ahead    1780.7     102.51    1.89x
//      + per-token scale factored to the epilogue  1700.9     107.32    1.97x
//      + permuted shared k, forced LDS.128         1719.3     106.17    1.95x  (no change)
//   2: fragment-ordered layout, cp.async, packed shared
//      cp.async over pre-permuted W and X          1600.5     114.05    2.10x
//      + scales as a third contiguous async plane  1427.5     127.88    2.35x
//      + cp.async.cg rather than .ca               1415.2     128.99    2.37x
//
// Structure 2 is a data-layout change, not a kernel tweak, and it is what broke a plateau six
// separate kernel-level attempts could not:
//
//   1. W and X are pre-permuted into MMA-fragment order, so one (block, group) is a single
//      contiguous run and cp.async -- which cannot scatter -- can move it. In the real Op this is
//      a repack at materialization, free at runtime, and exactly why ninfer already carries
//      RowSplit and BlockScaleK16M128x4 layouts.
//   2. Shared holds W as PACKED nibbles. A's shared read traffic halves from 32 to 16 bytes per
//      m-tile; the unpack moves to the consumer, which costs ALU that was sitting idle at IPC
//      1.15 of 4.
//   3. cp.async removes the register staging and every shared store from the compute warps -- 16
//      STS.32 per thread per group off the MIO pipe. Registers fell 102 -> 85, shared 42.5 -> 26 KB.
//   4. The scales became a third async plane. Reading them synchronously inside the prefetch made
//      the thread wait on a global load in the one path whose purpose is not to wait; fixing that
//      alone was worth 2.10x -> 2.35x, the single largest step in this file.
//
// Things tried that did NOT help, recorded so they are not retried:
//
//   a. Occupancy. Structure 1 at 160 registers (8 warps/SM) vs 98 (16 warps/SM): identical time.
//   b. Barrier count. Two __syncthreads per group vs one over two stages: identical.
//   c. Fewer, wider shared loads -- the fix Nsight Compute prescribes. Its profile of the 1700.9us
//      version reports MIO throttle as the top stall (4.8 of 13.7 warp cycles between issues,
//      "Est. Local Speedup: 34.62%") and advises "fewer but wider loads". Permuting k inside each
//      shared row collapsed 32 LDS.32 per thread per group into 8 LDS.128, confirmed emitted by
//      issuing ld.shared.v4.u32 directly. No change -- because a 128-bit load moves the same four
//      128-byte wavefronts as four 32-bit ones. Instruction count was never the lever; only fewer
//      shared BYTES were, which is what structure 2 finally delivered.
//   d. More independent MMA chains. Giving all eight (m,n) tiles their own s32 accumulator with
//      the k-halves outermost compiled to 102 registers, identical to the byte: ptxas was already
//      scheduling them that way. A semantic no-op.
//   e. More work per shared byte. Widening the warp tile 2x4 -> 4x4 raises MMA-per-shared-byte by
//      a third but costs 173 registers, halving resident warps. 1890.3us, 10% worse.
//   f. A third pipeline stage. 1831.9us, worse: registers 85 -> 93, shared 26 -> 38 KB, plus a
//      three-way branch on the wait_group immediate every iteration. Two stages is the sweet spot.
//   g. Pointer arrays for the stages. sa_b[2] indexed by a runtime buffer number spills to local
//      memory and cost 32%; offset arithmetic fixes it.
//
// At 128.99 TOP/s this reaches 61% of the 211 TOP/s the inner-loop probe says the arithmetic
// allows (w4a8_inner_loop_probe.cu). Activations arrive pre-quantized and pre-permuted; in the Op
// that is one fused pass over X, O(K*T) against the GEMM's O(N*K*T), so it is 0.03% of the work
// at this shape. Correctness is checked against a double-precision reference on sampled outputs;
// 3.4e-3 is bf16 store rounding.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int N      = 34816;
constexpr int K      = 5120;
constexpr int T      = 512;
constexpr int BM     = 128;
constexpr int BN     = 128;
constexpr int BK     = 64; // one Q4 scale group
constexpr int GROUPS = K / BK;
constexpr int MTILES = BM / 16; // 8 m-tiles per block
constexpr int NTILES = BN / 8;  // 16 n-tiles per block
constexpr int WSTAGE = MTILES * 32 * 16; // 4096 packed bytes
constexpr int XSTAGE = NTILES * 32 * 16; // 8192 s8 bytes
constexpr int SSTAGE = BM * 2;                  // 128 FP16 scales, one per row, per group
constexpr int STAGE  = WSTAGE + XSTAGE + SSTAGE;
constexpr int STAGES = 2;                       // deeper prefetch: two groups always in flight

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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr), "l"(gmem));
}

// Four packed bytes hold eight codes as (low,high) nibble pairs. Returns the two s8 words the
// pairs (b0,b1) and (b2,b3) decode to, centred by subtracting 8.
__device__ __forceinline__ void unpack4b(unsigned packed, unsigned& w0, unsigned& w1) {
    const unsigned mask = 0x0f0f0f0fu;
    const unsigned even = __vsub4(packed & mask, 0x08080808u);
    const unsigned odd  = __vsub4((packed >> 4) & mask, 0x08080808u);
    w0                  = __byte_perm(even, odd, 0x5140);
    w1                  = __byte_perm(even, odd, 0x7362);
}

__global__ __launch_bounds__(512) void w4a8_async(const char* __restrict__ w_perm,
                                                  const __half* __restrict__ w_scale,
                                                  const char* __restrict__ x_perm,
                                                  const float* __restrict__ x_scale,
                                                  __nv_bfloat16* __restrict__ out) {
    extern __shared__ char smem[];
    char* const s_base = smem;
    float* const sxs   = reinterpret_cast<float*>(smem + STAGES * STAGE);

    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warp     = tid >> 5;
    const int group_id = lane >> 2;
    const int tig      = lane & 3;
    const int warp_m   = warp >> 2; // 0..3
    const int warp_n   = warp & 3;  // 0..3

    const int row_block = blockIdx.x * BM;
    const int col_block = blockIdx.y * BN;
    if (tid < BN) { sxs[tid] = x_scale[col_block + tid]; }

    // Each (block,group) reads one contiguous run from each pre-permuted tensor.
    const char* const w_blk = w_perm + (size_t)blockIdx.x * GROUPS * WSTAGE;
    const char* const x_blk = x_perm + (size_t)blockIdx.y * GROUPS * XSTAGE;
    // Scales are pre-permuted to [row_block][group][BM] so a group's 128 of them are contiguous.
    // Loading them synchronously inside the prefetch, as the first version did, made the thread
    // wait on a global read in the very path whose point is not to wait.
    const char* const s_blk = reinterpret_cast<const char*>(w_scale) +
                              (size_t)blockIdx.x * GROUPS * SSTAGE;

    auto issue = [&](int g, int buf) {
        char* const dst = s_base + buf * STAGE;
        if (tid < WSTAGE / 16) {
            cp_async16(dst + tid * 16, w_blk + (size_t)g * WSTAGE + tid * 16);
        }
        cp_async16(dst + WSTAGE + tid * 16, x_blk + (size_t)g * XSTAGE + tid * 16);
        if (tid < SSTAGE / 16) {
            cp_async16(dst + WSTAGE + XSTAGE + tid * 16, s_blk + (size_t)g * SSTAGE + tid * 16);
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc[2][4][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0f;

#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        if (i < GROUPS) { issue(i, i); }
    }

    for (int g = 0; g < GROUPS; ++g) {
        const int buf = g % STAGES;
        if (g + STAGES - 1 < GROUPS) { issue(g + STAGES - 1, (g + STAGES - 1) % STAGES); }
        // Groups issued so far are 0..g+STAGES-2, of which g must have landed. wait_group takes an
        // immediate, so branch on how many may stay outstanding.
        const int issued  = (g + STAGES < GROUPS) ? (g + STAGES) : GROUPS;
        const int allowed = issued - (g + 1);
        if (allowed >= 2) {
            asm volatile("cp.async.wait_group 2;");
        } else if (allowed == 1) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const char* const sa   = s_base + buf * STAGE;
        const char* const sb   = sa + WSTAGE;
        const __half* const ws = reinterpret_cast<const __half*>(sb + XSTAGE);

        unsigned af[2][2][4];
        unsigned bf[4][2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int rt   = warp_m * 2 + m;
            const uint4 pk = lds128(sa + (rt * 32 + lane) * 16);
            // bytes 0-3 -> (a0,a2) of ks0 ; 4-7 -> (a0,a2) of ks1
            // bytes 8-11 -> (a1,a3) of ks0 ; 12-15 -> (a1,a3) of ks1
            unpack4b(pk.x, af[m][0][0], af[m][0][2]);
            unpack4b(pk.y, af[m][1][0], af[m][1][2]);
            unpack4b(pk.z, af[m][0][1], af[m][0][3]);
            unpack4b(pk.w, af[m][1][1], af[m][1][3]);
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int nt  = warp_n * 4 + n;
            const uint4 b = lds128(sb + (nt * 32 + lane) * 16);
            bf[n][0][0] = b.x;
            bf[n][0][1] = b.y;
            bf[n][1][0] = b.z;
            bf[n][1][1] = b.w;
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float ws0 = __half2float(ws[(warp_m * 2 + m) * 16 + group_id]);
            const float ws1 = __half2float(ws[(warp_m * 2 + m) * 16 + group_id + 8]);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
                acc[m][n][0] = fmaf((float)s[0], ws0, acc[m][n][0]);
                acc[m][n][1] = fmaf((float)s[1], ws0, acc[m][n][1]);
                acc[m][n][2] = fmaf((float)s[2], ws1, acc[m][n][2]);
                acc[m][n][3] = fmaf((float)s[3], ws1, acc[m][n][3]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int r0 = row_block + (warp_m * 2 + m) * 16 + group_id;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c0    = col_block + (warp_n * 4 + n) * 8 + tig * 2;
            const float xs0 = sxs[(warp_n * 4 + n) * 8 + tig * 2];
            const float xs1 = sxs[(warp_n * 4 + n) * 8 + tig * 2 + 1];
            out[(size_t)r0 * T + c0]           = __float2bfloat16(acc[m][n][0] * xs0);
            out[(size_t)r0 * T + c0 + 1]       = __float2bfloat16(acc[m][n][1] * xs1);
            out[(size_t)(r0 + 8) * T + c0]     = __float2bfloat16(acc[m][n][2] * xs0);
            out[(size_t)(r0 + 8) * T + c0 + 1] = __float2bfloat16(acc[m][n][3] * xs1);
        }
    }
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));

    const size_t w_bytes  = (size_t)N * K / 2;
    const size_t ws_count = (size_t)N * GROUPS;
    const size_t x_count  = (size_t)T * K;

    std::vector<unsigned char> hw(w_bytes);
    std::vector<__half> hws(ws_count);
    std::vector<signed char> hx(x_count);
    std::vector<float> hxs(T);
    srand(1234);
    for (size_t i = 0; i < w_bytes; ++i) hw[i] = (unsigned char)(rand() & 0xff);
    for (size_t i = 0; i < ws_count; ++i) hws[i] = __float2half(0.002f + 0.001f * ((i % 7) / 7.0f));
    for (size_t i = 0; i < x_count; ++i) hx[i] = (signed char)((rand() % 255) - 127);
    for (int i = 0; i < T; ++i) hxs[i] = 0.0031f + 0.0004f * ((i % 5) / 5.0f);

    // ---- the repack a real materialization pass would do once ----
    std::vector<unsigned char> hwp(w_bytes);
    for (int rb = 0; rb < N / BM; ++rb)
        for (int g = 0; g < GROUPS; ++g)
            for (int rt = 0; rt < MTILES; ++rt)
                for (int l = 0; l < 32; ++l) {
                    const int gid = l >> 2, tg = l & 3;
                    unsigned char* dst =
                        &hwp[(((size_t)rb * GROUPS + g) * MTILES + rt) * 512 + (size_t)l * 16];
                    for (int half = 0; half < 2; ++half) {
                        const int row = rb * BM + rt * 16 + gid + half * 8;
                        for (int ks = 0; ks < 2; ++ks)
                            for (int hi = 0; hi < 2; ++hi) {
                                const int k = g * BK + ks * 32 + tg * 4 + hi * 16;
                                const int slot = half * 8 + ks * 4 + hi * 2;
                                dst[slot]     = hw[(size_t)row * (K / 2) + k / 2];
                                dst[slot + 1] = hw[(size_t)row * (K / 2) + k / 2 + 1];
                            }
                    }
                }
    // Scales repacked to [row_block][group][BM] so each group's 128 are contiguous.
    std::vector<__half> hwsp(ws_count);
    for (int rb = 0; rb < N / BM; ++rb)
        for (int g = 0; g < GROUPS; ++g)
            for (int r = 0; r < BM; ++r)
                hwsp[((size_t)rb * GROUPS + g) * BM + r] =
                    hws[(size_t)(rb * BM + r) * GROUPS + g];

    std::vector<signed char> hxp(x_count);
    for (int cb = 0; cb < T / BN; ++cb)
        for (int g = 0; g < GROUPS; ++g)
            for (int nt = 0; nt < NTILES; ++nt)
                for (int l = 0; l < 32; ++l) {
                    const int gid = l >> 2, tg = l & 3;
                    const int col = cb * BN + nt * 8 + gid;
                    signed char* dst =
                        &hxp[(((size_t)cb * GROUPS + g) * NTILES + nt) * 512 + (size_t)l * 16];
                    for (int ks = 0; ks < 2; ++ks)
                        for (int hi = 0; hi < 2; ++hi)
                            for (int j = 0; j < 4; ++j)
                                dst[ks * 8 + hi * 4 + j] =
                                    hx[(size_t)col * K + g * BK + ks * 32 + tg * 4 + hi * 16 + j];
                }

    void *dw, *dws, *dx, *dxs, *dout, *dflush;
    CHECK(cudaMalloc(&dw, w_bytes));
    CHECK(cudaMalloc(&dws, ws_count * sizeof(__half)));
    CHECK(cudaMalloc(&dx, x_count));
    CHECK(cudaMalloc(&dxs, T * sizeof(float)));
    CHECK(cudaMalloc(&dout, (size_t)N * T * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dflush, 256u << 20));
    CHECK(cudaMemcpy(dw, hwp.data(), w_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dws, hwsp.data(), ws_count * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hxp.data(), x_count, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dxs, hxs.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(N / BM, T / BN);
    const size_t smem = STAGES * STAGE + BN * sizeof(float);
    printf("GPU: %s   grid=(%d,%d) block=512  smem=%zu B\n", p.name, grid.x, grid.y, smem);

    w4a8_async<<<grid, 512, smem>>>((const char*)dw, (const __half*)dws, (const char*)dx,
                                    (const float*)dxs, (__nv_bfloat16*)dout);
    CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hout((size_t)N * T);
    CHECK(cudaMemcpy(hout.data(), dout, (size_t)N * T * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (int s = 0; s < 64; ++s) {
        const int r = (int)((size_t)rand() * 7919 % N);
        const int c = rand() % T;
        double ref  = 0.0;
        for (int g = 0; g < GROUPS; ++g) {
            long long dot = 0;
            for (int j = 0; j < BK; ++j) {
                const int k       = g * BK + j;
                const unsigned by = hw[(size_t)r * (K / 2) + k / 2];
                const int code    = (int)((k % 2 == 0) ? (by & 0xf) : ((by >> 4) & 0xf)) - 8;
                dot += (long long)code * hx[(size_t)c * K + k];
            }
            ref += (double)dot * (double)__half2float(hws[(size_t)r * GROUPS + g]) * hxs[c];
        }
        const double got = (double)__bfloat162float(hout[(size_t)r * T + c]);
        worst = std::max(worst, std::abs(got - ref) / std::max(std::abs(ref), 1e-6));
    }
    printf("correctness: 64 sampled outputs, worst relative error %.3e  (%s)\n", worst,
           worst < 5e-3 ? "OK" : "MISMATCH");
    if (worst >= 5e-3) {
        printf("  aborting: kernel is wrong, timing would be meaningless\n");
        return 1;
    }

    const int reps = 12;
    std::vector<float> ms(reps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaMemsetAsync(dflush, i & 0xff, 256u << 20));
        CHECK(cudaEventRecord(a));
        w4a8_async<<<grid, 512, smem>>>((const char*)dw, (const __half*)dws, (const char*)dx,
                                        (const float*)dxs, (__nv_bfloat16*)dout);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double median_us = ms[reps / 2] * 1000.0;
    const double ops       = 2.0 * N * K * T;
    printf("\n  %-34s %9.1f us   %6.2f TOP/s\n", "W4A8 cp.async + fragment layout", median_us,
           ops / (median_us * 1e-6) / 1e12);
    printf("  %-34s %9.1f us   %6.2f TOP/s\n", "W4A8 first structure", 1715.2, ops / (1715.2e-6) / 1e12);
    printf("  %-34s %9.1f us   %6.2f TFLOP/s\n", "A16 route (measured)", 3357.696,
           ops / (3357.696e-6) / 1e12);
    printf("  %-34s %9.2fx  (vs A16)\n", "speedup", 3357.696 / median_us);
    return 0;
}
