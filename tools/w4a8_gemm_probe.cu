// B2 prototype: W4A8 tiled GEMM for the Qwen3.8-27B mlp/gate_up prefill shape.
//
//   C[N,T] = W[N,K] (Q4, group-64 FP16 scales) x X[K,T] (per-token s8)
//   N = 34816, K = 5120, T = 512
//
// Compared against the measured A16 route (q4_rowsplit_gemm_mma) at the same shape:
// 3357.7 us median, 54.36 TFLOP/s.
//
// Tiling: 128x128 block tile, BK=64 == exactly one scale group, so the s32 accumulator is
// rescaled once per group with no partial-group bookkeeping. 16 warps in a 4x4 grid, each warp
// owning 32 rows x 32 columns as 2x4 m16n8 tiles, over two alternating shared stages.
//
// Activations arrive pre-quantized to s8. In the real Op that is one fused pass over X, which is
// O(K*T) against the GEMM's O(N*K*T) -- 0.03% of the work at this shape -- so excluding it here
// does not flatter the result meaningfully.
//
// Measured on an RTX 3090 (build: nvcc -O3 -arch=sm_86):
//
//   step                                        us      TOP/s   vs A16
//   first working version                   2254.8      80.95    1.49x
//   + word stores for the unpack, hoisted B 2183.2      83.61    1.54x
//   + global loads pipelined one group ahead1780.7     102.51    1.89x
//   + per-token scale factored to epilogue  1700.9     107.32    1.97x
//
// The inner-loop probe says the arithmetic supports 3.45x, so at 1.97x this kernel reaches 57%
// of what the MMAs alone would allow. What did NOT move it: raising occupancy from 8 to 16 warps
// per SM (98 vs 160 registers, identical time), and cutting from two barriers per group to one.
// The remaining gap is the shared-memory operand path -- 32 separate LDS.32 per warp per group to
// build fragments. ldmatrix over a swizzled stage is the next thing to try, then cp.async for the
// global path.

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

constexpr int N  = 34816;
constexpr int K  = 5120;
constexpr int T  = 512;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64; // one Q4 scale group
constexpr int GROUPS = K / BK;
constexpr int SROW = BK + 16; // padded row stride: 16-byte aligned for uint4 stores, and 20
                              // words apart so the 8 rows a warp touches hit 8 distinct banks

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two Q4 codes per byte: low nibble is the even k, high nibble the odd k. Codes are offset
// binary, so subtract 8 to centre. __vsub4 does the four lanes at once.
__device__ __forceinline__ void unpack8(unsigned packed, unsigned& even, unsigned& odd) {
    const unsigned mask = 0x0f0f0f0fu;
    even                = __vsub4(packed & mask, 0x08080808u);
    odd                 = __vsub4((packed >> 4) & mask, 0x08080808u);
}

__global__ __launch_bounds__(512) void w4a8_gemm(const uint2* __restrict__ w_packed,
                                                 const __half* __restrict__ w_scale,
                                                 const int8_t* __restrict__ x_q,
                                                 const float* __restrict__ x_scale,
                                                 __nv_bfloat16* __restrict__ out) {
    // Two shared stages. With one stage the loop needs two barriers per group -- one after the
    // MMAs and one after the refill -- and the refill cannot overlap compute. Alternating stages
    // leaves a single barrier at the top of the loop and lets the store for g+1 run underneath the
    // MMAs for g.
    constexpr int STAGE = BM * SROW + BN * SROW;
    extern __shared__ char smem_raw[];
    // Stage pointers are computed by arithmetic, never indexed out of an array: a pointer array
    // indexed by a runtime buffer number lands in local memory and costs more than the barrier
    // this double buffering removes.
    int8_t* const s_base   = reinterpret_cast<int8_t*>(smem_raw);
    float* const sws_base  = reinterpret_cast<float*>(smem_raw + 2 * STAGE);
    float* const sxs       = sws_base + 2 * BM;                     // [BN]

    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warp     = tid >> 5;
    const int group_id = lane >> 2;  // 0..7
    const int tig      = lane & 3;   // 0..3
    const int warp_m   = warp >> 2;  // 0..3
    const int warp_n   = warp & 3;   // 0..3

    const int row_block = blockIdx.x * BM;
    const int col_block = blockIdx.y * BN;

    // Per-token activation scales are constant over k: load once.
    if (tid < BN) { sxs[tid] = x_scale[col_block + tid]; }

    float acc[2][4][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0f;

    // Global-load mapping: 4 threads per row for both tiles.
    const int ld_row = tid >> 2;        // 0..127
    const int ld_q   = tid & 3;         // 0..3, each covers 16 k

    // Software pipeline: the global reads for group g+1 are issued before the MMAs for group g,
    // so their latency is covered by compute instead of stalling behind __syncthreads(). At 157
    // registers only one block is resident per SM, so there is no second block to hide it for us.
    struct GlobalFrag {
        uint2 w;   // 8 packed bytes -> 16 codes
        uint4 x;   // 16 s8
        float ws;
    };
    const size_t w_row  = (size_t)(row_block + ld_row) * (K / 16) + ld_q;
    const size_t x_row  = (size_t)(col_block + ld_row) * K + ld_q * 16;
    const size_t ws_row = (size_t)(row_block + tid) * GROUPS;

    auto load_global = [&](int g, GlobalFrag& f) {
        f.w = w_packed[w_row + (size_t)(g * BK / 16)];
        f.x = *reinterpret_cast<const uint4*>(x_q + x_row + (size_t)(g * BK));
        if (tid < BM) { f.ws = __half2float(w_scale[ws_row + g]); }
    };
    auto store_smem = [&](const GlobalFrag& f, int buf) {
        int8_t* const sa_w  = s_base + buf * STAGE;
        const unsigned p[2] = {f.w.x, f.w.y};
        unsigned* dst       = reinterpret_cast<unsigned*>(sa_w + ld_row * SROW + ld_q * 16);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            unsigned even, odd;
            unpack8(p[i], even, odd);
            // Interleave back into k order: byte j of the packed word holds k=2j (low nibble)
            // and k=2j+1 (high). byte_perm builds both output words without byte stores.
            dst[i * 2]     = __byte_perm(even, odd, 0x5140);
            dst[i * 2 + 1] = __byte_perm(even, odd, 0x7362);
        }
        *reinterpret_cast<uint4*>(sa_w + BM * SROW + ld_row * SROW + ld_q * 16) = f.x;
        if (tid < BM) { sws_base[buf * BM + tid] = f.ws; }
    };

    GlobalFrag cur;
    load_global(0, cur);
    store_smem(cur, 0);

    for (int g = 0; g < GROUPS; ++g) {
        const int buf          = g & 1;
        const int8_t* const sa = s_base + buf * STAGE;
        const int8_t* const sb = sa + BM * SROW;
        const float* const sws = sws_base + buf * BM;
        __syncthreads();
        GlobalFrag next;
        if (g + 1 < GROUPS) { load_global(g + 1, next); }

        // One scale group: accumulate k=64 in s32, then rescale into f32.
        // Both operand sets are hoisted out of the MMA nest so each shared word is read once per
        // group rather than once per (m,n) pair.
        unsigned af[2][2][4];
        unsigned bf[4][2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int r0 = warp_m * 32 + m * 16 + group_id;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int k0 = ks * 32;
                af[m][ks][0] = *reinterpret_cast<const unsigned*>(sa + r0 * SROW + k0 + tig * 4);
                af[m][ks][1] =
                    *reinterpret_cast<const unsigned*>(sa + (r0 + 8) * SROW + k0 + tig * 4);
                af[m][ks][2] =
                    *reinterpret_cast<const unsigned*>(sa + r0 * SROW + k0 + tig * 4 + 16);
                af[m][ks][3] =
                    *reinterpret_cast<const unsigned*>(sa + (r0 + 8) * SROW + k0 + tig * 4 + 16);
            }
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c0col = warp_n * 32 + n * 8 + group_id;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int k0 = ks * 32;
                bf[n][ks][0] = *reinterpret_cast<const unsigned*>(sb + c0col * SROW + k0 + tig * 4);
                bf[n][ks][1] =
                    *reinterpret_cast<const unsigned*>(sb + c0col * SROW + k0 + tig * 4 + 16);
            }
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float ws0 = sws[warp_m * 32 + m * 16 + group_id];
            const float ws1 = sws[warp_m * 32 + m * 16 + group_id + 8];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
                // c0,c1 -> row r0 ; c2,c3 -> row r0+8 ; columns tig*2 and tig*2+1.
                // The per-token activation scale is constant over k, so it factors out of the
                // sum entirely and is applied once in the epilogue. That removes four float
                // multiplies per tile per group -- 2560 per thread over the whole k loop -- and
                // is exact rather than an approximation.
                acc[m][n][0] = fmaf((float)s[0], ws0, acc[m][n][0]);
                acc[m][n][1] = fmaf((float)s[1], ws0, acc[m][n][1]);
                acc[m][n][2] = fmaf((float)s[2], ws1, acc[m][n][2]);
                acc[m][n][3] = fmaf((float)s[3], ws1, acc[m][n][3]);
            }
        }
        if (g + 1 < GROUPS) { store_smem(next, buf ^ 1); }
    }

#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int r0 = row_block + warp_m * 32 + m * 16 + group_id;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c0    = col_block + warp_n * 32 + n * 8 + tig * 2;
            const float xs0 = sxs[warp_n * 32 + n * 8 + tig * 2];
            const float xs1 = sxs[warp_n * 32 + n * 8 + tig * 2 + 1];
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
    std::vector<int8_t> hx(x_count);
    std::vector<float> hxs(T);
    srand(1234);
    for (size_t i = 0; i < w_bytes; ++i) hw[i] = (unsigned char)(rand() & 0xff);
    for (size_t i = 0; i < ws_count; ++i) hws[i] = __float2half(0.002f + 0.001f * ((i % 7) / 7.0f));
    for (size_t i = 0; i < x_count; ++i) hx[i] = (int8_t)((rand() % 255) - 127);
    for (int i = 0; i < T; ++i) hxs[i] = 0.0031f + 0.0004f * ((i % 5) / 5.0f);

    void *dw, *dws, *dx, *dxs, *dout, *dflush;
    CHECK(cudaMalloc(&dw, w_bytes));
    CHECK(cudaMalloc(&dws, ws_count * sizeof(__half)));
    CHECK(cudaMalloc(&dx, x_count));
    CHECK(cudaMalloc(&dxs, T * sizeof(float)));
    CHECK(cudaMalloc(&dout, (size_t)N * T * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dflush, 256u << 20));
    CHECK(cudaMemcpy(dw, hw.data(), w_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dws, hws.data(), ws_count * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hx.data(), x_count, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dxs, hxs.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(N / BM, T / BN);
    const size_t smem = 2 * ((size_t)BM * SROW + (size_t)BN * SROW) + (2 * BM + BN) * sizeof(float);
    printf("GPU: %s   grid=(%d,%d) block=512  smem=%zu B\n", p.name, grid.x, grid.y, smem);

    w4a8_gemm<<<grid, 512, smem>>>((const uint2*)dw, (const __half*)dws, (const int8_t*)dx,
                                   (const float*)dxs, (__nv_bfloat16*)dout);
    CHECK(cudaDeviceSynchronize());

    // ---- correctness: sample outputs against a double-precision reference ----
    std::vector<__nv_bfloat16> hout((size_t)N * T);
    CHECK(cudaMemcpy(hout.data(), dout, (size_t)N * T * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    double worst = 0.0;
    int checked  = 0;
    for (int s = 0; s < 64; ++s) {
        const int r = (int)((size_t)rand() * 7919 % N);
        const int c = rand() % T;
        double ref  = 0.0;
        for (int g = 0; g < GROUPS; ++g) {
            long long dot = 0;
            for (int j = 0; j < BK; ++j) {
                const int k        = g * BK + j;
                const unsigned by  = hw[(size_t)r * (K / 2) + k / 2];
                const int code     = (int)((k % 2 == 0) ? (by & 0xf) : ((by >> 4) & 0xf)) - 8;
                dot += (long long)code * hx[(size_t)c * K + k];
            }
            ref += (double)dot * (double)__half2float(hws[(size_t)r * GROUPS + g]) * hxs[c];
        }
        const double got = (double)__bfloat162float(hout[(size_t)r * T + c]);
        const double rel = std::abs(got - ref) / std::max(std::abs(ref), 1e-6);
        worst            = std::max(worst, rel);
        ++checked;
    }
    printf("correctness: %d sampled outputs, worst relative error %.3e  (%s)\n", checked, worst,
           worst < 5e-3 ? "OK — bf16 store rounding" : "MISMATCH");
    if (worst >= 5e-3) { printf("  aborting: kernel is wrong, timing would be meaningless\n"); return 1; }

    // ---- timing, cold L2 between samples like the repo bench ----
    const int reps = 12;
    std::vector<float> ms(reps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaMemsetAsync(dflush, i & 0xff, 256u << 20));
        CHECK(cudaEventRecord(a));
        w4a8_gemm<<<grid, 512, smem>>>((const uint2*)dw, (const __half*)dws, (const int8_t*)dx,
                                       (const float*)dxs, (__nv_bfloat16*)dout);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double median_us = ms[reps / 2] * 1000.0;
    const double ops       = 2.0 * N * K * T;
    const double tops      = ops / (median_us * 1e-6) / 1e12;

    const double a16_us    = 3357.696; // measured q4_rowsplit_gemm_mma at this shape
    printf("\n  %-34s %9.1f us   %6.2f TOP/s\n", "W4A8 prototype", median_us, tops);
    printf("  %-34s %9.1f us   %6.2f TFLOP/s\n", "A16 route (measured)", a16_us, ops / (a16_us * 1e-6) / 1e12);
    printf("  %-34s %9.2fx\n", "speedup", a16_us / median_us);
    return 0;
}
