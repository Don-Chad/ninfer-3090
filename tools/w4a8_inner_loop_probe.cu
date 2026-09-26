// Gate B1, arithmetic level.
//
// The premise of Workstream B is that INT8 tensor cores give ~4.7x the BF16 f32-accumulate rate
// on GA102. The stated risk is that a W4A8 route cannot keep that advantage, because it must do
// two things the BF16 dequantising GEMM does not:
//
//   1. unpack Q4G64 4-bit codes into s8 operands, and
//   2. rescale the s32 accumulator once per 64-deep scale group (cvt + fma on CUDA cores).
//
// Both run on the CUDA cores and compete with the MMA pipe. This probe measures the inner loop at
// the real arithmetic intensity of the 27B mlp/gate_up shape so the gate is decided on issue rates
// rather than on a half-tuned GEMM, which could produce a false negative.
//
//   A: pure s8 MMA                          -> the ceiling
//   B: s8 MMA + per-group rescale           -> adds the group-64 cvt/fma
//   C: s8 MMA + rescale + 4-bit unpack      -> the full W4A8 inner loop
//   D: bf16 MMA                             -> what the code issues today
//
// Useful work counted for every variant is the same: 2*M*N*K MACs of the logical GEMM.

#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int kChains = 4;   // independent accumulator chains per warp
constexpr int kGroups = 512; // scale groups of 64 processed per kernel call

// One k=64 scale group == two m16n8k32 s8 MMAs.
#define MMA_S8(c, a0, a1, a2, a3, b0, b1)                                                          \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "                                \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                         \
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])                                  \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__global__ void probe_a(float* sink, unsigned seed) {
    unsigned a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = seed * (i + 1u) + 0x9e37u; }
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    int c[kChains][4] = {};
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            MMA_S8(c[i], a[0], a[1], a[2], a[3], b0, b1);
            MMA_S8(c[i], a[4], a[5], a[6], a[7], b0, b1);
        }
    }
    int acc = 0;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (acc == 0x7fffffff) { sink[0] = (float)acc; }
}

__global__ void probe_b(float* sink, unsigned seed) {
    unsigned a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = seed * (i + 1u) + 0x9e37u; }
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    float f[kChains][4] = {};
    const float wscale = 0.013f + 1e-6f * (float)seed;
    const float xscale = 0.0071f;
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            int c[4] = {0, 0, 0, 0};
            MMA_S8(c, a[0], a[1], a[2], a[3], b0, b1);
            MMA_S8(c, a[4], a[5], a[6], a[7], b0, b1);
            // group-64 rescale: s32 -> f32, fused into the f32 accumulator
            const float s = wscale * xscale;
#pragma unroll
            for (int j = 0; j < 4; ++j) { f[i][j] = fmaf((float)c[j], s, f[i][j]); }
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += f[i][0] + f[i][1] + f[i][2] + f[i][3]; }
    if (acc == 1234.5678f) { sink[0] = acc; }
}

// Unpack eight 4-bit codes held in one 32-bit word into two 32-bit words of four s8 lanes.
__device__ __forceinline__ void unpack_q4(unsigned packed, unsigned& lo, unsigned& hi) {
    // low nibble of each byte -> lo lanes, high nibble -> hi lanes, both sign-corrected by -8.
    const unsigned low_mask = 0x0f0f0f0fu;
    const unsigned l        = packed & low_mask;
    const unsigned h        = (packed >> 4) & low_mask;
    lo                      = __vsub4(l, 0x08080808u);
    hi                      = __vsub4(h, 0x08080808u);
}

__global__ void probe_c(float* sink, unsigned seed) {
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { packed[i] = seed * (i + 3u) + 0x5bd1u; }
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    float f[kChains][4] = {};
    const float wscale = 0.013f + 1e-6f * (float)seed;
    const float xscale = 0.0071f;
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            // 32 s8 A-lanes per thread for k=64 come from 4 packed words.
            unsigned a[8];
            unpack_q4(packed[0], a[0], a[1]);
            unpack_q4(packed[1], a[2], a[3]);
            unpack_q4(packed[2], a[4], a[5]);
            unpack_q4(packed[3], a[6], a[7]);
            int c[4] = {0, 0, 0, 0};
            MMA_S8(c, a[0], a[1], a[2], a[3], b0, b1);
            MMA_S8(c, a[4], a[5], a[6], a[7], b0, b1);
            const float s = wscale * xscale;
#pragma unroll
            for (int j = 0; j < 4; ++j) { f[i][j] = fmaf((float)c[j], s, f[i][j]); }
            packed[0] += 1u; // keep the unpack live across iterations
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += f[i][0] + f[i][1] + f[i][2] + f[i][3]; }
    if (acc == 1234.5678f) { sink[0] = acc; }
}

__global__ void probe_d(float* sink, unsigned seed) {
    unsigned a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = seed * (i + 1u) + 0x9e37u; }
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    float c[kChains][4] = {};
    // bf16 m16n8k16: four MMAs cover the same k=64 the two s8 MMAs above cover.
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            }
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (acc == 1234.5678f) { sink[0] = acc; }
}

template <class K> double run(K kernel, int blocks, int threads, float* sink) {
    kernel<<<blocks, threads>>>(sink, 1u);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    const int reps = 5;
    for (int r = 0; r < reps; ++r) { kernel<<<blocks, threads>>>(sink, 1u); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    // Useful work is identical across variants: kChains warp-tiles of m16 x n8 over k=64,
    // for kGroups groups. 2*M*N*K MACs each.
    const double warps = (double)blocks * (threads / 32);
    const double ops =
        warps * (double)kGroups * kChains * (2.0 * 16.0 * 8.0 * 64.0) * (double)reps;
    return ops / (ms / 1000.0) / 1e12;
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    const int threads = 256;
    const int blocks  = p.multiProcessorCount * 8;
    float* sink       = nullptr;
    cudaMalloc(&sink, sizeof(float));

    const double d = run(probe_d, blocks, threads, sink);
    const double a = run(probe_a, blocks, threads, sink);
    const double b = run(probe_b, blocks, threads, sink);
    const double c = run(probe_c, blocks, threads, sink);

    printf("GPU: %s  sm_%d%d\n\n", p.name, p.major, p.minor);
    printf("  %-46s %7.1f T/s   1.00x   <- today's route\n", "D  bf16 MMA (f32 acc)", d);
    printf("  %-46s %7.1f T/s   %.2fx\n", "A  s8 MMA, no overhead", a, a / d);
    printf("  %-46s %7.1f T/s   %.2fx\n", "B  s8 MMA + group-64 rescale", b, b / d);
    printf("  %-46s %7.1f T/s   %.2fx   <- W4A8 inner loop\n", "C  s8 MMA + rescale + Q4 unpack", c,
           c / d);
    printf("\n  Gate B1 threshold on the inner loop: 1.80x\n");
    printf("  Result: %s (%.2fx)\n", (c / d >= 1.80) ? "PASS" : "FAIL", c / d);

    cudaFree(sink);
    return 0;
}
