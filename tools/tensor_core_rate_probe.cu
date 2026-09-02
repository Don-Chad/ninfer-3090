// Measures achieved tensor-core throughput on this GPU for the MMA shapes that matter to
// ninfer-3090: the BF16 f32-accumulate path the kernels use today, the INT8 path the KV
// attention kernel already uses, and the INT4 path nothing uses yet.
//
// Each warp runs CHAINS independent accumulator chains so MMA latency is hidden and the
// measurement reflects issue throughput rather than dependency stalls. Operands stay in
// registers; results are consumed at the end so nothing is optimised away.

#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);                 \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

constexpr int kChains = 8;
constexpr int kIters  = 8192;

// ---- BF16, m16n8k16, f32 accumulate (what ninfer's mma_bf16 emits) ----
__global__ void bf16_f32_probe(float* sink, unsigned seed) {
    unsigned a0 = seed, a1 = seed ^ 0x5a5au, a2 = seed + 7u, a3 = seed * 3u;
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    float c[kChains][4];
#pragma unroll
    for (int i = 0; i < kChains; ++i) { c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f; }

    for (int it = 0; it < kIters; ++it) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (acc == 1234.5678f) { sink[0] = acc; }
}

// ---- FP16, m16n8k16, f16 accumulate (the un-halved GeForce rate, for reference) ----
__global__ void f16_f16_probe(float* sink, unsigned seed) {
    unsigned a0 = seed, a1 = seed ^ 0x5a5au, a2 = seed + 7u, a3 = seed * 3u;
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    unsigned c[kChains][2];
#pragma unroll
    for (int i = 0; i < kChains; ++i) { c[i][0] = c[i][1] = 0u; }

    for (int it = 0; it < kIters; ++it) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                         "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                         : "+r"(c[i][0]), "+r"(c[i][1])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    unsigned acc = 0u;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += c[i][0] + c[i][1]; }
    if (acc == 0xdeadbeefu) { sink[0] = (float)acc; }
}

// ---- INT8, m16n8k32, s32 accumulate (what mma_s8 emits) ----
__global__ void s8_probe(float* sink, unsigned seed) {
    unsigned a0 = seed, a1 = seed ^ 0x5a5au, a2 = seed + 7u, a3 = seed * 3u;
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    int c[kChains][4];
#pragma unroll
    for (int i = 0; i < kChains; ++i) { c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0; }

    for (int it = 0; it < kIters; ++it) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    int acc = 0;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (acc == 0x7fffffff) { sink[0] = (float)acc; }
}

// ---- INT8, m16n8k16 (the finer-k shape a group-16 scale would force) ----
__global__ void s8_k16_probe(float* sink, unsigned seed) {
    unsigned a0 = seed, a1 = seed ^ 0x5a5au;
    unsigned b0 = seed ^ 0x1234u;
    int c[kChains][4];
#pragma unroll
    for (int i = 0; i < kChains; ++i) { c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0; }

    for (int it = 0; it < kIters; ++it) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                         "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                         : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                         : "r"(a0), "r"(a1), "r"(b0));
        }
    }
    int acc = 0;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (acc == 0x7fffffff) { sink[0] = (float)acc; }
}

// ---- INT4, m16n8k64, s32 accumulate ----
__global__ void s4_probe(float* sink, unsigned seed) {
    unsigned a0 = seed, a1 = seed ^ 0x5a5au, a2 = seed + 7u, a3 = seed * 3u;
    unsigned b0 = seed ^ 0x1234u, b1 = seed + 11u;
    int c[kChains][4];
#pragma unroll
    for (int i = 0; i < kChains; ++i) { c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0; }

    for (int it = 0; it < kIters; ++it) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) {
            asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    int acc = 0;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += c[i][0] + c[i][1] + c[i][2] + c[i][3]; }
    if (acc == 0x7fffffff) { sink[0] = (float)acc; }
}

template <class K> double run(K kernel, int blocks, int threads, float* sink, double ops_per_mma) {
    // warmup
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

    const double warps    = (double)blocks * (threads / 32);
    const double mmas     = warps * (double)kIters * kChains * reps;
    const double total_op = mmas * ops_per_mma;
    return total_op / (ms / 1000.0) / 1e12; // TOPS or TFLOPS
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s  sm_%d%d  %d SMs  %.0f MHz  mem %.0f MHz x %d-bit\n\n", p.name, p.major,
           p.minor, p.multiProcessorCount, p.clockRate / 1000.0, p.memoryClockRate / 1000.0,
           p.memoryBusWidth);

    const int threads = 256;
    const int blocks  = p.multiProcessorCount * 8;
    float* sink       = nullptr;
    CHECK(cudaMalloc(&sink, sizeof(float)));

    // FLOPs/OPs per warp-level MMA = 2*M*N*K
    const double f_k16 = 2.0 * 16 * 8 * 16;
    const double f_k32 = 2.0 * 16 * 8 * 32;
    const double f_k64 = 2.0 * 16 * 8 * 64;

    const double bf16 = run(bf16_f32_probe, blocks, threads, sink, f_k16);
    const double f16  = run(f16_f16_probe, blocks, threads, sink, f_k16);
    const double s8   = run(s8_probe, blocks, threads, sink, f_k32);
    const double s8k16 = run(s8_k16_probe, blocks, threads, sink, f_k16);
    const double s4   = run(s4_probe, blocks, threads, sink, f_k64);

    printf("  %-34s %8.1f TFLOPS   (1.00x baseline)\n", "BF16 m16n8k16, f32 accumulate", bf16);
    printf("  %-34s %8.1f TFLOPS   %.2fx\n", "FP16 m16n8k16, f16 accumulate", f16, f16 / bf16);
    printf("  %-34s %8.1f TOPS     %.2fx\n", "INT8 m16n8k32", s8, s8 / bf16);
    printf("  %-34s %8.1f TOPS     %.2fx\n", "INT8 m16n8k16", s8k16, s8k16 / bf16);
    printf("  %-34s %8.1f TOPS     %.2fx\n", "INT4 m16n8k64", s4, s4 / bf16);

    cudaFree(sink);
    return 0;
}
