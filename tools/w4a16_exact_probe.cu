// W4 x A16-equivalent linear_swiglu on INT8 tensor cores, at NO accuracy cost.
//
// The single-plane W4A8 probe is fast (2.14-2.58x) but adds ~0.9% relative L2 that the model has
// never seen, because 8-bit fixed point over a group of 64 gives small elements in the group too
// few levels. This version removes that entirely.
//
// A bf16 activation is exactly an 8-bit signed mantissa times a power of two, so matching bf16
// needs roughly 16 bits of fixed point across a group, not 8. Carry the activation as a 16-bit
// integer split over two int8 planes:
//
//     q  = round(x / s),  s = absmax_group / 32512,  q in [-32512, 32512]
//     hi = (q + 128) >> 8            in [-127, 127]
//     lo = q - 256 * hi              in [-128, 127]
//     dot(w, q) = 256 * dot(w, hi) + dot(w, lo)         -- both exact in s8 x s8 -> s32
//
// Weight codes are integers in [-7,7], so they are exact in int8 too: the whole dot product is
// integer-exact, and the only rounding left is the 16-bit activation step. That step is
// absmax/32512, giving ~8.9e-6 relative error against bf16's own 2^-9 = 2e-3 -- about 200x finer
// than the route this would replace. Accumulation stays FP32 per group, as in the A16 route.
//
// The cost is two s8 MMAs per k-chunk where the single-plane version needed one. The open question
// this probe answers is whether the 4.74x rate advantage of s8 over bf16-with-f32-accumulate
// survives being halved. Overflow is safe: |dot_hi| <= 64*7*127 = 56,896, times 256 is 14.6e6,
// plus |dot_lo| stays well inside s32.
//
// Reports its own error and the A16 route's error against the same FP64 reference, so the
// comparison is like for like. What both routes still carry is the bf16 activation input and the
// bf16 output store; the integer path adds nothing measurable on top.
//
// MEASURED, real layer-0 gate_up:
//
//   outliers          W4+2xINT8      A16 route
//   none              1.213e-03      1.206e-03
//   12 chans x 16     1.681e-03      1.669e-03
//   12 chans x 64     7.542e-04      7.531e-04
//
//   2712 us median of five runs (2385-2735 spread), against A16's 3357.7us: 1.24x.
//
// Tuning history. Holding both planes' B fragments at once costs 126 registers; distributing the
// planes over the accumulator instead -- they are summed with weights 256 and 1, so hi can run for
// every tile and then lo, reusing one fragment set -- drops it to 107 and takes 1.21x to 1.24x.
// Retiling from 2m x 4n to 4m x 2n to cut the doubled B traffic went the wrong way: af grows to 32
// registers, total 127, and it fell to 1.10x. Note also that single runs of this kernel spread
// about 10%, so the 1.40x an early run reported was a boosted clock, not a result.
//
// ON WHAT THE ERROR IS RELATIVE TO, and what the 5090 actually does differently.
//
// The ~0.9% that single-plane W4A8 adds is measured against an exact FP64 evaluation of the SAME
// Q4 codes against the SAME bf16 activations. Q4 weight quantisation is not part of that
// comparison at all: both routes decode identical codes exactly, and that error is already baked
// into the checkpoint. The 0.9% is purely the cost of rounding activations to 8-bit fixed point.
//
// Upstream's A8 and A4 routes on sm_120a do the same thing, and upstream budgets for it explicitly.
// From tests/ops/linear/linear_test_common.cpp:
//
//     A16 relative-L2 allowance = kBf16UnitRoundoff        = 1/256 = 3.9e-3
//     A8  relative-L2 allowance = kA8QuantizationAllowance = 0.04   (4%)
//     A4  relative-L2 allowance = kA4QuantizationAllowance = 0.16   (16%)
//
// So the A8 path is permitted ten times the A16 error and A4 forty times, and the constants are
// named for what they are. The A4 route also runs against dedicated artifacts --
// model-cards/Qwen3.8-27B-nvfp4-NInfer, Qwen3.6-27B-nvfp4-NInfer -- whose weights are built for
// 4-bit activation execution rather than being groupwise-int reinterpreted.
//
// That means the sm_120a speed advantage is not a free gap the sm_86 fork is missing. It is partly
// a deliberate accuracy trade, and the A16 route the 3090 falls back to is the MORE accurate one.
// The 0.9% measured for single-plane W4A8 sits 4.4x inside upstream's own A8 allowance, so
// choosing it would be in policy, not a regression against how upstream ships. This file exists
// for the other choice: the same integer tensor cores with no accuracy cost at all.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
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
constexpr int NOUT    = N / 2;
constexpr int K      = 5120;
constexpr int T      = 512;
constexpr int BM     = 128;
constexpr int BN     = 128;
constexpr int BK     = 64;
constexpr int GROUPS = K / BK;
constexpr int MTILES = BM / 16;
constexpr int NTILES = BN / 8;
constexpr int WSTAGE = MTILES * 32 * 16;
constexpr int XSTAGE = NTILES * 32 * 16; // one activation plane
constexpr int SSTAGE = BM * 2;
constexpr int ASTAGE = BN * 2;
constexpr int STAGE  = WSTAGE + 2 * XSTAGE + SSTAGE + ASTAGE;
constexpr size_t LOW_PLANE   = (size_t)N * GROUPS * 32;
constexpr size_t SCALE_PLANE = (size_t)N * GROUPS * 2;
constexpr float QMAX = 32512.0f; // 127 * 256, so hi stays inside int8

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(a));
    return r;
}
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void unpack4b(unsigned packed, unsigned& w0, unsigned& w1) {
    packed ^= 0x88888888u;
    const unsigned mask = 0x0f0f0f0fu;
    const unsigned even = __vsub4(packed & mask, 0x08080808u);
    const unsigned odd  = __vsub4((packed >> 4) & mask, 0x08080808u);
    w0                  = __byte_perm(even, odd, 0x5140);
    w1                  = __byte_perm(even, odd, 0x7362);
}

__global__ void quantize_x2(const __nv_bfloat16* __restrict__ x, signed char* __restrict__ xhi,
                            signed char* __restrict__ xlo, __half* __restrict__ xs) {
    const int t  = blockIdx.x;
    const int cb = t / BN, cl = t % BN;
    const int nt = cl / 8, gid = cl % 8;
    for (int g = threadIdx.x; g < GROUPS; g += blockDim.x) {
        float amax = 0.0f;
        for (int j = 0; j < BK; ++j) {
            amax = fmaxf(amax, fabsf(__bfloat162float(x[(size_t)t * K + g * BK + j])));
        }
        amax            = fmaxf(amax, 1e-20f);
        const float inv = QMAX / amax;
        xs[((size_t)cb * GROUPS + g) * BN + cl] = __float2half(amax / QMAX);
        for (int j = 0; j < BK; ++j) {
            const float v = __bfloat162float(x[(size_t)t * K + g * BK + j]) * inv;
            int q         = __float2int_rn(v);
            q             = max(-32512, min(32512, q));
            const int hi  = (q + 128) >> 8;
            const int lo  = q - 256 * hi;
            const int ks = j / 32, rr = j % 32, h = rr / 16, tg = (rr % 16) / 4, jj = j % 4;
            const size_t idx = (((size_t)cb * GROUPS + g) * NTILES + nt) * 512 +
                               (size_t)(gid * 4 + tg) * 16 + ks * 8 + h * 4 + jj;
            xhi[idx] = (signed char)hi;
            xlo[idx] = (signed char)lo;
        }
    }
}

__global__ __launch_bounds__(512) void w4a16_swiglu(const char* __restrict__ w,
                                                     const char* __restrict__ ws,
                                                     const char* __restrict__ xhi,
                                                     const char* __restrict__ xlo,
                                                     const char* __restrict__ xs,
                                                     __nv_bfloat16* __restrict__ out) {
    extern __shared__ char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3, warp_m = warp >> 2, warp_n = warp & 3;
    const int rb = blockIdx.x, cb = blockIdx.y;

    const char* const wb  = w + (size_t)rb * GROUPS * WSTAGE;
    const char* const hb  = xhi + (size_t)cb * GROUPS * XSTAGE;
    const char* const lb  = xlo + (size_t)cb * GROUPS * XSTAGE;
    const char* const sb_ = ws + (size_t)rb * GROUPS * SSTAGE;
    const char* const ab_ = xs + (size_t)cb * GROUPS * ASTAGE;

    auto issue = [&](int g, int buf) {
        char* const d = smem + buf * STAGE;
        if (tid < WSTAGE / 16) { cp_async16(d + tid * 16, wb + (size_t)g * WSTAGE + tid * 16); }
        cp_async16(d + WSTAGE + tid * 16, hb + (size_t)g * XSTAGE + tid * 16);
        cp_async16(d + WSTAGE + XSTAGE + tid * 16, lb + (size_t)g * XSTAGE + tid * 16);
        if (tid < SSTAGE / 16) {
            cp_async16(d + WSTAGE + 2 * XSTAGE + tid * 16, sb_ + (size_t)g * SSTAGE + tid * 16);
        }
        if (tid < ASTAGE / 16) {
            cp_async16(d + WSTAGE + 2 * XSTAGE + SSTAGE + tid * 16,
                       ab_ + (size_t)g * ASTAGE + tid * 16);
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

    issue(0, 0);
    for (int g = 0; g < GROUPS; ++g) {
        const int buf = g & 1;
        if (g + 1 < GROUPS) { issue(g + 1, buf ^ 1); }
        if (g + 1 < GROUPS) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const char* const sa    = smem + buf * STAGE;
        const char* const sh    = sa + WSTAGE;
        const char* const sl    = sh + XSTAGE;
        const __half* const sws = reinterpret_cast<const __half*>(sl + XSTAGE);
        const __half* const sas = reinterpret_cast<const __half*>(sl + XSTAGE + SSTAGE);

        unsigned af[2][2][4];
        unsigned bf[4][2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int rt   = warp_m + m * 4;
            const uint4 pk = lds128(sa + (rt * 32 + lane) * 16);
            unpack4b(pk.x, af[m][0][0], af[m][0][2]);
            unpack4b(pk.y, af[m][1][0], af[m][1][2]);
            unpack4b(pk.z, af[m][0][1], af[m][0][3]);
            unpack4b(pk.w, af[m][1][1], af[m][1][3]);
        }
        // The two planes are summed with weights 256 and 1, so they distribute over the
        // accumulator: run the hi plane for every tile, then the lo plane, reusing one set of B
        // fragment registers. Holding both planes at once cost 16 registers and pushed the kernel
        // to 126, one block per SM. The arithmetic is unchanged and still integer-exact.
#pragma unroll
        for (int plane = 0; plane < 2; ++plane) {
            const char* const src   = (plane == 0) ? sh : sl;
            const float plane_scale = (plane == 0) ? 256.0f : 1.0f;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const uint4 b = lds128(src + ((warp_n * 4 + n) * 32 + lane) * 16);
                bf[n][0][0] = b.x; bf[n][0][1] = b.y; bf[n][1][0] = b.z; bf[n][1][1] = b.w;
            }
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int rt    = warp_m + m * 4;
                const float ws0 = __half2float(sws[rt * 16 + gid]) * plane_scale;
                const float ws1 = __half2float(sws[rt * 16 + gid + 8]) * plane_scale;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    int s[4] = {0, 0, 0, 0};
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks) {
                        mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                               af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                    }
                    const int c     = (warp_n * 4 + n) * 8 + tig * 2;
                    const float xa0 = __half2float(sas[c]);
                    const float xa1 = __half2float(sas[c + 1]);
                    acc[m][n][0] = fmaf((float)s[0], ws0 * xa0, acc[m][n][0]);
                    acc[m][n][1] = fmaf((float)s[1], ws0 * xa1, acc[m][n][1]);
                    acc[m][n][2] = fmaf((float)s[2], ws1 * xa0, acc[m][n][2]);
                    acc[m][n][3] = fmaf((float)s[3], ws1 * xa1, acc[m][n][3]);
                }
            }
        }
    }

#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int c0 = cb * BN + (warp_n * 4 + n) * 8 + tig * 2;
        const int r0 = rb * 64 + warp_m * 16 + gid;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row  = r0 + half * 8;
            const float g0 = acc[0][n][half * 2], u0 = acc[1][n][half * 2];
            const float g1 = acc[0][n][half * 2 + 1], u1 = acc[1][n][half * 2 + 1];
            out[(size_t)row * T + c0]     = __float2bfloat16(g0 / (1.0f + __expf(-g0)) * u0);
            out[(size_t)row * T + c0 + 1] = __float2bfloat16(g1 / (1.0f + __expf(-g1)) * u1);
        }
    }
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "gate_up_L0.bin";
    std::vector<unsigned char> raw(LOW_PLANE + SCALE_PLANE);
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { printf("cannot open %s\n", path); return 1; }
    const size_t got = std::fread(raw.data(), 1, raw.size(), f);
    std::fclose(f);
    if (got != raw.size()) { printf("short read\n"); return 1; }
    const unsigned char* low = raw.data();
    const __half* wsc        = reinterpret_cast<const __half*>(raw.data() + LOW_PLANE);
    printf("real weight: %s\n\n", path);

    std::vector<unsigned char> wp(LOW_PLANE);
    std::vector<__half> sp((size_t)N * GROUPS);
    for (int rb = 0; rb < NOUT / 64; ++rb)
        for (int g = 0; g < GROUPS; ++g)
            for (int rt = 0; rt < MTILES; ++rt) {
                const int base = ((rt >= 4) ? NOUT : 0) + rb * 64 + (rt % 4) * 16;
                for (int l = 0; l < 32; ++l) {
                    const int gid = l >> 2, tg = l & 3;
                    unsigned char* dst =
                        &wp[(((size_t)rb * GROUPS + g) * MTILES + rt) * 512 + (size_t)l * 16];
                    for (int half = 0; half < 2; ++half)
                        for (int ks = 0; ks < 2; ++ks)
                            for (int hi = 0; hi < 2; ++hi) {
                                const int k    = g * BK + ks * 32 + tg * 4 + hi * 16;
                                const int slot = half * 8 + ks * 4 + hi * 2;
                                const size_t s = (size_t)(base + gid + half * 8) * (K / 2) + k / 2;
                                dst[slot]      = low[s];
                                dst[slot + 1]  = low[s + 1];
                            }
                }
                for (int r = 0; r < 16; ++r)
                    sp[((size_t)rb * GROUPS + g) * BM + rt * 16 + r] =
                        wsc[(size_t)(base + r) * GROUPS + g];
            }

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<int> och;
    for (int i = 0; i < 12; ++i) och.push_back((int)(rng() % K));

    void *dw, *dsp, *dhi, *dlo, *dxs, *dx, *dout, *dflush;
    CHECK(cudaMalloc(&dw, LOW_PLANE));
    CHECK(cudaMalloc(&dsp, sp.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dhi, (size_t)T * K));
    CHECK(cudaMalloc(&dlo, (size_t)T * K));
    CHECK(cudaMalloc(&dxs, (size_t)(T / BN) * GROUPS * BN * sizeof(__half)));
    CHECK(cudaMalloc(&dx, (size_t)T * K * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dout, (size_t)NOUT * T * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dflush, 256u << 20));
    CHECK(cudaMemcpy(dw, wp.data(), LOW_PLANE, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dsp, sp.data(), sp.size() * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 grid(NOUT / 64, T / BN);
    const size_t smem = 2 * STAGE;
    printf("grid=(%d,%d) block=512 smem=%zu B\n\n", grid.x, grid.y, smem);
    printf("  %-20s %13s %13s\n", "outliers", "W4+2xINT8", "A16 route");
    printf("  %-20s %13s %13s\n", "--------", "rel L2", "rel L2");

    for (double amp : {1.0, 16.0, 64.0}) {
        std::vector<__nv_bfloat16> hxb((size_t)T * K);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                float v = nd(rng);
                for (int c : och)
                    if (c == k) v *= (float)amp;
                hxb[(size_t)t * K + k] = __float2bfloat16(v);
            }
        CHECK(cudaMemcpy(dx, hxb.data(), hxb.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
        quantize_x2<<<T, 128>>>((const __nv_bfloat16*)dx, (signed char*)dhi, (signed char*)dlo,
                                (__half*)dxs);
        CHECK(cudaDeviceSynchronize());
        w4a16_swiglu<<<grid, 512, smem>>>((const char*)dw, (const char*)dsp, (const char*)dhi,
                                          (const char*)dlo, (const char*)dxs,
                                          (__nv_bfloat16*)dout);
        CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> hout((size_t)NOUT * T);
        CHECK(cudaMemcpy(hout.data(), dout, hout.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost));

        double se = 0.0, sref = 0.0, se16 = 0.0;
        for (int s = 0; s < 48; ++s) {
            const int i = (int)(rng() % NOUT), t = (int)(rng() % T);
            double gs = 0.0, us = 0.0;
            float gs16 = 0.0f, us16 = 0.0f; // the A16 route: exact codes, bf16 x, fp32 accumulate
            for (int g = 0; g < GROUPS; ++g) {
                double gd = 0.0, ud = 0.0;
                float gf = 0.0f, uf = 0.0f;
                for (int j = 0; j < BK; ++j) {
                    const int k       = g * BK + j;
                    const unsigned bg = low[(size_t)i * (K / 2) + k / 2];
                    const unsigned bu = low[(size_t)(NOUT + i) * (K / 2) + k / 2];
                    const int cg = (((k % 2 == 0) ? (bg & 0xf) : (bg >> 4)) ^ 8) - 8;
                    const int cu = (((k % 2 == 0) ? (bu & 0xf) : (bu >> 4)) ^ 8) - 8;
                    const float xv = __bfloat162float(hxb[(size_t)t * K + k]);
                    gd += (double)cg * xv;
                    ud += (double)cu * xv;
                    gf += (float)cg * xv;
                    uf += (float)cu * xv;
                }
                const double sg = __half2float(wsc[(size_t)i * GROUPS + g]);
                const double su = __half2float(wsc[(size_t)(NOUT + i) * GROUPS + g]);
                gs += gd * sg;
                us += ud * su;
                gs16 += gf * (float)sg;
                us16 += uf * (float)su;
            }
            const double ref  = gs / (1.0 + std::exp(-gs)) * us;
            const double got  = (double)__bfloat162float(hout[(size_t)i * T + t]);
            const double a16f = (double)__bfloat162float(
                __float2bfloat16(gs16 / (1.0f + std::exp(-gs16)) * us16));
            se += (got - ref) * (got - ref);
            se16 += (a16f - ref) * (a16f - ref);
            sref += ref * ref;
        }
        char lab[32];
        std::snprintf(lab, sizeof(lab), (amp == 1.0) ? "none" : "12 chans x %d", (int)amp);
        printf("  %-20s %13.3e %13.3e\n", lab, std::sqrt(se / std::max(sref, 1e-30)),
               std::sqrt(se16 / std::max(sref, 1e-30)));
    }

    const int reps = 12;
    std::vector<float> ms(reps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaMemsetAsync(dflush, i & 0xff, 256u << 20));
        CHECK(cudaEventRecord(a));
        w4a16_swiglu<<<grid, 512, smem>>>((const char*)dw, (const char*)dsp, (const char*)dhi,
                                          (const char*)dlo, (const char*)dxs,
                                          (__nv_bfloat16*)dout);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double us  = ms[reps / 2] * 1000.0;
    const double ops = 2.0 * N * K * T;
    printf("\n  %-40s %9.1f us  %6.2f TOP/s\n", "W4 + 2x INT8 planes (no accuracy cost)", us,
           ops / (us * 1e-6) / 1e12);
    printf("  %-40s %9.1f us  %6.2f TOP/s\n", "W4A8 single plane (adds ~0.9% error)", 1565.7,
           ops / (1565.7e-6) / 1e12);
    printf("  %-40s %9.1f us  %6.2f TFLOP/s\n", "A16 route (measured)", 3357.7,
           ops / (3357.7e-6) / 1e12);
    printf("  %-40s %9.2fx\n", "speedup vs A16", 3357.7 / us);
    return 0;
}
