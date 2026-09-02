// W4A8 linear_swiglu against a REAL Qwen3.8-27B weight.
//
// Reads text/layers/0/mlp/gate_up straight out of the artifact's payload -- Q4G64_F16S in the
// row-split-k128-v1 layout, [34816,5120], 94,699,520 bytes: an 89,128,960-byte code plane of
// 32 bytes per 64-code group followed by a 5,570,560-byte FP16 scale plane, both row-major. Codes
// are two's complement, decoded (n^8)-8, which the real weight confirms: the histogram is a
// symmetric bell about zero and -8 never appears, so the encoder uses [-7,+7].
//
// Two things this measures that synthetic weights cannot:
//
//   1. The fused SwiGLU falls out of the repack for free. linear_swiglu wants
//      out[i,t] = SiLU(gate_up[i,t]) * gate_up[17408+i,t], and gate row i sits 136 blocks away
//      from its up row. Packing each block as 64 gate rows plus their 64 matching up rows, ordered
//      so one warp holds tile gt as m=0 and tile gt+4 as m=1, puts a row and its partner in the
//      same thread's registers. The epilogue is then SiLU(acc[0]) * acc[1] with no shared
//      exchange, and the permutation absorbs the whole thing at zero kernel cost.
//
//   2. Whether int8 activation quantisation survives real transformer activations. This is the
//      open quality risk for the whole W4A8 route, and it is a property of the activation
//      distribution rather than of the weights. Hidden states carry a handful of fixed outlier
//      channels an order of magnitude above the rest, so the probe sweeps outlier strength.
//
// RESULT: the speed case holds on real weights and the quality case does not.
//
//   activation scale     outliers        rel L2      speed
//   ------------------------------------------------------
//   per token            none          1.228e-02      2.58x
//   per token            12 x 4        4.113e-02
//   per token            12 x 16       7.505e-02
//   per token            12 x 64       1.288e-01
//   per group of 64      none          9.122e-03      2.14x
//   per group of 64      12 x 4        8.158e-03
//   per group of 64      12 x 16       1.776e-02
//   per group of 64      12 x 64       2.016e-02
//
// Per-group scales do what they were meant to: an outlier can only spoil its own group of 64, so
// error stops tracking outlier strength and flattens to 1-2% -- a 5-6x improvement on the worst
// case. They cost 20% of the speed, 2.58x -> 2.14x, for the extra multiply per tile per group and
// a fourth async plane.
//
// The problem is the floor. Even with no outliers at all, per-group int8 leaves ~0.9% relative L2,
// and that is arithmetic rather than a defect: an absmax over 64 Gaussian samples is about 2.7
// sigma, so the step is 2.7/127 = 0.021 sigma, quantisation error is step/sqrt(12) = 0.006 sigma,
// and over a K=5120 dot product that stays ~0.6% before SwiGLU roughly doubles it by multiplying
// two noisy terms. Finer groups only crawl: group-16 would predict ~0.7% for twice the scale
// traffic.
//
// For scale, the rk8v4 KV work spends 3e-3 relative L2 in attention for +0.082% perplexity. This
// puts 9e-3 on the MLP output of every one of 64 layers, and unlike weight quantisation -- which
// the checkpoint was trained through -- it is error the model has never seen. The A16 route it
// would replace introduces no activation error at all.
//
// So Gate C is amber, not green, and it cannot be settled from here: it needs a perplexity run,
// which needs the Op integration. What this probe does settle is that no further kernel work will
// fix it, because the floor is set by the number of levels int8 has, not by the schedule.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
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

constexpr int N       = 34816; // gate rows then up rows
constexpr int NOUT    = N / 2; // 17408
constexpr int K       = 5120;
constexpr int T       = 512;
constexpr int BM      = 128; // 64 gate rows + their 64 up rows
constexpr int BN      = 128;
constexpr int BK      = 64;
constexpr int GROUPS  = K / BK;
constexpr int MTILES  = BM / 16;
constexpr int NTILES  = BN / 8;
constexpr int WSTAGE  = MTILES * 32 * 16;
constexpr int XSTAGE  = NTILES * 32 * 16;
constexpr int SSTAGE  = BM * 2;   // FP16 weight group scale, one per row
constexpr int ASTAGE  = BN * 2;   // FP16 activation group scale, one per token
constexpr int STAGE   = WSTAGE + XSTAGE + SSTAGE + ASTAGE;
constexpr size_t LOW_PLANE   = (size_t)N * GROUPS * 32;
constexpr size_t SCALE_PLANE = (size_t)N * GROUPS * 2;

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
// Two's-complement Q4: flip bit 3 of every nibble, mask, subtract 8.
__device__ __forceinline__ void unpack4b(unsigned packed, unsigned& w0, unsigned& w1) {
    packed ^= 0x88888888u;
    const unsigned mask = 0x0f0f0f0fu;
    const unsigned even = __vsub4(packed & mask, 0x08080808u);
    const unsigned odd  = __vsub4((packed >> 4) & mask, 0x08080808u);
    w0                  = __byte_perm(even, odd, 0x5140);
    w1                  = __byte_perm(even, odd, 0x7362);
}

// ---- activation quantisation: per-token absmax, then straight into fragment order -------------
// One scale per (token, group of 64) rather than one per token. A per-token absmax over all 5120
// channels is set by whichever channel is largest, and every other channel then lands in a handful
// of the 255 levels; with a group of 64 an outlier only spoils its own group. It is also free where
// it is used: the kernel already rescales once per group, so the activation scale folds into the
// weight scale there instead of being applied in the epilogue.
__global__ void quantize_x(const __nv_bfloat16* __restrict__ x, signed char* __restrict__ xq,
                           __half* __restrict__ xs) {
    const int t  = blockIdx.x; // one block per token
    const int cb = t / BN, cl = t % BN;
    const int nt = cl / 8, gid = cl % 8;
    for (int g = threadIdx.x; g < GROUPS; g += blockDim.x) {
        float amax = 0.0f;
        for (int j = 0; j < BK; ++j) {
            amax = fmaxf(amax, fabsf(__bfloat162float(x[(size_t)t * K + g * BK + j])));
        }
        amax            = fmaxf(amax, 1e-12f);
        const float inv = 127.0f / amax;
        xs[((size_t)cb * GROUPS + g) * BN + cl] = __float2half(amax / 127.0f);
        for (int j = 0; j < BK; ++j) {
            const float v  = __bfloat162float(x[(size_t)t * K + g * BK + j]) * inv;
            const int code = max(-127, min(127, __float2int_rn(v)));
            const int ks = j / 32, rr = j % 32, hi = rr / 16, tg = (rr % 16) / 4, jj = j % 4;
            const int lane = gid * 4 + tg;
            xq[(((size_t)cb * GROUPS + g) * NTILES + nt) * 512 + (size_t)lane * 16 + ks * 8 +
               hi * 4 + jj] = (signed char)code;
        }
    }
}

__global__ __launch_bounds__(512) void w4a8_swiglu(const char* __restrict__ w, const char* __restrict__ ws,
                                                    const char* __restrict__ xq,
                                                    const char* __restrict__ xs,
                                                    __nv_bfloat16* __restrict__ out) {
    extern __shared__ char smem[];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3, warp_m = warp >> 2, warp_n = warp & 3;
    const int rb = blockIdx.x, cb = blockIdx.y;

    const char* const wb  = w + (size_t)rb * GROUPS * WSTAGE;
    const char* const xb  = xq + (size_t)cb * GROUPS * XSTAGE;
    const char* const sb_ = ws + (size_t)rb * GROUPS * SSTAGE;
    const char* const ab_ = xs + (size_t)cb * GROUPS * ASTAGE;

    auto issue = [&](int g, int buf) {
        char* const d = smem + buf * STAGE;
        if (tid < WSTAGE / 16) { cp_async16(d + tid * 16, wb + (size_t)g * WSTAGE + tid * 16); }
        cp_async16(d + WSTAGE + tid * 16, xb + (size_t)g * XSTAGE + tid * 16);
        if (tid < SSTAGE / 16) {
            cp_async16(d + WSTAGE + XSTAGE + tid * 16, sb_ + (size_t)g * SSTAGE + tid * 16);
        }
        if (tid < ASTAGE / 16) {
            cp_async16(d + WSTAGE + XSTAGE + SSTAGE + tid * 16,
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
        const char* const sx    = sa + WSTAGE;
        const __half* const sws = reinterpret_cast<const __half*>(sx + XSTAGE);
        const __half* const sas = reinterpret_cast<const __half*>(sx + XSTAGE + SSTAGE);

        unsigned af[2][2][4];
        unsigned bf[4][2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int rt   = warp_m + m * 4; // m=0 gate tile, m=1 its up partner
            const uint4 pk = lds128(sa + (rt * 32 + lane) * 16);
            unpack4b(pk.x, af[m][0][0], af[m][0][2]);
            unpack4b(pk.y, af[m][1][0], af[m][1][2]);
            unpack4b(pk.z, af[m][0][1], af[m][0][3]);
            unpack4b(pk.w, af[m][1][1], af[m][1][3]);
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const uint4 b = lds128(sx + ((warp_n * 4 + n) * 32 + lane) * 16);
            bf[n][0][0] = b.x; bf[n][0][1] = b.y; bf[n][1][0] = b.z; bf[n][1][1] = b.w;
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int rt    = warp_m + m * 4;
            const float ws0 = __half2float(sws[rt * 16 + gid]);
            const float ws1 = __half2float(sws[rt * 16 + gid + 8]);
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

    // Fused SwiGLU: acc[0] is the gate row, acc[1] its up partner, same thread, same registers.
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
    if (f == nullptr) {
        printf("cannot open %s\n", path);
        return 1;
    }
    const size_t got = std::fread(raw.data(), 1, raw.size(), f);
    std::fclose(f);
    if (got != raw.size()) {
        printf("short read: %zu of %zu\n", got, raw.size());
        return 1;
    }
    const unsigned char* low = raw.data();
    const __half* wsc        = reinterpret_cast<const __half*>(raw.data() + LOW_PLANE);
    printf("loaded real weight: %s (%zu bytes)\n\n", path, raw.size());

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));

    // ---- repack: gate tile gt as m=0, its up partner as m=1, in fragment order ----
    std::vector<unsigned char> wp(LOW_PLANE);
    std::vector<__half> sp((size_t)N * GROUPS);
    for (int rb = 0; rb < NOUT / 64; ++rb)
        for (int g = 0; g < GROUPS; ++g) {
            for (int rt = 0; rt < MTILES; ++rt) {
                const bool up  = rt >= 4;
                const int gt   = rt % 4;
                const int base = (up ? NOUT : 0) + rb * 64 + gt * 16;
                for (int l = 0; l < 32; ++l) {
                    const int gid = l >> 2, tg = l & 3;
                    unsigned char* dst =
                        &wp[(((size_t)rb * GROUPS + g) * MTILES + rt) * 512 + (size_t)l * 16];
                    for (int half = 0; half < 2; ++half) {
                        const int row = base + gid + half * 8;
                        for (int ks = 0; ks < 2; ++ks)
                            for (int hi = 0; hi < 2; ++hi) {
                                const int k    = g * BK + ks * 32 + tg * 4 + hi * 16;
                                const int slot = half * 8 + ks * 4 + hi * 2;
                                const size_t s = (size_t)row * (K / 2) + k / 2;
                                dst[slot]      = low[s];
                                dst[slot + 1]  = low[s + 1];
                            }
                    }
                }
                for (int r = 0; r < 16; ++r) {
                    sp[((size_t)rb * GROUPS + g) * BM + rt * 16 + r] =
                        wsc[(size_t)(base + r) * GROUPS + g];
                }
            }
        }

    // ---- activations: Gaussian hidden states plus fixed outlier channels ----
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<int> outlier_ch;
    for (int i = 0; i < 12; ++i) { outlier_ch.push_back((int)(rng() % K)); }

    void *dw, *dsp, *dxq, *dxs, *dx, *dout, *dflush;
    CHECK(cudaMalloc(&dw, LOW_PLANE));
    CHECK(cudaMalloc(&dsp, sp.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dxq, (size_t)T * K));
    CHECK(cudaMalloc(&dxs, (size_t)(T / BN) * GROUPS * BN * sizeof(__half)));
    CHECK(cudaMalloc(&dx, (size_t)T * K * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dout, (size_t)NOUT * T * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dflush, 256u << 20));
    CHECK(cudaMemcpy(dw, wp.data(), LOW_PLANE, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dsp, sp.data(), sp.size() * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 grid(NOUT / 64, T / BN);
    const size_t smem = 2 * STAGE;
    printf("grid=(%d,%d) block=512 smem=%zu B   real Q4 weight, fused SwiGLU\n\n", grid.x, grid.y,
           smem);
    printf("  %-22s %11s %11s %11s\n", "outlier channels", "max |err|", "rel L2", "worst rel");
    printf("  %-22s %11s %11s %11s\n", "----------------", "---------", "------", "---------");

    for (double amp : {1.0, 4.0, 16.0, 64.0}) {
        std::vector<float> hx((size_t)T * K);
        std::vector<__nv_bfloat16> hxb((size_t)T * K);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                float v = nd(rng);
                for (int c : outlier_ch)
                    if (c == k) { v *= (float)amp; }
                hx[(size_t)t * K + k]  = v;
                hxb[(size_t)t * K + k] = __float2bfloat16(v);
            }
        CHECK(cudaMemcpy(dx, hxb.data(), hxb.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice));
        quantize_x<<<T, 128>>>((const __nv_bfloat16*)dx, (signed char*)dxq, (__half*)dxs);
        CHECK(cudaDeviceSynchronize());
        w4a8_swiglu<<<grid, 512, smem>>>((const char*)dw, (const char*)dsp, (const char*)dxq,
                                        (const char*)dxs, (__nv_bfloat16*)dout);
        CHECK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> hout((size_t)NOUT * T);
        CHECK(cudaMemcpy(hout.data(), dout, hout.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost));

        double maxabs = 0.0, se = 0.0, sref = 0.0, worstrel = 0.0;
        for (int s = 0; s < 48; ++s) {
            const int i = (int)(rng() % NOUT), t = (int)(rng() % T);
            double gsum = 0.0, usum = 0.0;
            for (int g = 0; g < GROUPS; ++g) {
                double gd = 0.0, ud = 0.0;
                for (int j = 0; j < BK; ++j) {
                    const int k = g * BK + j;
                    const unsigned bg = low[(size_t)i * (K / 2) + k / 2];
                    const unsigned bu = low[(size_t)(NOUT + i) * (K / 2) + k / 2];
                    const int cg = (((k % 2 == 0) ? (bg & 0xf) : (bg >> 4)) ^ 8) - 8;
                    const int cu = (((k % 2 == 0) ? (bu & 0xf) : (bu >> 4)) ^ 8) - 8;
                    const double xv = __bfloat162float(hxb[(size_t)t * K + k]);
                    gd += cg * xv;
                    ud += cu * xv;
                }
                gsum += gd * (double)__half2float(wsc[(size_t)i * GROUPS + g]);
                usum += ud * (double)__half2float(wsc[(size_t)(NOUT + i) * GROUPS + g]);
            }
            const double ref = gsum / (1.0 + std::exp(-gsum)) * usum;
            const double got = (double)__bfloat162float(hout[(size_t)i * T + t]);
            const double e   = std::abs(got - ref);
            maxabs           = std::max(maxabs, e);
            se += e * e;
            sref += ref * ref;
            worstrel = std::max(worstrel, e / std::max(std::abs(ref), 1e-9));
        }
        printf("  %-22s %11.3e %11.3e %11.3e\n",
               (amp == 1.0) ? "none (plain N(0,1))"
                            : (std::string("12 chans x ") + std::to_string((int)amp)).c_str(),
               maxabs, std::sqrt(se / std::max(sref, 1e-30)), worstrel);
    }

    // ---- timing ----
    const int reps = 12;
    std::vector<float> ms(reps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaMemsetAsync(dflush, i & 0xff, 256u << 20));
        CHECK(cudaEventRecord(a));
        w4a8_swiglu<<<grid, 512, smem>>>((const char*)dw, (const char*)dsp, (const char*)dxq,
                                        (const char*)dxs, (__nv_bfloat16*)dout);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double us  = ms[reps / 2] * 1000.0;
    const double ops = 2.0 * N * K * T;
    printf("\n  %-38s %9.1f us  %6.2f TOP/s\n", "W4A8 linear_swiglu, real weight", us,
           ops / (us * 1e-6) / 1e12);
    printf("  %-38s %9.1f us  %6.2f TFLOP/s\n", "A16 linear GEMM at this shape (measured)", 3357.7,
           ops / (3357.7e-6) / 1e12);
    printf("  %-38s %9.2fx\n", "speedup", 3357.7 / us);
    return 0;
}
