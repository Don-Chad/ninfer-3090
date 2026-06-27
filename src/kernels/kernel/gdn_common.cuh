#pragma once

// Shared GDN kernel helpers localized from ~/chunked_gdn. This header is
// host-safe: host-only tests can include it for head_map, while CUDA-only
// intrinsic helpers are compiled only in nvcc translation units.

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#    define QUS_KERNELS_HOST_DEVICE __host__ __device__
#else
#    define QUS_KERNELS_HOST_DEVICE
#endif

namespace qus::kernels {

inline uint3 init_fastdiv_values(std::uint64_t d64) {
    if (d64 == 0 || d64 > static_cast<std::uint64_t>(0xffffffffu)) {
        std::fprintf(stderr, "qus::kernels::init_fastdiv_values: invalid divisor %llu\n",
                     static_cast<unsigned long long>(d64));
        std::abort();
    }

    const auto d    = static_cast<std::uint32_t>(d64);
    std::uint32_t L = 0;
    while (L < 32 && (static_cast<std::uint32_t>(1) << L) < d) { ++L; }
    const auto mp = static_cast<std::uint32_t>(
        ((static_cast<std::uint64_t>(1) << 32) * ((static_cast<std::uint64_t>(1) << L) - d)) / d +
        1);
    return make_uint3(mp, L, d);
}

inline bool is_supported_gdn_head_dim(std::int64_t S) {
    return S == 16 || S == 32 || S == 64 || S == 128;
}

inline bool are_gdn_head_counts_valid(std::int64_t H_qk, std::int64_t H_v) {
    return H_qk > 0 && H_v >= H_qk && (H_v % H_qk) == 0;
}

#ifdef __CUDACC__

inline constexpr int WARP_SIZE = 32;

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#    pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void cuda_memcpy_1(void* __restrict__ dst,
                                                     const void* __restrict__ src) {
    static_assert(nbytes % (alignment == 0 ? 1 : alignment) == 0, "bad alignment");
    constexpr int nb_per_cpy = (alignment == 0) ? nbytes : alignment;

#    pragma unroll
    for (int i = 0; i < nbytes / nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            static_cast<char*>(dst)[i] = static_cast<const char*>(src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            static_cast<short*>(dst)[i] = static_cast<const short*>(src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            static_cast<int*>(dst)[i] = static_cast<const int*>(src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            static_cast<int2*>(dst)[i] = static_cast<const int2*>(src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            static_cast<int4*>(dst)[i] = static_cast<const int4*>(src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ std::uint32_t fastdiv(std::uint32_t n, uint3 fastdiv_values) {
    const std::uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ std::uint32_t fastdivide(std::uint32_t n, uint3 fastdiv_values) {
    return fastdiv(n, fastdiv_values);
}

template <int BLOCK_SIZE>
static __device__ __forceinline__ float block_reduce_sum(float x) {
    static_assert(BLOCK_SIZE >= WARP_SIZE && BLOCK_SIZE <= 1024,
                  "block_reduce_sum: BLOCK_SIZE must be in [WARP_SIZE, 1024]");
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "block_reduce_sum: BLOCK_SIZE must be a power of two");

    constexpr int N_WARPS = BLOCK_SIZE / WARP_SIZE;
    x                     = warp_reduce_sum<WARP_SIZE>(x);
    if constexpr (N_WARPS == 1) { return x; }

    __shared__ float warp_sums[N_WARPS];
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) { warp_sums[warp] = x; }
    __syncthreads();

    float v = (threadIdx.x < N_WARPS) ? warp_sums[threadIdx.x] : 0.0f;
    if (warp == 0) { v = warp_reduce_sum<N_WARPS>(v); }
    if (warp == 0 && lane == 0) { warp_sums[0] = v; }
    __syncthreads();
    return warp_sums[0];
}

static __device__ __forceinline__ float4 vec4_load(const float* src) {
    assert((reinterpret_cast<std::uintptr_t>(src) & 0xF) == 0 && "vec4_load: src not 16B aligned");
    return *reinterpret_cast<const float4*>(src);
}

static __device__ __forceinline__ float4 vec4_load_ldg(const float* src) {
    assert((reinterpret_cast<std::uintptr_t>(src) & 0xF) == 0 &&
           "vec4_load_ldg: src not 16B aligned");
    return __ldg(reinterpret_cast<const float4*>(src));
}

static __device__ __forceinline__ void vec4_store(float* dst, float4 v) {
    assert((reinterpret_cast<std::uintptr_t>(dst) & 0xF) == 0 && "vec4_store: dst not 16B aligned");
    *reinterpret_cast<float4*>(dst) = v;
}

template <int N_BYTES>
static __device__ __forceinline__ void async_copy_global_to_shared(void* smem_dst,
                                                                   const void* gmem_src) {
    static_assert(N_BYTES == 4 || N_BYTES == 8 || N_BYTES == 16,
                  "async_copy_global_to_shared: N_BYTES must be 4, 8, or 16");
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(
                     static_cast<unsigned>(__cvta_generic_to_shared(smem_dst))),
                 "l"(gmem_src), "n"(N_BYTES));
#    else
    if constexpr (N_BYTES == 16) {
        *reinterpret_cast<float4*>(smem_dst) = *reinterpret_cast<const float4*>(gmem_src);
    } else if constexpr (N_BYTES == 8) {
        *reinterpret_cast<float2*>(smem_dst) = *reinterpret_cast<const float2*>(gmem_src);
    } else {
        *reinterpret_cast<float*>(smem_dst) = *reinterpret_cast<const float*>(gmem_src);
    }
#    endif
}

template <int N_BYTES>
static __device__ __forceinline__ void
async_copy_global_to_shared_pred(void* smem_dst, const void* gmem_src, int src_bytes) {
    static_assert(N_BYTES == 16, "async_copy_global_to_shared_pred: only N_BYTES=16 supported");
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(
                     static_cast<unsigned>(__cvta_generic_to_shared(smem_dst))),
                 "l"(gmem_src), "n"(N_BYTES), "r"(src_bytes));
#    else
    if (src_bytes == 16) {
        *reinterpret_cast<float4*>(smem_dst) = *reinterpret_cast<const float4*>(gmem_src);
    } else {
        *reinterpret_cast<float4*>(smem_dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
#    endif
}

static __device__ __forceinline__ void async_copy_commit() {
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#    endif
}

template <int N_GROUPS_REMAINING>
static __device__ __forceinline__ void async_copy_wait() {
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N_GROUPS_REMAINING));
#    endif
}

static __device__ __forceinline__ void async_copy_wait_all() {
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n");
#    endif
}

static __device__ __forceinline__ unsigned f32_to_tf32(float v) { return __float_as_uint(v); }

static __device__ __forceinline__ void mma_m16n8k8_tf32(float& d0, float& d1, float& d2, float& d3,
                                                        float a0, float a1, float a2, float a3,
                                                        float b0, float b1) {
    const unsigned ua0 = f32_to_tf32(a0);
    const unsigned ua1 = f32_to_tf32(a1);
    const unsigned ua2 = f32_to_tf32(a2);
    const unsigned ua3 = f32_to_tf32(a3);
    const unsigned ub0 = f32_to_tf32(b0);
    const unsigned ub1 = f32_to_tf32(b1);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(ua0), "r"(ua1), "r"(ua2), "r"(ua3), "r"(ub0), "r"(ub1));
}

static __device__ __forceinline__ void mma_m16n8k8_tf32_b32(float& d0, float& d1, float& d2,
                                                            float& d3, unsigned a0, unsigned a1,
                                                            unsigned a2, unsigned a3, unsigned b0,
                                                            unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ unsigned smem_addr_b32(const void* smem_ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
}

static __device__ __forceinline__ void ldmatrix_x4_b16(unsigned& a0, unsigned& a1, unsigned& a2,
                                                       unsigned& a3, unsigned smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %2, %1, %3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(smem_addr));
}

static __device__ __forceinline__ void ldmatrix_x2_b16(unsigned& b0, unsigned& b1,
                                                       unsigned smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(smem_addr));
}

template <int STRIDE>
struct SmemTile {
    float* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE >= 32,
                  "SmemTile: only STRIDE in {16, 32, 64, 128, ...} supported");

    __device__ __forceinline__ int swz_xor(int row) const {
        if constexpr (STRIDE >= 32) {
            return ((row & 3) << 3) | (row & 4);
        } else {
            return ((row >> 1) & 3) << 2;
        }
    }

    __device__ __forceinline__ float& at(int row, int col) const {
        return base[row * STRIDE + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const {
        return *reinterpret_cast<float4*>(&base[row * STRIDE + (col ^ swz_xor(row))]);
    }
};

template <int ROWS, int STRIDE, int THREADS, class View>
static __device__ __forceinline__ void
issue_async_load_vec4(View view, const float* __restrict__ gmem_base_row0,
                      std::int64_t gmem_row_stride_floats, int cl, int tid) {
    static_assert(STRIDE % 4 == 0, "issue_async_load_vec4: STRIDE must be a multiple of 4");
    constexpr int VEC_PER_ROW = STRIDE / 4;
    constexpr int N_VEC       = ROWS * VEC_PER_ROW;
#    pragma unroll
    for (int v = tid; v < N_VEC; v += THREADS) {
        const int row       = v / VEC_PER_ROW;
        const int col4      = v - row * VEC_PER_ROW;
        float* smem_ptr     = reinterpret_cast<float*>(&view.vec4_at(row, col4 * 4));
        const bool in_range = row < cl;
        if (in_range) {
            const float* gmem_ptr =
                gmem_base_row0 + static_cast<std::int64_t>(row) * gmem_row_stride_floats + col4 * 4;
            async_copy_global_to_shared<16>(smem_ptr, gmem_ptr);
        } else {
            *reinterpret_cast<float4*>(smem_ptr) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

struct mma_lane_t {
    int lane;
    int warp;
    int lane_g;
    int lane_t;

    static __device__ __forceinline__ mma_lane_t decode(int tid) {
        mma_lane_t L{};
        L.lane   = tid & (WARP_SIZE - 1);
        L.warp   = tid >> 5;
        L.lane_g = L.lane >> 2;
        L.lane_t = L.lane & 3;
        return L;
    }
};

struct chunk_bounds_t {
    std::int64_t cs;
    std::int64_t ce;
    int cl;

    static __device__ __forceinline__ chunk_bounds_t of(int chunk_idx, std::int64_t T, int BT) {
        chunk_bounds_t b{};
        b.cs                    = static_cast<std::int64_t>(chunk_idx) * BT;
        const std::int64_t ce64 = b.cs + BT;
        b.ce                    = (ce64 < T) ? ce64 : T;
        b.cl                    = static_cast<int>(b.ce - b.cs);
        return b;
    }
};

inline constexpr float RCP_LN2_F = 1.4426950408889634f;

static __device__ __forceinline__ float exp2_fast(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

#endif // __CUDACC__

struct head_map {
    int H_qk;
    int H_v;
    uint3 group_magic;

    static head_map of(int H_qk_, int H_v_) {
        const int G = H_v_ / H_qk_;
        return head_map{H_qk_, H_v_, init_fastdiv_values(static_cast<std::uint64_t>(G))};
    }

    QUS_KERNELS_HOST_DEVICE int group_size() const { return H_v / H_qk; }

    QUS_KERNELS_HOST_DEVICE int qk_head(int h_v) const {
#if defined(__CUDA_ARCH__)
        return static_cast<int>(fastdivide(static_cast<std::uint32_t>(h_v), group_magic));
#else
        return h_v / group_size();
#endif
    }

    QUS_KERNELS_HOST_DEVICE int cta_h_v(int cta_h) const { return cta_h; }
};

} // namespace qus::kernels

#undef QUS_KERNELS_HOST_DEVICE
