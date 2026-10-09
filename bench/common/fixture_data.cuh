#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::bench::fixture {

// Stateless, reproducible samples. Mixing the seed separately prevents two operands from
// becoming shifted views of the same sequence. The full 64-bit element index participates.
__host__ __device__ inline std::uint64_t mix(std::uint64_t value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__host__ __device__ inline std::uint32_t bits(std::uint64_t index, std::uint64_t seed) {
    return static_cast<std::uint32_t>(mix(index ^ mix(seed + 0x9e3779b97f4a7c15ULL)) >> 32);
}

__host__ __device__ inline float uniform(std::uint64_t index, std::uint64_t seed, float low,
                                         float high) {
    return low + (high - low) * (static_cast<float>(bits(index, seed) >> 8) / 16777216.0F);
}

inline int grid(std::size_t count) {
    return static_cast<int>(std::min<std::size_t>(4096, (count + 255) / 256));
}

template <class T>
static __global__ void fill_values_kernel(T* values, std::size_t count, std::uint64_t seed,
                                          float low, float high) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        values[i] = T(uniform(i, seed, low, high));
    }
}

template <class T>
inline cudaError_t fill_values(T* values, std::size_t count, std::uint64_t seed, float low,
                               float high, cudaStream_t stream = nullptr) {
    if (count == 0) return cudaSuccess;
    fill_values_kernel<<<grid(count), 256, 0, stream>>>(values, count, seed, low, high);
    return cudaGetLastError();
}

static __global__ void fill_bytes_kernel(std::uint32_t* words, std::size_t bytes,
                                         std::uint64_t seed) {
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < (bytes + 3) / 4; i += stride) {
        const auto value = bits(i, seed);
        if (i < bytes / 4) {
            words[i] = value;
        } else {
            auto* tail = reinterpret_cast<std::uint8_t*>(words) + i * 4;
            for (std::size_t j = 0; j < bytes % 4; ++j) tail[j] = value >> (j * 8);
        }
    }
}

inline cudaError_t fill_bytes(void* data, std::size_t bytes, std::uint64_t seed,
                              cudaStream_t stream = nullptr) {
    if (bytes == 0) return cudaSuccess;
    fill_bytes_kernel<<<grid((bytes + 3) / 4), 256, 0, stream>>>(static_cast<std::uint32_t*>(data),
                                                                 bytes, seed);
    return cudaGetLastError();
}

} // namespace ninfer::bench::fixture
