// Lifetime and integrity qualification for the VMM-backed evictable weight pool:
// stable arena addresses across transactions, byte-exact restore of a partially
// evicted tail from the pinned mirror, and rejection of invalid transactions.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        if (!ninfer::EvictableWeightPool::supported(device.device)) {
            std::cout << "SKIP: device does not support CUDA virtual memory management\n";
            return 77;
        }

        constexpr std::size_t kChunk = ninfer::EvictableWeightPool::kChunkBytes;
        const std::size_t arena_bytes = 6 * kChunk - (kChunk / 2);   // deliberately unaligned
        const std::size_t tail_bytes  = 3 * kChunk;

        ninfer::EvictableWeightPool pool(ninfer::EvictableWeightPool::Config{
            .arena_bytes          = arena_bytes,
            .evictable_tail_bytes = tail_bytes,
            .overlay_bytes        = 2 * kChunk,
            .device               = device.device,
        });

        int failures = 0;
        const ninfer::DeviceSpan arena = pool.arena();
        failures += expect(arena.bytes == arena_bytes, "arena span covers the configured bytes");

        // Deterministic "weights".
        std::vector<std::uint32_t> pattern(arena_bytes / sizeof(std::uint32_t));
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            pattern[i] = static_cast<std::uint32_t>(i * 2654435761ULL + 12345U);
        }
        CUDA_CHECK(cudaMemcpyAsync(arena.data, pattern.data(),
                                   pattern.size() * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, device.stream));
        pool.capture_tail_mirror(device.stream);

        const void* stable_base = arena.data;
        std::vector<std::uint32_t> readback(pattern.size());

        for (int cycle = 0; cycle < 2; ++cycle) {
            std::size_t mapped   = 0;
            std::byte* staging   = pool.evict(kChunk + kChunk / 2, &mapped);
            failures += expect(mapped == 2 * kChunk, "evict rounds the extent up to chunks");
            failures += expect(pool.evicted(), "transaction reports evicted state");
            CUDA_CHECK(cudaMemsetAsync(staging, 0xC3, mapped, device.stream));
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            pool.restore(device.stream);
            failures += expect(!pool.evicted(), "restore closes the transaction");
            failures += expect(pool.arena().data == stable_base,
                               "arena base is stable across transactions");

            CUDA_CHECK(cudaMemcpyAsync(readback.data(), arena.data,
                                       readback.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, device.stream));
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            failures += expect(std::memcmp(readback.data(), pattern.data(),
                                           pattern.size() * sizeof(std::uint32_t)) == 0,
                               "restored arena is byte-identical to the mirror image");
        }

        pool.restore(device.stream);   // idempotent when nothing is evicted

        bool rejected = false;
        try {
            std::size_t mapped = 0;
            (void)pool.evict(64 * kChunk, &mapped);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += expect(rejected, "evict beyond the tail capacity is rejected");
        failures += expect(!pool.evicted(), "rejected evict leaves the pool resident");

        std::cout << "evict_s=" << pool.last_evict_seconds()
                  << " restore_s=" << pool.last_restore_seconds() << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "evictable weight pool test failed: " << error.what() << '\n';
        return 1;
    }
}
