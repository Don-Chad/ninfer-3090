// Lifetime and integrity qualification for the VMM-backed evictable weight pool: stable arena
// addresses across transactions, byte-exact restore of the borrowed chunks from the pinned mirror,
// restore confined to the dirtied extent, and rejection of invalid transactions.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
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

std::byte* at(const ninfer::DeviceSpan& arena, std::size_t offset) {
    return static_cast<std::byte*>(arena.data) + offset;
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
        if (!ninfer::EvictableWeightPool::supported(device)) {
            std::cout << "SKIP: device does not support CUDA virtual memory management\n";
            return 77;
        }

        constexpr std::size_t kChunk  = ninfer::EvictableWeightPool::kChunkBytes;
        const std::size_t arena_bytes = 6 * kChunk - (kChunk / 2); // deliberately unaligned
        const std::size_t tail_bytes  = 3 * kChunk;                // chunks at 3c, 4c, 5c

        ninfer::EvictableWeightPool pool(device, ninfer::EvictableWeightPool::Config{
                                                     .arena_bytes           = arena_bytes,
                                                     .evictable_tail_bytes  = tail_bytes,
                                                     .window_capacity_bytes = 2 * kChunk,
                                                 });

        int failures                   = 0;
        const ninfer::DeviceSpan arena = pool.arena();
        failures += expect(arena.bytes == arena_bytes, "arena span covers the configured bytes");
        failures += expect(pool.window_capacity_bytes() == 2 * kChunk,
                           "window capacity rounds to whole chunks");
        failures += expect(pool.mirror_bytes() == arena_bytes - 4 * kChunk,
                           "mirror covers exactly the window's weight bytes");

        std::vector<std::uint32_t> pattern(arena_bytes / sizeof(std::uint32_t));
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            pattern[i] = static_cast<std::uint32_t>(i * 2654435761ULL + 12345U);
        }
        CUDA_CHECK(cudaMemcpyAsync(arena.data, pattern.data(),
                                   pattern.size() * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, device.stream));
        bool early_rejected = false;
        try {
            (void)pool.evict(kChunk, device.stream);
        } catch (const std::logic_error&) { early_rejected = true; }
        failures += expect(early_rejected, "evict before the mirror capture is rejected");
        pool.capture_window_mirror(device.stream);
        failures += expect(pool.mirror_captured(), "mirror capture is recorded");

        const void* stable_base = arena.data;
        std::vector<std::uint32_t> readback(pattern.size());
        const auto read_arena = [&] {
            CUDA_CHECK(cudaMemcpyAsync(readback.data(), arena.data,
                                       readback.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, device.stream));
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
        };

        for (int cycle = 0; cycle < 2; ++cycle) {
            auto transaction = pool.evict(kChunk + kChunk / 2, device.stream);
            failures += expect(transaction.open(), "evict opens a transaction");
            failures += expect(transaction.leased().bytes == 2 * kChunk,
                               "evict rounds the extent up to chunks");
            failures += expect(pool.transaction_open(), "pool reports the open transaction");
            CUDA_CHECK(cudaMemsetAsync(transaction.leased().data, 0xC3, transaction.leased().bytes,
                                       device.stream));
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            transaction.close();
            failures += expect(!transaction.open() && !pool.transaction_open(),
                               "close ends the transaction");
            failures += expect(!pool.poisoned(), "a clean close does not poison the pool");
            failures += expect(pool.arena().data == stable_base,
                               "arena base is stable across transactions");
            transaction.close(); // idempotent
            read_arena();
            failures += expect(std::memcmp(readback.data(), pattern.data(),
                                           pattern.size() * sizeof(std::uint32_t)) == 0,
                               "restored arena is byte-identical to the mirror image");
        }

        {
            // Restore must touch only the borrowed chunk: a byte outside the window (chunk 3c)
            // and a byte inside the window but outside a one-chunk transaction (chunk 4c) both
            // keep a deliberate corruption, while the borrowed chunk (5c) comes back exact.
            const std::byte poison{0xEE};
            CUDA_CHECK(cudaMemcpy(at(arena, 3 * kChunk), &poison, 1, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(at(arena, 4 * kChunk), &poison, 1, cudaMemcpyHostToDevice));
            {
                auto transaction = pool.evict(kChunk, device.stream);
                failures += expect(transaction.leased().bytes == kChunk, "one-chunk window");
                CUDA_CHECK(cudaMemsetAsync(transaction.leased().data, 0x11, kChunk,
                                           device.stream));
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
            } // destructor closes the transaction
            failures += expect(!pool.transaction_open(), "destructor closes the transaction");
            read_arena();
            const auto* bytes = reinterpret_cast<const std::byte*>(readback.data());
            failures += expect(bytes[3 * kChunk] == poison,
                               "restore did not rewrite bytes outside the window");
            failures += expect(bytes[4 * kChunk] == poison,
                               "restore did not rewrite the undirtied window chunk");
            failures += expect(std::memcmp(bytes + 5 * kChunk,
                                           reinterpret_cast<const std::byte*>(pattern.data()) +
                                               5 * kChunk,
                                           arena_bytes - 5 * kChunk) == 0,
                               "the borrowed chunk was restored byte-exact");
        }

        bool nested_rejected = false;
        {
            auto outer = pool.evict(kChunk, device.stream);
            try {
                (void)pool.evict(kChunk, device.stream);
            } catch (const std::logic_error&) { nested_rejected = true; }
        }
        failures += expect(nested_rejected, "a nested transaction is rejected");

        bool oversize_rejected = false;
        try {
            (void)pool.evict(3 * kChunk, device.stream);
        } catch (const std::invalid_argument&) { oversize_rejected = true; }
        failures += expect(oversize_rejected, "evict beyond the window capacity is rejected");
        failures += expect(!pool.transaction_open(), "rejected evict leaves the pool resident");

        bool oversize_window_rejected = false;
        try {
            ninfer::EvictableWeightPool too_wide(device, ninfer::EvictableWeightPool::Config{
                                                             .arena_bytes           = arena_bytes,
                                                             .evictable_tail_bytes  = tail_bytes,
                                                             .window_capacity_bytes = 4 * kChunk,
                                                         });
        } catch (const std::invalid_argument&) { oversize_window_rejected = true; }
        failures += expect(oversize_window_rejected, "a window wider than the tail is rejected");

        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "evictable weight pool test failed: " << error.what() << '\n';
        return 1;
    }
}
