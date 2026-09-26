// Proves the residency mechanism behind the vision VRAM overlay: physical chunks
// backing a stable VA range can be unmapped, remapped into a different reserved
// range, used there, remapped home, and re-uploaded from a pinned host mirror --
// all while a CUDA graph captured against the home addresses stays launchable
// and correct. Also reports per-chunk map/unmap latency for the overlay budget.

#include "core/arena.h"
#include "core/device.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

void cu_check(CUresult result, const char* expr) {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(expr) + " failed: " +
                             (name != nullptr ? name : "unknown CUresult"));
}

#define CU_CHECK(expr) ::cu_check((expr), #expr)

constexpr std::size_t kChunkCount   = 4;
constexpr std::size_t kEvictedFirst = 2;   // evict the chunk suffix [2, 4)

__global__ void checksum_kernel(const std::uint32_t* words, std::size_t count,
                                unsigned long long* out) {
    unsigned long long local = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local += words[i];
    }
    atomicAdd(out, local);
}

void map_rw(CUdeviceptr va, std::size_t bytes, CUmemGenericAllocationHandle handle, int device) {
    CU_CHECK(cuMemMap(va, bytes, 0, handle, 0));
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = device;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(va, bytes, &access, 1));
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
        CU_CHECK(cuInit(0));

        CUmemAllocationProp prop{};
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = device.device;
        std::size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        const std::size_t chunk =
            ((64ULL << 20) + granularity - 1) / granularity * granularity;
        const std::size_t total = chunk * kChunkCount;

        CUdeviceptr home = 0, overlay = 0;
        CU_CHECK(cuMemAddressReserve(&home, total, granularity, 0, 0));
        CU_CHECK(cuMemAddressReserve(&overlay, total, granularity, 0, 0));

        std::vector<CUmemGenericAllocationHandle> handles(kChunkCount);
        for (std::size_t i = 0; i < kChunkCount; ++i) {
            CU_CHECK(cuMemCreate(&handles[i], chunk, &prop, 0));
            map_rw(home + i * chunk, chunk, handles[i], device.device);
        }

        // Deterministic pattern, mirrored in pinned host memory like the
        // production weight mirrors.
        ninfer::PinnedHostBuffer mirror(total);
        auto* mirror_words = static_cast<std::uint32_t*>(mirror.data());
        for (std::size_t i = 0; i < total / sizeof(std::uint32_t); ++i) {
            mirror_words[i] = static_cast<std::uint32_t>(i * 2654435761ULL);
        }
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(home), mirror.data(), total,
                                   cudaMemcpyHostToDevice, device.stream));

        unsigned long long* result = nullptr;
        CUDA_CHECK(cudaMalloc(&result, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(unsigned long long), device.stream));
        device.synchronize();

        // Reference checksum with everything resident.
        const std::size_t words = total / sizeof(std::uint32_t);
        checksum_kernel<<<256, 256, 0, device.stream>>>(
            reinterpret_cast<const std::uint32_t*>(home), words, result);
        unsigned long long expected = 0;
        CUDA_CHECK(cudaMemcpyAsync(&expected, result, sizeof(expected), cudaMemcpyDeviceToHost,
                                   device.stream));
        device.synchronize();

        // Capture the checksum as a graph against the home addresses.
        cudaGraph_t graph         = nullptr;
        cudaGraphExec_t execution = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal));
        CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(unsigned long long), device.stream));
        checksum_kernel<<<256, 256, 0, device.stream>>>(
            reinterpret_cast<const std::uint32_t*>(home), words, result);
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0));

        // Overlay transaction: unmap the chunk suffix, remap the same physical
        // pages into the overlay range, scribble there (vision stand-in),
        // remap home, and restore only the evicted bytes from the mirror.
        const std::size_t evicted_bytes  = (kChunkCount - kEvictedFirst) * chunk;
        const std::size_t evicted_offset = kEvictedFirst * chunk;
        const auto evict_start           = std::chrono::steady_clock::now();
        for (std::size_t i = kEvictedFirst; i < kChunkCount; ++i) {
            CU_CHECK(cuMemUnmap(home + i * chunk, chunk));
            map_rw(overlay + (i - kEvictedFirst) * chunk, chunk, handles[i], device.device);
        }
        const auto evict_end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(overlay), 0xA5, evicted_bytes,
                                   device.stream));
        device.synchronize();
        const auto restore_start = std::chrono::steady_clock::now();
        for (std::size_t i = kEvictedFirst; i < kChunkCount; ++i) {
            CU_CHECK(cuMemUnmap(overlay + (i - kEvictedFirst) * chunk, chunk));
            map_rw(home + i * chunk, chunk, handles[i], device.device);
        }
        const auto restore_end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(home + evicted_offset),
                                   mirror_words + evicted_offset / sizeof(std::uint32_t),
                                   evicted_bytes, cudaMemcpyHostToDevice, device.stream));
        device.synchronize();

        // The pre-captured graph must still run and see restored bytes.
        CUDA_CHECK(cudaGraphLaunch(execution, device.stream));
        unsigned long long actual = 0;
        CUDA_CHECK(cudaMemcpyAsync(&actual, result, sizeof(actual), cudaMemcpyDeviceToHost,
                                   device.stream));
        device.synchronize();

        const auto micros = [](auto begin, auto end) {
            return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        };
        std::cout << "chunk_bytes=" << chunk << " granularity=" << granularity
                  << " evict_map_us=" << micros(evict_start, evict_end)
                  << " restore_map_us=" << micros(restore_start, restore_end) << '\n';

        int failures = 0;
        if (actual != expected) {
            std::cerr << "graph checksum after overlay transaction expected " << expected
                      << ", got " << actual << '\n';
            ++failures;
        }

        CUDA_CHECK(cudaGraphExecDestroy(execution));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaFree(result));
        for (std::size_t i = 0; i < kChunkCount; ++i) {
            CU_CHECK(cuMemUnmap(home + i * chunk, chunk));
            CU_CHECK(cuMemRelease(handles[i]));
        }
        CU_CHECK(cuMemAddressFree(home, total));
        CU_CHECK(cuMemAddressFree(overlay, total));
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "vmm graph remap test failed: " << error.what() << '\n';
        return 1;
    }
}
