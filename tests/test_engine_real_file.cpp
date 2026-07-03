#include "qus/runtime/engine.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kTextPayloadBytes = 16378329088ULL;
constexpr std::size_t kMtpPayloadBytes = 451267584ULL;
constexpr std::size_t kDefaultArenaSlackBytes = 256ULL * kMiB;

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

bool enough_free_memory(std::size_t bytes) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        std::cout << "SKIP: cudaMemGetInfo failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    if (free_bytes < bytes) {
        std::cout << "SKIP: not enough free GPU memory; need " << bytes << ", free "
                  << free_bytes << '\n';
        return false;
    }
    return true;
}

void print_stats(const char* label, const qus::EngineMemoryStats& stats) {
    std::cout << "ENGINE_REAL " << label << " loaded_payload="
              << stats.q5090_loaded_payload_bytes << " weight_capacity="
              << stats.weights.capacity_bytes << " weight_used=" << stats.weights.used_bytes
              << " tensor_count=" << stats.q5090_tensor_count
              << " quant_count=" << stats.q5090_quant_count << '\n';
}

qus::EngineMemoryStats load_real_engine(const std::filesystem::path& weights_path,
                                        int mtp_draft_tokens) {
    qus::EngineOptions options;
    options.device = 0;
    options.max_ctx = 128;
    options.work_bytes = 64ULL * kMiB;
    options.mtp_draft_tokens = mtp_draft_tokens;

    qus::Engine engine(options);
    engine.load(weights_path.string());
    const qus::EngineMemoryStats stats = engine.memory_stats();
    cudaDeviceSynchronize();
    return stats;
}

int expect_stats(const qus::EngineMemoryStats& stats, std::size_t expected_loaded_payload,
                 std::size_t expected_weight_capacity, const char* label) {
    int failures = 0;
    failures += stats.loaded ? 0 : fail(std::string(label) + " did not report loaded");
    failures += stats.weights.present ? 0 : fail(std::string(label) + " missing weight arena");
    failures += stats.weights.capacity_bytes == expected_weight_capacity
                    ? 0
                    : fail(std::string(label) + " weight arena capacity mismatch");
    failures += stats.weights.used_bytes >= expected_loaded_payload
                    ? 0
                    : fail(std::string(label) + " weight arena used below loaded payload");
    failures += stats.weights.used_bytes <= stats.weights.capacity_bytes
                    ? 0
                    : fail(std::string(label) + " weight arena used above capacity");
    failures += stats.q5090_loaded_payload_bytes == expected_loaded_payload
                    ? 0
                    : fail(std::string(label) + " loaded payload mismatch");
    failures += stats.q5090_tensor_count == 1164
                    ? 0
                    : fail(std::string(label) + " tensor count mismatch");
    failures += stats.q5090_quant_count > 0 ? 0 : fail(std::string(label) + " quant count zero");
    return failures;
}

} // namespace

int main() {
    const std::filesystem::path root(QUS_SOURCE_DIR);
    const std::filesystem::path weights_path =
        root / "out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus";
    if (!std::filesystem::exists(weights_path)) {
        std::cout << "SKIP: real q5090 file not present\n";
        return 0;
    }

    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device for real Engine load\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    const std::size_t file_size = std::filesystem::file_size(weights_path);
    const std::size_t expected_weight_capacity =
        file_size + kDefaultArenaSlackBytes + kMtpPayloadBytes;
    if (!enough_free_memory(expected_weight_capacity + kGiB)) { return 0; }

    int failures = 0;
    {
        const qus::EngineMemoryStats stats = load_real_engine(weights_path, 0);
        print_stats("mtp=0", stats);
        failures += expect_stats(stats, kTextPayloadBytes, expected_weight_capacity, "mtp=0");
    }
    {
        const qus::EngineMemoryStats stats = load_real_engine(weights_path, 1);
        print_stats("mtp=1", stats);
        failures += expect_stats(stats, kTextPayloadBytes + kMtpPayloadBytes,
                                 expected_weight_capacity, "mtp=1");
    }
    return failures == 0 ? 0 : fail("real Engine load test failed");
}
