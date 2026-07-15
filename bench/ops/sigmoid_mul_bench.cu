#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

__global__ void sigmoid_mul_payload_control(const uint4* gate, uint4* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const uint4 a = gate[i];
        uint4 b       = x[i];
        b.x ^= a.x;
        b.y ^= a.y;
        b.z ^= a.z;
        b.w ^= a.w;
        x[i] = b;
    }
}

void run(int tokens, bool control) {
    constexpr int d     = 4096;
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(tokens);
    DBuf gate           = make_bf16(n);
    DBuf x              = make_bf16(n);
    Tensor tgate(gate.p, DType::BF16, {256, 16, tokens});
    Tensor tx(x.p, DType::BF16, {256, 16, tokens});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block   = 256;
                constexpr int maxGrid = 4096;
                const auto packs      = static_cast<std::int64_t>(n / 8);
                const int grid        = static_cast<int>(std::min<std::int64_t>(
                    maxGrid, std::max<std::int64_t>(1, (packs + block - 1) / block)));
                sigmoid_mul_payload_control<<<grid, block, 0, stream>>>(
                    static_cast<const uint4*>(gate.p), static_cast<uint4*>(x.p), packs);
            } else {
                ops::sigmoid_mul(tgate, tx, stream);
            }
        },
        static_cast<double>(n) * 6.0);

    char tag[80];
    std::snprintf(tag, sizeof(tag), "%s [256,16,%-4d]", control ? "control" : "sigmoid_mul",
                  tokens);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_tokens = 0;
    bool control        = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            selected_tokens = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T] [--control]\n", argv[0]);
            return 2;
        }
    }
    if (selected_tokens < 0) {
        std::fprintf(stderr, "tokens must be positive\n");
        return 2;
    }

    if (selected_tokens > 0) {
        run(selected_tokens, control);
        return 0;
    }
    for (const int tokens : {1, 2, 3, 4, 5, 6, 1024}) { run(tokens, control); }
    return 0;
}
