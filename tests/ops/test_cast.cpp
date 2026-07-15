#include "ninfer/ops/cast.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<float> make_values(std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.0137f) * 32.0f +
                    static_cast<float>(static_cast<int>(i % 29) - 14) * 0.03125f;
    }
    const std::vector<float> edge{-10.5f,          -1.00390625f, -0.0f,      0.0f,
                                  0.333251953125f, 1.00390625f,  3.1415927f, 65504.0f};
    for (std::size_t i = 0; i < std::min(count, edge.size()); ++i) { values[i] = edge[i]; }
    return values;
}

int cast_case(std::int32_t d, std::int32_t columns, std::size_t source_offset = 0,
              std::size_t destination_offset = 0) {
    const std::size_t count         = static_cast<std::size_t>(d) * columns;
    const std::vector<float> source = make_values(count);
    DBuf device_source(source_offset + count * sizeof(float));
    DBuf device_destination(destination_offset + count * sizeof(std::uint16_t));
    auto* source_ptr      = static_cast<std::byte*>(device_source.p) + source_offset;
    auto* destination_ptr = static_cast<std::byte*>(device_destination.p) + destination_offset;
    cudaMemcpy(source_ptr, source.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    Tensor source_tensor(source_ptr, DType::FP32, {d, columns});
    Tensor destination_tensor(destination_ptr, DType::BF16, {d, columns});
    ops::cast_fp32_to_bf16(source_tensor, destination_tensor, nullptr);
    cudaDeviceSynchronize();

    std::vector<std::uint16_t> actual(count);
    cudaMemcpy(actual.data(), destination_ptr, count * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < count; ++i) {
        if (actual[i] != f32_to_bf16(source[i])) {
            std::cerr << "cast mismatch D=" << d << " C=" << columns << " at " << i << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += cast_case(4, 2);        // x4
    failures += cast_case(3, 2);        // x2
    failures += cast_case(7, 1);        // scalar tail domain
    failures += cast_case(12, 1, 4, 2); // deliberately unaligned scalar route
    failures += cast_case(1536, 8);     // minimum video
    failures += cast_case(1536, 256);   // minimum image
    failures += cast_case(1536, 4096);  // canonical image
    std::cout << (failures ? "FAIL" : "OK") << " cast_fp32_to_bf16\n";
    return failures ? 1 : 0;
}
