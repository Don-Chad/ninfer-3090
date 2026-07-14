#include "ninfer/ops/position.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    constexpr int count = 9;
    DBuf filled(static_cast<std::size_t>(count) * sizeof(int));
    Tensor filled_tensor(filled.p, DType::I32, {count});
    ops::fill_i32_positions(filled_tensor, 17, nullptr);

    DBuf delta = to_device_i32({-5});
    DBuf offset(static_cast<std::size_t>(count) * sizeof(int));
    Tensor delta_tensor(delta.p, DType::I32, {1});
    Tensor offset_tensor(offset.p, DType::I32, {count});
    ops::offset_i32_positions(filled_tensor, delta_tensor, offset_tensor, nullptr);
    cudaDeviceSynchronize();

    std::vector<int> expected_filled(count);
    std::vector<int> expected_offset(count);
    for (int i = 0; i < count; ++i) {
        expected_filled[static_cast<std::size_t>(i)] = 17 + i;
        expected_offset[static_cast<std::size_t>(i)] = 12 + i;
    }
    if (from_device_i32(filled, count) != expected_filled) {
        std::cerr << "fill_i32_positions mismatch\n";
        return 1;
    }
    if (from_device_i32(offset, count) != expected_offset) {
        std::cerr << "offset_i32_positions mismatch\n";
        return 1;
    }
    std::cout << "OK position Ops\n";
    return 0;
}
