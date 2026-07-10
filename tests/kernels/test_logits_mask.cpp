#include "qus/kernels/logits_mask.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace qus;
using namespace qus::test;

int main() {
    constexpr int rows  = 8;
    constexpr int cols  = 3;
    constexpr int valid = 5;
    std::vector<float> input(static_cast<std::size_t>(rows * cols));
    for (std::size_t i = 0; i < input.size(); ++i) { input[i] = static_cast<float>(i) / 8.0F; }
    round_to_bf16(input);

    DBuf device = to_device_bf16(input);
    Tensor logits(device.p, DType::BF16, {rows, cols});
    kernels::mask_invalid_token_logits(logits, valid, nullptr);
    cudaDeviceSynchronize();
    const std::vector<double> got = from_device_bf16(device, input.size());

    int failures = 0;
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const std::size_t index = static_cast<std::size_t>(col * rows + row);
            if (row < valid) {
                if (got[index] != static_cast<double>(input[index])) {
                    std::cerr << "valid logit changed at " << index << '\n';
                    ++failures;
                }
            } else if (!std::isinf(got[index]) || got[index] >= 0.0) {
                std::cerr << "reserved logit was not -inf at " << index << '\n';
                ++failures;
            }
        }
    }
    return failures == 0 ? 0 : 1;
}
