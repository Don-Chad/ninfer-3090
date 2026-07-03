#include "qus/kernels/gdn_commit.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

std::vector<unsigned char> copy_bytes(const Tensor& tensor) {
    std::vector<unsigned char> out(tensor.bytes());
    cudaMemcpy(out.data(), tensor.data, out.size(), cudaMemcpyDeviceToHost);
    return out;
}

int expect_slot_byte(const Tensor& tensor, int slot_dim, int slot, unsigned char expected,
                     const char* label) {
    const Tensor view = tensor.slice(slot_dim, slot, 1);
    const std::vector<unsigned char> bytes = copy_bytes(view);
    for (const unsigned char value : bytes) {
        if (value != expected) {
            std::cerr << label << " expected byte 0x" << std::hex << static_cast<int>(expected)
                      << " got 0x" << static_cast<int>(value) << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

void set_slot_byte(const Tensor& tensor, int slot_dim, int slot, unsigned char value) {
    const Tensor view = tensor.slice(slot_dim, slot, 1);
    cudaMemset(view.data, value, view.bytes());
}

int commit_case(int accepted, unsigned char expected_slot0_conv, unsigned char expected_slot0_ssm,
                const char* label) {
    constexpr int C = 8;
    constexpr int K = 4;
    constexpr int V = 5;
    constexpr int H = 3;
    constexpr int S = 3;

    DBuf dconv(static_cast<std::size_t>(C) * 3u * S * 2u);
    DBuf dssm(static_cast<std::size_t>(K) * V * H * S * 4u);
    DBuf da(sizeof(std::int32_t));
    cudaMemcpy(da.p, &accepted, sizeof(accepted), cudaMemcpyHostToDevice);

    Tensor conv(dconv.p, DType::BF16, {C, 3, S});
    Tensor ssm(dssm.p, DType::FP32, {K, V, H, S});
    Tensor a(da.p, DType::I32, {1});
    set_slot_byte(conv, 2, 0, 0x10);
    set_slot_byte(conv, 2, 1, 0x20);
    set_slot_byte(conv, 2, 2, 0x30);
    set_slot_byte(ssm, 3, 0, 0x40);
    set_slot_byte(ssm, 3, 1, 0x50);
    set_slot_byte(ssm, 3, 2, 0x60);

    kernels::gdn_commit(conv, ssm, a, nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    const std::string prefix(label);
    failures += expect_slot_byte(conv, 2, 0, expected_slot0_conv,
                                 (prefix + " conv slot0").c_str());
    failures += expect_slot_byte(ssm, 3, 0, expected_slot0_ssm, (prefix + " ssm slot0").c_str());
    failures += expect_slot_byte(conv, 2, 1, 0x20, (prefix + " preserves conv slot1").c_str());
    failures += expect_slot_byte(conv, 2, 2, 0x30, (prefix + " preserves conv slot2").c_str());
    failures += expect_slot_byte(ssm, 3, 1, 0x50, (prefix + " preserves ssm slot1").c_str());
    failures += expect_slot_byte(ssm, 3, 2, 0x60, (prefix + " preserves ssm slot2").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += commit_case(0, 0x10, 0x40, "gdn_commit a=0 no-op");
    failures += commit_case(1, 0x20, 0x50, "gdn_commit a=1");
    failures += commit_case(2, 0x30, 0x60, "gdn_commit a=2");
    failures += commit_case(-1, 0x10, 0x40, "gdn_commit negative no-write");
    failures += commit_case(3, 0x10, 0x40, "gdn_commit out-of-range no-write");

    std::cout << (failures ? "FAIL" : "OK") << " gdn_commit correctness\n";
    return failures == 0 ? 0 : fail("gdn_commit test failed");
}
