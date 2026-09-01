// Correctness cover for the integer-activation prefill routes: the Q4G64 gate_up LinearSwiGLU and
// the Q5G64 down LinearAdd.
//
// These routes are reached from the target variant rather than through ops::linear_swiglu, so the
// shared run_profile harness cannot see them. They are verified here the same way: a deterministic
// row-split payload from the quantized-weight fixture, an FP64 oracle over exactly decoded weights
// and represented BF16 activations, and the activation-compute criterion for A8.
//
// The pass bound is upstream's own kA8QuantizationAllowance (0.04), the allowance the A8 activation
// compute path is held to in tests/ops/linear/linear_test_common.cpp. The observed relative L2 is
// printed on every case so drift inside that allowance is visible rather than silent -- it measures
// about 0.009 on the real 27B weight, so a four-fold regression would still pass the gate but be
// obvious in the output.

#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::test::ReductionCriterion;
using ninfer::test::ReductionStats;
using ninfer::test::compute_reduction_stats;
using ninfer::test::verify_reduction;
using ninfer::test::quantized_weight::PackedWeight;
namespace qw = ninfer::test::quantized_weight;

// One BF16 unit roundoff, and the A8 activation-quantisation allowance, both mirroring
// tests/ops/linear/linear_test_common.cpp.
constexpr double kBf16UnitRoundoff        = 1.0 / 256.0;
constexpr double kA8QuantizationAllowance = 0.04;
constexpr ReductionCriterion kA8Criterion{kA8QuantizationAllowance, kBf16UnitRoundoff,
                                          1.5 * kA8QuantizationAllowance};

// Hidden states are roughly Gaussian with a few fixed channels an order of magnitude larger. The
// outlier channels are what per-token activation scaling gets wrong, so they belong in the fixture.
std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t r = 0; r < rows; ++r) {
            const std::uint64_t h = qw::detail::mix64(
                (static_cast<std::uint64_t>(t) << 32) ^ static_cast<std::uint64_t>(r) ^ seed);
            float v = static_cast<float>(static_cast<std::int32_t>(h & 0xffffu) - 32768) / 32768.0F;
            if ((r % 431) == 7) { v *= 24.0F; } // fixed outlier channels
            // BF16, round-to-nearest-even, so the oracle sees exactly what the kernel reads.
            std::uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            const std::uint32_t lsb = (bits >> 16) & 1u;
            bits += 0x7fffu + lsb;
            out[static_cast<std::size_t>(t) * rows + r] = static_cast<std::uint16_t>(bits >> 16);
        }
    }
    return out;
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::vector<double> read_bf16(const test::GuardedDeviceBuffer& buffer, std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    buffer.copy_to_host(host.data(), elements * sizeof(std::uint16_t));
    std::vector<double> out(elements);
    for (std::size_t i = 0; i < elements; ++i) { out[i] = bf16_value(host[i]); }
    return out;
}

// Sampled (row, token) pairs, so the FP64 oracle stays affordable at registered shapes.
struct Samples {
    std::vector<double> actual;
    std::vector<double> reference;
};

int run_gate_up(std::int32_t tokens) {
    constexpr std::int32_t kRows = 34816;
    constexpr std::int32_t kCols = 5120;
    constexpr std::int32_t kOut  = kRows / 2;

    const PackedWeight host_weight =
        qw::make_patterned_weight(QType::Q4G64_F16S, kRows, kCols, 4801U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 91U);

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));

    const std::size_t out_elements = static_cast<std::size_t>(kOut) * tokens;
    test::GuardedDeviceBuffer output(out_elements * sizeof(std::uint16_t));
    output.fill(0xff);

    WorkspaceArena workspace(
        std::max<std::size_t>(ops::detail::q4a8_swiglu_workspace_capacity_bytes(tokens, tokens), 256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor destination(output.data(), DType::BF16, {kOut, tokens});
    ops::detail::q4a8_swiglu_launch(x, weight, destination, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize q4a8 swiglu");

    const std::string label = "LinearSwiGLU Q4_A8INT T=" + std::to_string(tokens);
    int failures            = 0;
    failures += output.verify_guards(label);

    const std::vector<double> got = read_bf16(output, out_elements);
    Samples s;
    std::vector<float> input(static_cast<std::size_t>(kCols));
    for (int pick = 0; pick < 24; ++pick) {
        const std::int32_t token = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 7919U + 3U) % static_cast<std::uint64_t>(tokens));
        const std::int32_t row = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 104729U + 11U) % static_cast<std::uint64_t>(kOut));
        for (std::int32_t k = 0; k < kCols; ++k) {
            input[static_cast<std::size_t>(k)] =
                bf16_value(activation[static_cast<std::size_t>(token) * kCols + k]);
        }
        const double gate = qw::dot_fp64(host_weight, row, input.data(), kCols);
        const double up   = qw::dot_fp64(host_weight, kOut + row, input.data(), kCols);
        s.reference.push_back(gate / (1.0 + std::exp(-gate)) * up);
        s.actual.push_back(got[static_cast<std::size_t>(token) * kOut + row]);
    }

    const ReductionStats stats = compute_reduction_stats(
        s.actual.data(), s.reference.data(), static_cast<std::int64_t>(s.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kA8QuantizationAllowance << ")\n";
    failures += verify_reduction(label, s.actual, s.reference, kA8Criterion);
    return failures;
}

int run_down(std::int32_t tokens) {
    constexpr std::int32_t kRows = 5120;
    constexpr std::int32_t kCols = 17408;

    const PackedWeight host_weight =
        qw::make_patterned_weight(QType::Q5G64_F16S, kRows, kCols, 5501U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 77U);
    const std::vector<std::uint16_t> residual0  = make_activation(kRows, tokens, 13U);

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));

    const std::size_t res_elements = static_cast<std::size_t>(kRows) * tokens;
    test::GuardedDeviceBuffer device_residual(res_elements * sizeof(std::uint16_t));
    device_residual.copy_from_host(residual0.data(), res_elements * sizeof(std::uint16_t));

    WorkspaceArena workspace(
        std::max<std::size_t>(ops::detail::q5a8_add_workspace_capacity_bytes(tokens, tokens), 256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor residual(device_residual.data(), DType::BF16, {kRows, tokens});
    ops::detail::q5a8_add_launch(x, weight, residual, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize q5a8 add");

    const std::string label = "LinearAdd Q5_A8INT T=" + std::to_string(tokens);
    int failures            = 0;
    failures += device_residual.verify_guards(label);

    const std::vector<double> got = read_bf16(device_residual, res_elements);
    Samples s;
    std::vector<float> input(static_cast<std::size_t>(kCols));
    for (int pick = 0; pick < 24; ++pick) {
        const std::int32_t token = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 6151U + 5U) % static_cast<std::uint64_t>(tokens));
        const std::int32_t row = static_cast<std::int32_t>(
            qw::detail::mix64(pick * 199933U + 17U) % static_cast<std::uint64_t>(kRows));
        for (std::int32_t k = 0; k < kCols; ++k) {
            input[static_cast<std::size_t>(k)] =
                bf16_value(activation[static_cast<std::size_t>(token) * kCols + k]);
        }
        const double acc = qw::dot_fp64(host_weight, row, input.data(), kCols);
        const double base =
            bf16_value(residual0[static_cast<std::size_t>(token) * kRows + row]);
        s.reference.push_back(base + acc);
        s.actual.push_back(got[static_cast<std::size_t>(token) * kRows + row]);
    }

    const ReductionStats stats = compute_reduction_stats(
        s.actual.data(), s.reference.data(), static_cast<std::int64_t>(s.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kA8QuantizationAllowance << ")\n";
    failures += verify_reduction(label, s.actual, s.reference, kA8Criterion);
    return failures;
}

// The routes must decline anything they do not cover, so the caller falls back to A16 rather than
// producing a wrong answer. Decode and partial prefill chunks depend on this.
int run_admission() {
    int failures = 0;
    const PackedWeight q4 =
        qw::make_patterned_weight(QType::Q4G64_F16S, 34816, 5120, 1U);
    const PackedWeight q5 =
        qw::make_patterned_weight(QType::Q5G64_F16S, 5120, 17408, 2U);
    const PackedWeight wrong_shape =
        qw::make_patterned_weight(QType::Q4G64_F16S, 4096, 5120, 3U);

    void* fake = reinterpret_cast<void*>(static_cast<std::uintptr_t>(4096));
    struct Case {
        const char* what;
        bool got;
        bool want;
    };
    const Case cases[] = {
        {"q4 accepts 128 tokens", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 128), true},
        {"q4 accepts 1024 tokens", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 1024), true},
        {"q4 declines decode", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 1), false},
        {"q4 declines partial tile", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 200), false},
        {"q4 declines zero tokens", ops::detail::q4a8_swiglu_supported(q4.device_weight(fake), 0), false},
        {"q4 declines a Q5 weight", ops::detail::q4a8_swiglu_supported(q5.device_weight(fake), 128), false},
        {"q4 declines another shape", ops::detail::q4a8_swiglu_supported(wrong_shape.device_weight(fake), 128), false},
        {"q4 declines a null payload", ops::detail::q4a8_swiglu_supported(q4.device_weight(nullptr), 128), false},
        {"q5 accepts 128 tokens", ops::detail::q5a8_add_supported(q5.device_weight(fake), 128), true},
        {"q5 declines decode", ops::detail::q5a8_add_supported(q5.device_weight(fake), 1), false},
        {"q5 declines partial tile", ops::detail::q5a8_add_supported(q5.device_weight(fake), 129), false},
        {"q5 declines a Q4 weight", ops::detail::q5a8_add_supported(q4.device_weight(fake), 128), false},
        {"q5 declines a null high plane", ops::detail::q5a8_add_supported(q5.device_weight(nullptr), 128), false},
    };
    for (const Case& c : cases) {
        if (c.got != c.want) {
            std::cerr << "admission: " << c.what << " returned " << (c.got ? "true" : "false")
                      << ", expected " << (c.want ? "true" : "false") << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        int failures = run_admission();
        for (const std::int32_t tokens : {128, 256, 512}) {
            failures += run_gate_up(tokens);
            failures += run_down(tokens);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " integer-activation prefill routes correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "integer-activation prefill route test failed: " << error.what() << '\n';
        return 1;
    }
}
