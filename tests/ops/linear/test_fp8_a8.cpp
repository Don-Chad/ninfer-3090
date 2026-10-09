#include "core/weight.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_fp8_a8() {
    constexpr std::array attn_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures =
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {14336, 5120, 827U, Comparison::SampledRows, true, attn_decode_invocations});
    constexpr std::array attn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1023, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 829U, Comparison::SampledRows, true, attn_invocations});
    constexpr std::array attn_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 833U, Comparison::SampledRows, true, attn_bulk_invocations});
    constexpr std::array gdn_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {16384, 5120, 837U, Comparison::SampledRows, true, gdn_decode_invocations});
    constexpr std::array gdn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 839U, Comparison::SampledRows, true, gdn_invocations});
    constexpr std::array gdn_bulk_invocations{
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 841U, Comparison::SampledRows, true, gdn_bulk_invocations});
    constexpr std::array mlp_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {34816, 5120, 851U, Comparison::SampledRows, true, mlp_decode_invocations});
    constexpr std::array mlp_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 853U, Comparison::SampledRows, true, mlp_invocations});

    constexpr std::array mlp_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{511, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 863U, Comparison::SampledRows, true, mlp_bulk_invocations});

    constexpr std::array residual6144_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 6144, 855U, Comparison::SampledRows, true, residual6144_decode_invocations});
    constexpr std::array residual6144_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {5120, 6144, 857U, Comparison::SampledRows, true, residual6144_invocations});
    constexpr std::array residual17408_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 17408, 867U, Comparison::SampledRows, true, residual17408_decode_invocations});
    constexpr std::array residual17408_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape(
        "FP8_A8", ActivationCompute::A8, make_fp8_weight,
        {5120, 17408, 859U, Comparison::SampledRows, true, residual17408_invocations});

    struct Problem {
        std::int32_t rows;
        std::int32_t input_rows;
        bool a8_at_one;
        bool a8_at_two;
    };

    for (const Problem problem :
         {Problem{14336, 5120, false, false}, Problem{16384, 5120, false, false},
          Problem{34816, 5120, false, false}, Problem{5120, 6144, false, false},
          Problem{5120, 17408, false, false}}) {
        const std::size_t one = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1, 1);
        const std::size_t two = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 2, 2);
        const std::size_t forty_eight = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 48, 48);
        const std::size_t early_interval = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 2, 4);
        const std::size_t hot_interval = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1, 48);
        const std::size_t exact_1024 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1024, 1024);
        const std::size_t exact_1048 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1048, 1048);
        const std::size_t spanning = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1000, 1048);
        const std::size_t a16 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::A16Only, 1, 2048);
        if ((one != 0) != problem.a8_at_one || (two != 0) != problem.a8_at_two ||
            early_interval != 0 || forty_eight <= two || hot_interval != forty_eight ||
            exact_1024 <= forty_eight || exact_1048 <= exact_1024 || spanning != exact_1048 ||
            a16 != 0) {
            std::cerr << "FP8 A8 workspace interval contract mismatch for N=" << problem.rows
                      << " K=" << problem.input_rows << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_a8();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
