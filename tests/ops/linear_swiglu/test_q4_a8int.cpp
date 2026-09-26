#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // AllowA8Int is permissive: the integer route covers token counts that are positive
        // multiples of 128 and the resolver falls back to A16 for the rest. These cases straddle
        // that boundary deliberately -- 127/128/129 and 511/512/513 -- so both sides of the
        // admission rule are exercised through the public Op, and every case is held to the same
        // A8 activation allowance whichever route ran.
        constexpr std::array<std::int32_t, 14> kTokenCases{
            1, 2, 33, 127, 128, 129, 256, 257, 384, 511, 512, 513, 640, 1024,
        };
        const int failures = run_profile(
            "LinearSwiGLU Q4_A8INT",
            {QType::Q4G64_F16S, 34816, 5120, 17408, 1409U, ActivationCompute::A8Int},
            kTokenCases);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4_A8INT correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A8INT test failed: " << error.what() << '\n';
        return 1;
    }
}
