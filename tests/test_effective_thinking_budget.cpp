#include "runtime/engine/effective_thinking_budget.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

using ninfer::runtime::EffectiveThinkingBudget;
using ninfer::runtime::effective_thinking_budget;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_c_around_b() {
    int failures = 0;
    constexpr std::uint32_t R = 28;
    constexpr std::uint32_t B = 50;

    // C = B - 1: C <= B -> B, available
    {
        const auto res = effective_thinking_budget(B, B - 1, R);
        failures += check(res.effective_budget == B && res.early_close_available,
                          "C = B - 1 did not return B, available");
    }

    // C = B: C <= B -> B, available
    {
        const auto res = effective_thinking_budget(B, B, R);
        failures += check(res.effective_budget == B && res.early_close_available,
                          "C = B did not return B, available");
    }

    // C = B + 1: C > R (51 > 28) and C - B < R (1 < 28) -> C - R, available
    {
        const auto res = effective_thinking_budget(B, B + 1, R);
        failures += check(res.effective_budget == (B + 1) - R && res.early_close_available,
                          "C = B + 1 did not return C - R, available");
    }

    return failures;
}

int test_c_around_b_plus_r() {
    int failures = 0;
    constexpr std::uint32_t R = 28;
    constexpr std::uint32_t B = 50;

    // C = B + R - 1: C > R and C - B = R - 1 < R -> C - R, available
    {
        const auto res = effective_thinking_budget(B, B + R - 1, R);
        failures += check(res.effective_budget == B - 1 && res.early_close_available,
                          "C = B + R - 1 did not return C - R = B - 1, available");
    }

    // C = B + R: C - B = R >= R -> B, available
    {
        const auto res = effective_thinking_budget(B, B + R, R);
        failures += check(res.effective_budget == B && res.early_close_available,
                          "C = B + R did not return B, available");
    }

    // C = B + R + 1: C - B = R + 1 >= R -> B, available
    {
        const auto res = effective_thinking_budget(B, B + R + 1, R);
        failures += check(res.effective_budget == B && res.early_close_available,
                          "C = B + R + 1 did not return B, available");
    }

    return failures;
}

int test_c_around_r_and_r_plus_1() {
    int failures = 0;
    constexpr std::uint32_t R = 28;
    constexpr std::uint32_t B = 10; // B < R

    // C = R - 1: C <= R and C > B -> B, not available (Case 3)
    {
        const auto res = effective_thinking_budget(B, R - 1, R);
        failures += check(res.effective_budget == B && !res.early_close_available,
                          "C = R - 1 (with B < R) did not return B, unavailable");
    }

    // C = R: C <= R and C > B -> B, not available (Case 3)
    {
        const auto res = effective_thinking_budget(B, R, R);
        failures += check(res.effective_budget == B && !res.early_close_available,
                          "C = R (with B < R) did not return B, unavailable");
    }

    // C = R + 1: C > R and C - B = (R + 1) - 10 = 19 < R -> C - R = 1, available (Case 2)
    {
        const auto res = effective_thinking_budget(B, R + 1, R);
        failures += check(res.effective_budget == 1 && res.early_close_available,
                          "C = R + 1 (with B < R) did not return 1, available");
    }

    // C = R + 2: C > R and C - B = 20 < R -> C - R = 2, available (Case 2)
    {
        const auto res = effective_thinking_budget(B, R + 2, R);
        failures += check(res.effective_budget == 2 && res.early_close_available,
                          "C = R + 2 (with B < R) did not return 2, available");
    }

    // When B >= R: C <= B dominates for C <= R, so behavior is Case 1
    {
        constexpr std::uint32_t B_large = 30;
        const auto res_rm1 = effective_thinking_budget(B_large, R - 1, R);
        failures += check(res_rm1.effective_budget == B_large && res_rm1.early_close_available,
                          "C = R - 1 (with B >= R) did not return B, available");
        const auto res_r = effective_thinking_budget(B_large, R, R);
        failures += check(res_r.effective_budget == B_large && res_r.early_close_available,
                          "C = R (with B >= R) did not return B, available");
        const auto res_rp1 = effective_thinking_budget(B_large, R + 1, R);
        failures += check(res_rp1.effective_budget == B_large && res_rp1.early_close_available,
                          "C = R + 1 (with B >= R) did not return B, available");
    }

    return failures;
}

int test_b_zero() {
    int failures = 0;
    constexpr std::uint32_t R = 28;
    constexpr std::uint32_t B = 0;

    // C = 0: C <= B -> B = 0, available
    {
        const auto res = effective_thinking_budget(B, 0, R);
        failures += check(res.effective_budget == 0 && res.early_close_available,
                          "B = 0, C = 0 did not return 0, available");
    }

    // 0 < C < R: C > B and C <= R and C - B < R -> B = 0, not available (Case 3)
    {
        const auto res_1 = effective_thinking_budget(B, 1, R);
        failures += check(res_1.effective_budget == 0 && !res_1.early_close_available,
                          "B = 0, C = 1 did not return 0, unavailable");

        const auto res_rm1 = effective_thinking_budget(B, R - 1, R);
        failures += check(res_rm1.effective_budget == 0 && !res_rm1.early_close_available,
                          "B = 0, C = R - 1 did not return 0, unavailable");
    }

    // C >= R: C - B >= R -> B = 0, available (Case 1)
    {
        const auto res_r = effective_thinking_budget(B, R, R);
        failures += check(res_r.effective_budget == 0 && res_r.early_close_available,
                          "B = 0, C = R did not return 0, available");

        const auto res_rp1 = effective_thinking_budget(B, R + 1, R);
        failures += check(res_rp1.effective_budget == 0 && res_rp1.early_close_available,
                          "B = 0, C = R + 1 did not return 0, available");

        const auto res_100 = effective_thinking_budget(B, 100, R);
        failures += check(res_100.effective_budget == 0 && res_100.early_close_available,
                          "B = 0, C = 100 did not return 0, available");
    }

    return failures;
}

int test_b_one() {
    int failures = 0;
    constexpr std::uint32_t R = 28;
    constexpr std::uint32_t B = 1;

    // C = 0: C <= B -> B = 1, available
    {
        const auto res = effective_thinking_budget(B, 0, R);
        failures += check(res.effective_budget == 1 && res.early_close_available,
                          "B = 1, C = 0 did not return 1, available");
    }

    // C = 1: C <= B -> B = 1, available
    {
        const auto res = effective_thinking_budget(B, 1, R);
        failures += check(res.effective_budget == 1 && res.early_close_available,
                          "B = 1, C = 1 did not return 1, available");
    }

    // 1 < C <= R: C > B and C <= R -> B = 1, not available (Case 3)
    {
        const auto res_2 = effective_thinking_budget(B, 2, R);
        failures += check(res_2.effective_budget == 1 && !res_2.early_close_available,
                          "B = 1, C = 2 did not return 1, unavailable");

        const auto res_r = effective_thinking_budget(B, R, R);
        failures += check(res_r.effective_budget == 1 && !res_r.early_close_available,
                          "B = 1, C = R did not return 1, unavailable");
    }

    // C = R + 1: C - B = R >= R -> B = 1, available (Case 1)
    {
        const auto res_rp1 = effective_thinking_budget(B, R + 1, R);
        failures += check(res_rp1.effective_budget == 1 && res_rp1.early_close_available,
                          "B = 1, C = R + 1 did not return 1, available");
    }

    // C = R + 2: C - B = R + 1 >= R -> B = 1, available (Case 1)
    {
        const auto res_rp2 = effective_thinking_budget(B, R + 2, R);
        failures += check(res_rp2.effective_budget == 1 && res_rp2.early_close_available,
                          "B = 1, C = R + 2 did not return 1, available");
    }

    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_c_around_b();
    failures += test_c_around_b_plus_r();
    failures += test_c_around_r_and_r_plus_1();
    failures += test_b_zero();
    failures += test_b_one();
    if (failures == 0) {
        std::cout << "All effective_thinking_budget tests passed.\n";
    }
    return failures;
}
