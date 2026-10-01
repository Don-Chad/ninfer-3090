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

int test_output_capacity() {
    using ninfer::runtime::effective_output_capacity;
    int failures = 0;
    // The request limit binds while the context has room.
    failures += check(effective_output_capacity(100, 1000, 200) == 100,
                      "request limit did not bind capacity");
    // The context binds otherwise, and the last context position is writable (+1).
    failures += check(effective_output_capacity(5000, 1000, 200) == 801,
                      "context window did not bound capacity as max_context - prompt + 1");
    // A prompt that fills the context still licenses exactly one token.
    failures += check(effective_output_capacity(5000, 1000, 1000) == 1,
                      "full-context prompt did not license exactly one token");
    // Derived output budgets arrive as the largest representable limit.
    failures += check(effective_output_capacity(UINT32_MAX, 159744, 143000) == 16745,
                      "derived output limit did not collapse to the remaining context");
    return failures;
}

int test_apply_to_request_options() {
    using ninfer::ThinkingControlOptions;
    using ninfer::runtime::apply_effective_thinking_budget;
    int failures = 0;
    constexpr std::uint32_t control_tokens = 27; // R = 28

    // No cap: the options are left untouched, whatever the capacity.
    {
        ThinkingControlOptions options;
        apply_effective_thinking_budget(options, 5, control_tokens);
        failures += check(!options.effective_budget && options.early_close_available,
                          "uncapped request was modified");
    }
    // Boundary window above R: the cap is lowered so control plus one token fits.
    {
        ThinkingControlOptions options{.budget = 100};
        apply_effective_thinking_budget(options, 110, control_tokens);
        failures += check(options.budget == 100 && options.effective_budget == 82 &&
                              options.early_close_available,
                          "boundary window did not lower the effective budget to C - R");
    }
    // At or below R: the requested value is kept and early close is withdrawn.
    {
        ThinkingControlOptions options{.budget = 10};
        apply_effective_thinking_budget(options, 20, control_tokens);
        failures += check(options.budget == 10 && options.effective_budget == 10 &&
                              !options.early_close_available,
                          "capacity below R did not disable early close");
    }
    // Plenty of room: unchanged.
    {
        ThinkingControlOptions options{.budget = 100};
        apply_effective_thinking_budget(options, 4096, control_tokens);
        failures += check(options.effective_budget == 100 && options.early_close_available,
                          "ample capacity changed the budget");
    }
    // A template with no control suffix needs R = 1, so any C > B already fits the close.
    {
        ThinkingControlOptions options{.budget = 100};
        apply_effective_thinking_budget(options, 101, 0);
        failures += check(options.effective_budget == 100 && options.early_close_available,
                          "bare-close template changed the budget");
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
    failures += test_output_capacity();
    failures += test_apply_to_request_options();
    if (failures == 0) {
        std::cout << "All effective_thinking_budget tests passed.\n";
    }
    return failures;
}
