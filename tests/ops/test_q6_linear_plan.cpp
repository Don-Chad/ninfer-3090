#include "ops/linear/q6/q6_rowsplit_plan.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using ninfer::ops::detail::Q6KernelVariant;
using ninfer::ops::detail::Q6Plan;
using ninfer::ops::detail::Q6Problem;
using ninfer::ops::detail::Q6ScheduleId;

namespace {

using S = Q6ScheduleId;
using V = Q6KernelVariant;

int failures = 0;

void fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    ++failures;
}

bool same_plan(Q6Plan lhs, Q6Plan rhs) {
    return lhs.schedule == rhs.schedule && lhs.variant == rhs.variant;
}

std::string plan_name(Q6Plan plan) {
    return std::string(ninfer::ops::detail::q6_schedule_name(plan.schedule)) + "." +
           ninfer::ops::detail::q6_kernel_variant_name(plan.variant);
}

Q6KernelVariant expected_variant(Q6ScheduleId schedule, const Q6Problem& problem) {
    if (schedule == S::SimtR8C4) { return V::None; }
    const int cols = schedule == S::MmaR64C64 ? 64 : 128;
    return problem.rows % 64 == 0 && problem.cols % cols == 0 && problem.k == problem.padded_k &&
                   problem.k % 64 == 0
               ? V::Full
               : V::Predicated;
}

Q6Plan expected_plan(const Q6Problem& problem) {
    S schedule;
    if (problem.rows == 248320 && problem.k == 5120) {
        schedule = S::SimtR8C4;
    } else if (problem.rows == 1152 && problem.k == 1536) {
        const int cols = problem.cols;
        if (cols <= 96) {
            schedule = S::SimtR8C4;
        } else if (cols <= 704) {
            schedule = S::MmaR64C64;
        } else if (cols <= 828) {
            schedule = S::MmaR64C128;
        } else if (cols <= 832) {
            schedule = S::MmaR64C64;
        } else if (cols <= 896) {
            schedule = S::MmaR64C128;
        } else if (cols <= 960) {
            schedule = S::MmaR64C64;
        } else if (cols <= 1024) {
            schedule = S::MmaR64C128;
        } else if (cols <= 1088) {
            schedule = S::MmaR64C64;
        } else {
            schedule = S::MmaR64C128;
        }
    } else {
        throw std::logic_error("test oracle received an unregistered Q6 problem");
    }
    return {schedule, expected_variant(schedule, problem)};
}

void expect_plan(const std::string& label, const Q6Problem& problem, Q6Plan expected) {
    if (!ninfer::ops::detail::q6_rowsplit_admits(problem)) {
        fail(label, "production admission rejected the problem");
        return;
    }
    try {
        const Q6Plan actual = ninfer::ops::detail::q6_rowsplit_resolve_plan(problem);
        if (!same_plan(actual, expected)) {
            fail(label, "expected " + plan_name(expected) + ", got " + plan_name(actual));
        }
        if (!ninfer::ops::detail::q6_candidate_is_legal(actual.schedule, actual.variant, problem)) {
            fail(label, "resolver returned a physically illegal plan");
        }
    } catch (const std::exception& error) {
        fail(label, std::string("resolver threw: ") + error.what());
    }
}

void full_support_scan() {
    for (std::int32_t cols = 0; cols <= 7; ++cols) {
        const Q6Problem problem{248320, 5120, 5120, cols};
        const bool admitted = cols >= 1 && cols <= 6;
        if (ninfer::ops::detail::q6_rowsplit_admits(problem) != admitted) {
            fail("head admission", admitted ? "rejected admitted cols" : "accepted rejected cols");
        }
        if (admitted) { expect_plan("head route", problem, expected_plan(problem)); }
    }

    for (std::int32_t cols = 0; cols <= 131076; ++cols) {
        const Q6Problem problem{1152, 1536, 1536, cols};
        const bool admitted = cols >= 4 && cols <= 131072 && cols % 4 == 0;
        if (ninfer::ops::detail::q6_rowsplit_admits(problem) != admitted) {
            fail("vision admission",
                 admitted ? "rejected admitted cols" : "accepted rejected cols");
        }
        if (admitted) { expect_plan("vision route", problem, expected_plan(problem)); }
    }
}

struct BoundaryCase {
    Q6Problem problem;
    Q6Plan expected;
};

void route_boundaries() {
    constexpr std::array<BoundaryCase, 20> cases{{
        {{248320, 5120, 5120, 1}, {S::SimtR8C4, V::None}},
        {{248320, 5120, 5120, 6}, {S::SimtR8C4, V::None}},
        {{1152, 1536, 1536, 4}, {S::SimtR8C4, V::None}},
        {{1152, 1536, 1536, 96}, {S::SimtR8C4, V::None}},
        {{1152, 1536, 1536, 100}, {S::MmaR64C64, V::Predicated}},
        {{1152, 1536, 1536, 704}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 708}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 768}, {S::MmaR64C128, V::Full}},
        {{1152, 1536, 1536, 828}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 832}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 836}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 896}, {S::MmaR64C128, V::Full}},
        {{1152, 1536, 1536, 900}, {S::MmaR64C64, V::Predicated}},
        {{1152, 1536, 1536, 960}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 964}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 1024}, {S::MmaR64C128, V::Full}},
        {{1152, 1536, 1536, 1028}, {S::MmaR64C64, V::Predicated}},
        {{1152, 1536, 1536, 1088}, {S::MmaR64C64, V::Full}},
        {{1152, 1536, 1536, 1092}, {S::MmaR64C128, V::Predicated}},
        {{1152, 1536, 1536, 131072}, {S::MmaR64C128, V::Full}},
    }};

    for (const BoundaryCase& test : cases) {
        expect_plan("route boundary", test.problem, test.expected);
    }
}

void rejection_contract() {
    constexpr std::array<Q6Problem, 5> rejected{{
        {248320, 5120, 5120, 7},
        {65536, 5120, 5120, 1},
        {4096, 5120, 5120, 4},
        {1152, 1536, 1536, 5},
        {248320, 2048, 2048, 1},
    }};

    for (const Q6Problem& problem : rejected) {
        if (ninfer::ops::detail::q6_rowsplit_admits(problem)) {
            fail("rejection", "production admission accepted an unregistered problem");
        }
        try {
            const Q6Plan plan = ninfer::ops::detail::q6_rowsplit_resolve_plan(problem);
            fail("rejection", "resolver returned " + plan_name(plan));
        } catch (const std::invalid_argument&) {
        } catch (const std::exception& error) {
            fail("rejection", std::string("resolver threw wrong exception: ") + error.what());
        }
    }
}

} // namespace

int main() {
    full_support_scan();
    route_boundaries();
    rejection_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q6 Linear production plan\n";
    return failures == 0 ? 0 : 1;
}
