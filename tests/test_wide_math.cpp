// Validates the portable 128-bit primitives against closed-form uint64 oracles and against the
// compiler's own unsigned __int128 where that type exists. The Windows build has no __int128, so
// the oracle branch is compiled out there and the structural cases below carry the check.

#include "core/wide_math.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

using ninfer::core::Uint128;
using ninfer::core::wide_add;
using ninfer::core::wide_clamp_to_uint64;
using ninfer::core::wide_fits_uint64;
using ninfer::core::wide_less;
using ninfer::core::wide_multiply;
using ninfer::core::wide_shift_right;

void test_multiply_known_values() {
    check(wide_multiply(0, 0).low == 0 && wide_multiply(0, 0).high == 0, "0*0");
    check(wide_multiply(kMax, 1).low == kMax && wide_multiply(kMax, 1).high == 0, "max*1");

    // (2^64-1)^2 == 2^128 - 2^65 + 1 -> high = 2^64-2, low = 1
    const Uint128 square = wide_multiply(kMax, kMax);
    check(square.low == 1U && square.high == kMax - 1U, "max*max");

    // 2^32 * 2^32 == 2^64 -> high = 1, low = 0
    const Uint128 pow64 = wide_multiply(1ULL << 32U, 1ULL << 32U);
    check(pow64.low == 0U && pow64.high == 1U, "2^32*2^32");

    // 2^63 * 2 == 2^64
    const Uint128 shifted = wide_multiply(1ULL << 63U, 2U);
    check(shifted.low == 0U && shifted.high == 1U, "2^63*2");
}

void test_multiply_against_uint64_oracle() {
    // Products that stay inside uint64 must agree with plain uint64 multiplication.
    std::mt19937_64 rng(20260829U);
    for (int iteration = 0; iteration < 20000; ++iteration) {
        const std::uint64_t left  = rng() >> 32U; // < 2^32
        const std::uint64_t right = rng() >> 32U; // < 2^32
        const Uint128 product     = wide_multiply(left, right);
        check(product.high == 0U && product.low == left * right, "32x32 product");
    }
}

#if defined(__SIZEOF_INT128__)
void test_multiply_against_int128() {
    std::mt19937_64 rng(7U);
    for (int iteration = 0; iteration < 200000; ++iteration) {
        const std::uint64_t left  = rng();
        const std::uint64_t right = rng();
        const Uint128 got         = wide_multiply(left, right);
        const unsigned __int128 expected =
            static_cast<unsigned __int128>(left) * static_cast<unsigned __int128>(right);
        check(got.low == static_cast<std::uint64_t>(expected) &&
                  got.high == static_cast<std::uint64_t>(expected >> 64U),
              "wide_multiply vs __int128");
        if (failures != 0) { return; }
    }
}

void test_add_against_int128() {
    std::mt19937_64 rng(11U);
    for (int iteration = 0; iteration < 200000; ++iteration) {
        const Uint128 left{rng(), rng()};
        const Uint128 right{rng(), rng()};
        bool overflowed   = false;
        const Uint128 got = wide_add(left, right, overflowed);

        const unsigned __int128 wide_left =
            (static_cast<unsigned __int128>(left.high) << 64U) | left.low;
        const unsigned __int128 wide_right =
            (static_cast<unsigned __int128>(right.high) << 64U) | right.low;
        const unsigned __int128 expected = wide_left + wide_right;
        const bool expected_overflow     = expected < wide_left;

        check(got.low == static_cast<std::uint64_t>(expected) &&
                  got.high == static_cast<std::uint64_t>(expected >> 64U),
              "wide_add value vs __int128");
        check(overflowed == expected_overflow, "wide_add carry vs __int128");
        if (failures != 0) { return; }
    }
}

void test_shift_against_int128() {
    std::mt19937_64 rng(13U);
    for (int iteration = 0; iteration < 50000; ++iteration) {
        const Uint128 value{rng(), rng()};
        const unsigned bits = static_cast<unsigned>(rng() % 64U);
        const Uint128 got   = wide_shift_right(value, bits);
        const unsigned __int128 expected =
            ((static_cast<unsigned __int128>(value.high) << 64U) | value.low) >> bits;
        check(got.low == static_cast<std::uint64_t>(expected) &&
                  got.high == static_cast<std::uint64_t>(expected >> 64U),
              "wide_shift_right vs __int128");
        if (failures != 0) { return; }
    }
}
#endif // __SIZEOF_INT128__

void test_add_carry_boundaries() {
    bool overflowed = false;

    (void)wide_add(Uint128{kMax, kMax}, Uint128{0, 0}, overflowed);
    check(!overflowed, "max128 + 0 must not overflow");

    (void)wide_add(Uint128{kMax, kMax}, Uint128{1, 0}, overflowed);
    check(overflowed, "max128 + 1 must overflow");

    // Carry out of the low limb alone must propagate without reporting 128-bit overflow.
    const Uint128 carried = wide_add(Uint128{kMax, 0}, Uint128{1, 0}, overflowed);
    check(!overflowed && carried.low == 0U && carried.high == 1U, "low-limb carry propagates");

    // Carry out of the low limb that then overflows the high limb must report overflow.
    (void)wide_add(Uint128{kMax, kMax}, Uint128{1, 0}, overflowed);
    check(overflowed, "low carry into saturated high limb overflows");
}

void test_compare_shift_and_clamp() {
    check(wide_less(Uint128{0, 0}, Uint128{1, 0}), "0 < 1");
    check(wide_less(Uint128{kMax, 0}, Uint128{0, 1}), "high limb dominates comparison");
    check(!wide_less(Uint128{0, 1}, Uint128{kMax, 0}), "comparison is not reversed");
    check(!wide_less(Uint128{5, 5}, Uint128{5, 5}), "equal values are not less");

    const Uint128 value{0, 1}; // 2^64
    check(wide_shift_right(value, 32U).low == (1ULL << 32U), "2^64 >> 32");
    check(wide_shift_right(value, 0U).high == 1U, "shift by zero is identity");

    check(wide_fits_uint64(Uint128{kMax, 0}), "max uint64 fits");
    check(!wide_fits_uint64(Uint128{0, 1}), "2^64 does not fit");
    check(wide_clamp_to_uint64(Uint128{7, 0}) == 7U, "clamp passes small values");
    check(wide_clamp_to_uint64(Uint128{0, 1}) == kMax, "clamp saturates");
}

} // namespace

int main() {
    test_multiply_known_values();
    test_multiply_against_uint64_oracle();
    test_add_carry_boundaries();
    test_compare_shift_and_clamp();
#if defined(__SIZEOF_INT128__)
    test_multiply_against_int128();
    test_add_against_int128();
    test_shift_against_int128();
#endif
    if (failures != 0) {
        std::fprintf(stderr, "%d wide-math check(s) failed\n", failures);
        return 1;
    }
    std::printf("wide math OK\n");
    return 0;
}
