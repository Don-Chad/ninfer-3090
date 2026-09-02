#pragma once

// ninfer::core - portable unsigned 128-bit intermediates for saturating host arithmetic.
//
// GCC and Clang provide `unsigned __int128`; MSVC does not, and this fork builds natively on
// Windows. Rather than fork the arithmetic behind #ifdef, this header defines one explicit
// hi/lo representation used by every target, so the Linux and Windows builds evaluate the
// identical expression and one set of tests covers both.
//
// The operations here are only used by host-side planning and cost accounting, never in a
// device kernel or a decode-step hot path, so the 32-bit-limb multiply costs nothing that
// matters.

#include <cstdint>
#include <limits>

namespace ninfer::core {

struct Uint128 {
    std::uint64_t low  = 0;
    std::uint64_t high = 0;
};

/// Full 64x64 -> 128 unsigned product. Exact for every input; never overflows.
[[nodiscard]] constexpr Uint128 wide_multiply(std::uint64_t left, std::uint64_t right) noexcept {
    const std::uint64_t left_low   = left & 0xffffffffULL;
    const std::uint64_t left_high  = left >> 32U;
    const std::uint64_t right_low  = right & 0xffffffffULL;
    const std::uint64_t right_high = right >> 32U;

    const std::uint64_t low_product = left_low * right_low;
    std::uint64_t carry             = left_high * right_low + (low_product >> 32U);
    const std::uint64_t middle      = left_low * right_high + (carry & 0xffffffffULL);

    Uint128 result;
    result.low  = (middle << 32U) | (low_product & 0xffffffffULL);
    result.high = left_high * right_high + (carry >> 32U) + (middle >> 32U);
    return result;
}

/// Wrapping 128-bit addition. `overflowed` reports carry out of the full 128-bit width.
[[nodiscard]] constexpr Uint128 wide_add(Uint128 left, Uint128 right, bool& overflowed) noexcept {
    Uint128 result;
    result.low                = left.low + right.low;
    const std::uint64_t carry = result.low < left.low ? 1U : 0U;

    const std::uint64_t high_sum = left.high + right.high;
    const bool high_wrapped      = high_sum < left.high;
    result.high                  = high_sum + carry;
    overflowed                   = high_wrapped || result.high < high_sum;
    return result;
}

[[nodiscard]] constexpr bool wide_less(Uint128 left, Uint128 right) noexcept {
    return left.high != right.high ? left.high < right.high : left.low < right.low;
}

[[nodiscard]] constexpr bool wide_fits_uint64(Uint128 value) noexcept { return value.high == 0; }

/// Logical right shift by `bits`, which must be in [0, 64).
[[nodiscard]] constexpr Uint128 wide_shift_right(Uint128 value, unsigned bits) noexcept {
    if (bits == 0) { return value; }
    Uint128 result;
    result.low  = (value.low >> bits) | (value.high << (64U - bits));
    result.high = value.high >> bits;
    return result;
}

/// Saturating narrowing to uint64.
[[nodiscard]] constexpr std::uint64_t wide_clamp_to_uint64(Uint128 value) noexcept {
    return wide_fits_uint64(value) ? value.low : std::numeric_limits<std::uint64_t>::max();
}

} // namespace ninfer::core
