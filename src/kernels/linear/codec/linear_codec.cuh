#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels::detail {

struct Q4Codec {
    static constexpr int kBits = 4;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 32;            // ceil(64*4/8)
    __device__ static void load_group(const std::uint8_t* codes, const std::uint8_t* high,
                                      const std::uint8_t* scales, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        (void)high;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + group;
        const std::uint16_t sb  = static_cast<std::uint16_t>(scales[group_index * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed = codes + group_index * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = packed[lane >> 1];
            const int u = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int s = (u & 0x08) ? (u - 16) : u;          // sign-extend bit 3
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

struct Q5Codec {
    static constexpr int kBits = 5;
    static constexpr int kGroupK = 64;
    static constexpr int kNibbleBytesPerRowPerGroup = 32;
    static constexpr int kHighBytesPerRowPerGroup = 8;
    __device__ static void load_group(const std::uint8_t* codes, const std::uint8_t* high,
                                      const std::uint8_t* scales, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + group;
        const std::uint16_t sb  = static_cast<std::uint16_t>(scales[group_index * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* nibble = codes + group_index * kNibbleBytesPerRowPerGroup;
        const std::uint8_t* high_bits = high + group_index * kHighBytesPerRowPerGroup;
#pragma unroll
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = nibble[lane >> 1];
            const int low = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int hi = (high_bits[lane >> 3] >> (lane & 7)) & 0x01;
            const int u = low | (hi << 4);
            const int s = (u & 0x10) ? (u - 32) : u;
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

struct Q6Codec {
    static constexpr int kBits = 6;
    static constexpr int kGroupK = 64;
    static constexpr int kNibbleBytesPerRowPerGroup = 32;
    static constexpr int kHighBytesPerRowPerGroup = 16;
    __device__ static void load_group(const std::uint8_t* codes, const std::uint8_t* high,
                                      const std::uint8_t* scales, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + group;
        const std::uint16_t sb  = static_cast<std::uint16_t>(scales[group_index * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(scales[group_index * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* nibble = codes + group_index * kNibbleBytesPerRowPerGroup;
        const std::uint8_t* high_bits = high + group_index * kHighBytesPerRowPerGroup;
#pragma unroll
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = nibble[lane >> 1];
            const int low = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int bitpos = lane * 2;
            const int hi = (high_bits[bitpos >> 3] >> (bitpos & 7)) & 0x03;
            const int u = low | (hi << 4);
            const int s = (u & 0x20) ? (u - 64) : u;
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

} // namespace qus::kernels::detail
