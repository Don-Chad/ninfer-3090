#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/kv_cache_lloyd4_oracle.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim          = 256;
constexpr std::int32_t kQuantGroup       = 64;
constexpr std::int32_t kQuantGroups      = kHeadDim / kQuantGroup;
constexpr std::int32_t kFp8QuantGroup    = kHeadDim;
constexpr std::int32_t kFp8QuantGroups   = 1;
constexpr std::int32_t kNvfp4QuantGroup  = 16;
constexpr std::int32_t kNvfp4QuantGroups = kHeadDim / kNvfp4QuantGroup;
constexpr std::int32_t kNvfp4CodeBytes   = kHeadDim / 2;
// The rk8v4 value plane groups 32 values per scale, not the key plane's 64 (see
// D256KVCacheProfile); its packed byte holds the dimension pair (2p, 2p+1).
constexpr std::int32_t kRk8ValueGroup  = 32;
constexpr std::int32_t kRk8ValueGroups = kHeadDim / kRk8ValueGroup;
constexpr float kAttentionScale        = 0.0625f;
constexpr std::uint16_t kOutputCanary  = 0x7fc1u;

// The Op plans launches and workspace from the physical SM count of the executing device;
// capacity queries and launches must use the same count.
DeviceExecutionView test_execution(cudaStream_t stream = nullptr) {
    static const std::int32_t multiprocessor_count = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return static_cast<std::int32_t>(count);
    }();
    return {stream, multiprocessor_count};
}

// The Op contract pairs the family/key-plane dtype with the value plane's code dtype
// (D256KVCacheProfile); the harness names the pair. A DType::U8 value code selects the rk8v4
// packed signed int4 profile.
struct CachePlan {
    DType dtype;
    DType value_code_dtype;

    // rk8v4 is the one plan whose value plane is a packed code stream rather than one code per
    // stored element, and it is the only plan whose value code dtype differs from its key dtype.
    [[nodiscard]] constexpr bool packed_values() const { return value_code_dtype == DType::U8; }

    // rk4v4 stores its rotated keys as packed 4-bit Lloyd-Max indices over the INT8 family's G64
    // scale plane, and pairs them with rk8v4's packed value plane.
    bool packed_keys = false;
};

constexpr CachePlan kPlanBf16{DType::BF16, DType::BF16};
constexpr CachePlan kPlanInt8{DType::I8, DType::I8};
constexpr CachePlan kPlanRk8v4{DType::I8, DType::U8};
constexpr CachePlan kPlanRk4v4{DType::I8, DType::U8, true};
constexpr CachePlan kPlanFp8{DType::FP8_E4M3FN, DType::FP8_E4M3FN};

// The public KvCacheStorage selection a CachePlan corresponds to, for call sites (workspace
// sizing) that only accept the public enum, not the CachePlan itself.
KvCacheStorage cache_plan_storage(const CachePlan& plan) {
    if (plan.dtype == DType::BF16) return KvCacheStorage::BFloat16;
    if (plan.dtype == DType::FP8_E4M3FN) return KvCacheStorage::Fp8E4M3Row256;
    if (plan.packed_keys) return KvCacheStorage::RotatedLloyd4KeyInt4Value;
    return plan.packed_values() ? KvCacheStorage::RotatedInt8KeyInt4ValueGroup64
                                : KvCacheStorage::Int8Group64;
}

// A1 and A3 use one fixed criterion for each registered storage profile; token count, geometry,
// execution envelope, and private launch route do not select or relax it.
// These three profiles all end in a BF16 output store, so their gross bound sits at
// kBf16GrossRelativeFloor -- see op_check.h for the derivation. The bounds they used to carry
// (2.7e-3, 2.2e-3, 2.7e-3) were 0.56 to 0.69 of a single BF16 rounding step, and two of them were
// below the error actually observed here: they admitted their own cases only because
// `gross_absolute` carried them, so the relative term was doing no work.
//
// `relative_l2` is unchanged and stays the accuracy gate. It is what separates the three profiles,
// and the separation is real. Measured over the 46 rk8v4 cases against the 35 INT8 cases:
//
//     profile     max relative_l2
//     bf16            2.604e-3
//     int8-g64        2.990e-3
//     rk8v4           3.040e-3
//
// rk8v4 stages values through the same FP16 path as the INT8 coding (one represented FP16
// code-scale product per dimension, one FP16 P/V MMA, one BF16 output store) and its keys are the
// same rotated INT8 G64 plane, so those shared stages dominate and it lands just above INT8 rather
// than an order of magnitude above it. Re-derive with NINFER_DUMP_ATTN_STATS=1, which prints
// per-case stats.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ kBf16GrossRelativeFloor,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ kBf16GrossRelativeFloor,
};

constexpr ReductionCriterion kAttentionRk8v4Criterion{
    /*relative_l2*/ 3.25e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ kBf16GrossRelativeFloor,
};

// rk4v4 decodes its keys exactly to INT8 codes before the same s8 QK path rk8v4 uses, and the
// oracle consumes those decoded keys, so every remaining error term is rk8v4's.
constexpr ReductionCriterion kAttentionRk4v4Criterion = kAttentionRk8v4Criterion;

constexpr ReductionCriterion kAttentionFp8Criterion{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 9.0e-3,
};

constexpr ReductionCriterion kAttentionNvfp4Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-2,
};

constexpr ReductionCriterion kAttentionK8V4Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-2,
};

struct TestVectorLayout {
    DType code_dtype;
    std::int32_t code_extent;
    DType scale_dtype;
    std::int32_t scale_extent;
};

struct TestCacheLayout {
    TestVectorLayout key;
    TestVectorLayout value;
};

TestCacheLayout test_cache_layout(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {{DType::BF16, kHeadDim, DType::U8, 0}, {DType::BF16, kHeadDim, DType::U8, 0}};
    case KvCacheStorage::Int8Group64:
        return {{DType::I8, kHeadDim, DType::FP16, kQuantGroups},
                {DType::I8, kHeadDim, DType::FP16, kQuantGroups}};
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        // rk8v4: the value plane packs two signed 4-bit codes per byte (U8, half the head
        // dimension) over a finer 32-value group, so its scale plane is twice as wide.
        return {{DType::I8, kHeadDim, DType::FP16, kQuantGroups},
                {DType::U8, kHeadDim / 2, DType::FP16, kRk8ValueGroups}};
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        // rk4v4: packed 4-bit key indices over the G64 key scale plane, rk8v4's value plane.
        return {{DType::U8, kHeadDim / 2, DType::FP16, kQuantGroups},
                {DType::U8, kHeadDim / 2, DType::FP16, kRk8ValueGroups}};
    case KvCacheStorage::Fp8E4M3Row256:
        return {{DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups},
                {DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups}};
    case KvCacheStorage::Nvfp4Group16:
        return {{DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups},
                {DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups}};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {{DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups},
                {DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups}};
    }
    throw std::invalid_argument("unsupported test KV storage");
}

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"d256-h24-kv4", 24, 4},
    {"d256-h16-kv2", 16, 2},
};

ops::AttentionHeadGeometry op_geometry(const Geometry& geometry) {
    return {kHeadDim, geometry.q_heads, geometry.kv_heads};
}

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
    bool zero_q       = false;
    bool graph_replay = false;
    // Half-width of the uniform V distribution, for the cached rows and the appended ones.
    float v_amplitude = 1.0f;
};

enum class MappingPattern { Identity, Offset, Fragmented };

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    }
    return table;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t logical_plane_index(std::int32_t leading_extent, const Geometry& geometry,
                                std::int32_t padded_context, std::int32_t head,
                                std::int32_t position, std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

// Packed rk8v4 value plane: leading extent kHeadDim/2, one byte per adjacent dimension pair.
std::size_t packed_cache_index(const Geometry& geometry, std::int32_t padded_context,
                               std::int32_t head, std::int32_t position, std::int32_t byte) {
    return static_cast<std::size_t>(byte) +
           static_cast<std::size_t>(kHeadDim / 2) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t plane_scale_index(const Geometry& geometry, std::int32_t padded_context,
                              std::int32_t head, std::int32_t position, std::int32_t group,
                              std::int32_t extent) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(extent) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t plane_scale_elements(const Geometry& geometry, std::int32_t padded_context,
                                 std::int32_t extent) {
    return static_cast<std::size_t>(extent) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t fp8_scale_index(const Geometry& geometry, std::int32_t padded_context,
                            std::int32_t head, std::int32_t position) {
    (void)geometry;
    return static_cast<std::size_t>(position) +
           static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head);
}

std::size_t fp8_scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

std::size_t physical_plane_elements(std::int32_t leading_extent, const Geometry& geometry,
                                    std::int32_t physical_pages) {
    return static_cast<std::size_t>(leading_extent) * kPagedKVPageSize * geometry.kv_heads *
           physical_pages;
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

float round_to_f16(float value) { return f16_bits_to_f32(f32_to_f16_bits(value)); }

std::vector<std::uint16_t> to_f16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_f16_bits(values[i]); }
    return bits;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

float decode_e4m3fn_positive(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if (exponent == 0) return std::ldexp(static_cast<float>(mantissa), -9);
    return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

float decode_e4m3fn(std::uint8_t code) {
    const float magnitude = decode_e4m3fn_positive(code & 0x7fU);
    return (code & 0x80U) == 0 ? magnitude : -magnitude;
}

std::uint8_t encode_e4m3fn_rne_satfinite(float value) {
    const bool negative   = std::signbit(value);
    const float magnitude = std::abs(value);
    std::uint8_t selected = 0x7eU;
    if (magnitude < 448.0f) {
        for (std::uint8_t upper = 1; upper <= 0x7eU; ++upper) {
            const float upper_value = decode_e4m3fn_positive(upper);
            if (upper_value < magnitude) continue;
            const std::uint8_t lower = static_cast<std::uint8_t>(upper - 1);
            const float lower_value  = decode_e4m3fn_positive(lower);
            const float lower_error  = magnitude - lower_value;
            const float upper_error  = upper_value - magnitude;
            selected                 = lower_error < upper_error   ? lower
                                       : upper_error < lower_error ? upper
                                                                   : ((lower & 1U) == 0U ? lower : upper);
            break;
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 0x80U : 0U));
}

float decode_e2m1(std::uint8_t code) {
    constexpr std::array<float, 8> magnitude{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = magnitude.at(static_cast<std::size_t>(code & 0x07U));
    return (code & 0x08U) == 0 ? value : -value;
}

std::uint8_t encode_e2m1_rne_satfinite(float value) {
    const bool negative   = std::signbit(value);
    const float magnitude = std::abs(value);
    std::uint8_t selected = 7;
    if (magnitude < 6.0f) {
        for (std::uint8_t upper = 1; upper <= 7; ++upper) {
            const float upper_value = decode_e2m1(upper);
            if (upper_value < magnitude) continue;
            const std::uint8_t lower = static_cast<std::uint8_t>(upper - 1);
            const float lower_value  = decode_e2m1(lower);
            const float lower_error  = magnitude - lower_value;
            const float upper_error  = upper_value - magnitude;
            selected                 = lower_error < upper_error   ? lower
                                       : upper_error < lower_error ? upper
                                                                   : ((lower & 1U) == 0U ? lower : upper);
            break;
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 0x08U : 0U));
}

struct HostCache {
    Geometry geometry;
    DType dtype;
    DType value_code_dtype;
    KvCacheStorage storage;
    std::int32_t max_context;
    std::int32_t logical_capacity;
    ops::D256KVCacheProfile profile;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_fp16;
    std::vector<std::int8_t> k_i8;
    // rk4v4 packed key indices: one byte holds the dimension pair (2p, 2p+1).
    std::vector<std::uint8_t> k_packed;
    std::vector<std::int8_t> v_i8;
    // rk8v4 packed signed int4 value codes: one byte holds the dimension pair (2p, 2p+1).
    std::vector<std::uint8_t> v_packed;
    std::vector<std::uint8_t> k_fp8;
    std::vector<std::uint8_t> v_fp8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
    std::vector<std::uint8_t> k_nvfp4;
    std::vector<std::uint8_t> v_nvfp4;
    std::vector<std::uint8_t> k_nvfp4_scale;
    std::vector<std::uint8_t> v_nvfp4_scale;
    // Original-coordinate logical K after decoding the private rotated quantized representation.
    // This is fixture input to the independent Attention oracle, not a production staging oracle.
    std::vector<float> logical_k_quantized;
    std::vector<float> rotated_k_quantized;
    std::vector<float> rotated_v_quantized;

    [[nodiscard]] bool packed_values() const { return profile.packed_int4_values(); }
    [[nodiscard]] bool packed_keys() const { return storage == KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    [[nodiscard]] std::int32_t value_extent() const { return profile.value_leading_extent; }
    [[nodiscard]] std::int32_t k_scale_extent() const { return profile.scale_leading_extent; }
    [[nodiscard]] std::int32_t v_scale_extent() const { return profile.value_scale_leading_extent; }
};

void encode_group(std::span<const float> source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

// rk8v4 value group encoding: FP16-RNE(absmax/7) over a kRk8ValueGroup-value group, RNE-even
// codes clamped to +/-7, and one byte per adjacent dimension pair with the even dimension in the
// low nibble. Mirrors the persistent codec contract in ninfer/ops/kv_cache_append.h. Each byte
// is written whole, so re-encoding an already populated row cannot leave stale nibbles.
void encode_group_i4(std::span<const float> source, std::size_t source_base,
                     std::vector<std::uint8_t>& packed, std::size_t packed_base,
                     std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kRk8ValueGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const std::uint16_t scale_bits = f32_to_f16_bits(absmax / 7.0f);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t byte = 0; byte < kRk8ValueGroup / 2; ++byte) {
        const float low = source[source_base + static_cast<std::size_t>(2 * byte)];
        const float high = source[source_base + static_cast<std::size_t>(2 * byte + 1)];
        const std::int32_t low_code =
            stored_scale == 0.0f ? 0
                                 : std::clamp(round_even_to_i32(low * inverse_scale), -7, 7);
        const std::int32_t high_code =
            stored_scale == 0.0f ? 0
                                 : std::clamp(round_even_to_i32(high * inverse_scale), -7, 7);
        packed[packed_base + static_cast<std::size_t>(byte)] = static_cast<std::uint8_t>(
            (static_cast<unsigned>(low_code) & 0x0fu) |
            ((static_cast<unsigned>(high_code) & 0x0fu) << 4));
    }
}

// Inverse of the packed byte layout: dimension d sits in the low nibble of byte d/2 when d is
// even and the high nibble when d is odd.
std::int8_t unpack_int4_code(std::uint8_t packed, std::int32_t high_nibble) {
    const unsigned nibble = high_nibble != 0 ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

void normalized_hadamard_d256(std::array<float, kHeadDim>& values) {
    for (std::int32_t stride = 1; stride < kHeadDim; stride *= 2) {
        for (std::int32_t base = 0; base < kHeadDim; base += 2 * stride) {
            for (std::int32_t offset = 0; offset < stride; ++offset) {
                const float low  = values[static_cast<std::size_t>(base + offset)];
                const float high = values[static_cast<std::size_t>(base + offset + stride)];
                values[static_cast<std::size_t>(base + offset)]          = low + high;
                values[static_cast<std::size_t>(base + offset + stride)] = low - high;
            }
        }
    }
    for (float& value : values) { value *= 0x1p-4f; }
}

void normalized_hadamard_d256(std::array<double, kHeadDim>& values) {
    for (std::int32_t stride = 1; stride < kHeadDim; stride *= 2) {
        for (std::int32_t base = 0; base < kHeadDim; base += 2 * stride) {
            for (std::int32_t offset = 0; offset < stride; ++offset) {
                const double low  = values[static_cast<std::size_t>(base + offset)];
                const double high = values[static_cast<std::size_t>(base + offset + stride)];
                values[static_cast<std::size_t>(base + offset)]          = low + high;
                values[static_cast<std::size_t>(base + offset + stride)] = low - high;
            }
        }
    }
    for (double& value : values) value *= 0x1p-4;
}

void encode_nvfp4_rotated_row(std::span<const float> source, std::size_t source_base,
                              std::vector<std::uint8_t>& codes, std::size_t code_base,
                              std::vector<std::uint8_t>& scales, std::size_t scale_base,
                              std::vector<float>& represented_rotated,
                              std::size_t represented_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    for (std::int32_t group = 0; group < kNvfp4QuantGroups; ++group) {
        const std::int32_t d0 = group * kNvfp4QuantGroup;
        float absmax          = 0.0f;
        for (std::int32_t i = 0; i < kNvfp4QuantGroup; ++i) {
            absmax = std::max(absmax, std::abs(rotated[static_cast<std::size_t>(d0 + i)]));
        }
        std::uint8_t scale_code = 0;
        float represented_scale = 0.0f;
        if (absmax != 0.0f) {
            const float bounded = std::clamp(absmax / 6.0f, std::ldexp(1.0f, -9), 448.0f);
            scale_code          = encode_e4m3fn_rne_satfinite(bounded);
            represented_scale   = decode_e4m3fn(scale_code);
        }
        scales[scale_base + static_cast<std::size_t>(group)] = scale_code;
        for (std::int32_t pair = 0; pair < kNvfp4QuantGroup / 2; ++pair) {
            const std::int32_t low_d  = d0 + 2 * pair;
            const std::int32_t high_d = low_d + 1;
            const std::uint8_t low =
                represented_scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(rotated[static_cast<std::size_t>(low_d)] /
                                                represented_scale);
            const std::uint8_t high =
                represented_scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(rotated[static_cast<std::size_t>(high_d)] /
                                                represented_scale);
            codes[code_base + static_cast<std::size_t>(group * 8 + pair)] =
                static_cast<std::uint8_t>(low | (high << 4));
            represented_rotated[represented_base + static_cast<std::size_t>(low_d)] =
                decode_e2m1(low) * represented_scale;
            represented_rotated[represented_base + static_cast<std::size_t>(high_d)] =
                decode_e2m1(high) * represented_scale;
        }
    }
}

std::vector<float> represent_f16_rotated_queries(const std::vector<float>& q,
                                                 const Geometry& geometry, std::int32_t tokens) {
    std::vector<float> represented(q.size(), 0.0f);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < geometry.q_heads; ++head) {
            std::array<float, kHeadDim> rotated{};
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                rotated[static_cast<std::size_t>(d)] = q[q_index(geometry, head, d, token)];
            }
            normalized_hadamard_d256(rotated);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                represented[q_index(geometry, head, d, token)] =
                    round_to_f16(rotated[static_cast<std::size_t>(d)]);
            }
        }
    }
    return represented;
}

void encode_fp8_row(const std::array<float, kHeadDim>& values, std::vector<std::uint8_t>& codes,
                    std::size_t code_base, std::vector<std::uint16_t>& scales,
                    std::size_t scale_offset) {
    float absmax = 0.0f;
    for (const float value : values) absmax = std::max(absmax, std::abs(value));
    std::uint16_t scale_bits = 0;
    float represented_scale  = 0.0f;
    if (absmax != 0.0f) {
        const float raw_scale = absmax / 448.0f;
        scale_bits        = f32_to_f16_bits(std::clamp(raw_scale, std::ldexp(1.0f, -24), 65504.0f));
        represented_scale = f16_bits_to_f32(scale_bits);
    }
    scales[scale_offset]   = scale_bits;
    const float reciprocal = represented_scale == 0.0f ? 0.0f : 1.0f / represented_scale;
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        codes[code_base + static_cast<std::size_t>(d)] =
            represented_scale == 0.0f
                ? 0
                : encode_e4m3fn_rne_satfinite(values[static_cast<std::size_t>(d)] * reciprocal);
    }
}

void encode_fp8_rotated_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::uint8_t>& codes, std::size_t code_base,
                            std::vector<std::uint16_t>& scales, std::size_t scale_offset,
                            std::vector<float>& represented_rotated, std::size_t represented_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    encode_fp8_row(rotated, codes, code_base, scales, scale_offset);
    const float represented_scale = f16_bits_to_f32(scales[scale_offset]);
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        represented_rotated[represented_base + static_cast<std::size_t>(d)] =
            decode_e4m3fn(codes[code_base + static_cast<std::size_t>(d)]) * represented_scale;
    }
}

std::vector<float> quantize_fp8_rotated_queries(const std::vector<float>& q,
                                                const Geometry& geometry, std::int32_t tokens) {
    std::vector<float> represented(q.size(), 0.0f);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kHeadDim));
    std::vector<std::uint16_t> scales(1);
    std::vector<float> row(static_cast<std::size_t>(kHeadDim));
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < geometry.q_heads; ++head) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                row[static_cast<std::size_t>(d)] = q[q_index(geometry, head, d, token)];
            }
            std::vector<float> decoded(static_cast<std::size_t>(kHeadDim));
            encode_fp8_rotated_row(row, 0, codes, 0, scales, 0, decoded, 0);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                represented[q_index(geometry, head, d, token)] =
                    decoded[static_cast<std::size_t>(d)];
            }
        }
    }
    return represented;
}

void encode_rotated_fp8_key_row(std::span<const float> source, std::size_t source_base,
                                std::vector<std::uint8_t>& codes, std::size_t code_base,
                                std::vector<std::uint16_t>& scales, std::size_t scale_offset,
                                std::vector<float>& logical, std::size_t logical_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    encode_fp8_row(rotated, codes, code_base, scales, scale_offset);

    const float represented_scale = f16_bits_to_f32(scales[scale_offset]);
    std::array<float, kHeadDim> decoded{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        decoded[static_cast<std::size_t>(d)] =
            decode_e4m3fn(codes[code_base + static_cast<std::size_t>(d)]) * represented_scale;
    }
    normalized_hadamard_d256(decoded);
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        logical[logical_base + static_cast<std::size_t>(d)] = decoded[static_cast<std::size_t>(d)];
    }
}

void encode_rotated_key_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::int8_t>& codes, std::size_t code_base,
                            std::vector<std::uint16_t>& scales, std::size_t scale_base,
                            std::vector<float>& logical, std::size_t logical_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    for (std::int32_t group = 0; group < kQuantGroups; ++group) {
        const std::size_t d = static_cast<std::size_t>(group * kQuantGroup);
        encode_group(rotated, d, codes, code_base + d, scales,
                     scale_base + static_cast<std::size_t>(group));
    }

    std::array<float, kHeadDim> decoded{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        const std::size_t offset = static_cast<std::size_t>(d);
        const std::size_t group  = static_cast<std::size_t>(d / kQuantGroup);
        decoded[offset]          = static_cast<float>(codes[code_base + offset]) *
                          f16_bits_to_f32(scales[scale_base + group]);
    }
    normalized_hadamard_d256(decoded);
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        logical[logical_base + static_cast<std::size_t>(d)] = decoded[static_cast<std::size_t>(d)];
    }
}

// rk4v4 key coding: rotate, encode each G64 group with the shared independent oracle
// (ops/kv_cache_lloyd4_oracle.h), and decode the represented codes back to original coordinates
// for the attention oracle.
void encode_rotated_lloyd4_key_row(std::span<const float> source, std::size_t source_base,
                                   std::vector<std::uint8_t>& packed, std::size_t packed_base,
                                   std::vector<std::uint16_t>& scales, std::size_t scale_base,
                                   std::vector<float>& logical, std::size_t logical_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);

    std::array<std::uint8_t, kHeadDim> indices{};
    std::array<float, kHeadDim> decoded{};
    for (std::int32_t group = 0; group < kQuantGroups; ++group) {
        const Lloyd4Group encoded =
            lloyd4_encode_group(&rotated[static_cast<std::size_t>(group * kQuantGroup)]);
        const std::uint16_t scale_bits = f32_to_f16_bits(encoded.scale);
        scales[scale_base + static_cast<std::size_t>(group)] = scale_bits;
        const float represented = f16_bits_to_f32(scale_bits);
        for (std::int32_t i = 0; i < kQuantGroup; ++i) {
            const std::size_t d = static_cast<std::size_t>(group * kQuantGroup + i);
            indices[d]          = encoded.index[static_cast<std::size_t>(i)];
            decoded[d]          = static_cast<float>(lloyd4_code(indices[d])) * represented;
        }
    }
    for (std::int32_t byte = 0; byte < kHeadDim / 2; ++byte) {
        packed[packed_base + static_cast<std::size_t>(byte)] = static_cast<std::uint8_t>(
            indices[static_cast<std::size_t>(2 * byte)] |
            (indices[static_cast<std::size_t>(2 * byte + 1)] << 4));
    }
    normalized_hadamard_d256(decoded);
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        logical[logical_base + static_cast<std::size_t>(d)] = decoded[static_cast<std::size_t>(d)];
    }
}

// CachePlan overload: covers this fork's four originally-registered profiles, including rk8v4
// (kPlanRk8v4), which the public KvCacheStorage enum did not distinguish from plain INT8 before
// this fork added RotatedInt8KeyInt4ValueGroup64. cache.storage is still populated here (from
// plan) so shared consumers below (DeviceCache, verify_cache) can dispatch on it uniformly
// regardless of which overload built the HostCache.
HostCache make_cache(const Geometry& geometry, const CachePlan& plan, std::int32_t max_context,
                     std::uint32_t seed) {
    const ops::D256KVCacheProfile profile =
        plan.packed_keys ? ops::d256_kv_cache_profile(KvCacheStorage::RotatedLloyd4KeyInt4Value)
                         : ops::d256_kv_cache_profile(plan.dtype, plan.value_code_dtype);
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v        = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    const KvCacheStorage storage = cache_plan_storage(plan);
    HostCache cache{.geometry         = geometry,
                    .dtype            = plan.dtype,
                    .value_code_dtype = plan.value_code_dtype,
                    .storage          = storage,
                    .max_context      = max_context,
                    .logical_capacity = logical_capacity,
                    .profile          = profile};
    if (plan.dtype == DType::BF16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_fp16 = to_bf16_bits(logical_v);
        return cache;
    }

    cache.k_scale.assign(plane_scale_elements(geometry, logical_capacity,
                                              profile.scale_leading_extent),
                         0);
    cache.v_scale.assign(plane_scale_elements(geometry, logical_capacity,
                                              profile.value_scale_leading_extent),
                         0);
    cache.logical_k_quantized.assign(elements, 0.0f);
    if (plan.dtype == DType::FP8_E4M3FN) {
        cache.k_fp8.assign(elements, 0);
        cache.v_fp8.assign(elements, 0);
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, 0);
                const std::size_t scale =
                    fp8_scale_index(geometry, logical_capacity, head, position);
                encode_rotated_fp8_key_row(logical_k, code, cache.k_fp8, code, cache.k_scale, scale,
                                           cache.logical_k_quantized, code);
                std::array<float, kHeadDim> v_row{};
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    v_row[static_cast<std::size_t>(d)] =
                        logical_v[code + static_cast<std::size_t>(d)];
                }
                encode_fp8_row(v_row, cache.v_fp8, code, cache.v_scale, scale);
            }
        }
        return cache;
    }

    if (plan.packed_keys) {
        cache.k_packed.assign(elements / 2, 0);
    } else {
        cache.k_i8.assign(elements, 0);
    }
    if (profile.packed_int4_values()) {
        cache.v_packed.assign(elements / 2, 0);
    } else {
        cache.v_i8.assign(elements, 0);
    }
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::size_t code  = cache_index(geometry, logical_capacity, head, position, 0);
            const std::size_t scale = scale_index(geometry, logical_capacity, head, position, 0);
            if (plan.packed_keys) {
                encode_rotated_lloyd4_key_row(
                    logical_k, code, cache.k_packed,
                    packed_cache_index(geometry, logical_capacity, head, position, 0),
                    cache.k_scale, scale, cache.logical_k_quantized, code);
            } else {
                encode_rotated_key_row(logical_k, code, cache.k_i8, code, cache.k_scale, scale,
                                       cache.logical_k_quantized, code);
            }
            if (profile.packed_int4_values()) {
                for (std::int32_t group = 0; group < kRk8ValueGroups; ++group) {
                    const std::int32_t d = group * kRk8ValueGroup;
                    const std::size_t group_code =
                        cache_index(geometry, logical_capacity, head, position, d);
                    const std::size_t group_packed =
                        packed_cache_index(geometry, logical_capacity, head, position,
                                           group * (kRk8ValueGroup / 2));
                    const std::size_t group_scale =
                        plane_scale_index(geometry, logical_capacity, head, position, group,
                                          profile.value_scale_leading_extent);
                    encode_group_i4(logical_v, group_code, cache.v_packed, group_packed,
                                    cache.v_scale, group_scale);
                }
            } else {
                for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                    const std::int32_t d = group * kQuantGroup;
                    const std::size_t group_code =
                        cache_index(geometry, logical_capacity, head, position, d);
                    const std::size_t group_scale =
                        scale_index(geometry, logical_capacity, head, position, group);
                    encode_group(logical_v, group_code, cache.v_i8, group_code, cache.v_scale,
                                group_scale);
                }
            }
        }
    }
    return cache;
}

// KvCacheStorage overload: covers every publicly-selectable profile except rk8v4 (this fork's
// own RotatedInt8KeyInt4ValueGroup64, which only the CachePlan overload above builds). dtype,
// value_code_dtype and profile are populated too for the three profiles d256_kv_cache_profile
// covers (BFloat16/Int8Group64/Fp8E4M3Row256), so DeviceCache/verify_cache work identically
// whichever overload produced the HostCache; nvfp4/k8v4 never read those fields.
HostCache make_cache(const Geometry& geometry, KvCacheStorage storage, std::int32_t max_context,
                     std::uint32_t seed, float v_amplitude = 1.0f) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v =
        make_bf16_values(elements, seed + 1u, -v_amplitude, v_amplitude);

    if (storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        HostCache cache{.geometry         = geometry,
                        .storage          = storage,
                        .max_context      = max_context,
                        .logical_capacity = logical_capacity};
        if (storage == KvCacheStorage::Nvfp4Group16) {
            const std::size_t code_elements =
                static_cast<std::size_t>(kNvfp4CodeBytes) * logical_capacity * geometry.kv_heads;
            const std::size_t scale_elements =
                static_cast<std::size_t>(kNvfp4QuantGroups) * logical_capacity * geometry.kv_heads;
            cache.k_nvfp4.assign(code_elements, 0);
            cache.v_nvfp4.assign(code_elements, 0);
            cache.k_nvfp4_scale.assign(scale_elements, 0);
            cache.v_nvfp4_scale.assign(scale_elements, 0);
            cache.rotated_k_quantized.assign(elements, 0.0f);
            cache.rotated_v_quantized.assign(elements, 0.0f);
            for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
                for (std::int32_t position = 0; position < logical_capacity; ++position) {
                    const std::size_t source =
                        cache_index(geometry, logical_capacity, head, position, 0);
                    const std::size_t code = logical_plane_index(
                        kNvfp4CodeBytes, geometry, logical_capacity, head, position, 0);
                    const std::size_t scale = logical_plane_index(
                        kNvfp4QuantGroups, geometry, logical_capacity, head, position, 0);
                    encode_nvfp4_rotated_row(logical_k, source, cache.k_nvfp4, code,
                                             cache.k_nvfp4_scale, scale, cache.rotated_k_quantized,
                                             source);
                    encode_nvfp4_rotated_row(logical_v, source, cache.v_nvfp4, code,
                                             cache.v_nvfp4_scale, scale, cache.rotated_v_quantized,
                                             source);
                }
            }
            return cache;
        }

        const std::size_t fp8_scales = fp8_scale_elements(geometry, logical_capacity);
        const std::size_t v_codes =
            static_cast<std::size_t>(kNvfp4CodeBytes) * logical_capacity * geometry.kv_heads;
        const std::size_t v_scales =
            static_cast<std::size_t>(kNvfp4QuantGroups) * logical_capacity * geometry.kv_heads;
        cache.k_fp8.assign(elements, 0);
        cache.k_scale.assign(fp8_scales, 0);
        cache.v_nvfp4.assign(v_codes, 0);
        cache.v_nvfp4_scale.assign(v_scales, 0);
        cache.rotated_k_quantized.assign(elements, 0.0f);
        cache.rotated_v_quantized.assign(elements, 0.0f);
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                const std::size_t source =
                    cache_index(geometry, logical_capacity, head, position, 0);
                const std::size_t k_scale =
                    fp8_scale_index(geometry, logical_capacity, head, position);
                const std::size_t v_code  = logical_plane_index(kNvfp4CodeBytes, geometry,
                                                                logical_capacity, head, position, 0);
                const std::size_t v_scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, logical_capacity, head, position, 0);
                encode_fp8_rotated_row(logical_k, source, cache.k_fp8, source, cache.k_scale,
                                       k_scale, cache.rotated_k_quantized, source);
                encode_nvfp4_rotated_row(logical_v, source, cache.v_nvfp4, v_code,
                                         cache.v_nvfp4_scale, v_scale, cache.rotated_v_quantized,
                                         source);
            }
        }
        return cache;
    }

    const ops::D256KVCacheProfile profile = ops::d256_kv_cache_profile(storage);
    const DType dtype = storage == KvCacheStorage::BFloat16        ? DType::BF16
                        : storage == KvCacheStorage::Fp8E4M3Row256 ? DType::FP8_E4M3FN
                                                                    : DType::I8;
    HostCache cache{.geometry         = geometry,
                    .dtype            = dtype,
                    .value_code_dtype = dtype,
                    .storage          = storage,
                    .max_context      = max_context,
                    .logical_capacity = logical_capacity,
                    .profile          = profile};
    if (storage == KvCacheStorage::BFloat16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_fp16 = to_bf16_bits(logical_v);
        return cache;
    }

    cache.k_scale.assign(plane_scale_elements(geometry, logical_capacity,
                                              profile.scale_leading_extent),
                         0);
    cache.v_scale.assign(plane_scale_elements(geometry, logical_capacity,
                                              profile.value_scale_leading_extent),
                         0);
    cache.logical_k_quantized.assign(elements, 0.0f);
    if (storage == KvCacheStorage::Fp8E4M3Row256) {
        cache.k_fp8.assign(elements, 0);
        cache.v_fp8.assign(elements, 0);
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::int32_t position = 0; position < logical_capacity; ++position) {
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, 0);
                const std::size_t scale =
                    fp8_scale_index(geometry, logical_capacity, head, position);
                encode_rotated_fp8_key_row(logical_k, code, cache.k_fp8, code, cache.k_scale, scale,
                                           cache.logical_k_quantized, code);
                std::array<float, kHeadDim> v_row{};
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    v_row[static_cast<std::size_t>(d)] =
                        logical_v[code + static_cast<std::size_t>(d)];
                }
                encode_fp8_row(v_row, cache.v_fp8, code, cache.v_scale, scale);
            }
        }
        return cache;
    }

    // storage == KvCacheStorage::Int8Group64 (rk8v4 is never selected via this overload).
    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::size_t code  = cache_index(geometry, logical_capacity, head, position, 0);
            const std::size_t scale = scale_index(geometry, logical_capacity, head, position, 0);
            encode_rotated_key_row(logical_k, code, cache.k_i8, code, cache.k_scale, scale,
                                   cache.logical_k_quantized, code);
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d = group * kQuantGroup;
                const std::size_t group_code =
                    cache_index(geometry, logical_capacity, head, position, d);
                const std::size_t group_scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                encode_group(logical_v, group_code, cache.v_i8, group_code, cache.v_scale,
                            group_scale);
            }
        }
    }
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.storage == KvCacheStorage::BFloat16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_fp16[target] = f32_to_bf16(v[source]);
                }
                continue;
            }

            const std::size_t source = kv_input_index(geometry, head, 0, token);
            const std::size_t target =
                cache_index(geometry, cache.logical_capacity, head, position, 0);
            if (cache.storage == KvCacheStorage::Nvfp4Group16) {
                const std::size_t code = logical_plane_index(
                    kNvfp4CodeBytes, geometry, cache.logical_capacity, head, position, 0);
                const std::size_t scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, cache.logical_capacity, head, position, 0);
                encode_nvfp4_rotated_row(k, source, cache.k_nvfp4, code, cache.k_nvfp4_scale, scale,
                                         cache.rotated_k_quantized, target);
                encode_nvfp4_rotated_row(v, source, cache.v_nvfp4, code, cache.v_nvfp4_scale, scale,
                                         cache.rotated_v_quantized, target);
                continue;
            }
            if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
                const std::size_t k_scale =
                    fp8_scale_index(geometry, cache.logical_capacity, head, position);
                const std::size_t v_code = logical_plane_index(
                    kNvfp4CodeBytes, geometry, cache.logical_capacity, head, position, 0);
                const std::size_t v_scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, cache.logical_capacity, head, position, 0);
                encode_fp8_rotated_row(k, source, cache.k_fp8, target, cache.k_scale, k_scale,
                                       cache.rotated_k_quantized, target);
                encode_nvfp4_rotated_row(v, source, cache.v_nvfp4, v_code, cache.v_nvfp4_scale,
                                         v_scale, cache.rotated_v_quantized, target);
                continue;
            }
            if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
                const std::size_t scale =
                    fp8_scale_index(geometry, cache.logical_capacity, head, position);
                encode_rotated_fp8_key_row(k, source, cache.k_fp8, target, cache.k_scale, scale,
                                           cache.logical_k_quantized, target);
                std::array<float, kHeadDim> v_row{};
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    v_row[static_cast<std::size_t>(d)] =
                        v[kv_input_index(geometry, head, d, token)];
                }
                encode_fp8_row(v_row, cache.v_fp8, target, cache.v_scale, scale);
                continue;
            }
            const std::size_t scale =
                scale_index(geometry, cache.logical_capacity, head, position, 0);
            if (cache.packed_keys()) {
                encode_rotated_lloyd4_key_row(
                    k, source, cache.k_packed,
                    packed_cache_index(geometry, cache.logical_capacity, head, position, 0),
                    cache.k_scale, scale, cache.logical_k_quantized, target);
            } else {
                encode_rotated_key_row(k, source, cache.k_i8, target, cache.k_scale, scale,
                                       cache.logical_k_quantized, target);
            }
            if (cache.packed_values()) {
                for (std::int32_t group = 0; group < kRk8ValueGroups; ++group) {
                    const std::int32_t d = group * kRk8ValueGroup;
                    const std::size_t group_source = kv_input_index(geometry, head, d, token);
                    const std::size_t group_target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    const std::size_t group_packed =
                        packed_cache_index(geometry, cache.logical_capacity, head, position,
                                           group * (kRk8ValueGroup / 2));
                    const std::size_t group_scale =
                        plane_scale_index(geometry, cache.logical_capacity, head, position, group,
                                          cache.v_scale_extent());
                    encode_group_i4(v, group_source, cache.v_packed, group_packed, cache.v_scale,
                                    group_scale);
                }
            } else {
                for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                    const std::int32_t d           = group * kQuantGroup;
                    const std::size_t group_source = kv_input_index(geometry, head, d, token);
                    const std::size_t group_target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    const std::size_t group_scale =
                        scale_index(geometry, cache.logical_capacity, head, position, group);
                    encode_group(v, group_source, cache.v_i8, group_target, cache.v_scale,
                                 group_scale);
                }
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, std::int32_t head, std::int32_t position,
                   std::int32_t d) {
    const std::size_t code = cache_index(cache.geometry, cache.logical_capacity, head, position, d);
    if (cache.storage == KvCacheStorage::BFloat16) {
        return key ? static_cast<double>(bf16_to_f32(cache.k_bf16[code]))
                   : static_cast<double>(bf16_to_f32(cache.v_fp16[code]));
    }

    if (key) { return static_cast<double>(cache.logical_k_quantized[code]); }

    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        const std::size_t scale =
            fp8_scale_index(cache.geometry, cache.logical_capacity, head, position);
        return static_cast<double>(decode_e4m3fn(cache.v_fp8[code]) *
                                   f16_bits_to_f32(cache.v_scale[scale]));
    }

    if (cache.packed_values()) {
        const std::size_t byte =
            packed_cache_index(cache.geometry, cache.logical_capacity, head, position, d / 2);
        const std::size_t scale =
            plane_scale_index(cache.geometry, cache.logical_capacity, head, position,
                              d / kRk8ValueGroup, cache.v_scale_extent());
        const float decoded =
            static_cast<float>(unpack_int4_code(cache.v_packed[byte], d & 1)) *
            f16_bits_to_f32(cache.v_scale[scale]);
        return static_cast<double>(decoded);
    }

    const std::size_t scale =
        scale_index(cache.geometry, cache.logical_capacity, head, position, d / kQuantGroup);
    const float decoded =
        static_cast<float>(cache.v_i8[code]) * f16_bits_to_f32(cache.v_scale[scale]);
    return static_cast<double>(decoded);
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));
    if (cache.storage == KvCacheStorage::Nvfp4Group16 ||
        cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        const std::vector<float> q_represented =
            cache.storage == KvCacheStorage::Fp8KeyNvfp4Value
                ? quantize_fp8_rotated_queries(q, geometry, tokens)
                : represent_f16_rotated_queries(q, geometry, tokens);
        naive_dense_softmax_attention(
            op_geometry(geometry), tokens, positions.back() + 1,
            static_cast<double>(kAttentionScale),
            [&](int d, int head, int token) {
                return static_cast<double>(q_represented[q_index(geometry, head, d, token)]);
            },
            [&](int d, int head, int position) {
                const float represented = cache.rotated_k_quantized[cache_index(
                    geometry, cache.logical_capacity, head, position, d)];
                return static_cast<double>(cache.storage == KvCacheStorage::Nvfp4Group16
                                               ? round_to_f16(represented)
                                               : represented);
            },
            [&](int d, int head, int position) {
                return static_cast<double>(round_to_f16(cache.rotated_v_quantized[cache_index(
                    geometry, cache.logical_capacity, head, position, d)]));
            },
            [&](int token, int position) {
                return position <= positions[static_cast<std::size_t>(token)];
            },
            [&](int d, int head, int token, double value) {
                output[q_index(geometry, head, d, token)] = value;
            });
        for (std::int32_t token = 0; token < tokens; ++token) {
            for (std::int32_t head = 0; head < geometry.q_heads; ++head) {
                std::array<double, kHeadDim> row{};
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    row[static_cast<std::size_t>(d)] = output[q_index(geometry, head, d, token)];
                }
                normalized_hadamard_d256(row);
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    output[q_index(geometry, head, d, token)] = row[static_cast<std::size_t>(d)];
                }
            }
        }
        return output;
    }

    naive_dense_softmax_attention(
        op_geometry(geometry), tokens, positions.back() + 1, static_cast<double>(kAttentionScale),
        [&](int d, int head, int token) {
            return static_cast<double>(q[q_index(geometry, head, d, token)]);
        },
        [&](int d, int head, int position) { return cache_value(cache, true, head, position, d); },
        [&](int d, int head, int position) { return cache_value(cache, false, head, position, d); },
        [&](int token, int position) {
            return position <= positions[static_cast<std::size_t>(token)];
        },
        [&](int d, int head, int token, double value) {
            output[q_index(geometry, head, d, token)] = value;
        });
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

class DeviceCache {
public:
    DeviceCache(const HostCache& cache, MappingPattern mapping)
        : geometry_(cache.geometry), storage_(cache.storage), layout_(test_cache_layout(storage_)),
          max_context_(cache.max_context), logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          k_scale_extent_(layout_.key.scale_extent), v_scale_extent_(layout_.value.scale_extent),
          k_code_elements_(
              physical_plane_elements(layout_.key.code_extent, geometry_, physical_pages_)),
          v_code_elements_(
              physical_plane_elements(layout_.value.code_extent, geometry_, physical_pages_)),
          k_scale_elements_(
              physical_plane_elements(layout_.key.scale_extent, geometry_, physical_pages_)),
          v_scale_elements_(
              physical_plane_elements(layout_.value.scale_extent, geometry_, physical_pages_)),
          k_(k_code_elements_ * dtype_size(layout_.key.code_dtype)),
          v_(v_code_elements_ * dtype_size(layout_.value.code_dtype)),
          k_scale_(layout_.key.scale_extent == 0
                       ? 1
                       : k_scale_elements_ * dtype_size(layout_.key.scale_dtype)),
          v_scale_(layout_.value.scale_extent == 0
                       ? 1
                       : v_scale_elements_ * dtype_size(layout_.value.scale_dtype)),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
        if (storage_ == KvCacheStorage::BFloat16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_fp16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::Int8Group64) {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, k_scale_extent_, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            const auto v_physical =
                scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, v_scale_extent_, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, k_scale_extent_, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            const auto v_physical =
                scatter_paged(cache.v_packed, layout_.value.code_extent, geometry_,
                              logical_capacity_, block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, v_scale_extent_, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            v_.copy_from_host(v_physical.data(), v_physical.size());
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
            const auto k_physical =
                scatter_paged(cache.k_packed, layout_.key.code_extent, geometry_,
                              logical_capacity_, block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, k_scale_extent_, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            const auto v_physical =
                scatter_paged(cache.v_packed, layout_.value.code_extent, geometry_,
                              logical_capacity_, block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, v_scale_extent_, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            v_.copy_from_host(v_physical.data(), v_physical.size());
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            const auto k_physical =
                scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_fp8, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            v_.copy_from_host(v_physical.data(), v_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            const auto k_physical =
                scatter_paged(cache.k_fp8, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            v_.copy_from_host(v_physical.data(), v_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size());
        } else {
            const auto k_physical =
                scatter_paged(cache.k_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_nvfp4_scale, kNvfp4QuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size());
            v_.copy_from_host(v_physical.data(), v_physical.size());
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size());
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size());
        }
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        result.k_pages = Tensor(
            k_.data(), layout_.key.code_dtype,
            {layout_.key.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages = Tensor(
            v_.data(), layout_.value.code_dtype,
            {layout_.value.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_table  = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.storage      = storage_;
        if (layout_.key.scale_extent != 0) {
            result.k_scale_pages = Tensor(
                k_scale_.data(), layout_.key.scale_dtype,
                {layout_.key.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        }
        if (layout_.value.scale_extent != 0) {
            result.v_scale_pages = Tensor(v_scale_.data(), layout_.value.scale_dtype,
                                          {layout_.value.scale_extent, kPagedKVPageSize,
                                           geometry_.kv_heads, physical_pages_});
        }
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .storage       = direct.storage,
        };
    }

    // dtype/value_code_dtype/profile are populated for the four profiles d256_kv_cache_profile
    // covers (BFloat16/Int8Group64/Fp8E4M3Row256/RotatedInt8KeyInt4ValueGroup64) so verify_cache's
    // packed_values() check still works; nvfp4/k8v4 snapshots leave them at their harmless
    // defaults since verify_cache never reads them for those two storages.
    HostCache snapshot() const {
        HostCache cache{.geometry = geometry_, .storage = storage_, .max_context = max_context_,
                        .logical_capacity = logical_capacity_};
        if (storage_ != KvCacheStorage::Nvfp4Group16 && storage_ != KvCacheStorage::Fp8KeyNvfp4Value) {
            const ops::D256KVCacheProfile profile = ops::d256_kv_cache_profile(storage_);
            cache.dtype            = profile.key_code_dtype;
            cache.value_code_dtype = profile.value_code_dtype;
            cache.profile          = profile;
        }
        if (storage_ == KvCacheStorage::BFloat16) {
            const auto k_physical = copy_from_guarded<std::uint16_t>(k_, k_code_elements_);
            const auto v_physical = copy_from_guarded<std::uint16_t>(v_, v_code_elements_);
            cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_fp16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::Int8Group64) {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::int8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
            cache.k_i8             = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.v_i8             = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, k_scale_extent_, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, v_scale_extent_, geometry_,
                                                        logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, k_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            cache.k_i8    = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_,
                                                      logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, k_scale_extent_, geometry_,
                                                        logical_capacity_, block_table_host_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
            cache.v_packed = gather_paged<std::uint8_t>(v_physical, layout_.value.code_extent,
                                                        geometry_, logical_capacity_,
                                                        block_table_host_);
            cache.v_scale  = gather_paged<std::uint16_t>(vs_physical, v_scale_extent_, geometry_,
                                                         logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
            cache.k_packed = gather_paged<std::uint8_t>(k_physical, layout_.key.code_extent,
                                                        geometry_, logical_capacity_,
                                                        block_table_host_);
            cache.k_scale  = gather_paged<std::uint16_t>(ks_physical, k_scale_extent_, geometry_,
                                                         logical_capacity_, block_table_host_);
            cache.v_packed = gather_paged<std::uint8_t>(v_physical, layout_.value.code_extent,
                                                        geometry_, logical_capacity_,
                                                        block_table_host_);
            cache.v_scale  = gather_paged<std::uint16_t>(vs_physical, v_scale_extent_, geometry_,
                                                         logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_);
            cache.k_fp8            = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_fp8            = gather_paged<std::uint8_t>(v_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, k_scale_extent_, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, v_scale_extent_, geometry_,
                                                        logical_capacity_, block_table_host_);
        } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_);
            cache.k_fp8            = gather_paged<std::uint8_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_nvfp4 = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeBytes, geometry_,
                                                       logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kFp8QuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_nvfp4_scale = gather_paged<std::uint8_t>(
                vs_physical, kNvfp4QuantGroups, geometry_, logical_capacity_, block_table_host_);
        } else {
            const auto k_physical  = copy_from_guarded<std::uint8_t>(k_, k_code_elements_);
            const auto v_physical  = copy_from_guarded<std::uint8_t>(v_, v_code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint8_t>(k_scale_, k_scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_);
            cache.k_nvfp4       = gather_paged<std::uint8_t>(k_physical, kNvfp4CodeBytes, geometry_,
                                                             logical_capacity_, block_table_host_);
            cache.v_nvfp4       = gather_paged<std::uint8_t>(v_physical, kNvfp4CodeBytes, geometry_,
                                                             logical_capacity_, block_table_host_);
            cache.k_nvfp4_scale = gather_paged<std::uint8_t>(
                ks_physical, kNvfp4QuantGroups, geometry_, logical_capacity_, block_table_host_);
            cache.v_nvfp4_scale = gather_paged<std::uint8_t>(
                vs_physical, kNvfp4QuantGroups, geometry_, logical_capacity_, block_table_host_);
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (layout_.key.scale_extent != 0) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
        }
        if (layout_.value.scale_extent != 0) {
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        failures += block_table_.verify_guards((label + " block-table").c_str());
        failures +=
            verify_exact((label + " block-table unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()),
                         block_table_host_);
        return failures;
    }

private:
    Geometry geometry_;
    KvCacheStorage storage_;
    TestCacheLayout layout_;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::int32_t k_scale_extent_;
    std::int32_t v_scale_extent_;
    std::size_t k_code_elements_;
    std::size_t v_code_elements_;
    std::size_t k_scale_elements_;
    std::size_t v_scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping)
        : geometry_(rows.front().geometry), storage_(rows.front().storage),
          layout_(test_cache_layout(storage_)), rows_(rows.size()),
          logical_capacity_(rows.front().logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(mapping == MappingPattern::Fragmented
                              ? 2 * static_cast<std::int32_t>(rows_) * logical_pages_ + 1
                              : static_cast<std::int32_t>(rows_) * logical_pages_),
          block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_)),
          k_scale_extent_(layout_.key.scale_extent), v_scale_extent_(layout_.value.scale_extent),
          k_code_elements_(
              physical_plane_elements(layout_.key.code_extent, geometry_, physical_pages_)),
          v_code_elements_(
              physical_plane_elements(layout_.value.code_extent, geometry_, physical_pages_)),
          k_scale_elements_(
              physical_plane_elements(layout_.key.scale_extent, geometry_, physical_pages_)),
          v_scale_elements_(
              physical_plane_elements(layout_.value.scale_extent, geometry_, physical_pages_)),
          k_(k_code_elements_ * dtype_size(layout_.key.code_dtype)),
          v_(v_code_elements_ * dtype_size(layout_.value.code_dtype)),
          k_scale_(layout_.key.scale_extent == 0
                       ? 1
                       : k_scale_elements_ * dtype_size(layout_.key.scale_dtype)),
          v_scale_(layout_.value.scale_extent == 0
                       ? 1
                       : v_scale_elements_ * dtype_size(layout_.value.scale_dtype)),
          block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
        for (std::size_t row = 0; row < rows_; ++row) {
            const HostCache& cache = rows[row];
            if (cache.geometry.q_heads != geometry_.q_heads ||
                cache.geometry.kv_heads != geometry_.kv_heads || cache.storage != storage_ ||
                cache.logical_capacity != logical_capacity_) {
                throw std::invalid_argument("batch cache rows must share one physical geometry");
            }
            for (std::int32_t logical = 0; logical < logical_pages_; ++logical) {
                const std::int32_t linear =
                    static_cast<std::int32_t>(row) * logical_pages_ + logical;
                block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                    mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
            }
        }
        block_tables_.copy_from_host(block_tables_host_.data(),
                                     block_tables_host_.size() * sizeof(std::int32_t));
        upload_rows(rows);
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        result.k_pages = Tensor(
            k_.data(), layout_.key.code_dtype,
            {layout_.key.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages = Tensor(
            v_.data(), layout_.value.code_dtype,
            {layout_.value.code_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_tables = Tensor(block_tables_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(rows_)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.storage      = storage_;
        if (layout_.key.scale_extent != 0) {
            result.k_scale_pages = Tensor(
                k_scale_.data(), layout_.key.scale_dtype,
                {layout_.key.scale_extent, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        }
        if (layout_.value.scale_extent != 0) {
            result.v_scale_pages = Tensor(v_scale_.data(), layout_.value.scale_dtype,
                                          {layout_.value.scale_extent, kPagedKVPageSize,
                                           geometry_.kv_heads, physical_pages_});
        }
        return result;
    }

    int verify(const std::string& label, std::span<const HostCache> expected) const {
        if (expected.size() != rows_) {
            std::cerr << label << ": expected cache row count mismatch\n";
            return 1;
        }
        int failures = 0;
        if (storage_ == KvCacheStorage::BFloat16) {
            std::vector<std::uint16_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint16_t> expected_v(v_code_elements_, 0);
            scatter_bf16_rows(expected, expected_k, expected_v);
            failures +=
                verify_exact((label + " cache-k").c_str(),
                             copy_from_guarded<std::uint16_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v").c_str(),
                             copy_from_guarded<std::uint16_t>(v_, v_code_elements_), expected_v);
        } else if (storage_ == KvCacheStorage::Int8Group64) {
            std::vector<std::int8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].v_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::int8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        } else if (storage_ == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
                   storage_ == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].v_packed, layout_.value.code_extent, geometry_,
                                   logical_capacity_, table, expected_v);
                scatter_paged_into(expected[row].v_scale, v_scale_extent_, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        } else if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, k_scale_extent_, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, v_scale_extent_, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        } else if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(k_scale_elements_, 0);
            std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_fp8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_nvfp4, kNvfp4CodeBytes, geometry_,
                                   logical_capacity_, table, expected_v);
                scatter_paged_into(expected[row].k_scale, kFp8QuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, k_scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        } else {
            std::vector<std::uint8_t> expected_k(k_code_elements_, 0);
            std::vector<std::uint8_t> expected_v(v_code_elements_, 0);
            std::vector<std::uint8_t> expected_ks(k_scale_elements_, 0);
            std::vector<std::uint8_t> expected_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_nvfp4, kNvfp4CodeBytes, geometry_,
                                   logical_capacity_, table, expected_k);
                scatter_paged_into(expected[row].v_nvfp4, kNvfp4CodeBytes, geometry_,
                                   logical_capacity_, table, expected_v);
                scatter_paged_into(expected[row].k_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::uint8_t>(k_, k_code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::uint8_t>(v_, v_code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint8_t>(k_scale_, k_scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint8_t>(v_scale_, v_scale_elements_),
                                     expected_vs);
        }
        failures +=
            verify_exact((label + " block tables unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                         block_tables_host_);
        failures += k_.verify_guards((label + " cache-k guard").c_str());
        failures += v_.verify_guards((label + " cache-v guard").c_str());
        if (layout_.key.scale_extent != 0) {
            failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
        }
        if (layout_.value.scale_extent != 0) {
            failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
        }
        failures += block_tables_.verify_guards((label + " block tables guard").c_str());
        return failures;
    }

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const {
        return std::span<const std::int32_t>(block_tables_host_.data() +
                                                 row * static_cast<std::size_t>(logical_pages_),
                                             static_cast<std::size_t>(logical_pages_));
    }

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const {
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
            scatter_paged_into(rows[row].v_fp16, kHeadDim, geometry_, logical_capacity_, table, v);
        }
    }

    void upload_rows(std::span<const HostCache> rows) {
        if (storage_ == KvCacheStorage::BFloat16) {
            std::vector<std::uint16_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint16_t> physical_v(v_code_elements_, 0);
            scatter_bf16_rows(rows, physical_k, physical_v);
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::Int8Group64) {
            std::vector<std::int8_t> physical_k(k_code_elements_, 0);
            std::vector<std::int8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_v);
                scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_,
                                   table, physical_ks);
                scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_,
                                   table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
            std::vector<std::int8_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_packed, layout_.value.code_extent, geometry_,
                                   logical_capacity_, table, physical_v);
                scatter_paged_into(rows[row].k_scale, k_scale_extent_, geometry_,
                                   logical_capacity_, table, physical_ks);
                scatter_paged_into(rows[row].v_scale, v_scale_extent_, geometry_,
                                   logical_capacity_, table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
            v_.copy_from_host(physical_v.data(), physical_v.size());
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
            std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_packed, layout_.key.code_extent, geometry_,
                                   logical_capacity_, table, physical_k);
                scatter_paged_into(rows[row].v_packed, layout_.value.code_extent, geometry_,
                                   logical_capacity_, table, physical_v);
                scatter_paged_into(rows[row].k_scale, k_scale_extent_, geometry_,
                                   logical_capacity_, table, physical_ks);
                scatter_paged_into(rows[row].v_scale, v_scale_extent_, geometry_,
                                   logical_capacity_, table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size());
            v_.copy_from_host(physical_v.data(), physical_v.size());
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::Fp8E4M3Row256) {
            std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint16_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_v);
                scatter_paged_into(rows[row].k_scale, k_scale_extent_, geometry_,
                                   logical_capacity_, table, physical_ks);
                scatter_paged_into(rows[row].v_scale, v_scale_extent_, geometry_,
                                   logical_capacity_, table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size());
            v_.copy_from_host(physical_v.data(), physical_v.size());
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
            return;
        }
        if (storage_ == KvCacheStorage::Fp8KeyNvfp4Value) {
            std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
            std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
            std::vector<std::uint16_t> physical_ks(k_scale_elements_, 0);
            std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(rows[row].k_fp8, kHeadDim, geometry_, logical_capacity_, table,
                                   physical_k);
                scatter_paged_into(rows[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                                   table, physical_v);
                scatter_paged_into(rows[row].k_scale, kFp8QuantGroups, geometry_, logical_capacity_,
                                   table, physical_ks);
                scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                                   logical_capacity_, table, physical_vs);
            }
            k_.copy_from_host(physical_k.data(), physical_k.size());
            v_.copy_from_host(physical_v.data(), physical_v.size());
            k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(physical_vs.data(), physical_vs.size());
            return;
        }
        std::vector<std::uint8_t> physical_k(k_code_elements_, 0);
        std::vector<std::uint8_t> physical_v(v_code_elements_, 0);
        std::vector<std::uint8_t> physical_ks(k_scale_elements_, 0);
        std::vector<std::uint8_t> physical_vs(v_scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, physical_k);
            scatter_paged_into(rows[row].v_nvfp4, kNvfp4CodeBytes, geometry_, logical_capacity_,
                               table, physical_v);
            scatter_paged_into(rows[row].k_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, physical_ks);
            scatter_paged_into(rows[row].v_nvfp4_scale, kNvfp4QuantGroups, geometry_,
                               logical_capacity_, table, physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size());
        v_.copy_from_host(physical_v.data(), physical_v.size());
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size());
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size());
    }

    Geometry geometry_;
    KvCacheStorage storage_;
    TestCacheLayout layout_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::int32_t k_scale_extent_;
    std::int32_t v_scale_extent_;
    std::size_t k_code_elements_;
    std::size_t v_code_elements_;
    std::size_t k_scale_elements_;
    std::size_t v_scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_tables_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected,
                 bool verify_private_key_representation) {
    int failures = 0;
    if (expected.storage == KvCacheStorage::BFloat16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_fp16, expected.v_fp16);
    } else if (expected.storage == KvCacheStorage::Int8Group64 ||
               expected.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
               expected.storage == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
        if (verify_private_key_representation && expected.packed_keys()) {
            failures +=
                verify_exact((label + " cache-k-code").c_str(), got.k_packed, expected.k_packed);
            failures +=
                verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        } else if (verify_private_key_representation) {
            failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
            failures +=
                verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        }
        if (expected.packed_values()) {
            failures +=
                verify_exact((label + " cache-v-code").c_str(), got.v_packed, expected.v_packed);
        } else {
            failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        }
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    } else if (expected.storage == KvCacheStorage::Fp8E4M3Row256) {
        if (verify_private_key_representation) {
            failures += verify_exact((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
            failures +=
                verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        }
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_fp8, expected.v_fp8);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    } else if (expected.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        if (verify_private_key_representation) {
            failures += verify_exact((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
            failures +=
                verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        }
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_nvfp4_scale,
                                 expected.v_nvfp4_scale);
    } else {
        if (verify_private_key_representation) {
            failures +=
                verify_exact((label + " cache-k-code").c_str(), got.k_nvfp4, expected.k_nvfp4);
            failures += verify_exact((label + " cache-k-scale").c_str(), got.k_nvfp4_scale,
                                     expected.k_nvfp4_scale);
        }
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_nvfp4_scale,
                                 expected.v_nvfp4_scale);
    }
    return failures;
}

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

const char* cache_name(const CachePlan& plan) {
    if (plan.dtype == DType::BF16) return "bf16";
    if (plan.packed_keys) return "rk4v4";
    if (plan.dtype == DType::I8) return plan.packed_values() ? "rk8v4" : "int8-g64";
    return "fp8-e4m3fn-row256";
}

ReductionCriterion attention_criterion(const CachePlan& plan) {
    if (plan.dtype == DType::BF16) return kAttentionBf16Criterion;
    if (plan.packed_keys) return kAttentionRk4v4Criterion;
    if (plan.dtype == DType::I8) {
        return plan.packed_values() ? kAttentionRk8v4Criterion : kAttentionInt8Criterion;
    }
    return kAttentionFp8Criterion;
}

const char* cache_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-g64";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rk8v4";
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return "rk4v4";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3fn-row256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4-g16";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

ReductionCriterion attention_criterion(KvCacheStorage storage) {
    if (storage == KvCacheStorage::BFloat16) return kAttentionBf16Criterion;
    if (storage == KvCacheStorage::Int8Group64) return kAttentionInt8Criterion;
    if (storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) return kAttentionRk8v4Criterion;
    if (storage == KvCacheStorage::RotatedLloyd4KeyInt4Value) return kAttentionRk4v4Criterion;
    if (storage == KvCacheStorage::Fp8E4M3Row256) return kAttentionFp8Criterion;
    if (storage == KvCacheStorage::Nvfp4Group16) return kAttentionNvfp4Criterion;
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) return kAttentionK8V4Criterion;
    throw std::logic_error("unregistered causal-attention test storage");
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    if (std::getenv("NINFER_DUMP_ATTN_STATS") != nullptr) {
        const ReductionStats s = compute_reduction_stats(
            actual.data(), reference.data(), static_cast<std::int64_t>(actual.size()));
        const double limit = gross_error_limit(s, criterion);
        std::printf("STATS\t%s\tl2=%.4e\tmax_abs=%.4e\tmax_ref=%.4e\tlimit=%.4e\tuse=%.3f\n",
                    label.c_str(), s.relative_l2, s.maximum_absolute_error,
                    s.maximum_absolute_reference, limit, s.maximum_absolute_error / limit);
    }
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

std::string case_label(const char* entry, const Geometry& geometry, const CachePlan& plan,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(plan) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max) +
           (test_case.graph_replay ? " graph-replay" : "");
}

std::string case_label(const char* entry, const Geometry& geometry, KvCacheStorage storage,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(storage) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max) +
           (test_case.graph_replay ? " graph-replay" : "");
}

template <class Launch>
void launch_attention_case(Launch&& launch, bool graph_replay) {
    if (!graph_replay) {
        launch(nullptr);
        cuda_synchronize();
        return;
    }

    // Prime route-local CUDA function attributes before stream capture. The append route writes
    // the same represented values to the same positions, so this does not change its oracle.
    launch(nullptr);
    cuda_synchronize();

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create causal-attention graph stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin causal-attention capture");
    launch(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end causal-attention capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate causal-attention graph");
    for (int replay = 0; replay < 2; ++replay) {
        cuda_check(cudaGraphLaunch(executable, stream), "launch causal-attention graph");
    }
    cuda_check(cudaStreamSynchronize(stream), "synchronize causal-attention graph");
    cuda_check(cudaGraphExecDestroy(executable), "destroy causal-attention graph executable");
    cuda_check(cudaGraphDestroy(graph), "destroy causal-attention graph");
    cuda_check(cudaStreamDestroy(stream), "destroy causal-attention graph stream");
}

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

int run_a1_case(const Geometry& geometry, const CachePlan& plan, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};

    const HostCache initial = make_cache(geometry, plan, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), cache_plan_storage(plan), envelope, 1, test_case.tokens,
        test_case.tokens, test_execution());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    launch_attention_case(
        [&](cudaStream_t stream) {
            ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, ttable_row,
                                          op_geometry(geometry), kAttentionScale,
                                          cache.batch_view(), envelope, workspace, tout,
                                          test_execution(stream));
        },
        test_case.graph_replay);

    const std::string label =
        case_label("causal_softmax_attention", geometry, plan, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(plan));
    failures += verify_cache(label, cache.snapshot(), expected, plan.dtype == DType::BF16);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

// Qualify large production widths against selected independent oracle rows;
// execution, cache-state checks and guards still cover the entire request.
template <class T>
std::vector<T> select_query_columns(const std::vector<T>& values, std::size_t stride,
                                    std::span<const int> queries) {
    if (queries.empty()) return values;
    std::vector<T> result(stride * queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i)
        std::copy_n(values.begin() + queries[i] * stride, stride, result.begin() + i * stride);
    return result;
}

int run_a1_case(const Geometry& geometry, KvCacheStorage storage, const AttentionCase& test_case,
                MappingPattern mapping, std::span<const int> oracle_queries = {},
                std::span<const std::uint32_t> graph_limits = {}) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u,
                                            -test_case.v_amplitude, test_case.v_amplitude);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                   test_case.envelope_max};

    const HostCache initial = make_cache(geometry, storage, max_context, test_case.seed + 10u,
                                         test_case.v_amplitude);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference =
        ideal_attention(select_query_columns(q, kHeadDim * geometry.q_heads, oracle_queries),
                        expected, select_query_columns(positions, 1, oracle_queries));
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, 1, test_case.tokens, test_case.tokens,
        test_execution());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    int graph_failures = 0;
    const auto launch  = [&](cudaStream_t stream) {
        ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, ttable_row, op_geometry(geometry),
                                       kAttentionScale, cache.batch_view(), envelope, workspace,
                                       tout, test_execution(stream));
    };
    if (graph_limits.empty()) {
        launch_attention_case(launch, test_case.graph_replay);
    } else {
        DeviceContext device(0);
        DecodeGraphExecutable executable;
        for (const auto limit : graph_limits) {
            envelope.max_visible_keys = limit;
            launch(device.stream);
            cuda_synchronize();
            DecodeGraphDefinition definition;
            definition.capture(device.stream, [&] { launch(device.stream); });
            if (executable.ready())
                executable.update(definition);
            else
                executable.instantiate(definition);
            dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));
            cuda_synchronize();
            executable.launch(device.stream);
            cuda_synchronize();
            const auto actual = copy_from_guarded<std::uint16_t>(dout, q_bits.size());
            graph_failures +=
                verify_attention("BF16 attention envelope update",
                                 bf16_bits_to_double(select_query_columns(
                                     actual, kHeadDim * geometry.q_heads, oracle_queries)),
                                 reference, attention_criterion(storage));
        }
    }

    const std::string label =
        case_label("causal_softmax_attention", geometry, storage, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = graph_failures +
                   verify_attention(label,
                                    bf16_bits_to_double(select_query_columns(
                                        output_bits, kHeadDim * geometry.q_heads, oracle_queries)),
                                    reference, attention_criterion(storage));
    failures += verify_cache(label, cache.snapshot(), expected,
                             storage == KvCacheStorage::BFloat16 ||
                                 storage == KvCacheStorage::Nvfp4Group16 ||
                                 storage == KvCacheStorage::Fp8KeyNvfp4Value);
    if (storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        DeviceCache standalone(initial, mapping);
        ops::kv_cache_append(tk, tv, tp, standalone.view(), nullptr);
        cuda_synchronize();
        failures += verify_cache(label + " fused/standalone byte parity", cache.snapshot(),
                                 standalone.snapshot(), true);
        failures += standalone.verify_guards(label + " standalone append");
    }
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, const CachePlan& plan, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};

    const HostCache cache_host = make_cache(geometry, plan, max_context, test_case.seed + 10u);
    const std::vector<double> reference = ideal_attention(q, cache_host, positions);
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), cache_plan_storage(plan), envelope, 1, test_case.tokens,
        test_case.tokens, test_execution());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    launch_attention_case(
        [&](cudaStream_t stream) {
            ops::causal_softmax_attention_cached(tq, tp, op_geometry(geometry), kAttentionScale,
                                                 cache.view(), envelope, workspace, tout,
                                                 test_execution(stream));
        },
        test_case.graph_replay);

    const std::string label =
        case_label("causal_softmax_attention_cached", geometry, plan, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(plan));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host, true);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, KvCacheStorage storage, const AttentionCase& test_case,
                MappingPattern mapping, std::span<const int> oracle_queries = {}) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    if (test_case.zero_q) std::fill(q.begin(), q.end(), 0.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};

    const HostCache cache_host = make_cache(geometry, storage, max_context, test_case.seed + 10u,
                                            test_case.v_amplitude);
    const std::vector<double> reference =
        ideal_attention(select_query_columns(q, kHeadDim * geometry.q_heads, oracle_queries),
                        cache_host, select_query_columns(positions, 1, oracle_queries));
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, 1, test_case.tokens, test_case.tokens,
        test_execution());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    launch_attention_case(
        [&](cudaStream_t stream) {
            ops::causal_softmax_attention_cached(tq, tp, op_geometry(geometry), kAttentionScale,
                                                 cache.view(), envelope, workspace, tout,
                                                 test_execution(stream));
        },
        test_case.graph_replay);

    const std::string label =
        case_label("causal_softmax_attention_cached", geometry, storage, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label,
                                    bf16_bits_to_double(select_query_columns(
                                        output_bits, kHeadDim * geometry.q_heads, oracle_queries)),
                                    reference, attention_criterion(storage));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host, true);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

// One batched attention launch.
//
// The three vectors are parallel and one entry long per *request*, so their common length is the
// batch size. `table_rows` is the one that trips people up: its entries are indices into the KV
// cache table, not positions in the batch, and they are chosen independently of the batch size.
// A single request addressing table row 7 is a legitimate and interesting case -- it is a server
// with eight cache slots serving one request that happens to live in slot 7 -- so the cache table
// is sized from the rows addressed (cache_table_row_count), never from the batch.
//
// Getting that wrong is not a clean failure. Sizing the table by the batch made
// upstream's DFlash2 sweep index a one-element vector at [7], unchecked, and the resulting
// undefined behaviour presented as a different symptom on almost every run -- "vector too long",
// "bad allocation", an access violation, or a nonsense geometry error -- none of which pointed at
// the fixture. validate_batch_case below exists so a malformed profile says so instead.
struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// How many rows the cache table needs to hold every row this case addresses. Deliberately not the
// batch size: table_rows carries indices into the table, so a batch of one addressing row 7 needs
// eight rows, not one.
std::size_t cache_table_row_count(const std::vector<std::int32_t>& table_rows) {
    std::size_t highest = 0;
    for (const std::int32_t row : table_rows) {
        highest = std::max(highest, static_cast<std::size_t>(row));
    }
    return highest + 1;
}

// Reject a malformed profile with a message naming the field and the offending value, rather than
// letting it become undefined behaviour somewhere downstream. Both run_batch_case overloads call
// this; it replaces the size-only check they each used to carry.
//
// The distinctness rule on table_rows is the non-obvious one. Two requests in one launch writing
// the same cache row is not merely unusual, it has no defined answer: the host reference appends
// them one after another in request order, while the kernel writes both concurrently. A case like
// that would fail as an intermittent numeric mismatch, which is a bad way to learn the profile was
// wrong.
void validate_batch_case(const BatchAttentionCase& test_case) {
    const std::size_t batch = test_case.contexts.size();
    const auto fail         = [](const std::string& detail) {
        throw std::invalid_argument("causal-attention batch test profile: " + detail);
    };

    if (batch == 0) { fail("empty batch"); }
    if (test_case.valid_columns.size() != batch) {
        fail("valid_columns has " + std::to_string(test_case.valid_columns.size()) +
             " entries for a batch of " + std::to_string(batch));
    }
    if (test_case.table_rows.size() != batch) {
        fail("table_rows has " + std::to_string(test_case.table_rows.size()) +
             " entries for a batch of " + std::to_string(batch));
    }
    if (test_case.width <= 0) { fail("width " + std::to_string(test_case.width) + " is not positive"); }

    std::vector<std::int32_t> seen;
    seen.reserve(batch);
    for (std::size_t request = 0; request < batch; ++request) {
        const std::int32_t context = test_case.contexts[request];
        const std::int32_t valid   = test_case.valid_columns[request];
        const std::int32_t row     = test_case.table_rows[request];
        const std::string at       = " at request " + std::to_string(request);

        if (context < 0) { fail("negative context " + std::to_string(context) + at); }
        if (valid < 0 || valid > test_case.width) {
            fail("valid_columns " + std::to_string(valid) + at + " is outside [0, width=" +
                 std::to_string(test_case.width) + "]");
        }
        if (row < 0) { fail("negative table row " + std::to_string(row) + at); }
        // A request with valid == 0 writes nothing: the small-T kernels return with neutral
        // output before touching the cache for it. Two such requests sharing a table row is not a
        // race -- neither is a writer -- so only requests that actually write are checked against
        // (and added to) the distinctness set.
        if (valid > 0) {
            if (std::find(seen.begin(), seen.end(), row) != seen.end()) {
                fail("table row " + std::to_string(row) + " is addressed twice" + at +
                     "; two requests writing one cache row in a single launch has no defined result");
            }
            seen.push_back(row);
        }
    }
}

int run_batch_case(const Geometry& geometry, const CachePlan& plan, const BatchAttentionCase& test_case) {
    validate_batch_case(test_case);
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context = maximum_visible + 3;
    const ops::CausalAttentionExecutionEnvelope envelope{
        static_cast<std::uint32_t>(maximum_visible), static_cast<std::uint32_t>(maximum_visible)};
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    // The cache table is sized by the rows the batch *addresses*, not by the batch size. Those are
    // different numbers: a case may run one request against table row 7, which is the realistic
    // shape -- a server with eight cache slots serving a single request that lives in slot 7.
    // Sizing this by `batch` is what made upstream's DFlash2 sweep die on its very first case,
    // B=1 with table_rows={7}: expected[7] on a one-element vector, unchecked, so the symptom
    // varied run to run and never pointed at the real fault.
    const std::size_t table_rows = cache_table_row_count(test_case.table_rows);
    std::vector<HostCache> initial;
    initial.reserve(table_rows);
    for (std::size_t row = 0; row < table_rows; ++row) {
        initial.push_back(make_cache(geometry, plan, max_context,
                                     test_case.seed + 20u + 3u * static_cast<unsigned>(row)));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected.at(static_cast<std::size_t>(table_row)), row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected.at(static_cast<std::size_t>(table_row)), row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), cache_plan_storage(plan), envelope, batch, test_case.width,
        test_case.width, test_execution());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::causal_softmax_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows,
                                  op_geometry(geometry), kAttentionScale, cache.view(), envelope,
                                  workspace, tout, test_execution());
    cuda_synchronize();

    const std::string label = std::string("causal_softmax_attention batch ") + geometry.name + " " +
                              cache_name(plan) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(plan));
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, KvCacheStorage storage,
                   const BatchAttentionCase& test_case, int envelope_max = 0) {
    validate_batch_case(test_case);
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const ops::CausalAttentionExecutionEnvelope envelope{
        static_cast<std::uint32_t>(maximum_visible),
        static_cast<std::uint32_t>(std::max(maximum_visible, envelope_max))};
    const std::int32_t max_context      = static_cast<std::int32_t>(envelope.max_visible_keys) + 3;
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    // Sized by the table rows the batch addresses, not by the batch. See the note on the CachePlan
    // overload above for why those differ and what it cost.
    const std::size_t table_rows = cache_table_row_count(test_case.table_rows);
    std::vector<HostCache> initial;
    initial.reserve(table_rows);
    for (std::size_t row = 0; row < table_rows; ++row) {
        initial.push_back(make_cache(geometry, storage, max_context,
                                     test_case.seed + 20u + 3u * static_cast<unsigned>(row)));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected.at(static_cast<std::size_t>(table_row)), row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected.at(static_cast<std::size_t>(table_row)), row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, batch, test_case.width, test_case.width,
        test_execution());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::causal_softmax_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows,
                                  op_geometry(geometry), kAttentionScale, cache.view(), envelope,
                                  workspace, tout, test_execution());
    cuda_synchronize();

    const std::string label = std::string("causal_softmax_attention batch ") + geometry.name + " " +
                              cache_name(storage) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(storage));
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int report_quantization_quality(KvCacheStorage storage, std::uint32_t seed) {
    const Geometry& geometry       = kGeometries[0];
    constexpr std::int32_t tokens  = 6;
    constexpr std::int32_t base    = 61;
    constexpr std::int32_t context = 128;
    const std::size_t q_elements   = static_cast<std::size_t>(kHeadDim) * geometry.q_heads * tokens;
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * tokens;
    std::vector<float> q          = make_bf16_values(q_elements, seed, -0.25f, 0.25f);
    std::vector<float> k          = make_bf16_values(kv_elements, seed + 1u, -0.25f, 0.25f);
    std::vector<float> v          = make_bf16_values(kv_elements, seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    std::vector<std::int32_t> positions(tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    // Both fixtures start from the same public BF16 logical cache. The first oracle applies the
    // target codec exactly; the second evaluates the corresponding unquantized BF16/FP64
    // attention formula. This evidence is deliberately independent of production output.
    HostCache represented = make_cache(geometry, storage, context, seed + 10u);
    HostCache unquantized = make_cache(geometry, KvCacheStorage::BFloat16, context, seed + 10u);
    append_cache(represented, k, v, positions);
    append_cache(unquantized, k, v, positions);
    const std::vector<double> quantized_output = ideal_attention(q, represented, positions);
    const std::vector<double> bf16_output      = ideal_attention(q, unquantized, positions);

    double sum_abs       = 0.0;
    double sum_sq        = 0.0;
    double reference_sq  = 0.0;
    double maximum_error = 0.0;
    for (std::size_t i = 0; i < quantized_output.size(); ++i) {
        const double error = quantized_output[i] - bf16_output[i];
        sum_abs += std::abs(error);
        sum_sq += error * error;
        reference_sq += bf16_output[i] * bf16_output[i];
        maximum_error = std::max(maximum_error, std::abs(error));
    }
    const double count         = static_cast<double>(quantized_output.size());
    const double mae           = sum_abs / count;
    const double rmse          = std::sqrt(sum_sq / count);
    const double reference_rms = std::sqrt(reference_sq / count);
    const double relative_rmse = reference_rms == 0.0 ? 0.0 : rmse / reference_rms;
    if (!std::isfinite(mae) || !std::isfinite(rmse) || !std::isfinite(relative_rmse) ||
        !std::isfinite(maximum_error)) {
        std::cerr << cache_name(storage)
                  << " quantization quality produced non-finite statistics\n";
        return 1;
    }
    std::cout << cache_name(storage) << " quantized-vs-bf16 oracle quality" << " mae=" << mae
              << " rmse=" << rmse << " rel_rmse=" << relative_rmse << " max_abs=" << maximum_error
              << '\n';
    return 0;
}

int run_quantized_batch_cases(KvCacheStorage storage, std::uint32_t seed) {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], storage,
                               {1, {63}, {1}, {0}, MappingPattern::Identity, seed});
    failures +=
        run_batch_case(kGeometries[1], storage,
                       {6, {61, 127}, {6, 3}, {1, 0}, MappingPattern::Fragmented, seed + 1u});
    failures += run_batch_case(
        kGeometries[0], storage,
        {6, {0, 63, 127, 2048}, {1, 6, 0, 3}, {2, 0, 3, 1}, MappingPattern::Fragmented, seed + 2u});
    failures += run_batch_case(kGeometries[1], storage,
                               {1,
                                {0, 1, 63, 64, 127, 128, 2048, 8191},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Fragmented,
                                seed + 3u});
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], kPlanInt8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], kPlanBf16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], kPlanBf16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], kPlanInt8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], kPlanInt8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], kPlanBf16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    // rk8v4 packed values in the batched routes: the small-T fused-append rows (T=6) and
    // the single-token long-context rows that drive the batched prompt layout.
    failures += run_batch_case(
        kGeometries[0], kPlanRk8v4, {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1},
                                     MappingPattern::Fragmented, 506u});
    failures += run_batch_case(
        kGeometries[1], kPlanRk8v4, {1, {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                     {1, 1, 1, 1, 1, 1, 1, 1}, {7, 0, 5, 2, 6, 1, 4, 3},
                                     MappingPattern::Identity, 507u});
    // rk4v4 packed keys through the same batched routes.
    failures += run_batch_case(
        kGeometries[0], kPlanRk4v4, {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1},
                                     MappingPattern::Fragmented, 508u});
    failures += run_batch_case(
        kGeometries[1], kPlanRk4v4, {1, {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                     {1, 1, 1, 1, 1, 1, 1, 1}, {7, 0, 5, 2, 6, 1, 4, 3},
                                     MappingPattern::Identity, 509u});
    failures += run_case_allowing_arch_skip(
        "causal_softmax_attention FP8 KV cache batched", [&] {
            return run_batch_case(kGeometries[0], kPlanFp8,
                                  {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1},
                                   MappingPattern::Fragmented, 505u});
        });
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    for (const CachePlan& plan : {kPlanBf16, kPlanInt8, kPlanRk8v4}) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_a1_case(geometry, plan, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, plan, {1, 128, 129, 191u}, mapping);
        }

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, plan, test_case, MappingPattern::Identity);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, plan, test_case, MappingPattern::Identity);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, plan, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures += run_a3_case(geometry, plan, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, plan, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, plan, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }
    return failures;
}

// A grouped BF16 call keeps one update-compatible two-stage execution across short and long
// envelopes. Verify each replay against the independent oracle. Upstream's version of this sweep
// also exercises the batched path via run_batch_case(..., graph_replay=true, envelope_max), but
// this fork's run_batch_case(KvCacheStorage, ...) overload has no phased capture/update/replay
// loop (BatchAttentionCase carries no graph_replay field) -- porting that would mean porting a
// second, larger piece of upstream's batch-case test infrastructure that predates this commit and
// is not part of this cherry-pick chain. Left out rather than added as a call that would compile
// but silently not exercise what its name promises.
int run_bf16_graph_envelope_cases() {
    int failures = 0;
    constexpr std::array<std::uint32_t, 5> short_limits{64, 128, 2048, 64, 2048};
    constexpr std::array<std::uint32_t, 4> long_limits{32769, 65536, 131072, 32769};
    for (const auto& geometry : kGeometries) {
        for (int width : {1, 4, 16})
            failures += run_a1_case(geometry, KvCacheStorage::BFloat16, {width, 17, 2048, 965u},
                                    MappingPattern::Fragmented, {}, short_limits);
        failures += run_a1_case(geometry, KvCacheStorage::BFloat16, {1, 32768, 131072, 966u},
                                MappingPattern::Fragmented, {}, long_limits);
    }
    return failures;
}

// BF16 grouped split-KV attention now serves single-request prefill up to W=256 (upstream
// d44ab584); the tiled kernel takes wider calls. Cover both sides of that boundary, long envelopes
// with short live rows, the cached entry and a batched prefill-width row against the oracle.
// Upstream's version also raises the Q/K amplitude on two cases; this fork's AttentionCase has no
// amplitude field, so those run at the default fixture amplitude.
int run_bf16_small_prefill_cases() {
    int failures = 0;
    const std::array<int, 4> append_queries{0, 7, 16, 33};
    const std::array<int, 4> cached_queries{0, 63, 64, 128};
    const std::array<int, 3> boundary_queries{0, 127, 256};
    for (const auto& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::BFloat16,
                                {34, 8191, 32768, 1301u, false, true}, MappingPattern::Fragmented,
                                append_queries);
        failures += run_a3_case(geometry, KvCacheStorage::BFloat16, {129, 32768, 131072, 1302u},
                                MappingPattern::Fragmented, cached_queries);
        failures += run_a3_case(geometry, KvCacheStorage::BFloat16, {257, 8192, 16384, 1303u},
                                MappingPattern::Offset, boundary_queries);
        failures += run_batch_case(geometry, KvCacheStorage::BFloat16,
                                   {65, {8191}, {17}, {0}, MappingPattern::Fragmented, 1304u},
                                   32768);
    }
    return failures;
}

// FP8 decode attention (small_t, T<=6) dequantizes K to BF16 and V to FP16 before ordinary MMA
// and is ported to sm_86/sm_89; the prompt (T>6) kernel still needs the FP8 mma.sync...f8f6f4
// QK matmul (Blackwell-only) rewritten the same way. Kept in separate functions -- rather than
// one run_fp8_cases() wrapped by a single run_case_allowing_arch_skip -- so a still-rejected
// prompt case can't unwind past (and hide the result of) an already-ported decode case that
// happened to run first in the same try block.
int run_fp8_decode_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, kPlanFp8, {1, 128, 192, 603u},
                                MappingPattern::Fragmented);
        failures +=
            run_a3_case(geometry, kPlanFp8, {1, 128, 192, 604u}, MappingPattern::Offset);
        failures += run_a1_case(geometry, kPlanFp8, {6, 61, 192, 605u},
                                MappingPattern::Fragmented);
    }
    failures += run_a3_case(kGeometries[0], kPlanFp8, {3, 63, 192, 606u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[0], kPlanFp8, {1, 16384, 16385, 607u},
                            MappingPattern::Fragmented);
    return failures;
}

int run_fp8_prompt_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, kPlanFp8, {65, 63, 192, 601u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, kPlanFp8, {65, 63, 192, 602u},
                                MappingPattern::Fragmented);
    }
    return failures;
}

// RotorQuant-style rk8v4: rotated INT8 G64 keys plus unrotated packed signed-INT4
// values with one FP16 scale per 32-value group (G32). This is the shipped compact KV
// profile, but the consumer attention kernels only ever ran against BF16/INT8/FP8 caches,
// so the packed value plane (and its shared-memory scale layout in both the prompt and
// small-T routes) needs direct oracle coverage here. Unlike FP8, INT4 packing needs no
// arch feature, so these cases run on every build the INT8 cases run on.
int run_rk8v4_cases() {
    std::cout << "  rk8v4 (INT8-G64 rotated keys + packed INT4 G32 values):\n";
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        // T >= 7: prompt route over the packed cache; the append writes packed V and
        // G32 scales that the prompt kernel then decodes (both static-smem T and the
        // dynamic-arena T >= 7 layout).
        failures += run_a1_case(geometry, kPlanRk8v4, {65, 63, 192, 611u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, kPlanRk8v4, {65, 63, 192, 612u},
                                MappingPattern::Fragmented);
        // Small-T prompt route (T <= 6) with the fused append writing the packed plane.
        failures += run_a1_case(geometry, kPlanRk8v4, {1, 128, 192, 613u},
                                MappingPattern::Fragmented);
        failures +=
            run_a3_case(geometry, kPlanRk8v4, {1, 128, 192, 614u}, MappingPattern::Offset);
        failures += run_a1_case(geometry, kPlanRk8v4, {6, 61, 192, 615u},
                                MappingPattern::Fragmented);
    }
    failures += run_a3_case(kGeometries[0], kPlanRk8v4, {3, 63, 192, 616u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[0], kPlanRk8v4, {1, 16384, 16385, 617u},
                            MappingPattern::Fragmented);
    return failures;
}

// rk4v4: rotated 4-bit Lloyd-Max key indices over a G64 FP16 scale plus rk8v4's packed INT4 G32
// values. Both attention kernels load the packed keys into registers and expand them to INT8 codes
// in the shared K tile, so this covers each kernel's staging at its tile shapes: the prompt kernel
// (T >= 7, one 64-key tile per page), every small-T warp/key-block schedule (the window tiers
// select 32- and 64-key blocks and 6- to 32-warp CTAs), the fused append, graph replay, and a
// long split-KV window.
int run_rk4v4_cases() {
    std::cout << "  rk4v4 (Lloyd-Max 4-bit rotated keys + packed INT4 G32 values):\n";
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_a1_case(geometry, kPlanRk4v4, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, kPlanRk4v4, {1, 128, 129, 191u}, mapping);
        }
        failures += run_a1_case(geometry, kPlanRk4v4, {65, 63, 192, 621u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, kPlanRk4v4, {65, 63, 192, 622u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, kPlanRk4v4, {1, 0, 1, 623u}, MappingPattern::Identity);
        failures += run_a1_case(geometry, kPlanRk4v4, {1, 128, 192, 624u},
                                MappingPattern::Fragmented);
        failures +=
            run_a3_case(geometry, kPlanRk4v4, {1, 128, 192, 625u}, MappingPattern::Offset);
        failures += run_a1_case(geometry, kPlanRk4v4, {6, 61, 192, 626u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, kPlanRk4v4, {7, 17, 512, 627u},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, kPlanRk4v4, {17, 31, 48, 628u},
                                MappingPattern::Identity);
        // Window tiers of the small-T schedule: <= 1029/2054, <= 4096/8198 (64-key blocks under
        // T >= 6), and the long split-KV tier.
        for (const std::int32_t depth : {1000, 2000, 6000, 20000}) {
            failures += run_a1_case(geometry, kPlanRk4v4,
                                    {1, depth, static_cast<std::uint32_t>(depth + 1), 629u},
                                    MappingPattern::Fragmented);
            failures += run_a1_case(geometry, kPlanRk4v4,
                                    {6, depth, static_cast<std::uint32_t>(depth + 6), 630u},
                                    MappingPattern::Fragmented);
        }
        failures += run_a1_case(geometry, kPlanRk4v4, {4, 3000, 3004, 631u, false, true},
                                MappingPattern::Fragmented);
    }
    failures += run_a3_case(kGeometries[0], kPlanRk4v4, {3, 63, 192, 632u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[0], kPlanRk4v4, {8, 5000, 5008, 633u},
                            MappingPattern::Fragmented);
    return failures;
}

// Nvfp4Group16 decode attention (small_t, T<=6) is ported to sm_86/sm_89 (its e2m1 codec is now
// a software encode/decode, no Blackwell tensor-core dependency); the prompt (T>6) kernel is
// warp-specialized (setmaxnreg dynamic producer/consumer register reallocation), a genuine
// Hopper+ feature, and still isn't. Split the same way as the FP8 cases above, for the same
// reason: a still-rejected prompt case must not unwind past an already-ported decode case
// wrapped in the same run_case_allowing_arch_skip.
int run_nvfp4_decode_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 0, 1, 701u},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 64, 65, 702u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 32, 33, 707u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 1, 2, 708u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 7, 8, 709u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 15, 16, 710u, true},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 31, 32, 711u, true},
                                MappingPattern::Identity);
        failures += run_a1_case(geometry, KvCacheStorage::Nvfp4Group16, {6, 61, 192, 703u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {1, 2048, 2049, 706u},
                                MappingPattern::Fragmented);
    }
    failures += run_a1_case(kGeometries[0], KvCacheStorage::Nvfp4Group16,
                            {1, 64, 65, 712u, false, true}, MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[0], KvCacheStorage::Nvfp4Group16,
                            {1, 64, 65, 713u, false, true}, MappingPattern::Fragmented);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {5, 17, 22, 714u},
                            MappingPattern::Identity);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 8191, 8192, 717u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 16383, 16384, 718u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {1, 65535, 65536, 719u},
                            MappingPattern::Fragmented);
    return failures;
}

int run_nvfp4_prompt_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::Nvfp4Group16, {64, 0, 128, 704u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Nvfp4Group16, {65, 63, 192, 705u},
                                MappingPattern::Offset);
    }
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {7, 17, 24, 715u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Nvfp4Group16, {7, 511, 518, 716u},
                            MappingPattern::Fragmented);
    return failures;
}

// K8V4 decode attention (small_t, T<=6) dequantizes its FP8 key plane to BF16 before ordinary MMA
// (same fallback pattern as plain FP8 above) and is ported to sm_86/sm_89; its value plane
// (NVFP4) was already portable. The prompt (T>6) kernel still needs the same QK matmul rewrite as
// plain FP8's prompt kernel. Split the same way and for the same reason as the FP8/NVFP4 cases
// above: a still-rejected prompt case must not unwind past an already-ported decode case wrapped
// in the same run_case_allowing_arch_skip.
int run_k8v4_decode_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 0, 1, 801u},
                                MappingPattern::Identity);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 64, 65, 802u},
                                MappingPattern::Fragmented);
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {6, 61, 192, 803u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {1, 2048, 2049, 806u},
                                MappingPattern::Fragmented);
    }
    failures += run_a1_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 64, 65, 807u, false, true}, MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[0], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 64, 65, 808u, false, true}, MappingPattern::Fragmented);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {5, 17, 22, 809u},
                            MappingPattern::Identity);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {1, 8191, 8192, 812u},
                            MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 16383, 16384, 813u}, MappingPattern::Fragmented);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value,
                            {1, 65535, 65536, 814u}, MappingPattern::Fragmented);
    return failures;
}

int run_k8v4_prompt_cases() {
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        failures += run_a1_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {64, 0, 128, 804u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, KvCacheStorage::Fp8KeyNvfp4Value, {65, 63, 192, 805u},
                                MappingPattern::Offset);
    }
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {7, 17, 24, 810u},
                            MappingPattern::Identity);
    failures += run_a1_case(kGeometries[1], KvCacheStorage::Fp8KeyNvfp4Value, {7, 511, 518, 811u},
                            MappingPattern::Fragmented);
    return failures;
}

// The INT8-G64 prompt route over more than 256 visible keys (fewer keep the tiled kernel, which
// run_geometry covers): register-resident FA2-style CTAs of four or eight warps, FP16 PV
// accumulation per 64-key tile under a power-of-two V-scale shift, and, from W = 17, key splits
// merged in FP32.
// Cases follow Infernix's qualification of the same kernel (V amplitude on either side of the FP16
// shift, 3584- and 4096-token chunks at context 0 and after an 8K history with sampled query rows)
// plus split launches over 4K and 33K histories with partial last row blocks and pages, an envelope
// past the populated keys, the plan's own width and history boundaries at b-1/b/b+1, the cached
// entry, a masked batch row and graph replay. Whether a launch splits is read from the public
// launch shape (a split adds the merge kernel), so the plan constants can move without the cases
// silently losing their split coverage.
bool int8_prompt_splits(const Geometry& geometry, std::int32_t tokens, std::uint32_t visible_keys) {
    return ops::causal_softmax_attention_launch_shape(
               op_geometry(geometry), KvCacheStorage::Int8Group64, {1, visible_keys}, 1, tokens,
               test_execution())
               .kernel_nodes == 3U;
}

std::vector<int> sampled_query_rows(std::int32_t tokens) {
    std::vector<int> rows;
    for (const int row : {0, 1, 63, 64, 127, 128, tokens / 2, tokens - 129, tokens - 128,
                          tokens - 65, tokens - 64, tokens - 2, tokens - 1}) {
        if (row >= 0 && row < tokens && std::find(rows.begin(), rows.end(), row) == rows.end()) {
            rows.push_back(row);
        }
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

int run_int8_prompt_cases() {
    std::cout << "  int8-g64 FA2 prompt route (FP16 PV accumulation, key splits):\n";
    constexpr KvCacheStorage kInt8 = KvCacheStorage::Int8Group64;
    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        int split_launches = 0;
        const auto a1 = [&](const AttentionCase& test_case, MappingPattern mapping,
                            bool sample_rows = true) {
            if (int8_prompt_splits(geometry, test_case.tokens, test_case.envelope_max)) {
                ++split_launches;
            }
            const std::vector<int> rows =
                sample_rows ? sampled_query_rows(test_case.tokens) : std::vector<int>{};
            return run_a1_case(geometry, kInt8, test_case, mapping, rows);
        };
        const auto a3 = [&](const AttentionCase& test_case, MappingPattern mapping) {
            if (int8_prompt_splits(geometry, test_case.tokens, test_case.envelope_max)) {
                ++split_launches;
            }
            const std::vector<int> rows = sampled_query_rows(test_case.tokens);
            return run_a3_case(geometry, kInt8, test_case, mapping, rows);
        };

        // V group scales of 2048/127 force the FP16 partial shift; 900/127 sits just inside the
        // unshifted 64 * 127 * 8 bound. Both narrow (four-warp) and split launches.
        failures += a1({65, 263, 392, 2101u, false, false, 2048.0f}, MappingPattern::Fragmented,
                       false);
        failures += a1({65, 263, 392, 2102u, false, false, 900.0f}, MappingPattern::Fragmented,
                       false);
        failures += a1({200, 4100, 4300, 2103u, false, false, 2048.0f}, MappingPattern::Fragmented);
        failures += a3({300, 2000, 2300, 2104u, false, false, 900.0f}, MappingPattern::Offset);

        // Full prefill chunks as the first chunk and after an 8K history.
        const std::array<int, 6> chunk_rows_3584{0, 127, 128, 1791, 3584 - 128, 3584 - 1};
        const std::array<int, 6> chunk_rows_4096{0, 127, 128, 1791, 4096 - 128, 4096 - 1};
        failures += run_a1_case(geometry, kInt8, {3584, 0, 3584, 2111u},
                                MappingPattern::Fragmented, chunk_rows_3584);
        failures += run_a1_case(geometry, kInt8, {4096, 0, 4096, 2112u},
                                MappingPattern::Fragmented, chunk_rows_4096);
        failures += run_a1_case(geometry, kInt8, {3584, 8192, 8192 + 3584, 2113u},
                                MappingPattern::Fragmented, chunk_rows_3584);
        failures += run_a1_case(geometry, kInt8, {4096, 8192, 8192 + 4096, 2114u},
                                MappingPattern::Fragmented, chunk_rows_4096);

        // Short follow-ups over long histories: the split domain, with partial last row blocks
        // and a partial last page (neither history is a multiple of 64).
        std::uint32_t seed = 2120u;
        for (const std::int32_t base : {4100, 33000}) {
            for (const std::int32_t tokens : {17, 200, 257, 513, 1100}) {
                failures += a1({tokens, base, static_cast<std::uint32_t>(base + tokens), seed++},
                               MappingPattern::Fragmented, tokens > 17);
            }
        }
        // Envelopes past the populated keys plan more splits than the live rows can fill, so
        // trailing splits sweep no key and publish neutral statistics.
        failures += a1({200, 4100, 40000, seed++}, MappingPattern::Fragmented);
        failures += a3({257, 4100, 40000, seed++}, MappingPattern::Fragmented);
        // Cached entry, and graph replay of a split and of an unsplit launch.
        failures += a3({513, 33000, 33513, seed++}, MappingPattern::Fragmented);
        failures += a1({200, 33000, 33200, seed++, false, true}, MappingPattern::Fragmented);
        failures += a1({100, 400, 500, seed++, false, true}, MappingPattern::Fragmented, false);
        // The first launch past the tiled-kernel window, and the last one inside it.
        failures += a1({17, 240, 257, seed++}, MappingPattern::Fragmented, false);
        failures += a1({17, 239, 256, seed++}, MappingPattern::Fragmented, false);

        // The plan's own boundaries: the widths at which a launch over a 33K history starts or
        // stops splitting, and the history at which a 200-column launch starts.
        constexpr std::int32_t kBoundaryBase = 33000;
        int width_boundaries = 0;
        for (std::int32_t tokens = 18; tokens <= 2048 && width_boundaries < 2; ++tokens) {
            const auto keys = [&](std::int32_t t) {
                return static_cast<std::uint32_t>(kBoundaryBase + t);
            };
            if (int8_prompt_splits(geometry, tokens, keys(tokens)) ==
                int8_prompt_splits(geometry, tokens - 1, keys(tokens - 1))) {
                continue;
            }
            ++width_boundaries;
            for (const std::int32_t t : {tokens - 1, tokens, tokens + 1}) {
                failures += a1({t, kBoundaryBase, keys(t), seed++}, MappingPattern::Fragmented);
            }
        }
        for (std::uint32_t keys = 257; keys <= 40000; ++keys) {
            if (!int8_prompt_splits(geometry, 200, keys)) { continue; }
            for (const std::uint32_t k : {keys - 1, keys, keys + 1}) {
                failures += a1({200, static_cast<std::int32_t>(k) - 200, k, seed++},
                               MappingPattern::Offset);
            }
            break;
        }

        // A masked single-row batch: 17 live columns of 65 over an 8K history, under a 33K
        // envelope that plans a split launch.
        if (int8_prompt_splits(geometry, 65, 33000)) { ++split_launches; }
        failures += run_batch_case(geometry, kInt8,
                                   {65, {8191}, {17}, {0}, MappingPattern::Fragmented, seed++},
                                   33000);

        if (split_launches == 0) {
            std::cerr << "int8-g64 prompt cases " << geometry.name
                      << " exercised no split launch\n";
            ++failures;
        }
    }
    return failures;
}

// A plan at an envelope's maximum must bound the scratch of every launch inside it: the capacity
// at fewer visible keys never exceeds the capacity at more. The split partials also stay inside
// the internal budget.
int verify_int8_prompt_split_capacity() {
    int failures = 0;
    for (const auto& geometry : kGeometries) {
        for (const int sm_count : {82, 170}) {
            const DeviceExecutionView execution{nullptr, sm_count};
            for (const std::int32_t width : {17, 63, 64, 65, 128, 129, 200, 257, 384, 513, 1024,
                                             1100, 2048, 4096}) {
                std::size_t previous = 0;
                for (std::uint32_t keys = static_cast<std::uint32_t>(width); keys <= 262144;
                     keys = keys * 9 / 8 + 1) {
                    const std::size_t capacity =
                        ops::causal_softmax_attention_workspace_capacity_bytes(
                            op_geometry(geometry), KvCacheStorage::Int8Group64, {1, keys}, 1,
                            width, width, execution);
                    if (capacity < previous || capacity > (std::size_t{64} << 20) + 512) {
                        std::cerr << "int8-g64 prompt split capacity " << geometry.name
                                  << " SM=" << sm_count << " W=" << width << " keys=" << keys
                                  << ": " << capacity << " after " << previous << '\n';
                        ++failures;
                    }
                    previous = std::max(previous, capacity);
                }
            }
        }
    }
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    // NVFP4/K8V4 are recognized KvCacheStorage selections but not ported on this fork (see
    // d256_kv_cache_profile), so causal_softmax_attention_workspace_capacity_bytes always rejects
    // them; there is no capacity contract to verify for a format this fork never plans a route
    // for.
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256}) {
        constexpr ops::CausalAttentionExecutionEnvelope envelope{1, 1025};
        constexpr ops::AttentionHeadGeometry geometry{kHeadDim, 16, 2};
        const std::size_t interval = ops::causal_softmax_attention_workspace_capacity_bytes(
            geometry, storage, envelope, 1, 1, 17, test_execution());
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::causal_softmax_attention_workspace_capacity_bytes(
                                            geometry, storage, envelope, 1, tokens, tokens,
                                            test_execution()));
        }
        if (interval != witness) {
            std::cerr << "causal_softmax_attention interval capacity has no exact route witness\n";
            ++failures;
        }

        // Prefill split counts can decrease as query width grows. The interval query must still
        // cover every supported point, including before a drop.
        for (const auto& item : kGeometries) {
            constexpr ops::CausalAttentionExecutionEnvelope prefill_envelope{1, 131072};
            std::size_t largest = 0;
            for (int width = 17; width <= 1025; ++width)
                largest = std::max(largest, ops::causal_softmax_attention_workspace_capacity_bytes(
                                                op_geometry(item), storage, prefill_envelope, 1,
                                                width, width, test_execution()));
            const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
                op_geometry(item), storage, prefill_envelope, 1, 17, 1025, test_execution());
            if (capacity != largest) {
                std::cerr << "causal_softmax_attention prefill interval capacity mismatch\n";
                ++failures;
            }
        }

        // Device capacity is a planning input (the BF16 plan derives its split budget from it).
        // Host-only queries cover batched decode and single-request prefill on several SM counts.
        for (const int sm_count : {82, 96, 160, 170, 192}) {
            const DeviceExecutionView capacity_execution{nullptr, sm_count};
            constexpr ops::AttentionHeadGeometry capacity_geometry{kHeadDim, 24, 4};
            constexpr ops::CausalAttentionExecutionEnvelope capacity_envelope{1, 131072};
            for (const int batch : {1, 3}) {
                const int max_width = batch == 1 ? 257 : 16;
                std::size_t largest = 0;
                for (int width = 1; width <= max_width; ++width)
                    largest = std::max(largest,
                                       ops::causal_softmax_attention_workspace_capacity_bytes(
                                           capacity_geometry, storage, capacity_envelope, batch,
                                           width, width, capacity_execution));
                const auto capacity = ops::causal_softmax_attention_workspace_capacity_bytes(
                    capacity_geometry, storage, capacity_envelope, batch, 1, max_width,
                    capacity_execution);
                if (capacity != largest) {
                    std::cerr << "causal_softmax_attention capacity interval mismatch for SM="
                              << sm_count << " B=" << batch << '\n';
                    ++failures;
                }
            }
        }
        for (const int sm_count : {0, -1}) {
            try {
                (void)ops::causal_softmax_attention_workspace_capacity_bytes(
                    geometry, storage, envelope, 1, 1, 17, {nullptr, sm_count});
                std::cerr << "causal_softmax_attention accepted a nonpositive SM count\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
        }
    }
    try {
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            {kHeadDim, 16, 2}, KvCacheStorage::BFloat16,
            {1, ops::kCausalAttentionMaximumVisibleKeys}, 1, 1, 1, test_execution());
    } catch (const std::invalid_argument&) {
        std::cerr << "causal_softmax_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::causal_softmax_attention_workspace_capacity_bytes(
            {kHeadDim, 16, 2}, KvCacheStorage::BFloat16,
            {1, ops::kCausalAttentionMaximumVisibleKeys + 1}, 1, 1, 1,
            test_execution());
        std::cerr << "causal_softmax_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

int run_softmax_attention_nvfp4_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_case_allowing_arch_skip(
        "causal_softmax_attention nvfp4 independent correctness (decode)", [] {
            int decode_failures = run_nvfp4_decode_cases();
            decode_failures += run_quantized_batch_cases(KvCacheStorage::Nvfp4Group16, 720u);
            decode_failures += report_quantization_quality(KvCacheStorage::Nvfp4Group16, 724u);
            return decode_failures;
        });
    failures += run_case_allowing_arch_skip(
        "causal_softmax_attention nvfp4 independent correctness (prompt)",
        [] { return run_nvfp4_prompt_cases(); });
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention nvfp4 independent correctness\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_k8v4_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_case_allowing_arch_skip(
        "causal_softmax_attention k8v4 independent correctness (decode)", [] {
            int decode_failures = run_k8v4_decode_cases();
            decode_failures += run_quantized_batch_cases(KvCacheStorage::Fp8KeyNvfp4Value, 815u);
            decode_failures += report_quantization_quality(KvCacheStorage::Fp8KeyNvfp4Value, 819u);
            return decode_failures;
        });
    failures += run_case_allowing_arch_skip(
        "causal_softmax_attention k8v4 independent correctness (prompt)",
        [] { return run_k8v4_prompt_cases(); });
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention k8v4 independent correctness\n";
    return failures == 0 ? 0 : 1;
}

// DFlash2 verification shapes: narrow widths over many batch rows, ragged valid-column counts and
// fragmented page mappings, swept across every registered cache storage. Ported from upstream and
// adapted to this fork's fixture: rk8v4 is added (upstream has no such storage), and the batch
// cases drop upstream's per-case graph flag because run_batch_case here has no graph-replay path.
// Graph coverage is retained through the a1/a3 cases below, whose AttentionCase does carry it.
//
// `order` is a permutation of table rows, so a case with batch < 8 still addresses rows up to 7 --
// which is the point: it exercises a request sitting in a high cache slot. That is what
// cache_table_row_count exists for.
int run_dflash2_cases() {
    constexpr int order[]{7, 0, 4, 2, 6, 1, 5, 3};
    // run_batch_case/run_a1_case/run_a3_case are each overloaded on KvCacheStorage and CachePlan,
    // so one generic sweep body covers every registered profile -- including rk8v4, which the
    // public KvCacheStorage enum cannot select and only the CachePlan overload builds.
    // `order` has to be captured explicitly: reading `order[b]` at a runtime index odr-uses it,
    // and a lambda with no default capture cannot reach an automatic variable of the enclosing
    // function however constant it is. Introducing this lambda without the capture is why this
    // translation unit stopped compiling.
    const auto sweep = [&order](auto storage) {
        int failures = 0;
        const auto run = [&](int width, int batch, int base) {
            BatchAttentionCase c{width, {}, {}, {}, MappingPattern::Fragmented,
                                 static_cast<unsigned>(1700 + width + 31 * batch)};
            for (int b = 0; b < batch; ++b) {
                c.contexts.push_back(base + (b % 3 == 0 ? 0 : b % 3 == 1 ? 17 : 61));
                c.valid_columns.push_back(b % 4 == 0   ? width
                                          : b % 4 == 1 ? width - 1
                                          : b % 4 == 2 ? 1
                                                       : 0);
                c.table_rows.push_back(order[b]);
            }
            return run_batch_case(kGeometries[0], storage, c);
        };
        for (int width = 2; width <= 16; ++width)
            for (int batch : {1, 8}) failures += run(width, batch, 0);
        for (int width : {2, 8, 16})
            for (int batch = 2; batch <= 7; ++batch) failures += run(width, batch, 0);
        for (int width : {7, 8, 9, 16})
            for (int batch : {1, 8}) failures += run(width, batch, 127);
        failures += run(16, 8, 2048);
        for (int width : {8, 9, 16}) {
            failures +=
                run_a1_case(kGeometries[0], storage,
                            {width, 2048, static_cast<unsigned>(2048 + width), 1801u, false, true},
                            MappingPattern::Fragmented);
            failures +=
                run_a3_case(kGeometries[0], storage,
                            {width, 8192, static_cast<unsigned>(8192 + width), 1802u, false, true},
                            MappingPattern::Fragmented);
        }
        return failures;
    };

    int failures = 0;
    for (auto storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        failures += sweep(storage);
    }
    failures += sweep(kPlanRk8v4);
    failures += sweep(kPlanRk4v4);
    return failures;
}

// The validator is the thing standing between a mistyped profile and undefined behaviour, so it
// gets its own coverage. Needs no GPU: every case here must be rejected before any device work.
int run_batch_profile_validation_cases() {
    struct Rejected {
        const char* what;
        BatchAttentionCase profile;
    };
    const Rejected rejected[]{
        {"empty batch", {2, {}, {}, {}, MappingPattern::Identity, 1u}},
        {"valid_columns length", {2, {0, 0}, {2}, {0, 1}, MappingPattern::Identity, 2u}},
        {"table_rows length", {2, {0, 0}, {2, 2}, {0}, MappingPattern::Identity, 3u}},
        {"non-positive width", {0, {0}, {0}, {0}, MappingPattern::Identity, 4u}},
        {"negative context", {2, {-1}, {2}, {0}, MappingPattern::Identity, 5u}},
        {"valid_columns above width", {2, {0}, {3}, {0}, MappingPattern::Identity, 6u}},
        {"negative table row", {2, {0}, {2}, {-1}, MappingPattern::Identity, 7u}},
        {"duplicate table row", {2, {0, 0}, {2, 2}, {3, 3}, MappingPattern::Identity, 8u}},
    };

    int failures = 0;
    for (const Rejected& item : rejected) {
        bool threw = false;
        try {
            validate_batch_case(item.profile);
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) {
            std::cerr << "validate_batch_case accepted a profile it must reject: " << item.what
                      << '\n';
            ++failures;
        }
    }

    // And the shape the DFlash2 sweep actually uses -- one request addressing a high table row --
    // must be accepted, or the validator has over-corrected into rejecting the real cases.
    try {
        validate_batch_case({2, {0}, {2}, {7}, MappingPattern::Fragmented, 9u});
        if (cache_table_row_count({7}) != 8) {
            std::cerr << "cache_table_row_count({7}) must be 8, not the batch size\n";
            ++failures;
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << "validate_batch_case rejected the DFlash2 shape it must accept: "
                  << error.what() << '\n';
        ++failures;
    }

    // Two requests sharing a table row where neither is a writer -- valid_columns == 0 -- is not a
    // race and must be accepted. This is exactly the shape run_dflash2_cases sweeps: every fourth
    // batch row is zero-valid, and order[] repeats table rows across the b%4==3 (zero-valid) and
    // other cases at different batch positions.
    try {
        validate_batch_case({2, {0, 2}, {0, 2}, {3, 3}, MappingPattern::Fragmented, 10u});
    } catch (const std::invalid_argument& error) {
        std::cerr << "validate_batch_case rejected a zero-valid row sharing a table row with a "
                     "writer: "
                  << error.what() << '\n';
        ++failures;
    }
    return failures;
}

int run_softmax_attention_int8_prompt_tests() {
    int failures = verify_int8_prompt_split_capacity();
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    failures += run_int8_prompt_cases();
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention int8-g64 prompt route\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_dflash2_tests() {
    // Profile validation first, and without a GPU: it is pure host logic, and if it is broken the
    // sweep below is the thing it was meant to protect.
    int failures = run_batch_profile_validation_cases();
    if (failures != 0) {
        std::cout << "FAIL causal_softmax_attention DFlash2 (profile validation)\n";
        return 1;
    }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    failures += run_case_allowing_arch_skip("causal_softmax_attention DFlash2 verification",
                                            [] { return run_dflash2_cases(); });
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " causal_softmax_attention DFlash2\n";
    return failures == 0 ? 0 : 1;
}

int run_softmax_attention_causal_cache_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = verify_workspace_capacity_contract();
    failures += run_case_allowing_arch_skip("causal_softmax_attention nvfp4 KV cache (decode)", [] {
        int nvfp4_failures = run_nvfp4_decode_cases();
        nvfp4_failures += run_quantized_batch_cases(KvCacheStorage::Nvfp4Group16, 720u);
        nvfp4_failures += report_quantization_quality(KvCacheStorage::Nvfp4Group16, 724u);
        return nvfp4_failures;
    });
    failures += run_case_allowing_arch_skip("causal_softmax_attention nvfp4 KV cache (prompt)",
                                            [] { return run_nvfp4_prompt_cases(); });
    failures += run_case_allowing_arch_skip("causal_softmax_attention k8v4 KV cache (decode)", [] {
        int k8v4_failures = run_k8v4_decode_cases();
        k8v4_failures += run_quantized_batch_cases(KvCacheStorage::Fp8KeyNvfp4Value, 815u);
        k8v4_failures += report_quantization_quality(KvCacheStorage::Fp8KeyNvfp4Value, 819u);
        return k8v4_failures;
    });
    failures += run_case_allowing_arch_skip("causal_softmax_attention k8v4 KV cache (prompt)",
                                            [] { return run_k8v4_prompt_cases(); });
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    failures += run_case_allowing_arch_skip("causal_softmax_attention FP8 KV cache (decode)",
                                            [&] { return run_fp8_decode_cases(); });
    failures += run_case_allowing_arch_skip("causal_softmax_attention FP8 KV cache (prompt)",
                                            [&] { return run_fp8_prompt_cases(); });
    failures += run_rk8v4_cases();
    failures += run_rk4v4_cases();
    failures += run_bf16_graph_envelope_cases();
    failures += run_bf16_small_prefill_cases();
    failures += run_batch_cases();
    const std::array<int, 7> prefill_queries{0, 63, 64, 127, 128, 511, 1023};
    for (const auto& geometry : kGeometries)
        failures += run_a1_case(geometry, KvCacheStorage::BFloat16, {1024, 8192, 9216, 951u},
                                MappingPattern::Fragmented, prefill_queries);
    failures += run_a3_case(kGeometries[1], KvCacheStorage::BFloat16, {1024, 8192, 9216, 952u},
                            MappingPattern::Fragmented, prefill_queries);
    failures += run_a3_case(kGeometries[0], KvCacheStorage::BFloat16, {1, 131072, 262144, 953u},
                            MappingPattern::Fragmented);
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
