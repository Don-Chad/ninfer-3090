#pragma once

#include "core/tensor.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kNvfp4GdnQueryRows = 2048;
inline constexpr std::int32_t kNvfp4GdnKeyRows   = 2048;
inline constexpr std::int32_t kNvfp4GdnValueRows = 6144;
inline constexpr std::int32_t kNvfp4GdnChannels =
    kNvfp4GdnQueryRows + kNvfp4GdnKeyRows + kNvfp4GdnValueRows;
inline constexpr std::int32_t kNvfp4GdnZRows      = 6144;
inline constexpr std::int32_t kNvfp4GdnParentRows = kNvfp4GdnChannels + kNvfp4GdnZRows;

template <int Tokens, class Publish>
struct Nvfp4GdnConvOutput {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    __device__ __forceinline__ void store_row(std::int32_t parent_row,
                                              const float (&projected)[Tokens]) const {
        if (parent_row < kNvfp4GdnChannels) {
            conv.store(parent_row, projected);
            return;
        }
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            z[static_cast<std::int64_t>(token) * kNvfp4GdnZRows + parent_row - kNvfp4GdnChannels] =
                __float2bfloat16_rn(projected[token]);
        }
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float projected) const {
        static_assert(Tokens == 1);
        (void)token;
        const float row[1]{projected};
        store_row(parent_row, row);
    }
};

template <int Tokens, class Publish>
Nvfp4GdnConvOutput<Tokens, Publish>
make_nvfp4_gdn_conv_output(const Tensor& conv_weight, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, Publish publish) {
    return {
        {
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data),
            kNvfp4GdnChannels,
            kNvfp4GdnQueryRows,
            kNvfp4GdnKeyRows,
            kNvfp4GdnValueRows,
            0,
            Tokens,
            0,
            publish,
        },
        static_cast<__nv_bfloat16*>(z.data),
    };
}

} // namespace ninfer::ops::detail
