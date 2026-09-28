#pragma once

#include "core/paged_kv_cache.h"
#include "core/arena.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace ninfer::ops::detail {

struct Bf16KvOperands {
    const __nv_bfloat16* q;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    float scale;
    int width;
    int batch;
    int visible_capacity;
    int head_dim;
    int query_heads;
};

template <bool Writable>
struct Bf16KvCacheView {
    static constexpr bool kWritable = Writable;
    using Key   = std::conditional_t<Writable, __nv_bfloat16, const __nv_bfloat16>;
    // V is pure BF16 in the cache (this fork's KV-plane typing), not upstream's FP16 V.
    using Value = std::conditional_t<Writable, __nv_bfloat16, const __nv_bfloat16>;
    Key* keys;
    Value* values;
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride;
    int head_dim;
    int kv_heads;
};

using Bf16KvReadView  = Bf16KvCacheView<false>;
using Bf16KvWriteView = Bf16KvCacheView<true>;

struct Bf16KvAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct Bf16KvCachedInput {
    static constexpr bool writes_cache = false;
};

struct Bf16KvPartialView {
    float* acc;
    float* maximum;
    float* sum;
    int splits;
};

inline Bf16KvOperands bf16_kv_operands(const Tensor& q, const Tensor& positions, Tensor& out,
                                       float scale, int capacity) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<__nv_bfloat16*>(out.data),
            scale,
            q.ne[2],
            q.ne[3],
            capacity,
            q.ne[0],
            q.ne[1]};
}

template <bool Writable>
Bf16KvCacheView<Writable> bf16_kv_cache_view(const PagedKVBatchLayerView& cache,
                                             const Tensor* valid = nullptr,
                                             const Tensor* rows  = nullptr) {
    return {static_cast<typename Bf16KvCacheView<Writable>::Key*>(cache.k_pages.data),
            static_cast<typename Bf16KvCacheView<Writable>::Value*>(cache.v_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.block_tables.ne[0],
            cache.head_dim,
            cache.num_kv_heads};
}

struct Bf16KvPartialStorage {
    Tensor acc, maximum, sum;

    Bf16KvPartialView view(int splits) const {
        return {static_cast<float*>(acc.data), static_cast<float*>(maximum.data),
                static_cast<float*>(sum.data), splits};
    }
};

template <class Allocator>
Bf16KvPartialStorage bf16_kv_allocate_partials(Allocator& allocator, int heads, int width,
                                               int splits, int batch) {
    return {allocator.alloc(DType::FP32, {256, heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch})};
}

} // namespace ninfer::ops::detail
