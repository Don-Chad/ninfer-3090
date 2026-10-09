// ninfer::ops::detail - asymmetric FP8-K/NVFP4-V append launch ownership.
#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/k8v4_kernel.cuh"
#include "ops/kv_cache/plane_types.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_k8v4_for(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                     Metadata metadata, cudaStream_t stream) {
    // K8V4 is the one storage that mixes scale codings within a single cache: an FP16 scale over
    // the FP8 key plane and a raw E4M3 byte over the NVFP4 value plane. Naming these from the
    // profile keeps that asymmetry in one place -- and note that "k8v4" here means
    // Fp8KeyNvfp4Value, not the similarly abbreviated rk8v4 (RotatedInt8KeyInt4ValueGroup64),
    // whose key plane is INT8 and whose value scale is FP16.
    constexpr auto kStorage = KvCacheStorage::Fp8KeyNvfp4Value;
    const auto tokens       = static_cast<std::int32_t>(k.ne[2]);
    auto* cache_k           = static_cast<KvKeyCodeT<kStorage>*>(cache.k_pages.data);
    auto* cache_v           = static_cast<KvValueCodeT<kStorage>*>(cache.v_pages.data);
    auto* scale_k           = static_cast<KvKeyScaleT<kStorage>*>(cache.k_scale_pages.data);
    auto* scale_v           = static_cast<KvValueScaleT<kStorage>*>(cache.v_scale_pages.data);
    assert_kv_planes<kStorage, decltype(cache_k), decltype(cache_v), decltype(scale_k),
                     decltype(scale_v)>();
    if (tokens >= 128 && Geometry::KVHeads == 2) {
        constexpr int TokensPerTile = 8;
        const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
        const dim3 grid(static_cast<unsigned>(max_tiles), static_cast<unsigned>(Geometry::KVHeads));
        kv_cache_append_full_k8v4_page_kernel<Geometry, Metadata><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v, scale_k,
            scale_v, tokens);
    } else {
        constexpr int FillWarps       = kBlock / 32;
        const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
        const int grid = static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
        kv_cache_append_full_k8v4_kernel<Geometry, Metadata><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata, cache_k, cache_v, scale_k,
            scale_v, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void dispatch_k8v4(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                   Metadata metadata, cudaStream_t stream) {
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_k8v4_for<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_k8v4_for<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

} // namespace

void kv_cache_append_k8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                 PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch_k8v4(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_k8v4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                       const Tensor& valid_columns, const Tensor& table_rows,
                                       PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (k.ne[3] > 1) {
            const auto append = [&]<class G>() {
                const dim3 grid(div_up(k.ne[2] * G::KVHeads, 8), 1, k.ne[3]);
                kv_cache_append_full_k8v4_kernel<G, PagedKVBatchMetadata<Masked>, true>
                    <<<grid, kBlock, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(k.data),
                        static_cast<const __nv_bfloat16*>(v.data),
                        static_cast<const std::int32_t*>(positions.data), metadata,
                        static_cast<std::uint8_t*>(cache.k_pages.data),
                        static_cast<std::uint8_t*>(cache.v_pages.data),
                        static_cast<__half*>(cache.k_scale_pages.data),
                        static_cast<std::uint8_t*>(cache.v_scale_pages.data), k.ne[2]);
            };
            if (k.ne[1] == 4)
                append.template operator()<KVCacheAppendD256Kv4>();
            else
                append.template operator()<KVCacheAppendD256Kv2>();
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        dispatch_k8v4(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
