// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fa2_plan.h"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // BFloat16 storage never reaches this function: both causal_softmax_attention and
    // causal_softmax_attention_cached early-return to bf16_kv_{append,cached}_attention
    // before route resolution. rk8v4 and rk4v4 always land here; INT8-G64 only over at most
    // kCausalPromptFa2MinVisibleKeys keys and takes the FA2-style kernel (prompt_i8_fa2.cu) above.
    configure_cuda_device_once([&] {
        return cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    });
    configure_cuda_device_once([&] {
        return cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    });
    configure_cuda_device_once([&] {
        return cudaFuncSetAttribute(
            causal_attention_prompt_i8_kernel<Geometry, Metadata, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    });

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    // The INT8 family is the one place that deliberately does not take its plane types from
    // ops/kv_cache/plane_types.h. One kernel serves both codings, so its value parameter is
    // std::int8_t* for int8-g64 and for rk8v4 alike, and the packed-int4 path re-casts to
    // std::uint8_t where it unpacks (causal_prompt_i4_dequant_f16x8). Substituting
    // KvValueCodeT<RotatedInt8KeyInt4ValueGroup64>, which is U8 because the profile describes
    // the plane's storage rather than this kernel's signature, would not be a cleanup. Leave
    // it; the dtype check below is the guard that matters here.
    // A U8 key plane is the rk4v4 Lloyd-Max coding; a U8 value plane is the packed signed
    // int4 coding it shares with rk8v4.
    if (cache_k.dtype == DType::U8) {
        causal_attention_prompt_i8_kernel<Geometry, Metadata, true, true>
            <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else if (cache_v.dtype == DType::U8) {
        causal_attention_prompt_i8_kernel<Geometry, Metadata, true>
            <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else {
        causal_attention_prompt_i8_kernel<Geometry, Metadata, false>
            <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache,
                                              CausalAttentionExecutionEnvelope envelope,
                                              WorkspaceArena& workspace,
                                              std::int32_t multiprocessor_count, Tensor& out,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Int8Group64 &&
        envelope.max_visible_keys > kCausalPromptFa2MinVisibleKeys) {
        causal_attention_prompt_i8_fa2_attention_launch(q, positions, scale, cache, envelope,
                                                        workspace, multiprocessor_count, out,
                                                        stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache,
                                    CausalAttentionExecutionEnvelope envelope,
                                    WorkspaceArena& workspace, std::int32_t multiprocessor_count,
                                    Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    if (cache.storage == KvCacheStorage::Int8Group64 &&
        envelope.max_visible_keys > kCausalPromptFa2MinVisibleKeys) {
        causal_attention_prompt_i8_fa2_batch_attention_launch(q, positions, valid_columns,
                                                              table_rows, scale, cache, envelope,
                                                              workspace, multiprocessor_count, out,
                                                              stream);
        return;
    }
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
