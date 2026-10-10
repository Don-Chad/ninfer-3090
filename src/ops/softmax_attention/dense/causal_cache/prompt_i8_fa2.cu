// ninfer::ops - INT8-G64 causal prompt attention launcher for the FA2-style kernel: plan the CTA
// shape and key splits, launch the attention kernel and, for a split launch, the FP32 merge.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fa2.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fa2_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename Metadata>
void launch_for(const Tensor& q, const Tensor& positions, float scale, const Tensor& k_pages,
                const Tensor& v_pages, const Tensor& k_scale_pages, const Tensor& v_scale_pages,
                Metadata metadata, CausalAttentionExecutionEnvelope envelope,
                WorkspaceArena& workspace, std::int32_t multiprocessor_count, Tensor& out,
                cudaStream_t stream) {
    const auto width = static_cast<std::int32_t>(q.ne[2]);
    const CausalPromptFa2Plan plan = causal_prompt_fa2_plan(
        Geometry::QHeads, width, envelope.max_visible_keys, multiprocessor_count);
    CausalPromptFa2Partials partials{};
    if (plan.splits > 1) {
        partials =
            allocate_causal_prompt_fa2_partials(workspace, Geometry::QHeads, width, plan.splits);
    }

    const auto launch = [&]<int Warps, bool Split>() {
        using Shape = CausalPromptFa2Shape<Warps>;
        const auto kernel =
            causal_attention_prompt_i8_fa2_kernel<Geometry, Metadata, Warps, Split>;
        configure_cuda_device_once([] {
            return cudaFuncSetAttribute(
                causal_attention_prompt_i8_fa2_kernel<Geometry, Metadata, Warps, Split>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
        });
        const dim3 grid(static_cast<unsigned>(div_up(width, Shape::Br)),
                        static_cast<unsigned>(Geometry::QHeads),
                        static_cast<unsigned>(plan.splits));
        kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const std::int8_t*>(k_pages.data),
            static_cast<const std::int8_t*>(v_pages.data),
            static_cast<const __half*>(k_scale_pages.data),
            static_cast<const __half*>(v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), width, static_cast<float*>(partials.rows.data),
            static_cast<float2*>(partials.stats.data));
        CUDA_CHECK(cudaGetLastError());
    };

    if (plan.splits == 1) {
        if (plan.warps == 4) {
            launch.template operator()<4, false>();
        } else {
            launch.template operator()<8, false>();
        }
        return;
    }
    // Only eight-warp CTAs split.
    launch.template operator()<8, true>();
    constexpr float Log2E = 1.4426950408889634074f;
    causal_attention_prompt_i8_fa2_merge_kernel<Geometry, Metadata>
        <<<dim3(static_cast<unsigned>(width), static_cast<unsigned>(Geometry::QHeads)),
           kCausalPromptHeadDim, 0, stream>>>(
            static_cast<const float*>(partials.rows.data),
            static_cast<const float2*>(partials.stats.data), metadata, width, plan.splits,
            scale * Log2E, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Metadata, typename Cache>
void launch_geometry(const Tensor& q, const Tensor& positions, float scale, const Cache& cache,
                     Metadata metadata, CausalAttentionExecutionEnvelope envelope,
                     WorkspaceArena& workspace, std::int32_t multiprocessor_count, Tensor& out,
                     cudaStream_t stream) {
    if (cache.storage != KvCacheStorage::Int8Group64) {
        throw std::logic_error("FA2 prompt attention serves the INT8-G64 cache only");
    }
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_for<CausalD256H24Kv4>(q, positions, scale, cache.k_pages, cache.v_pages,
                                     cache.k_scale_pages, cache.v_scale_pages, metadata, envelope,
                                     workspace, multiprocessor_count, out, stream);
        return;
    }
    launch_for<CausalD256H16Kv2>(q, positions, scale, cache.k_pages, cache.v_pages,
                                 cache.k_scale_pages, cache.v_scale_pages, metadata, envelope,
                                 workspace, multiprocessor_count, out, stream);
}

} // namespace

void causal_attention_prompt_i8_fa2_attention_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
    std::int32_t multiprocessor_count, Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    launch_geometry(q, positions, scale, cache, metadata, envelope, workspace,
                    multiprocessor_count, out, stream);
}

void causal_attention_prompt_i8_fa2_batch_attention_launch(
    const Tensor& q, const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, const PagedKVBatchLayerView& cache, CausalAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, std::int32_t multiprocessor_count, Tensor& out,
    cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        launch_geometry(q, positions, scale, cache, metadata, envelope, workspace,
                        multiprocessor_count, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
