#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/kv_cache/append/launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, Bf16KvInstance Instance, bool Writable, class Input>
void grouped(const Bf16KvOperands& p, Bf16KvCacheView<Writable> cache, Input input,
             const Bf16KvCausalPlan& plan, Bf16KvPartialView partial, cudaStream_t stream) {
    using S           = typename Bf16KvInstanceTraits<Instance>::Schedule;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_bf16_kv_grouped_mma<G, S, MultiBatch, Masked>(p, cache, input, plan.partition,
                                                             partial, stream);
        launch_bf16_kv_merge<G, Bf16KvMergeSchedule<256>, MultiBatch, Masked>(
            p, cache, plan.partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, bool Writable, class Input>
void grouped_instance(const Bf16KvOperands& p, Bf16KvCacheView<Writable> cache, Input input,
                      const Bf16KvCausalPlan& plan, Bf16KvPartialView partial,
                      cudaStream_t stream) {
    switch (plan.instance) {
    case Bf16KvInstance::GroupedDecode:
        return grouped<G, Bf16KvInstance::GroupedDecode>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped32:
        return grouped<G, Bf16KvInstance::Grouped32>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped64:
        return grouped<G, Bf16KvInstance::Grouped64>(p, cache, input, plan, partial, stream);
    default:
        throw std::logic_error("BF16 grouped launch received a tiled plan");
    }
}

void tiled(const Bf16KvOperands& p, Bf16KvReadView cache, const Bf16KvCausalPlan& plan,
           cudaStream_t stream) {
    const auto invoke = [&]<class G>() {
        if (plan.instance == Bf16KvInstance::Tiled128)
            launch_bf16_kv_tiled_mma<
                G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled128>::Schedule>(p, cache,
                                                                                      stream);
        else
            launch_bf16_kv_tiled_mma<
                G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(p, cache,
                                                                                     stream);
    };
    if (p.query_heads == 24)
        invoke.template operator()<Bf16KvD256H24Kv4>();
    else
        invoke.template operator()<Bf16KvD256H16Kv2>();
}

template <class Input>
void execute_grouped(const Tensor& q, Input input, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     const Bf16KvCausalPlan& plan, WorkspaceArena& workspace, Tensor& out,
                     cudaStream_t stream) {
    auto scope         = workspace.scope();
    auto storage       = bf16_kv_allocate_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    const auto p       = bf16_kv_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    const auto view    = bf16_kv_cache_view<Input::writes_cache>(cache, valid, rows);
    const auto partial = storage.view();
    if (p.query_heads == 24)
        grouped_instance<Bf16KvD256H24Kv4>(p, view, input, plan, partial, stream);
    else
        grouped_instance<Bf16KvD256H16Kv2>(p, view, input, plan, partial, stream);
}
} // namespace

void bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan = make_bf16_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope);
    if (!plan.grouped()) {
        kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
        tiled(bf16_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
              bf16_kv_cache_view<false>(cache, &valid_columns, &table_rows), plan, stream);
    } else {
        execute_grouped(q,
                        Bf16KvAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        positions, scale, cache, &valid_columns, &table_rows, plan, workspace, out,
                        stream);
    }
}

void bf16_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan = make_bf16_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (!plan.grouped()) {
        tiled(bf16_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
              bf16_kv_cache_view<false>(view), plan, stream);
    } else {
        execute_grouped(q, Bf16KvCachedInput{}, positions, scale, view, nullptr, nullptr, plan,
                        workspace, out, stream);
    }
}
} // namespace ninfer::ops::detail
