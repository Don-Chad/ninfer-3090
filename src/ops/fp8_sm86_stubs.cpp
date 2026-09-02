// sm_86 replacements for the FP8 A8 compute path.
//
// Every FP8 A8 route bottoms out in mma_fp8_e4m3, which emits
// `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`. The `kind::f8f6f4`
// qualifier is a Blackwell (sm_100a/sm_120a) instruction; Ampere has no FP8 tensor-core path at
// all, so the owning translation units cannot be compiled for sm_86 and are filtered out of
// ninfer_ops. These definitions keep the Op boundary link-complete and convert a mis-selected
// route into a precise runtime error instead of a link failure.
//
// FP8 *weights* remain usable on sm_86: LinearPolicy::A16Only routes them through the A16
// dequantizing GEMM (see kFp8TextPolicy in targets/qwen3_6_27b/impl/variant.cpp), whose
// translation units are compiled normally. Only A8 activation compute is unavailable.

#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_fp8_a8() {
    throw std::runtime_error("FP8 A8 execution requires an sm_100a or sm_120a GPU");
}

[[noreturn]] void reject_fp8_kv() {
    throw std::runtime_error(
        "FP8 E4M3 KV-cache attention requires an sm_100a or sm_120a GPU; use --kv-dtype int8 "
        "or --kv-dtype bf16");
}

} // namespace

// --- Linear / projection A8 routes -------------------------------------------------------------

void launch_fp8_a8_quantize(const Tensor&, const Weight&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void fp8_attn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void fp8_gdn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    reject_fp8_a8();
}

void fp8_linear_add_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                              cudaStream_t) {
    reject_fp8_a8();
}

void fp8_linear_swiglu_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t) {
    reject_fp8_a8();
}

// --- FP8 KV-cache causal attention -------------------------------------------------------------

void causal_attention_small_t_fp8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, float,
                                         PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                         std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                         Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

void causal_attention_cached_small_t_fp8_launch(const Tensor&, const Tensor&, float,
                                                const PagedKVLayerView&,
                                                CausalAttentionExecutionEnvelope, Tensor&, Tensor&,
                                                Tensor&, Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

void causal_attention_prompt_fp8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                        Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

void causal_attention_prompt_fp8_attention_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&, Tensor&, cudaStream_t) {
    reject_fp8_kv();
}

} // namespace ninfer::ops::detail
