// SM8X stubs for FP8 routes that require the Blackwell-only mma .kind::f8f6f4
// encoding. The A16 decode routes stay compiled, LinearPolicy is forced to
// A16Only under NINFER_SM8X_COMPAT, and the FP8 KV cache is rejected at
// startup, so none of these can be reached on a correctly configured engine.

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
    throw std::runtime_error("FP8 A8 execution requires Blackwell mma support (sm_120a)");
}

} // namespace

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

void causal_attention_small_t_fp8_launch(const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, const Tensor&, float,
                                         PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                         std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                         Tensor&, cudaStream_t) {
    reject_fp8_a8();
}

void causal_attention_cached_small_t_fp8_launch(const Tensor&, const Tensor&, float,
                                                const PagedKVLayerView&,
                                                CausalAttentionExecutionEnvelope, Tensor&, Tensor&,
                                                Tensor&, Tensor&, cudaStream_t) {
    reject_fp8_a8();
}

void causal_attention_prompt_fp8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                        Tensor&, cudaStream_t) {
    reject_fp8_a8();
}

void causal_attention_prompt_fp8_attention_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&, Tensor&, cudaStream_t) {
    reject_fp8_a8();
}

} // namespace ninfer::ops::detail
