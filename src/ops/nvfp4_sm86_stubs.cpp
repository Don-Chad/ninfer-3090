#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_nvfp4_a4() {
    throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
}

// NVFP4/K8V4 KV-cache formats are recognized (parsed, formatted, logged) but not yet ported on
// this fork: their causal-attention and kv_cache_append kernels use the Blackwell-only
// cvt.rn.satfinite.e2m1x2 PTX instruction, which has no sm_86 encoding. d256_kv_cache_profile and
// paged_kv_storage_layout already reject these KvCacheStorage selections before any cache reaches
// these Ops, so the definitions below only guard the link boundary.
[[noreturn]] void reject_nvfp4_kv() {
    throw std::runtime_error(
        "NVFP4 KV-cache attention requires an sm_100a or sm_120a GPU; use --kv-dtype bf16, int8, "
        "or rk8v4");
}

[[noreturn]] void reject_k8v4_kv() {
    throw std::runtime_error(
        "K8V4 (FP8 key / NVFP4 value) KV-cache attention requires an sm_100a or sm_120a GPU; use "
        "--kv-dtype bf16, int8, or rk8v4");
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_w4a4(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

// --- NVFP4/K8V4 KV-cache causal attention -------------------------------------------------------

void causal_attention_small_t_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                           const Tensor&, const Tensor&, const Tensor&, float,
                                           PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                           std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                           Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&,
                                                  CausalAttentionExecutionEnvelope, Tensor&,
                                                  Tensor&, Tensor&, Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_prompt_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_prompt_nvfp4_attention_launch(const Tensor&, const Tensor&, float,
                                                    const PagedKVLayerView&, Tensor&,
                                                    cudaStream_t) {
    reject_nvfp4_kv();
}

void causal_attention_small_t_k8v4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, CausalAttentionExecutionEnvelope,
                                          std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
                                          Tensor&, cudaStream_t) {
    reject_k8v4_kv();
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&,
                                                 CausalAttentionExecutionEnvelope, Tensor&, Tensor&,
                                                 Tensor&, Tensor&, cudaStream_t) {
    reject_k8v4_kv();
}

void causal_attention_prompt_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                         Tensor&, cudaStream_t) {
    reject_k8v4_kv();
}

void causal_attention_prompt_k8v4_attention_launch(const Tensor&, const Tensor&, float,
                                                   const PagedKVLayerView&, Tensor&, cudaStream_t) {
    reject_k8v4_kv();
}

// --- NVFP4/K8V4 KV-cache append ------------------------------------------------------------------

void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    reject_nvfp4_kv();
}

void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    reject_nvfp4_kv();
}

void kv_cache_append_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                 cudaStream_t) {
    reject_k8v4_kv();
}

void kv_cache_append_k8v4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    reject_k8v4_kv();
}

} // namespace ninfer::ops::detail
