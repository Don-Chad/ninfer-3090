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
//
// FP8 KV-cache attention (both small_t decode and the prompt kernel) is fully ported: it
// dequantizes K to BF16 and V to FP16 before ordinary MMA instead of using mma_fp8_e4m3, the same
// fallback pattern this fork's INT8 path already uses.

#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_fp8_a8() {
    throw std::runtime_error("FP8 A8 execution requires an sm_100a or sm_120a GPU");
}

} // namespace

// --- Linear / projection A8 routes -------------------------------------------------------------

// The plans size an A8 partial-sum region even when the route is A16-only; with no A8 kernel on
// this architecture that region is empty.
std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t) { return 0; }
std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t) { return 0; }
std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t) { return 0; }
std::size_t fp8_linear_add_partial_capacity_bytes(std::int32_t, std::int32_t) { return 0; }

void launch_fp8_a8_quantize(const Tensor&, const Weight&, Fp8A8Workspace, cudaStream_t) {
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

} // namespace ninfer::ops::detail
