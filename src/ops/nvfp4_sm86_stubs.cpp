#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_nvfp4_a4() {
    throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
}

} // namespace

void launch_nvfp4_a4_quantize(const Tensor&, const Weight&, Nvfp4A4Workspace, Nvfp4ScaleLayout,
                              cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_swiglu_a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                   cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_add_a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace,
                                cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_attn_input_a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                Nvfp4A4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_gdn_input_a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Nvfp4A4Workspace,
                               cudaStream_t) {
    reject_nvfp4_a4();
}

} // namespace ninfer::ops::detail
