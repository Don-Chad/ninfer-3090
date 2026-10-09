#pragma once

#include "core/weight.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q5LinearAddScheduleId {
    Split2ExactResidual,
    MmaResidualR64C16,
    MmaResidualR64C24,
    MmaResidualR64C32,
    MmaResidualR64C64,
    MmaResidualR64C32S3,
    MmaResidualR64C32S4,
    MmaResidualR64C128,
    SmallTMmaResidual,
    MmaResidualR64C128Tail,
    // Upstream's sliced-K and narrow MMA tiles. Not routed on sm_86 (the tables in
    // q5_linear_add_plan.cpp); upstream's R64T128 and its narrow-tail composite are the
    // MmaResidualR64C128 and MmaResidualR64C128Tail ids above.
    SlicedR16T8W4S2,
    SlicedR16T16W4S2,
    SlicedR16T24W4S2,
    SlicedR32T32W4S2,
    SlicedR32T24W4S2Pairwise,
    SlicedR32T32W4S1,
    SlicedR32T32W2S2,
    SlicedR32T64W2S1,
    MmaResidualR32T32K128,
    MmaResidualR32T128,
};

struct Q5LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q5LinearAddPlan {
    Q5LinearAddScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept;

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept;
Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem);

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols);

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream);
void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
