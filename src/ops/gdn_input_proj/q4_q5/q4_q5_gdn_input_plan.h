#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q5GdnInputScheduleId {
    IndependentDirectFixed,
    GroupedMixedMmaR64C8,
    GroupedMixedMmaR64C16,
    GroupedMixedMmaR64C32,
    GroupedMixedMmaR64C64,
    GroupedMixedMmaR64C128,
    SmallTMma,
    // Upstream's column-band tiles. Not routed on sm_86 (kRoutes); reachable through
    // q4_q5_gdn_input_execute_schedule for the schedule bench.
    GroupedMixedMmaR32C32S2,
    GroupedMixedMmaR32C64S4,
    GroupedMixedMmaR64C128S2,
};

struct Q4Q5GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qk_rows;
    std::int32_t value_z_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q5GdnInputPlan {
    Q4Q5GdnInputScheduleId schedule;
};

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept;

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept;
Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem);

// Runs a schedule without first checking that it is the one resolve_plan would pick;
// q4_q5_gdn_input_execute_plan is exactly this plus that check.
//
// It exists so a bench can time every candidate schedule at the same column count, which is the
// only way to tell whether a route boundary sits in the right place. This Op needed it most and
// had it last: its {{1, 6}} / {{7, 32}} boundary is where DFlash2 loses 15% between five and six
// draft tokens, and where a C8 decode cohort starts paying a 32-wide tile for eight live columns,
// and until now neither could be checked against the alternative. Mirrors
// q4_linear_swiglu_execute_schedule and q8_pair_execute_schedule.
//
// Nothing on the inference path should call this: the check execute_plan adds is what keeps a plan
// from being executed against a problem it was not resolved for.
void q4_q5_gdn_input_execute_schedule(Q4Q5GdnInputScheduleId schedule, const Tensor& x,
                                      const Weight& qk_weight, const Weight& value_z_weight,
                                      Tensor& qkv, Tensor& z, cudaStream_t stream);
void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, cudaStream_t stream);
void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream);

// Integer-activation route, registered for the 27B profile at full prefill tiles. It declines
// everything else so the caller falls back to the A16 schedules rather than producing a wrong
// answer; decode and partial chunks depend on that.
[[nodiscard]] bool q4_q5_gdn_input_a8_supported(const Weight& qk, const Weight& value_z,
                                                std::int32_t tokens);
[[nodiscard]] std::size_t q4_q5_gdn_input_a8_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);
void q4_q5_gdn_input_a8_launch(const Tensor& x, const Weight& qk, const Weight& value_z,
                               Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
