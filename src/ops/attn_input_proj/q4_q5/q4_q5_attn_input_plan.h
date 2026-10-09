#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q5AttnInputScheduleId {
    ParentSplitFixed,
    GroupedHomogeneousPairMmaR32C32S4,
    GroupedHomogeneousPairMmaR32C64S4,
    MixedR32C32S2,
    MixedR32C64S3,
    PairR32C64S3,
    MixedR64C128S2,
    PairR32C64S4,
    SmallTMma,
};

struct Q4Q5AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q5AttnInputPlan {
    Q4Q5AttnInputScheduleId schedule;
};

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept;

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept;
Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem);

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   cudaStream_t stream);
void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream);

// Integer-activation route, registered for the 27B profile at full prefill tiles. It declines
// everything else so the caller falls back to the A16 schedules; decode and partial chunks depend
// on that.
[[nodiscard]] bool q4_q5_attn_input_a8_supported(const Weight& query_key, const Weight& gate_value,
                                                 std::int32_t tokens);
[[nodiscard]] std::size_t q4_q5_attn_input_a8_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);
void q4_q5_attn_input_a8_launch(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
