#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q6ScheduleId {
    SimtR8C4,
    SimtR8C8,
    MmaR64C64,
    MmaR64C128,
};

enum class Q6KernelVariant {
    None,
    Full,
    Predicated,
};

struct Q6Problem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q6Plan {
    Q6ScheduleId schedule;
    Q6KernelVariant variant;
};

const char* q6_schedule_name(Q6ScheduleId schedule);
const char* q6_kernel_variant_name(Q6KernelVariant variant);
bool q6_schedule_uses_mma(Q6ScheduleId schedule);

bool q6_candidate_is_legal(Q6ScheduleId schedule, Q6KernelVariant variant,
                           const Q6Problem& problem);

void q6_rowsplit_launch_fixed(Q6Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void q6_rowsplit_launch_candidate(Q6ScheduleId schedule, Q6KernelVariant variant, const Tensor& x,
                                  const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
