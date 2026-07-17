#pragma once

#include "core/tensor.h"
#include "ops/linear/q4/q4_rowsplit_launch.h"
#include "ops/linear/q5/q5_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_q5_attn_input_small_t_launch(Q4Plan q4_plan, Q5Plan q5_plan, const Tensor& x,
                                     const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void q4_q5_attn_input_grouped_mma_launch(Q4KernelVariant variant, const Tensor& x,
                                         const Weight& query_key_weight,
                                         const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
