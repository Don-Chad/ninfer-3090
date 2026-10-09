#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream);
void q4_q5_gdn_input_small_t_launch(const Tensor& x, const Weight& qk_weight,
                                    const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                    Tensor& z, cudaStream_t stream);

void q4_q5_gdn_input_grouped_mma_c8_launch(const Tensor& x, const Weight& qk_weight,
                                           const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                           cudaStream_t stream);

void q4_q5_gdn_input_grouped_mma_c16_launch(const Tensor& x, const Weight& qk_weight,
                                            const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                            cudaStream_t stream);

void q4_q5_gdn_input_grouped_mma_c32_launch(const Tensor& x, const Weight& qk_weight,
                                            const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                            cudaStream_t stream);

void q4_q5_gdn_input_grouped_mma_c64_launch(const Tensor& x, const Weight& qk_weight,
                                            const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                            cudaStream_t stream);

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        Q4Q5GdnInputScheduleId schedule, cudaStream_t stream);

} // namespace ninfer::ops::detail
