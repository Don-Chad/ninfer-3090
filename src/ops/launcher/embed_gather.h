#pragma once

// ninfer::ops::detail - private launch prototypes for embedding variants.

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class Q8EmbedRoute {
    Auto,
    Grouped,
    Row,
};

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream);
void embed_gather_q4_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_q8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream);
void embed_gather_q8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 Q8EmbedRoute route, cudaStream_t stream);

} // namespace ninfer::ops::detail
