#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Bf16Problem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t cols;
};

// Reserved format boundary. The registry is intentionally empty until a BF16
// pure-Linear implementation and its exact product problems are qualified.
Bf16Problem bf16_contiguous_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept;
bool bf16_contiguous_admits(const Bf16Problem& problem) noexcept;

[[noreturn]] void bf16_contiguous_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
