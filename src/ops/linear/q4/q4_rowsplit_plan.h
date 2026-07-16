#pragma once

#include "ops/linear/q4/q4_rowsplit_launch.h"

namespace ninfer::ops::detail {

Q4Problem q4_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept;

bool q4_rowsplit_admits(const Q4Problem& problem) noexcept;
Q4Plan q4_rowsplit_resolve_plan(const Q4Problem& problem);

void q4_rowsplit_execute_plan(Q4Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void q4_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
