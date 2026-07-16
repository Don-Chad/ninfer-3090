#pragma once

#include "ops/linear/q6/q6_rowsplit_launch.h"

namespace ninfer::ops::detail {

Q6Problem q6_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept;

bool q6_rowsplit_admits(const Q6Problem& problem) noexcept;
Q6Plan q6_rowsplit_resolve_plan(const Q6Problem& problem);

void q6_rowsplit_execute_plan(Q6Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void q6_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
