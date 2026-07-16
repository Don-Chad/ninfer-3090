#include "ops/linear/bf16/bf16_contiguous_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Bf16Problem bf16_contiguous_problem(const Tensor& x, const Weight& w,
                                    const Tensor& /*out*/) noexcept {
    return {w.n, w.k, x.ne[1]};
}

bool bf16_contiguous_admits(const Bf16Problem& /*problem*/) noexcept { return false; }

[[noreturn]] void bf16_contiguous_dispatch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t /*stream*/) {
    const Bf16Problem problem = bf16_contiguous_problem(x, w, out);
    if (!bf16_contiguous_admits(problem)) {
        throw std::invalid_argument("bf16 linear: no pure Linear problem is registered");
    }
    throw std::logic_error("bf16 linear: admitted problem has no schedule");
}

} // namespace ninfer::ops::detail
