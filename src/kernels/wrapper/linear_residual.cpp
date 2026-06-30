#include "qus/kernels/linear_residual.h"

#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_down.cuh"
#include "kernels/linear/gemv/linear_rowsplit_gemv_out_6144.cuh"

#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("linear_residual_add: invalid ") + name);
    }
}

void require_q5(const Weight& w) {
    if (w.qtype != QType::Q5G64_F16S || w.group_size != 64 || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_residual_add: weight must be Q5G64_F16S row-split");
    }
}

} // namespace

void linear_residual_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                         WorkspaceArena& ws, cudaStream_t stream) {
    require_q5(w);
    require_tensor(x, DType::BF16, w.k, "x");
    require_tensor(residual_out, DType::BF16, w.n, "residual_out");

    if (w.n == 5120 && w.k == 17408) {
        detail::linear_rowsplit_gemv_mlp_down_residual_q5_launch(x, w, residual_out, ws, stream);
        return;
    }
    if (w.n == 5120 && w.k == 6144) {
        detail::linear_rowsplit_gemv_out_6144_residual_q5_launch(x, w, residual_out, ws, stream);
        return;
    }
    throw std::invalid_argument("linear_residual_add: unsupported Q5 shape");
}

} // namespace qus::kernels
