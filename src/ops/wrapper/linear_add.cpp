#include "ninfer/ops/linear_add.h"

#include "ops/linear/reference/linear_generic.h"
#include "ops/linear/gemv/linear_rowsplit_gemv_mlp_down.cuh"
#include "ops/linear/gemv/linear_rowsplit_gemv_out_6144.cuh"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t columns,
                    const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != columns || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("linear_add: invalid ") + name);
    }
}

void require_q5(const Weight& w) {
    if (w.qtype != QType::Q5G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 64 || w.group != 64 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be Q5G64_F16S row-split");
    }
}

} // namespace

std::size_t linear_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                       std::int32_t tokens) {
    if (output_rows <= 0 || input_rows <= 0 || tokens <= 0) {
        throw std::invalid_argument("linear_add_workspace_bytes: dimensions must be positive");
    }
    if (tokens == 1 || tokens > 16) { return 0; }
    return Tensor(nullptr, DType::BF16, {output_rows, tokens}).bytes();
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, WorkspaceArena& ws,
                cudaStream_t stream) {
    require_q5(w);
    const std::int32_t t = x.ne[1];
    require_tensor(x, DType::BF16, w.k, t, "x");
    require_tensor(residual_out, DType::BF16, w.n, t, "residual_out");

    const bool supported_shape = (w.n == 5120 && w.k == 17408) || (w.n == 5120 && w.k == 6144);
    if (!supported_shape) { throw std::invalid_argument("linear_add: unsupported Q5 shape"); }

    if (t > 16 && (w.k % 8) == 0) {
        detail::linear_rowsplit_gemm_mma_residual_q5_launch(x, w, residual_out, stream);
        return;
    }

    if (t == 1 && w.k == 17408) {
        detail::linear_rowsplit_gemv_mlp_down_residual_q5_launch(x, w, residual_out, ws, stream);
        return;
    }
    if (t == 1 && w.k == 6144) {
        detail::linear_rowsplit_gemv_out_6144_residual_q5_launch(x, w, residual_out, ws, stream);
        return;
    }

    auto scratch_scope = ws.scope();
    Tensor linear_out  = ws.alloc(DType::BF16, {w.n, t});
    linear(x, w, linear_out, ws, stream);
    residual_add(linear_out, residual_out, stream);
}

} // namespace ninfer::ops
