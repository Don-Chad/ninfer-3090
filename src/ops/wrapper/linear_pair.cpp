#include "ninfer/ops/linear_pair.h"

#include "ops/linear_pair/w8/w8_pair_plan.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

void require_weight(const Weight& weight, const char* label) {
    constexpr std::uint64_t kPayloadBytes = 5'570'560;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != 1024 || weight.k != 5120 || weight.shape[0] != 1024 ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != 1024 ||
        weight.padded_shape[1] != 5120 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < kPayloadBytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

} // namespace

void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    require_matrix(x, 5120, cols, "x");
    require_matrix(first_out, 1024, cols, "first output");
    require_matrix(second_out, 1024, cols, "second output");
    require_weight(first_weight, "first weight");
    require_weight(second_weight, "second weight");

    (void)ws;
    detail::w8_pair_dispatch(x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops
