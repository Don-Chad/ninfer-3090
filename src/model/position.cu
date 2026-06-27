#include "model/position.h"

#include "qus/core/device.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::model::detail {
namespace {

constexpr int kBlock = 256;

void require_i32_contiguous_nonnull(const Tensor& t, const char* name) {
    if (t.dtype != DType::I32) {
        throw std::invalid_argument(std::string(name) + ": tensor must be I32");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + ": tensor must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(name) + ": tensor data must be non-null");
    }
}

void require_pos_shape(const Tensor& pos, const char* name) {
    if (pos.ne[0] != 1 || pos.ne[1] != 1 || pos.ne[2] != 1 || pos.ne[3] != 1) {
        throw std::invalid_argument(std::string(name) + ": pos must have shape [1]");
    }
}

void require_vector_shape(const Tensor& t, const char* name) {
    if (t.ne[0] <= 0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(name) + ": positions must have shape [T]");
    }
}

__global__ void fill_positions_kernel(std::int32_t* positions, std::int32_t n) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { positions[i] = i; }
}

__global__ void set_pos_kernel(std::int32_t* pos, std::int32_t value) { pos[0] = value; }

__global__ void advance_pos_kernel(std::int32_t* pos) { ++pos[0]; }

} // namespace

void fill_positions(Tensor& positions, cudaStream_t stream) {
    require_i32_contiguous_nonnull(positions, "fill_positions");
    require_vector_shape(positions, "fill_positions");
    const int n    = positions.ne[0];
    const int grid = (n + kBlock - 1) / kBlock;
    fill_positions_kernel<<<grid, kBlock, 0, stream>>>(static_cast<std::int32_t*>(positions.data),
                                                       n);
    CUDA_CHECK(cudaGetLastError());
}

void set_pos(Tensor& pos, int value, cudaStream_t stream) {
    require_i32_contiguous_nonnull(pos, "set_pos");
    require_pos_shape(pos, "set_pos");
    set_pos_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(pos.data), value);
    CUDA_CHECK(cudaGetLastError());
}

void advance_pos(Tensor& pos, cudaStream_t stream) {
    require_i32_contiguous_nonnull(pos, "advance_pos");
    require_pos_shape(pos, "advance_pos");
    advance_pos_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(pos.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::model::detail
