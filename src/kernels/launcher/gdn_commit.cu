#include "kernels/launcher/gdn_commit.h"

#include "kernels/kernel/gdn_commit.cuh"
#include "qus/core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

int grid_for_bytes(std::int64_t bytes, int block) {
    const std::int64_t grid = (bytes + static_cast<std::int64_t>(block) - 1) / block;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("gdn_commit: grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

} // namespace

void gdn_commit_launch(Tensor& conv_states, Tensor& ssm_states, const Tensor& accepted,
                       cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int64_t conv_slot_bytes =
        static_cast<std::int64_t>(conv_states.ne[0]) * conv_states.ne[1] *
        static_cast<std::int64_t>(dtype_size(conv_states.dtype));
    const std::int64_t ssm_slot_bytes =
        static_cast<std::int64_t>(ssm_states.ne[0]) * ssm_states.ne[1] * ssm_states.ne[2] *
        static_cast<std::int64_t>(dtype_size(ssm_states.dtype));
    const std::int32_t slots = conv_states.ne[2];

    gdn_commit_copy_kernel<<<grid_for_bytes(conv_slot_bytes, kBlock), kBlock, 0, stream>>>(
        static_cast<unsigned char*>(conv_states.data), conv_slot_bytes, slots,
        static_cast<const std::int32_t*>(accepted.data));
    CUDA_CHECK(cudaGetLastError());
    gdn_commit_copy_kernel<<<grid_for_bytes(ssm_slot_bytes, kBlock), kBlock, 0, stream>>>(
        static_cast<unsigned char*>(ssm_states.data), ssm_slot_bytes, slots,
        static_cast<const std::int32_t*>(accepted.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
