#pragma once

#include <cstdint>

namespace qus::kernels {

__global__ void gdn_commit_copy_kernel(unsigned char* data, std::int64_t slot_bytes,
                                       std::int32_t slots, const std::int32_t* accepted) {
    const std::int32_t a = accepted[0];
    if (a <= 0 || a >= slots) { return; }

    const std::int64_t start =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const unsigned char* src  = data + static_cast<std::int64_t>(a) * slot_bytes;
    for (std::int64_t i = start; i < slot_bytes; i += stride) { data[i] = src[i]; }
}

} // namespace qus::kernels
