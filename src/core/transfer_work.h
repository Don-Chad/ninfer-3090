#pragma once

#include <cstdint>

namespace ninfer {

// Physical work represented by one logical context transfer. Payload bytes exclude host-arena
// padding. Completed observations count actual cudaMemcpy/cudaMemcpy2D calls. Before Device
// destination allocation, each known Host run is priced as one contiguous Device run using the
// same pool transfer plan; subsequent physical fragmentation can increase the actual count.
struct TransferWork {
    std::uint64_t payload_bytes   = 0;
    std::uint32_t copy_operations = 0;

    [[nodiscard]] friend constexpr bool operator==(TransferWork, TransferWork) noexcept = default;
};

} // namespace ninfer
