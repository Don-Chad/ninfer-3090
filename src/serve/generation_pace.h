#pragma once

// How fast one finished request streamed its tokens after the first. A healthy engine decodes at
// a few milliseconds per token; the "slow state" of issue #208 shows as a fixed multi-second stall
// right after the first token, which inflates this by orders of magnitude even for a short answer.
// A supervisor can compare it against its own threshold to restart a healthy-but-slow process.

#include <cstdint>

namespace ninfer::serve {

struct GenerationPace {
    std::uint32_t completion_tokens = 0;
    // First through last accepted output token (N-1 token intervals).
    double generation_wall_seconds = 0.0;
    // Host time decode rounds exposed to this request, device wait excluded.
    double decode_host_seconds = 0.0;

    [[nodiscard]] std::uint64_t token_intervals() const noexcept {
        return completion_tokens > 1 ? completion_tokens - 1 : 0;
    }
    [[nodiscard]] double inter_token_seconds() const noexcept {
        const std::uint64_t intervals = token_intervals();
        return intervals == 0 ? 0.0 : generation_wall_seconds / static_cast<double>(intervals);
    }
};

} // namespace ninfer::serve
