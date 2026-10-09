#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string_view>

namespace ninfer::product {

// --reasoning-loop off|stop|conclude, shared by the CLI and ninfer-serve.
[[nodiscard]] inline ReasoningLoopAction parse_reasoning_loop_action(std::string_view value) {
    if (value == "off") { return ReasoningLoopAction::Off; }
    if (value == "stop") { return ReasoningLoopAction::Stop; }
    if (value == "conclude") { return ReasoningLoopAction::Conclude; }
    throw std::invalid_argument("--reasoning-loop must be off, stop or conclude");
}

[[nodiscard]] inline const char* reasoning_loop_name(ReasoningLoopAction action) noexcept {
    switch (action) {
    case ReasoningLoopAction::Stop:
        return "stop";
    case ReasoningLoopAction::Conclude:
        return "conclude";
    case ReasoningLoopAction::Off:
        break;
    }
    return "off";
}

} // namespace ninfer::product
