#pragma once

#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>
#include <ninfer/targets/qwen3_6/diagnostics.h>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

namespace schedule {
using Phase             = qwen3_6::TextPhase;
using TapId             = qwen3_6::TextTapId;
using VisionTapId       = qwen3_6::VisionTapId;
using TextTapCallback   = qwen3_6::TextTapCallback;
using VisionTapCallback = qwen3_6::VisionTapCallback;
} // namespace schedule

// Target-private diagnostic attachment. The product Engine never exposes this seam; the dedicated
// activation-dump executable uses it to observe the exact Program schedule it executes.
class ActivationDumpAccess {
public:
    static void attach(Package::Program& program, void* context, schedule::TextTapCallback text,
                       schedule::VisionTapCallback vision = nullptr);
    static void detach(Package::Program& program) noexcept;
};

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
