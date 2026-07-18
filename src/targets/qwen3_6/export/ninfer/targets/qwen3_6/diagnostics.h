#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6 {

enum class TextPhase {
    Prefill,
    Verify,
};

enum class TextTapId {
    AfterEmbed,
    AfterMixer,
    AfterMlp,
    AfterFinalNorm,
    AfterLogits,
};

enum class VisionTapId {
    PatchEmbed,
    Block,
    Merger,
};

using TextTapCallback   = void (*)(void*, TextTapId, int, TextPhase, const Tensor&, cudaStream_t);
using VisionTapCallback = void (*)(void*, std::uint32_t, VisionTapId, int, const Tensor&,
                                   cudaStream_t);

} // namespace ninfer::targets::qwen3_6
