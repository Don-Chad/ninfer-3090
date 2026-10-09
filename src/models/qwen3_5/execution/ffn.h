#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

// `verify` selects the Verify-phase gate_up policy (DenseParameters::verify_gate_up_policy).
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool verify = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp = false, bool verify = false);

} // namespace ninfer::models::qwen3_5::execution
