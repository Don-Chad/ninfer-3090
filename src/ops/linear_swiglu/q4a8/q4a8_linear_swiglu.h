#pragma once
//
// Integer-activation route for the Qwen3.8/3.6-27B mlp/gate_up LinearSwiGLU, on sm_86.
//
// Reads the artifact's native Q4G64_F16S row-split weight with no repack: for Q4 that layout's
// code plane is exactly row-major [n][k/2] with two codes per byte, and its scale plane exactly
// row-major [n][k/64] FP16, so Weight::qdata and Weight::scales can be indexed directly.
//
// Activations are quantised to s8 with one scale per (token, group of 64) so that an outlier
// channel can only spoil its own group, then fed to mma.m16n8k32.s32.s8.s8.s32 -- about 4.7x the
// rate of the bf16 f32-accumulate MMA the A16 route issues on this hardware.
//
// This adds activation quantisation error the A16 route does not have: about 0.9% relative L2,
// against upstream's own kA8QuantizationAllowance of 4% for A8 activation compute (see
// tests/ops/linear/linear_test_common.cpp). It is opt-in for that reason.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// True when this route can serve the call: the registered 27B gate_up profile, a row-split Q4G64
// weight, and a token count the 128-wide tiling covers exactly. Partial prefill chunks and decode
// fall through to the A16 route.
[[nodiscard]] bool q4a8_swiglu_supported(const Weight& gate_up, std::int32_t tokens);

// Transient bytes this route needs for the quantised activation planes over [min,max] tokens.
[[nodiscard]] std::size_t q4a8_swiglu_workspace_capacity_bytes(std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

// x is contiguous BF16 [k, tokens]; out is contiguous BF16 [n/2, tokens].
void q4a8_swiglu_launch(const Tensor& x, const Weight& gate_up, Tensor& out,
                        WorkspaceArena& workspace, cudaStream_t stream);

// Companion route for mlp/down, a Q5G64 row-split LinearAdd. Q5 adds an 8-byte-per-group high
// plane carrying each code's fifth bit; a code decodes as ((low4 | hbit << 4) ^ 0x10) - 0x10 over
// [-16,15], which int8 still holds exactly.
[[nodiscard]] bool q5a8_add_supported(const Weight& down, std::int32_t tokens);
[[nodiscard]] std::size_t q5a8_add_workspace_capacity_bytes(std::int32_t min_tokens,
                                                            std::int32_t max_tokens);
void q5a8_add_launch(const Tensor& x, const Weight& down, Tensor& residual,
                     WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
