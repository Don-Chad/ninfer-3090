#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient FP32 arena capacity required by gated_delta_rule for `tokens`. It is zero
 * when no complete internal 64-token chunk is present.
 */
[[nodiscard]] std::size_t gated_delta_rule_workspace_bytes(std::int32_t tokens);

/**
 * Applies the Gated DeltaNet recurrence independently for each value head h. Its Q/K head is
 * qh=floor(h/3). Starting from S_h, for t in increasing order:
 *
 *   alpha       = exp(g[h,t])
 *   delta       = beta[h,t] * (v[:,h,t] - alpha * S_h * k[:,qh,t])
 *   S_h         = alpha * S_h + outer(delta, k[:,qh,t])
 *   out[:,h,t]  = BF16(scale * S_h * q[:,qh,t]).
 *
 * Shapes/dtypes are contiguous q/k BF16 [128,16,T], v/out BF16 [128,48,T], g/beta FP32 [48,T],
 * and state FP32 [128,128,48]. `scale` is 1/sqrt(128). Inputs and out do not overlap state or one
 * another. `ws` supplies transient chunked-prefill storage reported by
 * gated_delta_rule_workspace_bytes; scratch is scoped to the call.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage. No other
 * arguments may overlap either state.
 */
void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream);

/**
 * Snapshot form of the same recurrence. `ssm_states` is contiguous FP32
 * [128,128,48,Slots], `initial_slot` is a contiguous I32 scalar in [0,Slots), and Slots>=T. The
 * selected slot supplies the initial state; after token t, the new state is written to slot t.
 * Slots at and above T are unchanged. This form uses no arena allocation, and `ssm_states` is the
 * only persistent state mutated.
 */
void gated_delta_rule_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, WorkspaceArena& ws,
                               Tensor& ssm_states, const Tensor& initial_slot, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::ops
