#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   x'[i]                  = BF16(float(x[i]) * cos(phi) - float(x[i+R/2]) * sin(phi))
 *   x'[i+rotary_dim/2]     = BF16(float(x[i+R/2]) * cos(phi) + float(x[i]) * sin(phi)).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], head_dim=256, even 0<rotary_dim<=256,
 *   phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with contiguous head features and heads; token stride may be padded. Text Q
 * has 24 heads and Text K has 4, while both Vision tensors have 16. q and k must not overlap one
 * another or positions. The Op mutates only the supplied Q/K tensor storage and uses no workspace
 * or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. Text Q/K role is inferred from
// 24 versus 4 heads; Vision uses 16 heads.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
