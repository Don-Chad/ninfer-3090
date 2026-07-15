#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: scatter
 *
 * Math / indexing:
 *   destination[:, indices[i]] = source[:, i] for 0 <= i < V.
 *
 * Logical shapes:
 *   BF16 source [D,V], I32 indices [V], BF16 destination [D,T], all contiguous.
 *
 * Numeric:
 *   Exact BF16 element copies. Callers provide indices in [0,T) with no duplicates when
 *   deterministic overwrite order is required.
 *
 * Effects:
 *   Overwrites only the selected destination columns; all other columns retain their old values.
 *   Inputs and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect beyond the selected destination writes.
 *
 * Supported routes:
 *   D divisible by eight with 16-byte-aligned source/destination uses BF16x8 copies; other aligned
 *   even D uses BF16x2; remaining contiguous dimensions use scalar copies.
 */
void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);

/**
 * Op: extract_bf16_columns
 *
 * Math / indexing:
 *   For source [D,T] and destination [D',T]:
 *   destination[d,t] = source[source_column + d,t], 0<=d<D', 0<=t<T.
 *
 * Logical shapes:
 *   Contiguous BF16 source [D,T] and destination [D',T]. Despite the historical parameter name,
 *   source_column is an offset in the fastest (ne[0]) dimension; 0<=source_column and
 *   source_column+D'<=D.
 *
 * Numeric:
 *   Exact BF16 element copies.
 *
 * Effects:
 *   Writes the full destination; source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void extract_bf16_columns(const Tensor& source, std::int32_t source_column, Tensor& destination,
                          cudaStream_t stream);

} // namespace ninfer::ops
