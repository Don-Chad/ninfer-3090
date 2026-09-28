#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// A capture fixes a tile-aligned KV chunk and its upper-bound grid capacity.
// Device lengths only reduce the active prefix; they never change the topology.
struct Bf16KvPartition {
    int capacity       = 1;
    int keys_per_split = 64;

    __host__ __device__ int active_splits(int visible) const {
        const int count = (visible + keys_per_split - 1) / keys_per_split;
        return count < capacity ? count : capacity;
    }
};

} // namespace ninfer::ops::detail
