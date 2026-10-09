#pragma once
#include "core/device.h"

namespace ninfer::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* q8_shared_storage() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        return dynamic_shared;
    } else {
        __shared__ __align__(16) unsigned char storage[Bytes];
        return storage;
    }
}

template <int Bytes, auto Kernel>
int q8_prepare_shared() {
    if constexpr (Bytes > 48 * 1024) {
        // Function attributes are per device context; a process-wide static would leave the
        // kernel unconfigured on a second GPU (see configure_cuda_device_once).
        configure_cuda_device_once([] {
            return cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Bytes);
        });
        return Bytes;
    } else
        return 0;
}
} // namespace ninfer::ops::detail
