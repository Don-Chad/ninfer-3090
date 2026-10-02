// Integer-activation route for the Q5G64 row-split LinearAdds, companion to the gate_up route.
//
// Two registered profiles, differing only in K: mlp/down [5120,17408], and the attention o_proj and
// GDN out_proj [5120,6144]. Q5 adds an 8-byte-per-group high plane carrying each code's fifth bit;
// a code decodes as ((low4 | hbit << 4) ^ 0x10) - 0x10, two's complement over [-16,15], which int8
// represents exactly.
//
// The schedule, the staging and the token-tile reasoning all live in the shared header.

#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include "core/device.h"
#include "core/weight_view.h"
#include "ops/common/rowsplit_a8_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

namespace a8 = rowsplit_a8;

constexpr std::int32_t kRows      = 5120;
constexpr std::int32_t kDownCols  = 17408;
constexpr std::int32_t kMixerCols = 6144;

using Rows = a8::ContiguousRows<1>;

template <std::int32_t kCols, int NT>
void launch(const Tensor& x, const Weight& down, Tensor& residual, std::int32_t tokens,
            std::int8_t* codes, __half* scales, cudaStream_t stream) {
    constexpr int BN = a8::kWarpsN * NT * 8;
    a8::quantize_activations<kCols, BN><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, codes, scales);
    CUDA_CHECK(cudaGetLastError());

    const std::size_t smem = a8::shared_bytes<a8::Q5Codec, 1, NT, Rows>(kCols);
    const dim3 grid(kRows / Rows::kRowsPerBlock, tokens / BN);
    auto* kernel = a8::a8_mma_kernel<a8::Q5Codec, kCols, 1, NT, Rows, a8::ResidualAddEpilogue>;
    if (smem > 48 * 1024) {
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem));
        });
    }
    kernel<<<grid, a8::kThreads, smem, stream>>>(
        static_cast<const std::uint8_t*>(down.qdata), static_cast<const std::uint8_t*>(down.qhigh),
        static_cast<const __half*>(down.scales), codes, scales, tokens, Rows{0},
        a8::ResidualAddEpilogue{reinterpret_cast<__nv_bfloat16*>(residual.data), kRows},
        row_split_panel_shift(down.layout));
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t kCols>
void launch_for_tile(const Tensor& x, const Weight& down, Tensor& residual, std::int32_t tokens,
                     std::int8_t* codes, __half* scales, cudaStream_t stream) {
    switch (a8::token_tile(tokens, 512)) {
    case 512: launch<kCols, 16>(x, down, residual, tokens, codes, scales, stream); return;
    case 256: launch<kCols, 8>(x, down, residual, tokens, codes, scales, stream); return;
    default: launch<kCols, 4>(x, down, residual, tokens, codes, scales, stream); return;
    }
}

} // namespace

bool q5a8_tokens_supported(std::int32_t tokens) { return a8::tokens_supported(tokens); }

bool q5a8_add_supported(const Weight& down, std::int32_t tokens) {
    return down.qtype == QType::Q5_G64_FP16 && is_row_split(down.layout) &&
           down.n == kRows && (down.k == kDownCols || down.k == kMixerCols) &&
           down.group == a8::kGroup && down.qdata != nullptr && down.qhigh != nullptr &&
           down.scales != nullptr && a8::tokens_supported(tokens);
}

std::size_t q5a8_add_workspace_capacity_bytes(std::int32_t input_rows, std::int32_t min_tokens,
                                              std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q5a8 add workspace: invalid token interval");
    }
    if (input_rows != kDownCols && input_rows != kMixerCols) {
        throw std::invalid_argument("q5a8 add workspace: unregistered input width");
    }
    return a8::activation_workspace_bytes(input_rows, max_tokens);
}

void q5a8_add_launch(const Tensor& x, const Weight& down, Tensor& residual,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!q5a8_add_supported(down, tokens)) {
        throw std::invalid_argument("q5a8 add: unsupported profile");
    }

    auto scope = workspace.scope();
    const DeviceSpan codes =
        workspace.alloc_bytes(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(down.k));
    const DeviceSpan scales = workspace.alloc_bytes(static_cast<std::size_t>(tokens) *
                                                    (static_cast<std::size_t>(down.k) / a8::kGroup) *
                                                    sizeof(__half));
    auto* code_data  = reinterpret_cast<std::int8_t*>(codes.data);
    auto* scale_data = reinterpret_cast<__half*>(scales.data);

    if (down.k == kDownCols) {
        launch_for_tile<kDownCols>(x, down, residual, tokens, code_data, scale_data, stream);
    } else {
        launch_for_tile<kMixerCols>(x, down, residual, tokens, code_data, scale_data, stream);
    }
}

} // namespace ninfer::ops::detail
