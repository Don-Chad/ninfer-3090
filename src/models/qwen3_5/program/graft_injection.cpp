#include "models/qwen3_5/program/graft_injection.h"
#include "models/qwen3_5/program/program_impl.h"

#include "ninfer/ops/kv_cache_append.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

// Transpose graft conv data from safetensors row-major [conv_channels, conv_k] to ninfer's
// column-major [conv_channels, conv_width] layout, dropping the oldest (conv_k - conv_width)
// columns from the left.
void transpose_conv_layer(const std::uint8_t* src, std::uint8_t* dst,
                          std::uint64_t channels, std::uint64_t conv_k,
                          std::uint64_t conv_width) {
    // src row-major: element(ch, w) at (ch * conv_k + w) * 2
    // dst ninfer dim-0 innermost: element at (ch + w * channels) * 2
    const std::uint64_t drop = conv_k - conv_width;
    for (std::uint64_t w = 0; w < conv_width; ++w) {
        for (std::uint64_t ch = 0; ch < channels; ++ch) {
            const std::size_t src_off = (ch * conv_k + w + drop) * 2;
            const std::size_t dst_off = (ch + w * channels) * 2;
            std::memcpy(dst + dst_off, src + src_off, 2);
        }
    }
}

// Transpose graft recurrent data from safetensors row-major [Hv, Dk, Dv] (FP32) to ninfer's
// column-major [Dk, Dv, Hv] layout. When the target dtype is FP16, also truncate each float.
void transpose_rec_layer(const std::uint8_t* src_fp32, std::uint8_t* dst,
                         std::uint64_t Hv, std::uint64_t Dk, std::uint64_t Dv,
                         DType target_dtype) {
    // src row-major: element(h, k, v) at (h * Dk * Dv + k * Dv + v) * 4
    // dst ninfer dim-0 innermost: element(k, v, h) at (k + v * Dk + h * Dk * Dv) * elem_size
    const std::size_t elem = (target_dtype == DType::FP32) ? 4 : 2;
    for (std::uint64_t h = 0; h < Hv; ++h) {
        for (std::uint64_t k = 0; k < Dk; ++k) {
            for (std::uint64_t v = 0; v < Dv; ++v) {
                const std::size_t src_off = (h * Dk * Dv + k * Dv + v) * 4;
                const std::size_t dst_off = (k + v * Dk + h * Dk * Dv) * elem;
                if (target_dtype == DType::FP32) {
                    std::memcpy(dst + dst_off, src_fp32 + src_off, 4);
                } else {
                    // FP32 -> FP16 truncation: drop the lower 16 bits of the mantissa.
                    // This is a rough truncation (not IEEE round-to-nearest-even) but matches
                    // what ninfer does when gdn_state_fp16 is enabled.
                    float val;
                    std::memcpy(&val, src_fp32 + src_off, 4);
                    std::uint32_t bits;
                    std::memcpy(&bits, &val, 4);
                    // IEEE FP32 -> FP16: shift sign+exponent+mantissa, clamp to FP16 range.
                    const std::uint32_t sign     = (bits >> 16U) & 0x8000U;
                    const std::int32_t exponent  = static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127;
                    const std::uint32_t mantissa = bits & 0x7FFFFFU;
                    std::uint16_t fp16;
                    if (exponent > 15) {
                        fp16 = static_cast<std::uint16_t>(sign | 0x7C00U);
                    } else if (exponent < -14) {
                        fp16 = static_cast<std::uint16_t>(sign);
                    } else {
                        fp16 = static_cast<std::uint16_t>(
                            sign | (static_cast<std::uint32_t>(exponent + 15) << 10U) |
                            (mantissa >> 13U));
                    }
                    std::memcpy(dst + dst_off, &fp16, 2);
                }
            }
        }
    }
}

} // namespace

void inject_direct_graft(detail::ProgramImpl& program, const PromptGraft& graft,
                         cudaStream_t stream) {
    if (graft.kind == GraftKind::PrefillKV) {
        throw std::invalid_argument("inject_direct_graft: prefill_kv grafts use token replay, "
                                    "not direct injection");
    }
    if (!graft.tensors) {
        throw std::invalid_argument("inject_direct_graft: graft '" + graft.name +
                                    "' has no tensor data");
    }
    const GraftTensors& tensors = *graft.tensors;
    const auto n_slots      = static_cast<std::uint32_t>(tensors.n_slots);
    const auto n_attn       = static_cast<std::uint32_t>(tensors.n_attn_layers);
    const auto n_linear     = static_cast<std::uint32_t>(tensors.n_linear_layers);
    const auto n_kv_heads   = static_cast<std::int32_t>(tensors.n_kv_heads);
    const auto head_dim     = static_cast<std::int32_t>(tensors.head_dim);
    const auto conv_ch      = static_cast<std::int32_t>(tensors.conv_channels);
    const auto graft_conv_k = static_cast<std::int32_t>(tensors.conv_width);
    const auto Hv           = static_cast<std::int32_t>(tensors.value_heads);
    const auto Dk           = static_cast<std::int32_t>(tensors.key_head_dim);
    const auto Dv           = static_cast<std::int32_t>(tensors.value_head_dim);

    // --- Step 1: find a free shared prefix slot ---
    std::uint32_t slot_index = program.shared_prefix_capacity;
    for (std::uint32_t i = 0; i < program.shared_prefix_capacity; ++i) {
        if (program.shared_prefix_slots[i].role == detail::SharedPrefixSlotRole::Free) {
            slot_index = i;
            break;
        }
    }
    if (slot_index == program.shared_prefix_capacity) {
        throw std::runtime_error("inject_direct_graft: no free shared prefix slot for graft '" +
                                 graft.name + "'");
    }

    // --- Step 2: allocate and populate GDN state image ---
    std::optional<detail::StateImageHandle> state_handle = program.state_store->reserve_reset(stream);
    if (!state_handle) {
        throw std::runtime_error("inject_direct_graft: no free state image slot for graft '" +
                                 graft.name + "'");
    }
    const std::int32_t phys_slot = program.state_store->physical_slot(*state_handle);
    const auto& linear_pool     = program.state_images->linear();
    const auto& spec            = linear_pool.spec();
    const std::int32_t ninfer_conv_width = spec.conv_width;
    const DType rec_dtype                = spec.recurrent_dtype;

    // Upload conv state: transpose + column-drop per linear layer.
    // The staging buffer is pinned and reused, so each layer must sync before the next
    // transpose overwrites it.
    {
        const std::size_t conv_layer_bytes =
            static_cast<std::size_t>(conv_ch) * ninfer_conv_width * 2;
        PinnedHostBuffer conv_staging(conv_layer_bytes);
        const std::size_t graft_conv_layer_bytes =
            static_cast<std::size_t>(conv_ch) * graft_conv_k * 2;

        for (std::uint32_t layer = 0; layer < n_linear; ++layer) {
            if (layer > 0) { cudaStreamSynchronize(stream); }
            const std::uint8_t* src = tensors.conv_data() + layer * graft_conv_layer_bytes;
            auto* staging = static_cast<std::uint8_t*>(conv_staging.data());
            transpose_conv_layer(src, staging, conv_ch, graft_conv_k, ninfer_conv_width);

            Tensor device_conv = linear_pool.conv_slot(layer, phys_slot);
            cudaMemcpyAsync(device_conv.data, staging, conv_layer_bytes,
                            cudaMemcpyHostToDevice, stream);
        }
    }

    // Upload recurrent state: transpose per linear layer
    {
        const std::size_t rec_elem = (rec_dtype == DType::FP32) ? 4 : 2;
        const std::size_t rec_layer_bytes =
            static_cast<std::size_t>(Dk) * Dv * Hv * rec_elem;
        PinnedHostBuffer rec_staging(rec_layer_bytes);
        const std::size_t graft_rec_layer_bytes =
            static_cast<std::size_t>(Hv) * Dk * Dv * 4; // always FP32 in graft

        for (std::uint32_t layer = 0; layer < n_linear; ++layer) {
            if (layer > 0) { cudaStreamSynchronize(stream); }
            const std::uint8_t* src = tensors.rec_data() + layer * graft_rec_layer_bytes;
            auto* staging = static_cast<std::uint8_t*>(rec_staging.data());
            transpose_rec_layer(src, staging, Hv, Dk, Dv, rec_dtype);

            Tensor device_rec = linear_pool.recurrent_slot(layer, phys_slot);
            cudaMemcpyAsync(device_rec.data, staging, rec_layer_bytes,
                            cudaMemcpyHostToDevice, stream);
        }
    }

    // Ensure all state writes complete before freezing
    cudaStreamSynchronize(stream);
    program.state_store->freeze(*state_handle);
    program.state_store->retain_checkpoint_reference(*state_handle);

    // --- Step 3: create and populate text KV address space ---
    // The graft's K/V are stored as [n_attn, n_slots, n_kv_heads, head_dim] in safetensors
    // row-major. In ninfer's column-major convention, each per-layer slice is already
    // [head_dim, n_kv_heads, n_slots] — exactly what kv_cache_append expects.
    const std::uint32_t kv_entitlement =
        (n_slots + 63U) / 64U; // pages needed (page size = 64 tokens)

    std::optional<detail::KVAddressSpaceHandle> text_kv = program.text_kv_addresses->create_inactive();
    if (!text_kv) {
        throw std::runtime_error("inject_direct_graft: no free KV address space for graft '" +
                                 graft.name + "'");
    }
    program.text_kv_addresses->activate(*text_kv, kv_entitlement, 0);
    program.text_kv_addresses->ensure_mapped_to_tokens(*text_kv, n_slots, stream);

    // Build device positions tensor [0, 1, 2, ..., n_slots-1]
    PinnedHostBuffer positions_host(static_cast<std::size_t>(n_slots) * sizeof(std::int32_t));
    auto* pos_data = static_cast<std::int32_t*>(positions_host.data());
    for (std::uint32_t i = 0; i < n_slots; ++i) { pos_data[i] = static_cast<std::int32_t>(i); }

    void* positions_device = nullptr;
    cudaMalloc(&positions_device, static_cast<std::size_t>(n_slots) * sizeof(std::int32_t));
    cudaMemcpyAsync(positions_device, pos_data,
                    static_cast<std::size_t>(n_slots) * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice, stream);
    Tensor positions_tensor(positions_device, DType::I32, {static_cast<std::int32_t>(n_slots)});

    // Upload K/V per attention layer. Two device staging buffers are reused across layers.
    const std::size_t kv_layer_bytes =
        static_cast<std::size_t>(n_slots) * n_kv_heads * head_dim * 2; // BF16

    void* k_device_buf = nullptr;
    void* v_device_buf = nullptr;
    cudaMalloc(&k_device_buf, kv_layer_bytes);
    cudaMalloc(&v_device_buf, kv_layer_bytes);

    const auto& exec_row = program.text_kv_addresses->execution_row(*text_kv);
    PagedKVCacheView kv_view = program.decoder->text_kv.execution_view(exec_row);

    for (std::uint32_t layer = 0; layer < n_attn; ++layer) {
        // Wait for the previous layer's kv_cache_append to finish before overwriting buffers
        if (layer > 0) { cudaStreamSynchronize(stream); }

        const std::uint8_t* k_src = tensors.k_data() + layer * kv_layer_bytes;
        const std::uint8_t* v_src = tensors.v_data() + layer * kv_layer_bytes;
        cudaMemcpyAsync(k_device_buf, k_src, kv_layer_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(v_device_buf, v_src, kv_layer_bytes, cudaMemcpyHostToDevice, stream);

        Tensor k_tensor(k_device_buf, DType::BF16,
                        {head_dim, n_kv_heads, static_cast<std::int32_t>(n_slots)});
        Tensor v_tensor(v_device_buf, DType::BF16,
                        {head_dim, n_kv_heads, static_cast<std::int32_t>(n_slots)});
        PagedKVLayerView layer_view = kv_view.layer_view(layer);
        ops::kv_cache_append(k_tensor, v_tensor, positions_tensor, layer_view, stream);
    }

    cudaStreamSynchronize(stream);
    cudaFree(k_device_buf);
    cudaFree(v_device_buf);
    cudaFree(positions_device);

    program.text_kv_addresses->commit_frontier(*text_kv, n_slots);
    program.text_kv_addresses->deactivate(*text_kv);

    // --- Step 4: populate the shared prefix entry ---
    cudaStreamSynchronize(stream);

    auto& shared       = program.shared_prefix_states[slot_index];
    auto& slot         = program.shared_prefix_slots[slot_index];

    shared.kv = detail::SequenceKVBundle{.text = *text_kv, .backend = std::nullopt};
    shared.state             = *state_handle;
    shared.identity          = nullptr; // grafted requests bypass identity matching
    shared.frontier          = n_slots;
    shared.backend_frontier  = 0;
    shared.rope_delta        = 0;
    shared.tail_hidden_valid = false;
    shared.rebuild_work      = runtime::PrefillWork{.tokens = n_slots};
    shared.active_references = 0;

    slot.role = detail::SharedPrefixSlotRole::Pinned;

    program.graft_prefix_slots[graft.name] = {slot_index, slot.generation};
}

} // namespace ninfer::models::qwen3_5
