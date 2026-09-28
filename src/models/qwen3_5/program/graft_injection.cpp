#include "models/qwen3_5/program/graft_injection.h"
#include "models/qwen3_5/program/program_impl.h"

#include "ninfer/ops/kv_cache_append.h"
#include "core/arena.h"
#include "core/device.h"
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

void synchronize_all_ranks(const DeviceContext& device) {
    for (std::size_t rank = 0; rank < device.size(); ++rank) { device.synchronize_rank(rank); }
}

} // namespace

void inject_direct_graft(detail::ProgramImpl& program, const PromptGraft& graft) {
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

    // A pipeline split places each layer's KV planes, block-table replica and Linear Attention
    // state on that layer's own device, so every write goes on the owning rank's stream. One
    // device is the single-rank case of the same path.
    const DeviceContext& device = program.device;
    const RankStreams streams   = RankStreams::compute(device);

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

    // --- Step 2: allocate and populate the GDN state image ---
    std::optional<detail::StateImageHandle> state_handle =
        program.state_store->reserve_reset(streams);
    if (!state_handle) {
        throw std::runtime_error("inject_direct_graft: no free state image slot for graft '" +
                                 graft.name + "'");
    }
    const std::int32_t phys_slot = program.state_store->physical_slot(*state_handle);

    // The host image keeps every Linear Attention layer at its global offset whichever shard holds
    // it, and copy_from_host routes each shard's layers to its rank. The regions not written here
    // (continuation hidden, DFlash local state) stay zero, as reserve_reset left them.
    {
        const StateImageHostLayout& layout       = program.state_images->host_layout();
        const LinearAttentionStatePoolSpec& spec = layout.spec.linear;
        const std::int32_t ninfer_conv_width     = spec.conv_width;
        const DType rec_dtype                    = spec.recurrent_dtype;
        const std::size_t rec_elem               = (rec_dtype == DType::FP32) ? 4 : 2;
        if (spec.conv_dtype != DType::BF16 || spec.layers != n_linear ||
            layout.linear_conv_layer_bytes !=
                static_cast<std::size_t>(conv_ch) * ninfer_conv_width * 2 ||
            layout.linear_recurrent_layer_bytes !=
                static_cast<std::size_t>(Dk) * Dv * Hv * rec_elem) {
            throw std::runtime_error("inject_direct_graft: graft '" + graft.name +
                                     "' does not match the StateImage layout");
        }

        PinnedHostBuffer image(layout.image_bytes);
        auto* host = static_cast<std::uint8_t*>(image.data());
        std::memset(host, 0, layout.image_bytes);

        const std::size_t graft_conv_layer_bytes =
            static_cast<std::size_t>(conv_ch) * graft_conv_k * 2;
        const std::size_t graft_rec_layer_bytes =
            static_cast<std::size_t>(Hv) * Dk * Dv * 4; // always FP32 in graft
        for (std::uint32_t layer = 0; layer < n_linear; ++layer) {
            transpose_conv_layer(tensors.conv_data() + layer * graft_conv_layer_bytes,
                                 host + layout.linear_conv.offset +
                                     layer * layout.linear_conv_layer_bytes,
                                 conv_ch, graft_conv_k, ninfer_conv_width);
            transpose_rec_layer(tensors.rec_data() + layer * graft_rec_layer_bytes,
                                host + layout.linear_recurrent.offset +
                                    layer * layout.linear_recurrent_layer_bytes,
                                Hv, Dk, Dv, rec_dtype);
        }
        program.state_images->copy_from_host(
            HostStateImageConstView{.data   = reinterpret_cast<const std::byte*>(host),
                                    .layout = &layout},
            phys_slot, streams);
        // The pinned image is released at scope exit, so every rank's copy must finish first.
        synchronize_all_ranks(device);
    }
    program.state_store->freeze(*state_handle);
    program.state_store->retain_checkpoint_reference(*state_handle);

    // --- Step 3: create and populate text KV address space ---
    // The graft's K/V are stored as [n_attn, n_slots, n_kv_heads, head_dim] in safetensors
    // row-major. In ninfer's column-major convention, each per-layer slice is already
    // [head_dim, n_kv_heads, n_slots] -- exactly what kv_cache_append expects.
    const std::uint32_t kv_entitlement =
        (n_slots + 63U) / 64U; // pages needed (page size = 64 tokens)

    std::optional<detail::KVAddressSpaceHandle> text_kv =
        program.text_kv_addresses->create_inactive();
    if (!text_kv) {
        throw std::runtime_error("inject_direct_graft: no free KV address space for graft '" +
                                 graft.name + "'");
    }
    // Page ids name the same slice on every rank; activation publishes the row to every replica.
    program.text_kv_addresses->activate(*text_kv, kv_entitlement, 0, streams);
    program.text_kv_addresses->ensure_mapped_to_tokens(*text_kv, n_slots, streams);

    const std::size_t positions_bytes = static_cast<std::size_t>(n_slots) * sizeof(std::int32_t);
    PinnedHostBuffer positions_host(positions_bytes);
    auto* pos_data = static_cast<std::int32_t*>(positions_host.data());
    for (std::uint32_t i = 0; i < n_slots; ++i) { pos_data[i] = static_cast<std::int32_t>(i); }

    const std::size_t kv_layer_bytes =
        static_cast<std::size_t>(n_slots) * n_kv_heads * head_dim * 2; // BF16

    // Positions and K/V staging live on each rank that owns attention layers, allocated when the
    // first of its layers is reached. A rank's next layer reuses its staging, so that rank's
    // stream is synchronized before the overwrite.
    struct RankStaging {
        DeviceBuffer positions;
        DeviceBuffer k;
        DeviceBuffer v;
    };
    std::vector<RankStaging> staging(device.size());

    const auto& exec_row     = program.text_kv_addresses->execution_row(*text_kv);
    PagedKVCacheView kv_view = program.decoder->text_kv.execution_view(exec_row);

    for (std::uint32_t layer = 0; layer < n_attn; ++layer) {
        const std::size_t rank         = program.decoder->text_kv.layer_rank(layer);
        const cudaStream_t rank_stream = streams[rank];
        RankBinding bind(device, rank);
        RankStaging& buffers = staging[rank];
        if (buffers.positions.p == nullptr) {
            buffers.positions = DeviceBuffer(positions_bytes);
            buffers.k         = DeviceBuffer(kv_layer_bytes);
            buffers.v         = DeviceBuffer(kv_layer_bytes);
            cudaMemcpyAsync(buffers.positions.p, pos_data, positions_bytes,
                            cudaMemcpyHostToDevice, rank_stream);
        } else {
            device.synchronize_rank(rank);
        }

        const std::uint8_t* k_src = tensors.k_data() + layer * kv_layer_bytes;
        const std::uint8_t* v_src = tensors.v_data() + layer * kv_layer_bytes;
        cudaMemcpyAsync(buffers.k.p, k_src, kv_layer_bytes, cudaMemcpyHostToDevice, rank_stream);
        cudaMemcpyAsync(buffers.v.p, v_src, kv_layer_bytes, cudaMemcpyHostToDevice, rank_stream);

        Tensor positions_tensor(buffers.positions.p, DType::I32,
                                {static_cast<std::int32_t>(n_slots)});
        Tensor k_tensor(buffers.k.p, DType::BF16,
                        {head_dim, n_kv_heads, static_cast<std::int32_t>(n_slots)});
        Tensor v_tensor(buffers.v.p, DType::BF16,
                        {head_dim, n_kv_heads, static_cast<std::int32_t>(n_slots)});
        PagedKVLayerView layer_view = kv_view.layer_view(layer);
        ops::kv_cache_append(k_tensor, v_tensor, positions_tensor, layer_view, rank_stream);
    }

    synchronize_all_ranks(device);
    for (std::size_t rank = 0; rank < staging.size(); ++rank) {
        RankBinding bind(device, rank);
        staging[rank] = RankStaging{};
    }

    program.text_kv_addresses->commit_frontier(*text_kv, n_slots);
    program.text_kv_addresses->deactivate(*text_kv);

    // --- Step 4: populate the shared prefix entry ---
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
