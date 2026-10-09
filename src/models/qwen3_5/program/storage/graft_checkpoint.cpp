// Installation of externally supplied context (direct_kv and softprompt_kv prompt grafts) as a
// pinned, immutable SharedPrefix checkpoint.
//
// A graft carries the target's attention K/V for its n slots and the Gated DeltaNet convolution and
// recurrent state after them, in the phantom-kv safetensors layout. Installation writes the K/V
// through the ordinary append Op into fresh Main KV pages (so every KV storage profile receives the
// same consumable representation prefill would have written), uploads the state into a StateImage,
// and publishes one checkpoint whose identity is the graft's placeholder ids at positions [0, n).
// A request that selects the graft carries exactly those ids, so ordinary prefix matching binds it
// to this checkpoint as an exact source.
//
// Nothing in the container describes a speculative draft's context. MTP KV and DFlash context
// features over the graft are zero-filled and the continuation hidden is zero (tail_hidden_valid is
// false), so drafts over the first suffix tokens are weaker; the target verifies every proposal, so
// the output distribution is that of the target over the installed state.
//
// The checkpoint is pinned for the Program's life (a permanent lease): it cannot be released,
// demoted or chosen by a reclaim plan, and no binding may consume it.

#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "ninfer/ops/kv_cache_append.h"
#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Graft conv data is row-major [conv_channels, conv_k] per layer (the last conv_k inputs, oldest
// first). The StateImage keeps the trailing conv_width of them as BF16 [conv_channels, conv_width]
// with the channel innermost, oldest first, so the oldest conv_k - conv_width graft columns drop.
void transpose_conv_layer(const std::uint8_t* src, std::byte* dst, std::uint64_t channels,
                          std::uint64_t conv_k, std::uint64_t conv_width) {
    const std::uint64_t drop = conv_k - conv_width;
    for (std::uint64_t w = 0; w < conv_width; ++w) {
        for (std::uint64_t ch = 0; ch < channels; ++ch) {
            std::memcpy(dst + (ch + w * channels) * 2, src + (ch * conv_k + w + drop) * 2, 2);
        }
    }
}

// IEEE binary32 -> binary16, round to nearest even, with subnormals, overflow to infinity and NaN
// preserved: the same narrowing the runtime applies to recurrent state stored as FP16.
std::uint16_t float_to_half_rne(float value) {
    std::uint32_t x;
    std::memcpy(&x, &value, 4);
    const std::uint32_t sign = (x >> 16U) & 0x8000U;
    const std::uint32_t ax   = x & 0x7fffffffU;
    if (ax >= 0x7f800000U) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | ((ax & 0x007fffffU) != 0 ? 0x0200U : 0U));
    }
    const auto round_shift = [](std::uint32_t v, int shift) {
        const std::uint32_t base = v >> shift;
        const std::uint32_t rem  = v & ((1U << shift) - 1U);
        const std::uint32_t half = 1U << (shift - 1);
        return base + ((rem > half || (rem == half && (base & 1U) != 0U)) ? 1U : 0U);
    };
    int exponent             = static_cast<int>(ax >> 23U) - 127 + 15;
    const std::uint32_t mant = ax & 0x007fffffU;
    if (exponent <= 0) {
        if (exponent < -10) { return static_cast<std::uint16_t>(sign); }
        return static_cast<std::uint16_t>(sign | round_shift(mant | 0x00800000U, 14 - exponent));
    }
    if (exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    std::uint32_t half_mant = round_shift(mant, 13);
    if (half_mant == 0x0400U) {
        half_mant = 0;
        if (++exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10U) |
                                      half_mant);
}

// Graft recurrent data is row-major FP32 [Hv, Dk, Dv] per layer. The StateImage stores it as
// [Dk, Dv, Hv] with Dk innermost, in FP32 or narrowed to FP16 with round-to-nearest-even.
void transpose_rec_layer(const std::uint8_t* src_fp32, std::byte* dst, std::uint64_t Hv,
                         std::uint64_t Dk, std::uint64_t Dv, DType target_dtype) {
    const std::size_t elem = target_dtype == DType::FP32 ? 4 : 2;
    for (std::uint64_t h = 0; h < Hv; ++h) {
        for (std::uint64_t k = 0; k < Dk; ++k) {
            for (std::uint64_t v = 0; v < Dv; ++v) {
                const std::size_t src_off = (h * Dk * Dv + k * Dv + v) * 4;
                const std::size_t dst_off = (k + v * Dk + h * Dk * Dv) * elem;
                if (target_dtype == DType::FP32) {
                    std::memcpy(dst + dst_off, src_fp32 + src_off, 4);
                } else {
                    float value;
                    std::memcpy(&value, src_fp32 + src_off, 4);
                    const std::uint16_t half = float_to_half_rne(value);
                    std::memcpy(dst + dst_off, &half, 2);
                }
            }
        }
    }
}

[[noreturn]] void reject(const PromptGraft& graft, const std::string& message) {
    throw std::invalid_argument("graft '" + graft.name + "': " + message);
}

} // namespace

std::uint32_t ProgramImpl::prefix_identity_tag() const noexcept {
    return static_cast<std::uint32_t>(speculative_backend) |
           (static_cast<std::uint32_t>(proposal_head) << 8U) |
           (static_cast<std::uint32_t>(kv_storage) << 16U);
}

CheckpointHandle ProgramImpl::install_external_checkpoint(const PromptGraft& graft) {
    if (graft.kind == GraftKind::PrefillKV) {
        reject(graft, "a prefill_kv graft is replayed through prefill, not installed");
    }
    if (!graft.tensors) { reject(graft, "the graft carries no tensor data"); }
    if (causal_scoring || !context_cache.enabled) {
        throw std::logic_error("external checkpoints need a generation Program with a context cache");
    }
    if (context_transaction_ || pending_transaction_ ||
        std::any_of(requests.begin(), requests.end(),
                    [](const RequestControl& request) { return request.lifecycle != Lifecycle::Empty; })) {
        throw std::logic_error("external checkpoint installation requires an idle Program");
    }
    const GraftTensors& tensors = *graft.tensors;
    const std::uint32_t slots   = graft.n_slots;
    const auto& config          = parameters.model.config().text;
    if (slots == 0 || tensors.n_slots != slots || graft.placeholder_ids.size() != slots) {
        reject(graft, "slot count is inconsistent");
    }
    if (slots >= capacity) { reject(graft, "leaves no room for a prompt within max_context"); }
    if (!config.attention || tensors.n_attn_layers != config.full_attention_layers ||
        tensors.n_linear_layers != config.linear_attention_layers ||
        tensors.n_kv_heads != config.attention->num_key_value_heads ||
        tensors.head_dim != config.attention->head_dim) {
        reject(graft, "attention geometry differs from the resident model");
    }
    const auto public_tokens = dimension(parameters.model.resources().public_token_count);
    for (const TokenId id : graft.placeholder_ids) {
        if (id < 0 || id >= public_tokens) { reject(graft, "placeholder id is not a public token"); }
    }

    const auto& layout = state_images->host_layout();
    const auto& linear = layout.spec.linear;
    const std::size_t rec_elem = linear.recurrent_dtype == DType::FP32 ? 4 : 2;
    if (linear.conv_dtype != DType::BF16 || linear.layers != tensors.n_linear_layers ||
        static_cast<std::uint64_t>(linear.conv_channels) != tensors.conv_channels ||
        static_cast<std::uint64_t>(linear.conv_width) > tensors.conv_width ||
        static_cast<std::uint64_t>(linear.value_heads) != tensors.value_heads ||
        static_cast<std::uint64_t>(linear.key_head_dim) != tensors.key_head_dim ||
        static_cast<std::uint64_t>(linear.value_head_dim) != tensors.value_head_dim ||
        layout.linear_conv_layer_bytes !=
            static_cast<std::size_t>(linear.conv_channels) * linear.conv_width * 2 ||
        layout.linear_recurrent_layer_bytes != tensors.key_head_dim * tensors.value_head_dim *
                                                   tensors.value_heads * rec_elem) {
        reject(graft, "Gated DeltaNet state does not match the StateImage layout");
    }

    // The identity requests carry for this graft: placeholder ids, text positions 0..n-1.
    PreparedPromptData prefix;
    prefix.token_ids = graft.placeholder_ids;
    prefix.token_types.assign(slots, 0);
    prefix.positions.resize(3ULL * slots);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::uint32_t i = 0; i < slots; ++i) {
            prefix.positions[axis * slots + i] = static_cast<std::int32_t>(i);
        }
    }
    auto identity = std::make_shared<PreparedCaptureBacking>();
    identity->digests.assign(prefix);
    identity->ledger = prefix.token_ids;
    identity->prefix_identity.assign(prefix);

    // Ownerless until publication, so a failed installation releases its addresses explicitly.
    auto history = std::make_shared<KVHistory>();

    // The draft's context frontier follows the capture convention: MTP KV ends one token before
    // the target's, full DFlash context features end with it, other backends keep none.
    const std::uint32_t backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp      ? slots - 1U
        : speculative_backend == SpeculativeBackend::DFlash ? slots
                                                            : 0U;
    const auto handle = reserve_checkpoint();
    if (!handle) { throw std::runtime_error("no checkpoint descriptor for graft '" + graft.name + "'"); }
    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> text;
    std::optional<KVAddressSpaceHandle> backend;
    bool state_frozen = false;
    const auto rollback = [&]() noexcept {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        if (backend) {
            if (backend_kv_addresses->active(*backend)) { backend_kv_addresses->deactivate(*backend); }
            (void)backend_kv_addresses->release(*backend);
        }
        if (text) {
            if (text_kv_addresses->active(*text)) { text_kv_addresses->deactivate(*text); }
            (void)text_kv_addresses->release(*text);
        }
        if (state) {
            if (state_frozen) {
                (void)state_store->release_checkpoint_owner(*state);
            } else {
                (void)state_store->release(*state);
            }
        }
        checkpoints[handle->index].reserved = false;
    };
    try {
        // --- StateImage: Gated DeltaNet state; continuation hidden and DFlash local stay zero ---
        state = state_store->reserve_reset(device.stream);
        if (!state) {
            throw std::runtime_error("no free StateImage slot for graft '" + graft.name + "'");
        }
        {
            std::vector<std::byte> image(layout.image_bytes, std::byte{0});
            const std::size_t graft_conv_layer =
                static_cast<std::size_t>(tensors.conv_channels) * tensors.conv_width * 2;
            const std::size_t graft_rec_layer = static_cast<std::size_t>(tensors.value_heads) *
                                                tensors.key_head_dim * tensors.value_head_dim * 4;
            for (std::uint32_t layer = 0; layer < linear.layers; ++layer) {
                transpose_conv_layer(tensors.conv_data() + layer * graft_conv_layer,
                                     image.data() + layout.linear_conv.offset +
                                         layer * layout.linear_conv_layer_bytes,
                                     tensors.conv_channels, tensors.conv_width,
                                     static_cast<std::uint64_t>(linear.conv_width));
                transpose_rec_layer(tensors.rec_data() + layer * graft_rec_layer,
                                    image.data() + layout.linear_recurrent.offset +
                                        layer * layout.linear_recurrent_layer_bytes,
                                    tensors.value_heads, tensors.key_head_dim,
                                    tensors.value_head_dim, linear.recurrent_dtype);
            }
            state_images->copy_from_host(
                HostStateImageConstView{.data = image.data(), .layout = &layout},
                state_store->physical_slot(*state), RankStreams(device.stream));
            device.synchronize(); // the host image is released at scope exit
        }
        state_store->freeze(*state);
        state_store->retain_checkpoint_reference(*state);
        state_frozen = true;

        // --- Main KV: the ordinary append Op over staged BF16 rows, chunked to the workspace ---
        const std::uint32_t pages = kv_pages_for_frontier(slots);
        text = text_kv_addresses->create_active(pages, 0, device.stream);
        if (!text) {
            throw std::runtime_error("no Main KV address space or pages for graft '" + graft.name +
                                     "'");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*text, slots, device.stream);
        const PagedKVCacheView view =
            decoder->text_kv.execution_view(text_kv_addresses->execution_row(*text));
        const auto heads       = static_cast<std::int32_t>(tensors.n_kv_heads);
        const auto head_dim    = static_cast<std::int32_t>(tensors.head_dim);
        const std::size_t row  = static_cast<std::size_t>(heads) * head_dim * 2;
        const std::size_t span = 3U * 256U; // alignment slack for the three staging tensors
        work.reset();
        if (work.capacity() <= span + 2 * row + sizeof(std::int32_t)) {
            throw std::logic_error("Program workspace cannot stage one graft KV row");
        }
        const auto chunk = static_cast<std::uint32_t>(std::min<std::size_t>(
            slots, (work.capacity() - span) / (2 * row + sizeof(std::int32_t))));
        std::vector<std::int32_t> positions(chunk);
        for (std::uint32_t begin = 0; begin < slots; begin += chunk) {
            const std::uint32_t count = std::min(chunk, slots - begin);
            work.reset();
            Tensor position = work.alloc(DType::I32, {static_cast<std::int32_t>(count)});
            Tensor k = work.alloc(DType::BF16, {head_dim, heads, static_cast<std::int32_t>(count)});
            Tensor v = work.alloc(DType::BF16, {head_dim, heads, static_cast<std::int32_t>(count)});
            for (std::uint32_t i = 0; i < count; ++i) {
                positions[i] = static_cast<std::int32_t>(begin + i);
            }
            CUDA_CHECK(cudaMemcpyAsync(position.data, positions.data(), count * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, device.stream));
            // Each layer's slice is [n_slots, heads, head_dim] row-major: head_dim innermost, then
            // heads, then slots -- the [head_dim, heads, T] operand the append Op consumes.
            const std::size_t layer_bytes = static_cast<std::size_t>(slots) * row;
            for (std::uint32_t layer = 0; layer < tensors.n_attn_layers; ++layer) {
                const std::size_t offset = layer * layer_bytes + static_cast<std::size_t>(begin) * row;
                CUDA_CHECK(cudaMemcpyAsync(k.data, tensors.k_data() + offset, count * row,
                                           cudaMemcpyHostToDevice, device.stream));
                CUDA_CHECK(cudaMemcpyAsync(v.data, tensors.v_data() + offset, count * row,
                                           cudaMemcpyHostToDevice, device.stream));
                ops::kv_cache_append(k, v, position, view.layer_view(layer), device.stream);
            }
            device.synchronize(); // the staging rows and positions are reused by the next chunk
        }
        work.reset();
        text_kv_addresses->commit_frontier(*text, slots);
        text_kv_addresses->deactivate(*text);

        // --- Speculative draft context: present, mapped over the graft, and zero ---
        if (backend_kv_addresses) {
            backend = backend_kv_addresses->create_inactive();
            if (!backend) {
                throw std::runtime_error("no backend KV address space for graft '" + graft.name +
                                         "'");
            }
            if (backend_frontier != 0) {
                const std::uint32_t backend_pages = kv_pages_for_frontier(backend_frontier);
                backend_kv_addresses->activate(*backend, backend_pages, 0, device.stream);
                backend_kv_addresses->ensure_mapped_to_tokens(*backend, backend_frontier,
                                                              device.stream);
                std::vector<DeviceKVPageHandle> physical;
                physical.reserve(backend_pages);
                for (std::uint32_t page = 0; page < backend_pages; ++page) {
                    physical.push_back(backend_kv_addresses->physical_page(*backend, page));
                }
                backend_kv_pages->physical_pool().zero_pages(physical, RankStreams(device.stream));
                backend_kv_addresses->commit_frontier(*backend, backend_frontier);
                backend_kv_addresses->deactivate(*backend);
            }
        }
        device.synchronize();
    } catch (...) {
        rollback();
        throw;
    }

    // Publication performs no allocation and cannot fail.
    history->text    = *text;
    history->backend = backend;
    history->owner   = this;
    auto& slot       = checkpoints[handle->index];
    slot.value       = CheckpointState{
              .kv                = history,
              .role              = runtime::CheckpointRole::SharedPrefix,
              .state             = *state,
              .identity          = identity,
              .key               = {identity->digests.at(slots), slots, prefix_identity_tag()},
              .frontier          = slots,
              .backend_frontier  = backend_frontier,
              .rope_delta        = 0,
              .tail_hidden_valid = false,
    };
    slot.reserved = false;
    slot.pins     = 1; // permanent lease: never released, demoted, reclaimed or consumed
    slot.external = true;
    refresh_history_requirements(history);
    return *handle;
}

} // namespace ninfer::models::qwen3_5::detail
