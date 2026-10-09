#include "models/qwen3_5/state/state_image.h"

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t kStateImageAlignment = 256;

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(label);
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

std::uint32_t padded_dflash_capacity(std::uint32_t capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("StateImage DFlash capacity must be positive");
    }
    constexpr std::uint64_t alignment = 128;
    const std::uint64_t padded =
        (static_cast<std::uint64_t>(capacity) + alignment - 1U) & ~(alignment - 1U);
    if (padded > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("StateImage DFlash padded capacity exceeds int32");
    }
    return static_cast<std::uint32_t>(padded);
}

bool same_linear_spec(const LinearAttentionStatePoolSpec& left,
                      const LinearAttentionStatePoolSpec& right) noexcept {
    return left.layers == right.layers && left.conv_channels == right.conv_channels &&
           left.conv_width == right.conv_width && left.value_heads == right.value_heads &&
           left.value_head_dim == right.value_head_dim && left.key_head_dim == right.key_head_dim &&
           left.slot_count == right.slot_count && left.conv_dtype == right.conv_dtype &&
           left.recurrent_dtype == right.recurrent_dtype;
}

bool same_dflash_spec(const std::optional<DFlashLocalStateSpec>& left,
                      const std::optional<DFlashLocalStateSpec>& right) noexcept {
    if (left.has_value() != right.has_value()) { return false; }
    if (!left) { return true; }
    return left->layers == right->layers && left->capacity == right->capacity &&
           left->kv_heads == right->kv_heads && left->head_dim == right->head_dim;
}

bool same_region(const LayoutRegion& left, const LayoutRegion& right) noexcept {
    return left.offset == right.offset && left.bytes == right.bytes &&
           left.alignment == right.alignment;
}

bool same_optional_region(const std::optional<LayoutRegion>& left,
                          const std::optional<LayoutRegion>& right) noexcept {
    return left.has_value() == right.has_value() && (!left || same_region(*left, *right));
}

bool same_host_layout(const StateImageHostLayout& left,
                      const StateImageHostLayout& right) noexcept {
    return same_linear_spec(left.spec.linear, right.spec.linear) &&
           left.spec.hidden == right.spec.hidden &&
           same_dflash_spec(left.spec.dflash_local, right.spec.dflash_local) &&
           same_region(left.linear_conv, right.linear_conv) &&
           left.linear_conv_layer_bytes == right.linear_conv_layer_bytes &&
           same_region(left.linear_recurrent, right.linear_recurrent) &&
           left.linear_recurrent_layer_bytes == right.linear_recurrent_layer_bytes &&
           same_region(left.continuation_hidden, right.continuation_hidden) &&
           same_optional_region(left.dflash_local_k, right.dflash_local_k) &&
           same_optional_region(left.dflash_local_v, right.dflash_local_v) &&
           left.dflash_local_layer_bytes == right.dflash_local_layer_bytes &&
           left.image_bytes == right.image_bytes;
}

StateImageHostLayout plan_host_state_image(const StateImageSpec& spec) {
    if (spec.linear.layers == 0 || spec.linear.conv_channels <= 0 || spec.linear.conv_width <= 0 ||
        spec.linear.value_heads <= 0 || spec.linear.value_head_dim <= 0 ||
        spec.linear.key_head_dim <= 0 || spec.linear.slot_count <= 0 ||
        (spec.linear.conv_dtype != DType::BF16 && spec.linear.conv_dtype != DType::FP32) ||
        (spec.linear.recurrent_dtype != DType::FP32 &&
         spec.linear.recurrent_dtype != DType::FP16) ||
        spec.hidden <= 0) {
        throw std::invalid_argument("StateImage host geometry is invalid");
    }
    if (spec.dflash_local && (spec.dflash_local->layers == 0 || spec.dflash_local->kv_heads <= 0 ||
                              spec.dflash_local->head_dim <= 0)) {
        throw std::invalid_argument("StateImage host DFlash geometry is invalid");
    }

    LayoutBuilder builder;
    StateImageHostLayout host;
    host.spec = spec;
    const Tensor conv_slot(nullptr, spec.linear.conv_dtype,
                           {spec.linear.conv_channels, spec.linear.conv_width});
    const Tensor recurrent_slot(
        nullptr, spec.linear.recurrent_dtype,
        {spec.linear.key_head_dim, spec.linear.value_head_dim, spec.linear.value_heads});
    const Tensor hidden_slot(nullptr, DType::BF16, {spec.hidden});
    host.linear_conv_layer_bytes = conv_slot.bytes();
    host.linear_conv = builder.add(checked_mul(host.linear_conv_layer_bytes, spec.linear.layers,
                                               "StateImage host convolution bytes overflow"),
                                   kStateImageAlignment, "StateImage host convolution");
    host.linear_recurrent_layer_bytes = recurrent_slot.bytes();
    host.linear_recurrent =
        builder.add(checked_mul(host.linear_recurrent_layer_bytes, spec.linear.layers,
                                "StateImage host recurrent bytes overflow"),
                    kStateImageAlignment, "StateImage host recurrent");
    host.continuation_hidden = builder.add(hidden_slot.bytes(), kStateImageAlignment,
                                           "StateImage host continuation hidden");
    if (spec.dflash_local) {
        const Tensor local_slot(
            nullptr, DType::BF16,
            {spec.dflash_local->head_dim,
             static_cast<std::int32_t>(padded_dflash_capacity(spec.dflash_local->capacity)),
             spec.dflash_local->kv_heads});
        host.dflash_local_layer_bytes = local_slot.bytes();
        const std::size_t component_bytes =
            checked_mul(host.dflash_local_layer_bytes, spec.dflash_local->layers,
                        "StateImage host DFlash local bytes overflow");
        host.dflash_local_k =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local K");
        host.dflash_local_v =
            builder.add(component_bytes, kStateImageAlignment, "StateImage host DFlash local V");
    }
    host.image_bytes = builder.finish(kStateImageAlignment, "StateImage host image");
    return host;
}

std::byte* byte_offset(std::byte* base, std::size_t offset) noexcept { return base + offset; }

const std::byte* byte_offset(const std::byte* base, std::size_t offset) noexcept {
    return base + offset;
}

void validate_slot(std::int32_t slot, std::int32_t slot_count, const char* label) {
    if (slot < 0 || slot >= slot_count) { throw std::out_of_range(label); }
}

} // namespace

StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                    const StateImageSpec& spec) {
    LayoutBuilder* const builders[] = {&builder};
    const StateImageShard whole[]   = {{.rank = 0, .first_layer = 0, .layers = spec.linear.layers}};
    return plan_state_image_device_pool(builders, whole, spec);
}

StateImageDeviceLayout plan_state_image_device_pool(std::span<LayoutBuilder* const> builders,
                                                    std::span<const StateImageShard> shards,
                                                    const StateImageSpec& spec) {
    if (spec.hidden <= 0) {
        throw std::invalid_argument("StateImage hidden width must be positive");
    }
    if (shards.empty()) { throw std::invalid_argument("StateImage needs at least one shard"); }
    std::uint32_t next_layer = 0;
    for (const StateImageShard& shard : shards) {
        if (shard.first_layer != next_layer || shard.layers == 0 || shard.rank >= builders.size() ||
            builders[shard.rank] == nullptr) {
            throw std::invalid_argument("StateImage shards must cover the layers in order");
        }
        next_layer += shard.layers;
    }
    if (next_layer != spec.linear.layers) {
        throw std::invalid_argument("StateImage shards do not cover every Linear Attention layer");
    }

    StateImageDeviceLayout out;
    out.shards.assign(shards.begin(), shards.end());
    for (const StateImageShard& shard : shards) {
        LinearAttentionStatePoolSpec shard_spec = spec.linear;
        shard_spec.layers                       = shard.layers;
        out.linear.push_back(plan_linear_attention_state_pool(*builders[shard.rank], shard_spec));
    }
    // The continuation hidden and DFlash local state are read by the head and the draft, which run
    // on rank 0.
    LayoutBuilder& builder = *builders[0];
    out.continuation_hidden =
        builder.add_tensor(DType::BF16, {spec.hidden, spec.linear.slot_count}, kStateImageAlignment,
                           "StateImage continuation hidden");
    if (spec.dflash_local) {
        const DFlashLocalStateSpec& dflash = *spec.dflash_local;
        out.dflash_local =
            plan_cyclic_kv_cache(builder, dflash.layers, dflash.capacity, dflash.kv_heads,
                                 dflash.head_dim, spec.linear.slot_count);
    }

    out.host = plan_host_state_image(spec);
    return out;
}

TransferWork state_image_transfer_work(const StateImageHostLayout& layout) {
    std::size_t payload = checked_add(layout.linear_conv.bytes, layout.linear_recurrent.bytes,
                                      "StateImage transfer payload overflow");
    payload             = checked_add(payload, layout.continuation_hidden.bytes,
                                      "StateImage transfer payload overflow");
    if (layout.dflash_local_k) {
        if (!layout.spec.dflash_local || !layout.dflash_local_v) {
            throw std::invalid_argument("StateImage DFlash transfer layout is incomplete");
        }
        const std::size_t component_bytes =
            checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                        "StateImage transfer payload overflow");
        payload = checked_add(
            payload, checked_mul(component_bytes, 2U, "StateImage transfer payload overflow"),
            "StateImage transfer payload overflow");
    }
    const std::uint64_t operations =
        2ULL * layout.spec.linear.layers + 1ULL +
        (layout.spec.dflash_local ? 2ULL * layout.spec.dflash_local->layers : 0ULL);
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout) {
    if (!layout.spec.dflash_local || !layout.dflash_local_k || !layout.dflash_local_v) {
        throw std::invalid_argument("StateImage has no DFlash local component");
    }
    const std::size_t component_bytes =
        checked_mul(layout.dflash_local_layer_bytes, layout.spec.dflash_local->layers,
                    "StateImage DFlash transfer payload overflow");
    const std::uint64_t operations = 2ULL * layout.spec.dflash_local->layers;
    if (component_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
        operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage DFlash transfer work exceeds its representation");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(2U * component_bytes),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

HostStatePool::HostStatePool(HostContextArena& arena, StateImageHostLayout layout)
    : layout_(std::move(layout)), arena_(&arena) {
    if (!same_host_layout(layout_, plan_host_state_image(layout_.spec))) {
        throw std::invalid_argument("HostStatePool image layout is invalid");
    }
    if (layout_.image_bytes < arena_->minimum_allocation_bytes()) {
        throw std::invalid_argument("Host state geometry is below the shared arena minimum");
    }
    const std::size_t maximum_slots = arena_->capacity_bytes() / layout_.image_bytes;
    if (maximum_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host state descriptor capacity exceeds uint32");
    }
    const auto capacity = static_cast<std::uint32_t>(maximum_slots);
    slots_.resize(capacity);
    free_slots_.resize(capacity);
    free_count_ = capacity;
    for (std::uint32_t index = 0; index < capacity; ++index) {
        free_slots_[index] = capacity - 1U - index;
    }
}

std::optional<HostStateSlotHandle> HostStatePool::allocate() noexcept {
    if (free_count_ == 0) { return std::nullopt; }
    auto storage = arena_->allocate(layout_.image_bytes);
    if (!storage) { return std::nullopt; }
    const std::uint32_t index = free_slots_[--free_count_];
    Slot& slot                = slots_[index];
    slot.storage              = std::move(*storage);
    ++occupied_;
    return HostStateSlotHandle{.index = index, .generation = slot.generation, .owner = this};
}

bool HostStatePool::can_allocate() const noexcept {
    return free_count_ != 0 && arena_->can_allocate(layout_.image_bytes);
}

bool HostStatePool::publish(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    slots_[handle.index].storage.publish();
    return true;
}

bool HostStatePool::release(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    Slot& slot = slots_[handle.index];
    (void)slot.storage.release();
    if (++slot.generation == 0) { ++slot.generation; }
    free_slots_[free_count_++] = handle.index;
    --occupied_;
    return true;
}

HostStateImageView HostStatePool::writable_view(HostStateSlotHandle handle) {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

HostStateImageConstView HostStatePool::view(HostStateSlotHandle handle) const {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

std::uint32_t HostStatePool::capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
}

bool HostStatePool::valid(HostStateSlotHandle handle) const noexcept {
    return handle.owner == this && handle.index < slots_.size() &&
           slots_[handle.index].storage.valid() &&
           slots_[handle.index].generation == handle.generation;
}

std::byte* HostStatePool::slot_data(std::uint32_t index) const noexcept {
    return slots_[index].storage.data();
}

StateImageDevicePool::StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout)
    : StateImageDevicePool(std::span<const DeviceSpan>(&backing, 1), layout) {}

StateImageDevicePool::StateImageDevicePool(std::span<const DeviceSpan> backings,
                                           const StateImageDeviceLayout& layout)
    : shards_(layout.shards),
      continuation_hidden_(layout.continuation_hidden.bind(backings.front())),
      host_layout_(layout.host) {
    if (layout.shards.empty() || layout.shards.size() != layout.linear.size()) {
        throw std::invalid_argument("StateImage shard inventory is inconsistent");
    }
    linear_.reserve(layout.shards.size());
    for (std::size_t index = 0; index < layout.shards.size(); ++index) {
        if (layout.shards[index].rank >= backings.size()) {
            throw std::invalid_argument("StateImage shard names a rank with no backing");
        }
        linear_.push_back(std::make_unique<LinearAttentionStatePool>(
            backings[layout.shards[index].rank], layout.linear[index]));
    }
    if (continuation_hidden_.dtype != DType::BF16 || !continuation_hidden_.is_contiguous() ||
        continuation_hidden_.ne[0] != host_layout_.spec.hidden ||
        continuation_hidden_.ne[1] != linear_.front()->slot_count()) {
        throw std::invalid_argument("StateImage continuation hidden layout is inconsistent");
    }
    if (layout.dflash_local.has_value() != host_layout_.spec.dflash_local.has_value()) {
        throw std::invalid_argument("StateImage DFlash layout is inconsistent");
    }
    // The device components must add up to the host image's geometry: rebuild the whole-model spec
    // from the shards and compare.
    LinearAttentionStatePoolSpec whole = layout.linear.front().spec;
    whole.layers                       = 0;
    for (const auto& shard_layout : layout.linear) { whole.layers += shard_layout.spec.layers; }
    StateImageSpec device_spec{.linear = whole, .hidden = continuation_hidden_.ne[0]};
    if (layout.dflash_local) {
        device_spec.dflash_local = DFlashLocalStateSpec{
            .layers   = static_cast<std::uint32_t>(layout.dflash_local->k.size()),
            .capacity = layout.dflash_local->capacity,
            .kv_heads = layout.dflash_local->num_kv_heads,
            .head_dim = layout.dflash_local->head_dim,
        };
    }
    if (!same_host_layout(host_layout_, plan_host_state_image(device_spec))) {
        throw std::invalid_argument("StateImage host layout does not match its device components");
    }
    if (layout.dflash_local) {
        if (layout.dflash_local->lane_capacity != linear_.front()->slot_count()) {
            throw std::invalid_argument("StateImage components do not share one slot geometry");
        }
        dflash_local_.emplace(backings.front(), *layout.dflash_local);
    }
}

LinearAttentionStatePool* StateImageDevicePool::single_shard() const {
    if (linear_.size() != 1) {
        throw std::logic_error("StateImage pool is sharded across ranks: address a shard");
    }
    return linear_.front().get();
}

StateImageDeviceSlotView StateImageDevicePool::slot_view(std::int32_t slot) const {
    StateImageDeviceSlotView view{
        .continuation_hidden = continuation_hidden_slot(slot),
    };
    for (const auto& pool : linear_) { view.linear.push_back(pool->slot_view(slot)); }
    if (dflash_local_) { view.dflash_local = dflash_local_->slot_view(slot); }
    return view;
}

Tensor StateImageDevicePool::continuation_hidden_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage slot is out of range");
    return continuation_hidden_.slice(1, slot, 1).view({host_layout_.spec.hidden});
}

CyclicKVCache* StateImageDevicePool::dflash_local() noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

const CyclicKVCache* StateImageDevicePool::dflash_local() const noexcept {
    return dflash_local_ ? &*dflash_local_ : nullptr;
}

void StateImageDevicePool::zero_slot(std::int32_t slot, RankStreams streams) {
    validate_slot(slot, slot_count(), "StateImage zero slot is out of range");
    for (std::size_t index = 0; index < linear_.size(); ++index) {
        linear_[index]->zero_slot(slot, streams[shards_[index].rank]);
    }
    const cudaStream_t stream = streams[0];
    const Tensor hidden       = continuation_hidden_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(hidden.data, 0, hidden.bytes(), stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, slot, 1);
            const Tensor v                    = view.v.slice(3, slot, 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::zero_all(RankStreams streams) {
    for (std::size_t index = 0; index < linear_.size(); ++index) {
        linear_[index]->zero_all(streams[shards_[index].rank]);
    }
    const cudaStream_t stream = streams[0];
    CUDA_CHECK(cudaMemsetAsync(continuation_hidden_.data, 0, continuation_hidden_.bytes(), stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), stream));
        }
    }
}

void StateImageDevicePool::copy_slot(std::int32_t source, std::int32_t destination,
                                     RankStreams streams) {
    validate_slot(source, slot_count(), "StateImage copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage copy destination is out of range");
    if (source == destination) { return; }
    for (std::size_t index = 0; index < linear_.size(); ++index) {
        linear_[index]->copy_slot(source, destination, streams[shards_[index].rank]);
    }
    const cudaStream_t stream       = streams[0];
    const Tensor source_hidden      = continuation_hidden_slot(source);
    const Tensor destination_hidden = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(destination_hidden.data, source_hidden.data,
                               destination_hidden.bytes(), cudaMemcpyDeviceToDevice, stream));
    if (dflash_local_) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
    }
}

void StateImageDevicePool::copy_dflash_local(std::int32_t source, std::int32_t destination,
                                             RankStreams streams) {
    const cudaStream_t stream = streams[0];
    validate_slot(source, slot_count(), "StateImage DFlash copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage DFlash copy destination is out of range");
    if (!dflash_local_) { throw std::logic_error("StateImage has no DFlash local component"); }
    if (source != destination) {
        dflash_local_->copy_slot_from(*dflash_local_, source, destination, stream);
    }
}

void StateImageDevicePool::validate_host_layout(const StateImageHostLayout* layout,
                                                const std::byte* data) const {
    if (layout == nullptr || data == nullptr || !same_host_layout(*layout, host_layout_)) {
        throw std::invalid_argument("Host and device StateImage layouts do not match");
    }
}

void StateImageDevicePool::copy_to_host(std::int32_t source, HostStateImageView destination,
                                        RankStreams streams) const {
    validate_slot(source, slot_count(), "StateImage copy-to-host source is out of range");
    validate_host_layout(destination.layout, destination.data);
    // Each shard's layers land at their global offsets in the one host image.
    for (std::size_t index = 0; index < linear_.size(); ++index) {
        const cudaStream_t shard_stream = streams[shards_[index].rank];
        for (std::uint32_t local = 0; local < linear_[index]->layer_count(); ++local) {
            const std::size_t layer = shards_[index].first_layer + local;
            const Tensor conv       = linear_[index]->conv_slot(local, source);
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data, host_layout_.linear_conv.offset +
                                                  layer * host_layout_.linear_conv_layer_bytes),
                conv.data, conv.bytes(), cudaMemcpyDeviceToHost, shard_stream));
            const Tensor recurrent = linear_[index]->recurrent_slot(local, source);
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data,
                            host_layout_.linear_recurrent.offset +
                                layer * host_layout_.linear_recurrent_layer_bytes),
                recurrent.data, recurrent.bytes(), cudaMemcpyDeviceToHost, shard_stream));
        }
    }
    const cudaStream_t stream = streams[0];
    const Tensor hidden       = continuation_hidden_slot(source);
    CUDA_CHECK(
        cudaMemcpyAsync(byte_offset(destination.data, host_layout_.continuation_hidden.offset),
                        hidden.data, hidden.bytes(), cudaMemcpyDeviceToHost, stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, source, 1);
            const Tensor v                    = view.v.slice(3, source, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data, host_layout_.dflash_local_k->offset +
                                                  layer * host_layout_.dflash_local_layer_bytes),
                k.data, k.bytes(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                byte_offset(destination.data, host_layout_.dflash_local_v->offset +
                                                  layer * host_layout_.dflash_local_layer_bytes),
                v.data, v.bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }
}

void StateImageDevicePool::copy_from_host(HostStateImageConstView source, std::int32_t destination,
                                          RankStreams streams) {
    validate_slot(destination, slot_count(),
                  "StateImage copy-from-host destination is out of range");
    validate_host_layout(source.layout, source.data);
    for (std::size_t index = 0; index < linear_.size(); ++index) {
        const cudaStream_t shard_stream = streams[shards_[index].rank];
        for (std::uint32_t local = 0; local < linear_[index]->layer_count(); ++local) {
            const std::size_t layer = shards_[index].first_layer + local;
            const Tensor conv       = linear_[index]->conv_slot(local, destination);
            CUDA_CHECK(cudaMemcpyAsync(
                conv.data,
                byte_offset(source.data, host_layout_.linear_conv.offset +
                                             layer * host_layout_.linear_conv_layer_bytes),
                conv.bytes(), cudaMemcpyHostToDevice, shard_stream));
            const Tensor recurrent = linear_[index]->recurrent_slot(local, destination);
            CUDA_CHECK(cudaMemcpyAsync(
                recurrent.data,
                byte_offset(source.data, host_layout_.linear_recurrent.offset +
                                             layer * host_layout_.linear_recurrent_layer_bytes),
                recurrent.bytes(), cudaMemcpyHostToDevice, shard_stream));
        }
    }
    const cudaStream_t stream = streams[0];
    const Tensor hidden       = continuation_hidden_slot(destination);
    CUDA_CHECK(cudaMemcpyAsync(hidden.data,
                               byte_offset(source.data, host_layout_.continuation_hidden.offset),
                               hidden.bytes(), cudaMemcpyHostToDevice, stream));
    if (dflash_local_) {
        for (std::uint32_t layer = 0; layer < dflash_local_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = dflash_local_->layer_view(layer);
            const Tensor k                    = view.k.slice(3, destination, 1);
            const Tensor v                    = view.v.slice(3, destination, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                k.data,
                byte_offset(source.data, host_layout_.dflash_local_k->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                k.bytes(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                v.data,
                byte_offset(source.data, host_layout_.dflash_local_v->offset +
                                             layer * host_layout_.dflash_local_layer_bytes),
                v.bytes(), cudaMemcpyHostToDevice, stream));
        }
    }
}

} // namespace ninfer::models::qwen3_5
