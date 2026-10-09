#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/host_context_arena.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/transfer_work.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct DFlashLocalStateSpec {
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
    std::int32_t kv_heads  = 0;
    std::int32_t head_dim  = 0;
};

struct StateImageSpec {
    LinearAttentionStatePoolSpec linear;
    std::int32_t hidden = 0;
    std::optional<DFlashLocalStateSpec> dflash_local;
};

struct StateImageHostLayout {
    StateImageSpec spec;
    LayoutRegion linear_conv;
    std::size_t linear_conv_layer_bytes = 0;
    LayoutRegion linear_recurrent;
    std::size_t linear_recurrent_layer_bytes = 0;
    LayoutRegion continuation_hidden;
    std::optional<LayoutRegion> dflash_local_k;
    std::optional<LayoutRegion> dflash_local_v;
    std::size_t dflash_local_layer_bytes = 0;
    std::size_t image_bytes              = 0;
};

// A run of consecutive Linear Attention layers whose state one rank holds. The host image keeps the
// whole model's layers at their global offsets, so where a layer sits on a device is invisible to it.
struct StateImageShard {
    std::size_t rank          = 0;
    std::uint32_t first_layer = 0;
    std::uint32_t layers      = 0;
};

struct StateImageDeviceLayout {
    // One shard per rank that holds Linear Attention layers, in layer order, and each shard's pool.
    // The continuation hidden and the DFlash local state stay on rank 0.
    std::vector<StateImageShard> shards;
    std::vector<LinearAttentionStatePoolLayout> linear;
    TensorRegion continuation_hidden;
    std::optional<CyclicKVCacheLayout> dflash_local;
    StateImageHostLayout host;
};

[[nodiscard]] TransferWork state_image_transfer_work(const StateImageHostLayout& layout);
[[nodiscard]] TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout);

// Every Linear Attention layer in one shard on rank 0.
[[nodiscard]] StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                                  const StateImageSpec& spec);
// `shards` must cover spec.linear.layers exactly, in order; each is laid out in its rank's builder.
[[nodiscard]] StateImageDeviceLayout
plan_state_image_device_pool(std::span<LayoutBuilder* const> builders,
                             std::span<const StateImageShard> shards, const StateImageSpec& spec);

struct HostStateImageView {
    std::byte* data                    = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateImageConstView {
    const std::byte* data              = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

class HostStatePool;

struct HostStateSlotHandle {
    std::uint32_t index        = 0;
    std::uint32_t generation   = 0;
    const HostStatePool* owner = nullptr;
};

/** Typed StateImage descriptors backed on demand by the shared Host arena; owns no cache policy.
 */
class HostStatePool {
public:
    HostStatePool(HostContextArena& arena, StateImageHostLayout layout);

    HostStatePool(const HostStatePool&)            = delete;
    HostStatePool& operator=(const HostStatePool&) = delete;
    HostStatePool(HostStatePool&&)                 = delete;
    HostStatePool& operator=(HostStatePool&&)      = delete;

    [[nodiscard]] std::optional<HostStateSlotHandle> allocate() noexcept;
    [[nodiscard]] bool can_allocate() const noexcept;
    [[nodiscard]] bool publish(HostStateSlotHandle handle) noexcept;
    [[nodiscard]] bool release(HostStateSlotHandle handle) noexcept;

    [[nodiscard]] HostStateImageView writable_view(HostStateSlotHandle handle);
    [[nodiscard]] HostStateImageConstView view(HostStateSlotHandle handle) const;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] std::uint32_t occupied() const noexcept { return occupied_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept {
        return static_cast<std::size_t>(occupied_) * layout_.image_bytes;
    }

    [[nodiscard]] const StateImageHostLayout& layout() const noexcept { return layout_; }

private:
    struct Slot {
        HostContextAllocation storage;
        std::uint32_t generation = 1;
    };

    [[nodiscard]] bool valid(HostStateSlotHandle handle) const noexcept;
    [[nodiscard]] std::byte* slot_data(std::uint32_t index) const noexcept;

    StateImageHostLayout layout_;
    HostContextArena* arena_ = nullptr;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t free_count_ = 0;
    std::uint32_t occupied_   = 0;
};

struct StateImageDeviceSlotView {
    // One view per shard, in layer order.
    std::vector<LinearAttentionStateSlotView> linear;
    Tensor continuation_hidden;
    std::optional<CyclicKVCacheSlotView> dflash_local;
};

/**
 * Caller-backed fixed storage for Qwen3.6 continuation state.
 *
 * Every absolute slot contains common GDN/hidden state and, for a DFlash Program, its local cyclic
 * K/V state. The pool owns neither slot roles nor logical checkpoint identity.
 */
class StateImageDevicePool {
public:
    // `backings[r]` is rank r's persistent device memory.
    StateImageDevicePool(std::span<const DeviceSpan> backings, const StateImageDeviceLayout& layout);
    // Everything on rank 0.
    StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout);

    StateImageDevicePool(const StateImageDevicePool&)            = delete;
    StateImageDevicePool& operator=(const StateImageDevicePool&) = delete;
    StateImageDevicePool(StateImageDevicePool&&)                 = delete;
    StateImageDevicePool& operator=(StateImageDevicePool&&)      = delete;

    [[nodiscard]] std::int32_t slot_count() const noexcept { return linear_.front()->slot_count(); }

    [[nodiscard]] StateImageDeviceSlotView slot_view(std::int32_t slot) const;
    [[nodiscard]] Tensor continuation_hidden_slot(std::int32_t slot) const;

    // The Linear Attention state pools, one per shard. `linear()` is for the single-shard case and
    // refuses a sharded pool rather than quietly answering for only its first layers.
    [[nodiscard]] std::size_t shard_count() const noexcept { return linear_.size(); }
    [[nodiscard]] const StateImageShard& shard(std::size_t index) const { return shards_.at(index); }
    [[nodiscard]] LinearAttentionStatePool& linear(std::size_t shard_index) {
        return *linear_.at(shard_index);
    }
    [[nodiscard]] const LinearAttentionStatePool& linear(std::size_t shard_index) const {
        return *linear_.at(shard_index);
    }
    [[nodiscard]] LinearAttentionStatePool& linear() { return *single_shard(); }
    [[nodiscard]] const LinearAttentionStatePool& linear() const { return *single_shard(); }

    [[nodiscard]] Tensor& continuation_hidden_store() noexcept { return continuation_hidden_; }

    [[nodiscard]] const Tensor& continuation_hidden_store() const noexcept {
        return continuation_hidden_;
    }

    [[nodiscard]] CyclicKVCache* dflash_local() noexcept;
    [[nodiscard]] const CyclicKVCache* dflash_local() const noexcept;

    [[nodiscard]] const StateImageHostLayout& host_layout() const noexcept { return host_layout_; }

    // Each shard's copies go on the stream of the rank that holds it; the continuation hidden and
    // the DFlash local state are rank 0's.
    void zero_slot(std::int32_t slot, RankStreams streams = {});
    void zero_all(RankStreams streams = {});
    void copy_slot(std::int32_t source, std::int32_t destination, RankStreams streams = {});
    void copy_dflash_local(std::int32_t source, std::int32_t destination,
                           RankStreams streams = {});
    void copy_to_host(std::int32_t source, HostStateImageView destination,
                      RankStreams streams = {}) const;
    void copy_from_host(HostStateImageConstView source, std::int32_t destination,
                        RankStreams streams = {});

private:
    void validate_host_layout(const StateImageHostLayout* layout, const std::byte* data) const;
    [[nodiscard]] LinearAttentionStatePool* single_shard() const;

    std::vector<StateImageShard> shards_;
    std::vector<std::unique_ptr<LinearAttentionStatePool>> linear_;
    Tensor continuation_hidden_;
    std::optional<CyclicKVCache> dflash_local_;
    StateImageHostLayout host_layout_;
};

} // namespace ninfer::models::qwen3_5
