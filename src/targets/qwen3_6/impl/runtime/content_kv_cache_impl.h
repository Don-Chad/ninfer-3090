#pragma once

// Content-addressed host tier for resident context: chain-keyed KV pages plus recurrent-state
// anchors, so a previously computed prefix restores through PCIe instead of being re-prefilled,
// and harness branches sharing a prefix deduplicate against the same stored pages.
//
// Save runs at terminal completion on the serve thread; restore runs inside begin_request
// before the suffix prefill. Both borrow device staging from the idle workspace arena and move
// payloads with the hostpack kernels, so steady-state decode never sees this machinery.

#include "core/arena.h"
#include "core/device.h"
#include "core/kv_host_cache.h"
#include "core/paged_kv_hostpack.h"
#include "targets/qwen3_6/impl/runtime/content_chain.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

class ContentKvCache {
public:
    using RestorePlan = qwen3_6::detail::ContentRestorePlan;

    ContentKvCache(std::size_t budget_bytes, std::uint64_t salt, const PagedKVPool& text_pool,
                   const PagedKVPool* backend_pool, const LinearAttentionStatePool& linear,
                   std::uint32_t max_concurrency)
        : salt_(salt),
          linear_(linear),
          max_concurrency_(max_concurrency),
          text_geometry_(hostpack_geometry(text_pool)),
          has_backend_(backend_pool != nullptr),
          page_ids_device_(sizeof(std::int32_t) *
                           (text_pool.page_group_count() +
                            (backend_pool != nullptr ? backend_pool->page_group_count() : 0U))) {
        if (has_backend_) { backend_geometry_ = hostpack_geometry(*backend_pool); }
        KvHostCache::Config config;
        config.budget_bytes       = budget_bytes;
        config.text_page_bytes    = text_geometry_.packed_page_bytes;
        config.backend_page_bytes = has_backend_ ? backend_geometry_.packed_page_bytes : 0;
        config.anchor_state_bytes = anchor_state_bytes();
        store_                    = std::make_unique<KvHostCache>(config);
    }

    [[nodiscard]] const KvHostCache::Stats& stats() const noexcept { return store_->stats(); }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return store_->used_bytes(); }
    [[nodiscard]] std::size_t budget_bytes() const noexcept { return store_->budget_bytes(); }

    [[nodiscard]] std::optional<RestorePlan> probe(const PreparedPromptData& prompt) const {
        const auto tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
        if (tokens < 2 * kPagedKVPageSize) { return std::nullopt; }
        const auto stream                = content_chain::stream_of(prompt);
        const content_chain::Chain chain = content_chain::build(stream, tokens, salt_);
        const std::uint32_t present =
            store_->present_prefix(KvHostCache::PageKind::Text, chain.page_keys);
        for (std::uint32_t pages = present; pages >= 1; --pages) {
            RestorePlan best{};
            for (const std::uint64_t anchor_key :
                 store_->find_anchor_keys(chain.page_keys[pages - 1])) {
                const auto meta = store_->find_anchor(anchor_key);
                if (!meta) { continue; }
                if (meta->frontier >= tokens ||
                    meta->frontier / kPagedKVPageSize != pages ||
                    meta->rope_delta != prompt.rope_delta ||
                    meta->frontier <= best.frontier) {
                    continue;
                }
                if (chain.key_at(stream, meta->frontier) != anchor_key) { continue; }
                best = RestorePlan{meta->frontier, meta->backend_tokens, anchor_key};
            }
            if (best.frontier != 0) { return best; }
        }
        return std::nullopt;
    }

    // Sequence KV must already be reserved, bound, and materialized past the plan's frontier.
    void restore(SequenceState& sequence, const RestorePlan& plan,
                 const PreparedPromptData& prompt, WorkspaceArena& work, DeviceContext& device) {
        const auto stream                = content_chain::stream_of(prompt);
        const content_chain::Chain chain = content_chain::build(
            stream, static_cast<std::uint32_t>(prompt.token_ids.size()), salt_);
        restore_pool(KvHostCache::PageKind::Text, text_geometry_, sequence.kv->text,
                     page_keys_for(KvHostCache::PageKind::Text, chain, plan.frontier,
                                   plan.anchor_key),
                     work, device);
        if (has_backend_ && plan.backend_tokens != 0 && sequence.kv->backend) {
            restore_pool(KvHostCache::PageKind::Backend, backend_geometry_,
                         *sequence.kv->backend,
                         page_keys_for(KvHostCache::PageKind::Backend, chain, plan.backend_tokens,
                                       plan.anchor_key),
                         work, device);
        }
        const auto slices = anchor_state_slices(
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency_));
        store_->restore_anchor_state(plan.anchor_key, slices, device.stream);
        store_->note_hit(plan.frontier);
        store_->touch_segment(plan.anchor_key);
    }

    void save(const SequenceState& sequence, WorkspaceArena& work, DeviceContext& device) {
        if (!sequence.retained || !sequence.kv) { return; }
        const std::uint32_t frontier =
            std::min(sequence.execution_frontier, sequence.text_kv_valid);
        if (frontier < 2 * kPagedKVPageSize || sequence.ledger.size() < frontier ||
            sequence.prefix_identity.size() < frontier) {
            return;
        }
        const auto stream = content_chain::stream_of(sequence.ledger, sequence.prefix_identity);
        const content_chain::Chain chain = content_chain::build(stream, frontier, salt_);
        const std::uint32_t backend_tokens =
            sequence.kv->backend ? std::min(sequence.mtp_kv_valid, frontier) : 0;
        bool copied = false;
        if (!store_->find_anchor(chain.final_key)) {
            copied |= stage_segment(
                sequence, chain, frontier, chain.final_key, backend_tokens,
                LinearStateSlots::current_state_slot(sequence.lane, max_concurrency_), work,
                device);
        } else {
            store_->touch_segment(chain.final_key);
        }
        const std::uint32_t checkpoint = sequence.turn_checkpoint.frontier;
        if (sequence.turn_checkpoint.valid && checkpoint >= 2 * kPagedKVPageSize &&
            checkpoint < frontier) {
            const std::uint64_t checkpoint_key = chain.key_at(stream, checkpoint);
            if (!store_->find_anchor(checkpoint_key)) {
                copied |= stage_segment(
                    sequence, chain, checkpoint, checkpoint_key,
                    std::min(backend_tokens, checkpoint),
                    LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency_),
                    work, device);
            }
        }
        if (copied) { CUDA_CHECK(cudaStreamSynchronize(device.stream)); }
    }

private:
    [[nodiscard]] std::size_t anchor_state_bytes() const {
        std::size_t bytes = 0;
        for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
            bytes += slot_bytes(linear_.conv[layer]);
            bytes += slot_bytes(linear_.recurrent[layer]);
        }
        // Distinct tail pad keeps the anchor size-class from colliding with a page class.
        return bytes + 16;
    }

    [[nodiscard]] std::size_t slot_bytes(const Tensor& tensor) const {
        const std::size_t total =
            static_cast<std::size_t>(tensor.ne[3]) * static_cast<std::size_t>(tensor.nb[3]);
        return total / static_cast<std::size_t>(linear_.slot_count());
    }

    [[nodiscard]] std::vector<KvHostCopySlice> anchor_state_slices(std::int32_t slot) const {
        std::vector<KvHostCopySlice> slices;
        slices.reserve(2ULL * linear_.layer_count() + 1);
        for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
            for (const Tensor* tensor : {&linear_.conv[layer], &linear_.recurrent[layer]}) {
                const std::size_t bytes = slot_bytes(*tensor);
                slices.push_back(KvHostCopySlice{
                    static_cast<std::byte*>(tensor->data) + static_cast<std::size_t>(slot) * bytes,
                    bytes});
            }
        }
        slices.push_back(KvHostCopySlice{sentinel_.p, 16});
        return slices;
    }

    [[nodiscard]] std::vector<std::uint64_t>
    page_keys_for(KvHostCache::PageKind kind, const content_chain::Chain& chain,
                  std::uint32_t tokens, std::uint64_t anchor_key) const {
        const std::uint32_t full  = tokens / kPagedKVPageSize;
        const std::uint32_t total = (tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
        std::vector<std::uint64_t> keys(total);
        for (std::uint32_t page = 0; page < total; ++page) {
            if (page < full) {
                keys[page] = kind == KvHostCache::PageKind::Text
                                 ? chain.page_keys[page]
                                 : content_chain::fold(chain.page_keys[page], 0x6261636b656e64ULL);
            } else {
                keys[page] = partial_page_key(anchor_key, kind);
            }
        }
        return keys;
    }

    void restore_pool(KvHostCache::PageKind kind, const HostpackGeometry& geometry,
                      const PagedKVAllocation& allocation, std::span<const std::uint64_t> keys,
                      WorkspaceArena& work, DeviceContext& device) {
        if (keys.empty()) { return; }
        const auto scope = work.scope();
        auto* staging    = static_cast<std::byte*>(
            work.alloc_bytes(geometry.packed_page_bytes * keys.size()).data);
        for (std::size_t page = 0; page < keys.size(); ++page) {
            const KvHostCopySlice slice{staging + geometry.packed_page_bytes * page,
                                        geometry.packed_page_bytes};
            store_->restore_page(kind, keys[page], std::span(&slice, 1), device.stream);
        }
        upload_page_ids(allocation.page_ids().subspan(0, keys.size()), device);
        hostpack_scatter(geometry, static_cast<const std::int32_t*>(page_ids_device_.p),
                         static_cast<std::uint32_t>(keys.size()), staging, device.stream);
        // Arena staging is reused stream-ordered; the scatter completes before any later
        // same-stream producer can overwrite it.
    }

    [[nodiscard]] bool stage_segment(const SequenceState& sequence,
                                     const content_chain::Chain& chain, std::uint32_t frontier,
                                     std::uint64_t anchor_key, std::uint32_t backend_tokens,
                                     std::int32_t state_slot, WorkspaceArena& work,
                                     DeviceContext& device) {
        const std::vector<std::uint64_t> text_keys =
            page_keys_for(KvHostCache::PageKind::Text, chain, frontier, anchor_key);
        const std::vector<std::uint64_t> backend_keys =
            backend_tokens == 0
                ? std::vector<std::uint64_t>{}
                : page_keys_for(KvHostCache::PageKind::Backend, chain, backend_tokens, anchor_key);
        const bool staged_text = stage_pool(KvHostCache::PageKind::Text, text_geometry_,
                                            sequence.kv->text, text_keys, work, device);
        const bool staged_backend =
            backend_keys.empty() ||
            stage_pool(KvHostCache::PageKind::Backend, backend_geometry_, *sequence.kv->backend,
                       backend_keys, work, device);
        KvHostCache::AnchorMeta meta;
        meta.frontier       = frontier;
        meta.backend_tokens = backend_tokens;
        meta.rope_delta     = sequence.rope_delta;
        const bool staged_anchor =
            staged_text && staged_backend &&
            store_->stage_anchor(anchor_key, meta, anchor_state_slices(state_slot), device.stream);
        if (!staged_anchor) {
            store_->abort_segment();
            return false;
        }
        store_->seal_segment(anchor_key, text_keys, backend_keys,
                             chain.page_keys[frontier / kPagedKVPageSize - 1]);
        return true;
    }

    [[nodiscard]] bool stage_pool(KvHostCache::PageKind kind, const HostpackGeometry& geometry,
                                  const PagedKVAllocation& allocation,
                                  std::span<const std::uint64_t> keys, WorkspaceArena& work,
                                  DeviceContext& device) {
        std::vector<std::uint32_t> missing;
        for (std::uint32_t page = 0; page < keys.size(); ++page) {
            if (!store_->has_page(kind, keys[page])) { missing.push_back(page); }
        }
        if (missing.empty()) { return true; }
        const auto scope = work.scope();
        auto* staging    = static_cast<std::byte*>(
            work.alloc_bytes(geometry.packed_page_bytes * missing.size()).data);
        std::vector<std::int32_t> gather_ids(missing.size());
        const std::span<const std::int32_t> page_ids = allocation.page_ids();
        for (std::size_t index = 0; index < missing.size(); ++index) {
            gather_ids[index] = page_ids[missing[index]];
        }
        upload_page_ids(gather_ids, device);
        hostpack_gather(geometry, static_cast<const std::int32_t*>(page_ids_device_.p),
                        static_cast<std::uint32_t>(missing.size()), staging, device.stream);
        for (std::size_t index = 0; index < missing.size(); ++index) {
            const KvHostCopySlice slice{staging + geometry.packed_page_bytes * index,
                                        geometry.packed_page_bytes};
            if (!store_->stage_page(kind, keys[missing[index]], std::span(&slice, 1),
                                    device.stream)) {
                return false;
            }
        }
        // Staged D2H copies read arena staging that the scope releases on return.
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        return true;
    }

    void upload_page_ids(std::span<const std::int32_t> ids, DeviceContext& device) {
        CUDA_CHECK(cudaMemcpyAsync(page_ids_device_.p, ids.data(),
                                   ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device.stream));
    }

    [[nodiscard]] static std::uint64_t partial_page_key(std::uint64_t anchor_key,
                                                        KvHostCache::PageKind kind) {
        return content_chain::fold(anchor_key, kind == KvHostCache::PageKind::Text
                                                   ? 0x7061727469616cULL
                                                   : 0x7061727469616c62ULL);
    }

    std::uint64_t salt_;
    const LinearAttentionStatePool& linear_;
    std::uint32_t max_concurrency_;
    HostpackGeometry text_geometry_;
    HostpackGeometry backend_geometry_;
    bool has_backend_;
    DeviceBuffer page_ids_device_;
    DeviceBuffer sentinel_{16};
    std::unique_ptr<KvHostCache> store_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
