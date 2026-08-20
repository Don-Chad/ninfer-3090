#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include <ninfer/targets/qwen3_6/vision.h>
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace overlay_detail {

using OverlayClock = std::chrono::steady_clock;

inline std::size_t staging_align(std::size_t bytes) {
    constexpr std::size_t kAlign = 256;
    return (bytes + kAlign - 1) / kAlign * kAlign;
}

inline std::ptrdiff_t byte_delta(const std::byte* target, const std::byte* source) {
    return target - source;
}

inline Tensor rebase(Tensor tensor, std::ptrdiff_t delta) {
    tensor.data = static_cast<std::byte*>(tensor.data) + delta;
    return tensor;
}

inline Weight rebase(Weight weight, std::ptrdiff_t delta) {
    const auto shift = [delta](const void* pointer) -> const void* {
        return pointer == nullptr
                   ? nullptr
                   : static_cast<const void*>(static_cast<const std::byte*>(pointer) + delta);
    };
    weight.payload = shift(weight.payload);
    weight.qdata   = shift(weight.qdata);
    weight.qhigh   = shift(weight.qhigh);
    weight.scales  = shift(weight.scales);
    return weight;
}

inline qwen3_6::VisionLayerWeights rebase_layer(const qwen3_6::VisionLayerWeights& source,
                                                std::ptrdiff_t delta) {
    qwen3_6::VisionLayerWeights out = source;
    out.qkv                         = rebase(source.qkv, delta);
    out.qkv_bias                    = rebase(source.qkv_bias, delta);
    out.output                      = rebase(source.output, delta);
    out.output_bias                 = rebase(source.output_bias, delta);
    out.fc1                         = rebase(source.fc1, delta);
    out.fc1_bias                    = rebase(source.fc1_bias, delta);
    out.fc2                         = rebase(source.fc2, delta);
    out.fc2_bias                    = rebase(source.fc2_bias, delta);
    out.norm1_weight                = rebase(source.norm1_weight, delta);
    out.norm1_bias                  = rebase(source.norm1_bias, delta);
    out.norm2_weight                = rebase(source.norm2_weight, delta);
    out.norm2_bias                  = rebase(source.norm2_bias, delta);
    return out;
}

} // namespace overlay_detail

// Streams vision weights from the pinned block through borrowed device staging:
// a fixed prelude region (patch/position embedding), a fixed merger region, and
// two layer slots refilled on the load stream one layer ahead of compute. All
// synchronization is device-side (events); the host never blocks between
// layers.
class VisionWeightStream {
public:
    VisionWeightStream(DeviceContext& device, const qwen3_6::VisionOverlayAssets& assets,
                       std::byte* staging)
        : device_(device), assets_(assets) {
        using overlay_detail::staging_align;
        const qwen3_6::VisionOverlayLayout& layout = assets.layout;
        prelude_ = staging;
        merger_  = prelude_ + staging_align(layout.prelude_bytes);
        slot_[0] = merger_ + staging_align(layout.merger_bytes);
        slot_[1] = slot_[0] + staging_align(layout.slot_bytes);
        for (cudaEvent_t& event : uploaded_) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&prelude_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&merger_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_fence_, cudaEventDisableTiming));

        const std::byte* block = assets.pinned_block;
        CUDA_CHECK(cudaMemcpyAsync(prelude_, block + layout.prelude_begin, layout.prelude_bytes,
                                   cudaMemcpyHostToDevice, copy_stream()));
        CUDA_CHECK(cudaEventRecord(prelude_event_, copy_stream()));
        CUDA_CHECK(cudaMemcpyAsync(merger_, block + layout.merger_begin, layout.merger_bytes,
                                   cudaMemcpyHostToDevice, copy_stream()));
        CUDA_CHECK(cudaEventRecord(merger_event_, copy_stream()));
        upload_bytes_ = layout.prelude_bytes + layout.merger_bytes;
        reset(device_.stream);
    }

    ~VisionWeightStream() {
        // The window synchronizes the device before restore; events are idle here.
        for (cudaEvent_t event : uploaded_) { (void)cudaEventDestroy(event); }
        (void)cudaEventDestroy(prelude_event_);
        (void)cudaEventDestroy(merger_event_);
        (void)cudaEventDestroy(compute_fence_);
    }

    VisionWeightStream(const VisionWeightStream&)            = delete;
    VisionWeightStream& operator=(const VisionWeightStream&) = delete;

    // Rebased view of the host weights: every layer's tensors point at the slot
    // that will hold the layer when arrive(layer) admits it.
    [[nodiscard]] qwen3_6::VisionWeights window_weights(const qwen3_6::VisionWeights& host) const {
        using overlay_detail::byte_delta;
        using overlay_detail::rebase;
        using overlay_detail::rebase_layer;
        const qwen3_6::VisionOverlayLayout& layout = assets_.layout;
        const std::byte* block                     = assets_.pinned_block;

        qwen3_6::VisionWeights out = host;
        const std::ptrdiff_t prelude_delta =
            byte_delta(prelude_, block + layout.prelude_begin);
        out.common.patch_embedding      = rebase(host.common.patch_embedding, prelude_delta);
        out.common.patch_embedding_bias = rebase(host.common.patch_embedding_bias, prelude_delta);
        out.common.position_embedding   = rebase(host.common.position_embedding, prelude_delta);
        for (std::size_t layer = 0; layer < host.common.layers.size(); ++layer) {
            const std::ptrdiff_t delta =
                byte_delta(slot_[layer % 2], block + layout.layer_begin[layer]);
            out.common.layers[layer] = rebase_layer(host.common.layers[layer], delta);
        }
        const std::ptrdiff_t merger_delta = byte_delta(merger_, block + layout.merger_begin);
        out.common.merger_fc1             = rebase(host.common.merger_fc1, merger_delta);
        out.common.merger_fc1_bias        = rebase(host.common.merger_fc1_bias, merger_delta);
        out.common.merger_norm_weight     = rebase(host.common.merger_norm_weight, merger_delta);
        out.common.merger_norm_bias       = rebase(host.common.merger_norm_bias, merger_delta);
        out.merger_fc2                    = rebase(host.merger_fc2, merger_delta);
        out.merger_fc2_bias               = rebase(host.merger_fc2_bias, merger_delta);
        return out;
    }

    // Prepare the next encode pass: uploads of layers 0 and 1 are issued after
    // everything already submitted on the compute stream (the previous item's
    // tail layers still own the slots until then).
    void reset(cudaStream_t compute) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
        next_upload_ = 0;
        upload_next_layer();
        upload_next_layer();
    }

    void prelude_ready(cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
    }

    void merger_ready(cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
    }

    // Called at the top of the encoder loop for `layer`: gates compute on the
    // slot upload, then refills the slot the previous layer just vacated.
    void arrive(std::uint32_t layer, cudaStream_t compute) {
        CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
        // At this point every op of layers < layer is issued; the fence makes the
        // copy stream wait for them before refilling the slot that layer-1 held
        // with layer+1's weights.
        if (next_upload_ == layer + 1 &&
            next_upload_ < static_cast<std::uint32_t>(VisionScheduleConfig::layers)) {
            CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
            upload_next_layer();
        }
    }

    [[nodiscard]] std::size_t uploaded_bytes() const noexcept { return upload_bytes_; }

private:
    [[nodiscard]] cudaStream_t copy_stream() const noexcept { return device_.load_stream; }

    void upload_next_layer() {
        if (next_upload_ >= static_cast<std::uint32_t>(VisionScheduleConfig::layers)) { return; }
        const std::uint32_t layer = next_upload_++;
        const qwen3_6::VisionOverlayLayout& layout = assets_.layout;
        CUDA_CHECK(cudaMemcpyAsync(slot_[layer % 2],
                                   assets_.pinned_block + layout.layer_begin[layer],
                                   layout.layer_bytes[layer], cudaMemcpyHostToDevice,
                                   copy_stream()));
        CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], copy_stream()));
        upload_bytes_ += layout.layer_bytes[layer];
    }

    DeviceContext& device_;
    const qwen3_6::VisionOverlayAssets& assets_;
    std::byte* prelude_       = nullptr;
    std::byte* merger_        = nullptr;
    std::byte* slot_[2]       = {nullptr, nullptr};
    cudaEvent_t uploaded_[2]  = {nullptr, nullptr};
    cudaEvent_t prelude_event_ = nullptr;
    cudaEvent_t merger_event_  = nullptr;
    cudaEvent_t compute_fence_ = nullptr;
    std::uint32_t next_upload_ = 0;
    std::size_t upload_bytes_  = 0;
};

// Encode every vision item of one request inside a single overlay window:
// evict the staging extent from the weight pool tail, stream the vision tower
// through it, land each item's merged embeddings in pinned host memory, and
// restore the evicted text weights before returning. The caller must hold the
// engine's exclusive GPU execution turn with no other work in flight.
inline std::vector<PinnedVisionResult>
encode_items_overlay(DeviceContext& device, const LoadedModelData& model,
                     qwen3_6::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                     std::size_t first_item, VisionOverlayWindowStats* stats) {
    using overlay_detail::OverlayClock;
    using overlay_detail::staging_align;
    if (!model.vision || !model.vision_overlay) {
        throw std::logic_error("overlay window requires vision weights and overlay assets");
    }
    const qwen3_6::VisionOverlayAssets& assets = *model.vision_overlay;
    EvictableWeightPool& pool                  = *assets.pool;
    if (plan.control == nullptr || plan.uses.empty()) {
        throw std::invalid_argument("overlay window has no vision items");
    }

    // The window is self-contained: weight staging, encode workspace, and the
    // item output all live in memory borrowed from the evicted weight tail, so
    // the resident workspace/transient reservations carry no vision terms.
    std::size_t workspace_need = 0;
    std::size_t output_need    = 0;
    for (std::size_t index = first_item; index < plan.control->items.size(); ++index) {
        const qwen3_6::VisionItemControl& control = plan.control->items[index];
        workspace_need =
            std::max(workspace_need, VisionContext::workspace_bytes(control));
        output_need = std::max(
            output_need, VisionContext::output_transient_bytes(control.merged_count));
    }

    const auto window_start = OverlayClock::now();
    std::size_t mapped      = 0;
    std::byte* staging      = pool.evict(assets.layout.staging_bytes +
                                             staging_align(output_need) +
                                             staging_align(workspace_need),
                                         &mapped);
    std::byte* lease_output    = staging + staging_align(assets.layout.staging_bytes);
    std::byte* lease_workspace = lease_output + staging_align(output_need);

    struct RestoreGuard {
        EvictableWeightPool& pool;
        cudaStream_t stream;
        ~RestoreGuard() {
            // On the failure path in-flight work may still reference the overlay
            // range; quiesce before remapping so restore stays safe.
            (void)cudaDeviceSynchronize();
            pool.restore(stream);
        }
    } restore_guard{pool, device.stream};

    std::vector<PinnedVisionResult> results;
    std::size_t staged_bytes = 0;
    {
        VisionWeightStream stream(device, assets, staging);
        const qwen3_6::VisionWeights window_view = stream.window_weights(*model.vision);
        const VisionContext context(device, window_view);
        results.reserve(plan.control->items.size());
        for (std::size_t skipped = 0; skipped < first_item; ++skipped) {
            results.push_back(PinnedVisionResult{});   // prefix-reused: never consumed
        }

        WorkspaceArena window_workspace(DeviceSpan{lease_workspace, workspace_need});
        for (std::size_t index = first_item; index < plan.control->items.size(); ++index) {
            const qwen3_6::VisionItemControl& control = plan.control->items[index];
            const qwen3_6::VisionItem& source         = prompt.vision_items.at(index);
            const std::size_t patch_offset =
                control.patch_begin * static_cast<std::size_t>(VisionScheduleConfig::patch_dim);
            const std::size_t patch_elements =
                control.patch_count * static_cast<std::size_t>(VisionScheduleConfig::patch_dim);
            if (source.patch_begin != control.patch_begin ||
                patch_offset > prompt.patches.size() ||
                patch_elements > prompt.patches.size() - patch_offset) {
                throw std::invalid_argument("overlay item patch range exceeds prepared payload");
            }
            Tensor output(lease_output, DType::BF16,
                          {VisionScheduleConfig::out_hidden,
                           static_cast<std::int32_t>(control.merged_count)});

            if (index != first_item) { stream.reset(device.stream); }
            context.encode(
                VisionItemView{
                    std::span<const float>(prompt.patches).subspan(patch_offset, patch_elements),
                    &control},
                output, window_workspace, &stream);
            window_workspace.reset();

            const std::size_t embedding_bytes = output.bytes();
            results.push_back(PinnedVisionResult{
                std::make_unique<PinnedHostBuffer>(embedding_bytes), embedding_bytes});
            CUDA_CHECK(cudaMemcpyAsync(results.back().buffer->data(), output.data, embedding_bytes,
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
        staged_bytes = stream.uploaded_bytes();
    }

    pool.restore(device.stream);   // the guard's later call becomes a no-op

    if (stats != nullptr) {
        stats->window_seconds  = std::chrono::duration<double>(OverlayClock::now() - window_start)
                                    .count();
        stats->evict_seconds   = pool.last_evict_seconds();
        stats->restore_seconds = pool.last_restore_seconds();
        stats->evicted_bytes   = mapped;
        stats->staged_bytes    = staged_bytes;
    }
    return results;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
