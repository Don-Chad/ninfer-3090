#pragma once

#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/stage_link.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// What a forward pass across pipeline stages needs beyond one device's state. The Program owns one
// when the model is split over several devices and hands a pointer to each TextContext; it is null on
// one device, where none of this exists.
//
// A stage owns its layers whole, so each stage reads its own device's KV planes and Linear Attention
// state. The embedding, head and round state stay on rank 0: the residual stream leaves rank 0 after
// its layers, crosses each later stage in turn, and the last stage sends it back.
struct StageRuntime {
    // The Linear Attention state pool of each shard, and the global index of the first layer in it.
    std::vector<LinearAttentionStatePool*> state;
    std::vector<std::uint32_t> state_first_layer;
    // The ReplaySSM record storage of each shard, in the same order; empty without speculative
    // decoding. A stage records only its own layers, on its own device.
    std::vector<const GdnReplayRecords*> replay;

    // forward[s] carries the residual from stage s to stage s+1; `back` from the last stage to rank
    // 0. `control[s-1]` carries the small position, row and slot tensors the layers read from rank 0
    // to stage s, whose device cannot follow a pointer into rank 0's memory.
    std::vector<StageLink> forward;
    std::optional<StageLink> back;
    std::vector<StageLink> control;
    // features[s-1] carries the hidden states of the DFlash feature layers stage s owns back to rank
    // 0, where the draft reads them. Present only for a masked-draft Program whose feature layers
    // reach past stage 0; a stage without such a layer has no link.
    std::vector<std::optional<StageLink>> features;

    // One fence per rank. A CUDA graph launched on rank 0's stream also runs the other stages'
    // layers, so before the launch rank 0's stream waits for work already queued on every other
    // rank's stream (block-table publishes, state copies), and after it every other rank's stream
    // waits for the graph. Eager execution orders itself through the links and needs neither.
    std::vector<CudaCompletionEvent> rank_fences;

    // Alternates per forward pass so consecutive passes use different slots of every link.
    std::uint32_t next_slot = 0;

    // Rank 0's `entry` stream waits for everything already queued on the other ranks' streams.
    void join_into(const DeviceContext& device, cudaStream_t entry) {
        for (std::size_t rank = 1; rank < rank_fences.size(); ++rank) {
            rank_fences[rank].record(device.rank(rank).stream);
            rank_fences[rank].wait(entry);
        }
    }
    // Every other rank's stream waits for everything queued on rank 0's `entry` stream so far.
    void release_from(const DeviceContext& device, cudaStream_t entry) {
        if (rank_fences.size() < 2) { return; }
        rank_fences[0].record(entry);
        for (std::size_t rank = 1; rank < rank_fences.size(); ++rank) {
            rank_fences[0].wait(device.rank(rank).stream);
        }
    }
};

} // namespace ninfer::models::qwen3_5::execution
