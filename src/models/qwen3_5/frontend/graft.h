#pragma once

#include "models/qwen3_5/config.h"

#include <ninfer/types.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

// phantom-kv prompt grafts (format_version 1).
//
// A graft is a hidden conversation prefix that sits in front of a request's rendered prompt.
//
// prefill_kv grafts carry replayable token ids: NInfer replays them through its own prefill,
// which is exact for NInfer's weights and KV storage, covers every layer kind including MTP,
// and lets the shared-prefix cache hold the replayed state so each graft is prefilled once.
//
// direct_kv and softprompt_kv grafts carry trained cache tensors with no replayable token ids.
// Their K/V and GDN state are injected directly into a synthesized shared-prefix entry at
// startup. MTP KV is zero-filled (no HF MTP implementation exists to produce it).
//
// Loading validates the whole container against the resident model -- every tensor shape,
// the hybrid layer layout, and the payload digest -- so a graft built for another model or
// damaged on disk is refused at startup instead of silently steering a different network.

namespace ninfer::models::qwen3_5 {

enum class GraftKind : std::uint8_t {
    PrefillKV,   // replay token ids through ninfer's own prefill
    DirectKV,    // inject trained K/V/conv/rec directly
    SoftpromptKV // inject trained embedding-derived state directly
};

struct GraftTensorRegion {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct GraftTensors {
    std::vector<std::uint8_t> payload;
    GraftTensorRegion k;
    GraftTensorRegion v;
    GraftTensorRegion conv;
    GraftTensorRegion rec;
    std::uint64_t n_slots         = 0;
    std::uint64_t n_attn_layers   = 0;
    std::uint64_t n_linear_layers = 0;
    std::uint64_t n_kv_heads      = 0;
    std::uint64_t head_dim        = 0;
    std::uint64_t conv_channels   = 0;
    std::uint64_t conv_width      = 0; // graft's kernel dim (e.g. 4)
    std::uint64_t value_heads     = 0;
    std::uint64_t key_head_dim    = 0;
    std::uint64_t value_head_dim  = 0;

    [[nodiscard]] const std::uint8_t* k_data() const noexcept { return payload.data() + k.begin; }
    [[nodiscard]] const std::uint8_t* v_data() const noexcept { return payload.data() + v.begin; }
    [[nodiscard]] const std::uint8_t* conv_data() const noexcept {
        return payload.data() + conv.begin;
    }
    [[nodiscard]] const std::uint8_t* rec_data() const noexcept {
        return payload.data() + rec.begin;
    }
};

struct PromptGraft {
    std::string name;
    GraftKind kind = GraftKind::PrefillKV;
    std::vector<TokenId> tokens;              // replay ids (prefill_kv only)
    std::optional<GraftTensors> tensors;      // tensor data (direct_kv / softprompt_kv)
    std::uint32_t n_slots = 0;                // slot count (all kinds)
    std::string payload_sha256;
};

// Reads `source.path` and its `.json` sidecar. Throws std::invalid_argument naming the graft and
// the first inconsistency found.
[[nodiscard]] PromptGraft load_prompt_graft(const GraftSource& source, const TextConfig& text);

[[nodiscard]] std::vector<PromptGraft> load_prompt_grafts(const std::vector<GraftSource>& sources,
                                                         const TextConfig& text);

} // namespace ninfer::models::qwen3_5
