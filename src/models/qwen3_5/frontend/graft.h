#pragma once

#include "models/qwen3_5/config.h"

#include <ninfer/types.h>

#include <string>
#include <vector>

// phantom-kv prompt grafts (format_version 1, kind "prefill_kv").
//
// A graft is a hidden conversation prefix that sits in front of a request's rendered prompt. The
// container also carries the cache state an HF reference computed for that prefix (attention K/V,
// Gated DeltaNet conv windows and recurrent states), but NInfer does not inject those tensors: it
// replays the graft's own token ids through its own prefill. That is exact for NInfer's weights and
// KV storage, covers every layer kind including the MTP draft layer, and lets the shared-prefix
// cache hold the replayed state so each graft is prefilled once rather than per request.
//
// Loading still validates the whole container against the resident model -- every tensor shape,
// the hybrid layer layout, and the payload digest -- so a graft built for another model or damaged
// on disk is refused at startup instead of silently steering a different network.

namespace ninfer::models::qwen3_5 {

struct PromptGraft {
    std::string name;
    std::vector<TokenId> tokens;
    std::string payload_sha256; // hex, over the safetensors tensor bytes
};

// Reads `source.path` and its `.json` sidecar. Throws std::invalid_argument naming the graft and
// the first inconsistency found.
[[nodiscard]] PromptGraft load_prompt_graft(const GraftSource& source, const TextConfig& text);

[[nodiscard]] std::vector<PromptGraft> load_prompt_grafts(const std::vector<GraftSource>& sources,
                                                         const TextConfig& text);

} // namespace ninfer::models::qwen3_5
