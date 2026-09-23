// Prompt-graft container validation against a small hybrid model geometry. The container is the
// phantom-kv format_version 1 safetensors file plus its JSON sidecar, written here byte for byte.
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/graft.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

namespace q = ninfer::models::qwen3_5;
using Json  = nlohmann::json;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Three Gated DeltaNet layers then one attention layer, the Qwen3.5 hybrid period.
q::TextConfig model() {
    q::TextConfig text;
    text.vocab_size        = 1000;
    text.num_hidden_layers = 4;
    text.layer_types       = {q::MixerKind::LinearAttention, q::MixerKind::LinearAttention,
                              q::MixerKind::LinearAttention, q::MixerKind::FullAttention};
    text.full_attention_layers   = 1;
    text.linear_attention_layers = 3;
    text.attention = q::AttentionConfig{.num_attention_heads = 4, .num_key_value_heads = 2,
                                        .head_dim = 8};
    text.gdn = q::GdnConfig{.linear_num_key_heads   = 2,
                            .linear_key_head_dim    = 4,
                            .linear_num_value_heads = 4,
                            .linear_value_head_dim  = 4,
                            .linear_conv_kernel_dim = 4};
    return text;
}

struct