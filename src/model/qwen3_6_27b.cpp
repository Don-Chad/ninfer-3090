#include "qus/model/model.h"

#include "qus/core/weight_store_parser.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::model {
namespace {

constexpr ModuleKind kText = ModuleKind::TextCore;

std::uint32_t sk(SourceKind kind) { return static_cast<std::uint32_t>(kind); }

std::string source_label(const char* field, SourceKind kind, std::uint32_t layer) {
    return std::string(field) + " source_kind=" + std::to_string(sk(kind)) +
           " layer=" + (layer == kQ5090NoLayer ? std::string("NO_LAYER") : std::to_string(layer));
}

const Weight* require_weight(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                             const char* field) {
    const Weight* weight = store.qweight(kText, sk(kind), layer);
    if (weight == nullptr) {
        throw std::runtime_error("missing q5090 weight: " + source_label(field, kind, layer));
    }
    return weight;
}

const Tensor* require_tensor(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                             const char* field) {
    const Tensor* tensor = store.tensor(kText, sk(kind), layer);
    if (tensor == nullptr) {
        throw std::runtime_error("missing q5090 tensor: " + source_label(field, kind, layer));
    }
    return tensor;
}

Weight bind_dense_weight(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                         const char* field) {
    const Tensor* tensor = require_tensor(store, kind, layer, field);
    Weight weight        = weight_from_dense(*tensor);
    weight.module        = kText;
    weight.source_kind   = sk(kind);
    weight.source_layer  = layer;
    return weight;
}

Tensor bind_conv1d_view(const WeightStore& store, SourceKind kind, std::uint32_t layer,
                        const char* field) {
    const Tensor* tensor = require_tensor(store, kind, layer, field);
    if (tensor->ne[1] == 1 && tensor->ne[2] == kCfg.gdn_conv_k && tensor->is_contiguous()) {
        return tensor->view({tensor->ne[0], kCfg.gdn_conv_k});
    }
    if (tensor->ne[1] == kCfg.gdn_conv_k && tensor->ne[2] == 1) { return *tensor; }
    throw std::runtime_error("unexpected q5090 conv1d shape: " + source_label(field, kind, layer));
}

MlpW bind_mlp(const WeightStore& store, std::uint32_t layer) {
    return MlpW{require_weight(store, SourceKind::MlpGate, layer, "mlp.gate"),
                require_weight(store, SourceKind::MlpUp, layer, "mlp.up"),
                require_weight(store, SourceKind::MlpDown, layer, "mlp.down")};
}

} // namespace

Qwen3_6_27B::Qwen3_6_27B(DeviceContext& ctx, WeightStore& weights, WorkspaceArena& work,
                         KVCache& kv, GdnState& state, StepState& io)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), state_(state), io_(io) {
    bind();
}

void Qwen3_6_27B::bind() {
    embed_      = require_weight(weights_, SourceKind::Embed, kQ5090NoLayer, "embed");
    final_norm_ = require_tensor(weights_, SourceKind::FinalNorm, kQ5090NoLayer, "final_norm");
    lm_head_    = require_weight(weights_, SourceKind::LmHead, kQ5090NoLayer, "lm_head");

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        const auto source_layer = static_cast<std::uint32_t>(layer);
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm  = require_tensor(weights_, SourceKind::InputLayernorm, source_layer,
                                             "full.input_norm");
            out.q_proj = require_weight(weights_, SourceKind::AttnQ, source_layer, "full.q_proj");
            out.gate_proj =
                require_weight(weights_, SourceKind::AttnGate, source_layer, "full.gate_proj");
            out.k_proj = require_weight(weights_, SourceKind::AttnK, source_layer, "full.k_proj");
            out.v_proj = require_weight(weights_, SourceKind::AttnV, source_layer, "full.v_proj");
            out.o_proj = require_weight(weights_, SourceKind::AttnO, source_layer, "full.o_proj");
            out.q_norm =
                require_tensor(weights_, SourceKind::AttnQNorm, source_layer, "full.q_norm");
            out.k_norm =
                require_tensor(weights_, SourceKind::AttnKNorm, source_layer, "full.k_norm");
            out.post_attn_norm = require_tensor(weights_, SourceKind::PostAttnLayernorm,
                                                source_layer, "full.post_attn_norm");
            out.mlp            = bind_mlp(weights_, source_layer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            out.input_norm = require_tensor(weights_, SourceKind::InputLayernorm, source_layer,
                                            "gdn.input_norm");
            out.in_q = require_weight(weights_, SourceKind::GdnInProjQ, source_layer, "gdn.in_q");
            out.in_k = require_weight(weights_, SourceKind::GdnInProjK, source_layer, "gdn.in_k");
            out.in_v = require_weight(weights_, SourceKind::GdnInProjV, source_layer, "gdn.in_v");
            out.in_z = require_weight(weights_, SourceKind::GdnInProjZ, source_layer, "gdn.in_z");
            gdn_in_a_[gidx] =
                bind_dense_weight(weights_, SourceKind::GdnInProjA, source_layer, "gdn.in_a");
            gdn_in_b_[gidx] =
                bind_dense_weight(weights_, SourceKind::GdnInProjB, source_layer, "gdn.in_b");
            out.in_a = &gdn_in_a_[gidx];
            out.in_b = &gdn_in_b_[gidx];
            gdn_conv1d_views_[gidx] =
                bind_conv1d_view(weights_, SourceKind::GdnConv1d, source_layer, "gdn.conv1d");
            out.conv1d = &gdn_conv1d_views_[gidx];
            out.a_log  = require_tensor(weights_, SourceKind::GdnALog, source_layer, "gdn.a_log");
            out.dt_bias =
                require_tensor(weights_, SourceKind::GdnDtBias, source_layer, "gdn.dt_bias");
            out.gdn_norm =
                require_tensor(weights_, SourceKind::GdnNorm, source_layer, "gdn.gdn_norm");
            out.out_proj =
                require_weight(weights_, SourceKind::GdnOutProj, source_layer, "gdn.out_proj");
            out.post_attn_norm = require_tensor(weights_, SourceKind::PostAttnLayernorm,
                                                source_layer, "gdn.post_attn_norm");
            out.mlp            = bind_mlp(weights_, source_layer);
        }
    }
}

} // namespace qus::model
