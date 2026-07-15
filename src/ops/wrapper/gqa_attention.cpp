// ninfer::ops - gqa_attention wrapper: public api validation and phase dispatch.
#include "ninfer/ops/gqa_attention.h"

#include "core/layout.h"

#include "ops/launcher/gqa_attention.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr float kExpectedScale  = 0.0625f;

std::int32_t kv_heads_for_q_heads(std::int32_t q_heads, const char* op) {
    if (q_heads == 24) { return 4; }
    if (q_heads == 16) { return 2; }
    throw std::invalid_argument(std::string(op) + ": unsupported Q/KV head geometry");
}

void require_kv_heads(std::int32_t kv_heads, const char* op) {
    if (kv_heads != 4 && kv_heads != 2) {
        throw std::invalid_argument(std::string(op) + ": unsupported KV head geometry");
    }
}

std::int32_t checked_i32(std::uint32_t value, const char* op, const char* name) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": " + name + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void validate_cache(KVCache& kv, int layer, std::int32_t kv_heads, const char* op) {
    if (layer < 0 || static_cast<std::uint32_t>(layer) >= kv.layer_count()) {
        throw std::invalid_argument(std::string(op) + ": layer out of range");
    }
    if ((kv.dtype != DType::BF16 && kv.dtype != DType::I8) || kv.num_kv_heads != kv_heads ||
        kv.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache geometry or dtype");
    }
    if (kv.dtype == DType::BF16 && kv.quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KVCache must not have quant_group");
    }
    if (kv.dtype == DType::I8 && kv.quant_group != kKvQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KVCache must use quant_group 64");
    }
    if (kv.padded_context < kv.max_context) {
        throw std::invalid_argument(std::string(op) + ": KVCache padded_context is too small");
    }
    if (kv.k.size() != kv.v.size() || kv.k.size() != kv.layer_count()) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache layer vectors");
    }

    const Tensor& cache_k            = kv.k[static_cast<std::uint32_t>(layer)];
    const Tensor& cache_v            = kv.v[static_cast<std::uint32_t>(layer)];
    const DType expected_cache_dtype = kv.dtype == DType::I8 ? DType::I8 : DType::BF16;
    if (cache_k.dtype != expected_cache_dtype || cache_v.dtype != expected_cache_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache code tensor dtype");
    }
    const std::int32_t padded_context = checked_i32(kv.padded_context, op, "padded_context");
    if (cache_k.ne[0] != kHeadDim || cache_k.ne[1] != padded_context || cache_k.ne[2] != kv_heads ||
        cache_k.ne[3] != 1 || cache_v.ne[0] != kHeadDim || cache_v.ne[1] != padded_context ||
        cache_v.ne[2] != kv_heads || cache_v.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache tensor shape");
    }
    require_contiguous_nonnull(cache_k, op, "cache k");
    require_contiguous_nonnull(cache_v, op, "cache v");

    if (kv.dtype == DType::BF16) {
        if (!kv.k_scale.empty() || !kv.v_scale.empty()) {
            throw std::invalid_argument(std::string(op) + ": BF16 KVCache must not have scales");
        }
        return;
    }

    if (kv.k_scale.size() != kv.layer_count() || kv.v_scale.size() != kv.layer_count()) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache scale vectors");
    }
    const Tensor& cache_k_scale         = kv.k_scale[static_cast<std::uint32_t>(layer)];
    const Tensor& cache_v_scale         = kv.v_scale[static_cast<std::uint32_t>(layer)];
    constexpr std::int32_t kScaleGroups = kHeadDim / kKvQuantGroup;
    if (cache_k_scale.dtype != DType::FP16 || cache_v_scale.dtype != DType::FP16 ||
        cache_k_scale.ne[0] != kScaleGroups || cache_k_scale.ne[1] != padded_context ||
        cache_k_scale.ne[2] != kv_heads || cache_k_scale.ne[3] != 1 ||
        cache_v_scale.ne[0] != kScaleGroups || cache_v_scale.ne[1] != padded_context ||
        cache_v_scale.ne[2] != kv_heads || cache_v_scale.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid KVCache scale tensor shape");
    }
    require_contiguous_nonnull(cache_k_scale, op, "cache k scale");
    require_contiguous_nonnull(cache_v_scale, op, "cache v scale");
}

} // namespace

std::size_t gqa_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens) {
    if (tokens <= 0 || !detail::gqa_attention_uses_small_t(tokens)) { return 0; }
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, "gqa_attention_workspace_bytes");
    const std::int32_t splits   = detail::gqa_attention_decode_splits(q_heads, kv_heads);
    const Tensor acc(nullptr, DType::BF16, {kHeadDim, q_heads, tokens, splits});
    const Tensor stat(nullptr, DType::FP32, {q_heads, tokens, splits});
    LayoutBuilder layout;
    (void)layout.add(acc.bytes(), 256, "GQA partial accumulator");
    (void)layout.add(stat.bytes(), 256, "GQA partial max");
    (void)layout.add(stat.bytes(), 256, "GQA partial sum");
    return layout.finish(256, "GQA workspace");
}

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                   cudaStream_t stream) {
    constexpr const char* op = "gqa_attention";
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: q/k/v/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_attention: positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument("gqa_attention: scale must be 1/sqrt(256)");
    }

    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_attention: T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(k, kHeadDim, kv_heads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kv_heads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");

    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    validate_cache(kv, layer, kv_heads, op);

    const std::uint32_t token_count = static_cast<std::uint32_t>(tokens);
    if (token_count > kv.max_context) {
        throw std::invalid_argument("gqa_attention: T exceeds KVCache max_context");
    }
    if (token_count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("gqa_attention: T exceeds int32");
    }

    auto scratch_scope = ws.scope();
    Tensor partial_acc;
    Tensor partial_m;
    Tensor partial_l;
    Tensor* partial_acc_ptr = nullptr;
    Tensor* partial_m_ptr   = nullptr;
    Tensor* partial_l_ptr   = nullptr;
    if (detail::gqa_attention_uses_small_t(tokens)) {
        const std::int32_t splits = detail::gqa_attention_decode_splits(q_heads, kv_heads);
        partial_acc               = ws.alloc(DType::BF16, {kHeadDim, q_heads, tokens, splits});
        partial_m                 = ws.alloc(DType::FP32, {q_heads, tokens, splits});
        partial_l                 = ws.alloc(DType::FP32, {q_heads, tokens, splits});
        partial_acc_ptr           = &partial_acc;
        partial_m_ptr             = &partial_m;
        partial_l_ptr             = &partial_l;
    }

    detail::gqa_attention_launch(q, k, v, positions, scale, kv, layer, partial_acc_ptr,
                                 partial_m_ptr, partial_l_ptr, out, stream);
}

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, KVCache& kv,
                   int layer, cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_append";
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_kv_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_kv_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    require_kv_heads(kv_heads, op);
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_kv_append: T must be positive"); }
    require_shape(k, kHeadDim, kv_heads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kv_heads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    validate_cache(kv, layer, kv_heads, op);
    if (static_cast<std::uint32_t>(tokens) > kv.max_context) {
        throw std::invalid_argument("gqa_kv_append: T exceeds KVCache max_context");
    }
    detail::gqa_kv_append_launch(k, v, positions, kv, layer, stream);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale, KVCache& kv,
                          int layer, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gqa_attention_cached";
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention_cached: q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_attention_cached: positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument("gqa_attention_cached: scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_attention_cached: T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    validate_cache(kv, layer, kv_heads, op);
    if (static_cast<std::uint32_t>(tokens) > kv.max_context) {
        throw std::invalid_argument("gqa_attention_cached: T exceeds KVCache max_context");
    }
    detail::gqa_attention_prompt_attention_launch(q, positions, scale, kv, layer, out, stream);
}

} // namespace ninfer::ops
