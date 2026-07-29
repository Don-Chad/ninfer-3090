#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void target_verify(TextContext& card, State& state, const Tensor& ids, const Tensor& positions,
                   ops::GqaExecutionEnvelope envelope) {
    if (state.diagnostic_text_tap != nullptr) {
        card.diagnostic_target_verify(ids, positions, envelope, state.diagnostic_context,
                                      state.diagnostic_text_tap);
    } else {
        card.target_verify(ids, positions, envelope);
    }
}

} // namespace

void mtp_bridge_and_propose(State& state, const Tensor& next_token, const Tensor& previous_hidden,
                            std::int32_t position, std::span<const std::int32_t> rope_position,
                            bool build_proposal) {
    if (state.mtp_kv == nullptr) { throw std::logic_error("MTP bridge requires MTP storage"); }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.work.reset();
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                     state.prefill_hidden, state.prefill_chunk, state.text_kv_base, state.mtp_kv);
    configure_text_card(card, state);

    Tensor position_view = state.io.positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.device.stream);
    Tensor mtp_hidden         = state.io.mtp_ar_hidden;
    Tensor logits             = state.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.io.drafts.slice(0, 0, 1);
    Tensor rope_position_view = state.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view);
    if (!build_proposal) { return; }

    ops::set_i32_scalar(state.io.ar_pos, position + 1, state.device.stream);
    for (int i = 1; i < state.io.drafts.ne[0]; ++i) {
        Tensor previous_token = state.io.drafts.slice(0, i - 1, 1);
        Tensor next_draft     = state.io.drafts.slice(0, i, 1);
        Tensor next_hidden    = state.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.io.mtp_ar_hidden, state.io.ar_pos, envelope,
                                 next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.io.mtp_ar_hidden.data, next_hidden.data,
                                   state.io.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                   state.device.stream));
        ops::increment_i32_scalar(state.io.ar_pos, state.device.stream);
    }
}

void mtp_realign_and_propose(State& state, const Tensor& hidden, std::span<const TokenId> tokens,
                             std::uint32_t first_position, std::uint32_t proposal_k) {
    if (state.mtp_kv == nullptr) { throw std::logic_error("MTP realignment requires MTP storage"); }
    if (tokens.empty() || proposal_k == 0 ||
        proposal_k > static_cast<std::uint32_t>(state.io.drafts.ne[0])) {
        throw std::invalid_argument("MTP realignment has an invalid token or proposal window");
    }
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden ||
        hidden.ne[1] != static_cast<std::int32_t>(tokens.size()) || hidden.ne[2] != 1 ||
        hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("MTP realignment hidden history has an invalid shape");
    }
    if (tokens.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() - first_position)) {
        throw std::overflow_error("MTP realignment position overflows uint32");
    }

    state.work.reset();
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                     state.prefill_hidden, state.prefill_chunk, state.text_kv_base, state.mtp_kv);
    configure_text_card(card, state);

    const std::size_t chunk_capacity =
        std::min<std::size_t>(state.io.verify_ids.ne[0], state.prefill_chunk);
    std::vector<std::int32_t> positions(chunk_capacity);
    std::size_t offset = 0;
    while (offset < tokens.size()) {
        const std::size_t count = std::min(chunk_capacity, tokens.size() - offset);
        auto ids                = state.io.verify_ids.slice(0, 0, static_cast<int>(count));
        auto pos                = state.io.positions.slice(0, 0, static_cast<int>(count));
        CUDA_CHECK(cudaMemcpyAsync(ids.data, tokens.data() + offset, count * sizeof(TokenId),
                                   cudaMemcpyHostToDevice, state.device.stream));
        for (std::size_t i = 0; i < count; ++i) {
            positions[i] = static_cast<std::int32_t>(first_position + offset + i);
        }
        CUDA_CHECK(cudaMemcpyAsync(pos.data, positions.data(), count * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, state.device.stream));

        auto source_hidden     = hidden.slice(1, static_cast<int>(offset), static_cast<int>(count));
        auto mtp_hidden        = state.prefill_hidden.slice(1, 0, static_cast<int>(count));
        const bool final_chunk = offset + count == tokens.size();
        auto logits            = state.io.logits.slice(1, 0, 1);
        auto draft0            = state.io.drafts.slice(0, 0, 1);
        const std::uint32_t visible = first_position + static_cast<std::uint32_t>(offset + count);
        card.mtp_forward_batch(ids, source_hidden, pos, {visible, visible}, mtp_hidden,
                               final_chunk ? static_cast<int>(count) - 1 : -1,
                               final_chunk ? &logits : nullptr, final_chunk ? &draft0 : nullptr);
        offset += count;
    }

    const std::uint32_t frontier = first_position + static_cast<std::uint32_t>(tokens.size());
    ops::set_i32_scalar(state.io.ar_pos, static_cast<std::int32_t>(frontier), state.device.stream);
    auto logits = state.io.logits.slice(1, 0, 1);
    for (std::uint32_t i = 1; i < proposal_k; ++i) {
        auto previous_token         = state.io.drafts.slice(0, static_cast<int>(i) - 1, 1);
        auto next_draft             = state.io.drafts.slice(0, static_cast<int>(i), 1);
        auto next_hidden            = state.prefill_hidden.slice(1, static_cast<int>(i), 1);
        const std::uint32_t visible = frontier + i;
        card.mtp_forward_ar_step(previous_token, state.io.mtp_ar_hidden, state.io.ar_pos,
                                 {visible, visible}, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.io.mtp_ar_hidden.data, next_hidden.data,
                                   state.io.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                   state.device.stream));
        ops::increment_i32_scalar(state.io.ar_pos, state.device.stream);
    }
}

template <class Body>
void run_prepared(State& state, DecodeGraph* graph, Body&& body) {
    if (graph != nullptr) {
        if (!graph->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        graph->launch(state.device.stream);
    } else {
        body();
    }
}

template <class Body>
void warm_capture(State& state, DecodeGraph& graph, const GraphPrepare& prepare, Body&& body) {
    prepare();
    state.device.synchronize();
    body();
    state.device.synchronize();
    prepare();
    state.device.synchronize();
    graph.capture(state.device.stream, body);
    prepare();
    state.device.synchronize();
    graph.launch(state.device.stream);
    state.device.synchronize();
}

auto ordinary_body(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope) {
    auto record = [&state, align_mtp, envelope] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                         align_mtp ? state.mtp_kv : nullptr);
        configure_text_card(card, state);

        Tensor verify_id = state.io.verify_ids.slice(0, 0, 1);
        Tensor position  = state.io.positions.slice(0, 0, 1);
        ops::assign_i32_scalar(state.io.token, verify_id, state.device.stream);
        ops::assign_i32_scalar(state.io.pos, position, state.device.stream);
        target_verify(card, state, verify_id, position, envelope);

        Tensor logits = state.io.logits.slice(1, 0, 1);
        ops::sample(logits, state.io.token, TextConfig::token_domain, state.sampling,
                    static_cast<const std::int32_t*>(state.io.pos.data), ops::kSamplePurposeDecode,
                    state.work, state.device.stream);

        if (align_mtp) {
            Tensor hidden     = state.io.verify_hidden.slice(1, 0, 1);
            Tensor mtp_hidden = state.io.mtp_ar_hidden;
            card.mtp_forward_batch(state.io.token, hidden, position, envelope, mtp_hidden, -1,
                                   nullptr, nullptr);
        }

        ops::increment_i32_scalar(state.io.pos, state.device.stream);
        ops::increment_i32_scalar(state.io.rope_pos, state.device.stream);
        ops::set_i32_scalar(state.io.gdn_initial_slot, 0, state.device.stream);
        if (state.mtp_kv != nullptr) {
            Tensor fallback_steps = state.io.stats.slice(0, 3, 1);
            ops::increment_i64_scalar(fallback_steps, state.device.stream);
        }
    };
    return record;
}

auto mtp_body(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes) {
    if (state.mtp_kv == nullptr || k == 0 ||
        k != static_cast<std::uint32_t>(state.io.drafts.ne[0])) {
        throw std::logic_error("MTP round storage does not match its configured window");
    }

    auto record = [&state, k, envelopes] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                         state.mtp_kv);
        configure_text_card(card, state);

        ops::mtp_prepare_verify_inputs(state.io.token, state.io.drafts, state.io.pos,
                                       state.io.window_base, state.io.verify_ids,
                                       state.io.positions, state.device.stream);
        target_verify(card, state, state.io.verify_ids, state.io.positions,
                      envelopes.target_verify);
        ops::mtp_accept_tokens(state.io.target_tokens, state.io.logits, state.io.drafts,
                               state.io.pos, state.io.token, state.io.sampled_out,
                               state.io.num_sampled, state.io.accepted, state.io.ar_pos,
                               state.io.stats, TextConfig::token_domain, state.sampling, state.work,
                               state.device.stream);
        ops::assign_i32_scalar(state.io.accepted, state.io.gdn_initial_slot, state.device.stream);

        const int columns = static_cast<int>(k) + 1;
        ops::mtp_prepare_shifted_ids(state.io.verify_ids, state.io.token, state.io.accepted,
                                     state.io.shifted_ids, state.device.stream);
        Tensor mtp_hidden = state.prefill_hidden.slice(1, 0, columns);
        card.mtp_forward_batch(state.io.shifted_ids, state.io.verify_hidden, state.io.positions,
                               envelopes.batch, mtp_hidden, -1, nullptr, nullptr);

        Tensor logits = state.io.logits.slice(1, 0, 1);
        Tensor draft0 = state.io.drafts.slice(0, 0, 1);
        card.mtp_sample_from_hidden_row(mtp_hidden, state.io.accepted, state.io.mtp_ar_hidden,
                                        logits, draft0);
        for (std::uint32_t i = 1; i < k; ++i) {
            Tensor previous_token = state.io.drafts.slice(0, static_cast<int>(i) - 1, 1);
            Tensor next_token     = state.io.drafts.slice(0, static_cast<int>(i), 1);
            Tensor next_hidden    = state.prefill_hidden.slice(1, static_cast<int>(i), 1);
            card.mtp_forward_ar_step(previous_token, state.io.mtp_ar_hidden, state.io.ar_pos,
                                     envelopes.ar[i - 1], next_hidden, logits, next_token);
            CUDA_CHECK(cudaMemcpyAsync(state.io.mtp_ar_hidden.data, next_hidden.data,
                                       state.io.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                       state.device.stream));
            ops::increment_i32_scalar(state.io.ar_pos, state.device.stream);
        }
    };
    return record;
}

auto prompt_lookup_body(State& state, std::uint32_t k, ops::GqaExecutionEnvelope envelope) {
    if (k == 0 || k != static_cast<std::uint32_t>(state.io.drafts.ne[0])) {
        throw std::logic_error("prompt lookup storage does not match its configured window");
    }
    return [&state, envelope] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_hidden, state.prefill_chunk, state.text_kv_base, nullptr);
        configure_text_card(card, state);
        ops::mtp_prepare_verify_inputs(state.io.token, state.io.drafts, state.io.pos,
                                       state.io.window_base, state.io.verify_ids,
                                       state.io.positions, state.device.stream);
        target_verify(card, state, state.io.verify_ids, state.io.positions, envelope);
        ops::mtp_accept_tokens(state.io.target_tokens, state.io.logits, state.io.drafts,
                               state.io.pos, state.io.token, state.io.sampled_out,
                               state.io.num_sampled, state.io.accepted, state.io.ar_pos,
                               state.io.stats, TextConfig::token_domain, state.sampling, state.work,
                               state.device.stream);
        ops::assign_i32_scalar(state.io.accepted, state.io.gdn_initial_slot, state.device.stream);
    };
}

void warm_capture_ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                                 const GraphPrepare& prepare, DecodeGraph& graph) {
    auto body = ordinary_body(state, align_mtp, envelope);
    warm_capture(state, graph, prepare, body);
}

void ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                    DecodeGraph* graph) {
    auto body = ordinary_body(state, align_mtp, envelope);
    run_prepared(state, graph, body);
}

void warm_capture_mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes,
                            const GraphPrepare& prepare, DecodeGraph& graph) {
    auto body = mtp_body(state, k, envelopes);
    warm_capture(state, graph, prepare, body);
}

void mtp_round(State& state, std::uint32_t k, MtpGqaEnvelopes envelopes, DecodeGraph* graph) {
    auto body = mtp_body(state, k, envelopes);
    run_prepared(state, graph, body);
}

void warm_capture_prompt_lookup_round(State& state, std::uint32_t k,
                                      ops::GqaExecutionEnvelope envelope,
                                      const GraphPrepare& prepare, DecodeGraph& graph) {
    auto body = prompt_lookup_body(state, k, envelope);
    warm_capture(state, graph, prepare, body);
}

void prompt_lookup_round(State& state, std::uint32_t k, ops::GqaExecutionEnvelope envelope,
                         DecodeGraph* graph) {
    auto body = prompt_lookup_body(state, k, envelope);
    run_prepared(state, graph, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
