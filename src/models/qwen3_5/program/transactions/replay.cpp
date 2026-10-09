#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/execution_context.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

PreparedPromptData replay_prompt(const RequestBasePlanImpl& base, const SequenceState& sequence) {
    if (!base.prompt || sequence.ledger.size() < base.prompt->token_ids.size()) {
        throw std::logic_error("replay ledger does not retain its prepared prompt");
    }
    const PreparedPromptData& source = *base.prompt;
    const std::size_t prompt_tokens  = source.token_ids.size();
    const std::size_t ledger_tokens  = sequence.ledger.size();
    if (source.token_types.size() != prompt_tokens ||
        source.positions.size() != 3ULL * prompt_tokens ||
        !std::equal(source.token_ids.begin(), source.token_ids.end(), sequence.ledger.begin())) {
        throw std::logic_error("replay input does not match the retained prompt");
    }

    PreparedPromptData out = source;
    out.token_ids          = sequence.ledger;
    out.token_types.resize(ledger_tokens, 0);
    out.positions.resize(3ULL * ledger_tokens);
    out.rope_delta = sequence.rope_delta;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        std::copy_n(source.positions.begin() + axis * prompt_tokens, prompt_tokens,
                    out.positions.begin() + axis * ledger_tokens);
        for (std::size_t token = prompt_tokens; token < ledger_tokens; ++token) {
            const std::int64_t position = static_cast<std::int64_t>(token) + sequence.rope_delta;
            if (position < std::numeric_limits<std::int32_t>::min() ||
                position > std::numeric_limits<std::int32_t>::max()) {
                throw std::overflow_error("replay RoPE position exceeds int32");
            }
            out.positions[axis * ledger_tokens + token] = static_cast<std::int32_t>(position);
        }
    }
    const auto splits = sequence.prefix_identity.execution_frontiers();
    out.identity.rewrite_execution_frontiers.assign(splits.begin(), splits.end());
    return out;
}

} // namespace

void ProgramImpl::install_resume_sampling(SequenceState& sequence, RequestControl& request) {
    if (!request.base || sequence.lane >= max_concurrency ||
        sequence.ledger.size() < request.base->summary.prompt_tokens) {
        throw std::logic_error("sampling restore has no complete request ledger");
    }
    // The configuration and occurrence storage are lane bindings. Timings and speculative
    // statistics belong to the original request and must survive every rebinding.
    request.sampling_host = request.base->sampling;
    const bool penalties  = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    Tensor counts =
        token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
            .view({execution::dimension(parameters.model.resources().public_token_count)});
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;

    try {
        if (penalties) {
            CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
            const auto generated = std::span<const TokenId>(sequence.ledger)
                                       .subspan(request.base->summary.prompt_tokens);
            validate_licensed_tokens(generated);
            // Reuse the fixed workspace in bounded chunks, including committed control tokens.
            // Replaying those tokens later never touches occurrence counts again.
            work.reset();
            if (!generated.empty()) {
                const auto chunk = static_cast<std::int32_t>(
                    std::min<std::size_t>(prefill_chunk, generated.size()));
                Tensor ids = work.alloc(DType::I32, {chunk});
                for (std::size_t begin = 0; begin < generated.size(); begin += chunk) {
                    const auto count = static_cast<std::int32_t>(
                        std::min<std::size_t>(chunk, generated.size() - begin));
                    Tensor current = ids.slice(0, 0, count);
                    CUDA_CHECK(cudaMemcpyAsync(current.data, generated.data() + begin,
                                               current.bytes(), cudaMemcpyHostToDevice,
                                               device.stream));
                    ops::increment_token_counts(current, counts, device.stream);
                }
            }
        }
        Tensor config = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
        CUDA_CHECK(cudaMemcpyAsync(config.data, &request.sampling_host,
                                   sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                                   device.stream));
        // Later workspace users and sampling execute on this same ordered compute stream.
        work.reset();
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        throw;
    }
}

ReplayProgress ProgramImpl::advance_replay(SequenceHandle handle,
                                           runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (!valid_sequence(handle) || pending_transaction_) {
        throw std::logic_error("replay sequence capability is unavailable");
    }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    SequenceState& sequence  = active_sequence(lane);
    RequestControl& request  = requests[lane];
    if (request.lifecycle != Lifecycle::Replaying || !request.base || !request.base->prompt ||
        request.replay_cursor > request.replay_target || request.replay_target > capacity ||
        request.replay_target > sequence.ledger.size() ||
        (request.resume_lifecycle != Lifecycle::Active &&
         request.resume_lifecycle != Lifecycle::Prefilling) ||
        (request.resume_lifecycle == Lifecycle::Active &&
         sequence.ledger.size() != static_cast<std::size_t>(request.replay_target) + 1U)) {
        throw std::logic_error("replay does not describe a committed recovery frontier");
    }
    if (request.capture_pending) {
        throw std::logic_error("replay cannot advance while a capture is pending");
    }
    require_unit(lane, ExecutionUnitKind::Replay);
    const double prior_vision_seconds =
        request.replay && request.replay->vision ? request.replay->vision->elapsed_seconds() : 0.0;
    const auto record_vision_time = [&] {
        if (request.replay && request.replay->vision) {
            request.timings.vision_seconds +=
                request.replay->vision->elapsed_seconds() - prior_vision_seconds;
        }
    };
    const auto complete_replay = [&] {
        sequence.mtp_draft_count = 0;
        request.lifecycle        = request.resume_lifecycle;
        request.replay.reset();
        initialize_captures(lane, request.replay_target,
                            request.lifecycle == Lifecycle::Prefilling
                                ? request.base->summary.prompt_tokens
                                : request.replay_target);
    };

    try {
        const bool bridge_mtp = speculative_backend == SpeculativeBackend::Mtp &&
                                request.replay_cursor != 0 &&
                                sequence.mtp_kv_valid + 1U == request.replay_cursor &&
                                request.replay_cursor < sequence.ledger.size();
        if (request.replay_cursor == request.replay_target && !bridge_mtp) {
            complete_replay();
            settle_unit(lane);
            return ReplayProgress{.complete = true, .timing = timing.finish()};
        }
        const UnitDemand permit = *request.permit;
        if (permit.main_frontier < request.replay_cursor ||
            (permit.main_frontier == request.replay_cursor && !bridge_mtp) ||
            permit.main_frontier > request.replay_target) {
            throw std::logic_error("replay permit does not cover its next chunk");
        }
        prepare_capture_boundary(lane);
        const std::optional<std::uint32_t> capture_frontier =
            request.capture_reservation
                ? std::optional<std::uint32_t>(request.capture_reservation->frontier)
                : std::nullopt;
        ensure_sequence_kv_mapped(sequence, permit.main_frontier, permit.backend_frontier);
        set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                            : 0);

        const std::uint32_t cursor        = request.replay_cursor;
        const std::uint32_t prompt_tokens = request.base->summary.prompt_tokens;
        std::uint32_t count               = std::min(prefill_chunk, permit.main_frontier - cursor);
        // The multimodal prefix and the generated suffix have different input bindings. Keep
        // their boundary explicit even if both fit in the same scheduling allowance.
        if (cursor < prompt_tokens) { count = std::min(count, prompt_tokens - cursor); }
        // An exact prefix hit can be paused before its first sample. There is no committed
        // next token for the last MTP alignment column yet. Rebuild that target column alone,
        // leaving the original exact-hit bridge to consume the first newly sampled token.
        const bool awaits_first_mtp_token = speculative_backend == SpeculativeBackend::Mtp &&
                                            request.resume_lifecycle == Lifecycle::Prefilling &&
                                            request.replay_target == prompt_tokens &&
                                            sequence.ledger.size() == prompt_tokens;
        if (awaits_first_mtp_token && cursor + 1U < prompt_tokens) {
            count = std::min(count, prompt_tokens - cursor - 1U);
        }
        const bool skip_mtp_alignment = awaits_first_mtp_token && cursor + 1U == prompt_tokens;
        const auto splits             = sequence.prefix_identity.execution_frontiers();
        const auto next_split         = std::upper_bound(splits.begin(), splits.end(), cursor);
        std::optional<std::uint32_t> split =
            next_split != splits.end() ? std::optional<std::uint32_t>(*next_split) : std::nullopt;
        if (capture_frontier && (!split || *capture_frontier < *split)) {
            split = capture_frontier;
        }

        const StateImageSelectors selectors = state_selectors(sequence);
        execution::PrefillContext context{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            text_kv_view(sequence),
            skip_mtp_alignment ? PagedKVCacheView{} : mtp_kv_view(sequence),
            decoder->text_kv,
            skip_mtp_alignment ? nullptr : decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            cursor,
            nullptr,
            nullptr,
            selectors.source,
            selectors.destination,
            0,
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0,
            dflash_prefill_host_ingress,
            sequence.rope_delta};
        context.prefill_gpu_timer = &prefill_gpu_timer_;
        mark_workspace_usage(speculative_backend == SpeculativeBackend::Mtp
                                 ? workspace_plan.mtp_prefill
                                 : workspace_plan.text_prefill);
        if (is_masked_draft_backend(speculative_backend)) {
            mark_workspace_usage(workspace_plan.dflash_context);
        }

        const bool multimodal = request.base->prompt->has_media() && cursor < prompt_tokens;
        if (multimodal && (!request.replay || !request.replay->vision)) {
            if (!workspace_plan.vision || !request.base->vision_control_plan) {
                throw std::logic_error("Vision replay has no prepared resource plan");
            }
            request.replay.reset();
            request.replay.emplace();
            auto& replay  = *request.replay;
            replay.prompt = replay_prompt(*request.base, sequence);
            VisionPrefillPlan plan;
            plan.control = std::make_shared<VisionControl>(
                build_vision_control(replay.prompt, *request.base->vision_control_plan, 0));
            const auto& items = request.base->vision_control_plan->items;
            for (std::size_t index = 0; index < items.size(); ++index) {
                const auto& item = items[index];
                const std::uint32_t begin =
                    speculative_backend == SpeculativeBackend::Mtp && item.token_begin != 0
                        ? item.token_begin - 1U
                        : item.token_begin;
                plan.uses.push_back(
                    VisionUseSpan{.begin               = begin,
                                  .end                 = item.token_end,
                                  .prepared_item_index = static_cast<std::uint32_t>(index),
                                  .control_index       = static_cast<std::uint32_t>(index)});
                plan.max_merged_count = std::max(plan.max_merged_count, item.merged_count);
            }
            replay.vision_plan.emplace(std::move(plan));
            replay.vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, replay.prompt, *replay.vision_plan, vision_handoff,
                vision_handoff_peak_bytes);
        }

        if (bridge_mtp) {
            if (!sequence.tail_hidden_valid || !io.mtp) {
                throw std::logic_error("MTP replay source has no continuation hidden");
            }
            std::array<std::int32_t, 3> rope;
            for (std::size_t axis = 0; axis < rope.size(); ++axis) {
                const std::int64_t position =
                    cursor <= prompt_tokens
                        ? request.base->prompt->positions[axis * prompt_tokens + cursor - 1U]
                        : static_cast<std::int64_t>(cursor - 1U) + sequence.rope_delta;
                if (position < std::numeric_limits<std::int32_t>::min() ||
                    position > std::numeric_limits<std::int32_t>::max()) {
                    throw std::overflow_error("MTP replay bridge RoPE position exceeds int32");
                }
                rope[axis] = static_cast<std::int32_t>(position);
            }
            const execution::MtpBridgeInput bridge{.previous_hidden = &sequence.tail_hidden,
                                                   .position =
                                                       static_cast<std::int32_t>(cursor - 1U),
                                                   .rope_position = rope};
            if (multimodal) {
                execution::mtp_bridge_multimodal(context, request.replay->prompt,
                                                 *request.replay->vision, bridge);
            } else {
                Tensor token = io.mtp->target_input_ids.slice(0, 0, 1);
                CUDA_CHECK(cudaMemcpyAsync(token.data, sequence.ledger.data() + cursor,
                                           sizeof(TokenId), cudaMemcpyHostToDevice, device.stream));
                execution::mtp_bridge_and_propose(context, token, sequence.tail_hidden,
                                                  bridge.position, rope, false);
            }
            sequence.mtp_kv_valid = cursor;
            commit_sequence_kv(sequence, sequence.text_kv_valid, cursor);
            if (count == 0) {
                timing.begin_wait();
                device.synchronize();
                timing.end_wait();
                record_vision_time();
                work.reset();
                complete_replay();
                settle_unit(lane);
                return ReplayProgress{.complete = true, .timing = timing.finish()};
            }
        }

        execution::PrefillChunkResult result;
        timing.pause();
        if (multimodal) {
            if (!request.replay->vision) {
                throw std::logic_error("Vision replay session is unavailable");
            }
            mark_workspace_usage(workspace_plan.vision->capacity_bytes);
            result = execution::prefill_multimodal_chunk(
                context, request.replay->prompt, *request.replay->vision, count, split, false);
        } else {
            // Pass the complete ledger: MTP alignment consumes the next already committed token
            // after the replayed frontier, rather than io.token from another request's round.
            result = execution::prefill_text_chunk(context, sequence.ledger, count, split, false);
        }
        timing.include(result.timing);
        timing.resume_post();
        if (result.finalized || result.processed_tokens == 0 || result.processed_tokens > count) {
            throw std::logic_error("replay chunk made invalid or sampling progress");
        }
        const std::uint32_t end = cursor + result.processed_tokens;
        sequence.text_kv_valid  = end;
        if (speculative_backend == SpeculativeBackend::Mtp) {
            sequence.mtp_kv_valid = skip_mtp_alignment ? end - 1U : end;
        } else if (is_masked_draft_backend(speculative_backend)) {
            sequence.dflash_context_frontier = end;
        }
        commit_sequence_kv(sequence, end, backend_kv_valid(sequence));
        settle_state_fork(sequence);
        timing.resume_submit();
        copy_tail(sequence, prefill_hidden.slice(
                                1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        record_vision_time();
        work.reset();
        sequence.tail_hidden_valid = true;
        sequence.mtp_draft_count   = 0;
        request.replay_cursor      = end;

        const bool capture_ready = capture_frontier && end == *capture_frontier;
        if (capture_ready) { request.capture_pending = true; }
        // A recovered semantic point is published through the same capture transaction as
        // prefill. Even at the replay target, retain this lifecycle until that capture settles.
        const bool complete = end == request.replay_target && !capture_ready;
        if (complete) { complete_replay(); }
        settle_unit(lane);
        return ReplayProgress{.capture_ready    = capture_ready,
                              .processed_tokens = result.processed_tokens,
                              .complete         = complete,
                              .timing           = timing.finish()};
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        work.reset();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
