#include "guarded_main.h"
#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/program.h"
#include <cstdlib>

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace qwen = ninfer::models::qwen3_5;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void test_handoff_lifetime(ninfer::DeviceContext& device,
                           const qwen::execution::Parameters& parameters) {
    const auto& config     = parameters.model.config().vision.value();
    const auto merge       = static_cast<std::int32_t>(config.spatial_merge_size);
    const auto patches     = static_cast<std::size_t>(merge) * merge;
    const auto make_prompt = [&](std::uint32_t seed) {
        qwen::PreparedPromptData prompt;
        prompt.token_ids.resize(5);
        prompt.token_types         = {0, static_cast<std::uint8_t>(qwen::PromptModality::Image), 0,
                                      static_cast<std::uint8_t>(qwen::PromptModality::Image), 0};
        prompt.prepare.media_items = 2;
        prompt.prepare.raw_patches = 2 * patches;
        prompt.prepare.vision_tokens = 2;
        for (std::size_t item = 0; item != 2; ++item) {
            prompt.vision_items.push_back(qwen::VisionItem{
                .modality    = qwen::PromptModality::Image,
                .grid        = {.temporal = 1, .height = merge, .width = merge},
                .patch_begin = item * patches,
                .patch_count = patches,
                .token_spans = {{.begin = 2 * item + 1, .count = 1}},
            });
            auto payload            = std::make_shared<qwen::PreparedMediaPayload>();
            payload->patch_elements = patches * config.patch_width();
            payload->patches        = std::make_unique<std::uint16_t[]>(payload->patch_elements);
            for (auto& value : payload->mutable_span()) {
                seed  = seed * 1664525U + 1013904223U;
                value = static_cast<std::uint16_t>(0x3f00U | ((seed >> 24) & 0x7fU) |
                                                   ((seed >> 16) & 0x8000U));
            }
            prompt.media_payloads.push_back(std::move(payload));
        }
        return prompt;
    };
    auto prompt_a = make_prompt(1);
    auto prompt_b = make_prompt(3);
    qwen::detail::VisionPrefillPlan plan;
    plan.control_plan =
        std::make_shared<qwen::VisionControlPlan>(qwen::plan_vision_control(prompt_a, config));
    plan.control = std::make_shared<qwen::VisionControl>(
        qwen::build_vision_control(prompt_a, *plan.control_plan, 0));
    plan.uses             = {{.begin = 0, .end = 2, .prepared_item_index = 0, .control_index = 0},
                             {.begin = 2, .end = 4, .prepared_item_index = 1, .control_index = 1}};
    plan.max_merged_count = 1;
    const auto workspace_plan =
        qwen::execution::VisionContext::plan_workspace(config, *parameters.vision, 1, 256);
    ninfer::DeviceBuffer workspace(workspace_plan.capacity_bytes);
    const ninfer::DeviceSpan backing{workspace.p, workspace.bytes};
    qwen::detail::VisionHandoffState handoff;
    std::size_t peak_bytes  = 0;
    const auto make_session = [&](const qwen::PreparedPromptData& prompt) {
        return std::make_unique<qwen::execution::VisionPrefillSession>(
            device, parameters, backing, workspace_plan, prompt, plan, handoff, peak_bytes);
    };
    const auto read = [&](const qwen::execution::VisionChunk& chunk) {
        require(chunk.control != nullptr, "Vision replay omitted its item");
        std::vector<std::uint16_t> output(chunk.embeddings.bytes() / sizeof(std::uint16_t));
        CUDA_CHECK(cudaMemcpyAsync(output.data(), chunk.embeddings.data, chunk.embeddings.bytes(),
                                   cudaMemcpyDeviceToHost, device.stream));
        device.synchronize();
        return output;
    };
    auto session_a      = make_session(prompt_a);
    auto session_b      = make_session(prompt_b);
    const auto original = read(session_a->prepare_chunk(0, 2));
    require(read(session_a->prepare_chunk(2, 2)) != original,
            "Vision handoff fixture needs distinguishable items");
    require(read(session_a->prepare_chunk(0, 2)) == original,
            "Vision replay failed to revisit an earlier item");
    require(session_a->prepare_chunk(4, 1).control == nullptr,
            "text after the last Vision item retained a scatter binding");
    require(read(session_a->prepare_chunk(0, 2)) == original,
            "Vision replay failed after all item spans were consumed");
    const auto other = read(session_b->prepare_chunk(0, 2));
    require(other != original, "Vision handoff fixture needs distinguishable requests");
    require(session_a->active_handoff_bytes() == 0 && session_b->active_handoff_bytes() != 0,
            "Vision handoff remained owned by an overwritten session");
    require(read(session_a->prepare_chunk(1, 1)) == original,
            "Vision request failed to reconstruct an overwritten handoff");
    session_b->retire_handoff();
    require(session_a->active_handoff_bytes() != 0,
            "retiring an overwritten session released the current Vision handoff");

    // Text/decode may replace all general scratch while a Vision item remains live.
    CUDA_CHECK(
        cudaMemsetAsync(workspace.p, 0xa5, workspace_plan.general_capacity_bytes, device.stream));
    require(read(session_a->prepare_chunk(1, 1)) == original,
            "general workspace use changed the live Vision handoff");
    { auto unused_session = make_session(prompt_b); }
    require(session_a->active_handoff_bytes() != 0,
            "destroying an unused session released another Vision handoff");
    require(read(session_b->prepare_chunk(0, 2)) == other,
            "retired Vision request could not reconstruct its handoff");
    session_a.reset();
    require(session_b->active_handoff_bytes() != 0,
            "destroying an overwritten session released the current Vision handoff");
    require(prompt_a.media_payloads[0] && prompt_a.media_payloads[1] &&
                prompt_b.media_payloads[0] && prompt_b.media_payloads[1],
            "Vision execution released durable prepared media");
}

} // namespace

namespace {

int run() {
    namespace qwen       = ninfer::models::qwen3_5;
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) { return 77; }
    static_assert(ninfer::models::qwen3_5::kMaximumVisionItemTokens == 16384);
    // The current Vision route fits below 827 MiB, including arena alignment and empty-scratch
    // backing. The context growth check below protects the single-item lifetime bound.
    constexpr std::size_t kWorkspaceCeiling = 827ULL << 20;
    try {
        int devices       = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) {
            return 77;
        }
        CUDA_CHECK(status);
        ninfer::DeviceContext device;
        ninfer::models::LoadOptions selected;
        selected.vision        = true;
        selected.speculative   = ninfer::SpeculativeBackend::Mtp;
        selected.proposal_head = ninfer::ProposalHead::Optimized;
        auto model             = qwen::load_model(artifact, selected, device);
        const qwen::execution::Parameters parameters(*model);
        const auto capacity = [&](std::uint32_t max_context) {
            ninfer::EngineOptions options;
            options.max_context         = max_context;
            options.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
            options.prefill_chunk       = 1024;
            options.kv_cache            = ninfer::KvCacheStorage::Fp8E4M3Row256;
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens         = 3;
            options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
            options.enable_vision                    = true;
            options.use_cuda_graph                   = false;
            options.context_cache.device_state_slots = 1;
            auto planner     = qwen::make_sequence_planner(parameters, device, options);
            const auto pages = planner.capacity_curve().minimum_main_page_groups;
            return std::move(planner).finalize(pages).workspace_capacity_bytes();
        };
        // Increasing total context cannot exceed the single Vision item's 16K workspace.
        const auto at_item_limit    = capacity(16384);
        const auto above_item_limit = capacity(131072);
        if (at_item_limit == 0 || at_item_limit > kWorkspaceCeiling ||
            above_item_limit != at_item_limit) {
            throw std::runtime_error("Vision workspace exceeded the single-item bound: at=" +
                                     std::to_string(at_item_limit) +
                                     " above=" + std::to_string(above_item_limit));
        }
        test_handoff_lifetime(device, parameters);
        std::cout << "Vision workspace remains bounded for long text context\n";
        std::cout << "Vision handoff survives request rotation and replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
