#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "core/dtype.h"
#include "core/evictable_weight_pool.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/stage_plan.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text = out->config.text;
    out->weights.text =
        loading::bind_text(bindings, text, options,
                           loading::plan_stage_plan(text.num_hidden_layers, options));
    if (options.overlay_vision() && !out->config.vision) {
        throw std::invalid_argument("--vision-residency overlay requires a Vision artifact");
    }
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(
            bindings, *out->config.vision, text,
            options.overlay_vision() ? artifact::Residency::Pinned : artifact::Residency::Device);
    }
    std::pair<std::size_t, std::size_t> mtp_parameters{bindings.weights.size(),
                                                       bindings.weights.size()};
    if (out->config.mtp) {
        out->weights.mtp      = loading::bind_mtp(bindings, text, out->weights.text);
        mtp_parameters.second = bindings.weights.size();
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    loading::apply_storage_trades(bindings, out->config, out->weights, options);
    if (options.overlay_vision()) {
        loading::apply_vision_overlay_placement(bindings, out->weights, mtp_parameters);
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish(
        options.overlay_vision() ? EvictableWeightPool::kChunkBytes : 1);
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

namespace {

// What one stage's device needs whatever layers it owns, beyond weights: the CUDA context, the
// workspace and graph memory of a forward pass over a full chunk. An estimate, since the Program
// plans these later; it only decides how the layers are dealt out, and `--stage-layers` overrides it.
constexpr std::uint64_t kStageFixedBytes = 1536ULL << 20;
// What rank 0 needs on top: the round buffers, logits and sampling state that stay with the head.
constexpr std::uint64_t kHeadStageFixedBytes = 1024ULL << 20;

std::uint64_t parameter_bytes(const artifact::Reader& reader, const loading::Bindings& bindings,
                              WeightId id) {
    std::uint64_t total = 0;
    for (const auto& part : bindings.at(id).reference.binding.parts) {
        total += reader.geometry(part.object).bytes;
    }
    return total;
}

} // namespace

std::vector<std::uint32_t> default_stage_layers(const artifact::Reader& reader,
                                                LoadOptions options, const StageSizing& sizing,
                                                std::span<const std::uint64_t> free_bytes) {
    if (free_bytes.size() < 2) {
        throw std::invalid_argument("default stage layers need at least two devices");
    }
    // Bind the text model on one stage purely to size its layers.
    options.ranks = 1;
    options.stage_layers.clear();
    const Config config = parse_config(reader.directory(), options);
    const auto& text    = config.text;
    artifact::Binder binder(reader);
    loading::Bindings bindings(binder);
    const TextWeights weights =
        loading::bind_text(bindings, text, options, StagePlan(text.num_hidden_layers));

    std::vector<LayerCost> layers;
    layers.reserve(weights.layers.size());
    std::uint64_t layer_bytes_total = 0;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        LayerCost cost;
        for (const WeightId id : loading::layer_weights(weights.layers[layer])) {
            cost.weight_bytes += parameter_bytes(reader, bindings, id);
        }
        if (text.layer_types[layer] == MixerKind::FullAttention && text.attention) {
            // One page group of this layer: every plane of a page, for every KV head, as stored.
            const PagedKVStorageLayout storage = paged_kv_storage_layout(
                sizing.kv_storage, static_cast<std::int32_t>(text.attention->head_dim));
            cost.kv_bytes_per_page_group = static_cast<std::uint64_t>(kPagedKVPageSize) *
                                           text.attention->num_key_value_heads *
                                           storage.physical_bytes_per_token_head();
        } else if (text.gdn) {
            // The convolution window and the recurrent matrix of every state slot.
            const auto& gdn = *text.gdn;
            const std::uint64_t conv_bytes =
                gdn.conv_channels() * (gdn.linear_conv_kernel_dim - 1) * dtype_size(DType::BF16);
            const std::uint64_t recurrent_bytes =
                static_cast<std::uint64_t>(gdn.linear_key_head_dim) * gdn.linear_value_head_dim *
                gdn.linear_num_value_heads *
                dtype_size(options.gdn_state_fp16 ? DType::FP16 : DType::FP32);
            cost.state_bytes = sizing.state_slots * (conv_bytes + recurrent_bytes);
        }
        layer_bytes_total += cost.weight_bytes;
        layers.push_back(cost);
    }

    // The head stage keeps the embedding, the head and the final norm; MTP's layer sits with them.
    std::uint64_t head_bytes = parameter_bytes(reader, bindings, weights.token_embedding) +
                               parameter_bytes(reader, bindings, weights.output_head) +
                               parameter_bytes(reader, bindings, weights.final_norm);
    if (config.mtp && options.speculative == SpeculativeBackend::Mtp) {
        head_bytes += layer_bytes_total / std::max<std::size_t>(layers.size(), 1);
    }
    // A resident Vision tower sits on the primary device too. An overlay tower is pinned on the
    // host and borrows the head's evictable tail, which `head_bytes` already counts.
    if (config.vision && options.vision && !options.overlay_vision()) {
        const std::size_t first = bindings.weights.size();
        (void)loading::bind_vision(bindings, *config.vision, text, artifact::Residency::Device);
        for (std::size_t index = first; index < bindings.weights.size(); ++index) {
            head_bytes += parameter_bytes(reader, bindings, WeightId{index});
        }
    }

    std::vector<StageBudget> budgets;
    budgets.reserve(free_bytes.size());
    for (std::size_t stage = 0; stage < free_bytes.size(); ++stage) {
        budgets.push_back({.available_bytes = free_bytes[stage],
                           .fixed_bytes     = kStageFixedBytes +
                                          (stage == 0 ? kHeadStageFixedBytes + head_bytes : 0)});
    }
    // The estimates above are rough. When they say nothing fits, deal the layers out evenly and let
    // the Program's exact planning accept the split or report which device runs out.
    const StagePlan plan = [&] {
        try {
            return solve_stage_plan(layers, budgets).plan;
        } catch (const std::runtime_error&) {
            return StagePlan::even(text.num_hidden_layers, free_bytes.size());
        }
    }();
    std::vector<std::uint32_t> counts;
    counts.reserve(plan.stages());
    for (std::size_t stage = 0; stage < plan.stages(); ++stage) {
        counts.push_back(plan.stage_layers(stage));
    }
    return counts;
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data = std::move(plan.impl_);
    std::unique_ptr<EvictableWeightPool> pool;
    if (data->options.overlay_vision()) {
        // The tail is sized against the encode window once execution planning knows it; the
        // pool's mirror is captured then (see Model::weight_pool).
        const auto& materialization = data->materialization;
        if (materialization.evictable_tail_bytes == 0 || materialization.pinned_objects.empty()) {
            throw std::logic_error("overlay Vision load plan has no evictable tail or pinned tower");
        }
        if (!EvictableWeightPool::supported(device)) {
            throw std::invalid_argument(
                "--vision-residency overlay requires CUDA virtual memory management support");
        }
        pool = std::make_unique<EvictableWeightPool>(
            device, EvictableWeightPool::Config{
                        .arena_bytes =
                            static_cast<std::size_t>(materialization.device_capacity(0)),
                        .evictable_tail_bytes =
                            static_cast<std::size_t>(materialization.evictable_tail_bytes),
                    });
    }
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer,
                                         std::move(pool));
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    std::optional<VisionOverlayLayout> vision_overlay;
    if (data->options.overlay_vision()) {
        vision_overlay =
            loading::vision_overlay_layout(*data->weights.vision, bound, backing.pinned_block());
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing),
        std::move(vision_overlay)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
