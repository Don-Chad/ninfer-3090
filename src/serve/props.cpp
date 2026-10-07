#include "serve/props.h"

#include "product/version/version.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

std::string make_props(const ServeOptions& options, const ModelDescription& model,
                       const ninfer::ModelSamplingDefaults& sampling_defaults) {
    using Json = nlohmann::json;

    // A request that states nothing runs in the server's default thinking mode (on unless
    // --no-thinking), whose registered preset the process flags then override; --greedy last.
    const ninfer::SamplingMode mode = options.enable_thinking.value_or(true)
                                          ? ninfer::SamplingMode::Thinking
                                          : ninfer::SamplingMode::NonThinking;
    const ninfer::SamplingPreset& preset     = sampling_defaults.for_mode(mode);
    const ninfer::SamplingOverrides& process = options.sampling_overrides;
    const float temperature = options.greedy ? 0.0F : process.temperature.value_or(preset.temperature);

    // llama.cpp's -1 means "no fixed cap": without --default-max-tokens NInfer derives each
    // request's budget from its prompt and the lane share, so no single number applies.
    const int n_predict = options.default_max_tokens.value_or(-1);
    Json params{{"n_predict", n_predict},
                {"max_tokens", n_predict},
                {"temperature", temperature},
                {"top_k", process.top_k.value_or(preset.top_k)},
                {"top_p", process.top_p.value_or(preset.top_p)},
                {"min_p", process.min_p.value_or(preset.min_p)},
                {"presence_penalty", process.presence_penalty.value_or(preset.presence_penalty)},
                {"frequency_penalty", process.frequency_penalty.value_or(preset.frequency_penalty)}};
    // An unset seed is a fresh random seed per request; llama.cpp's sentinel for that is not
    // NInfer's contract, so only a fixed --seed is reported.
    if (process.seed) { params["seed"] = *process.seed; }

    const Json props{
        {"default_generation_settings", Json{{"n_ctx", model.max_model_len}, {"params", params}}},
        {"total_slots", options.max_concurrency},
        {"model_alias", model.id},
        {"model_path", options.artifact_path},
        {"modalities", Json{{"vision", model.vision}, {"audio", false}}},
        {"build_info", std::string("ninfer ") + std::string(ninfer::product::build_version())},
    };
    return props.dump();
}

} // namespace ninfer::serve
