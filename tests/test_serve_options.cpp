#include "serve/generation_service.h"
#include "serve/props.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.context_cost_presets.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.media_cache_bytes == ninfer::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == ninfer::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.context_cache.host_state_slots == ninfer::kDefaultHostStateSlots &&
                          defaults.context_cache.host_kv_capacity_bytes ==
                              ninfer::kDefaultHostKvCapacityBytes,
                      "Host context-cache defaults mismatch");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    // Without --default-max-tokens an omitted request limit is derived per request from the Engine's
    // concurrent lane budget, bounded above by --max-context; the flag replaces it with a fixed cap.
    const ServeOptions long_context =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "262144"});
    const RequestLimits derived = request_limits(long_context);
    failures += check(!derived.default_max_tokens && derived.max_context == 262144,
                      "omitted --default-max-tokens did not select the derived lane budget");
    GenerationRequest omitted;
    apply_default_output_limit(omitted, derived);
    failures += check(omitted.derive_output_budget && omitted.max_tokens == 262144,
                      "an omitted limit was not marked for derivation under the context bound");
    const ServeOptions capped = parse(
        {"ninfer-serve", "model.ninfer", "--max-context", "262144", "--default-max-tokens", "4096"});
    GenerationRequest fixed;
    apply_default_output_limit(fixed, request_limits(capped));
    failures += check(!fixed.derive_output_budget && fixed.max_tokens == 4096,
                      "explicit --default-max-tokens was not the default output limit");
    bool zero_default_output_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--default-max-tokens", "0"});
    } catch (const std::invalid_argument&) { zero_default_output_rejected = true; }
    failures += check(zero_default_output_rejected, "--default-max-tokens 0 was accepted");

    failures += check(!defaults.default_reasoning_effort,
                      "a reasoning effort is unexpectedly configured by default");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", "high"})
                              .default_reasoning_effort == RequestedReasoningEffort::High,
                      "--reasoning-effort high was not parsed");
    for (const char* rejected : {"none", "extreme"}) {
        bool effort_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", rejected});
        } catch (const std::invalid_argument&) { effort_rejected = true; }
        const std::string message = std::string("--reasoning-effort ") + rejected + " was accepted";
        failures += check(effort_rejected, message.c_str());
    }

    const ServeOptions fp8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "fp8"});
    failures += check(fp8.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string kv_help = serve_usage_text("ninfer-serve");
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos,
                      "serve help omits a production KV storage mode");

    // Every one of these parses without error whether or not the service carries it to the Engine,
    // so the mapping from parsed options to Engine options is what has to be checked.
    const ninfer::EngineOptions default_engine = make_engine_options(defaults);
    failures += check(!default_engine.prefill_cublas && default_engine.prefill_cublas_projections &&
                          default_engine.speculative.lookup_ngram == 0,
                      "the cuBLAS prefill route or context lookup is on by default in serving");
    const ServeOptions route =
        parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3",
               "--lookup-ngram", "5", "--prefill-cublas", "--no-prefill-cublas-projections"});
    const ninfer::EngineOptions route_engine = make_engine_options(route);
    failures += check(route_engine.speculative.lookup_ngram == 5 &&
                          route_engine.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          route_engine.speculative.draft_tokens == 3,
                      "--lookup-ngram did not reach the Engine options next to --spec");
    failures += check(route_engine.prefill_cublas && !route_engine.prefill_cublas_projections,
                      "the cuBLAS prefill controls did not reach the Engine options");
    for (const char* flag : {"--prefill-cublas", "--no-prefill-cublas-projections", "--lookup-ngram"}) {
        failures += check(kv_help.find(flag) != std::string::npos,
                          "serve help omits an accepted prefill or drafting control");
    }

    // rk8v4 still parses to its storage value; the engine rejects it in
    // target_kv_cache_profile so the failure names the unported feature rather than an
    // unknown --kv-dtype token.
    const ServeOptions rotor = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk8v4"});
    failures += check(rotor.kv_cache == ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                      "--kv-dtype rk8v4 did not select rotated K8/V4 storage");
    failures += check(defaults.kv_cache == ninfer::KvCacheStorage::BFloat16,
                      "rk8v4 unexpectedly changed the default KV storage");
    const ServeOptions lloyd = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk4v4"});
    failures += check(lloyd.kv_cache == ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value,
                      "--kv-dtype rk4v4 did not select rotated Lloyd-Max K4/V4 storage");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"ninfer-serve", "model.ninfer", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.context_cost_presets == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += check(options.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              options.speculative.draft_tokens == k &&
                              options.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "serve options did not preserve DFlash2 configuration");
    }

    const ServeOptions dflash_vision = parse(
        {"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 15,
                      "serve options did not preserve combined DFlash and Vision features");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse({"ninfer-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.context_cache.host_state_slots == 0 &&
                          configured.context_cache.host_kv_capacity_bytes == 0,
                      "root-only server mode retained default Host capacities");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures += check(configured.preserve_thinking == true,
                      "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging = parse({"ninfer-serve", "model.ninfer", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "log level did not reach serving options");

    const ServeOptions context_cache =
        parse({"ninfer-serve", "model.ninfer", "--device-state-slots", "3", "--host-state-slots",
               "5", "--host-kv-mib", "64", "--max-private-continuations", "9",
               "--max-shared-prefixes", "4", "--max-long-anchors-per-continuation", "2"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_state_slots == 5 &&
                          context_cache.context_cache.host_kv_capacity_bytes == (64ULL << 20) &&
                          context_cache.context_cache.max_private_continuations == 9 &&
                          context_cache.context_cache.max_shared_prefixes == 4 &&
                          context_cache.context_cache.max_long_anchors_per_continuation == 2,
                      "context-cache capacities did not reach serving options");
    bool disabled_cache_capacity_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--host-kv-mib", "64"});
    } catch (const std::invalid_argument&) { disabled_cache_capacity_rejected = true; }
    failures += check(disabled_cache_capacity_rejected,
                      "root-only server mode accepted context-cache capacity options");

    const ServeOptions auto_cache = parse({"ninfer-serve", "model.ninfer", "--auto-host-cache"});
    failures += check(auto_cache.context_cache.auto_host_cache && auto_cache.context_cache.enabled &&
                          !defaults.context_cache.auto_host_cache,
                      "--auto-host-cache did not reach serving options or is on by default");
    for (const char* sized : {"--host-kv-mib", "--host-state-slots", "--max-private-continuations",
                              "--max-shared-prefixes"}) {
        bool conflict_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--auto-host-cache", sized, "16"});
        } catch (const std::invalid_argument&) { conflict_rejected = true; }
        failures += check(conflict_rejected,
                          "--auto-host-cache accepted an explicit Host capacity option");
    }
    bool auto_without_cache_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--auto-host-cache"});
    } catch (const std::invalid_argument&) { auto_without_cache_rejected = true; }
    failures += check(auto_without_cache_rejected,
                      "--auto-host-cache was accepted with the context cache disabled");

    const ServeOptions no_slots = parse({"ninfer-serve", "model.ninfer"});
    failures += check(no_slots.slot_save_path.empty() && !no_slots.auto_save_evicted,
                      "slot persistence was on by default");
    const ServeOptions slots = parse(
        {"ninfer-serve", "model.ninfer", "--slot-save-path", "sessions", "--auto-save-evicted"});
    failures += check(slots.slot_save_path == "sessions" && slots.auto_save_evicted,
                      "slot persistence options did not reach serving options");
    const auto rejected = [&](std::vector<std::string> argv) {
        try {
            (void)parse(std::move(argv));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--auto-save-evicted"}),
                      "--auto-save-evicted was accepted without --slot-save-path");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--slot-save-path", ""}),
                      "an empty --slot-save-path was accepted");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--no-prefix-reuse",
                                "--slot-save-path", "sessions"}),
                      "--slot-save-path was accepted with prefix reuse disabled");
    failures += check(serve_usage_text("ninfer-serve").find("--slot-save-path") !=
                          std::string::npos,
                      "serve help omits --slot-save-path");

    failures += check(!parse({"ninfer-serve", "model.ninfer"}).auto_prefix_grid,
                      "automatic prefix grid was on without --auto-prefix-grid");
    failures +=
        check(parse({"ninfer-serve", "model.ninfer", "--auto-prefix-grid"}).auto_prefix_grid,
              "--auto-prefix-grid did not reach serving options");
    bool grid_without_reuse_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--auto-prefix-grid"});
    } catch (const std::invalid_argument&) { grid_without_reuse_rejected = true; }
    failures += check(grid_without_reuse_rejected,
                      "--auto-prefix-grid was accepted with prefix reuse disabled");

    ninfer::ContextCacheOptions resolved_cache;
    resolved_cache.enabled                           = true;
    resolved_cache.max_long_anchors_per_continuation = 2U;
    failures += check(resolve_automatic_private_anchors(parse({"ninfer-serve", "model.ninfer"}),
                                                        resolved_cache) == 2U,
                      "automatic long anchors did not default to the retained-anchor cap");
    failures += check(
        resolve_automatic_private_anchors(
            parse({"ninfer-serve", "model.ninfer", "--auto-long-anchors", "1"}), resolved_cache) ==
            1U,
        "--auto-long-anchors did not select fewer anchors than the cap");
    failures += check(
        resolve_automatic_private_anchors(
            parse({"ninfer-serve", "model.ninfer", "--auto-long-anchors", "9"}), resolved_cache) ==
            2U,
        "--auto-long-anchors was not clamped to the retained-anchor cap");
    failures += check(
        resolve_automatic_private_anchors(
            parse({"ninfer-serve", "model.ninfer", "--auto-long-anchors", "0"}), resolved_cache) ==
            0U,
        "--auto-long-anchors 0 did not disable automatic anchors");
    ninfer::ContextCacheOptions disabled_cache = resolved_cache;
    disabled_cache.enabled                     = false;
    failures += check(resolve_automatic_private_anchors(parse({"ninfer-serve", "model.ninfer"}),
                                                        disabled_cache) == 0U,
                      "automatic long anchors survived a disabled context cache");
    bool anchors_without_reuse_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--auto-long-anchors",
                     "2"});
    } catch (const std::invalid_argument&) { anchors_without_reuse_rejected = true; }
    failures += check(anchors_without_reuse_rejected,
                      "--auto-long-anchors was accepted with prefix reuse disabled");
    // The conflict is on presence, like --auto-prefix-grid: even the disabling value is refused
    // in root-only mode, while it stays valid when prefix reuse is on.
    bool zero_anchors_without_reuse_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--auto-long-anchors",
                     "0"});
    } catch (const std::invalid_argument&) { zero_anchors_without_reuse_rejected = true; }
    failures += check(zero_anchors_without_reuse_rejected,
                      "--auto-long-anchors 0 was accepted with prefix reuse disabled");
    failures +=
        check(parse({"ninfer-serve", "model.ninfer", "--auto-long-anchors", "0"}).auto_long_anchors ==
                  0U,
              "--auto-long-anchors 0 was not accepted with prefix reuse enabled");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "20", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 20 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--top-k", "21"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens   = 1;
    const auto semantics = resolve_prompt_semantics(request, defaults);
    failures += check(!semantics.reasoning_effort && !semantics.enable_thinking &&
                          !semantics.reasoning_effort,
                      "omitted reasoning effort did not resolve to the template default");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const ninfer::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking = resolve_prompt_semantics(request, thinking_budget);
    failures += check(!non_thinking.reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = RequestedReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults);
    failures += check(explicit_effort.reasoning_effort == ninfer::ReasoningEffort::Low &&
                          explicit_effort.enable_thinking == true,
                      "explicit reasoning effort did not remain the effective effort");
    request.reasoning_effort.reset();
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == true,
                      "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == false,
                      "request preserve-thinking override did not win");

    // Client effort vocabularies are wider than the three rungs the maintained Qwen templates
    // accept: OpenAI and Claude Code send 'high', pi sends 'minimal' and 'max'. Those collapse
    // onto the nearest rung rather than making the template raise.
    const auto collapses = [&](RequestedReasoningEffort wire) {
        GenerationRequest aliased = GenerationRequest{};
        aliased.max_tokens        = 1;
        aliased.reasoning_effort  = wire;
        return resolve_prompt_semantics(aliased, defaults).reasoning_effort;
    };
    failures += check(collapses(RequestedReasoningEffort::Minimal) == ninfer::ReasoningEffort::Low,
                      "'minimal' did not collapse onto the template's low rung");
    failures += check(collapses(RequestedReasoningEffort::High) == ninfer::ReasoningEffort::XHigh,
                      "'high' did not collapse onto the template's xhigh rung");
    failures += check(collapses(RequestedReasoningEffort::Max) == ninfer::ReasoningEffort::XHigh,
                      "'max' did not collapse onto the template's xhigh rung");
    failures += check(collapses(RequestedReasoningEffort::Medium) == ninfer::ReasoningEffort::Medium,
                      "'medium' did not pass through unchanged");
    failures += check(collapses(RequestedReasoningEffort::None) == ninfer::ReasoningEffort::None,
                      "'none' did not pass through unchanged");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--auto-prefix-grid") != std::string::npos,
              "serve help omits --auto-prefix-grid");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--auto-long-anchors") != std::string::npos,
              "serve help omits --auto-long-anchors");

    // --devices selects the ordered CUDA devices the model's pipeline stages run on. Only the shape
    // is parsed here; matching compute capability is the engine's to validate, because it needs real
    // devices.
    failures += check(parse({"ninfer-serve", "model.ninfer"}).devices.empty(),
                      "devices defaulted to a non-empty list");
    const ServeOptions one_device = parse({"ninfer-serve", "model.ninfer", "--devices", "3"});
    failures += check(one_device.devices.size() == 1 && one_device.devices[0] == 3,
                      "--devices with one entry did not reach serving options");
    const ServeOptions pair = parse({"ninfer-serve", "model.ninfer", "--devices", "1,2"});
    failures += check(pair.devices.size() == 2 && pair.devices[0] == 1 && pair.devices[1] == 2,
                      "--devices did not preserve the ordered pair");
    const ServeOptions triple = parse({"ninfer-serve", "model.ninfer", "--devices", "0,1,2"});
    failures += check(triple.devices.size() == 3 && triple.devices[2] == 2,
                      "--devices did not accept more than two stages");

    for (const auto& [value, why] : std::vector<std::pair<std::string, const char*>>{
             {"0,1,2,3,4,5,6,7,8", "more devices than a context holds"},
             {"", "an empty list"},
             {"1,", "a trailing comma"}}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--devices", value});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, why);
    }

    // "0,0" is deliberately permitted: it puts both ranks on one card so the split path can be
    // exercised on a single-GPU machine.
    const ServeOptions same_card = parse({"ninfer-serve", "model.ninfer", "--devices", "0,0"});
    failures += check(same_card.devices.size() == 2 && same_card.devices[0] == 0 &&
                           same_card.devices[1] == 0,
                      "--devices 0,0 was not accepted for single-card split coverage");

    // --stage-layers gives the layers per stage, one count per device.
    const ServeOptions staged =
        parse({"ninfer-serve", "model.ninfer", "--devices", "0,0", "--stage-layers", "20,44"});
    failures += check(staged.stage_layers == std::vector<std::uint32_t>({20, 44}),
                      "--stage-layers did not reach serving options");
    bool stage_layers_alone_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--stage-layers", "20,44"});
    } catch (const std::invalid_argument&) { stage_layers_alone_rejected = true; }
    failures += check(stage_layers_alone_rejected,
                      "--stage-layers without a multi-device split was accepted");
    bool zero_stage_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--devices", "0,0", "--stage-layers", "0,64"});
    } catch (const std::invalid_argument&) { zero_stage_rejected = true; }
    failures += check(zero_stage_rejected, "a stage with no layers was accepted");

    bool exclusive_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--device", "0", "--devices", "0,1"});
    } catch (const std::invalid_argument&) { exclusive_rejected = true; }
    failures += check(exclusive_rejected, "--device and --devices were accepted together");

    failures += check(serve_usage_text("ninfer-serve").find("--devices") != std::string::npos,
                      "serve help omits --devices");
    failures += check(serve_usage_text("ninfer-serve").find("--host-kv-mib") != std::string::npos,
                      "serve help omits context-cache capacities");
    failures += check(serve_usage_text("ninfer-serve").find("device-state=max-concurrency") !=
                          std::string::npos,
                      "serve help omits context-cache defaults");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(serve_usage_text("ninfer-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--context-cost-presets") != std::string::npos,
              "serve help omits external context-cost presets");
    failures += check(serve_usage_text("ninfer-serve").find("metadata.name") != std::string::npos,
                      "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    if (failures == 0) { std::cout << "ok\n"; }

    {
        const ServeOptions overlay =
            parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-residency", "overlay",
                   "--vision-max-merged", "12288"});
        failures += check(overlay.vision_residency == ninfer::VisionResidency::Overlay,
                          "--vision-residency overlay did not select overlay residency");
        failures += check(overlay.vision_max_merged_tokens == 12288U,
                          "--vision-max-merged did not set the merged-token budget");
        const ServeOptions resident = parse({"ninfer-serve", "model.ninfer", "--vision"});
        failures += check(resident.vision_residency == ninfer::VisionResidency::Resident,
                          "vision residency does not default to resident");
        failures += check(resident.vision_max_merged_tokens == 16384U,
                          "vision merged-token budget does not default to the item maximum");
        bool overlay_without_vision_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--vision-residency", "overlay"});
        } catch (const std::invalid_argument&) { overlay_without_vision_rejected = true; }
        failures += check(overlay_without_vision_rejected,
                          "--vision-residency overlay without --vision was accepted");
        bool small_budget_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-max-merged", "32"});
        } catch (const std::invalid_argument&) { small_budget_rejected = true; }
        failures += check(small_budget_rejected, "--vision-max-merged 32 was accepted");
        bool unknown_mode_rejected = false;
        try {
            (void)parse(
                {"ninfer-serve", "model.ninfer", "--vision", "--vision-residency", "sometimes"});
        } catch (const std::invalid_argument&) { unknown_mode_rejected = true; }
        failures += check(unknown_mode_rejected, "--vision-residency sometimes was accepted");
    }

    {
        const ServeOptions grafted = parse({"ninfer-serve", "model.ninfer", "--graft",
                                            "product=C:/grafts/p.bin", "--graft", "b=b.bin"});
        failures += check(grafted.grafts.size() == 2 && grafted.grafts[0].name == "product" &&
                              grafted.grafts[0].path == "C:/grafts/p.bin" &&
                              grafted.grafts[1].name == "b",
                          "--graft NAME=PATH was not parsed in order");
        for (const char* malformed : {"product", "=p.bin", "product="}) {
            bool rejected = false;
            try {
                (void)parse({"ninfer-serve", "model.ninfer", "--graft", malformed});
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "malformed --graft was accepted");
        }

        GenerationRequest graft_request;
        graft_request.max_tokens = 1;
        graft_request.graft      = "product";
        failures += check(resolve_prompt_semantics(graft_request, grafted).graft == "product" &&
                              to_prompt_input(graft_request,
                                              resolve_prompt_semantics(graft_request, grafted), {})
                                      .options.graft == "product",
                          "a loaded graft did not reach the Engine prompt options");
        std::string unknown_code;
        try {
            (void)resolve_prompt_semantics(graft_request, defaults);
        } catch (const ApiException& error) { unknown_code = error.error().code; }
        failures += check(unknown_code == "unknown_graft",
                          "a graft the server did not load was not rejected");
        graft_request.graft.reset();
        failures += check(resolve_prompt_semantics(graft_request, grafted).graft.empty(),
                          "a request without a graft selected one with no default configured");
        graft_request.graft = "";
        failures += check(resolve_prompt_semantics(graft_request, grafted).graft.empty(),
                          "an empty graft selected one");
    }

    {
        const ServeOptions with_default =
            parse({"ninfer-serve", "model.ninfer", "--default-graft", "b", "--graft",
                   "product=p.bin", "--graft", "b=b.bin"});
        failures += check(with_default.default_graft == "b",
                          "--default-graft was not parsed independent of --graft order");

        GenerationRequest request;
        request.max_tokens = 1;
        failures += check(resolve_prompt_semantics(request, with_default).graft == "b",
                          "a request without a graft did not take the server default");
        request.graft = std::string{};
        failures += check(resolve_prompt_semantics(request, with_default).graft.empty(),
                          "an empty graft did not opt out of the server default");
        request.graft = "product";
        failures += check(resolve_prompt_semantics(request, with_default).graft == "product",
                          "an explicit graft did not override the server default");
        request.graft = "missing";
        std::string unknown_code;
        try {
            (void)resolve_prompt_semantics(request, with_default);
        } catch (const ApiException& error) { unknown_code = error.error().code; }
        failures += check(unknown_code == "unknown_graft",
                          "an unknown explicit graft was accepted under a server default");

        for (const std::vector<std::string>& invalid : {
                 std::vector<std::string>{"ninfer-serve", "model.ninfer", "--default-graft", "b"},
                 std::vector<std::string>{"ninfer-serve", "model.ninfer", "--graft", "a=a.bin",
                                          "--default-graft", "b"},
                 std::vector<std::string>{"ninfer-serve", "model.ninfer", "--graft", "a=a.bin",
                                          "--default-graft", ""}}) {
            bool rejected = false;
            try {
                (void)parse(invalid);
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "--default-graft naming no loaded graft was accepted");
        }
    }

    {
        // /props reports what a request that states nothing actually inherits.
        const ninfer::ModelSamplingDefaults presets{
            .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F},
            .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.8F,
                             .presence_penalty = 1.5F}};
        const ServeOptions served =
            parse({"ninfer-serve", "model.ninfer", "--max-context", "65536", "--max-concurrency",
                   "2", "--top-p", "0.9"});
        const nlohmann::json props = nlohmann::json::parse(make_props(
            served, ModelDescription{.id = "qwen", .max_model_len = 65536, .vision = true},
            presets));
        const nlohmann::json& settings = props.at("default_generation_settings");
        const nlohmann::json& params   = settings.at("params");
        failures += check(settings.at("n_ctx") == 65536 && props.at("total_slots") == 2 &&
                              props.at("model_alias") == "qwen" &&
                              props.at("model_path") == "model.ninfer",
                          "/props context, slots or identity mismatch");
        failures += check(params.at("n_predict") == -1 && params.at("max_tokens") == -1,
                          "/props did not report the remaining-context output default as -1");
        failures += check(params.at("temperature") == 1.0 && params.at("top_k") == 20 &&
                              params.at("top_p") == 0.9F && !params.contains("seed"),
                          "/props sampler is not the thinking preset under process overrides");
        failures += check(props.at("modalities") ==
                              nlohmann::json{{"vision", true}, {"audio", false}},
                          "/props modalities mismatch");

        const ServeOptions capped =
            parse({"ninfer-serve", "model.ninfer", "--no-thinking", "--default-max-tokens",
                   "2048", "--greedy", "--seed", "7"});
        const nlohmann::json capped_params =
            nlohmann::json::parse(
                make_props(capped, ModelDescription{.id = "qwen", .max_model_len = 8192}, presets))
                .at("default_generation_settings")
                .at("params");
        failures += check(capped_params.at("n_predict") == 2048 &&
                              capped_params.at("temperature") == 0.0 &&
                              capped_params.at("presence_penalty") == 1.5F &&
                              capped_params.at("seed") == 7,
                          "/props did not follow --default-max-tokens, --no-thinking, --greedy "
                          "and --seed");
    }

    return failures == 0 ? 0 : 1;
}
