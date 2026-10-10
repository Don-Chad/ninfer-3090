#include "serve/generation_service.h"
#include "serve/props.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <limits>
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
    failures += check(!defaults.context_cache.host_capacity_bytes.has_value(),
                      "default Host context capacity was resolved before Engine startup");
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

    // Without --default-max-tokens an omitted request limit is derived per request as the prompt's
    // remaining context, bounded above by --max-context; the flag replaces it with a fixed cap.
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

    // --max-output-tokens bounds every request's budget, stated or derived, and is off by default.
    failures += check(!defaults.max_output_tokens &&
                          bounded_output_budget(200000, defaults.max_output_tokens) == 200000,
                      "an output bound applied without --max-output-tokens");
    const ServeOptions bounded =
        parse({"ninfer-serve", "model.ninfer", "--max-output-tokens", "16384"});
    failures += check(bounded_output_budget(200000, bounded.max_output_tokens) == 16384 &&
                          bounded_output_budget(500, bounded.max_output_tokens) == 500,
                      "--max-output-tokens did not bound large budgets only");
    bool zero_output_bound_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--max-output-tokens", "0"});
    } catch (const std::invalid_argument&) { zero_output_bound_rejected = true; }
    failures += check(zero_output_bound_rejected, "--max-output-tokens 0 was accepted");

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
    failures += check(!default_engine.prefill_cublas && default_engine.prefill_cublas_projections,
                      "the cuBLAS prefill route is on by default in serving");
    const ServeOptions route =
        parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3",
               "--prefill-cublas", "--no-prefill-cublas-projections"});
    const ninfer::EngineOptions route_engine = make_engine_options(route);
    failures += check(route_engine.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          route_engine.speculative.draft_tokens == 3,
                      "--spec and --draft-tokens did not reach the Engine options");
    failures += check(route_engine.prefill_cublas && !route_engine.prefill_cublas_projections,
                      "the cuBLAS prefill controls did not reach the Engine options");
    for (const char* flag : {"--prefill-cublas", "--no-prefill-cublas-projections"}) {
        failures += check(kv_help.find(flag) != std::string::npos,
                          "serve help omits an accepted prefill control");
    }
    // The MTP-only context lookup was removed with no alias; n-gram copy drafting replaces it.
    bool lookup_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--lookup-ngram", "5"});
    } catch (const std::invalid_argument&) { lookup_rejected = true; }
    failures += check(lookup_rejected && kv_help.find("--lookup-ngram") == std::string::npos,
                      "serve still accepts or advertises the removed --lookup-ngram");

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

    failures += check(parse({"ninfer-serve", "model.ninfer"}).reasoning_loop ==
                          ninfer::ReasoningLoopAction::Off,
                      "the reasoning-loop guard is not off by default");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--reasoning-loop", "stop"})
                                  .reasoning_loop == ninfer::ReasoningLoopAction::Stop &&
                          parse({"ninfer-serve", "model.ninfer", "--reasoning-loop", "conclude"})
                                  .reasoning_loop == ninfer::ReasoningLoopAction::Conclude &&
                          parse({"ninfer-serve", "model.ninfer", "--reasoning-loop", "off"})
                                  .reasoning_loop == ninfer::ReasoningLoopAction::Off,
                      "--reasoning-loop did not select its action");
    bool bad_reasoning_loop_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--reasoning-loop", "on"});
    } catch (const std::invalid_argument&) { bad_reasoning_loop_rejected = true; }
    failures += check(bad_reasoning_loop_rejected, "--reasoning-loop on was accepted");

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

    for (const char* backend : {"mtp", "dflash", "dflash2"}) {
        const ServeOptions implied =
            parse({"ninfer-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3"});
        failures += check(implied.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "--spec did not imply the optimized proposal head");
    }

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
    failures += check(!configured.context_cache.enabled &&
                          configured.context_cache.host_capacity_bytes ==
                              defaults.context_cache.host_capacity_bytes &&
                          configured.context_cache.device_state_slots ==
                              defaults.context_cache.device_state_slots,
                      "--no-prefix-reuse changed context capacities or retained cache enablement");
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

    const ServeOptions context_cache = parse(
        {"ninfer-serve", "model.ninfer", "--device-state-slots", "3", "--host-context-mib", "64"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_capacity_bytes == (64ULL << 20),
                      "context-cache capacities did not reach serving options");

    const ServeOptions zero_host_context =
        parse({"ninfer-serve", "model.ninfer", "--host-context-mib", "0"});
    failures += check(zero_host_context.context_cache.enabled &&
                          zero_host_context.context_cache.host_capacity_bytes == 0,
                      "zero Host context capacity was not retained as an explicit budget");

    for (const auto& [mib, bytes] : std::vector<std::pair<std::string, std::size_t>>{
             {"146.822265625", 153954304},
             {"587.2890625", 615817216},
             {"186.822265625", 195897344},
             {"0.00000095367431640625", 1},
             {"0.000000953674316406250000", 1},
             {"000146.822265625000", 153954304},
             {"64.0000", 64ULL << 20},
             {"000.0000", 0},
         }) {
        const ServeOptions exact_host_context =
            parse({"ninfer-serve", "model.ninfer", "--host-context-mib", mib});
        failures += check(exact_host_context.context_cache.host_capacity_bytes == bytes,
                          ("Host context MiB did not preserve exact bytes: " + mib).c_str());
    }

    constexpr std::size_t bytes_per_mib  = 1ULL << 20;
    constexpr std::size_t max_host_bytes = std::numeric_limits<std::size_t>::max();
    std::string maximum_host_mib         = std::to_string(max_host_bytes / bytes_per_mib);
    std::size_t fractional_bytes         = max_host_bytes % bytes_per_mib;
    if (fractional_bytes != 0) { maximum_host_mib += '.'; }
    while (fractional_bytes != 0) {
        fractional_bytes *= 10;
        maximum_host_mib += static_cast<char>('0' + fractional_bytes / bytes_per_mib);
        fractional_bytes %= bytes_per_mib;
    }
    const ServeOptions maximum_host_context =
        parse({"ninfer-serve", "model.ninfer", "--host-context-mib", maximum_host_mib});
    failures += check(maximum_host_context.context_cache.host_capacity_bytes == max_host_bytes,
                      "maximum size_t Host context capacity was not accepted exactly");

    bool overflowing_host_context_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--host-context-mib",
                     std::to_string(std::numeric_limits<std::size_t>::max() / (1ULL << 20) + 1)});
    } catch (const std::invalid_argument&) { overflowing_host_context_rejected = true; }
    failures +=
        check(overflowing_host_context_rejected, "overflowing Host context capacity was accepted");

    for (const std::string mib : {"0.1", "0.0000001", "0.000000476837158203125", "-1", "-0", "+1",
                                  "nan", "inf", "1e3", "1.", ".5", "", " 1", "1 ", "1.2.3"}) {
        bool invalid_host_context_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--host-context-mib", mib});
        } catch (const std::invalid_argument&) { invalid_host_context_rejected = true; }
        failures +=
            check(invalid_host_context_rejected,
                  ("invalid or fractional-byte Host context MiB was accepted: " + mib).c_str());
    }

    for (const std::string option : {"--max-request-mib", "--media-cache-mib", "--media-live-mib",
                                     "--response-store-max-mib"}) {
        bool fractional_mib_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", option, "1.5"});
        } catch (const std::invalid_argument&) { fractional_mib_rejected = true; }
        failures += check(fractional_mib_rejected,
                          ("integer-only MiB option accepted a fraction: " + option).c_str());
    }

    for (const bool disable_first : {false, true}) {
        std::vector<std::string> arguments = {"ninfer-serve", "model.ninfer"};
        if (disable_first) { arguments.push_back("--no-prefix-reuse"); }
        arguments.insert(arguments.end(),
                         {"--device-state-slots", "3", "--host-context-mib", "64"});
        if (!disable_first) { arguments.push_back("--no-prefix-reuse"); }
        const ServeOptions disabled_cache = parse(std::move(arguments));
        failures +=
            check(!disabled_cache.allow_prefix_reuse && !disabled_cache.context_cache.enabled &&
                      disabled_cache.context_cache.device_state_slots == 3 &&
                      disabled_cache.context_cache.host_capacity_bytes == (64ULL << 20),
                  "--no-prefix-reuse did not preserve independently configured capacities");
    }

    const ServeOptions auto_cache = parse({"ninfer-serve", "model.ninfer", "--auto-host-cache"});
    failures += check(auto_cache.context_cache.auto_host_cache && auto_cache.context_cache.enabled &&
                          !defaults.context_cache.auto_host_cache,
                      "--auto-host-cache did not reach serving options or is on by default");
    bool conflict_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--auto-host-cache", "--host-context-mib", "16"});
    } catch (const std::invalid_argument&) { conflict_rejected = true; }
    failures += check(conflict_rejected, "--auto-host-cache accepted an explicit --host-context-mib");
    const ServeOptions reserve = parse(
        {"ninfer-serve", "model.ninfer", "--auto-host-cache", "--host-cache-reserve-mib", "512"});
    failures += check(reserve.context_cache.host_cache_reserve_bytes == (512ULL << 20) &&
                          auto_cache.context_cache.host_cache_reserve_bytes ==
                              ninfer::kDefaultHostCacheReserveBytes,
                      "--host-cache-reserve-mib did not reach serving options or has no default");
    bool reserve_without_auto_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--host-cache-reserve-mib", "512"});
    } catch (const std::invalid_argument&) { reserve_without_auto_rejected = true; }
    failures += check(reserve_without_auto_rejected,
                      "--host-cache-reserve-mib was accepted without --auto-host-cache");
    const ServeOptions host_capped = parse(
        {"ninfer-serve", "model.ninfer", "--auto-host-cache", "--host-cache-max-mib", "10240"});
    failures += check(host_capped.context_cache.host_cache_max_bytes == (10240ULL << 20) &&
                          !auto_cache.context_cache.host_cache_max_bytes,
                      "--host-cache-max-mib did not reach serving options or has a default");
    bool max_without_auto_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--host-cache-max-mib", "1024"});
    } catch (const std::invalid_argument&) { max_without_auto_rejected = true; }
    failures += check(max_without_auto_rejected,
                      "--host-cache-max-mib was accepted without --auto-host-cache");
    const ServeOptions host_percent = parse(
        {"ninfer-serve", "model.ninfer", "--auto-host-cache", "--host-cache-percent", "50"});
    failures += check(host_percent.context_cache.host_cache_percent == 50U &&
                          !auto_cache.context_cache.host_cache_percent,
                      "--host-cache-percent did not reach serving options or has a default");
    for (const char* bad : {"0", "101", "x"}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--auto-host-cache",
                         "--host-cache-percent", bad});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--host-cache-percent accepted a value outside [1,100]");
    }
    bool percent_without_auto_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--host-cache-percent", "50"});
    } catch (const std::invalid_argument&) { percent_without_auto_rejected = true; }
    failures += check(percent_without_auto_rejected,
                      "--host-cache-percent was accepted without --auto-host-cache");
    // Request pause/replay still uses the Host budget when history is off, so sizing it stays valid.
    const ServeOptions auto_root_only =
        parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--auto-host-cache"});
    failures += check(auto_root_only.context_cache.auto_host_cache &&
                          !auto_root_only.context_cache.enabled,
                      "--auto-host-cache was not kept with --no-prefix-reuse");

    failures += check(defaults.exit_on_engine_failure &&
                          !parse({"ninfer-serve", "model.ninfer", "--no-exit-on-engine-failure"})
                               .exit_on_engine_failure,
                      "exit-on-engine-failure must default on and be disabled by its flag");
    const auto rejected = [&](std::vector<std::string> argv) {
        try {
            (void)parse(std::move(argv));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    // Options of the previous context-cache engine are gone; a stale launcher must fail loudly
    // rather than run with a silently different configuration.
    for (const std::vector<std::string>& removed : {
             std::vector<std::string>{"--host-kv-mib", "64"},
             std::vector<std::string>{"--host-state-slots", "8"},
             std::vector<std::string>{"--max-private-continuations", "4"},
             std::vector<std::string>{"--max-shared-prefixes", "4"},
             std::vector<std::string>{"--max-long-anchors-per-continuation", "2"},
             std::vector<std::string>{"--max-cache-markers-per-request", "4"},
             std::vector<std::string>{"--auto-long-anchors", "2"},
             std::vector<std::string>{"--auto-prefix-grid"},
             std::vector<std::string>{"--progress-anchor-tokens", "4096"},
             std::vector<std::string>{"--max-prefill-lanes", "2"},
             std::vector<std::string>{"--prefill-max-skip", "8"},
             std::vector<std::string>{"--decode-rounds-per-prefill", "4"},
             std::vector<std::string>{"--output-reservation-tokens", "4096"},
             std::vector<std::string>{"--slot-save-path", "sessions"},
             std::vector<std::string>{"--auto-save-evicted"}}) {
        std::vector<std::string> argv{"ninfer-serve", "model.ninfer"};
        argv.insert(argv.end(), removed.begin(), removed.end());
        failures += check(rejected(argv), ("removed option was accepted: " + removed[0]).c_str());
        failures += check(serve_usage_text("ninfer-serve").find(removed[0] + " ") ==
                                  std::string::npos &&
                              serve_usage_text("ninfer-serve").find(removed[0] + "]") ==
                                  std::string::npos,
                          ("serve help still lists a removed option: " + removed[0]).c_str());
    }

    // The context store is off unless a directory is named; its tuning flags need the directory.
    const ServeOptions no_store = parse({"ninfer-serve", "model.ninfer"});
    failures += check(no_store.context_store_path.empty() && !no_store.context_store_max_gib &&
                          no_store.context_store_ttl_hours == 24 * 7 &&
                          no_store.context_store_idle_seconds == 30 &&
                          no_store.context_store_restore_seconds == 120 &&
                          no_store.context_store_flush_seconds == 60,
                      "the context store was on by default or its defaults changed");
    const ServeOptions store = parse({"ninfer-serve", "model.ninfer", "--context-store", "cache",
                                      "--context-store-max-gib", "40", "--context-store-ttl-hours",
                                      "48", "--context-store-idle-seconds", "10",
                                      "--context-store-restore-seconds", "0",
                                      "--context-store-flush-seconds", "7"});
    failures += check(store.context_store_path == "cache" && store.context_store_max_gib == 40 &&
                          store.context_store_ttl_hours == 48 &&
                          store.context_store_idle_seconds == 10 &&
                          store.context_store_restore_seconds == 0 &&
                          store.context_store_flush_seconds == 7,
                      "context store options did not reach serving options");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--context-store-max-gib", "4"}),
                      "--context-store-max-gib was accepted without --context-store");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--context-store-idle-seconds", "5"}),
                      "--context-store-idle-seconds was accepted without --context-store");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--context-store", ""}),
                      "an empty --context-store was accepted");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--context-store", "cache",
                                "--context-store-max-gib", "0"}),
                      "a zero --context-store-max-gib was accepted");
    failures += check(rejected({"ninfer-serve", "model.ninfer", "--no-prefix-reuse",
                                "--context-store", "cache"}),
                      "--context-store was accepted with prefix reuse disabled");
    failures += check(serve_usage_text("ninfer-serve").find("--context-store DIR") !=
                          std::string::npos,
                      "serve help omits --context-store");
    // The S3 copy of the context store is off unless both the endpoint and the bucket are given,
    // needs the store, and normalises its key prefix.
    const ServeOptions s3 = parse({"ninfer-serve", "model.ninfer", "--context-store", "dir",
                                   "--context-store-s3-endpoint", "https://s3.example.com",
                                   "--context-store-s3-bucket", "cache", "--context-store-s3-prefix",
                                   "prod"});
    failures += check(s3.context_store_s3_endpoint == "https://s3.example.com" &&
                          s3.context_store_s3_bucket == "cache" &&
                          s3.context_store_s3_prefix == "prod/" &&
                          s3.context_store_s3_region == "us-east-1" &&
                          parse({"ninfer-serve", "model.ninfer", "--context-store", "dir"})
                              .context_store_s3_endpoint.empty(),
                      "S3 context store options were not parsed or are not off by default");
    failures += check(
        rejected({"ninfer-serve", "model.ninfer", "--context-store", "dir",
                  "--context-store-s3-endpoint", "https://s3.example.com"}) &&
            rejected({"ninfer-serve", "model.ninfer", "--context-store-s3-endpoint",
                      "https://s3.example.com", "--context-store-s3-bucket", "cache"}) &&
            rejected({"ninfer-serve", "model.ninfer", "--context-store", "dir",
                      "--context-store-s3-endpoint", "ftp://s3.example.com",
                      "--context-store-s3-bucket", "cache"}) &&
            rejected({"ninfer-serve", "model.ninfer", "--context-store", "dir",
                      "--context-store-s3-endpoint", "https://s3.example.com",
                      "--context-store-s3-bucket", "cache", "--context-store-s3-prefix", "a b"}),
        "an invalid S3 context store configuration was accepted");

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
        check(serve_usage_text("ninfer-serve").find("--host-context-mib") != std::string::npos,
              "serve help omits context-cache capacities");

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

        // --max-output-tokens bounds every budget, so /props reports it as the effective cap.
        const auto predict = [&](std::vector<std::string> extra) {
            std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
            args.insert(args.end(), extra.begin(), extra.end());
            return nlohmann::json::parse(
                       make_props(parse(args), ModelDescription{.id = "qwen", .max_model_len = 8192},
                                  presets))
                .at("default_generation_settings")
                .at("params")
                .at("n_predict");
        };
        failures += check(predict({"--max-output-tokens", "4096"}) == 4096 &&
                              predict({"--default-max-tokens", "2048", "--max-output-tokens",
                                       "4096"}) == 2048 &&
                              predict({"--default-max-tokens", "8000", "--max-output-tokens",
                                       "4096"}) == 4096,
                          "/props did not report the effective --max-output-tokens cap");
    }

    return failures == 0 ? 0 : 1;
}
