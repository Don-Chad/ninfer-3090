#include "serve/load_report.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    ninfer::EngineOptions engine;
    engine.max_concurrency                  = 4;
    engine.max_pending_requests             = 32;
    engine.context_cache.device_state_slots = 4;

    ninfer::MemorySummary memory;
    memory.max_context               = 65536;
    memory.kv_capacity               = 131072; // 2048 page groups of 64 tokens
    memory.kv_capacity_page_groups   = 2048;
    memory.host_context_capacity_bytes = 16ULL << 30;

    const LoadCapacity capacity = make_load_capacity("qwen3.8-27b", engine, memory);
    failures += check(capacity.device_state_slots == 8,
                      "device StateImage capacity is lane slots plus extra checkpoint slots");

    LoadSample sample;
    sample.uptime_seconds                      = 12.5;
    sample.admitted_requests                   = 6;
    sample.stats.running_requests              = 4;
    sample.stats.prefilling_requests           = 1;
    sample.stats.decode_ready_requests         = 3;
    sample.stats.waiting_requests              = 2;
    sample.stats.paused_requests               = 1;
    sample.stats.replaying_requests            = 1;
    sample.stats.materializing_requests        = 0;
    sample.stats.device_main_kv_occupied_pages = 100;
    sample.stats.device_state_occupied_slots   = 5;
    sample.stats.host_state_occupied_slots     = 7;
    sample.stats.host_kv_occupied_bytes        = 1024;
    sample.stats.host_context_occupied_bytes   = 4096;
    sample.stats.computed_prefill_tokens       = std::numeric_limits<std::uint64_t>::max() - 1;
    sample.stats.committed_decode_tokens       = 987654321;
    sample.stats.reused_prompt_tokens          = 5555;
    sample.stats.decode_rounds                 = 42;
    sample.stats.decode_row_rounds             = 160;
    sample.stats.host_work.program_submit_ns   = 1'000'000'000;
    sample.stats.host_work.engine_boundary_ns  = 500'000'000;
    sample.stats.host_work.device_wait_ns      = 250'000'000;

    const Json report = Json::parse(make_load_report(capacity, sample));
    failures += check(report.at("object") == "ninfer.load", "object tag");
    failures += check(report.at("model") == "qwen3.8-27b", "public model id");
    failures += check(report.at("uptime_seconds") == 12.5, "uptime");

    const Json& cap = report.at("capacity");
    failures += check(cap.at("max_concurrency") == 4 && cap.at("max_pending_requests") == 32,
                      "concurrency limits");
    failures += check(cap.at("max_admitted_requests") == 36,
                      "admission limit is max_concurrency + max_pending_requests");
    failures += check(cap.at("max_context") == 65536, "max_context");
    failures += check(cap.at("kv_capacity_tokens") == 131072 && cap.at("kv_capacity_pages") == 2048,
                      "KV capacity");
    failures += check(cap.at("kv_page_tokens") == 64, "KV page size derives from capacity");
    failures += check(cap.at("device_state_slots") == 8 &&
                          cap.at("host_context_bytes") == (16ULL << 30),
                      "checkpoint capacities");

    const Json& requests = report.at("requests");
    failures += check(requests.at("admitted") == 6 && requests.at("running") == 4 &&
                          requests.at("prefilling") == 1 && requests.at("decode_ready") == 3 &&
                          requests.at("waiting") == 2 && requests.at("paused") == 1 &&
                          requests.at("replaying") == 1 && requests.at("materializing") == 0,
                      "request gauges");

    const Json& occupancy = report.at("occupancy");
    failures += check(occupancy.at("device_main_kv_pages") == 100 &&
                          occupancy.at("device_main_kv_tokens") == 6400,
                      "KV occupancy in pages and tokens");
    failures +=
        check(occupancy.at("device_state_slots") == 5 && occupancy.at("host_state_slots") == 7 &&
                  occupancy.at("host_kv_bytes") == 1024 &&
                  occupancy.at("host_context_bytes") == 4096,
              "checkpoint occupancy");

    const Json& counters = report.at("counters");
    failures += check(counters.at("computed_prefill_tokens").get<std::uint64_t>() ==
                          std::numeric_limits<std::uint64_t>::max() - 1,
                      "64-bit counters survive JSON rendering");
    failures +=
        check(counters.at("committed_decode_tokens") == 987654321 &&
                  counters.at("reused_prompt_tokens") == 5555 &&
                  counters.at("decode_rounds") == 42 && counters.at("decode_row_rounds") == 160,
              "decode counters");

    failures += check(report.at("host").at("active_seconds") == 1.5 &&
                          report.at("host").at("device_wait_seconds") == 0.25,
                      "host time splits active work from device wait");
    failures += check(report.at("last_request").is_null(),
                      "last_request is null before any request finished");
    sample.last_generation = GenerationPace{24, 6.9, 6.3};
    const Json paced       = Json::parse(make_load_report(capacity, sample));
    const Json& last       = paced.at("last_request");
    failures += check(last.at("completion_tokens") == 24 &&
                          last.at("generation_wall_seconds") == 6.9 &&
                          std::abs(last.at("inter_token_seconds").get<double>() - 0.3) < 1e-9 &&
                          last.at("decode_host_seconds") == 6.3,
                      "last_request reports the token pace");

    // A capacity without resolved pages (never expected after attach) must not divide by zero.
    LoadCapacity unresolved      = capacity;
    unresolved.kv_capacity_pages = 0;
    const Json degenerate        = Json::parse(make_load_report(unresolved, sample));
    failures += check(degenerate.at("capacity").at("kv_page_tokens") == 0 &&
                          degenerate.at("occupancy").at("device_main_kv_tokens") == 0,
                      "zero page groups renders zero page size");

    if (failures != 0) { std::cerr << failures << " load report check(s) failed\n"; }
    return failures == 0 ? 0 : 1;
}
