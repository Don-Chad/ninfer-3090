#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // A model split across devices: each holds its own layers' KV, so a page group costs each a
    // different amount, and the capacity is the smallest any one allows.
    ninfer::runtime::SequenceCapacityCurve split{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 20,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 100,
        .extra_ranks                          = {{.minimum_device_reservation_bytes     = 500,
                                                  .bytes_per_additional_main_page_group = 300}},
    };
    const std::size_t split_available[] = {1500};
    // Device 0 alone would allow 11 pages; device 1 allows 2 + (1500 - 50 - 500) / 300 = 5.
    const auto both = ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50),
                                                           split, 2000, split_available);
    failures += check(both.main_page_groups == 5 && both.binding_rank == 1 &&
                          both.extra_rank_reservation_bytes.size() == 1 &&
                          both.extra_rank_reservation_bytes[0] == 500 + 3 * 300,
                      "the capacity was not limited by the tighter device");
    failures += check(split.extra_rank_reservation_bytes(0, 5) == 1400,
                      "a further device's reservation does not follow its own curve");

    // A device holding no KV does not limit the count, so the primary device decides.
    split.extra_ranks[0].bytes_per_additional_main_page_group = 0;
    const auto free_rank = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), split, 2000, split_available);
    failures += check(free_rank.main_page_groups == 11 && free_rank.binding_rank == 0,
                      "a device without KV constrained the capacity");
    split.extra_ranks[0].bytes_per_additional_main_page_group = 300;

    // A failure names the device that could not hold the plan.
    const auto names_device = [&](auto&& call, const char* device) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what()).find(device) != std::string::npos;
        }
        return false;
    };
    const std::size_t starved[] = {520};
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000, starved);
                          },
                          "device 1"),
                      "a device below its minimum reservation was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::explicit_capacity(64 * 19), split, 100000,
                                  split_available);
                          },
                          "device 1"),
                      "an explicit capacity too large for one device was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000);
                          },
                          "every device"),
                      "missing free memory for a device was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
