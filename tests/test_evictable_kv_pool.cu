// Lifetime and integrity qualification for the VMM-backed KV arena: stable home addresses across
// leases, granules handed out contiguously at the overlay range, resident bytes untouched by a
// lease, and rejection of invalid leases.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_kv_pool.h"
#include "core/kv_loan.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <optional>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

std::byte* at(const ninfer::DeviceSpan& arena, std::size_t offset) {
    return static_cast<std::byte*>(arena.data) + offset;
}

void fill(void* destination, std::byte value, std::size_t bytes) {
    CUDA_CHECK(cudaMemset(destination, static_cast<int>(value), bytes));
}

std::byte read_one(const void* source) {
    std::byte value{};
    CUDA_CHECK(cudaMemcpy(&value, source, 1, cudaMemcpyDeviceToHost));
    return value;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        if (!ninfer::EvictableKVPool::supported(device)) {
            std::cout << "SKIP: device does not support CUDA virtual memory management\n";
            return 77;
        }

        int failures = 0;
        // Sized in granules once the pool reports the device granularity: build a throwaway pool
        // with a one-granule window to learn it, then the real fixture.
        std::size_t granule = 0;
        {
            ninfer::EvictableKVPool probe(device, ninfer::EvictableKVPool::Config{
                                                      .arena_bytes           = 8ULL << 20,
                                                      .lendable_prefix_bytes = 8ULL << 20,
                                                      .window_capacity_bytes = 1,
                                                  });
            granule = probe.granularity();
            failures += expect(granule != 0 && (granule & (granule - 1)) == 0,
                               "granularity is a nonzero power of two");
        }

        // Six granules of payload plus a deliberately unaligned resident remainder.
        const std::size_t payload_bytes = 6 * granule;
        const std::size_t arena_bytes   = payload_bytes + granule / 2 + 4096;
        ninfer::EvictableKVPool pool(device, ninfer::EvictableKVPool::Config{
                                                 .arena_bytes           = arena_bytes,
                                                 .lendable_prefix_bytes = payload_bytes,
                                                 .window_capacity_bytes = 3 * granule,
                                             });

        const ninfer::DeviceSpan arena = pool.arena();
        void* const stable_base        = arena.data;
        failures += expect(arena.bytes == arena_bytes, "arena span covers the configured bytes");
        failures += expect(reinterpret_cast<std::uintptr_t>(arena.data) % granule == 0,
                           "arena base is granularity aligned");
        failures += expect(pool.lendable_granules() == 6, "the payload prefix is six granules");
        failures += expect(pool.window_capacity_bytes() == 3 * granule,
                           "window capacity is reported in whole granules");
        failures += expect(pool.granule_of(0) == 0 && pool.granule_of(granule) == 1 &&
                               pool.granule_of(granule * 5 + 17) == 5,
                           "granule_of maps arena offsets to granules");
        failures += expect(pool.granule_bytes(2).data == at(arena, 2 * granule) &&
                               pool.granule_bytes(2).bytes == granule,
                           "granule_bytes describes the home range");

        // Whole arena carries a marker; the lease must leave everything outside it alone.
        constexpr std::byte kResident{0x5a};
        fill(arena.data, kResident, arena_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        {
            const std::size_t granules[] = {1, 3, 4};
            ninfer::EvictableKVPool::Transaction lease =
                pool.lease(std::span<const std::size_t>(granules), device.stream);
            failures += expect(lease.open() && pool.lease_open(), "lease opens a transaction");
            failures += expect(lease.leased().bytes == 3 * granule,
                               "the leased span covers the requested granules");
            failures += expect(lease.stats().mapped_bytes == 3 * granule,
                               "lease stats report the mapped bytes");

            // The window is usable memory: writing it must not disturb the resident granules.
            constexpr std::byte kWindow{0x27};
            fill(lease.leased().data, kWindow, lease.leased().bytes);
            CUDA_CHECK(cudaDeviceSynchronize());
            failures += expect(read_one(at(arena, 0)) == kResident,
                               "granule 0 is untouched while granule 1 is lent");
            failures += expect(read_one(at(arena, 2 * granule)) == kResident,
                               "granule 2 is untouched while granules 3 and 4 are lent");
            failures += expect(read_one(at(arena, 5 * granule)) == kResident,
                               "the last payload granule is untouched");
            failures += expect(read_one(at(arena, payload_bytes + 1024)) == kResident,
                               "the resident remainder is untouched");

            bool nested_rejected = false;
            const std::size_t other[] = {0};
            try {
                auto second = pool.lease(std::span<const std::size_t>(other), device.stream);
                (void)second;
            } catch (const std::logic_error&) { nested_rejected = true; }
            failures += expect(nested_rejected, "a nested lease is rejected");

            lease.close();
            failures += expect(!lease.open() && !pool.lease_open(),
                               "close returns the granules home");
            failures += expect(!pool.poisoned(), "a clean close does not poison the pool");
            failures += expect(pool.arena().data == stable_base, "the home base never moves");
        }

        // The pool is usable again and the previously lent granules are mapped, if garbage.
        {
            const std::size_t granules[] = {1};
            ninfer::EvictableKVPool::Transaction lease =
                pool.lease(std::span<const std::size_t>(granules), device.stream);
            failures += expect(lease.leased().bytes == granule, "a one-granule lease");
        }
        failures += expect(!pool.lease_open(), "the destructor closes the lease");
        fill(at(arena, granule), kResident, granule);
        CUDA_CHECK(cudaDeviceSynchronize());
        failures += expect(read_one(at(arena, granule)) == kResident,
                           "a returned granule is writable at its home address");

        const auto rejects = [&](const std::vector<std::size_t>& granules, const char* label) {
            bool rejected = false;
            try {
                auto lease = pool.lease(std::span<const std::size_t>(granules), device.stream);
                (void)lease;
            } catch (const std::exception&) { rejected = true; }
            failures += expect(rejected, label);
            failures += expect(!pool.lease_open(), "a rejected lease leaves the arena resident");
        };
        rejects({}, "an empty lease is rejected");
        rejects({6}, "a resident granule cannot be lent");
        rejects({0, 1, 2, 3}, "a lease beyond the window capacity is rejected");
        rejects({2, 2}, "duplicate granules are rejected");
        rejects({3, 1}, "unordered granules are rejected");

        bool window_rejected = false;
        try {
            ninfer::EvictableKVPool wide(device, ninfer::EvictableKVPool::Config{
                                                     .arena_bytes           = arena_bytes,
                                                     .lendable_prefix_bytes = 2 * granule,
                                                     .window_capacity_bytes = 3 * granule,
                                                 });
        } catch (const std::invalid_argument&) { window_rejected = true; }
        failures += expect(window_rejected, "a window wider than the payload prefix is rejected");

        // Loan planning over a page pool bound inside the arena: a K-sized plane whose stride
        // divides the granularity is lendable, a scale-sized plane is not.
        {
            const auto page_stride = static_cast<std::int32_t>(granule / 8 / 64 / 2);
            ninfer::KVPageGeometry geometry{
                .planes = {{ninfer::DType::I8, page_stride, 2, 256},
                           {ninfer::DType::FP16, 1, 2, 256}},
            };
            ninfer::LayoutBuilder builder;
            const ninfer::DeviceKVPagePoolLayout layout = ninfer::plan_device_kv_page_pool(
                builder, {.page_group_count = 512, .geometry = geometry});
            const std::size_t needed = builder.finish(256, "loan fixture");
            ninfer::EvictableKVPool fixture(device, ninfer::EvictableKVPool::Config{
                                                        .arena_bytes           = needed,
                                                        .lendable_prefix_bytes = needed,
                                                        .window_capacity_bytes = 4 * granule,
                                                    });
            ninfer::DeviceKVPagePool pages(fixture.arena(), layout);
            const std::uint32_t unit = ninfer::kv_loan_unit_pages(pages, granule);
            failures += expect(unit == granule / static_cast<std::size_t>(pages.plane(0).nb[3]),
                               "the loan unit follows the widest lendable plane");

            const ninfer::KVLoanPlan plan = ninfer::plan_kv_loan(fixture, pages, 2 * granule);
            failures += expect(plan.bytes >= 2 * granule, "the plan covers the request");
            failures += expect(!plan.runs.empty(), "the plan lends pages");
            std::size_t previous = 0;
            bool ordered         = true;
            for (const std::size_t index : plan.granules) {
                if (previous != 0 && index <= previous) { ordered = false; }
                previous = index;
            }
            failures += expect(ordered, "planned granules strictly increase");

            // A live page at the top must push the plan away from the granules it backs.
            std::vector<ninfer::DeviceKVPageLease> live;
            live.reserve(512);
            std::optional<ninfer::DeviceKVPageReservation> reservation = pages.reserve(512);
            pages.materialize(*reservation, 512, live);
            const ninfer::KVLoanPlan starved = ninfer::plan_kv_loan(fixture, pages, granule);
            failures += expect(starved.granules.empty(),
                               "a full pool cannot fund a loan");
            live.clear();
            reservation.reset();

            const ninfer::KVLoanPlan huge =
                ninfer::plan_kv_loan(fixture, pages, 64ULL * granule);
            failures += expect(huge.granules.empty(), "a request beyond the pool is refused");
        }

        if (failures != 0) {
            std::cerr << failures << " expectation(s) failed\n";
            return 1;
        }
        std::cout << "evictable KV pool test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "evictable KV pool test failed: " << error.what() << '\n';
        return 1;
    }
}
