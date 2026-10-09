// A KV page pool whose planes live on several ranks.
//
// The contract is that spreading planes over ranks changes where the bytes sit and which stream
// moves them, and nothing else: the page-group allocator, the handles it mints and the host image of
// a page must be identical to the same pool with every plane in one backing. So the oracle here is
// the single-rank pool itself. Both pools get the same operation script, and every observable is
// compared.
//
// Ranks share one card in this test (two backings on device 0, two streams), which exercises the
// per-plane stream selection and the split bookkeeping but not a wrong-device pointer; that is
// checked on real hardware.

#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

template <class Fn>
int expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "expectation failed: " << label << " (nothing was thrown)\n";
    return 1;
}

constexpr std::uint32_t kPages = 12;

ninfer::KVPageGeometry make_geometry(ninfer::PagedKVPlaneOrder order) {
    ninfer::KVPageGeometry geometry{
        .device_plane_order = order,
        .planes             = {{ninfer::DType::I8, 8, 2, 256},
                               {ninfer::DType::FP16, 1, 2, 256},
                               {ninfer::DType::I8, 8, 2, 256},
                               {ninfer::DType::FP16, 1, 2, 256}},
    };
    return geometry;
}

// Alternating planes go to alternating ranks, so both ranks own planes of every kind.
std::vector<std::size_t> alternating_ranks(std::size_t planes, std::size_t ranks) {
    std::vector<std::size_t> out(planes);
    for (std::size_t plane = 0; plane < planes; ++plane) { out[plane] = plane % ranks; }
    return out;
}

// A pool plus the arenas that back it, so the two travel together.
struct Cache {
    std::vector<ninfer::DeviceArena> arenas;
    std::optional<ninfer::DeviceKVPagePool> pool;
    // One block-table copy per rank, like the planes.
    std::optional<ninfer::KVExecutionTablePool> tables;

    Cache() = default;
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
};

void build_cache(Cache& cache, const ninfer::KVPageGeometry& geometry, std::size_t ranks) {
    const ninfer::DeviceKVPagePoolSpec spec{.page_group_count = kPages, .geometry = geometry};
    const std::vector<std::size_t> plane_rank = alternating_ranks(geometry.planes.size(), ranks);
    std::vector<ninfer::LayoutBuilder> builders(ranks);
    std::vector<ninfer::LayoutBuilder*> pointers;
    for (auto& builder : builders) { pointers.push_back(&builder); }
    const ninfer::DeviceKVPagePoolLayout layout =
        ninfer::plan_device_kv_page_pool(pointers, plane_rank, spec);
    std::vector<ninfer::KVExecutionTableLayout> table_layouts;
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        table_layouts.push_back(ninfer::plan_kv_execution_tables(
            *pointers[rank], {.logical_page_capacity = kPages, .table_rows = 2}, rank));
    }
    std::vector<ninfer::DeviceSpan> backings;
    cache.arenas.reserve(ranks);
    for (auto& builder : builders) {
        cache.arenas.emplace_back(builder.finish(256));
        backings.push_back({cache.arenas.back().base(), cache.arenas.back().capacity()});
    }
    cache.pool.emplace(backings, layout);
    cache.tables.emplace(backings, table_layouts, *cache.pool);
}

std::vector<std::int32_t> read_row(const ninfer::Tensor& row, std::size_t count) {
    std::vector<std::int32_t> out(count);
    CUDA_CHECK(cudaMemcpy(out.data(), row.data, out.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return out;
}

std::vector<ninfer::DeviceKVPageLease> materialize(ninfer::DeviceKVPagePool& pool,
                                                   std::uint32_t pages) {
    std::optional<ninfer::DeviceKVPageReservation> reservation = pool.reserve(pages);
    if (!reservation) { throw std::bad_alloc(); }
    std::vector<ninfer::DeviceKVPageLease> out;
    // The pool materializes into capacity the caller has already reserved.
    out.reserve(pages);
    pool.materialize(*reservation, pages, out);
    return out;
}

std::vector<ninfer::DeviceKVPageHandle> handles(std::span<const ninfer::DeviceKVPageLease> pages) {
    std::vector<ninfer::DeviceKVPageHandle> out;
    for (const ninfer::DeviceKVPageLease& page : pages) { out.push_back(page.handle()); }
    return out;
}

// The same bytes for the same plane index, whichever rank the plane sits on.
void fill(ninfer::DeviceKVPagePool& pool, ninfer::RankStreams streams) {
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        std::vector<unsigned char> host(plane.bytes());
        for (std::size_t index = 0; index < host.size(); ++index) {
            host[index] =
                static_cast<unsigned char>((index * 29U + plane_index * 61U + 17U) & 0xffU);
        }
        CUDA_CHECK(cudaMemcpyAsync(plane.data, host.data(), host.size(), cudaMemcpyHostToDevice,
                                   streams[pool.plane_rank(plane_index)]));
        // The host vector dies at the end of this iteration; the copy must be done with it first.
        CUDA_CHECK(cudaStreamSynchronize(streams[pool.plane_rank(plane_index)]));
    }
}

bool same_free_runs(const ninfer::DeviceKVPagePool& a, const ninfer::DeviceKVPagePool& b) {
    const auto left  = a.free_runs();
    const auto right = b.free_runs();
    if (left.size() != right.size()) { return false; }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i].begin != right[i].begin || left[i].count != right[i].count) { return false; }
    }
    return true;
}

bool same_counts(const ninfer::DeviceKVPagePool& a, const ninfer::DeviceKVPagePool& b) {
    return a.allocated_pages() == b.allocated_pages() && a.reserved_pages() == b.reserved_pages() &&
           a.available_pages() == b.available_pages() && a.capacity_pages() == b.capacity_pages();
}

// One host image of `pages` from `pool`, prefilled with a marker so bytes the transfer never writes
// stay recognisable.
std::vector<unsigned char> read_image(const ninfer::DeviceKVPagePool& pool,
                                      std::span<const ninfer::DeviceKVPageHandle> pages,
                                      ninfer::RankStreams streams,
                                      const ninfer::DeviceContext& context) {
    const ninfer::HostKVPageLayout host_layout = ninfer::plan_host_kv_page_layout(pool.geometry());
    const ninfer::HostKVPageLayout layouts[]   = {host_layout};
    ninfer::HostContextArena backing(host_layout.page_stride * (kPages + 2), host_layout.page_stride);
    ninfer::HostKVArena arena(backing, std::span<const ninfer::HostKVPageLayout>(layouts));
    std::optional<ninfer::HostKVAllocation> allocation =
        arena.allocate(host_layout, static_cast<std::uint32_t>(pages.size()));
    if (!allocation) { throw std::runtime_error("host allocation failed"); }
    ninfer::HostKVAllocationView view = arena.writable_view(*allocation);
    const std::size_t bytes           = host_layout.page_stride * view.page_count();
    std::memset(view.data(), 0xAB, bytes);
    pool.copy_to_host(pages, view, streams);
    context.synchronize();
    const auto* begin = reinterpret_cast<const unsigned char*>(view.data());
    return std::vector<unsigned char>(begin, begin + bytes);
}

int exercise(const ninfer::DeviceContext& context, ninfer::PagedKVPlaneOrder order,
             const std::string& label) {
    int failures                    = 0;
    const ninfer::KVPageGeometry g = make_geometry(order);
    Cache reference_cache, ranked_cache;
    build_cache(reference_cache, g, 1);
    build_cache(ranked_cache, g, 2);
    ninfer::DeviceKVPagePool& reference = *reference_cache.pool;
    ninfer::DeviceKVPagePool& ranked    = *ranked_cache.pool;
    const ninfer::RankStreams one_stream(context.rank(0).stream);
    const ninfer::RankStreams both_streams = ninfer::RankStreams::compute(context);

    failures += expect(reference.rank_count() == 1 && ranked.rank_count() == 2,
                       label + ": the pools span the ranks they were given");
    bool ranks_ok = true;
    for (std::size_t plane = 0; plane < ranked.plane_count(); ++plane) {
        ranks_ok = ranks_ok && ranked.plane_rank(plane) == plane % 2 && reference.plane_rank(plane) == 0;
    }
    failures += expect(ranks_ok, label + ": each plane reports the rank that holds it");

    // 1. The allocator is rank-agnostic: the same script leaves both pools in the same state.
    std::vector<ninfer::DeviceKVPageLease> ref_prefix   = materialize(reference, 3);
    std::vector<ninfer::DeviceKVPageLease> ref_blockers = materialize(reference, 3);
    std::vector<ninfer::DeviceKVPageLease> rk_prefix    = materialize(ranked, 3);
    std::vector<ninfer::DeviceKVPageLease> rk_blockers  = materialize(ranked, 3);
    ref_prefix.clear();
    rk_prefix.clear();
    std::vector<ninfer::DeviceKVPageLease> ref_pages = materialize(reference, 5);
    std::vector<ninfer::DeviceKVPageLease> rk_pages  = materialize(ranked, 5);
    failures += expect(same_free_runs(reference, ranked), label + ": free runs match after the script");
    failures += expect(same_counts(reference, ranked), label + ": page counts match after the script");
    const auto ref_handles = handles(ref_pages);
    const auto rk_handles  = handles(rk_pages);
    failures += expect(reference.contiguous_run_count(ref_handles) ==
                           ranked.contiguous_run_count(rk_handles),
                       label + ": both allocators fragment identically");

    // 1b. Every rank that runs attention holds its own copy of the block table, and publishing
    //     writes them all: each copy must read back the mapping a single-rank table holds.
    {
        ninfer::KVExecutionTablePool& ref_tables = *reference_cache.tables;
        ninfer::KVExecutionTablePool& rk_tables  = *ranked_cache.tables;
        failures += expect(ref_tables.replica_ranks().size() == 1 && rk_tables.replica_ranks().size() == 2,
                           label + ": a table copy per rank");
        ninfer::KVExecutionRowLease ref_row = ref_tables.acquire(1);
        ninfer::KVExecutionRowLease rk_row  = rk_tables.acquire(1);
        ref_tables.publish(ref_row.handle(), 0, ref_handles, one_stream);
        rk_tables.publish(rk_row.handle(), 0, rk_handles, both_streams);
        context.synchronize();
        const auto want = read_row(ref_tables.row(ref_row.handle()), ref_handles.size());
        for (const std::size_t rank : {std::size_t{0}, std::size_t{1}}) {
            failures += expect(read_row(rk_tables.row(rk_row.handle(), rank), rk_handles.size()) == want,
                               label + ": rank " + std::to_string(rank) +
                                   "'s table copy holds the mapping");
        }
        failures += expect_throws([&] { (void)ref_tables.matrix(1); },
                                  label + ": a rank with no copy is rejected");
        // A missing stream fails the whole publication before any copy is written.
        failures += expect_throws(
            [&] { rk_tables.publish(rk_row.handle(), 0, rk_handles, context.rank(0).stream); },
            label + ": one stream for a two-copy table is rejected");
    }

    // 2. The same bytes go in, so the same host image must come out, however the planes are split.
    fill(reference, one_stream);
    fill(ranked, both_streams);
    const auto ref_image = read_image(reference, ref_handles, one_stream, context);
    const auto rk_image  = read_image(ranked, rk_handles, both_streams, context);
    failures += expect(ref_image == rk_image, label + ": D2H host image is identical to single rank");
    bool marker_only = true;
    for (unsigned char byte : ref_image) { marker_only = marker_only && byte == 0xAB; }
    failures += expect(!marker_only, label + ": the transfer actually wrote payload");

    // 3. copy_page and zero_pages act on every plane, on whichever rank holds it.
    reference.copy_page(ref_handles[0], ref_handles[4], one_stream);
    ranked.copy_page(rk_handles[0], rk_handles[4], both_streams);
    reference.zero_pages(std::span(ref_handles).subspan(1, 2), one_stream);
    ranked.zero_pages(std::span(rk_handles).subspan(1, 2), both_streams);
    context.synchronize();
    const auto ref_after = read_image(reference, ref_handles, one_stream, context);
    const auto rk_after  = read_image(ranked, rk_handles, both_streams, context);
    failures += expect(ref_after == rk_after, label + ": copy and zero leave identical images");
    failures += expect(ref_after != ref_image, label + ": copy and zero changed something");

    // 4. A round trip through the host restores a ranked pool page for page.
    Cache other_cache;
    build_cache(other_cache, g, 2);
    ninfer::DeviceKVPagePool& other        = *other_cache.pool;
    std::vector<ninfer::DeviceKVPageLease> restored = materialize(other, 5);
    const auto restored_handles                     = handles(restored);
    {
        const ninfer::HostKVPageLayout host_layout = ninfer::plan_host_kv_page_layout(g);
        const ninfer::HostKVPageLayout layouts[]   = {host_layout};
        ninfer::HostContextArena backing(host_layout.page_stride * (kPages + 2),
                                         host_layout.page_stride);
        ninfer::HostKVArena arena(backing, std::span<const ninfer::HostKVPageLayout>(layouts));
        auto allocation = arena.allocate(host_layout, static_cast<std::uint32_t>(rk_handles.size()));
        ninfer::HostKVAllocationView view = arena.writable_view(*allocation);
        ranked.copy_to_host(rk_handles, view, both_streams);
        context.synchronize();
        other.zero_pages(restored_handles, both_streams);
        other.copy_from_host(arena.view(*allocation), restored_handles, both_streams);
        context.synchronize();
    }
    const auto restored_image = read_image(other, restored_handles, both_streams, context);
    failures += expect(restored_image == rk_after,
                       label + ": a host round trip restores the pool byte for byte");

    // 5. Misuse is refused rather than putting a rank's copy on another rank's stream.
    failures += expect_throws([&] { ranked.zero_pages(rk_handles, context.rank(0).stream); },
                              label + ": one stream for a two-rank pool is rejected");
    failures += expect_throws([&] { (void)ranked.plane_page_range(0, 0, 1); },
                              label + ": page ranges are refused on a multi-rank pool");
    if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
        failures += expect(reference.plane_page_range(0, 0, 1).bytes != 0,
                           label + ": a single-rank pool still serves page ranges");
    }
    return failures;
}

int exercise_layout_errors() {
    int failures = 0;
    const ninfer::KVPageGeometry g = make_geometry(ninfer::PagedKVPlaneOrder::PageMajor);
    const ninfer::DeviceKVPagePoolSpec spec{.page_group_count = kPages, .geometry = g};
    ninfer::LayoutBuilder a, b;
    ninfer::LayoutBuilder* const builders[] = {&a, &b};
    failures += expect_throws(
        [&] {
            const std::vector<std::size_t> short_map(g.planes.size() - 1, 0);
            (void)ninfer::plan_device_kv_page_pool(builders, short_map, spec);
        },
        "a rank map that misses a plane is rejected");
    failures += expect_throws(
        [&] {
            const std::vector<std::size_t> off_end(g.planes.size(), 2);
            (void)ninfer::plan_device_kv_page_pool(builders, off_end, spec);
        },
        "a plane on a rank with no builder is rejected");

    // A layout that names rank 1 needs two backings.
    Cache cache;
    build_cache(cache, g, 2);
    ninfer::LayoutBuilder c, d;
    ninfer::LayoutBuilder* const two[] = {&c, &d};
    const auto layout = ninfer::plan_device_kv_page_pool(two, alternating_ranks(g.planes.size(), 2), spec);
    const std::size_t bytes = c.finish(256);
    (void)d.finish(256);
    ninfer::DeviceArena only(bytes);
    failures += expect_throws(
        [&] {
            ninfer::DeviceKVPagePool bad(ninfer::DeviceSpan{only.base(), only.capacity()}, layout);
        },
        "a pool with fewer backings than ranks is rejected");
    return failures;
}

int run() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);
    CUDA_CHECK(cudaSetDevice(0));

    const int device_ids[2] = {0, 0};
    ninfer::DeviceContext context(std::span<const int>(device_ids, 2));

    int failures = 0;
    failures += exercise(context, ninfer::PagedKVPlaneOrder::PageMajor, "page-major");
    failures += exercise(context, ninfer::PagedKVPlaneOrder::HeadMajor, "head-major");
    failures += exercise_layout_errors();
    if (failures != 0) { return 1; }
    std::cout << "multi-rank kv pool matches the single-rank pool\n";
    return 0;
}

} // namespace

int run_kv_cache_ranks_test() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
