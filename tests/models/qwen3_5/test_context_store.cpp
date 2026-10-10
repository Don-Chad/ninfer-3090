#include "core/device.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_address_space.h"
#include "models/qwen3_5/program/storage/state_store.h"

#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36   = ninfer::models::qwen3_5;
namespace store = ninfer::models::qwen3_5::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::vector<std::int32_t> read_block_table(const ninfer::KVExecutionTablePool& tables,
                                           std::int32_t row, std::size_t count) {
    const ninfer::Tensor source = tables.matrix().slice(1, row, 1).view(
        {static_cast<std::int32_t>(tables.logical_page_capacity())});
    std::vector<std::int32_t> values(count);
    CUDA_CHECK(cudaMemcpy(values.data(), source.data, values.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return values;
}

void test_state_store(ninfer::DeviceContext& device) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 1,
                .conv_channels  = 8,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 4,
                .slot_count     = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 8,
        .dflash_local =
            q36::DFlashLocalStateSpec{.layers = 1, .capacity = 8, .kv_heads = 2, .head_dim = 4},
    };
    ninfer::LayoutBuilder builder;
    const q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    ninfer::DeviceArena arena(builder.finish(256));
    q36::StateImageDevicePool physical({arena.base(), arena.capacity()}, layout);
    ninfer::HostContextArena host_backing(layout.host.image_bytes * 2, layout.host.image_bytes);
    q36::HostStatePool host(host_backing, layout.host);
    store::StateImageStore images(
        physical, &host, static_cast<std::uint32_t>(physical.slot_count()) + host.capacity());

    const auto source = images.reserve_reset(device.stream);
    expect(source.has_value(), "state source allocation");
    const std::int32_t original_slot = images.physical_slot(*source);
    images.freeze(*source);
    images.move_checkpoint_to_active(*source);
    expect(images.physical_slot(*source) == original_slot &&
               images.role(*source) == store::StateImageRole::ActiveMutable,
           "private Move preserves the logical image and physical slot");
    images.freeze(*source);
    const auto destination = images.reserve_destination();
    expect(destination.has_value(), "state destination reservation");
    const store::StateImageSelectors selectors = images.begin_fork(*source, *destination);
    expect(selectors.source >= 0 && selectors.destination >= 0 &&
               selectors.source != selectors.destination,
           "fork resolves distinct physical selectors");
    expect(!images.release(*source) && !images.release(*destination),
           "fork pins both logical images");
    const auto concurrent_destination = images.reserve_destination();
    expect(concurrent_destination.has_value(), "concurrent fork destination reservation");
    (void)images.begin_fork(*source, *concurrent_destination);
    expect(images.source_pins(*source) == 2 && !images.release(*source),
           "independent forks each retain the source");
    images.abort_fork(*source, *concurrent_destination);
    expect(images.release(*concurrent_destination), "concurrent fork destination release");
    expect(images.source_pins(*source) == 1 && !images.release(*source) &&
               !images.release(*destination),
           "aborting one reader preserves the remaining fork's pins");
    images.commit_fork(*source, *destination);
    expect(images.role(*source) == store::StateImageRole::CheckpointImmutable &&
               images.role(*destination) == store::StateImageRole::ActiveMutable,
           "fork commit preserves source and activates destination");
    images.retain_checkpoint_reference(*source);
    expect(!images.release(*source), "checkpoint owner keeps the source alive after fork commit");
    expect(images.release_checkpoint_owner(*source) && images.release(*destination) &&
               images.occupied() == 0,
           "checkpoint and active state ownership close after release");
    expect(!images.valid(*source), "released state generation becomes stale");

    const auto reused = images.reserve_reset(device.stream);
    expect(reused.has_value() && *reused != *source, "state descriptor reuse advances generation");
    expect(images.release(*reused), "reused state image releases");

    for (const bool cancel_last_reader : {false, true}) {
        const auto shared       = images.reserve_reset(device.stream);
        const auto first_reader = images.reserve_destination();
        const auto last_reader  = images.reserve_destination();
        expect(shared && first_reader && last_reader, "shared State reader fixture allocates");
        const auto source_slot = images.physical_slot(*shared);
        const std::array<std::uint16_t, 8> content{1, 3, 5, 7, 9, 11, 13, 15};
        const auto hidden = physical.continuation_hidden_slot(source_slot);
        CUDA_CHECK(cudaMemcpyAsync(hidden.data, content.data(), sizeof(content),
                                   cudaMemcpyHostToDevice, device.stream));
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared); // Cache checkpoint.
        images.retain_checkpoint_reference(*shared); // Paused request snapshot.
        (void)images.begin_fork(*shared, *first_reader);
        (void)images.begin_fork(*shared, *last_reader);
        expect(images.release_checkpoint_owner(*shared) &&
                   images.release_checkpoint_owner(*shared) && images.valid(*shared) &&
                   images.checkpoint_references(*shared) == 0 && images.source_pins(*shared) == 2,
               "cache eviction and paused cancellation leave execution readers alive");
        expect(!images.can_release_after_checkpoint_references(*shared, 0) &&
                   images.device_occupied() == 3,
               "retired content owner does not advertise pinned State capacity as free");
        physical.copy_slot(source_slot, images.physical_slot(*first_reader), device.stream);
        device.synchronize();
        images.commit_fork(*shared, *first_reader);
        expect(images.valid(*shared) && images.source_pins(*shared) == 1,
               "one completed reader does not release the remaining reader's State");
        physical.copy_slot(source_slot, images.physical_slot(*last_reader), device.stream);
        std::array<std::uint16_t, 8> observed{};
        const auto last_hidden =
            physical.continuation_hidden_slot(images.physical_slot(*last_reader));
        CUDA_CHECK(cudaMemcpyAsync(observed.data(), last_hidden.data, sizeof(observed),
                                   cudaMemcpyDeviceToHost, device.stream));
        device.synchronize();
        expect(observed == content, "remaining State reader retains the exact source content");
        if (cancel_last_reader) {
            images.abort_fork(*shared, *last_reader);
        } else {
            images.commit_fork(*shared, *last_reader);
        }
        expect(!images.valid(*shared) && images.device_occupied() == 2,
               "last reader retirement releases the orphaned State automatically");
        const auto replacement = images.reserve_reset(device.stream);
        expect(replacement && *replacement != *shared &&
                   images.physical_slot(*replacement) == source_slot,
               "retired State storage is reusable under a new generation");
        expect(images.release(*replacement) && images.release(*first_reader) &&
                   images.release(*last_reader) && images.occupied() == 0,
               "retired checkpoint and execution readers leave no State allocation behind");
    }

    {
        const auto shared = images.reserve_reset(device.stream);
        const auto reader = images.reserve_destination();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        (void)images.begin_fork(*shared, *reader);
        expect(images.release_checkpoint_owner(*shared), "read lease outlives original checkpoint");
        // Pausing an unwritten fork converts its read lease into a new snapshot owner.
        images.retain_checkpoint_reference(*shared);
        images.abort_fork(*shared, *reader);
        expect(images.valid(*shared) && images.checkpoint_references(*shared) == 1 &&
                   images.source_pins(*shared) == 0,
               "new snapshot ownership is installed before its read lease retires");
        expect(images.release(*reader) && images.release_checkpoint_owner(*shared) &&
                   images.occupied() == 0,
               "reowned snapshot retires without leaking its previous read lease");
    }

    for (const bool cancel_transfer : {false, true}) {
        const auto shared = images.reserve_reset(device.stream);
        device.synchronize();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        auto transfer = images.begin_device_to_host(*shared, device.transfer_stream);
        expect(transfer && images.release_checkpoint_owner(*shared) && images.valid(*shared),
               "submitted State transfer retains a retired checkpoint source");
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        if (cancel_transfer) {
            images.abort_transfer(std::move(*transfer));
        } else {
            images.publish_transfer(std::move(*transfer), true);
        }
        expect(!images.valid(*shared) && images.occupied() == 0 &&
                   host_backing.occupied_bytes() == 0,
               "last transfer lease retirement frees both State replicas and reservations");
    }

    const auto host_source = images.reserve_reset(device.stream);
    expect(host_source.has_value(), "Host state source allocation");
    device.synchronize();
    images.freeze(*host_source);
    {
        auto cancelled = images.reserve_device_to_host(*host_source);
        expect(cancelled && host_backing.reserved_bytes() == layout.host.image_bytes &&
                   host_backing.live_bytes() == 0 && images.source_pins(*host_source) == 1 &&
                   !images.release(*host_source),
               "State transfer reservation charges destination while its source is pinned");
    }
    expect(host_backing.occupied_bytes() == 0 && images.source_pins(*host_source) == 0 &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "cancelled State transfer releases its actual Host extent and retains source state");
    auto d2h = images.begin_device_to_host(*host_source, device.transfer_stream);
    expect(d2h.has_value() &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "incomplete State D2H does not publish a Host replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    images.publish_transfer(std::move(*d2h), true);
    expect(images.residency(*host_source) == store::StateReplicaResidency::Both &&
               host_backing.live_bytes() == layout.host.image_bytes &&
               host_backing.reserved_bytes() == 0,
           "State D2H backup publishes stable Both residency");

    const auto moved_device = images.reserve_logical_destination();
    expect(moved_device.has_value(), "State replica split destination allocation");
    images.split_device_replica_identity(*host_source, *moved_device);
    expect(images.residency(*host_source) == store::StateReplicaResidency::HostOnly &&
               images.residency(*moved_device) == store::StateReplicaResidency::DeviceOnly,
           "State replica identity split keeps old Host content and moves Device ownership");

    const auto fork_one = images.reserve_logical_destination();
    const auto fork_two = images.reserve_logical_destination();
    expect(fork_one && fork_two, "concurrent Host State forks reserve logical destinations");
    auto h2d_one = images.begin_host_fork(*host_source, *fork_one, device.transfer_stream);
    auto h2d_two = images.begin_host_fork(*host_source, *fork_two, device.transfer_stream);
    expect(h2d_one && h2d_two && images.source_pins(*host_source) == 2,
           "immutable Host State source supports multiple fork pins");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    images.publish_transfer(std::move(*h2d_one), true);
    images.publish_transfer(std::move(*h2d_two), true);
    expect(images.source_pins(*host_source) == 0 &&
               images.residency(*fork_one) == store::StateReplicaResidency::DeviceOnly &&
               images.residency(*fork_two) == store::StateReplicaResidency::DeviceOnly,
           "Host State forks publish independent Device destinations");
    expect(images.release(*host_source) && images.release(*moved_device) &&
               images.release(*fork_one) && images.release(*fork_two) && host.occupied() == 0,
           "State Host/Device replica ownership closes without leaked slots");
}

void test_kv_store(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 8,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 4, .table_rows = 2});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(physical_pages.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostContextArena host_backing(host_layout.page_stride * 8, host_layout.page_stride);
    ninfer::HostKVArena host_arena(host_backing, host_layouts);
    store::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages() + 8U);
    store::HostKVExtentStore extents(host_arena, 8);
    store::KVAddressSpaceStore addresses(pages, physical_tables, 4, 4);

    const auto address = addresses.create_active(3, 0, device.stream);
    expect(address.has_value(), "active KV address allocation");
    addresses.ensure_mapped_to_tokens(*address, 65, device.stream);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 2 && addresses.committed_frontier(*address) == 0,
           "physical KV append does not publish canonical coverage before commit");
    expect(read_block_table(physical_tables, 0, 2) == std::vector<std::int32_t>({0, 1}),
           "batched KV materialization publishes the exact execution mapping");
    addresses.commit_frontier(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 &&
               addresses.reserved_growth_pages(*address) == 1 &&
               addresses.committed_frontier(*address) == 65 && addresses.bound_row(*address) == 0,
           "KV address tracks mapped pages, remaining growth, frontier, and execution row");
    expect(pages.active_address_references(addresses.logical_page(*address, 0)) == 1 &&
               pages.active_address_references(addresses.logical_page(*address, 1)) == 1,
           "active KV membership is counted on each logical page");
    addresses.ensure_mapped_to_tokens(*address, 129, device.stream);
    device.synchronize();
    expect(read_block_table(physical_tables, 0, 3) == std::vector<std::int32_t>({0, 1, 2}),
           "incremental KV materialization preserves the existing mapping prefix");
    addresses.destructive_truncate(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 &&
               addresses.reserved_growth_pages(*address) == 1 &&
               addresses.committed_frontier(*address) == 65,
           "same-frontier trim releases uncommitted materialized suffix pages");
    addresses.set_checkpoint_requirement(*address, 65);
    expect(pages.occupied() == 2 && addresses.mapped_pages(*address) == 2,
           "endpoint and rewrite requirements share one ordered page mapping");
    bool checkpoint_truncate_rejected = false;
    try {
        addresses.destructive_truncate(*address, 0);
    } catch (const std::logic_error&) { checkpoint_truncate_rejected = true; }
    expect(checkpoint_truncate_rejected && addresses.mapped_pages(*address) == 2 &&
               addresses.committed_frontier(*address) == 65 && pages.occupied() == 2,
           "whole-page truncation cannot discard a protected checkpoint frontier");
    const std::uint64_t old_epoch = addresses.content_epoch(*address, 0);

    addresses.deactivate(*address);
    expect(addresses.bound_row(*address) == -1 && addresses.reserved_growth_pages(*address) == 0,
           "catalogued KV address owns no execution row or growth reservation");

    const std::array logical_pages{addresses.logical_page(*address, 0),
                                   addresses.logical_page(*address, 1)};
    expect(pages.active_address_references(logical_pages[0]) == 0 &&
               pages.active_address_references(logical_pages[1]) == 0 &&
               !addresses.has_active_reference(logical_pages[0]),
           "KV deactivation clears logical-page active references");
    addresses.set_checkpoint_requirement(*address, 32);
    expect(pages.protected_columns(logical_pages[0]) == 32 &&
               pages.protected_columns(logical_pages[1]) == 0,
           "lowered checkpoint frontier releases stale suffix protection");
    addresses.set_checkpoint_requirement(*address, 65);
    {
        auto cancelled = extents.prepare(pages, logical_pages);
        expect(cancelled && host_backing.reserved_bytes() == host_layout.page_stride * 2 &&
                   pages.source_pins(logical_pages[0]) == 1 &&
                   pages.source_pins(logical_pages[1]) == 1,
               "KV transfer reserves real Host destination while pinning its source pages");
    }
    expect(host_backing.occupied_bytes() == 0 && pages.source_pins(logical_pages[0]) == 0 &&
               pages.source_pins(logical_pages[1]) == 0 && !pages.host_resident(logical_pages[0]) &&
               !pages.host_resident(logical_pages[1]),
           "cancelled KV reservation releases destination bytes without publishing replicas");
    auto host_backup = extents.prepare(pages, logical_pages);
    expect(host_backup.has_value(), "Host KV extent reservation");
    const std::vector<ninfer::DeviceKVPageHandle> sources = extents.device_sources(*host_backup);
    physical_pages.copy_to_host(sources, extents.writable_view(*host_backup),
                                device.transfer_stream);
    expect(!pages.host_resident(logical_pages[0]) && !pages.host_resident(logical_pages[1]),
           "incomplete KV D2H does not publish Host replicas");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    auto host_extent = extents.publish(std::move(*host_backup));
    expect(pages.host_resident(logical_pages[0]) && pages.host_resident(logical_pages[1]) &&
               host_backing.reserved_bytes() == 0 &&
               host_backing.live_bytes() == host_layout.page_stride * 2,
           "KV extent publication attaches every logical Host replica");
    const std::array first_host_release{logical_pages[0]};
    expect(extents.release_page_replicas(pages, first_host_release),
           "first Host page release transaction");
    const auto retained_host_extent = pages.host_replica(logical_pages[1]).extent;
    expect(!extents.valid(host_extent) && !pages.host_resident(logical_pages[0]) &&
               extents.valid(retained_host_extent) &&
               pages.host_replica(logical_pages[1]).page_offset == 0,
           "partial Host release partitions an extent and republishes the retained run");
    const std::array second_host_release{logical_pages[1]};
    expect(extents.release_page_replicas(pages, second_host_release),
           "second Host page release transaction");
    expect(!pages.host_resident(logical_pages[1]) && host_arena.occupied_bytes() == 0,
           "partitioned Host replicas release while Device replicas survive");

    auto second_host_backup = extents.prepare(pages, logical_pages);
    expect(second_host_backup.has_value(), "second Host KV extent reservation");
    const std::vector<ninfer::DeviceKVPageHandle> second_sources =
        extents.device_sources(*second_host_backup);
    physical_pages.copy_to_host(second_sources, extents.writable_view(*second_host_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto second_host_extent = extents.publish(std::move(*second_host_backup));
    expect(pages.drop_device_replica(logical_pages[0]) &&
               pages.drop_device_replica(logical_pages[1]) && !extents.release(second_host_extent),
           "Host-only KV pages remain valid and cannot lose their last replica");
    expect(
        host_arena.can_allocate(host_layout, 6) && !host_arena.can_allocate(host_layout, 7) &&
            host_backing.live_bytes() == host_layout.page_stride * 2 &&
            host_backing.reserved_bytes() == 0,
        "referenced Host-only pages retain their actual extents without predicted release credit");
    auto restore_reservation = physical_pages.reserve(2);
    expect(restore_reservation.has_value(), "KV Host restore Device reservation");
    const std::array restore_destinations{
        pages.reserve_device_replica(logical_pages[0], *restore_reservation),
        pages.reserve_device_replica(logical_pages[1], *restore_reservation)};
    physical_pages.copy_from_host(extents.view(second_host_extent), restore_destinations,
                                  device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    pages.publish_device_replica(logical_pages[0]);
    pages.publish_device_replica(logical_pages[1]);
    expect(extents.release(second_host_extent) && host_arena.occupied_bytes() == 0,
           "KV Host restore republishes Device replicas before releasing the extent");
    auto activation = addresses.prepare_activation(*address, 1, 1);
    expect(addresses.bound_row(*address) == -1 && addresses.reserved_growth_pages(*address) == 0 &&
               physical_pages.reserved_pages() == 1,
           "prepared KV activation preserves the catalogued mapping until publication");
    addresses.commit_activation(std::move(activation), device.stream);
    expect(pages.active_address_references(logical_pages[0]) == 1 &&
               pages.active_address_references(logical_pages[1]) == 1,
           "KV reactivation republishes logical-page active references");
    // Retire the checkpoint at 65 before rewriting beyond the retained prefix at 32.
    addresses.set_checkpoint_requirement(*address, 32);
    addresses.destructive_truncate(*address, 32);
    expect(addresses.mapped_pages(*address) == 1 &&
               addresses.reserved_growth_pages(*address) == 2 &&
               addresses.committed_frontier(*address) == 32 &&
               addresses.content_epoch(*address, 0) != old_epoch,
           "destructive rewrite truncates coverage and advances content epoch");

    // A same-frontier trim must not depend on tail-page exclusivity: the tail is not mutated, and
    // it may still carry a Host replica (auto-host-cache restore) while the address is active.
    // Host backup requires an inactive address (no writer reference), so publish the replica while
    // deactivated and reactivate before growing the suffix.
    addresses.deactivate(*address);
    const std::array host_tail_page{addresses.logical_page(*address, 0)};
    auto host_tail_backup = extents.prepare(pages, host_tail_page);
    expect(host_tail_backup.has_value(), "inactive tail Host KV backup reservation");
    if (!host_tail_backup) { return; }
    physical_pages.copy_to_host(extents.device_sources(*host_tail_backup),
                                extents.writable_view(*host_tail_backup), device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    (void)extents.publish(std::move(*host_tail_backup));
    addresses.commit_activation(addresses.prepare_activation(*address, 3, 1), device.stream);
    addresses.ensure_mapped_to_tokens(*address, 65, device.stream);
    device.synchronize();
    expect(pages.host_resident(host_tail_page[0]), "active tail page carries a Host replica");
    addresses.destructive_truncate(*address, 32);
    expect(addresses.mapped_pages(*address) == 1 && addresses.committed_frontier(*address) == 32,
           "same-frontier trim releases suffix pages despite a Host replica on the tail");
    // Drop the checkpoint protection for the attempt, so only the Host replica can refuse it.
    addresses.set_checkpoint_requirement(*address, 0);
    bool host_rollback_rejected = false;
    try {
        addresses.destructive_truncate(*address, 31);
    } catch (const std::logic_error& error) {
        host_rollback_rejected =
            std::string_view(error.what()).find("host_replica=1") != std::string_view::npos;
    }
    addresses.set_checkpoint_requirement(*address, 32);
    expect(host_rollback_rejected && addresses.committed_frontier(*address) == 32,
           "real rollback of a Host-backed tail is still rejected and names the failing state");
    expect(extents.release_page_replicas(pages, host_tail_page),
           "active tail Host replica releases");
    addresses.deactivate(*address);
    const std::array final_logical_page{addresses.logical_page(*address, 0)};
    auto final_host_backup = extents.prepare(pages, final_logical_page);
    expect(final_host_backup.has_value(), "final Host KV backup reservation");
    const auto final_sources = extents.device_sources(*final_host_backup);
    physical_pages.copy_to_host(final_sources, extents.writable_view(*final_host_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    (void)extents.publish(std::move(*final_host_backup));
    expect(addresses.release(*address), "catalogued KV address releases");
    expect(extents.release_unreferenced() == host_layout.page_stride &&
               host_arena.occupied_bytes() == 0,
           "address teardown reclaims its zero-reference Host extent");
    expect(!addresses.valid(*address) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "KV release invalidates generations and closes physical ownership");

    const auto occupied_prefix = addresses.create_active(1, 0, device.stream);
    expect(occupied_prefix.has_value(), "growth rollback retained prefix allocation");
    addresses.ensure_mapped_to_tokens(*occupied_prefix, 1, device.stream);
    addresses.settle_growth(*occupied_prefix, 1);
    addresses.deactivate(*occupied_prefix);
    const auto occupied_page = addresses.logical_page(*occupied_prefix, 0);
    addresses.activate(*occupied_prefix, 0, 0, device.stream);
    expect(addresses.active(*occupied_prefix) && addresses.mapped_pages(*occupied_prefix) == 1 &&
               addresses.logical_page(*occupied_prefix, 0) == occupied_page &&
               addresses.reserved_growth_pages(*occupied_prefix) == 0 &&
               addresses.growth_pages_for_tokens(*occupied_prefix, 64) == 0 &&
               pages.writer_references(occupied_page) == 1 && physical_pages.reserved_pages() == 0,
           "zero-growth reactivation retains its existing page and restores the writer");
    addresses.deactivate(*occupied_prefix);
    const auto first_empty  = addresses.create_active(0, 0, device.stream);
    const auto second_empty = addresses.create_active(0, 1, device.stream);
    expect(first_empty && second_empty && addresses.active(*first_empty) &&
               addresses.active(*second_empty) && addresses.bound_row(*first_empty) == 0 &&
               addresses.bound_row(*second_empty) == 1 &&
               addresses.mapped_pages(*first_empty) == 0 &&
               addresses.mapped_pages(*second_empty) == 0 &&
               addresses.reserved_growth_pages(*first_empty) == 0 &&
               addresses.reserved_growth_pages(*second_empty) == 0 &&
               physical_pages.allocated_pages() == 1 && physical_pages.reserved_pages() == 0,
           "zero-growth active addresses own rows without reserving physical pages");
    addresses.ensure_mapped_to_tokens(*first_empty, 0, device.stream);
    addresses.reserve_growth(*first_empty, 4);
    bool growth_rejected = false;
    try {
        addresses.reserve_growth(*second_empty, 4);
    } catch (const std::bad_alloc&) { growth_rejected = true; }
    expect(growth_rejected && addresses.reserved_growth_pages(*first_empty) == 4 &&
               addresses.reserved_growth_pages(*second_empty) == 0 &&
               addresses.mapped_pages(*first_empty) == 0 &&
               addresses.mapped_pages(*second_empty) == 0 && pages.occupied() == 1 &&
               addresses.occupied() == 3 && physical_pages.allocated_pages() == 1 &&
               physical_pages.reserved_pages() == 4 && physical_pages.available_pages() == 3,
           "failed later-address growth preserves prior reservations without partial ownership");
    addresses.release_growth(*first_empty);
    addresses.release_growth(*second_empty);
    expect(addresses.active(*first_empty) && addresses.active(*second_empty) &&
               addresses.reserved_growth_pages(*first_empty) == 0 &&
               addresses.reserved_growth_pages(*second_empty) == 0 &&
               physical_pages.allocated_pages() == 1 && physical_pages.reserved_pages() == 0 &&
               physical_pages.available_pages() == 7,
           "multi-address growth rollback releases all reservations and preserves active rows");
    addresses.reserve_growth(*first_empty, 3);
    addresses.reserve_growth(*second_empty, 4);
    expect(addresses.reserved_growth_pages(*first_empty) == 3 &&
               addresses.reserved_growth_pages(*second_empty) == 4 &&
               physical_pages.reserved_pages() == 7 && physical_pages.available_pages() == 0,
           "growth retry can reserve all capacity returned by rollback");
    addresses.release_growth(*first_empty);
    addresses.release_growth(*second_empty);
    addresses.deactivate(*first_empty);
    addresses.deactivate(*second_empty);
    expect(addresses.release(*first_empty) && addresses.release(*second_empty) &&
               addresses.release(*occupied_prefix) && addresses.occupied() == 0 &&
               pages.occupied() == 0 && physical_pages.allocated_pages() == 0 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 8,
           "zero-growth and failed reservation fixtures close all ownership");

    // Verify crosses a page boundary, but the terminal commit consumes only its first column.
    const auto terminal = addresses.create_active(3, 0, device.stream);
    expect(terminal.has_value(), "terminal boundary KV address allocation");
    addresses.ensure_mapped_to_tokens(*terminal, 63, device.stream);
    addresses.commit_frontier(*terminal, 63);
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    device.synchronize();
    const auto terminal_mapping   = read_block_table(physical_tables, 0, 2);
    const auto unchanged_terminal = [&] {
        return addresses.mapped_pages(*terminal) == 2 &&
               addresses.committed_frontier(*terminal) == 63 &&
               addresses.reserved_growth_pages(*terminal) == 1 &&
               physical_pages.allocated_pages() == 2 && physical_pages.reserved_pages() == 1 &&
               physical_pages.available_pages() == 5 &&
               read_block_table(physical_tables, 0, 2) == terminal_mapping;
    };
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    addresses.ensure_mapped_to_tokens(*terminal, 64, device.stream);
    device.synchronize();
    expect(unchanged_terminal(), "covered KV requests preserve speculative mappings and ownership");
    bool exceeded = false;
    try {
        addresses.ensure_mapped_to_tokens(*terminal, 193, device.stream);
    } catch (const std::invalid_argument&) { exceeded = true; }
    expect(exceeded && unchanged_terminal(),
           "KV coverage beyond reserved growth fails without mutation");
    addresses.commit_frontier(*terminal, 64);
    addresses.settle_growth(*terminal, 64);
    expect(addresses.mapped_pages(*terminal) == 1 &&
               addresses.committed_frontier(*terminal) == 64 &&
               addresses.reserved_growth_pages(*terminal) == 0 &&
               physical_pages.allocated_pages() == 1 && physical_pages.reserved_pages() == 0 &&
               physical_pages.available_pages() == 7 && addresses.active(*terminal),
           "growth settlement returns speculative and unused pages while keeping the row active");
    expect(addresses.growth_pages_for_tokens(*terminal, 64) == 0 &&
               addresses.growth_pages_for_tokens(*terminal, 65) == 1,
           "next unit reserves only pages beyond the settled mapping");
    addresses.reserve_growth(*terminal, addresses.growth_pages_for_tokens(*terminal, 65));
    expect(addresses.growth_pages_for_tokens(*terminal, 65) == 1 &&
               addresses.reserved_growth_pages(*terminal) == 1,
           "unit growth demand is independent of an already acquired reservation");
    addresses.ensure_mapped_to_tokens(*terminal, 65, device.stream);
    addresses.settle_growth(*terminal, 65);
    expect(addresses.mapped_pages(*terminal) == 2 &&
               addresses.committed_frontier(*terminal) == 65 &&
               addresses.reserved_growth_pages(*terminal) == 0 &&
               physical_pages.allocated_pages() == 2 && physical_pages.reserved_pages() == 0,
           "settled active address reserves and commits the following unit independently");
    addresses.deactivate(*terminal);
    expect(addresses.release(*terminal) && physical_pages.allocated_pages() == 0 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 8,
           "terminal settlement releases both mappings and unused growth");

    const auto alternating = addresses.create_active(4, 0, device.stream);
    expect(alternating.has_value(), "alternating Host release address allocation");
    addresses.ensure_mapped_to_tokens(*alternating, 193, device.stream);
    addresses.commit_frontier(*alternating, 193);
    addresses.deactivate(*alternating);
    const std::array alternating_pages{
        addresses.logical_page(*alternating, 0), addresses.logical_page(*alternating, 1),
        addresses.logical_page(*alternating, 2), addresses.logical_page(*alternating, 3)};
    auto alternating_backup = extents.prepare(pages, alternating_pages);
    expect(alternating_backup.has_value(), "alternating Host extent reservation");
    const auto alternating_sources = extents.device_sources(*alternating_backup);
    physical_pages.copy_to_host(alternating_sources, extents.writable_view(*alternating_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto alternating_extent = extents.publish(std::move(*alternating_backup));
    const std::array alternating_release{alternating_pages[0], alternating_pages[2]};
    expect(extents.release_page_replicas(pages, alternating_release),
           "alternating Host page release transaction");
    expect(!extents.valid(alternating_extent) && !pages.host_resident(alternating_pages[0]) &&
               pages.host_resident(alternating_pages[1]) &&
               !pages.host_resident(alternating_pages[2]) &&
               pages.host_resident(alternating_pages[3]) &&
               pages.host_replica(alternating_pages[1]).extent !=
                   pages.host_replica(alternating_pages[3]).extent &&
               host_arena.occupied_bytes() == 2U * host_layout.page_stride,
           "one batch partitions alternating Host release and retained runs exactly once");
    const std::array alternating_retained{alternating_pages[1], alternating_pages[3]};
    expect(extents.release_page_replicas(pages, alternating_retained),
           "retained Host run release transaction");
    expect(host_arena.occupied_bytes() == 0 && addresses.release(*alternating) &&
               pages.occupied() == 0,
           "alternating Host extent partitions close without leaked descriptors");

    const auto shared = addresses.create_active(3, 0, device.stream);
    expect(shared.has_value(), "shared-prefix source address allocation");
    addresses.ensure_mapped_to_tokens(*shared, 65, device.stream);
    addresses.commit_frontier(*shared, 65);
    addresses.set_checkpoint_requirement(*shared, 65);
    device.synchronize();
    const auto shared_mapping = read_block_table(physical_tables, 0, 2);
    addresses.deactivate(*shared);
    const auto shared_full = addresses.logical_page(*shared, 0);
    const auto shared_tail = addresses.logical_page(*shared, 1);

    const auto branch_one = addresses.create_inactive();
    expect(branch_one.has_value(), "first shared-prefix branch address allocation");
    auto first_fork = addresses.prepare_prefix_fork(*shared, *branch_one, 65, 0, 1);
    expect(first_fork.needs_tail_copy() && pages.address_references(shared_full) == 1,
           "prefix fork remains unpublished while its tail copy is pending");
    physical_pages.copy_page(addresses.prefix_fork_tail_source(first_fork),
                             addresses.prefix_fork_tail_destination(first_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(first_fork), device.stream);
    device.synchronize();
    const auto branch_one_mapping = read_block_table(physical_tables, 1, 2);
    const auto branch_one_tail    = addresses.logical_page(*branch_one, 1);
    expect(addresses.logical_page(*branch_one, 0) == shared_full &&
               branch_one_mapping[0] == shared_mapping[0] &&
               branch_one_mapping[1] != shared_mapping[1] && branch_one_tail != shared_tail &&
               pages.address_references(shared_full) == 2 &&
               pages.address_references(shared_tail) == 1 &&
               addresses.reserved_growth_pages(*branch_one) == 0,
           "zero-growth fork shares full pages and publishes its reserved private partial tail");
    addresses.ensure_mapped_to_tokens(*branch_one, 66, device.stream);
    addresses.commit_frontier(*branch_one, 66);
    expect(pages.committed_columns(shared_tail) == 1 &&
               pages.committed_columns(branch_one_tail) == 2,
           "private branch writes do not extend the shared partial tail");
    addresses.deactivate(*branch_one);
    addresses.destructive_truncate_inactive(*branch_one, 65);
    expect(addresses.committed_frontier(*branch_one) == 65 &&
               pages.committed_columns(branch_one_tail) == 1 &&
               pages.address_references(shared_full) == 2,
           "private suffix truncation retains shared full-prefix pages");

    const auto branch_two = addresses.create_inactive();
    expect(branch_two.has_value(), "second shared-prefix branch address allocation");
    auto second_fork = addresses.prepare_prefix_fork(*shared, *branch_two, 65, 1, 1);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(second_fork),
                             addresses.prefix_fork_tail_destination(second_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(second_fork), device.stream);
    device.synchronize();
    const auto branch_two_mapping = read_block_table(physical_tables, 1, 2);
    const auto branch_two_tail    = addresses.logical_page(*branch_two, 1);
    expect(branch_two_tail != shared_tail && branch_two_tail != branch_one_tail &&
               addresses.logical_page(*branch_two, 0) == shared_full &&
               branch_two_mapping[0] == shared_mapping[0] &&
               branch_two_mapping[1] != shared_mapping[1] &&
               branch_two_mapping[1] != branch_one_mapping[1] &&
               pages.address_references(shared_full) == 3 && pages.occupied() == 4 &&
               physical_pages.allocated_pages() == 4,
           "independent shared branches own distinct partial tails and one shared full page");
    addresses.deactivate(*branch_two);
    expect(addresses.release(*shared) && !pages.valid(shared_tail) && pages.valid(shared_full) &&
               pages.address_references(shared_full) == 2 &&
               addresses.logical_page(*branch_one, 0) == shared_full &&
               addresses.logical_page(*branch_two, 0) == shared_full &&
               physical_pages.allocated_pages() == 3,
           "derived KV directories retain shared page identity after releasing their source");
    expect(addresses.release(*branch_one) && !pages.valid(branch_one_tail) &&
               pages.valid(shared_full) && pages.address_references(shared_full) == 1 &&
               physical_pages.allocated_pages() == 2,
           "releasing one branch reclaims only its private tail");
    expect(addresses.release(*branch_two) && !pages.valid(branch_two_tail) &&
               !pages.valid(shared_full) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "shared full-page occupancy survives until its final address reference releases");

    // A prefix fork at a page-aligned frontier shares every page with its source, including the
    // tail, and needs no tail copy. After admission maps the pages the prompt and draft window
    // need, the post-prefill trim keeps the frontier and drops only those unused pages. The
    // shared tail is not modified, so that trim must succeed.
    const auto aligned_source = addresses.create_active(3, 0, device.stream);
    expect(aligned_source.has_value(), "page-aligned fork source allocation");
    addresses.ensure_mapped_to_tokens(*aligned_source, 128, device.stream);
    addresses.commit_frontier(*aligned_source, 128);
    addresses.set_checkpoint_requirement(*aligned_source, 128);
    addresses.deactivate(*aligned_source);
    const auto aligned_tail = addresses.logical_page(*aligned_source, 1);

    const auto aligned_branch = addresses.create_inactive();
    expect(aligned_branch.has_value(), "page-aligned fork destination allocation");
    auto aligned_fork = addresses.prepare_prefix_fork(*aligned_source, *aligned_branch, 128, 1, 1);
    expect(!aligned_fork.needs_tail_copy(), "page-aligned prefix fork needs no tail copy");
    addresses.commit_prefix_fork(std::move(aligned_fork), device.stream);
    expect(addresses.logical_page(*aligned_branch, 1) == aligned_tail &&
               pages.address_references(aligned_tail) == 2 &&
               addresses.committed_frontier(*aligned_branch) == 128,
           "page-aligned fork shares its tail page with the source");
    addresses.ensure_mapped_to_tokens(*aligned_branch, 192, device.stream);
    device.synchronize();
    expect(addresses.mapped_pages(*aligned_branch) == 3, "fork maps its draft-window page");
    addresses.destructive_truncate(*aligned_branch, 128);
    expect(addresses.mapped_pages(*aligned_branch) == 2 &&
               addresses.committed_frontier(*aligned_branch) == 128 &&
               addresses.logical_page(*aligned_branch, 1) == aligned_tail &&
               pages.address_references(aligned_tail) == 2,
           "post-prefill trim of a page-aligned fork releases only the unused suffix");
    addresses.deactivate(*aligned_branch);
    expect(addresses.release(*aligned_branch) && addresses.release(*aligned_source) &&
               pages.occupied() == 0 && physical_pages.allocated_pages() == 0,
           "page-aligned fork releases without leaked pages");

    const auto mixed_source = addresses.create_active(4, 0, device.stream);
    expect(mixed_source.has_value(), "mixed prefix retained source allocation");
    addresses.ensure_mapped_to_tokens(*mixed_source, 65, device.stream);
    addresses.commit_frontier(*mixed_source, 65);
    addresses.set_checkpoint_requirement(*mixed_source, 65);
    addresses.deactivate(*mixed_source);
    const auto mixed_shared_full = addresses.logical_page(*mixed_source, 0);

    const auto mixed_active = addresses.create_inactive();
    expect(mixed_active.has_value(), "mixed prefix active branch allocation");
    auto mixed_fork = addresses.prepare_prefix_fork(*mixed_source, *mixed_active, 65, 2, 0);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(mixed_fork),
                             addresses.prefix_fork_tail_destination(mixed_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(mixed_fork), device.stream);
    addresses.ensure_mapped_to_tokens(*mixed_active, 130, device.stream);
    addresses.commit_frontier(*mixed_active, 130);
    addresses.set_checkpoint_requirement(*mixed_active, 130);
    const auto mixed_mutable_full = addresses.logical_page(*mixed_active, 1);
    const auto mixed_tail         = addresses.logical_page(*mixed_active, 2);

    const auto partial_view = addresses.create_inactive();
    expect(partial_view.has_value(), "mixed prefix view destination allocation");
    const auto allocated_before_view = physical_pages.allocated_pages();
    {
        auto aborted = addresses.prepare_active_prefix_view(*mixed_active, *partial_view, 130);
        expect(pages.source_pins(mixed_shared_full) == 1 &&
                   pages.source_pins(mixed_mutable_full) == 1 && pages.source_pins(mixed_tail) == 1,
               "prefix-view preparation pins immutable, mutable full, and partial pages");
    }
    expect(addresses.active(*mixed_active) && pages.source_pins(mixed_shared_full) == 0 &&
               pages.source_pins(mixed_mutable_full) == 0 && pages.source_pins(mixed_tail) == 0 &&
               pages.writer_references(mixed_shared_full) == 0 &&
               pages.writer_references(mixed_mutable_full) == 1 &&
               pages.writer_references(mixed_tail) == 1 &&
               physical_pages.allocated_pages() == allocated_before_view,
           "aborted mixed prefix view restores pins, allocation, and page ownership");

    auto partial = addresses.prepare_active_prefix_view(*mixed_active, *partial_view, 130);
    physical_pages.copy_page(addresses.active_prefix_view_tail_source(partial),
                             addresses.active_prefix_view_tail_destination(partial),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_active_prefix_view(std::move(partial));
    const auto partial_tail = addresses.logical_page(*partial_view, 2);
    expect(addresses.active(*mixed_active) && !addresses.active(*partial_view) &&
               addresses.logical_page(*partial_view, 0) == mixed_shared_full &&
               addresses.logical_page(*partial_view, 1) == mixed_mutable_full &&
               partial_tail != mixed_tail && pages.address_references(mixed_shared_full) == 3 &&
               pages.address_references(mixed_mutable_full) == 2 &&
               pages.writer_references(mixed_mutable_full) == 0 &&
               pages.writer_references(mixed_tail) == 1 &&
               pages.writer_references(partial_tail) == 0,
           "mixed prefix view freezes full pages and isolates its tail while keeping the writer");

    addresses.ensure_mapped_to_tokens(*mixed_active, 192, device.stream);
    addresses.commit_frontier(*mixed_active, 192);
    addresses.set_checkpoint_requirement(*mixed_active, 192);
    const auto aligned_view = addresses.create_inactive();
    expect(aligned_view.has_value(), "aligned prefix view destination allocation");
    const std::array aligned_prefix{addresses.logical_page(*mixed_active, 0),
                                    addresses.logical_page(*mixed_active, 1),
                                    addresses.logical_page(*mixed_active, 2)};
    device.synchronize();
    const auto aligned_mapping               = read_block_table(physical_tables, 0, 3);
    const auto allocated_before_aligned_view = physical_pages.allocated_pages();
    auto aligned = addresses.prepare_active_prefix_view(*mixed_active, *aligned_view, 192);
    expect(!aligned.needs_tail_copy(), "aligned prefix view requires no KV copy");
    addresses.commit_active_prefix_view(std::move(aligned));
    bool aligned_identity_preserved = true;
    for (std::uint32_t page = 0; page < aligned_prefix.size(); ++page) {
        aligned_identity_preserved =
            aligned_identity_preserved &&
            addresses.logical_page(*aligned_view, page) == aligned_prefix[page];
    }
    expect(aligned_identity_preserved &&
               read_block_table(physical_tables, 0, 3) == aligned_mapping &&
               physical_pages.allocated_pages() == allocated_before_aligned_view,
           "aligned prefix view preserves every logical and physical page without allocation");
    addresses.ensure_mapped_to_tokens(*mixed_active, 193, device.stream);
    addresses.settle_growth(*mixed_active, 192);
    expect(addresses.mapped_pages(*mixed_active) == 3 &&
               addresses.reserved_growth_pages(*mixed_active) == 0 &&
               addresses.logical_page(*mixed_active, 2) == aligned_prefix[2] &&
               physical_pages.allocated_pages() == allocated_before_aligned_view,
           "settlement releases a private speculative suffix ending at a shared full page");
    addresses.reserve_growth(*mixed_active, 1);
    addresses.ensure_mapped_to_tokens(*mixed_active, 193, device.stream);
    addresses.commit_frontier(*mixed_active, 193);
    device.synchronize();
    const auto appended_page = addresses.logical_page(*mixed_active, 3);
    expect(addresses.mapped_pages(*aligned_view) == 3 &&
               addresses.committed_frontier(*aligned_view) == 192 &&
               addresses.committed_frontier(*partial_view) == 130 &&
               pages.committed_columns(partial_tail) == 2 &&
               read_block_table(physical_tables, 0, 3) == aligned_mapping &&
               pages.writer_references(appended_page) == 1 &&
               pages.address_references(appended_page) == 1 &&
               pages.active_address_references(appended_page) == 1 &&
               physical_pages.allocated_pages() == allocated_before_aligned_view + 1,
           "continuation appends a private page without changing retained prefix views");

    addresses.deactivate(*mixed_active);
    expect(addresses.release(*aligned_view) && addresses.release(*partial_view) &&
               addresses.release(*mixed_active) && addresses.release(*mixed_source) &&
               pages.occupied() == 0 && physical_pages.allocated_pages() == 0,
           "mixed prefix view ownership closes without leaked logical or physical pages");

    const auto filler = addresses.create_active(4, 0, device.stream);
    expect(filler.has_value(), "full-capacity retained-fork filler allocation");
    addresses.ensure_mapped_to_tokens(*filler, 193, device.stream);
    addresses.commit_frontier(*filler, 193);
    addresses.set_checkpoint_requirement(*filler, 193);
    addresses.deactivate(*filler);

    const auto retained = addresses.create_active(4, 0, device.stream);
    expect(retained.has_value(), "full-capacity retained source allocation");
    addresses.ensure_mapped_to_tokens(*retained, 65, device.stream);
    addresses.commit_frontier(*retained, 65);
    addresses.set_checkpoint_requirement(*retained, 65);
    addresses.deactivate(*retained);
    const auto retained_destination = addresses.create_inactive();
    expect(retained_destination.has_value(), "full-capacity retained-fork destination allocation");
    expect(physical_pages.allocated_pages() == 6 && physical_pages.available_pages() == 2,
           "retained-fork fixture has insufficient space for the tail copy and growth together");

    const auto retained_full    = addresses.logical_page(*retained, 0);
    const auto retained_tail    = addresses.logical_page(*retained, 1);
    bool retained_fork_rejected = false;
    try {
        auto rejected = addresses.prepare_prefix_fork(*retained, *retained_destination, 65, 2, 1);
        (void)rejected;
    } catch (const std::bad_alloc&) { retained_fork_rejected = true; }
    expect(retained_fork_rejected && !addresses.active(*retained_destination) &&
               addresses.mapped_pages(*retained_destination) == 0 &&
               addresses.bound_row(*retained_destination) == -1 &&
               pages.device_resident(retained_full) && pages.device_resident(retained_tail) &&
               pages.address_references(retained_full) == 1 &&
               pages.address_references(retained_tail) == 1 &&
               pages.source_pins(retained_full) == 0 && pages.source_pins(retained_tail) == 0 &&
               pages.occupied() == 6 && physical_pages.allocated_pages() == 6 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 2,
           "failed prefix fork preserves its resident source and releases all destination claims");

    expect(addresses.release(*filler) && physical_pages.available_pages() == 6,
           "releasing unrelated ownership makes the whole retained fork feasible");
    auto retained_fork = addresses.prepare_prefix_fork(*retained, *retained_destination, 65, 2, 1);
    expect(pages.device_resident(retained_tail) && pages.source_pins(retained_tail) == 1 &&
               physical_pages.allocated_pages() == 3 && physical_pages.reserved_pages() == 2,
           "retained fork reserves tail and growth while the source remains resident");
    physical_pages.copy_page(addresses.prefix_fork_tail_source(retained_fork),
                             addresses.prefix_fork_tail_destination(retained_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(retained_fork), device.stream);
    expect(pages.device_resident(retained_tail) && pages.source_pins(retained_tail) == 0 &&
               addresses.logical_page(*retained_destination, 0) == retained_full &&
               addresses.logical_page(*retained_destination, 1) != retained_tail &&
               addresses.reserved_growth_pages(*retained_destination) == 2 &&
               physical_pages.allocated_pages() == 3 && physical_pages.reserved_pages() == 2,
           "retried retained fork publishes a private tail and the complete growth reservation");

    addresses.deactivate(*retained_destination);
    expect(addresses.release(*retained_destination) && addresses.release(*retained) &&
               host_arena.occupied_bytes() == 0 && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "retained fork failure and retry close all logical and physical ownership");
}

void test_cancel_alias_during_prefix_fork(ninfer::DeviceContext& device) {
    constexpr std::uint32_t page_tokens = ninfer::kPagedKVPageSize;
    ninfer::LayoutBuilder builder;
    const ninfer::DeviceKVPagePoolLayout page_layout = ninfer::plan_device_kv_page_pool(
        builder,
        {.page_group_count = 4,
         .geometry         = {
                     .page_tokens        = page_tokens,
                     .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                     .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}}}});
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 3, .table_rows = 2});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    store::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages());
    store::KVAddressSpaceStore addresses(pages, physical_tables, 3, 3);
    const ninfer::Tensor& plane = physical_pages.plane(0);
    const auto page_data        = [&](std::int32_t index) {
        return static_cast<unsigned char*>(plane.data) + index * plane.nb[3];
    };
    const auto read_page = [&](std::int32_t index) {
        std::vector<unsigned char> bytes(plane.nb[3]);
        CUDA_CHECK(
            cudaMemcpy(bytes.data(), page_data(index), bytes.size(), cudaMemcpyDeviceToHost));
        return bytes;
    };

    for (const bool cancel_fork : {false, true}) {
        const auto source = addresses.create_active(2, 0, device.stream);
        expect(source.has_value(), "pinned prefix source allocation");
        addresses.ensure_mapped_to_tokens(*source, page_tokens + 1, device.stream);
        addresses.settle_growth(*source, page_tokens + 1);
        addresses.set_checkpoint_requirement(*source, page_tokens + 1);
        device.synchronize();
        const auto source_mapping = read_block_table(physical_tables, 0, 2);
        CUDA_CHECK(cudaMemsetAsync(page_data(source_mapping[0]), 0x35, plane.nb[3], device.stream));
        CUDA_CHECK(cudaMemsetAsync(page_data(source_mapping[1]), 0xa6, plane.nb[3], device.stream));
        device.synchronize();
        addresses.deactivate(*source);
        const auto full = addresses.logical_page(*source, 0);
        const auto tail = addresses.logical_page(*source, 1);

        const auto alias       = addresses.create_inactive();
        const auto destination = addresses.create_inactive();
        expect(alias && destination, "pinned prefix branch allocation");
        auto alias_fork = addresses.prepare_prefix_fork(*source, *alias, page_tokens, 0, 0);
        addresses.commit_prefix_fork(std::move(alias_fork), device.stream);
        device.synchronize();
        expect(pages.address_references(full) == 2 && pages.active_address_references(full) == 1,
               "active branch shares the retained full page");

        auto pending = addresses.prepare_prefix_fork(*source, *destination, page_tokens + 1, 1, 1);
        physical_pages.copy_page(addresses.prefix_fork_tail_source(pending),
                                 addresses.prefix_fork_tail_destination(pending),
                                 device.transfer_stream);
        expect(pages.source_pins(full) == 1 && pages.source_pins(tail) == 1 &&
                   !addresses.can_release(*source),
               "pending fork pins its source and protects the final tail reference");
        if (cancel_fork) {
            addresses.deactivate(*alias);
            expect(addresses.release(*alias),
                   "inactive alias can release its nonfinal transfer-pinned page reference");
        } else {
            expect(addresses.release_after_deactivate(*alias),
                   "active cancellation can release its nonfinal transfer-pinned page reference");
        }
        expect(!addresses.valid(*alias) && addresses.valid(*source) &&
                   pages.address_references(full) == 1 &&
                   pages.active_address_references(full) == 0 && pages.source_pins(full) == 1 &&
                   pages.source_pins(tail) == 1 && pages.device_resident(full) &&
                   pages.device_resident(tail) && !addresses.can_release(*source),
               "alias cancellation preserves both pinned source pages and their final ownership");
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        if (cancel_fork) {
            addresses.abort_prefix_fork(pending);
            expect(addresses.release(*destination),
                   "cancelled fork releases its empty destination");
        } else {
            addresses.commit_prefix_fork(std::move(pending), device.stream);
            device.synchronize();
            const auto destination_mapping = read_block_table(physical_tables, 1, 2);
            expect(destination_mapping[0] == source_mapping[0] &&
                       destination_mapping[1] != source_mapping[1] &&
                       read_page(destination_mapping[0]) ==
                           std::vector<unsigned char>(plane.nb[3], 0x35) &&
                       read_page(destination_mapping[1]) ==
                           std::vector<unsigned char>(plane.nb[3], 0xa6),
                   "fork completes with the original shared prefix and copied tail after alias "
                   "cancellation");
            expect(addresses.release_after_deactivate(*destination),
                   "completed fork releases its active destination and unused growth reservation");
        }
        expect(pages.source_pins(full) == 0 && pages.source_pins(tail) == 0 &&
                   read_page(source_mapping[0]) == std::vector<unsigned char>(plane.nb[3], 0x35) &&
                   read_page(source_mapping[1]) == std::vector<unsigned char>(plane.nb[3], 0xa6),
               "fork completion or cancellation retires pins without changing source contents");
        expect(addresses.release(*source) && addresses.occupied() == 0 && pages.occupied() == 0 &&
                   physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0 &&
                   physical_pages.available_pages() == physical_pages.capacity_pages(),
               "alias cancellation closes all address, logical page, and physical reservation "
               "ownership");
        auto row_zero = physical_tables.acquire(0);
        auto row_one  = physical_tables.acquire(1);
        expect(row_zero.release() && row_one.release(), "both execution rows remain reusable");
    }
}

void test_shared_kv_directory(ninfer::DeviceContext& device) {
    constexpr std::uint32_t page_tokens = ninfer::kPagedKVPageSize;
    constexpr std::uint32_t full_pages  = 65;
    constexpr std::uint32_t frontier    = full_pages * page_tokens + 1;
    ninfer::LayoutBuilder builder;
    const ninfer::DeviceKVPagePoolLayout page_layout = ninfer::plan_device_kv_page_pool(
        builder,
        {.page_group_count = 72,
         .geometry         = {
                     .page_tokens        = page_tokens,
                     .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                     .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}}}});
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 68, .table_rows = 1});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    store::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages());
    store::KVAddressSpaceStore addresses(pages, physical_tables, 3, 68);

    const auto source = addresses.create_active(full_pages + 1, 0, device.stream);
    expect(source.has_value(), "long shared-directory source allocation");
    addresses.ensure_mapped_to_tokens(*source, frontier, device.stream);
    addresses.settle_growth(*source, frontier);
    addresses.set_checkpoint_requirement(*source, frontier);
    device.synchronize();
    const auto source_mapping = read_block_table(physical_tables, 0, full_pages + 1);
    addresses.deactivate(*source);
    std::vector<store::LogicalKVPageHandle> shared_pages;
    for (std::uint32_t page = 0; page < full_pages; ++page) {
        shared_pages.push_back(addresses.logical_page(*source, page));
    }
    const auto source_tail = addresses.logical_page(*source, full_pages);
    const auto branch      = addresses.create_inactive();
    expect(branch.has_value(), "long shared-directory fork destination allocation");
    auto fork = addresses.prepare_prefix_fork(*source, *branch, frontier, 1, 0);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(fork),
                             addresses.prefix_fork_tail_destination(fork), device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(fork), device.stream);
    device.synchronize();
    const auto branch_mapping    = read_block_table(physical_tables, 0, full_pages + 1);
    bool fork_identity_preserved = true;
    for (std::uint32_t page = 0; page < full_pages; ++page) {
        fork_identity_preserved = fork_identity_preserved &&
                                  addresses.logical_page(*branch, page) == shared_pages[page] &&
                                  branch_mapping[page] == source_mapping[page] &&
                                  pages.address_references(shared_pages[page]) == 2;
    }
    const auto branch_tail = addresses.logical_page(*branch, full_pages);
    expect(fork_identity_preserved && branch_tail != source_tail &&
               branch_mapping.back() != source_mapping.back() &&
               physical_pages.allocated_pages() == full_pages + 2 &&
               addresses.reserved_growth_pages(*branch) == 1,
           "long prefix fork shares every full-page identity and copies only the partial tail");
    expect(addresses.release(*source) && !pages.valid(source_tail) &&
               physical_pages.allocated_pages() == full_pages + 1,
           "long fork preserves shared pages after the original directory is destroyed");
    bool surviving_prefix_preserved = true;
    for (std::uint32_t page = 0; page < full_pages; ++page) {
        surviving_prefix_preserved = surviving_prefix_preserved &&
                                     addresses.logical_page(*branch, page) == shared_pages[page] &&
                                     pages.address_references(shared_pages[page]) == 1;
    }
    expect(surviving_prefix_preserved &&
               read_block_table(physical_tables, 0, full_pages + 1) == branch_mapping,
           "every inherited page remains reachable from the surviving long directory");

    const std::uint32_t next_frontier = (full_pages + 1) * page_tokens + 1;
    addresses.ensure_mapped_to_tokens(*branch, next_frontier, device.stream);
    addresses.settle_growth(*branch, next_frontier);
    device.synchronize();
    const auto grown_branch_mapping = read_block_table(physical_tables, 0, full_pages + 2);
    const auto appended_tail        = addresses.logical_page(*branch, full_pages + 1);
    const auto view_destination     = addresses.create_inactive();
    expect(view_destination.has_value(), "long directory prefix-view destination allocation");
    addresses.set_checkpoint_requirement(*branch, next_frontier);
    auto view = addresses.prepare_active_prefix_view(*branch, *view_destination, next_frontier);
    physical_pages.copy_page(addresses.active_prefix_view_tail_source(view),
                             addresses.active_prefix_view_tail_destination(view),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_active_prefix_view(std::move(view));
    const auto view_tail         = addresses.logical_page(*view_destination, full_pages + 1);
    bool view_identity_preserved = true;
    for (std::uint32_t page = 0; page < full_pages; ++page) {
        view_identity_preserved =
            view_identity_preserved &&
            addresses.logical_page(*view_destination, page) == shared_pages[page] &&
            pages.address_references(shared_pages[page]) == 2;
    }
    expect(view_identity_preserved &&
               addresses.logical_page(*view_destination, full_pages) == branch_tail &&
               view_tail != appended_tail && !addresses.active(*view_destination) &&
               addresses.active(*branch) && addresses.mapped_pages(*branch) == full_pages + 2 &&
               addresses.committed_frontier(*branch) == next_frontier &&
               read_block_table(physical_tables, 0, full_pages + 2) == grown_branch_mapping &&
               physical_pages.allocated_pages() == full_pages + 3,
           "long prefix view shares the grown full prefix and isolates its immutable tail");
    expect(addresses.release_after_deactivate(*branch) && !pages.valid(appended_tail) &&
               pages.valid(branch_tail) && physical_pages.allocated_pages() == full_pages + 2,
           "long prefix view retains its directory after the intermediate branch is released");
    addresses.activate(*view_destination, 0, 0, device.stream);
    device.synchronize();
    const auto view_mapping = read_block_table(physical_tables, 0, full_pages + 2);
    expect(view_mapping.back() != grown_branch_mapping.back(),
           "activated prefix view uses its independent tail page");
    addresses.ensure_mapped_to_tokens(*view_destination, next_frontier + 1, device.stream);
    addresses.settle_growth(*view_destination, next_frontier + 1);
    device.synchronize();
    std::vector<std::int32_t> expected_mapping(source_mapping.begin(),
                                               source_mapping.begin() + full_pages);
    expected_mapping.push_back(branch_mapping[full_pages]);
    expected_mapping.push_back(view_mapping.back());
    expect(read_block_table(physical_tables, 0, full_pages + 2) == expected_mapping &&
               addresses.committed_frontier(*view_destination) == next_frontier + 1 &&
               pages.committed_columns(view_tail) == 2,
           "surviving long prefix publishes the full execution mapping and continues in its tail");
    addresses.deactivate(*view_destination);
    expect(addresses.release(*view_destination) && !pages.valid(branch_tail) &&
               !pages.valid(view_tail) && addresses.occupied() == 0 && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0 &&
               physical_pages.available_pages() == physical_pages.capacity_pages(),
           "last long-directory reference releases every logical and physical page");
}

std::vector<std::vector<unsigned char>> fill_device_pool(ninfer::DeviceKVPagePool& pool,
                                                         cudaStream_t stream) {
    std::vector<std::vector<unsigned char>> bytes;
    bytes.reserve(pool.plane_count());
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        std::vector<unsigned char> host(plane.bytes());
        for (std::size_t index = 0; index < host.size(); ++index) {
            host[index] =
                static_cast<unsigned char>((index * 29U + plane_index * 61U + 17U) & 0xffU);
        }
        const cudaError_t err =
            cudaMemcpyAsync(plane.data, host.data(), host.size(), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("device pool fill failed: ") +
                                     cudaGetErrorString(err));
        }
        bytes.push_back(std::move(host));
    }
    return bytes;
}

std::vector<std::byte>
expected_host_records(const ninfer::DeviceKVPagePool& pool,
                      std::span<const std::int32_t> physical_pages,
                      const ninfer::HostKVPageLayout& host_layout,
                      const std::vector<std::vector<unsigned char>>& device_planes) {
    std::vector<std::byte> out(host_layout.page_stride * physical_pages.size(), std::byte{0});
    for (std::size_t logical = 0; logical < physical_pages.size(); ++logical) {
        const std::int32_t physical = physical_pages[logical];
        for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
            const ninfer::Tensor& plane                 = pool.plane(plane_index);
            const ninfer::HostKVPlaneLayout& host_plane = host_layout.planes[plane_index];
            std::byte* destination =
                out.data() + logical * host_layout.page_stride + host_plane.offset;
            if (pool.geometry().device_plane_order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const unsigned char* source = device_planes[plane_index].data() +
                                              static_cast<std::size_t>(physical) * plane.nb[3];
                std::memcpy(destination, source, host_plane.page_payload_bytes);
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    const unsigned char* source = device_planes[plane_index].data() +
                                                  static_cast<std::size_t>(head) * plane.nb[3] +
                                                  static_cast<std::size_t>(physical) * plane.nb[2];
                    std::memcpy(destination +
                                    static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                                source, host_plane.head_payload_bytes);
                }
            }
        }
    }
    return out;
}

void test_history_prefix_view(ninfer::DeviceContext& context, ninfer::PagedKVPlaneOrder order) {
    constexpr auto page_size = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
    constexpr auto rewrite   = page_size + 13U;
    constexpr auto endpoint  = 3U * page_size + 11U;
    const ninfer::KVPageGeometry geometry{
        .device_plane_order = order,
        .planes             = {{ninfer::DType::I8, 8, 2, 256}, {ninfer::DType::FP16, 1, 2, 256}},
    };
    ninfer::LayoutBuilder builder;
    const auto page_layout =
        ninfer::plan_device_kv_page_pool(builder, {.page_group_count = 8, .geometry = geometry});
    const auto table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 6, .table_rows = 1});
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::DeviceKVPagePool pool({arena.base(), arena.capacity()}, page_layout);
    ninfer::KVExecutionTablePool tables({arena.base(), arena.capacity()}, table_layout, pool);
    store::LogicalKVPageStore pages(pool, 12);
    store::KVAddressSpaceStore addresses(pages, tables, 4, 6);
    const auto source      = addresses.create_active(5, 0, context.stream).value();
    const auto destination = addresses.create_inactive().value();
    addresses.ensure_mapped_to_tokens(source, endpoint, context.stream);
    addresses.commit_frontier(source, endpoint);
    addresses.set_checkpoint_requirement(source, rewrite);
    addresses.deactivate(source);
    addresses.activate(source, 1, 0, context.stream);
    bool protected_truncate_rejected = false;
    try {
        addresses.destructive_truncate(source, rewrite - 1U);
    } catch (const std::logic_error&) { protected_truncate_rejected = true; }
    expect(protected_truncate_rejected,
           "activation lost the internal rewrite checkpoint protection");

    const auto original_bytes = fill_device_pool(pool, context.stream);
    context.synchronize();
    const auto original_mapping = read_block_table(tables, 0, 4);
    const auto allocated = pool.allocated_pages(), reserved = pool.reserved_pages();
    {
        auto aborted = addresses.prepare_active_prefix_view(source, destination, rewrite);
        bool reader_truncate_rejected = false;
        try {
            addresses.destructive_truncate(source, rewrite);
        } catch (const std::logic_error&) { reader_truncate_rejected = true; }
        expect(reader_truncate_rejected,
               "an in-flight prefix-view reader allowed destructive truncation");
    }
    expect(pool.allocated_pages() == allocated && pool.reserved_pages() == reserved &&
               addresses.mapped_pages(destination) == 0 &&
               addresses.committed_frontier(source) == endpoint &&
               pages.source_pins(addresses.logical_page(source, 1)) == 0 &&
               pages.writer_references(addresses.logical_page(source, 0)) == 1,
           "aborted prefix view changed history ownership or leaked its destination");

    const auto aligned_destination = addresses.create_inactive().value();
    auto aligned = addresses.prepare_active_prefix_view(source, aligned_destination, page_size);
    expect(!aligned.needs_tail_copy() && pool.allocated_pages() == allocated,
           "page-aligned prefix export allocated a copy");
    addresses.commit_active_prefix_view(std::move(aligned));

    auto view = addresses.prepare_active_prefix_view(source, destination, rewrite);
    expect(view.needs_tail_copy(), "partial prefix export omitted its tail copy");
    pool.copy_page(addresses.active_prefix_view_tail_source(view),
                   addresses.active_prefix_view_tail_destination(view), context.stream);
    context.synchronize();
    addresses.commit_active_prefix_view(std::move(view));
    expect(
        addresses.active(source) && !addresses.active(destination) &&
            addresses.bound_row(source) == 0 && addresses.committed_frontier(source) == endpoint &&
            addresses.mapped_pages(source) == 4 && addresses.reserved_growth_pages(source) == 1 &&
            read_block_table(tables, 0, 4) == original_mapping,
        "prefix export changed the active row, growth or committed suffix");
    expect(addresses.logical_page(source, 0) == addresses.logical_page(destination, 0) &&
               addresses.logical_page(source, 1) != addresses.logical_page(destination, 1) &&
               pages.active_address_references(addresses.logical_page(source, 0)) == 1 &&
               pages.writer_references(addresses.logical_page(destination, 0)) == 0 &&
               pages.writer_references(addresses.logical_page(destination, 1)) == 0,
           "prefix view did not share full pages and isolate its immutable tail");

    const auto host_layout = ninfer::plan_host_kv_page_layout(geometry);
    const std::array layouts{host_layout};
    ninfer::HostContextArena host_backing(host_layout.page_stride * 6, host_layout.page_stride);
    ninfer::HostKVArena host_arena(host_backing, layouts);
    auto host            = host_arena.allocate(host_layout, 6).value();
    const auto host_view = host_arena.writable_view(host);
    std::memset(host_view.data(), 0, host_layout.page_stride * 6);
    const std::array observation{
        addresses.physical_page(source, 0),      addresses.physical_page(source, 1),
        addresses.physical_page(source, 2),      addresses.physical_page(source, 3),
        addresses.physical_page(destination, 0), addresses.physical_page(destination, 1)};
    pool.copy_to_host(observation, host_view, context.stream);
    context.synchronize();
    const auto expected_source =
        expected_host_records(pool, original_mapping, host_layout, original_bytes);
    const auto expected_view = expected_host_records(pool, std::span(original_mapping).first(2),
                                                     host_layout, original_bytes);
    expect(std::memcmp(host_view.data(), expected_source.data(), expected_source.size()) == 0 &&
               std::memcmp(host_view.data() + 4 * host_layout.page_stride, expected_view.data(),
                           expected_view.size()) == 0,
           "exported prefix or source suffix differs from the original KV planes");

    // Retire the internal R, then rewrite the source after an earlier valid prefix. The exported
    // view still needs the old tokens in the same physical-page range.
    addresses.set_checkpoint_requirement(source, page_size);
    addresses.destructive_truncate(source, page_size + 5U);
    const auto physical_tail = original_mapping[1];
    for (std::size_t i = 0; i < pool.plane_count(); ++i) {
        const auto& plane = pool.plane(i);
        for (std::uint32_t head = 0; head < geometry.planes[i].head_extent; ++head) {
            const auto offset =
                order == ninfer::PagedKVPlaneOrder::PageMajor
                    ? static_cast<std::size_t>(physical_tail) * plane.nb[3] + head * plane.nb[2]
                    : head * plane.nb[3] + static_cast<std::size_t>(physical_tail) * plane.nb[2];
            const auto error =
                cudaMemsetAsync(static_cast<std::byte*>(plane.data) + offset + 5 * plane.nb[1], 0,
                                (page_size - 5) * plane.nb[1], context.stream);
            if (error != cudaSuccess) { throw std::runtime_error("source suffix rewrite failed"); }
        }
    }
    addresses.commit_frontier(source, page_size + 21U);
    const std::array exported{addresses.physical_page(destination, 0),
                              addresses.physical_page(destination, 1)};
    pool.copy_to_host(exported, host_view.subview(0, 2), context.stream);
    context.synchronize();
    expect(std::memcmp(host_view.data(), expected_view.data(), expected_view.size()) == 0,
           "source rewrite changed the old exported prefix contents");
    expect(addresses.release(destination) && addresses.release(aligned_destination) &&
               addresses.release_after_deactivate(source) && pool.allocated_pages() == 0 &&
               pool.reserved_pages() == 0,
           "retiring the final history and views leaked KV storage");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    try {
        ninfer::DeviceContext device(0);
        test_state_store(device);
        test_kv_store(device);
        test_cancel_alias_during_prefix_fork(device);
        test_shared_kv_directory(device);
        test_history_prefix_view(device, ninfer::PagedKVPlaneOrder::PageMajor);
        test_history_prefix_view(device, ninfer::PagedKVPlaneOrder::HeadMajor);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
