#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                        })},
        },
        "materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 3 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 3 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 2 && plan.device_capacity_bytes == 260,
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                              cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 2 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment +
                                            kSecondTensor.size(),
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");

        // Overlay plan mechanics: an evict-ranked tensor lands in a chunk-aligned
        // arena tail backed by the eviction pool, a HostPinned tensor lands in the
        // pinned block, and an evict/restore transaction preserves every byte.
        if (ninfer::EvictableWeightPool::supported(device.device)) {
            constexpr std::size_t kChunk = ninfer::EvictableWeightPool::kChunkBytes;
            ninfer::artifact::Binder overlay_binder(reader);
            const auto overlay_resource = overlay_binder.require_resource(
                "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
            overlay_binder.retain_on_host(overlay_resource);
            const auto resident =
                overlay_binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                              ninfer::artifact::StorageLayout::ContiguousLeV1,
                                              tensor_shape);
            overlay_binder.materialize_on_device(resident);
            const auto evictable = overlay_binder.require_tensor(
                "weights/second", ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
            overlay_binder.materialize_on_device(evictable, /*evict_rank=*/700);
            const ninfer::artifact::MaterializationPlan overlay_plan =
                overlay_binder.finish(kChunk);
            require(overlay_plan.evictable_tail_bytes == kSecondTensor.size() &&
                        overlay_plan.device_objects.back().offset % kChunk == 0 &&
                        overlay_plan.device_capacity_bytes ==
                            kChunk + kSecondTensor.size(),
                    "evict-ranked tensor was not planned into a chunk-aligned arena tail");

            auto pool = std::make_unique<ninfer::EvictableWeightPool>(
                ninfer::EvictableWeightPool::Config{
                    .arena_bytes          = overlay_plan.device_capacity_bytes,
                    .evictable_tail_bytes = overlay_plan.evictable_tail_bytes,
                    .overlay_bytes        = kChunk,
                    .device               = device.device,
                });
            auto overlay_materialized = ninfer::artifact::materialize(
                reader, overlay_plan, device, nullptr, std::move(pool));
            ninfer::EvictableWeightPool* live_pool = overlay_materialized.eviction_pool();
            require(live_pool != nullptr, "pool-backed materialization dropped the pool");
            live_pool->capture_tail_mirror(device.load_stream);

            void* evictable_home = overlay_materialized.device_data(evictable);
            std::size_t mapped   = 0;
            std::byte* staging   = live_pool->evict(kChunk, &mapped);
            CUDA_CHECK(cudaMemset(staging, 0x5A, mapped));
            // The pool contract requires a quiesced device before remapping.
            CUDA_CHECK(cudaDeviceSynchronize());
            live_pool->restore(device.stream);
            require(overlay_materialized.device_data(evictable) == evictable_home,
                    "evictable tensor address changed across the overlay transaction");
            std::array<std::byte, kSecondTensor.size()> roundtrip{};
            CUDA_CHECK(cudaMemcpy(roundtrip.data(), evictable_home, roundtrip.size(),
                                  cudaMemcpyDeviceToHost));
            require(roundtrip == kSecondTensor,
                    "evictable tensor bytes were not restored from the mirror");
        } else {
            std::cout << "note: VMM unsupported, overlay plan mechanics not exercised\n";
        }

        {
            // HostPinned placement: payload lands in the pinned block, never on device.
            ninfer::artifact::Binder pinned_binder(reader);
            const auto pinned_resource = pinned_binder.require_resource(
                "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
            pinned_binder.retain_on_host(pinned_resource);
            const auto device_tensor =
                pinned_binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                             ninfer::artifact::StorageLayout::ContiguousLeV1,
                                             tensor_shape);
            pinned_binder.materialize_on_device(device_tensor);
            const auto pinned_tensor = pinned_binder.require_tensor(
                "weights/second", ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
            pinned_binder.materialize_on_host_pinned(pinned_tensor);
            const ninfer::artifact::MaterializationPlan pinned_plan = pinned_binder.finish();
            require(pinned_plan.pinned_objects.size() == 1 &&
                        pinned_plan.pinned_capacity_bytes == kSecondTensor.size(),
                    "pinned placement was not planned into the pinned block");

            auto pinned_materialized =
                ninfer::artifact::materialize(reader, pinned_plan, device);
            require(pinned_materialized.is_host_pinned(pinned_tensor),
                    "pinned tensor is not reported as host pinned");
            const auto block = pinned_materialized.pinned_block();
            require(block.size() == kSecondTensor.size() &&
                        std::equal(block.begin(), block.end(), kSecondTensor.begin(),
                                   kSecondTensor.end()),
                    "pinned block content differs from the artifact payload");
            require(pinned_materialized.storage_data(pinned_tensor) ==
                        const_cast<std::byte*>(block.data()) +
                            pinned_materialized.pinned_offset(pinned_tensor),
                    "pinned tensor storage pointer is outside the pinned block");
            require(pinned_materialized.stats().pinned_weight_bytes == kSecondTensor.size(),
                    "pinned weight bytes are not accounted");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
