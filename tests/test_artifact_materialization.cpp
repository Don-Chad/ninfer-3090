#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
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
constexpr std::size_t kFp8TensorBytes = 260;
constexpr std::size_t kTailReadBytes  = 256 + kFp8TensorBytes;

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
                            {{"name", "weights/fp8"},
                             {"kind", "tensor"},
                             {"shape", {2, 4}},
                             {"format", "FP8_E4M3FN_ROW_BF16S"},
                             {"layout", "row-scale-v1"},
                             {"offset", 8448},
                             {"bytes", kFp8TensorBytes}},
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
        constexpr std::array<std::uint64_t, 2> fp8_shape = {2, 4};
        const auto validated_fp8                         = validation_binder.require_tensor(
            "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        validation_binder.validate_only(validated_fp8);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 4 && validation_plan.host_objects.size() == 1 &&
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

        const auto fp8 = binder.require_tensor(
            "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        binder.materialize_on_device(fp8);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 4 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 3 && plan.device_capacity_bytes == 772,
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
        std::array<std::byte, kFp8TensorBytes> fp8_copied{};
        CUDA_CHECK(cudaMemcpy(fp8_copied.data(), materialized.device_data(fp8), fp8_copied.size(),
                              cudaMemcpyDeviceToHost));
        require(std::all_of(fp8_copied.begin(), fp8_copied.end(),
                            [](std::byte value) { return value == std::byte{4}; }),
                "FP8 device tensor payload differs from the artifact");

        const ninfer::Weight fp8_weight = ninfer::artifact::materialized_weight(
            materialized, fp8, ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S, 2, 4);
        require(fp8_weight.qtype == ninfer::QType::FP8_E4M3FN_ROW_BF16S &&
                    fp8_weight.layout == ninfer::QuantLayout::RowScale &&
                    fp8_weight.scale_dtype == ninfer::DType::BF16 && fp8_weight.n == 2 &&
                    fp8_weight.k == 4 && fp8_weight.group == 4 && fp8_weight.group_size == 4 &&
                    fp8_weight.qdata == fp8_weight.payload && fp8_weight.qhigh == nullptr &&
                    fp8_weight.scales == static_cast<const std::byte*>(fp8_weight.payload) + 256 &&
                    fp8_weight.payload_bytes == kFp8TensorBytes,
                "materialized FP8 Weight metadata is incomplete");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 3 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() + kFp8TensorBytes &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment +
                                            kTailReadBytes,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");
        // Overlay plan mechanics: an evict-ranked tensor lands in a chunk-aligned arena tail
        // backed by the eviction pool, a HostPinned tensor lands in the pinned block, and a
        // transaction preserves every byte of the borrowed chunk.
        if (ninfer::EvictableWeightPool::supported(device)) {
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
            const auto overlay_fp8 = overlay_binder.require_tensor(
                "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
                ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
            overlay_binder.validate_only(overlay_fp8);
            const ninfer::artifact::MaterializationPlan overlay_plan =
                overlay_binder.finish(kChunk);
            require(overlay_plan.evictable_tail_offset == kChunk &&
                        overlay_plan.evictable_tail_bytes == kSecondTensor.size() &&
                        overlay_plan.device_objects.back().offset ==
                            overlay_plan.evictable_tail_offset &&
                        overlay_plan.device_objects.back().alignment ==
                            overlay_plan.device_objects.front().alignment &&
                        overlay_plan.device_capacity_bytes == kChunk + kSecondTensor.size(),
                    "evict-ranked tensor was not planned into a chunk-aligned arena tail");

            auto pool = std::make_unique<ninfer::EvictableWeightPool>(
                device, ninfer::EvictableWeightPool::Config{
                            .arena_bytes           = overlay_plan.device_capacity_bytes,
                            .evictable_tail_bytes  = overlay_plan.evictable_tail_bytes,
                            .window_capacity_bytes = kChunk,
                        });
            auto overlay_materialized = ninfer::artifact::materialize(
                reader, overlay_plan, device, nullptr, std::move(pool));
            ninfer::EvictableWeightPool* live_pool = overlay_materialized.eviction_pool();
            require(live_pool != nullptr, "pool-backed materialization dropped the pool");
            require(overlay_materialized.device_arena().used() ==
                        overlay_plan.device_capacity_bytes,
                    "pool-backed arena does not account the planned capacity");
            live_pool->capture_window_mirror(device.transfer_stream);

            void* evictable_home = overlay_materialized.device_data(evictable);
            {
                auto transaction = live_pool->evict(kChunk, device.stream);
                require(transaction.leased().bytes == kChunk,
                        "window transaction mapped an unexpected extent");
                CUDA_CHECK(cudaMemsetAsync(transaction.leased().data, 0x5A,
                                           transaction.leased().bytes, device.stream));
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                transaction.close();
                require(!live_pool->poisoned(), "window transaction poisoned the pool");
            }
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
            const auto pinned_fp8 = pinned_binder.require_tensor(
                "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
                ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
            pinned_binder.validate_only(pinned_fp8);
            const ninfer::artifact::MaterializationPlan pinned_plan = pinned_binder.finish();
            require(pinned_plan.pinned_objects.size() == 1 &&
                        pinned_plan.pinned_capacity_bytes == kSecondTensor.size() &&
                        pinned_plan.device_capacity_bytes == kTensor.size(),
                    "pinned placement was not planned into the pinned block");

            auto pinned_materialized =
                ninfer::artifact::materialize(reader, pinned_plan, device);
            require(pinned_materialized.is_host_pinned(pinned_tensor) &&
                        !pinned_materialized.is_host_pinned(device_tensor),
                    "pinned residency is not reported per object");
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
