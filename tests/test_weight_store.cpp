#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/weight_store.h"
#include "qus/core/weight_store_parser.h"

#include <cuda_runtime.h>

#include <dlfcn.h>

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::atomic<bool> g_fail_next_event_record{false};
std::atomic<int> g_stream_sync_failures_remaining{0};
std::atomic<bool> g_track_upload_lifetime{false};
std::atomic<unsigned long long> g_lifetime_sequence{0};
std::atomic<unsigned long long> g_first_successful_sync{0};
std::atomic<unsigned long long> g_first_upload_free{0};

void record_first(std::atomic<unsigned long long>& target) {
    const unsigned long long sequence = g_lifetime_sequence.fetch_add(1) + 1;
    unsigned long long expected       = 0;
    (void)target.compare_exchange_strong(expected, sequence);
}

bool consume_stream_sync_failure() {
    int remaining = g_stream_sync_failures_remaining.load();
    while (remaining > 0) {
        if (g_stream_sync_failures_remaining.compare_exchange_weak(remaining, remaining - 1)) {
            return true;
        }
    }
    return false;
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (g_fail_next_event_record.exchange(false)) { return cudaErrorUnknown; }
    using Fn               = cudaError_t(CUDARTAPI*)(cudaEvent_t, cudaStream_t);
    static const auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaEventRecord"));
    return real == nullptr ? cudaErrorUnknown : real(event, stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    if (consume_stream_sync_failure()) { return cudaErrorUnknown; }
    using Fn                 = cudaError_t(CUDARTAPI*)(cudaStream_t);
    static const auto real   = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    const cudaError_t result = real == nullptr ? cudaErrorUnknown : real(stream);
    if (result == cudaSuccess && g_track_upload_lifetime) { record_first(g_first_successful_sync); }
    return result;
}

extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    if (g_track_upload_lifetime) { record_first(g_first_upload_free); }
    using Fn               = cudaError_t(CUDARTAPI*)(void*);
    static const auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaFreeHost"));
    return real == nullptr ? cudaErrorUnknown : real(pointer);
}

extern "C" cudaError_t CUDARTAPI cudaFree(void* pointer) {
    if (g_track_upload_lifetime) { record_first(g_first_upload_free); }
    using Fn               = cudaError_t(CUDARTAPI*)(void*);
    static const auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaFree"));
    return real == nullptr ? cudaErrorUnknown : real(pointer);
}

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

constexpr std::uint64_t kSegmentRecordSize = 32;

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "qus_q5090_weight_store_fixture.qus";
    const std::filesystem::path script =
        std::filesystem::path(QUS_SOURCE_DIR) / "tests/fixtures/make_q5090_fixture.py";
    const std::string command =
        "python3 \"" + script.string() + "\" --out \"" + path.string() + "\"";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("fixture generator failed"); }
    return path;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to open fixture"); }
    const std::vector<char> chars{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

void write_file(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("failed to create fixture"); }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) { throw std::runtime_error("failed to write fixture"); }
}

std::uint64_t read_u64(const std::vector<std::byte>& bytes, std::uint64_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void write_u32(std::vector<std::byte>& bytes, std::uint64_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

qus::Q5090Expectations expectations() {
    qus::Q5090Expectations expected;
    expected.layer_count             = 64;
    expected.hidden_size             = 5120;
    expected.intermediate_size       = 17408;
    expected.vocab_size              = 248320;
    expected.num_attention_heads     = 24;
    expected.num_key_value_heads     = 4;
    expected.head_dim                = 256;
    expected.gdn_key_heads           = 16;
    expected.gdn_value_heads         = 48;
    expected.gdn_key_head_dim        = 128;
    expected.gdn_value_head_dim      = 128;
    expected.gdn_conv_width          = 4;
    expected.full_attention_interval = 4;
    expected.max_position_embeddings = 262144;
    expected.validate_model_contract = false;
    return expected;
}

const qus::ParsedQ5090Tensor& find_tensor(const qus::ParsedQ5090File& parsed,
                                          std::string_view name) {
    for (const qus::ParsedQ5090Tensor& tensor : parsed.tensors) {
        if (tensor.name == name) { return tensor; }
    }
    throw std::runtime_error("tensor not found");
}

const qus::ParsedQ5090Segment& find_segment(const qus::ParsedQ5090File& parsed,
                                            std::string_view name) {
    for (const qus::ParsedQ5090Segment& segment : parsed.segments) {
        if (segment.name == name) { return segment; }
    }
    throw std::runtime_error("segment not found");
}

std::size_t find_segment_index(const qus::ParsedQ5090File& parsed, std::string_view name) {
    for (std::size_t i = 0; i < parsed.segments.size(); ++i) {
        if (parsed.segments[i].name == name) { return i; }
    }
    throw std::runtime_error("segment not found");
}

std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t align) {
    return ((value + align - 1) / align) * align;
}

std::uint64_t nibble_bytes_per_group(qus::QType qtype) {
    switch (qtype) {
    case qus::QType::Q4G64_F16S:
    case qus::QType::Q5G64_F16S:
    case qus::QType::Q6G64_F16S:
    case qus::QType::W8G32_F16S:
        return 32;
    case qus::QType::W8G128_F16S:
        return 128;
    default:
        throw std::runtime_error("unexpected row-split qtype");
    }
}

std::uint64_t high_bytes_per_group(qus::QType qtype) {
    switch (qtype) {
    case qus::QType::Q4G64_F16S:
    case qus::QType::W8G128_F16S:
    case qus::QType::W8G32_F16S:
        return 0;
    case qus::QType::Q5G64_F16S:
        return 8;
    case qus::QType::Q6G64_F16S:
        return 16;
    default:
        throw std::runtime_error("unexpected row-split qtype");
    }
}

int expect_plane_offsets(const qus::Weight& weight, const qus::ParsedQ5090Tensor& tensor,
                         const qus::ParsedQ5090Segment& segment, std::string_view label) {
    if (weight.payload == nullptr) { return fail(std::string(label) + " payload null"); }
    const std::uint64_t groups     = tensor.padded_shape[1] / tensor.group_size;
    const std::uint64_t nibble_row = groups * nibble_bytes_per_group(tensor.qtype);
    const std::uint64_t high_row   = groups * high_bytes_per_group(tensor.qtype);
    const std::uint64_t scale_row  = groups * 2ULL;
    const std::uint64_t high_rel   = align_up_u64(tensor.nibble_plane_bytes, 256);
    const std::uint64_t scale_rel  = high_rel + align_up_u64(tensor.high_plane_bytes, 256);
    const auto* base               = static_cast<const std::byte*>(weight.payload);
    int failures                   = 0;
    failures += weight.high_plane_bytes == tensor.high_plane_bytes
                    ? 0
                    : fail(std::string(label) + " high_plane_bytes mismatch");
    failures += weight.qdata == base + segment.row_begin * nibble_row
                    ? 0
                    : fail(std::string(label) + " qdata offset mismatch");
    if (high_row == 0) {
        failures += weight.qhigh == nullptr ? 0 : fail(std::string(label) + " qhigh non-null");
    } else {
        failures += weight.qhigh == base + high_rel + segment.row_begin * high_row
                        ? 0
                        : fail(std::string(label) + " qhigh offset mismatch");
    }
    failures += weight.scales == base + scale_rel + segment.row_begin * scale_row
                    ? 0
                    : fail(std::string(label) + " scales offset mismatch");
    return failures;
}

std::vector<std::byte> payload_bytes(const std::vector<std::byte>& file,
                                     const qus::ParsedQ5090Tensor& tensor) {
    const auto begin = file.begin() + static_cast<std::ptrdiff_t>(tensor.payload_offset);
    const auto end   = begin + static_cast<std::ptrdiff_t>(tensor.payload_bytes);
    return std::vector<std::byte>(begin, end);
}

int expect_device_bytes(const void* device, const std::vector<std::byte>& expected,
                        std::string_view label) {
    std::vector<std::byte> actual(expected.size());
    const cudaError_t err =
        cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " cudaMemcpy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual != expected) {
        std::cerr << label << " payload mismatch\n";
        return 1;
    }
    return 0;
}

std::uint32_t segment_row_sum(const qus::ParsedQ5090File& parsed,
                              const qus::ParsedQ5090Tensor& tensor) {
    std::uint32_t rows = 0;
    for (std::uint32_t i = 0; i < tensor.segment_count; ++i) {
        rows += parsed.segments[static_cast<std::size_t>(tensor.segment_begin + i)].row_count;
    }
    return rows;
}

int expect_counts(const qus::WeightStore& store, std::size_t quant_count,
                  std::uint64_t loaded_bytes) {
    int failures = 0;
    failures += store.tensor_count() == 20 ? 0 : fail("tensor_count mismatch");
    failures += store.quant_count() == quant_count ? 0 : fail("quant_count mismatch");
    failures +=
        store.module_tensor_count(qus::ModuleKind::TextCore) == 6 ? 0 : fail("TEXT count mismatch");
    failures +=
        store.module_tensor_count(qus::ModuleKind::MtpDraft) == 12 ? 0 : fail("MTP count mismatch");
    failures += store.module_tensor_count(qus::ModuleKind::VisionEncoder) == 2
                    ? 0
                    : fail("VISION count mismatch");
    failures +=
        store.loaded_payload_bytes() == loaded_bytes ? 0 : fail("loaded payload bytes mismatch");
    return failures;
}

int expect_default_text_load(const qus::WeightStore& store, const qus::ParsedQ5090File& parsed,
                             const std::vector<std::byte>& file) {
    int failures = 0;
    failures += expect_counts(store, 5, parsed.modules[0].payload_bytes);
    failures += store.module_loaded(qus::ModuleKind::TextCore) ? 0 : fail("TEXT not loaded");
    failures += !store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP loaded by default");
    failures +=
        !store.module_loaded(qus::ModuleKind::VisionEncoder) ? 0 : fail("VISION loaded by default");

    const auto& text_q             = find_tensor(parsed, "layers.0.mlp.down_proj.weight");
    const qus::Weight* text_weight = store.qweight("layers.0.mlp.down_proj.weight");
    failures += text_weight != nullptr ? 0 : fail("missing text quant weight");
    if (text_weight != nullptr) {
        failures += text_weight->payload != nullptr ? 0 : fail("text quant payload is null");
        failures += text_weight->payload_bytes == text_q.payload_bytes
                        ? 0
                        : fail("text payload bytes mismatch");
        failures +=
            text_weight->layout == qus::QuantLayout::RowSplit ? 0 : fail("text layout mismatch");
        failures += text_weight->qhigh != nullptr ? 0 : fail("text qhigh null");
        failures += text_weight->scales != nullptr ? 0 : fail("text scales null");
        failures += expect_plane_offsets(*text_weight, text_q,
                                         find_segment(parsed, "layers.0.mlp.down_proj.weight"),
                                         "text qweight");
        failures +=
            expect_device_bytes(text_weight->payload, payload_bytes(file, text_q), "text qweight");
    }

    const qus::Weight* gate = store.qweight("layers.0.mlp.gate_proj.weight");
    const qus::Weight* up   = store.qweight("layers.0.mlp.up_proj.weight");
    failures += gate != nullptr ? 0 : fail("missing fused gate segment");
    failures += up != nullptr ? 0 : fail("missing fused up segment");
    if (gate != nullptr && up != nullptr) {
        failures += gate->n == 5 && gate->k == 7 ? 0 : fail("gate segment shape mismatch");
        failures += up->n == 4 && up->k == 7 ? 0 : fail("up segment shape mismatch");
        failures += gate->payload == up->payload ? 0 : fail("fused segments should share payload");
        failures +=
            gate->qdata != nullptr && gate->scales != nullptr ? 0 : fail("gate planes null");
        failures += up->qdata != nullptr && up->scales != nullptr ? 0 : fail("up planes null");
        failures += gate->qhigh == nullptr ? 0 : fail("gate qhigh should be null");
        failures += up->qhigh == nullptr ? 0 : fail("up qhigh should be null");
        failures += expect_plane_offsets(*gate, find_tensor(parsed, "layers.0.mlp.gateup"),
                                         find_segment(parsed, "layers.0.mlp.gate_proj.weight"),
                                         "gate segment");
        failures +=
            expect_plane_offsets(*up, find_tensor(parsed, "layers.0.mlp.gateup"),
                                 find_segment(parsed, "layers.0.mlp.up_proj.weight"), "up segment");
    }
    const auto& gateup_tensor = find_tensor(parsed, "layers.0.mlp.gateup");
    const qus::Weight* gateup = store.qfused(qus::ModuleKind::TextCore, /*MLP_GATEUP*/ 3, 0, 0);
    failures += gateup != nullptr ? 0 : fail("missing fused gate/up block");
    if (gateup != nullptr && gate != nullptr) {
        failures += gateup->n == static_cast<std::int32_t>(segment_row_sum(parsed, gateup_tensor))
                        ? 0
                        : fail("fused gate/up row sum mismatch");
        failures += gateup->k == 7 ? 0 : fail("fused gate/up K mismatch");
        failures += gateup->qtype == qus::QType::Q4G64_F16S ? 0 : fail("fused gate/up qtype");
        failures += gateup->source_kind == static_cast<std::uint32_t>(qus::SourceKind::Other)
                        ? 0
                        : fail("fused gate/up source_kind");
        failures +=
            gateup->qdata == gate->qdata ? 0 : fail("fused gate/up qdata should start at gate");
        failures +=
            gateup->scales == gate->scales ? 0 : fail("fused gate/up scales should start at gate");
    }

    const auto& text_tensor_meta   = find_tensor(parsed, "layers.0.input_layernorm.weight");
    const qus::Tensor* text_tensor = store.tensor("layers.0.input_layernorm.weight");
    failures += text_tensor != nullptr ? 0 : fail("missing text tensor");
    if (text_tensor != nullptr) {
        failures += text_tensor->data != nullptr ? 0 : fail("text tensor payload is null");
        failures += expect_device_bytes(text_tensor->data, payload_bytes(file, text_tensor_meta),
                                        "text tensor");
    }

    const qus::Weight* by_source = store.qweight(
        qus::ModuleKind::TextCore, static_cast<std::uint32_t>(qus::SourceKind::MlpDown), 0);
    failures += by_source == text_weight ? 0 : fail("source lookup mismatch");

    failures += store.qweight("mtp.fc.weight") == nullptr
                    ? 0
                    : fail("unselected MTP descriptor was published");
    failures += store.qweight("model.visual.patch_embed.proj.weight") == nullptr
                    ? 0
                    : fail("unselected VISION descriptor was published");
    return failures;
}

int expect_module_payload(const qus::WeightStore& store, const qus::ParsedQ5090File& parsed,
                          const std::vector<std::byte>& file, std::string_view name,
                          bool should_be_loaded) {
    const qus::ParsedQ5090Tensor& meta = find_tensor(parsed, name);
    const qus::Weight* weight          = store.qweight(name);
    if (!should_be_loaded) {
        return weight == nullptr ? 0 : fail("unselected quant descriptor was published");
    }
    if (weight == nullptr) { return fail("missing loaded quant weight"); }
    int failures = 0;
    failures += weight->payload != nullptr ? 0 : fail("expected loaded payload");
    failures +=
        weight->payload_bytes == meta.payload_bytes ? 0 : fail("module payload size mismatch");
    if (weight->payload != nullptr) {
        failures += expect_device_bytes(weight->payload, payload_bytes(file, meta), name);
    }
    return failures;
}

int expect_mtp_expectations_reject_swapped_attn_segments(const std::vector<std::byte>& valid,
                                                         const qus::ParsedQ5090File& parsed,
                                                         qus::DeviceContext& ctx) {
    std::vector<std::byte> bad         = valid;
    const std::uint64_t segment_offset = read_u64(bad, 200);
    const std::size_t k_index = find_segment_index(parsed, "mtp.layers.0.self_attn.k_proj.weight");
    const std::size_t gate_index = find_segment_index(parsed, "mtp.layers.0.self_attn.q_proj.gate");
    write_u32(bad, segment_offset + k_index * kSegmentRecordSize,
              static_cast<std::uint32_t>(qus::SourceKind::AttnGate));
    write_u32(bad, segment_offset + gate_index * kSegmentRecordSize,
              static_cast<std::uint32_t>(qus::SourceKind::AttnK));

    const auto path =
        std::filesystem::temp_directory_path() / "qus_q5090_bad_mtp_attn_segments.qus";
    write_file(path, bad);

    qus::LoadOptions options;
    options.load_mtp = true;
    qus::WeightStore store(expectations());
    store.load(path.c_str(), ctx, options);
    try {
        store.require_mtp_module_expectations();
    } catch (const std::runtime_error&) { return 0; }
    return fail("swapped MTP attn segment source kinds were accepted");
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures                             = 0;
    const std::filesystem::path fixture_path = make_fixture();
    const std::vector<std::byte> file        = read_file(fixture_path);
    const qus::ParsedQ5090File parsed        = qus::parse_q5090_file(file, expectations());
    qus::DeviceContext ctx(0);

    if (argc == 2 && std::string_view(argv[1]) == "--fatal-drain") {
        qus::WeightStore store(expectations());
        store.prepare(fixture_path.c_str());
        g_stream_sync_failures_remaining = 2;
        try {
            store.upload(ctx);
        } catch (...) {
            // A recoverable exception means the guard returned despite failing to prove the stream
            // quiescent. Only std::terminate inside the guard may make this subprocess nonzero.
            return 0;
        }
        return 0;
    }

    qus::WeightStore default_store(expectations());
    default_store.load(fixture_path.c_str(), ctx);
    failures += expect_default_text_load(default_store, parsed, file);
    const qus::DeviceArena* default_arena = default_store.module_arena(qus::ModuleKind::TextCore);
    failures += default_arena != nullptr ? 0 : fail("default TEXT arena missing");
    failures +=
        default_arena != nullptr && default_arena->used() == default_store.loaded_payload_bytes()
            ? 0
            : fail("default arena exact used bytes mismatch");
    failures += default_arena != nullptr &&
                        default_arena->capacity() == default_store.loaded_payload_bytes()
                    ? 0
                    : fail("default arena exact capacity mismatch");
    failures += default_store.load_stats().total_file_read_bytes ==
                        default_store.load_plan().file_read_bytes
                    ? 0
                    : fail("default planned/actual file read bytes mismatch");
    failures +=
        expect_throws<std::runtime_error>([&] { default_store.prepare(fixture_path.c_str()); },
                                          "prepare replacing resident weights without clear");
    failures += default_store.module_arena(qus::ModuleKind::TextCore) == default_arena
                    ? 0
                    : fail("rejected prepare destroyed resident arena");
    default_store.clear();
    failures += default_store.tensor_count() == 0 ? 0 : fail("clear tensor_count mismatch");
    failures += default_store.quant_count() == 0 ? 0 : fail("clear quant_count mismatch");
    failures += default_store.qfused(qus::ModuleKind::TextCore, 3, 0, 0) == nullptr
                    ? 0
                    : fail("clear fused lookup mismatch");
    failures += default_store.loaded_payload_bytes() == 0 ? 0 : fail("clear loaded bytes mismatch");
    failures += !default_store.module_loaded(qus::ModuleKind::TextCore)
                    ? 0
                    : fail("clear TEXT module still loaded");

    qus::LoadOptions mtp_options;
    mtp_options.load_mtp = true;
    qus::WeightStore mtp_store(expectations());
    mtp_store.load(fixture_path.c_str(), ctx, mtp_options);
    mtp_store.require_mtp_module_expectations();
    failures += mtp_store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP not loaded");
    failures += !mtp_store.module_loaded(qus::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("VISION loaded with MTP");
    failures += expect_module_payload(mtp_store, parsed, file, "mtp.fc.weight", true);
    const qus::Weight* mtp_attn = mtp_store.qfused(qus::ModuleKind::MtpDraft, /*ATTN_IN*/ 1, 0, 0);
    failures += mtp_attn != nullptr ? 0 : fail("missing MTP attn fused block");
    if (mtp_attn != nullptr) {
        failures += mtp_attn->qtype == qus::QType::W8G32_F16S ? 0 : fail("MTP attn qtype");
        failures += mtp_attn->group_size == 32 ? 0 : fail("MTP attn group_size");
        failures += mtp_attn->qhigh == nullptr ? 0 : fail("MTP attn qhigh");
        failures += mtp_attn->high_plane_bytes == 0 ? 0 : fail("MTP attn high_plane_bytes");
    }
    failures += expect_module_payload(mtp_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", false);

    qus::LoadOptions vision_options;
    vision_options.load_vision = true;
    qus::WeightStore vision_store(expectations());
    vision_store.load(fixture_path.c_str(), ctx, vision_options);
    failures +=
        vision_store.module_loaded(qus::ModuleKind::VisionEncoder) ? 0 : fail("VISION not loaded");
    failures +=
        !vision_store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP loaded with VISION");
    failures += expect_module_payload(vision_store, parsed, file, "mtp.fc.weight", false);
    failures += expect_module_payload(vision_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", true);

    qus::LoadOptions all_options;
    all_options.load_mtp    = true;
    all_options.load_vision = true;
    qus::WeightStore all_store(expectations());
    all_store.load(fixture_path.c_str(), ctx, all_options);
    all_store.require_mtp_module_expectations();
    failures +=
        all_store.module_loaded(qus::ModuleKind::TextCore) ? 0 : fail("TEXT not loaded with all");
    failures +=
        all_store.module_loaded(qus::ModuleKind::MtpDraft) ? 0 : fail("MTP not loaded with all");
    failures += all_store.module_loaded(qus::ModuleKind::VisionEncoder)
                    ? 0
                    : fail("VISION not loaded with all");
    failures += expect_module_payload(all_store, parsed, file, "mtp.fc.weight", true);
    failures += expect_module_payload(all_store, parsed, file,
                                      "model.visual.patch_embed.proj.weight", true);
    failures += expect_mtp_expectations_reject_swapped_attn_segments(file, parsed, ctx);

    qus::WeightStore callback_failure_store(expectations());
    qus::Q5090Progress throwing_progress;
    throwing_progress.min_interval_bytes = 1;
    throwing_progress.callback = [](std::string_view phase, std::uint64_t done, std::uint64_t) {
        if (phase == "upload selected payloads" && done > 0) {
            throw std::runtime_error("injected upload progress failure");
        }
    };
    failures += expect_throws<std::runtime_error>(
        [&] {
            qus::LoadOptions options;
            options.progress = &throwing_progress;
            callback_failure_store.load(fixture_path.c_str(), ctx, options);
        },
        "upload callback failure");
    failures += callback_failure_store.module_arena(qus::ModuleKind::TextCore) == nullptr &&
                        callback_failure_store.qweight("layers.0.mlp.down_proj.weight") == nullptr
                    ? 0
                    : fail("failed upload published resident state");
    failures += callback_failure_store.load_stats().fail_stage == "upload"
                    ? 0
                    : fail("failed upload stage was not recorded");

    qus::WeightStore split_progress_store(expectations());
    int upload_progress_calls = 0;
    {
        qus::Q5090Progress short_lived_progress;
        short_lived_progress.min_interval_bytes = 1;
        short_lived_progress.callback           = [&](std::string_view phase, std::uint64_t done,
                                            std::uint64_t) {
            if (phase == "upload selected payloads" && done > 0) { ++upload_progress_calls; }
        };
        qus::LoadOptions options;
        options.progress = &short_lived_progress;
        split_progress_store.prepare(fixture_path.c_str(), options);
    }
    split_progress_store.upload(ctx);
    failures += upload_progress_calls > 0
                    ? 0
                    : fail("split prepare/upload did not retain progress callback safely");

    qus::WeightStore reentrant_clear_store(expectations());
    qus::Q5090Progress reentrant_clear_progress;
    reentrant_clear_progress.min_interval_bytes = 1;
    reentrant_clear_progress.callback           = [&](std::string_view phase, std::uint64_t done,
                                            std::uint64_t) {
        if (phase == "upload selected payloads" && done > 0) { reentrant_clear_store.clear(); }
    };
    {
        qus::LoadOptions options;
        options.progress = &reentrant_clear_progress;
        reentrant_clear_store.load(fixture_path.c_str(), ctx, options);
    }
    failures += reentrant_clear_store.module_arena(qus::ModuleKind::TextCore) == nullptr &&
                        reentrant_clear_store.tensor_count() == 0
                    ? 0
                    : fail("progress callback clear was not safely deferred");

    qus::WeightStore event_failure_store(expectations());
    event_failure_store.prepare(fixture_path.c_str());
    g_lifetime_sequence      = 0;
    g_first_successful_sync  = 0;
    g_first_upload_free      = 0;
    g_track_upload_lifetime  = true;
    g_fail_next_event_record = true;
    failures += expect_throws<std::runtime_error>([&] { event_failure_store.upload(ctx); },
                                                  "injected cudaEventRecord failure");
    g_track_upload_lifetime = false;
    failures += event_failure_store.load_stats().h2d_bytes > 0 &&
                        event_failure_store.module_arena(qus::ModuleKind::TextCore) == nullptr
                    ? 0
                    : fail("event failure did not drain and roll back staged upload");
    failures += g_first_successful_sync > 0 && g_first_upload_free > g_first_successful_sync
                    ? 0
                    : fail("event failure freed upload storage before stream drain");
    event_failure_store.upload(ctx);
    failures += event_failure_store.module_loaded(qus::ModuleKind::TextCore)
                    ? 0
                    : fail("event failure retry did not publish TEXT");
    failures += event_failure_store.load_stats().total_file_read_bytes ==
                            event_failure_store.load_plan().file_read_bytes &&
                        event_failure_store.load_stats().h2d_bytes ==
                            event_failure_store.load_plan().h2d_bytes
                    ? 0
                    : fail("event failure retry retained prior-attempt byte counters");

    qus::WeightStore stream_failure_store(expectations());
    stream_failure_store.prepare(fixture_path.c_str());
    g_lifetime_sequence              = 0;
    g_first_successful_sync          = 0;
    g_first_upload_free              = 0;
    g_track_upload_lifetime          = true;
    g_stream_sync_failures_remaining = 1;
    failures += expect_throws<std::runtime_error>([&] { stream_failure_store.upload(ctx); },
                                                  "injected cudaStreamSynchronize failure");
    g_track_upload_lifetime = false;
    failures += stream_failure_store.module_arena(qus::ModuleKind::TextCore) == nullptr
                    ? 0
                    : fail("stream wait failure published an arena");
    failures += g_first_successful_sync > 0 && g_first_upload_free > g_first_successful_sync
                    ? 0
                    : fail("stream wait failure freed upload storage before guard drain retry");
    stream_failure_store.upload(ctx);
    failures += stream_failure_store.module_loaded(qus::ModuleKind::TextCore)
                    ? 0
                    : fail("stream wait failure retry did not publish TEXT");
    failures += stream_failure_store.load_stats().total_file_read_bytes ==
                            stream_failure_store.load_plan().file_read_bytes &&
                        stream_failure_store.load_stats().h2d_bytes ==
                            stream_failure_store.load_plan().h2d_bytes
                    ? 0
                    : fail("stream wait failure retry retained prior-attempt byte counters");

    const auto short_read_path =
        std::filesystem::temp_directory_path() / "qus_q5090_weight_store_short_read.qus";
    std::filesystem::copy_file(fixture_path, short_read_path,
                               std::filesystem::copy_options::overwrite_existing);
    qus::WeightStore short_read_store(expectations());
    bool truncated = false;
    qus::Q5090Progress truncate_progress;
    truncate_progress.min_interval_bytes = 1;
    truncate_progress.callback = [&](std::string_view phase, std::uint64_t done, std::uint64_t) {
        if (phase != "upload selected payloads" || done != 0 || truncated) { return; }
        const auto& module = short_read_store.load_plan().modules.front();
        std::filesystem::resize_file(short_read_path, module.file_offset + module.file_bytes / 2);
        truncated = true;
    };
    {
        qus::LoadOptions options;
        options.progress = &truncate_progress;
        short_read_store.prepare(short_read_path.c_str(), options);
    }
    failures += expect_throws<std::runtime_error>([&] { short_read_store.upload(ctx); },
                                                  "injected partial payload read");
    failures += truncated && short_read_store.load_stats().h2d_bytes == 0 &&
                        short_read_store.load_stats().total_file_read_bytes >
                            short_read_store.load_plan().modules.front().file_offset &&
                        short_read_store.load_stats().total_file_read_bytes <
                            short_read_store.load_plan().file_read_bytes
                    ? 0
                    : fail("partial pread bytes were not retained in failed-attempt stats");

    failures += expect_throws<std::invalid_argument>(
        [&] {
            qus::LoadOptions options;
            options.load_lm_head_draft = true;
            qus::WeightStore store(expectations());
            store.prepare(fixture_path.c_str(), options);
        },
        "draft residency without MTP");

    failures += expect_throws<std::runtime_error>(
        [&] {
            qus::LoadOptions options;
            options.required_text_tensors.push_back("missing.text.tensor");
            qus::WeightStore store(expectations());
            store.prepare(fixture_path.c_str(), options);
        },
        "missing required text tensor");

    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
    const std::string death_command =
        "ulimit -c 0; \"" + self.string() + "\" --fatal-drain >/dev/null 2>&1";
    failures += std::system(death_command.c_str()) != 0
                    ? 0
                    : fail("double stream-sync failure did not terminate the subprocess");

    return failures == 0 ? 0 : fail("weight store q5090 test failed");
}
