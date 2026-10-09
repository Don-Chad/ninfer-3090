// Direct prompt grafts against the model's own prefill, on the real artifact.
//
// The oracle: a direct_kv graft whose K/V and Gated DeltaNet state are exactly what the target
// computes for a token prefix T must make a request continue exactly as the same request with T
// replayed as a prefill_kv graft. The fixture prefills T in a bare Program with BF16 KV storage,
// reads the attention K/V rows and the StateImage back, and writes both containers. The Engine
// phases then compare greedy output of the two grafts:
//   * no speculative backend, BF16 KV: token-identical, solo, repeated and at concurrency 2, and
//     still identical after context-cache pressure has evicted unrelated history (the pinned graft
//     checkpoint must survive every reclaim);
//   * rk4v4 KV (the 3090 launchers' storage): the BF16-storage capture is not the state rk4v4
//     prefill computes (its later layers read quantized history), so only binding and completion
//     are required and the first divergence is reported. The first attention layer, whose inputs
//     agree, checks that installation's standalone append stores the bytes prefill's fused append
//     stores;
//   * MTP and DFlash2: the draft context over a direct graft is zero-filled, so drafts and verify
//     widths differ from the replayed graft. The target still verifies every token, so the output
//     is required to match until a reported divergence, and every request must speculate, start
//     from the graft and finish its budget.
//
// NINFER_TEST_GRAFT_REFERENCE=<phantom-kv prefill_kv container> takes T from that container's
// replay ids and compares the captured tensors with its independently produced (Hugging Face)
// tensors, an oracle for the safetensors layout that the round trip above cannot see: a transposed
// recurrent state round-trips perfectly but disagrees with the reference.
#include "core/device.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include "ninfer/engine.h"
#include "ninfer/ops/kv_cache_append.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen      = models::qwen3_5;
namespace execution = qwen::execution;
using Json          = nlohmann::json;

constexpr std::uint32_t kMaxContext   = 4096;
constexpr std::uint32_t kPrefillChunk = 1024;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// The default prefix: a system turn and an acknowledged assistant turn, the shape trained grafts have.
constexpr const char* kPrefixText =
    "<|im_start|>system\nYou are a terse pirate. You answer every question in one short sentence of "
    "pirate speech and you always mention the sea.<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
    "</think>\n\nAye, I will answer every question in one short pirate sentence about the sea."
    "<|im_end|>\n";

struct TensorBytes {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> bytes;
};

struct Captured {
    qwen::TextConfig text;
    std::vector<TokenId> tokens;
    std::map<std::string, TensorBytes> tensors; // k, v, conv, rec
};

float bf16_value(const std::uint8_t* p) {
    std::uint16_t bits;
    std::memcpy(&bits, p, 2);
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
    float value;
    std::memcpy(&value, &wide, 4);
    return value;
}

float half_value(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000U) << 16U;
    std::uint32_t exponent   = (h >> 10U) & 0x1fU;
    std::uint32_t mantissa   = h & 0x3ffU;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1U;
                --exponent;
            }
            bits = sign | (exponent << 23U) | ((mantissa & 0x3ffU) << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23U) | (mantissa << 13U);
    }
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}

std::vector<std::uint8_t> device_bytes(const Tensor& tensor) {
    std::vector<std::uint8_t> bytes(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(bytes.data(), tensor.data, bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}

std::vector<TokenId> prefix_tokens(const qwen::Model& model) {
    qwen::FrontendOptions options;
    options.vision_enabled = false;
    options.max_context    = kMaxContext;
    return qwen::make_frontend(model.resources(), options).tokenize_text(kPrefixText);
}

// --- safetensors container I/O (phantom-kv format_version 1) ---

std::map<std::string, TensorBytes> read_container(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "cannot open " + path.string());
    std::vector<std::uint8_t> file((std::istreambuf_iterator<char>(in)), {});
    require(file.size() >= 8, "container is truncated");
    std::uint64_t header_bytes = 0;
    for (int byte = 7; byte >= 0; --byte) { header_bytes = (header_bytes << 8U) | file[byte]; }
    const Json header = Json::parse(file.begin() + 8, file.begin() + 8 + header_bytes);
    const std::size_t payload = 8 + header_bytes;
    std::map<std::string, TensorBytes> tensors;
    for (const auto& [name, entry] : header.items()) {
        if (name == "__metadata__") { continue; }
        TensorBytes tensor{entry.at("dtype").get<std::string>(),
                           entry.at("shape").get<std::vector<std::uint64_t>>(),
                           {}};
        const auto offsets = entry.at("data_offsets").get<std::vector<std::uint64_t>>();
        tensor.bytes.assign(file.begin() + payload + offsets[0], file.begin() + payload + offsets[1]);
        tensors.emplace(name, std::move(tensor));
    }
    return tensors;
}

std::filesystem::path write_container(const std::filesystem::path& dir, const std::string& stem,
                                      const std::map<std::string, TensorBytes>& tensors,
                                      Json meta) {
    Json header = Json::object();
    std::vector<std::uint8_t> payload;
    for (const auto& [name, tensor] : tensors) {
        const std::uint64_t begin = payload.size();
        payload.insert(payload.end(), tensor.bytes.begin(), tensor.bytes.end());
        header[name] = {{"dtype", tensor.dtype},
                        {"shape", tensor.shape},
                        {"data_offsets", {begin, payload.size()}}};
    }
    meta["sha256"]                = qwen::frontend::sha256_hex(qwen::frontend::sha256(payload));
    const std::string header_text = header.dump();
    std::vector<std::uint8_t> file(8);
    for (int byte = 0; byte < 8; ++byte) {
        file[byte] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(header_text.size()) >>
                                               (8U * byte));
    }
    file.insert(file.end(), header_text.begin(), header_text.end());
    file.insert(file.end(), payload.begin(), payload.end());
    const auto bin = dir / (stem + ".bin");
    std::ofstream(bin, std::ios::binary)
        .write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    std::ofstream(dir / (stem + ".json")) << meta.dump(2);
    return bin;
}

Json container_meta(const qwen::TextConfig& text, std::uint32_t slots, const char* kind) {
    Json layer_types = Json::array();
    for (const auto kind_of_layer : text.layer_types) {
        layer_types.push_back(kind_of_layer == qwen::MixerKind::FullAttention ? "full_attention"
                                                                              : "linear_attention");
    }
    Json meta = {{"format_version", 1},
                 {"kind", kind},
                 {"model_id", "ninfer-capture"},
                 {"layer_types", layer_types},
                 {"n_layers", text.layer_types.size()},
                 {"n_attn_layers", text.full_attention_layers},
                 {"n_slots", slots},
                 {"n_kv_heads", text.attention->num_key_value_heads},
                 {"head_dim", text.attention->head_dim},
                 {"conv_dim", text.gdn->conv_channels()},
                 {"conv_k", text.gdn->linear_conv_kernel_dim},
                 {"rec_shape",
                  {text.gdn->linear_num_value_heads, text.gdn->linear_key_head_dim,
                   text.gdn->linear_value_head_dim}},
                 {"quant", "none"}};
    if (std::string(kind) == "prefill_kv") { meta["replay"] = "ids"; }
    return meta;
}

// --- capture: prefill T in a bare Program and read its K/V rows and StateImage back ---

struct AddressOwner {
    qwen::detail::KVAddressSpaceStore* store = nullptr;
    qwen::detail::KVAddressSpaceHandle handle;
    ~AddressOwner() {
        if (store && store->active(handle)) { store->deactivate(handle); }
        if (store && !store->release_after_deactivate(handle)) { std::terminate(); }
    }
};

Captured capture(const char* artifact, std::optional<std::vector<TokenId>> reference_tokens) {
    DeviceContext device;
    models::LoadOptions selected;
    selected.speculative   = SpeculativeBackend::None;
    selected.proposal_head = ProposalHead::Full;
    auto model             = qwen::load_model(artifact, selected, device);
    const execution::Parameters parameters(*model);
    const auto& text = model->config().text;

    Captured out;
    out.text            = text;
    out.tokens          = reference_tokens ? *reference_tokens : prefix_tokens(*model);
    const auto slots    = static_cast<std::uint32_t>(out.tokens.size());
    const auto capacity = kMaxContext;
    const auto pages    = capacity / static_cast<std::uint32_t>(kPagedKVPageSize);
    require(slots > 0 && slots < capacity, "prefix does not fit the capture Program");

    EngineOptions options;
    options.max_context                       = capacity;
    options.prefill_chunk                     = kPrefillChunk;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(capacity);
    options.max_concurrency                   = 1;
    options.context_cache.device_state_slots  = 1;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph                    = false;
    options.kv_cache                          = KvCacheStorage::BFloat16;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});

    const auto state = program.state_store->reserve_reset(device.stream);
    require(state.has_value(), "capture state allocation failed");
    const auto slot = program.state_store->physical_slot(*state);
    const auto text_address =
        program.text_kv_addresses->create_active(pages, 0, device.stream);
    require(text_address.has_value(), "capture KV allocation failed");
    AddressOwner owner{program.text_kv_addresses.get(), *text_address};
    program.text_kv_addresses->ensure_mapped_to_tokens(*text_address, slots, device.stream);
    const auto view = program.decoder->text_kv.execution_view(
        program.text_kv_addresses->execution_row(*text_address));
    execution::PrefillContext context{
        {device, parameters, program.work, program.state_images->linear(),
         program.replay_records ? &*program.replay_records : nullptr, program.io,
         program.prefill_hidden, program.prefill_chunk, program.proposal_head},
        view,
        {},
        program.decoder->text_kv,
        nullptr,
        nullptr,
        0,
        nullptr,
        nullptr,
        slot,
        slot,
        0,
        0,
        nullptr};
    // The Engine prefills a prompt in chunks of prefill_chunk from the root and splits at the
    // graft frontier, its shared-prefix capture. Reproduce that decomposition exactly.
    while (context.text_kv_base < slots) {
        const auto nominal = std::min(kPrefillChunk, slots - context.text_kv_base);
        const auto result =
            execution::prefill_text_chunk(context, out.tokens, nominal, std::nullopt, false);
        require(result.processed_tokens == nominal, "capture prefill made partial progress");
        context.text_kv_base += result.processed_tokens;
    }
    device.synchronize();

    // Attention K/V rows, gathered through the block table into [layers, slots, heads, dim].
    const auto heads    = static_cast<std::uint64_t>(text.attention->num_key_value_heads);
    const auto head_dim = static_cast<std::uint64_t>(text.attention->head_dim);
    const auto table    = program.decoder->text_kv.execution_tables().matrix().slice(1, 0, 1);
    std::vector<std::int32_t> physical((slots + kPagedKVPageSize - 1) / kPagedKVPageSize);
    CUDA_CHECK(cudaMemcpy(physical.data(), table.data, physical.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    const bool page_major = program.decoder->text_kv.page_pool().geometry().device_plane_order ==
                            PagedKVPlaneOrder::PageMajor;
    const std::uint64_t row = heads * head_dim * 2;
    for (const char* name : {"k", "v"}) {
        TensorBytes tensor{"BF16", {text.full_attention_layers, slots, heads, head_dim}, {}};
        tensor.bytes.resize(text.full_attention_layers * slots * row);
        for (std::uint32_t layer = 0; layer < text.full_attention_layers; ++layer) {
            const auto layer_view = view.layer_view(layer);
            const Tensor& plane   = *name == 'k' ? layer_view.k_pages : layer_view.v_pages;
            require(plane.dtype == DType::BF16 && plane.ne[0] == static_cast<int>(head_dim),
                    "capture expects BF16 KV planes");
            const auto host = device_bytes(plane);
            for (std::uint32_t position = 0; position < slots; ++position) {
                const auto page   = physical[position / kPagedKVPageSize];
                const auto column = position % kPagedKVPageSize;
                for (std::uint64_t head = 0; head < heads; ++head) {
                    const std::size_t source =
                        column * plane.nb[1] + (page_major ? head * plane.nb[2] + page * plane.nb[3]
                                                           : page * plane.nb[2] + head * plane.nb[3]);
                    std::memcpy(tensor.bytes.data() + layer * slots * row + position * row +
                                    head * head_dim * 2,
                                host.data() + source, head_dim * 2);
                }
            }
        }
        out.tensors.emplace(name, std::move(tensor));
    }

    // StateImage, transposed into the container layout (the inverse of installation).
    const auto& layout = program.state_images->host_layout();
    std::vector<std::byte> image(layout.image_bytes);
    program.state_images->copy_to_host(slot, qwen::HostStateImageView{image.data(), &layout},
                                       RankStreams(device.stream));
    device.synchronize();
    const auto& linear        = layout.spec.linear;
    const std::uint64_t C     = static_cast<std::uint64_t>(linear.conv_channels);
    const std::uint64_t W     = static_cast<std::uint64_t>(linear.conv_width);
    const std::uint64_t K     = text.gdn->linear_conv_kernel_dim;
    const std::uint64_t Hv    = static_cast<std::uint64_t>(linear.value_heads);
    const std::uint64_t Dk    = static_cast<std::uint64_t>(linear.key_head_dim);
    const std::uint64_t Dv    = static_cast<std::uint64_t>(linear.value_head_dim);
    const std::uint64_t L     = linear.layers;
    const bool fp32_recurrent = linear.recurrent_dtype == DType::FP32;
    TensorBytes conv{"BF16", {L, C, K}, std::vector<std::uint8_t>(L * C * K * 2, 0)};
    TensorBytes rec{"F32", {L, Hv, Dk, Dv}, std::vector<std::uint8_t>(L * Hv * Dk * Dv * 4)};
    for (std::uint64_t layer = 0; layer < L; ++layer) {
        const std::byte* conv_layer =
            image.data() + layout.linear_conv.offset + layer * layout.linear_conv_layer_bytes;
        for (std::uint64_t w = 0; w < W; ++w) {
            for (std::uint64_t c = 0; c < C; ++c) { // the oldest K - W container columns stay zero
                std::memcpy(conv.bytes.data() + ((layer * C + c) * K + (K - W) + w) * 2,
                            conv_layer + (c + w * C) * 2, 2);
            }
        }
        const std::byte* rec_layer = image.data() + layout.linear_recurrent.offset +
                                     layer * layout.linear_recurrent_layer_bytes;
        for (std::uint64_t h = 0; h < Hv; ++h) {
            for (std::uint64_t k = 0; k < Dk; ++k) {
                for (std::uint64_t v = 0; v < Dv; ++v) {
                    const std::uint64_t source = k + v * Dk + h * Dk * Dv;
                    float value;
                    if (fp32_recurrent) {
                        std::memcpy(&value, rec_layer + source * 4, 4);
                    } else {
                        std::uint16_t half;
                        std::memcpy(&half, rec_layer + source * 2, 2);
                        value = half_value(half);
                    }
                    std::memcpy(rec.bytes.data() + (((layer * Hv + h) * Dk + k) * Dv + v) * 4,
                                &value, 4);
                }
            }
        }
    }
    out.tensors.emplace("conv", std::move(conv));
    out.tensors.emplace("rec", std::move(rec));
    std::cout << "captured " << slots << " slots (recurrent state "
              << (fp32_recurrent ? "FP32" : "FP16") << ")\n";
    return out;
}

// Quantized KV storage: the bytes prefill's fused append stores for T against the bytes the
// standalone append stores from the captured BF16 rows (what installation does), on the first
// attention layer, whose inputs agree in both runs. Every row must be byte-identical.
void quantized_append_agreement(const char* artifact, const Captured& captured,
                                KvCacheStorage storage) {
    DeviceContext device;
    models::LoadOptions selected;
    selected.speculative   = SpeculativeBackend::None;
    selected.proposal_head = ProposalHead::Full;
    auto model             = qwen::load_model(artifact, selected, device);
    const execution::Parameters parameters(*model);
    const auto slots = static_cast<std::uint32_t>(captured.tokens.size());
    const auto pages = kMaxContext / static_cast<std::uint32_t>(kPagedKVPageSize);
    EngineOptions options;
    options.max_context                       = kMaxContext;
    options.prefill_chunk                     = kPrefillChunk;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(2 * kMaxContext);
    options.max_concurrency                   = 2;
    options.context_cache.device_state_slots  = 1;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph                    = false;
    options.kv_cache                          = storage;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 2 * pages);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    const auto state = program.state_store->reserve_reset(device.stream);
    require(state.has_value(), "diagnostic state allocation failed");
    const auto slot     = program.state_store->physical_slot(*state);
    auto& addresses     = *program.text_kv_addresses;
    const auto prefill  = addresses.create_active(pages, 0, device.stream);
    const auto appended = addresses.create_active(pages, 1, device.stream);
    require(prefill && appended, "diagnostic KV allocation failed");
    AddressOwner prefill_owner{&addresses, *prefill};
    AddressOwner appended_owner{&addresses, *appended};
    addresses.ensure_mapped_to_tokens(*prefill, slots, device.stream);
    addresses.ensure_mapped_to_tokens(*appended, slots, device.stream);
    const auto prefill_view = program.decoder->text_kv.execution_view(addresses.execution_row(*prefill));
    const auto append_view =
        program.decoder->text_kv.execution_view(addresses.execution_row(*appended));
    execution::PrefillContext context{
        {device, parameters, program.work, program.state_images->linear(),
         program.replay_records ? &*program.replay_records : nullptr, program.io,
         program.prefill_hidden, program.prefill_chunk, program.proposal_head},
        prefill_view, {}, program.decoder->text_kv, nullptr, nullptr, 0, nullptr, nullptr, slot,
        slot, 0, 0, nullptr};
    while (context.text_kv_base < slots) {
        const auto nominal = std::min(kPrefillChunk, slots - context.text_kv_base);
        (void)execution::prefill_text_chunk(context, captured.tokens, nominal, std::nullopt, false);
        context.text_kv_base += nominal;
    }
    const auto& text    = captured.text;
    const auto heads    = static_cast<std::int32_t>(text.attention->num_key_value_heads);
    const auto head_dim = static_cast<std::int32_t>(text.attention->head_dim);
    const std::size_t row = static_cast<std::size_t>(heads) * head_dim * 2;
    program.work.reset();
    Tensor positions = program.work.alloc(DType::I32, {static_cast<std::int32_t>(slots)});
    Tensor k = program.work.alloc(DType::BF16, {head_dim, heads, static_cast<std::int32_t>(slots)});
    Tensor v = program.work.alloc(DType::BF16, {head_dim, heads, static_cast<std::int32_t>(slots)});
    std::vector<std::int32_t> host_positions(slots);
    for (std::uint32_t i = 0; i < slots; ++i) { host_positions[i] = static_cast<std::int32_t>(i); }
    CUDA_CHECK(cudaMemcpy(positions.data, host_positions.data(), slots * 4, cudaMemcpyHostToDevice));
    for (std::uint32_t layer = 0; layer < text.full_attention_layers; ++layer) {
        const std::size_t offset = layer * slots * row;
        CUDA_CHECK(cudaMemcpy(k.data, captured.tensors.at("k").bytes.data() + offset, slots * row,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v.data, captured.tensors.at("v").bytes.data() + offset, slots * row,
                              cudaMemcpyHostToDevice));
        ops::kv_cache_append(k, v, positions, append_view.layer_view(layer), device.stream);
        device.synchronize();
    }
    const auto tables = program.decoder->text_kv.execution_tables().matrix();
    const auto table_row = [&](std::int32_t r) {
        std::vector<std::int32_t> physical((slots + kPagedKVPageSize - 1) / kPagedKVPageSize);
        CUDA_CHECK(cudaMemcpy(physical.data(), tables.slice(1, r, 1).data,
                              physical.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        return physical;
    };
    const auto first  = table_row(0);
    const auto second = table_row(1);
    const bool page_major = program.decoder->text_kv.page_pool().geometry().device_plane_order ==
                            PagedKVPlaneOrder::PageMajor;
    // Only the first attention layer sees identical inputs in both runs: from then on, prefill in
    // a quantized storage reads its own quantized history, so later layers' K/V legitimately differ
    // from the BF16-storage capture.
    std::array<std::uint64_t, 4> differing{}, total{};
    for (std::uint32_t layer = 0; layer < 1; ++layer) {
        const auto view = prefill_view.layer_view(layer);
        const std::array<const Tensor*, 4> planes{&view.k_pages, &view.v_pages, &view.k_scale_pages,
                                                  &view.v_scale_pages};
        for (std::size_t index = 0; index < planes.size(); ++index) {
            const Tensor& plane = *planes[index];
            if (plane.data == nullptr) { continue; }
            const auto host  = device_bytes(plane);
            const auto bytes = static_cast<std::size_t>(plane.nb[1]);
            for (std::uint32_t position = 0; position < slots; ++position) {
                const auto column = position % kPagedKVPageSize;
                for (std::int64_t head = 0; head < (page_major ? plane.ne[2] : plane.ne[3]); ++head) {
                    const auto at = [&](std::int32_t page) {
                        return column * plane.nb[1] + (page_major ? head * plane.nb[2] + page * plane.nb[3]
                                                                  : page * plane.nb[2] + head * plane.nb[3]);
                    };
                    ++total[index];
                    if (std::memcmp(host.data() + at(first[position / kPagedKVPageSize]),
                                    host.data() + at(second[position / kPagedKVPageSize]), bytes) != 0) {
                        ++differing[index];
                    }
                }
            }
        }
    }
    program.work.reset();
    std::cout << "quantized append agreement, first attention layer (differing rows / rows): k " << differing[0] << "/"
              << total[0] << ", v " << differing[1] << "/" << total[1] << ", k scale "
              << differing[2] << "/" << total[2] << ", v scale " << differing[3] << "/" << total[3]
              << '\n';
    require(differing == std::array<std::uint64_t, 4>{} && total[0] != 0,
            "installation's append stores other quantized bytes than prefill for the same rows");
}

// --- reference comparison ---

double cosine(const TensorBytes& a, const TensorBytes& b, std::size_t begin, std::size_t end,
              std::size_t stride, std::size_t inner_begin, std::size_t inner_end,
              const std::function<std::size_t(std::size_t)>& b_index = {}) {
    double dot = 0, na = 0, nb = 0;
    const bool f32 = a.dtype == "F32";
    for (std::size_t outer = begin; outer < end; ++outer) {
        for (std::size_t inner = inner_begin; inner < inner_end; ++inner) {
            const std::size_t i = outer * stride + inner;
            const std::size_t j = b_index ? b_index(i) : i;
            double x, y;
            if (f32) {
                float fx, fy;
                std::memcpy(&fx, a.bytes.data() + i * 4, 4);
                std::memcpy(&fy, b.bytes.data() + j * 4, 4);
                x = fx;
                y = fy;
            } else {
                x = bf16_value(a.bytes.data() + i * 2);
                y = bf16_value(b.bytes.data() + j * 2);
            }
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
    }
    return na == 0 || nb == 0 ? 0.0 : dot / std::sqrt(na * nb);
}

void compare_reference(const Captured& captured, const std::map<std::string, TensorBytes>& reference) {
    for (const char* name : {"k", "v", "conv", "rec"}) {
        require(reference.contains(name) &&
                    reference.at(name).shape == captured.tensors.at(name).shape &&
                    reference.at(name).dtype == captured.tensors.at(name).dtype,
                std::string("reference tensor '") + name + "' does not match the capture shape");
    }
    const auto& k = captured.tensors.at("k");
    const auto& v = captured.tensors.at("v");
    const double k_cos =
        cosine(k, reference.at("k"), 0, k.bytes.size() / 2, 1, 0, 1);
    const double v_cos =
        cosine(v, reference.at("v"), 0, v.bytes.size() / 2, 1, 0, 1);
    // conv: only the trailing conv_width columns are carried by the runtime.
    const auto& conv  = captured.tensors.at("conv");
    const auto K      = conv.shape[2];
    const auto rows   = conv.shape[0] * conv.shape[1];
    const double c_cos = cosine(conv, reference.at("conv"), 0, rows, K, 1, K);
    const auto& rec   = captured.tensors.at("rec");
    const auto Dk     = rec.shape[2];
    const auto Dv     = rec.shape[3];
    const auto blocks = rec.shape[0] * rec.shape[1];
    const double r_cos = cosine(rec, reference.at("rec"), 0, blocks, Dk * Dv, 0, Dk * Dv);
    // The same state with Dk and Dv exchanged: what a transposition bug would produce.
    const double r_transposed =
        cosine(rec, reference.at("rec"), 0, blocks, Dk * Dv, 0, Dk * Dv, [&](std::size_t i) {
            const std::size_t block = i / (Dk * Dv), within = i % (Dk * Dv);
            const std::size_t kk = within / Dv, vv = within % Dv;
            return block * Dk * Dv + vv * Dk + kk;
        });
    std::cout << "reference cosine: k " << k_cos << ", v " << v_cos << ", conv " << c_cos
              << ", rec " << r_cos << " (transposed rec " << r_transposed << ")\n";
    require(k_cos > 0.9 && v_cos > 0.9 && c_cos > 0.9 && r_cos > 0.9,
            "captured tensors disagree with the independently produced reference graft");
    require(r_cos > r_transposed + 0.2,
            "the recurrent-state comparison cannot tell its layout from the transposed one");
}

// --- Engine phases ---

RequestOptions greedy(std::uint32_t outputs) {
    RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 0;
    request.stop.include_model_defaults       = false;
    return request;
}

PromptInput question(const std::string& graft, const std::string& text) {
    PromptInput input;
    input.messages.push_back(
        ChatMessage{.role = ChatRole::User, .parts = {MessagePart{.text = text}}});
    input.options.graft           = graft;
    input.options.enable_thinking = false;
    return input;
}

std::string first_difference(const std::vector<TokenId>& a, const std::vector<TokenId>& b) {
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    return i == a.size() && i == b.size() ? "identical"
                                          : "first difference at token " + std::to_string(i);
}

struct Phase {
    const char* name;
    SpeculativeBackend backend = SpeculativeBackend::None;
    KvCacheStorage kv          = KvCacheStorage::BFloat16;
    bool exact                 = true;
    bool pressure              = false;
};

EngineOptions engine_options(const char* artifact, const Phase& phase,
                             const std::filesystem::path& text_graft,
                             const std::filesystem::path& direct_graft) {
    EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = kMaxContext;
    options.prefill_chunk   = kPrefillChunk;
    options.max_concurrency = 2;
    options.kv_cache        = phase.kv;
    options.enable_vision   = false;
    // The pressure phase leaves room for two full requests and no Host tier, so unrelated history
    // must be released to admit more; the graft's pages are granted on top of that capacity.
    options.kv_capacity = phase.pressure ? KvCapacityPolicy::explicit_capacity(2 * kMaxContext)
                                         : KvCapacityPolicy::explicit_capacity(2 * kMaxContext);
    options.context_cache.device_state_slots  = phase.pressure ? 1U : 4U;
    options.context_cache.host_capacity_bytes = 0;
    options.speculative.backend               = phase.backend;
    // The registered DFlash2 proposal profile is the 7-token window the DFlash2 real test uses.
    options.speculative.draft_tokens = phase.backend == SpeculativeBackend::None      ? 0
                                       : phase.backend == SpeculativeBackend::DFlash2 ? 7
                                                                                      : 3;
    if (phase.backend == SpeculativeBackend::DFlash2) {
        options.speculative.proposal_head = ProposalHead::Optimized;
    }
    options.grafts = {GraftSource{.name = "text", .path = text_graft},
                      GraftSource{.name = "direct", .path = direct_graft}};
    return options;
}

int run_phase(const char* artifact, const Phase& phase, std::uint32_t slots,
              const std::filesystem::path& text_graft, const std::filesystem::path& direct_graft) {
    std::cout << "== " << phase.name << '\n';
    Engine engine(engine_options(artifact, phase, text_graft, direct_graft));
    int failures      = 0;
    const auto expect = [&](bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "FAIL [" << phase.name << "]: " << message << '\n';
            ++failures;
        }
    };
    const auto speculated = [&](const GenerationResult& result) {
        return phase.backend == SpeculativeBackend::None ||
               (result.speculative.backend == phase.backend &&
                result.speculative.rounds + result.speculative.fallback_steps != 0);
    };
    const auto compare = [&](const GenerationResult& direct, const GenerationResult& text,
                             const std::string& what) {
        const auto verdict = first_difference(direct.generated_token_ids, text.generated_token_ids);
        std::cout << "  " << what << ": " << verdict << " (direct reused "
                  << direct.reused_prompt_tokens << "/" << direct.prompt.prompt_tokens
                  << ", replayed reused " << text.reused_prompt_tokens << ")";
        if (phase.backend != SpeculativeBackend::None) {
            std::cout << " accepted " << direct.speculative.accepted_tokens << "/"
                      << direct.speculative.drafted_tokens << " vs "
                      << text.speculative.accepted_tokens << "/" << text.speculative.drafted_tokens;
        }
        std::cout << '\n';
        expect(direct.reused_prompt_tokens >= slots, what + ": direct graft was not bound");
        expect(direct.prompt.prompt_tokens == text.prompt.prompt_tokens,
               what + ": grafted prompts differ in length");
        expect(direct.finish_reason == FinishReason::OutputLimit &&
                   text.finish_reason == FinishReason::OutputLimit,
               what + ": a request did not finish its budget");
        expect(speculated(direct) && speculated(text), what + ": a request did not speculate");
        if (phase.exact) {
            expect(direct.generated_token_ids == text.generated_token_ids,
                   what + ": direct graft output differs from the replayed prefix");
        }
    };
    const std::string asked = "Who are you? Answer in one sentence.";
    const auto direct = engine.generate(engine.prepare(question("direct", asked)), greedy(32));
    // The first replayed request prefills the graft tokens; the second reads them from the shared
    // prefix the first published, exactly as the direct request reads the installed graft.
    const auto cold = engine.generate(engine.prepare(question("text", asked)), greedy(32));
    const auto text = engine.generate(engine.prepare(question("text", asked)), greedy(32));
    expect(cold.reused_prompt_tokens == 0 && text.reused_prompt_tokens >= slots,
           "the replayed graft was not prefilled once and then reused");
    compare(direct, text, "solo");
    std::cout << "  replayed cold vs reused: "
              << first_difference(cold.generated_token_ids, text.generated_token_ids) << '\n';
    if (phase.kv == KvCacheStorage::BFloat16 && phase.backend == SpeculativeBackend::None) {
        expect(direct.generated_token_ids == cold.generated_token_ids,
               "direct graft output differs from a cold replay of the prefix");
    }

    // A repeat binds the graft again, or a deeper checkpoint descending from it (the first
    // request's recovery point sits at the start of its turn, which here is the graft frontier).
    const auto direct_again = engine.generate(engine.prepare(question("direct", asked)), greedy(32));
    expect(direct_again.reused_prompt_tokens >= slots,
           "a repeated direct request did not start from the graft");
    if (phase.exact) {
        expect(direct_again.generated_token_ids == direct.generated_token_ids,
               "repeating a direct request changed its output");
    }

    // Concurrency 2: both lanes bind at once, one of them forking the pinned graft.
    {
        const std::string other = "Name three colours of the sea.";
        auto first  = engine.submit(engine.prepare(question("direct", other)), greedy(24));
        auto second = engine.submit(engine.prepare(question("text", other)), greedy(24));
        compare(first.wait(), second.wait(), "concurrent");
        auto a = engine.submit(engine.prepare(question("direct", asked)), greedy(16));
        auto b = engine.submit(engine.prepare(question("direct", other)), greedy(20));
        const auto ra = a.wait(), rb = b.wait();
        expect(ra.reused_prompt_tokens >= slots && rb.reused_prompt_tokens >= slots &&
                   ra.generated_token_ids.size() == 16 && rb.generated_token_ids.size() == 20,
               "two concurrent direct requests did not both bind the graft");
    }

    // An ungrafted request is unaffected by the installed graft.
    {
        PromptInput plain = question("", asked);
        const auto result = engine.generate(engine.prepare(std::move(plain)), greedy(8));
        expect(result.generated_token_ids.size() == 8 && result.reused_prompt_tokens < slots,
               "an ungrafted request did not run normally");
    }

    if (phase.pressure) {
        // Distinct long prompts fill and then overflow the Main KV pool and the state slots, so
        // reclaim must release earlier history. The pinned graft must survive all of it.
        std::vector<std::vector<TokenId>> fillers;
        for (int index = 0; index < 6; ++index) {
            std::string body = "Filler document " + std::to_string(index) + ":";
            for (int word = 0; word < 450; ++word) {
                body += " w" + std::to_string((word * 7919 + index * 104729) % 100003);
            }
            fillers.push_back(engine.tokenize_text(body));
        }
        for (const auto& filler : fillers) {
            const auto result = engine.generate(engine.prepare_tokens(filler), greedy(4));
            expect(result.generated_token_ids.size() == 4, "a filler request failed");
        }
        const auto revisited = engine.generate(engine.prepare_tokens(fillers.front()), greedy(4));
        std::cout << "  first filler after pressure reused " << revisited.reused_prompt_tokens << "/"
                  << fillers.front().size() << '\n';
        expect(revisited.reused_prompt_tokens == 0,
               "cache pressure did not evict unrelated history (the test proves nothing)");
        const auto after = engine.generate(engine.prepare(question("direct", asked)), greedy(32));
        compare(after, text, "after pressure");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const auto dir = std::filesystem::temp_directory_path() / "ninfer_graft_real_test";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);

        std::optional<std::map<std::string, TensorBytes>> reference;
        std::optional<std::vector<TokenId>> reference_tokens;
        if (const char* path = std::getenv("NINFER_TEST_GRAFT_REFERENCE"); path && *path) {
            reference = read_container(path);
            require(reference->contains("replay_ids"), "the reference graft carries no replay ids");
            const auto& ids = reference->at("replay_ids").bytes;
            reference_tokens.emplace(ids.size() / 8);
            for (std::size_t i = 0; i < reference_tokens->size(); ++i) {
                std::int64_t id;
                std::memcpy(&id, ids.data() + i * 8, 8);
                (*reference_tokens)[i] = static_cast<TokenId>(id);
            }
        }

        Captured captured = capture(artifact, reference_tokens);
        if (reference) { compare_reference(captured, *reference); }
        quantized_append_agreement(artifact, captured, KvCacheStorage::RotatedLloyd4KeyInt4Value);
        const auto slots = static_cast<std::uint32_t>(captured.tokens.size());

        const auto& text = captured.text;
        auto replayed    = captured.tensors;
        TensorBytes ids{"I64", {slots}, std::vector<std::uint8_t>(slots * 8)};
        for (std::uint32_t i = 0; i < slots; ++i) {
            const std::int64_t id = captured.tokens[i];
            std::memcpy(ids.bytes.data() + i * 8, &id, 8);
        }
        replayed.emplace("replay_ids", std::move(ids));
        const auto text_graft =
            write_container(dir, "replayed", replayed, container_meta(text, slots, "prefill_kv"));
        const auto direct_graft = write_container(dir, "direct", captured.tensors,
                                                  container_meta(text, slots, "direct_kv"));
        (void)argc;
        (void)argv;
        int failures = 0;
        const std::vector<Phase> phases{
            {.name = "no draft, BF16 KV, cache pressure", .pressure = true},
            // The capture is BF16-storage state; prefill in rk4v4 storage reads its own quantized
            // history from the second attention layer on, so the two prefixes are not the same
            // computation and only binding and completion are required.
            {.name    = "no draft, rk4v4 KV",
             .kv      = KvCacheStorage::RotatedLloyd4KeyInt4Value,
             .exact   = false},
            {.name = "MTP", .backend = SpeculativeBackend::Mtp, .exact = false},
            {.name = "DFlash2", .backend = SpeculativeBackend::DFlash2, .exact = false},
        };
        for (const auto& phase : phases) {
            failures += run_phase(artifact, phase, slots, text_graft, direct_graft);
        }
        std::filesystem::remove_all(dir);
        if (failures != 0) {
            std::cerr << failures << " graft checks failed\n";
            return 1;
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
