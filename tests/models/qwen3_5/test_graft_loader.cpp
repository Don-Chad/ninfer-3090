// Prompt-graft container validation against a small hybrid model geometry. The container is the
// phantom-kv format_version 1 safetensors file plus its JSON sidecar, written here byte for byte.
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/graft.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

namespace q = ninfer::models::qwen3_5;
using Json  = nlohmann::json;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Three Gated DeltaNet layers then one attention layer, the Qwen3.5 hybrid period.
q::TextConfig model() {
    q::TextConfig text;
    text.vocab_size        = 1000;
    text.num_hidden_layers = 4;
    text.layer_types       = {q::MixerKind::LinearAttention, q::MixerKind::LinearAttention,
                              q::MixerKind::LinearAttention, q::MixerKind::FullAttention};
    text.full_attention_layers   = 1;
    text.linear_attention_layers = 3;
    text.attention = q::AttentionConfig{.num_attention_heads = 4, .num_key_value_heads = 2,
                                        .head_dim = 8};
    text.gdn = q::GdnConfig{.linear_num_key_heads   = 2,
                            .linear_key_head_dim    = 4,
                            .linear_num_value_heads = 4,
                            .linear_value_head_dim  = 4,
                            .linear_conv_kernel_dim = 4};
    return text;
}

struct Tensor {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> bytes;
};

std::uint64_t element_count(const std::vector<std::uint64_t>& shape) {
    std::uint64_t count = 1;
    for (const std::uint64_t dimension : shape) { count *= dimension; }
    return count;
}

Tensor filled(std::string dtype, std::vector<std::uint64_t> shape, std::uint64_t element_bytes) {
    Tensor tensor{std::move(dtype), std::move(shape), {}};
    tensor.bytes.resize(element_count(tensor.shape) * element_bytes);
    for (std::size_t index = 0; index < tensor.bytes.size(); ++index) {
        tensor.bytes[index] = static_cast<std::uint8_t>(index * 31U + 7U);
    }
    return tensor;
}

Tensor replay_ids(const std::vector<std::int64_t>& ids) {
    Tensor tensor{"I64", {ids.size()}, std::vector<std::uint8_t>(ids.size() * sizeof(std::int64_t))};
    std::memcpy(tensor.bytes.data(), ids.data(), tensor.bytes.size());
    return tensor;
}

// A valid graft for model(): 3 slots, the tensor set phantom-kv writes for a hybrid prefill_kv graft.
struct Container {
    std::map<std::string, Tensor> tensors;
    Json meta;
};

Container valid_container() {
    Container c;
    c.tensors.emplace("k", filled("BF16", {1, 3, 2, 8}, 2));
    c.tensors.emplace("v", filled("BF16", {1, 3, 2, 8}, 2));
    c.tensors.emplace("conv", filled("BF16", {3, 32, 4}, 2));
    c.tensors.emplace("rec", filled("F32", {3, 4, 4, 4}, 4));
    c.tensors.emplace("replay_ids", replay_ids({11, 22, 33}));
    c.meta = {
        {"format_version", 1},
        {"kind", "prefill_kv"},
        {"model_id", "tiny/hybrid"},
        {"layer_types", {"linear_attention", "linear_attention", "linear_attention", "full_attention"}},
        {"n_layers", 4},
        {"n_attn_layers", 1},
        {"n_slots", 3},
        {"n_kv_heads", 2},
        {"head_dim", 8},
        {"conv_dim", 32},
        {"conv_k", 4},
        {"rec_shape", {4, 4, 4}},
        {"replay", "ids"},
        {"quant", "none"},
    };
    return c;
}

// Writes <dir>/<stem>.bin and .json; the sidecar sha256 covers the payload unless overridden.
std::filesystem::path write(const std::filesystem::path& dir, const std::string& stem,
                            Container container,
                            const std::function<void(std::vector<std::uint8_t>&)>& corrupt = {}) {
    Json header = Json::object();
    std::vector<std::uint8_t> payload;
    for (const auto& [name, tensor] : container.tensors) {
        const std::uint64_t begin = payload.size();
        payload.insert(payload.end(), tensor.bytes.begin(), tensor.bytes.end());
        header[name] = {{"dtype", tensor.dtype},
                        {"shape", tensor.shape},
                        {"data_offsets", {begin, payload.size()}}};
    }
    if (!container.meta.contains("sha256")) {
        container.meta["sha256"] = q::frontend::sha256_hex(q::frontend::sha256(payload));
    }
    if (corrupt) { corrupt(payload); }

    const std::string header_text = header.dump();
    std::vector<std::uint8_t> file(8);
    const std::uint64_t header_bytes = header_text.size();
    for (int byte = 0; byte < 8; ++byte) {
        file[byte] = static_cast<std::uint8_t>(header_bytes >> (8U * byte));
    }
    file.insert(file.end(), header_text.begin(), header_text.end());
    file.insert(file.end(), payload.begin(), payload.end());

    const std::filesystem::path bin = dir / (stem + ".bin");
    std::ofstream(bin, std::ios::binary).write(reinterpret_cast<const char*>(file.data()),
                                               static_cast<std::streamsize>(file.size()));
    std::ofstream(dir / (stem + ".json")) << container.meta.dump(2);
    return bin;
}

// Loads and returns the error text, or "" when the graft was accepted.
std::string load_error(const std::filesystem::path& path) {
    try {
        (void)q::load_prompt_graft(ninfer::GraftSource{.name = "g", .path = path}, model());
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

int expect_rejected(const std::filesystem::path& path, const std::string& fragment,
                    const std::string& what) {
    const std::string error = load_error(path);
    return check(!error.empty() && error.find(fragment) != std::string::npos,
                 what + " (error was: '" + error + "')");
}

} // namespace

int main() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "ninfer_graft_loader_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    int failures = 0;

    {
        const std::filesystem::path path = write(dir, "valid", valid_container());
        const q::PromptGraft graft =
            q::load_prompt_graft(ninfer::GraftSource{.name = "product", .path = path}, model());
        failures += check(graft.name == "product" &&
                              graft.tokens == std::vector<ninfer::TokenId>{11, 22, 33},
                          "a valid graft did not load its replay ids in order");
        failures += check(graft.payload_sha256.size() == 64, "payload digest was not recorded");
    }

    failures += expect_rejected(
        write(dir, "tampered", valid_container(), [](auto& payload) { payload[5] ^= 0x40U; }),
        "sha256", "a payload that no longer matches the sidecar digest was accepted");

    {
        Container c                = valid_container();
        c.meta["layer_types"][0] = "full_attention";
        failures += expect_rejected(write(dir, "layers", c), "layer 0",
                                    "a graft with another layer layout was accepted");
    }
    {
        Container c = valid_container();
        c.tensors.insert_or_assign("conv", filled("BF16", {3, 32, 3}, 2));
        failures += expect_rejected(write(dir, "conv", c), "conv",
                                    "a conv state of the wrong kernel width was accepted");
    }
    {
        Container c = valid_container();
        c.tensors.insert_or_assign("rec", filled("BF16", {3, 4, 4, 4}, 2));
        failures += expect_rejected(write(dir, "rec_dtype", c), "rec",
                                    "a recurrent state stored below FP32 was accepted");
    }
    {
        Container c         = valid_container();
        c.meta["n_kv_heads"] = 4;
        failures += expect_rejected(write(dir, "heads", c), "attention geometry",
                                    "a graft with another KV head count was accepted");
    }
    {
        Container c   = valid_container();
        c.meta["kind"] = "direct_kv";
        failures += expect_rejected(write(dir, "direct", c), "direct_kv",
                                    "a direct_kv graft (no replay ids) was accepted");
    }
    {
        Container c = valid_container();
        c.tensors.insert_or_assign("replay_ids", replay_ids({11, 1000, 33}));
        failures += expect_rejected(write(dir, "vocab", c), "vocabulary",
                                    "a replay id outside the vocabulary was accepted");
    }
    {
        Container c                  = valid_container();
        c.meta["format_version"] = 0;
        failures += expect_rejected(write(dir, "v0", c), "format_version",
                                    "a format_version 0 (all-attention) graft was accepted");
    }
    {
        const std::filesystem::path orphan = write(dir, "orphan", valid_container());
        std::filesystem::remove(dir / "orphan.json");
        failures += expect_rejected(orphan, "cannot open", "a graft without its sidecar was accepted");
    }
    {
        const std::filesystem::path path = write(dir, "dup", valid_container());
        bool rejected                    = false;
        try {
            (void)q::load_prompt_grafts({{.name = "a", .path = path}, {.name = "a", .path = path}},
                                        model());
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "two grafts with the same name were accepted");
    }

    std::filesystem::remove_all(dir);
    if (failures != 0) {
        std::cerr << failures << " graft loader checks failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
