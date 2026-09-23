#include "models/qwen3_5/frontend/graft.h"

#include "models/qwen3_5/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace ninfer::models::qwen3_5 {
namespace {

using Json = nlohmann::json;

struct TensorEntry {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

class GraftError {
public:
    explicit GraftError(const GraftSource& source) : prefix_("graft '" + source.name + "': ") {}

    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument(prefix_ + message);
    }

    void require(bool condition, const std::string& message) const {
        if (!condition) { fail(message); }
    }

private:
    std::string prefix_;
};

std::vector<std::uint8_t> read_file(const std::filesystem::path& path, const GraftError& error) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    error.require(in.good(), "cannot open " + path.string());
    const std::streamoff size = in.tellg();
    error.require(size >= 0, "cannot size " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    error.require(in.good() || in.eof(), "cannot read " + path.string());
    return bytes;
}

std::uint64_t element_bytes(const std::string& dtype, const GraftError& error) {
    if (dtype == "BF16" || dtype == "F16") { return 2; }
    if (dtype == "F32" || dtype == "I32") { return 4; }
    if (dtype == "I64" || dtype == "F64") { return 8; }
    error.fail("unsupported tensor dtype " + dtype);
}

std::uint64_t json_u64(const Json& meta, const char* key, const GraftError& error) {
    error.require(meta.contains(key) && meta.at(key).is_number_unsigned(),
                  std::string("metadata field '") + key + "' must be a non-negative integer");
    return meta.at(key).get<std::uint64_t>();
}

std::string json_string(const Json& meta, const char* key, const GraftError& error) {
    error.require(meta.contains(key) && meta.at(key).is_string(),
                  std::string("metadata field '") + key + "' must be a string");
    return meta.at(key).get<std::string>();
}

std::string shape_text(std::span<const std::uint64_t> shape) {
    std::string text = "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) { text += ", "; }
        text += std::to_string(shape[index]);
    }
    return text + "]";
}

// Safetensors framing: u64 little-endian header length, a JSON header mapping each tensor name to
// {dtype, shape, data_offsets} relative to the payload, then the payload.
std::map<std::string, TensorEntry> parse_safetensors(std::span<const std::uint8_t> file,
                                                     std::span<const std::uint8_t>& payload,
                                                     const GraftError& error) {
    error.require(file.size() >= 8, "container is shorter than a safetensors header");
    std::uint64_t header_bytes = 0;
    for (int byte = 7; byte >= 0; --byte) { header_bytes = (header_bytes << 8U) | file[byte]; }
    error.require(header_bytes <= file.size() - 8, "safetensors header runs past the file");
    const std::string_view header_text(reinterpret_cast<const char*>(file.data() + 8),
                                       static_cast<std::size_t>(header_bytes));
    Json header;
    try {
        header = Json::parse(header_text);
    } catch (const Json::exception& parse_error) {
        error.fail(std::string("safetensors header is not JSON: ") + parse_error.what());
    }
    error.require(header.is_object(), "safetensors header is not an object");
    payload = file.subspan(static_cast<std::size_t>(8 + header_bytes));

    std::map<std::string, TensorEntry> tensors;
    for (const auto& [name, value] : header.items()) {
        if (name == "__metadata__") { continue; }
        error.require(value.is_object() && value.contains("dtype") && value.contains("shape") &&
                          value.contains("data_offsets"),
                      "tensor '" + name + "' has an incomplete safetensors entry");
        TensorEntry entry;
        entry.dtype                  = value.at("dtype").get<std::string>();
        entry.shape                  = value.at("shape").get<std::vector<std::uint64_t>>();
        const auto offsets           = value.at("data_offsets").get<std::vector<std::uint64_t>>();
        std::uint64_t expected_bytes = element_bytes(entry.dtype, error);
        for (const std::uint64_t dimension : entry.shape) {
            error.require(dimension == 0 ||
                              expected_bytes <= std::numeric_limits<std::uint64_t>::max() / dimension,
                          "tensor '" + name + "' size overflows");
            expected_bytes *= dimension;
        }
        error.require(offsets.size() == 2 && offsets[0] <= offsets[1] &&
                          offsets[1] <= payload.size() && offsets[1] - offsets[0] == expected_bytes,
                      "tensor '" + name + "' data_offsets do not match its dtype and shape");
        entry.begin = offsets[0];
        entry.end   = offsets[1];
        tensors.emplace(name, std::move(entry));
    }
    return tensors;
}

void expect_tensor(const std::map<std::string, TensorEntry>& tensors, const std::string& name,
                   const std::string& dtype, const std::vector<std::uint64_t>& shape,
                   const GraftError& error) {
    const auto found = tensors.find(name);
    error.require(found != tensors.end(), "container has no '" + name + "' tensor");
    error.require(found->second.dtype == dtype,
                  "tensor '" + name + "' is " + found->second.dtype + ", expected " + dtype);
    error.require(found->second.shape == shape, "tensor '" + name + "' has shape " +
                                                    shape_text(found->second.shape) +
                                                    ", the model needs " + shape_text(shape));
}

} // namespace

PromptGraft load_prompt_graft(const GraftSource& source, const TextConfig& text) {
    const GraftError error(source);
    error.require(!source.name.empty(), "graft name is empty");

    std::filesystem::path sidecar_path = source.path;
    sidecar_path.replace_extension(".json");
    const std::vector<std::uint8_t> sidecar_bytes = read_file(sidecar_path, error);
    Json meta;
    try {
        meta = Json::parse(sidecar_bytes.begin(), sidecar_bytes.end());
    } catch (const Json::exception& parse_error) {
        error.fail("sidecar " + sidecar_path.string() + " is not JSON: " + parse_error.what());
    }
    error.require(meta.is_object(), "sidecar is not a JSON object");

    error.require(json_u64(meta, "format_version", error) == 1,
                  "only phantom-kv format_version 1 is supported");
    const std::string kind = json_string(meta, "kind", error);
    // Replaying ids is the whole mechanism here. direct_kv and softprompt_kv grafts carry no token
    // ids and would need their tensors written into the cache directly.
    error.require(kind == "prefill_kv",
                  "kind '" + kind + "' has no replayable token ids; only prefill_kv is supported");
    error.require(json_string(meta, "replay", error) == "ids",
                  "prefill_kv graft does not carry replay ids");

    // The hybrid layout must be this model's, layer for layer: the graft's attention and linear
    // layers are the ones it was computed through.
    error.require(meta.contains("layer_types") && meta.at("layer_types").is_array(),
                  "metadata field 'layer_types' must be an array");
    const Json& layer_types = meta.at("layer_types");
    error.require(layer_types.size() == text.layer_types.size(),
                  "graft has " + std::to_string(layer_types.size()) + " layers, the model has " +
                      std::to_string(text.layer_types.size()));
    for (std::size_t layer = 0; layer < layer_types.size(); ++layer) {
        const std::string type = layer_types[layer].is_string() ? layer_types[layer].get<std::string>()
                                                               : std::string();
        const MixerKind expected = text.layer_types[layer];
        const bool matches =
            (type == "full_attention" && expected == MixerKind::FullAttention) ||
            (type == "linear_attention" && expected == MixerKind::LinearAttention);
        error.require(matches, "layer " + std::to_string(layer) + " is '" + type +
                                   "' in the graft but not in the model");
    }
    error.require(json_u64(meta, "n_layers", error) == text.layer_types.size() &&
                      json_u64(meta, "n_attn_layers", error) == text.full_attention_layers,
                  "n_layers/n_attn_layers disagree with layer_types");
    error.require(text.attention.has_value() && text.gdn.has_value(),
                  "the model is not a hybrid attention/Gated DeltaNet model");
    const AttentionConfig& attention = *text.attention;
    const GdnConfig& gdn             = *text.gdn;

    const std::uint64_t slots = json_u64(meta, "n_slots", error);
    error.require(slots > 0, "graft has no slots");
    error.require(json_u64(meta, "n_kv_heads", error) == attention.num_key_value_heads &&
                      json_u64(meta, "head_dim", error) == attention.head_dim,
                  "attention geometry differs from the model");
    error.require(json_u64(meta, "conv_dim", error) == gdn.conv_channels() &&
                      json_u64(meta, "conv_k", error) == gdn.linear_conv_kernel_dim,
                  "Gated DeltaNet convolution geometry differs from the model");
    error.require(meta.contains("rec_shape") && meta.at("rec_shape").is_array() &&
                      meta.at("rec_shape").get<std::vector<std::uint64_t>>() ==
                          std::vector<std::uint64_t>{gdn.linear_num_value_heads,
                                                     gdn.linear_key_head_dim,
                                                     gdn.linear_value_head_dim},
                  "Gated DeltaNet recurrent-state geometry differs from the model");
    const std::string expected_sha256 = json_string(meta, "sha256", error);

    const std::vector<std::uint8_t> file = read_file(source.path, error);
    std::span<const std::uint8_t> payload;
    const std::map<std::string, TensorEntry> tensors = parse_safetensors(file, payload, error);

    const std::string actual_sha256 = frontend::sha256_hex(frontend::sha256(payload));
    error.require(actual_sha256 == expected_sha256,
                  "tensor payload sha256 " + actual_sha256 + " does not match the sidecar's " +
                      expected_sha256);

    const std::uint64_t attention_layers = text.full_attention_layers;
    const std::uint64_t linear_layers    = text.linear_attention_layers;
    const std::vector<std::uint64_t> kv_shape{attention_layers, slots,
                                              attention.num_key_value_heads, attention.head_dim};
    expect_tensor(tensors, "k", "BF16", kv_shape, error);
    expect_tensor(tensors, "v", "BF16", kv_shape, error);
    expect_tensor(tensors, "conv", "BF16",
                  {linear_layers, gdn.conv_channels(), gdn.linear_conv_kernel_dim}, error);
    expect_tensor(tensors, "rec", "F32",
                  {linear_layers, gdn.linear_num_value_heads, gdn.linear_key_head_dim,
                   gdn.linear_value_head_dim},
                  error);
    expect_tensor(tensors, "replay_ids", "I64", {slots}, error);
    error.require(tensors.size() == 5, "container holds tensors beyond k, v, conv, rec, replay_ids");

    const TensorEntry& ids = tensors.at("replay_ids");
    PromptGraft graft;
    graft.name           = source.name;
    graft.payload_sha256 = actual_sha256;
    graft.tokens.reserve(static_cast<std::size_t>(slots));
    for (std::uint64_t index = 0; index < slots; ++index) {
        std::int64_t token = 0;
        std::memcpy(&token, payload.data() + ids.begin + index * sizeof(token), sizeof(token));
        error.require(token >= 0 && static_cast<std::uint64_t>(token) < text.vocab_size,
                      "replay id " + std::to_string(token) + " at slot " + std::to_string(index) +
                          " is outside the model vocabulary");
        graft.tokens.push_back(static_cast<TokenId>(token));
    }
    return graft;
}

std::vector<PromptGraft> load_prompt_grafts(const std::vector<GraftSource>& sources,
                                            const TextConfig& text) {
    std::vector<PromptGraft> grafts;
    grafts.reserve(sources.size());
    std::unordered_set<std::string> names;
    for (const GraftSource& source : sources) {
        if (!names.insert(source.name).second) {
            throw std::invalid_argument("graft '" + source.name + "' is configured twice");
        }
        grafts.push_back(load_prompt_graft(source, text));
    }
    return grafts;
}

} // namespace ninfer::models::qwen3_5
