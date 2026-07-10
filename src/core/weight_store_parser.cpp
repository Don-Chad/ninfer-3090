#include "qus/core/weight_store_parser.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace qus {
namespace {

constexpr std::array<std::byte, 16> kMagic = {
    std::byte{0x51}, std::byte{0x35}, std::byte{0x30}, std::byte{0x39},
    std::byte{0x30}, std::byte{0x4D}, std::byte{0x49}, std::byte{0x58},
    std::byte{0x45}, std::byte{0x44}, std::byte{0x56}, std::byte{0x34},
    std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
};

constexpr std::uint32_t kVersion                = 4;
constexpr std::uint32_t kFormatMinor            = 1;
constexpr std::uint32_t kEndianTag              = 0x01020304U;
constexpr std::uint32_t kHeaderSize             = 4096;
constexpr std::uint32_t kModuleRecordSize       = 64;
constexpr std::uint32_t kTensorEntrySize        = 128;
constexpr std::uint32_t kSegmentRecordSize      = 32;
constexpr std::uint32_t kFusionGroupRecordSize  = 64;
constexpr std::uint32_t kTokenizerRecordSize    = 64;
constexpr std::uint32_t kTokenizerRecordCount   = 3;
constexpr std::uint32_t kCanonicalDraftRows     = 131072;
constexpr std::uint32_t kCanonicalDraftHidden   = 5120;
constexpr std::uint32_t kCanonicalTokenizerIds  = 248077;
constexpr std::uint32_t kMaxCatalogNameBytes    = 4096;
constexpr std::uint64_t kPayloadAlign           = 256;
constexpr std::uint64_t kRegionAlign            = 4096;
constexpr std::uint64_t kTokenizerAlign         = 64;
constexpr std::uint32_t kLmHeadDraftPresentFlag = 1U << 4;

constexpr std::array<std::array<std::uint8_t, 32>, 3> kCanonicalTokenizerSha256 = {{
    {0x87, 0xa7, 0x83, 0x0d, 0x63, 0xfc, 0xf4, 0x3b, 0xf2, 0x41, 0xc3,
     0xc5, 0x24, 0x2e, 0x96, 0xe6, 0x2d, 0xd3, 0xfd, 0xc2, 0x92, 0x24,
     0xca, 0x26, 0xfe, 0xd8, 0xea, 0x33, 0x3d, 0xb7, 0x2d, 0xe4},
    {0x3b, 0xd6, 0x40, 0xba, 0x6d, 0x8d, 0xa8, 0xf5, 0x84, 0x4f, 0x35,
     0x48, 0xb7, 0xe2, 0x18, 0x4f, 0xc6, 0x64, 0xdd, 0x67, 0x0e, 0x14,
     0x47, 0xe4, 0x25, 0xa4, 0x8d, 0xba, 0xaa, 0xd1, 0xef, 0x43},
    {0xe7, 0x0c, 0x13, 0x6c, 0x1b, 0x78, 0xdd, 0xc1, 0xfb, 0x09, 0x05,
     0xba, 0xc8, 0xe7, 0x33, 0xa4, 0xdc, 0x44, 0x8d, 0x4f, 0x85, 0x2a,
     0x5d, 0xd7, 0x51, 0x43, 0xff, 0xfc, 0x70, 0xbe, 0x55, 0x0e},
}};

struct Range {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

[[noreturn]] void parse_error(const char* message) { throw std::runtime_error(message); }

void require(bool ok, const char* message) {
    if (!ok) { parse_error(message); }
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        parse_error("q5090 integer addition overflow");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        parse_error("q5090 integer multiplication overflow");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t align) {
    if (align == 0) { parse_error("q5090 invalid alignment"); }
    const std::uint64_t add = align - 1;
    return (checked_add(value, add) / align) * align;
}

void require_range(std::span<const std::byte> file, std::uint64_t offset, std::uint64_t size,
                   const char* label) {
    const std::uint64_t end = checked_add(offset, size);
    if (offset > file.size() || end > file.size()) {
        throw std::runtime_error(std::string("q5090 range outside file: ") + label);
    }
}

void require_logical_range(std::uint64_t file_size, std::uint64_t offset, std::uint64_t size,
                           const char* label) {
    const std::uint64_t end = checked_add(offset, size);
    if (offset > file_size || end > file_size) {
        throw std::runtime_error(std::string("q5090 range outside file: ") + label);
    }
}

std::uint16_t read_u16(std::span<const std::byte> file, std::uint64_t offset) {
    require_range(file, offset, 2, "u16");
    const auto* p = file.data() + offset;
    return static_cast<std::uint16_t>(std::to_integer<unsigned char>(p[0])) |
           (static_cast<std::uint16_t>(std::to_integer<unsigned char>(p[1])) << 8);
}

std::uint32_t read_u32(std::span<const std::byte> file, std::uint64_t offset) {
    require_range(file, offset, 4, "u32");
    const auto* p = file.data() + offset;
    return static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[0])) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[1])) << 8) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[2])) << 16) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(p[3])) << 24);
}

std::uint64_t read_u64(std::span<const std::byte> file, std::uint64_t offset) {
    require_range(file, offset, 8, "u64");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(
                     std::to_integer<unsigned char>(file[static_cast<std::size_t>(offset) + i]))
                 << (8 * i);
    }
    return value;
}

bool all_zero(std::span<const std::byte> bytes) {
    return std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; });
}

void require_zero_range(std::span<const std::byte> file, std::uint64_t offset, std::uint64_t size,
                        const char* label) {
    require_range(file, offset, size, label);
    require(
        all_zero(file.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(size))),
        label);
}

template <typename T>
void validate_expected(std::uint32_t actual, const std::optional<T>& expected, const char* name) {
    if (expected.has_value() && actual != static_cast<std::uint32_t>(*expected)) {
        throw std::runtime_error(std::string("q5090 model metadata mismatch: ") + name);
    }
}

QType qtype_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0:
        return QType::Q4G64_F16S;
    case 1:
        return QType::Q5G64_F16S;
    case 2:
        return QType::Q6G64_F16S;
    case 3:
        return QType::W8G128_F16S;
    case 4:
        return QType::BF16_CTRL;
    case 5:
        return QType::FP32_CTRL;
    case 6:
        return QType::W8G32_F16S;
    case 7:
        return QType::I32_CTRL;
    default:
        parse_error("q5090 invalid qtype tag");
    }
}

QuantLayout layout_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0:
        return QuantLayout::RowSplit;
    case 1:
        return QuantLayout::Contiguous;
    default:
        parse_error("q5090 invalid layout tag");
    }
}

ModuleKind module_from_tag(std::uint32_t tag) {
    switch (tag) {
    case 0:
        return ModuleKind::TextCore;
    case 1:
        return ModuleKind::MtpDraft;
    case 2:
        return ModuleKind::VisionEncoder;
    case 3:
        return ModuleKind::LmHeadDraft;
    default:
        parse_error("q5090 invalid module tag");
    }
}

ScaleDType scale_from_tag(std::uint16_t tag) {
    switch (tag) {
    case 0:
        return ScaleDType::None;
    case 1:
        return ScaleDType::FP16;
    default:
        parse_error("q5090 invalid scale dtype tag");
    }
}

bool valid_source_kind(std::uint32_t kind) {
    switch (kind) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 30:
    case 31:
    case 32:
    case 33:
    case 34:
    case 35:
    case 36:
    case 40:
    case 41:
    case 42:
    case 50:
    case 51:
    case 52:
    case 53:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 67:
    case 68:
    case 69:
    case 70:
    case 71:
    case 72:
    case 73:
    case 74:
    case 75:
    case 76:
    case 77:
    case 78:
    case 79:
    case 80:
        return true;
    default:
        return false;
    }
}

bool valid_source_kind(ModuleKind module, std::uint32_t kind) {
    if (kind == static_cast<std::uint32_t>(SourceKind::Other)) { return true; }
    switch (module) {
    case ModuleKind::TextCore:
        return (kind >= 1 && kind <= 5) || (kind >= 10 && kind <= 20) ||
               (kind >= 30 && kind <= 36) || (kind >= 40 && kind <= 42);
    case ModuleKind::MtpDraft:
        return (kind >= 50 && kind <= 53) || kind == 4 || kind == 5 || (kind >= 30 && kind <= 36) ||
               (kind >= 40 && kind <= 42);
    case ModuleKind::VisionEncoder:
        return kind >= 60 && kind <= 80;
    case ModuleKind::LmHeadDraft:
        return kind == static_cast<std::uint32_t>(SourceKind::LmHeadDraft) ||
               kind == static_cast<std::uint32_t>(SourceKind::LmHeadDraftIdmap);
    }
    return false;
}

bool valid_fusion_group_id(std::uint32_t group_id) { return group_id >= 1 && group_id <= 3; }

bool is_quant_qtype(QType qtype) {
    return qtype == QType::Q4G64_F16S || qtype == QType::Q5G64_F16S || qtype == QType::Q6G64_F16S ||
           qtype == QType::W8G128_F16S || qtype == QType::W8G32_F16S;
}

std::uint32_t quant_group_size(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
        return 64;
    case QType::W8G128_F16S:
        return 128;
    case QType::W8G32_F16S:
        return 32;
    default:
        parse_error("q5090 qtype has no quant group size");
    }
}

std::uint64_t nibble_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::Q5G64_F16S:
    case QType::Q6G64_F16S:
        return 32;
    case QType::W8G128_F16S:
        return 128;
    case QType::W8G32_F16S:
        return 32;
    default:
        parse_error("q5090 qtype has no nibble plane bytes per group");
    }
}

std::uint64_t high_bytes_per_group(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
    case QType::W8G128_F16S:
    case QType::W8G32_F16S:
        return 0;
    case QType::Q5G64_F16S:
        return 8;
    case QType::Q6G64_F16S:
        return 16;
    default:
        parse_error("q5090 qtype has no high plane bytes per group");
    }
}

std::uint64_t element_size(QType qtype) {
    switch (qtype) {
    case QType::BF16_CTRL:
        return 2;
    case QType::FP32_CTRL:
        return 4;
    case QType::I32_CTRL:
        return 4;
    default:
        parse_error("q5090 qtype has no contiguous element size");
    }
}

std::uint64_t numel(const std::array<std::uint32_t, 4>& shape, std::uint32_t ndim) {
    std::uint64_t result = 1;
    for (std::uint32_t i = 0; i < ndim; ++i) { result = checked_mul(result, shape[i]); }
    return result;
}

void validate_shapes(const ParsedQ5090Tensor& tensor) {
    require(tensor.ndim >= 1 && tensor.ndim <= 4, "q5090 invalid tensor rank");
    for (std::uint32_t i = 0; i < 4; ++i) {
        if (i < tensor.ndim) {
            require(tensor.shape[i] != 0, "q5090 zero tensor shape dim");
            require(tensor.padded_shape[i] != 0, "q5090 zero padded tensor shape dim");
        } else {
            require(tensor.shape[i] == 1, "q5090 inactive shape dim must be one");
            require(tensor.padded_shape[i] == 1, "q5090 inactive padded dim must be one");
        }
    }
}

std::uint64_t expected_payload_bytes(const ParsedQ5090Tensor& tensor) {
    validate_shapes(tensor);
    switch (tensor.layout) {
    case QuantLayout::RowSplit: {
        require(tensor.ndim == 2, "q5090 ROW_SPLIT tensor must be 2D");
        require(is_quant_qtype(tensor.qtype), "q5090 ROW_SPLIT qtype mismatch");
        const std::uint32_t group = quant_group_size(tensor.qtype);
        require(tensor.group_size == group, "q5090 ROW_SPLIT group mismatch");
        require(tensor.scale_dtype == ScaleDType::FP16, "q5090 ROW_SPLIT scale mismatch");
        require(tensor.padded_shape[0] == tensor.shape[0], "q5090 ROW_SPLIT padded N mismatch");
        require(tensor.shape[2] == 1 && tensor.shape[3] == 1,
                "q5090 ROW_SPLIT trailing shape mismatch");
        require(tensor.padded_shape[1] == align_up(tensor.shape[1], 128),
                "q5090 ROW_SPLIT padded K mismatch");
        const std::uint64_t groups = tensor.padded_shape[1] / group;
        const std::uint64_t nibble =
            checked_mul(checked_mul(tensor.shape[0], groups), nibble_bytes_per_group(tensor.qtype));
        const std::uint64_t high =
            checked_mul(checked_mul(tensor.shape[0], groups), high_bytes_per_group(tensor.qtype));
        const std::uint64_t scale     = checked_mul(checked_mul(tensor.shape[0], groups), 2);
        const std::uint64_t high_rel  = align_up(nibble, kPayloadAlign);
        const std::uint64_t scale_rel = checked_add(high_rel, align_up(high, kPayloadAlign));
        const std::uint64_t payload   = checked_add(scale_rel, scale);
        require(tensor.nibble_plane_bytes == nibble, "q5090 ROW_SPLIT nibble plane byte mismatch");
        require(tensor.high_plane_bytes == high, "q5090 ROW_SPLIT high plane byte mismatch");
        require(tensor.scale_plane_bytes == scale, "q5090 ROW_SPLIT scale plane byte mismatch");
        return payload;
    }
    case QuantLayout::Contiguous: {
        require(tensor.qtype == QType::BF16_CTRL || tensor.qtype == QType::FP32_CTRL ||
                    tensor.qtype == QType::I32_CTRL,
                "q5090 CONTIGUOUS qtype mismatch");
        require(tensor.group_size == 0, "q5090 CONTIGUOUS group mismatch");
        require(tensor.scale_dtype == ScaleDType::None, "q5090 CONTIGUOUS scale mismatch");
        require(tensor.padded_shape == tensor.shape, "q5090 CONTIGUOUS padded shape mismatch");
        const std::uint64_t raw =
            checked_mul(numel(tensor.shape, tensor.ndim), element_size(tensor.qtype));
        require(tensor.nibble_plane_bytes == raw, "q5090 CONTIGUOUS payload byte mismatch");
        require(tensor.high_plane_bytes == 0, "q5090 CONTIGUOUS high byte mismatch");
        require(tensor.scale_plane_bytes == 0, "q5090 CONTIGUOUS scale byte mismatch");
        return raw;
    }
    }
    parse_error("q5090 invalid layout");
}

ParsedQ5090Header parse_header(std::span<const std::byte> file, std::uint64_t file_size,
                               const Q5090Expectations& expected) {
    require(file.size() >= kHeaderSize, "q5090 file too small for header");
    require(std::equal(kMagic.begin(), kMagic.end(), file.begin()), "q5090 bad magic");

    require(read_u32(file, 16) == kVersion, "q5090 bad version");
    require(read_u32(file, 20) == kEndianTag, "q5090 endian mismatch");
    require(read_u32(file, 24) == kHeaderSize, "q5090 bad header size");

    ParsedQ5090Header h;
    h.tensor_count            = read_u32(file, 28);
    h.module_count            = read_u32(file, 32);
    h.layer_count             = read_u32(file, 36);
    h.flags                   = read_u32(file, 40);
    h.segment_count           = read_u32(file, 44);
    h.module_index_offset     = read_u64(file, 48);
    h.module_index_bytes      = read_u64(file, 56);
    h.tensor_index_offset     = read_u64(file, 64);
    h.tensor_index_bytes      = read_u64(file, 72);
    h.string_table_offset     = read_u64(file, 80);
    h.string_table_bytes      = read_u64(file, 88);
    h.payload_offset          = read_u64(file, 96);
    h.payload_bytes           = read_u64(file, 104);
    h.hidden_size             = read_u32(file, 112);
    h.intermediate_size       = read_u32(file, 116);
    h.vocab_size              = read_u32(file, 120);
    h.num_attention_heads     = read_u32(file, 124);
    h.num_key_value_heads     = read_u32(file, 128);
    h.head_dim                = read_u32(file, 132);
    h.gdn_key_heads           = read_u32(file, 136);
    h.gdn_value_heads         = read_u32(file, 140);
    h.gdn_key_head_dim        = read_u32(file, 144);
    h.gdn_value_head_dim      = read_u32(file, 148);
    h.gdn_conv_width          = read_u32(file, 152);
    h.full_attention_interval = read_u32(file, 156);
    h.max_position_embeddings = read_u32(file, 160);
    h.fusion_group_count      = read_u32(file, 164);
    for (std::size_t i = 0; i < h.sha256_safetensors_index.size(); ++i) {
        h.sha256_safetensors_index[i] = std::to_integer<std::uint8_t>(file[168 + i]);
    }
    h.segment_index_offset      = read_u64(file, 200);
    h.segment_index_bytes       = read_u64(file, 208);
    h.fusion_group_index_offset = read_u64(file, 216);
    h.fusion_group_index_bytes  = read_u64(file, 224);
    h.format_minor              = read_u32(file, 232);
    h.tokenizer_record_count    = read_u32(file, 236);
    h.tokenizer_record_size     = read_u32(file, 240);
    h.tokenizer_flags           = read_u32(file, 244);
    h.tokenizer_index_offset    = read_u64(file, 248);
    h.tokenizer_index_bytes     = read_u64(file, 256);
    h.tokenizer_data_offset     = read_u64(file, 264);
    h.tokenizer_data_bytes      = read_u64(file, 272);

    require(h.module_count >= 1 && h.module_count <= 4, "q5090 invalid module count");
    require(h.tensor_count > 0 && h.tensor_count <= 2048, "q5090 invalid tensor count");
    require(h.segment_count > 0 && h.segment_count <= 4096, "q5090 invalid segment count");
    require(h.fusion_group_count <= 512, "q5090 invalid fusion group count");
    require(h.string_table_bytes <= 64ULL * 1024ULL * 1024ULL, "q5090 string table exceeds cap");
    require(h.layer_count == expected.layer_count.value_or(64), "q5090 layer count mismatch");
    require(h.full_attention_interval > 0, "q5090 full-attention interval must be nonzero");
    if (expected.validate_model_contract) {
        require(h.layer_count == 64 && h.hidden_size == 5120 && h.intermediate_size == 17408 &&
                    h.vocab_size == 248320 && h.num_attention_heads == 24 &&
                    h.num_key_value_heads == 4 && h.head_dim == 256 && h.gdn_key_heads == 16 &&
                    h.gdn_value_heads == 48 && h.gdn_key_head_dim == 128 &&
                    h.gdn_value_head_dim == 128 && h.gdn_conv_width == 4 &&
                    h.full_attention_interval == 4 && h.max_position_embeddings == 262144,
                "q5090 fixed Qwen3.6-27B model-card header mismatch");
    }
    require((h.flags & ~0x1FU) == 0, "q5090 unknown header flags");
    require(h.format_minor == kFormatMinor, "q5090 unsupported format minor");
    require(h.tokenizer_record_count == kTokenizerRecordCount,
            "q5090 tokenizer record count mismatch");
    require(h.tokenizer_record_size == kTokenizerRecordSize,
            "q5090 tokenizer record size mismatch");
    require(h.tokenizer_flags == 0, "q5090 tokenizer flags nonzero");
    require(all_zero(file.subspan(280, kHeaderSize - 280)), "q5090 header padding nonzero");

    require(h.module_index_offset == kHeaderSize, "q5090 bad module index offset");
    require(h.module_index_bytes == checked_mul(h.module_count, kModuleRecordSize),
            "q5090 bad module index bytes");
    require(h.tensor_index_offset == checked_add(h.module_index_offset, h.module_index_bytes),
            "q5090 bad tensor index offset");
    require(h.tensor_index_bytes == checked_mul(h.tensor_count, kTensorEntrySize),
            "q5090 bad tensor index bytes");
    require(h.segment_index_offset == checked_add(h.tensor_index_offset, h.tensor_index_bytes),
            "q5090 bad segment index offset");
    require(h.segment_index_bytes == checked_mul(h.segment_count, kSegmentRecordSize),
            "q5090 bad segment index bytes");
    require(h.fusion_group_index_offset ==
                checked_add(h.segment_index_offset, h.segment_index_bytes),
            "q5090 bad fusion group index offset");
    require(h.fusion_group_index_bytes == checked_mul(h.fusion_group_count, kFusionGroupRecordSize),
            "q5090 bad fusion group index bytes");
    require(h.string_table_offset ==
                checked_add(h.fusion_group_index_offset, h.fusion_group_index_bytes),
            "q5090 bad string table offset");
    const std::uint64_t string_end = checked_add(h.string_table_offset, h.string_table_bytes);
    require(h.tokenizer_index_offset == align_up(string_end, kTokenizerAlign),
            "q5090 bad tokenizer index offset");
    require(h.tokenizer_index_bytes == checked_mul(kTokenizerRecordCount, kTokenizerRecordSize),
            "q5090 bad tokenizer index bytes");
    require(h.tokenizer_data_offset ==
                align_up(checked_add(h.tokenizer_index_offset, h.tokenizer_index_bytes),
                         kTokenizerAlign),
            "q5090 bad tokenizer data offset");
    const std::uint64_t tokenizer_end =
        checked_add(h.tokenizer_data_offset, h.tokenizer_data_bytes);
    require(h.tokenizer_data_bytes <= 321ULL * 1024ULL * 1024ULL,
            "q5090 tokenizer data region exceeds cap");
    require(h.payload_offset <= 512ULL * 1024ULL * 1024ULL,
            "q5090 metadata/tokenizer prefix exceeds cap");
    require(h.payload_offset == align_up(tokenizer_end, kRegionAlign),
            "q5090 bad payload offset after tokenizer");
    require(h.payload_offset % kRegionAlign == 0, "q5090 payload region is not aligned");
    require(checked_add(h.payload_offset, h.payload_bytes) == file_size,
            "q5090 payload bytes do not match file size");

    require_logical_range(file_size, h.module_index_offset, h.module_index_bytes, "module index");
    require_logical_range(file_size, h.tensor_index_offset, h.tensor_index_bytes, "tensor index");
    require_logical_range(file_size, h.segment_index_offset, h.segment_index_bytes,
                          "segment index");
    require_logical_range(file_size, h.fusion_group_index_offset, h.fusion_group_index_bytes,
                          "fusion group index");
    require_logical_range(file_size, h.string_table_offset, h.string_table_bytes, "string table");
    require_logical_range(file_size, h.tokenizer_index_offset, h.tokenizer_index_bytes,
                          "tokenizer index");
    require_logical_range(file_size, h.tokenizer_data_offset, h.tokenizer_data_bytes,
                          "tokenizer data");
    require_logical_range(file_size, h.payload_offset, h.payload_bytes, "payload region");

    validate_expected(h.hidden_size, expected.hidden_size, "hidden_size");
    validate_expected(h.intermediate_size, expected.intermediate_size, "intermediate_size");
    validate_expected(h.vocab_size, expected.vocab_size, "vocab_size");
    validate_expected(h.num_attention_heads, expected.num_attention_heads, "num_attention_heads");
    validate_expected(h.num_key_value_heads, expected.num_key_value_heads, "num_key_value_heads");
    validate_expected(h.head_dim, expected.head_dim, "head_dim");
    validate_expected(h.gdn_key_heads, expected.gdn_key_heads, "gdn_key_heads");
    validate_expected(h.gdn_value_heads, expected.gdn_value_heads, "gdn_value_heads");
    validate_expected(h.gdn_key_head_dim, expected.gdn_key_head_dim, "gdn_key_head_dim");
    validate_expected(h.gdn_value_head_dim, expected.gdn_value_head_dim, "gdn_value_head_dim");
    validate_expected(h.gdn_conv_width, expected.gdn_conv_width, "gdn_conv_width");
    validate_expected(h.full_attention_interval, expected.full_attention_interval,
                      "full_attention_interval");
    validate_expected(h.max_position_embeddings, expected.max_position_embeddings,
                      "max_position_embeddings");
    return h;
}

std::uint32_t crc32(std::span<const std::byte> bytes) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const std::byte raw : bytes) {
        crc ^= std::to_integer<std::uint8_t>(raw);
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0U - (crc & 1U);
            crc                      = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

std::array<std::uint8_t, 32> sha256(std::span<const std::byte> bytes) {
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U,
    };
    std::array<std::uint32_t, 8> state = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    const auto rotate_right = [](std::uint32_t value, unsigned shift) {
        return (value >> shift) | (value << (32U - shift));
    };
    const auto process = [&](std::span<const std::byte, 64> block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] =
                (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(block[4 * i])) << 24U) |
                (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(block[4 * i + 1]))
                 << 16U) |
                (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(block[4 * i + 2]))
                 << 8U) |
                static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(block[4 * i + 3]));
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const std::uint32_t x  = words[i - 15];
            const std::uint32_t y  = words[i - 2];
            const std::uint32_t s0 = rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3U);
            const std::uint32_t s1 = rotate_right(y, 17) ^ rotate_right(y, 19) ^ (y >> 10U);
            words[i]               = words[i - 16] + s0 + words[i - 7] + s1;
        }
        std::uint32_t a = state[0];
        std::uint32_t b = state[1];
        std::uint32_t c = state[2];
        std::uint32_t d = state[3];
        std::uint32_t e = state[4];
        std::uint32_t f = state[5];
        std::uint32_t g = state[6];
        std::uint32_t h = state[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint32_t sum1 =
                rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ (~e & g);
            const std::uint32_t temp1  = h + sum1 + choose + k[i] + words[i];
            const std::uint32_t sum0 =
                rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2    = sum0 + majority;
            h                            = g;
            g                            = f;
            f                            = e;
            e                            = d + temp1;
            d                            = c;
            c                            = b;
            b                            = a;
            a                            = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    };

    const std::size_t full_bytes = bytes.size() - bytes.size() % 64;
    for (std::size_t offset = 0; offset < full_bytes; offset += 64) {
        process(std::span<const std::byte, 64>(bytes.data() + offset, 64));
    }
    std::array<std::byte, 128> tail{};
    const std::size_t remainder = bytes.size() - full_bytes;
    std::copy_n(bytes.data() + full_bytes, remainder, tail.data());
    tail[remainder]               = std::byte{0x80};
    const std::size_t tail_blocks = remainder < 56 ? 1 : 2;
    require(bytes.size() <= std::numeric_limits<std::uint64_t>::max() / 8,
            "q5090 tokenizer SHA-256 input is too large");
    const std::uint64_t bit_length  = static_cast<std::uint64_t>(bytes.size()) * 8U;
    const std::size_t length_offset = tail_blocks * 64 - 8;
    for (std::size_t i = 0; i < 8; ++i) {
        tail[length_offset + i] = static_cast<std::byte>((bit_length >> (56U - 8U * i)) & 0xffU);
    }
    for (std::size_t i = 0; i < tail_blocks; ++i) {
        process(std::span<const std::byte, 64>(tail.data() + 64 * i, 64));
    }

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t i = 0; i < state.size(); ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            digest[4 * i + j] = static_cast<std::uint8_t>(state[i] >> (24U - 8U * j));
        }
    }
    return digest;
}

bool valid_utf8(std::span<const std::byte> bytes) {
    std::size_t i = 0;
    while (i < bytes.size()) {
        const std::uint8_t lead = std::to_integer<std::uint8_t>(bytes[i]);
        if (lead <= 0x7FU) {
            ++i;
            continue;
        }
        std::size_t count   = 0;
        std::uint32_t value = 0;
        if (lead >= 0xC2U && lead <= 0xDFU) {
            count = 2;
            value = lead & 0x1FU;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            count = 3;
            value = lead & 0x0FU;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            count = 4;
            value = lead & 0x07U;
        } else {
            return false;
        }
        if (i + count > bytes.size()) { return false; }
        for (std::size_t j = 1; j < count; ++j) {
            const std::uint8_t next = std::to_integer<std::uint8_t>(bytes[i + j]);
            if ((next & 0xC0U) != 0x80U) { return false; }
            value = (value << 6U) | (next & 0x3FU);
        }
        if ((count == 3 && value < 0x800U) || (count == 4 && value < 0x10000U) ||
            (value >= 0xD800U && value <= 0xDFFFU) || value > 0x10FFFFU) {
            return false;
        }
        i += count;
    }
    return true;
}

Q5090TokenizerKind tokenizer_kind_from_tag(std::uint32_t tag) {
    switch (tag) {
    case 1:
        return Q5090TokenizerKind::TokenizerJson;
    case 2:
        return Q5090TokenizerKind::MergesTxt;
    case 3:
        return Q5090TokenizerKind::GenerationConfig;
    default:
        parse_error("q5090 invalid tokenizer kind");
    }
}

std::uint64_t tokenizer_max_bytes(Q5090TokenizerKind kind) {
    switch (kind) {
    case Q5090TokenizerKind::TokenizerJson:
        return 256ULL * 1024ULL * 1024ULL;
    case Q5090TokenizerKind::MergesTxt:
        return 64ULL * 1024ULL * 1024ULL;
    case Q5090TokenizerKind::GenerationConfig:
        return 1ULL * 1024ULL * 1024ULL;
    }
    parse_error("q5090 invalid tokenizer kind");
}

std::vector<ParsedQ5090TokenizerRecord> parse_tokenizer(std::span<const std::byte> file,
                                                        const ParsedQ5090Header& h,
                                                        Q5090TokenizerBundle& bundle) {
    std::vector<ParsedQ5090TokenizerRecord> records;
    records.reserve(kTokenizerRecordCount);
    std::uint64_t previous_end = h.tokenizer_data_offset;
    for (std::uint32_t i = 0; i < kTokenizerRecordCount; ++i) {
        const std::uint64_t off =
            checked_add(h.tokenizer_index_offset, checked_mul(i, kTokenizerRecordSize));
        ParsedQ5090TokenizerRecord record;
        record.kind        = tokenizer_kind_from_tag(read_u32(file, off));
        record.encoding    = read_u32(file, off + 4);
        record.data_offset = read_u64(file, off + 8);
        record.data_bytes  = read_u64(file, off + 16);
        record.crc32       = read_u32(file, off + 24);
        require(read_u32(file, off + 28) == 0, "q5090 tokenizer reserved field nonzero");
        for (std::size_t j = 0; j < record.sha256.size(); ++j) {
            record.sha256[j] = std::to_integer<std::uint8_t>(file[off + 32 + j]);
        }

        require(static_cast<std::uint32_t>(record.kind) == i + 1,
                "q5090 tokenizer records are not in canonical order");
        require(record.encoding == 0, "q5090 tokenizer encoding is not RAW_UTF8");
        require(record.data_bytes > 0, "q5090 tokenizer asset is empty");
        require(record.data_bytes <= tokenizer_max_bytes(record.kind),
                "q5090 tokenizer asset exceeds size cap");
        const std::uint64_t expected_offset =
            i == 0 ? h.tokenizer_data_offset : align_up(previous_end, kTokenizerAlign);
        require(record.data_offset == expected_offset,
                "q5090 tokenizer asset offset is not canonical");
        const std::uint64_t data_end = checked_add(record.data_offset, record.data_bytes);
        require(data_end <= checked_add(h.tokenizer_data_offset, h.tokenizer_data_bytes),
                "q5090 tokenizer asset outside data region");
        require_zero_range(file, previous_end, record.data_offset - previous_end,
                           "q5090 tokenizer inter-asset padding nonzero");
        const auto data = file.subspan(static_cast<std::size_t>(record.data_offset),
                                       static_cast<std::size_t>(record.data_bytes));
        require(crc32(data) == record.crc32, "q5090 tokenizer crc32 mismatch");
        require(sha256(data) == record.sha256, "q5090 tokenizer sha256 mismatch");
        require(valid_utf8(data), "q5090 tokenizer asset is not valid UTF-8");
        const auto* chars = reinterpret_cast<const char*>(data.data());
        std::string value(chars, chars + data.size());
        switch (record.kind) {
        case Q5090TokenizerKind::TokenizerJson:
            bundle.tokenizer_json = std::move(value);
            break;
        case Q5090TokenizerKind::MergesTxt:
            bundle.merges_txt = std::move(value);
            break;
        case Q5090TokenizerKind::GenerationConfig:
            bundle.generation_config_json = std::move(value);
            break;
        }
        previous_end = data_end;
        records.push_back(record);
    }
    require(h.tokenizer_data_bytes == previous_end - h.tokenizer_data_offset,
            "q5090 tokenizer data byte span mismatch");
    const std::uint64_t tokenizer_end =
        checked_add(h.tokenizer_data_offset, h.tokenizer_data_bytes);
    require_zero_range(file, tokenizer_end, h.payload_offset - tokenizer_end,
                       "q5090 tokenizer-to-payload padding nonzero");
    return records;
}

std::uint32_t parse_tokenizer_id(const nlohmann::json& value, std::uint32_t vocab_size,
                                 const char* label) {
    require(value.is_number_integer(), label);
    std::uint64_t id = 0;
    if (value.is_number_unsigned()) {
        id = value.get<std::uint64_t>();
    } else {
        const std::int64_t signed_id = value.get<std::int64_t>();
        require(signed_id >= 0, label);
        id = static_cast<std::uint64_t>(signed_id);
    }
    require(id < vocab_size, label);
    return static_cast<std::uint32_t>(id);
}

bool require_json_bool(const nlohmann::json& object, const char* field) {
    const auto it = object.find(field);
    require(it != object.end() && it->is_boolean(), "q5090 added token boolean field is invalid");
    return it->get<bool>();
}

void validate_tokenizer_semantics(const Q5090TokenizerBundle& bundle, std::uint32_t vocab_size,
                                  bool require_canonical) {
    using Json = nlohmann::json;
    try {
        const Json tokenizer = Json::parse(bundle.tokenizer_json);
        require(tokenizer.is_object(), "q5090 tokenizer.json root is not an object");
        const auto model_it = tokenizer.find("model");
        require(model_it != tokenizer.end() && model_it->is_object(),
                "q5090 tokenizer.json model is missing");
        const auto type_it = model_it->find("type");
        require(type_it != model_it->end() && type_it->is_string() &&
                    type_it->get<std::string>() == "BPE",
                "q5090 tokenizer.json model type is not BPE");
        const auto vocab_it = model_it->find("vocab");
        require(vocab_it != model_it->end() && vocab_it->is_object() && !vocab_it->empty(),
                "q5090 tokenizer.json vocab is missing or empty");

        std::unordered_set<std::uint32_t> occupied_ids;
        occupied_ids.reserve(vocab_it->size());
        for (const auto& item : vocab_it->items()) {
            const std::uint32_t id = parse_tokenizer_id(item.value(), vocab_size,
                                                        "q5090 tokenizer vocab id is out of range");
            require(occupied_ids.insert(id).second, "q5090 tokenizer vocab contains duplicate id");
        }

        const auto added_it = tokenizer.find("added_tokens");
        require(added_it != tokenizer.end() && added_it->is_array(),
                "q5090 tokenizer.json added_tokens is missing or invalid");
        std::unordered_set<std::uint32_t> added_ids;
        added_ids.reserve(added_it->size());
        for (const Json& item : *added_it) {
            require(item.is_object(), "q5090 added token entry is not an object");
            const auto id_it = item.find("id");
            require(id_it != item.end(), "q5090 added token id is missing");
            const std::uint32_t id =
                parse_tokenizer_id(*id_it, vocab_size, "q5090 added token id is out of range");
            require(!occupied_ids.contains(id), "q5090 added token id overlaps base vocab");
            require(added_ids.insert(id).second, "q5090 added token id is duplicated");
            const auto content_it = item.find("content");
            require(content_it != item.end() && content_it->is_string(),
                    "q5090 added token content is missing or invalid");
            const bool single_word = require_json_bool(item, "single_word");
            const bool lstrip      = require_json_bool(item, "lstrip");
            const bool rstrip      = require_json_bool(item, "rstrip");
            const bool normalized  = require_json_bool(item, "normalized");
            (void)require_json_bool(item, "special");
            require(!single_word && !lstrip && !rstrip && !normalized,
                    "q5090 tokenizer uses unsupported added-token matching flags");
            occupied_ids.insert(id);
        }

        if (require_canonical) {
            require(vocab_size == 248320 && occupied_ids.size() == kCanonicalTokenizerIds,
                    "q5090 canonical tokenizer occupied-id domain mismatch");
            for (std::uint32_t id = 0; id < kCanonicalTokenizerIds; ++id) {
                require(occupied_ids.contains(id),
                        "q5090 canonical tokenizer occupied-id domain is not contiguous");
            }
        }

        const Json generation = Json::parse(bundle.generation_config_json);
        require(generation.is_object(), "q5090 generation_config.json root is not an object");
        const auto eos_it = generation.find("eos_token_id");
        require(eos_it != generation.end(), "q5090 generation_config.json eos_token_id is missing");
        auto validate_eos = [&](const Json& value) {
            const std::uint32_t id =
                parse_tokenizer_id(value, vocab_size, "q5090 eos_token_id is out of range");
            require(occupied_ids.contains(id), "q5090 eos_token_id is absent from tokenizer vocab");
        };
        if (eos_it->is_array()) {
            require(!eos_it->empty(), "q5090 eos_token_id array is empty");
            for (const Json& value : *eos_it) { validate_eos(value); }
        } else {
            validate_eos(*eos_it);
        }
    } catch (const nlohmann::json::exception& ex) {
        throw std::runtime_error(std::string("q5090 tokenizer JSON is malformed: ") + ex.what());
    }

    require(bundle.merges_txt.find('\0') == std::string::npos, "q5090 merges.txt contains NUL");
    std::istringstream input(bundle.merges_txt);
    std::string line;
    bool saw_content = false;
    std::unordered_set<std::string> merge_pairs;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (line.empty()) { continue; }
        if (!saw_content && line.starts_with("#version")) {
            saw_content = true;
            continue;
        }
        saw_content                   = true;
        const std::size_t first_space = line.find(' ');
        require(first_space != std::string::npos && first_space != 0 &&
                    first_space + 1 < line.size() &&
                    line.find(' ', first_space + 1) == std::string::npos,
                "q5090 merges.txt contains a malformed merge");
        std::string key = line.substr(0, first_space);
        key.push_back('\0');
        key.append(line.substr(first_space + 1));
        require(merge_pairs.insert(std::move(key)).second,
                "q5090 merges.txt contains a duplicate merge pair");
    }
}

void validate_tokenizer_identity(const std::vector<ParsedQ5090TokenizerRecord>& records) {
    require(records.size() == kCanonicalTokenizerSha256.size(),
            "q5090 canonical tokenizer record count mismatch");
    for (std::size_t i = 0; i < records.size(); ++i) {
        require(records[i].sha256 == kCanonicalTokenizerSha256[i],
                "q5090 tokenizer identity does not match canonical Qwen3.6-27B");
    }
}

std::vector<ParsedQ5090Module> parse_modules(std::span<const std::byte> file,
                                             const ParsedQ5090Header& h) {
    std::vector<ParsedQ5090Module> modules;
    modules.reserve(h.module_count);
    std::uint64_t expected_begin       = 0;
    int previous_order                 = -1;
    std::uint64_t previous_payload_end = h.payload_offset;
    for (std::uint32_t i = 0; i < h.module_count; ++i) {
        const std::uint64_t off = h.module_index_offset + checked_mul(i, kModuleRecordSize);
        ParsedQ5090Module m;
        const std::uint32_t raw_kind = read_u32(file, off);
        m.module_kind                = module_from_tag(raw_kind);
        m.module_version             = read_u32(file, off + 4);
        m.tensor_index_begin         = read_u64(file, off + 8);
        m.tensor_index_count         = read_u64(file, off + 16);
        m.payload_offset             = read_u64(file, off + 24);
        m.payload_bytes              = read_u64(file, off + 32);
        require(read_u32(file, off + 40) == 0,
                "q5090 module load-policy field must be reserved zero");
        require(read_u32(file, off + 44) == 0, "q5090 module flags field must be reserved zero");
        require(all_zero(file.subspan(static_cast<std::size_t>(off + 48), 16)),
                "q5090 module reserved bytes nonzero");
        require(m.module_version == kVersion, "q5090 bad module version");
        require(m.tensor_index_count > 0, "q5090 empty module");
        require(m.tensor_index_begin == expected_begin,
                "q5090 module tensor ranges are not contiguous");
        expected_begin = checked_add(expected_begin, m.tensor_index_count);
        require(checked_add(m.tensor_index_begin, m.tensor_index_count) <= h.tensor_count,
                "q5090 module tensor range outside tensor index");
        int order = -1;
        switch (m.module_kind) {
        case ModuleKind::TextCore:
            order = 0;
            break;
        case ModuleKind::LmHeadDraft:
            order = 1;
            break;
        case ModuleKind::MtpDraft:
            order = 2;
            break;
        case ModuleKind::VisionEncoder:
            order = 3;
            break;
        }
        require(order > previous_order, "q5090 modules are not in canonical order");
        if (i == 0) {
            require(m.module_kind == ModuleKind::TextCore, "q5090 first module is not TEXT_CORE");
        }
        previous_order = order;
        require(m.payload_bytes > 0, "q5090 module payload span is empty");
        require(m.payload_offset >= h.payload_offset,
                "q5090 module payload begins before payload region");
        require(m.payload_offset >= previous_payload_end,
                "q5090 module payload spans are not ordered");
        require(checked_add(m.payload_offset, m.payload_bytes) <=
                    checked_add(h.payload_offset, h.payload_bytes),
                "q5090 module payload outside payload region");
        previous_payload_end = checked_add(m.payload_offset, m.payload_bytes);
        modules.push_back(m);
    }
    require(expected_begin == h.tensor_count,
            "q5090 module tensor ranges do not cover tensor index");
    return modules;
}

std::uint32_t module_flags(const std::vector<ParsedQ5090Module>& modules) {
    std::uint32_t flags = 0;
    for (const ParsedQ5090Module& module : modules) {
        switch (module.module_kind) {
        case ModuleKind::TextCore:
            flags |= 1U << 0;
            break;
        case ModuleKind::MtpDraft:
            flags |= 1U << 1;
            break;
        case ModuleKind::VisionEncoder:
            flags |= 1U << 2;
            break;
        case ModuleKind::LmHeadDraft:
            flags |= kLmHeadDraftPresentFlag;
            break;
        }
    }
    return flags;
}

const ParsedQ5090Module& module_for_tensor_index(const std::vector<ParsedQ5090Module>& modules,
                                                 std::uint64_t index) {
    for (const ParsedQ5090Module& module : modules) {
        const std::uint64_t end = checked_add(module.tensor_index_begin, module.tensor_index_count);
        if (index >= module.tensor_index_begin && index < end) { return module; }
    }
    parse_error("q5090 tensor index not covered by any module");
}

std::string read_name(std::span<const std::byte> file, const ParsedQ5090Header& h,
                      std::uint32_t name_offset, std::uint32_t name_len) {
    require(name_len > 0 && name_len <= kMaxCatalogNameBytes,
            "q5090 catalog name length exceeds cap");
    const std::uint64_t name_end = checked_add(name_offset, name_len);
    require(checked_add(name_end, 1) <= h.string_table_bytes, "q5090 name outside string table");
    const std::uint64_t absolute = checked_add(h.string_table_offset, name_offset);
    require(file[static_cast<std::size_t>(absolute + name_len)] == std::byte{0},
            "q5090 name is not NUL-terminated");
    const auto* chars = reinterpret_cast<const char*>(file.data() + absolute);
    const std::string name(chars, chars + name_len);
    require(name.find('\0') == std::string::npos, "q5090 catalog name contains embedded NUL");
    require(valid_utf8(file.subspan(static_cast<std::size_t>(absolute), name_len)),
            "q5090 catalog name is not valid UTF-8");
    return name;
}

std::vector<ParsedQ5090Tensor> parse_tensors(std::span<const std::byte> file,
                                             const ParsedQ5090Header& h,
                                             const std::vector<ParsedQ5090Module>& modules,
                                             bool validate_payload_padding) {
    std::vector<ParsedQ5090Tensor> tensors;
    tensors.reserve(h.tensor_count);
    std::vector<Range> ranges;
    ranges.reserve(h.tensor_count);
    std::set<std::string> seen_names;
    std::uint64_t previous_payload_end = h.payload_offset;
    for (std::uint32_t i = 0; i < h.tensor_count; ++i) {
        const std::uint64_t off = h.tensor_index_offset + checked_mul(i, kTensorEntrySize);
        ParsedQ5090Tensor t;
        t.name_offset = read_u32(file, off);
        t.name_len    = read_u32(file, off + 4);
        t.name_hash   = read_u64(file, off + 8);
        t.qtype       = qtype_from_tag(read_u16(file, off + 16));
        t.layout      = layout_from_tag(read_u16(file, off + 18));
        t.module_kind = module_from_tag(read_u16(file, off + 20));
        t.ndim        = read_u16(file, off + 22);
        for (int d = 0; d < 4; ++d) {
            t.shape[d]        = read_u32(file, off + 24 + d * 4);
            t.padded_shape[d] = read_u32(file, off + 40 + d * 4);
        }
        t.group_size         = read_u32(file, off + 56);
        t.scale_dtype        = scale_from_tag(read_u16(file, off + 60));
        t.segment_count      = read_u16(file, off + 62);
        t.payload_offset     = read_u64(file, off + 64);
        t.payload_bytes      = read_u64(file, off + 72);
        t.source_layer       = read_u32(file, off + 80);
        t.source_kind        = read_u32(file, off + 84);
        t.crc32              = read_u32(file, off + 88);
        t.segment_begin      = read_u32(file, off + 92);
        t.fusion_group_id    = read_u16(file, off + 96);
        t.fusion_index       = read_u16(file, off + 98);
        t.nibble_plane_bytes = read_u64(file, off + 100);
        t.high_plane_bytes   = read_u64(file, off + 108);
        t.scale_plane_bytes  = read_u64(file, off + 116);
        require(all_zero(file.subspan(static_cast<std::size_t>(off + 124), 4)),
                "q5090 tensor reserved bytes nonzero");

        const ParsedQ5090Module& module = module_for_tensor_index(modules, i);
        require(t.module_kind == module.module_kind,
                "q5090 tensor module does not match module range");
        t.name = read_name(file, h, t.name_offset, t.name_len);
        require(q5090_fnv1a64(t.name) == t.name_hash, "q5090 tensor name hash mismatch");
        require(t.segment_count > 0, "q5090 tensor has no segments");
        require(t.payload_bytes > 0, "q5090 tensor payload is empty");
        require(t.payload_offset % kPayloadAlign == 0, "q5090 tensor payload is not aligned");
        require(t.payload_offset >= h.payload_offset,
                "q5090 tensor payload begins before payload region");
        const std::uint64_t payload_end = checked_add(t.payload_offset, t.payload_bytes);
        require(payload_end <= checked_add(h.payload_offset, h.payload_bytes),
                "q5090 tensor payload outside payload region");
        require(t.payload_offset >= module.payload_offset &&
                    payload_end <= checked_add(module.payload_offset, module.payload_bytes),
                "q5090 tensor payload outside module span");
        require(t.payload_offset >= previous_payload_end,
                "q5090 tensor payload offsets are not ordered");
        if (validate_payload_padding) {
            require_zero_range(file, previous_payload_end, t.payload_offset - previous_payload_end,
                               "q5090 payload padding nonzero");
        }
        previous_payload_end = payload_end;
        require(t.source_layer <= 63 || t.source_layer == kQ5090NoLayer,
                "q5090 invalid source layer");
        require(valid_source_kind(t.module_kind, t.source_kind), "q5090 invalid source kind");
        require(t.fusion_group_id == 0 || valid_fusion_group_id(t.fusion_group_id),
                "q5090 invalid tensor fusion group id");
        if (t.fusion_group_id == 0) {
            require(t.fusion_index == 0, "q5090 standalone tensor has nonzero fusion index");
        }
        require(seen_names.insert(t.name).second, "q5090 duplicate tensor name");
        require(expected_payload_bytes(t) == t.payload_bytes, "q5090 tensor payload byte mismatch");
        ranges.push_back(Range{t.payload_offset, payload_end});
        tensors.push_back(std::move(t));
    }
    if (validate_payload_padding) {
        require_zero_range(file, previous_payload_end,
                           checked_add(h.payload_offset, h.payload_bytes) - previous_payload_end,
                           "q5090 payload padding nonzero");
    }

    std::sort(ranges.begin(), ranges.end(),
              [](const Range& a, const Range& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        require(ranges[i].begin >= ranges[i - 1].end, "q5090 tensor payload ranges overlap");
    }

    for (const ParsedQ5090Module& module : modules) {
        std::uint64_t min_begin = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t max_end   = 0;
        for (std::uint64_t i = module.tensor_index_begin;
             i < checked_add(module.tensor_index_begin, module.tensor_index_count); ++i) {
            const ParsedQ5090Tensor& tensor = tensors[static_cast<std::size_t>(i)];
            min_begin                       = std::min(min_begin, tensor.payload_offset);
            max_end = std::max(max_end, checked_add(tensor.payload_offset, tensor.payload_bytes));
        }
        require(module.payload_offset == min_begin, "q5090 module payload offset mismatch");
        require(checked_add(module.payload_offset, module.payload_bytes) == max_end,
                "q5090 module payload span mismatch");
    }
    return tensors;
}

std::vector<ParsedQ5090Segment> parse_segments(std::span<const std::byte> file,
                                               const ParsedQ5090Header& h) {
    std::vector<ParsedQ5090Segment> segments;
    segments.reserve(h.segment_count);
    std::set<std::string> seen_names;
    for (std::uint32_t i = 0; i < h.segment_count; ++i) {
        const std::uint64_t off = h.segment_index_offset + checked_mul(i, kSegmentRecordSize);
        ParsedQ5090Segment s;
        s.source_kind  = read_u32(file, off);
        s.source_layer = read_u32(file, off + 4);
        s.row_begin    = read_u32(file, off + 8);
        s.row_count    = read_u32(file, off + 12);
        s.name_offset  = read_u32(file, off + 16);
        s.name_len     = read_u32(file, off + 20);
        s.name_hash    = read_u64(file, off + 24);
        s.name         = read_name(file, h, s.name_offset, s.name_len);
        require(q5090_fnv1a64(s.name) == s.name_hash, "q5090 segment name hash mismatch");
        require(s.source_layer <= 63 || s.source_layer == kQ5090NoLayer,
                "q5090 invalid segment source layer");
        require(valid_source_kind(s.source_kind), "q5090 invalid segment source kind");
        require(s.row_count > 0, "q5090 segment row_count is zero");
        require(seen_names.insert(s.name).second, "q5090 duplicate segment name");
        segments.push_back(std::move(s));
    }
    return segments;
}

void validate_tensor_segments(const std::vector<ParsedQ5090Tensor>& tensors,
                              const std::vector<ParsedQ5090Segment>& segments) {
    std::set<std::tuple<std::uint16_t, std::uint32_t, std::uint32_t>> seen_sources;
    for (const ParsedQ5090Tensor& t : tensors) {
        const std::uint64_t end = checked_add(t.segment_begin, t.segment_count);
        require(end <= segments.size(), "q5090 tensor segment range outside table");
        const auto module = static_cast<std::uint16_t>(t.module_kind);
        std::uint64_t row = 0;
        for (std::uint32_t j = 0; j < t.segment_count; ++j) {
            const ParsedQ5090Segment& s = segments[static_cast<std::size_t>(t.segment_begin + j)];
            require(valid_source_kind(t.module_kind, s.source_kind),
                    "q5090 invalid segment source kind for module");
            if (t.layout == QuantLayout::RowSplit) {
                require(static_cast<std::uint64_t>(s.row_begin) == row,
                        "q5090 ROW_SPLIT segments are not contiguous");
                row = checked_add(row, s.row_count);
                require(row <= t.shape[0], "q5090 ROW_SPLIT segment exceeds N");
            } else {
                require(t.segment_count == 1, "q5090 CONTIGUOUS block must have one segment");
                require(s.row_begin == 0, "q5090 CONTIGUOUS segment row_begin must be zero");
                require(s.row_count == t.shape[0], "q5090 CONTIGUOUS segment row_count mismatch");
            }
            if (t.segment_count > 1) {
                require(s.source_layer == t.source_layer,
                        "q5090 fused block segment source_layer mismatch");
            }
            if (s.source_kind != static_cast<std::uint32_t>(SourceKind::Other)) {
                const auto key = std::make_tuple(module, s.source_kind, s.source_layer);
                require(seen_sources.insert(key).second, "q5090 duplicate segment source id");
            }
        }
        if (t.layout == QuantLayout::RowSplit) {
            require(row == t.shape[0], "q5090 ROW_SPLIT segments do not partition N");
        }

        const ParsedQ5090Segment& first = segments[static_cast<std::size_t>(t.segment_begin)];
        if (t.segment_count == 1) {
            require(t.source_kind == first.source_kind && t.source_layer == first.source_layer,
                    "q5090 single-segment source identity mismatch");
            if (t.fusion_group_id == 0) {
                require(t.name == first.name, "q5090 standalone segment name mismatch");
            }
        } else {
            require(t.source_kind == static_cast<std::uint32_t>(SourceKind::Other),
                    "q5090 fused block source_kind must be OTHER");
        }
    }
}

std::vector<ParsedQ5090FusionGroup> parse_fusion_groups(std::span<const std::byte> file,
                                                        const ParsedQ5090Header& h) {
    std::vector<ParsedQ5090FusionGroup> groups;
    groups.reserve(h.fusion_group_count);
    for (std::uint32_t i = 0; i < h.fusion_group_count; ++i) {
        const std::uint64_t off =
            h.fusion_group_index_offset + checked_mul(i, kFusionGroupRecordSize);
        ParsedQ5090FusionGroup g;
        g.group_id                 = read_u32(file, off);
        g.source_layer             = read_u32(file, off + 4);
        g.block_count              = read_u32(file, off + 8);
        g.shared_input_kind        = read_u32(file, off + 12);
        g.first_block_tensor_index = read_u64(file, off + 16);
        g.payload_offset           = read_u64(file, off + 24);
        g.payload_bytes            = read_u64(file, off + 32);
        g.total_n                  = read_u32(file, off + 40);
        g.shared_k                 = read_u32(file, off + 44);
        require(all_zero(file.subspan(static_cast<std::size_t>(off + 48), 16)),
                "q5090 fusion group reserved bytes nonzero");
        require(valid_fusion_group_id(g.group_id), "q5090 invalid fusion group id");
        require(g.source_layer <= 63, "q5090 invalid fusion group source layer");
        require(g.block_count > 0, "q5090 empty fusion group");
        require(valid_source_kind(g.shared_input_kind), "q5090 invalid fusion shared input kind");
        require(checked_add(g.first_block_tensor_index, g.block_count) <= h.tensor_count,
                "q5090 fusion group tensor range outside table");
        require(g.payload_offset >= h.payload_offset,
                "q5090 fusion group payload begins before payload region");
        require(checked_add(g.payload_offset, g.payload_bytes) <=
                    checked_add(h.payload_offset, h.payload_bytes),
                "q5090 fusion group payload outside payload region");
        groups.push_back(g);
    }
    return groups;
}

void validate_fusion_groups(const std::vector<ParsedQ5090Tensor>& tensors,
                            const std::vector<ParsedQ5090FusionGroup>& groups) {
    std::vector<bool> covered(tensors.size(), false);
    std::uint64_t previous_first = 0;
    bool have_previous           = false;
    for (const ParsedQ5090FusionGroup& g : groups) {
        if (have_previous) {
            require(g.first_block_tensor_index > previous_first,
                    "q5090 fusion groups are not ordered");
        }
        have_previous  = true;
        previous_first = g.first_block_tensor_index;

        std::uint64_t total_n                 = 0;
        const std::uint64_t first             = g.first_block_tensor_index;
        const std::uint64_t last              = first + g.block_count - 1;
        const ParsedQ5090Tensor& first_tensor = tensors[static_cast<std::size_t>(first)];
        const ParsedQ5090Tensor& last_tensor  = tensors[static_cast<std::size_t>(last)];
        require(g.payload_offset == first_tensor.payload_offset,
                "q5090 fusion payload offset mismatch");
        require(checked_add(g.payload_offset, g.payload_bytes) ==
                    checked_add(last_tensor.payload_offset, last_tensor.payload_bytes),
                "q5090 fusion payload span mismatch");

        for (std::uint32_t j = 0; j < g.block_count; ++j) {
            const std::size_t index = static_cast<std::size_t>(g.first_block_tensor_index + j);
            require(!covered[index], "q5090 overlapping fusion groups");
            covered[index]             = true;
            const ParsedQ5090Tensor& t = tensors[index];
            require(t.layout == QuantLayout::RowSplit, "q5090 fusion member must be ROW_SPLIT");
            require(t.fusion_group_id == g.group_id, "q5090 fusion member group id mismatch");
            require(t.fusion_index == j, "q5090 fusion member index mismatch");
            require(t.source_layer == g.source_layer, "q5090 fusion member layer mismatch");
            require(t.shape[1] == g.shared_k, "q5090 fusion member K mismatch");
            total_n = checked_add(total_n, t.shape[0]);
        }
        require(total_n == g.total_n, "q5090 fusion total_n mismatch");
    }

    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].fusion_group_id != 0) {
            require(covered[i], "q5090 tensor references missing fusion group");
        }
    }
}

const ParsedQ5090Tensor& require_model_tensor(const std::vector<ParsedQ5090Tensor>& tensors,
                                              std::string_view name, ModuleKind module, QType qtype,
                                              QuantLayout layout, SourceKind source_kind,
                                              std::uint32_t source_layer,
                                              std::initializer_list<std::uint32_t> shape) {
    const auto it =
        std::find_if(tensors.begin(), tensors.end(), [&](const ParsedQ5090Tensor& tensor) {
            return tensor.name == name && tensor.module_kind == module;
        });
    require(it != tensors.end(), "q5090 required model tensor is missing");
    require(it->qtype == qtype && it->layout == layout &&
                it->source_kind == static_cast<std::uint32_t>(source_kind) &&
                it->source_layer == source_layer && it->ndim == shape.size(),
            "q5090 required model tensor metadata mismatch");
    std::size_t axis = 0;
    for (const std::uint32_t extent : shape) {
        require(it->shape[axis++] == extent, "q5090 required model tensor shape mismatch");
    }
    return *it;
}

const ParsedQ5090Segment& require_model_segment(const ParsedQ5090Tensor& tensor,
                                                const std::vector<ParsedQ5090Segment>& segments,
                                                std::string_view name, SourceKind source_kind,
                                                std::uint32_t source_layer, std::uint32_t row_begin,
                                                std::uint32_t row_count) {
    const std::uint64_t end = checked_add(tensor.segment_begin, tensor.segment_count);
    for (std::uint64_t i = tensor.segment_begin; i < end; ++i) {
        const ParsedQ5090Segment& segment = segments[static_cast<std::size_t>(i)];
        if (segment.name == name) {
            require(segment.source_kind == static_cast<std::uint32_t>(source_kind) &&
                        segment.source_layer == source_layer && segment.row_begin == row_begin &&
                        segment.row_count == row_count,
                    "q5090 required model segment metadata mismatch");
            return segment;
        }
    }
    parse_error("q5090 required model segment is missing");
}

void validate_qwen36_model_contract(const ParsedQ5090Header& header,
                                    const std::vector<ParsedQ5090Module>& modules,
                                    const std::vector<ParsedQ5090Tensor>& tensors,
                                    const std::vector<ParsedQ5090Segment>& segments,
                                    const std::vector<ParsedQ5090FusionGroup>& groups) {
    const auto text_module = std::find_if(modules.begin(), modules.end(), [](const auto& module) {
        return module.module_kind == ModuleKind::TextCore;
    });
    require(text_module != modules.end() && text_module->tensor_index_begin == 0 &&
                text_module->tensor_index_count == 819,
            "q5090 TEXT_CORE tensor inventory mismatch");
    const std::size_t text_segment_count =
        std::count_if(segments.begin(), segments.end(), [&](const auto& segment) {
            const auto tensor = std::find_if(tensors.begin(), tensors.end(), [&](const auto& item) {
                const std::uint64_t begin = item.segment_begin;
                const std::uint64_t end   = begin + item.segment_count;
                const std::uint64_t index = static_cast<std::uint64_t>(&segment - segments.data());
                return item.module_kind == ModuleKind::TextCore && index >= begin && index < end;
            });
            return tensor != tensors.end();
        });
    require(text_segment_count == 963, "q5090 TEXT_CORE segment inventory mismatch");
    require(std::count_if(groups.begin(), groups.end(),
                          [&](const auto& group) {
                              return group.first_block_tensor_index <
                                     text_module->tensor_index_count;
                          }) == 128,
            "q5090 TEXT_CORE fusion inventory mismatch");

    const auto layer_name = [](std::uint32_t layer, std::string_view suffix) {
        return std::string("layers.") + std::to_string(layer) + "." + std::string(suffix);
    };
    const auto require_group = [&](std::uint32_t id, const ParsedQ5090Tensor& first,
                                   std::uint32_t block_count, std::uint32_t total_n,
                                   std::uint32_t shared_k, const char* message) {
        const std::uint64_t first_index = static_cast<std::uint64_t>(&first - tensors.data());
        const auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& group) {
            return group.group_id == id && group.first_block_tensor_index == first_index;
        });
        require(it != groups.end() && it->block_count == block_count &&
                    it->source_layer == first.source_layer && it->total_n == total_n &&
                    it->shared_k == shared_k,
                message);
    };

    (void)require_model_tensor(tensors, "model.language_model.embed_tokens.weight",
                               ModuleKind::TextCore, QType::Q6G64_F16S, QuantLayout::RowSplit,
                               SourceKind::Embed, kQ5090NoLayer,
                               {header.vocab_size, header.hidden_size});
    (void)require_model_tensor(tensors, "model.language_model.norm.weight", ModuleKind::TextCore,
                               QType::BF16_CTRL, QuantLayout::Contiguous, SourceKind::FinalNorm,
                               kQ5090NoLayer, {header.hidden_size});
    (void)require_model_tensor(tensors, "lm_head.weight", ModuleKind::TextCore, QType::Q6G64_F16S,
                               QuantLayout::RowSplit, SourceKind::LmHead, kQ5090NoLayer,
                               {header.vocab_size, header.hidden_size});

    const std::uint32_t hidden       = header.hidden_size;
    const std::uint32_t intermediate = header.intermediate_size;
    const std::uint32_t q_rows =
        static_cast<std::uint32_t>(checked_mul(header.num_attention_heads, header.head_dim));
    const std::uint32_t kv_rows =
        static_cast<std::uint32_t>(checked_mul(header.num_key_value_heads, header.head_dim));
    const std::uint32_t attn_block_rows = static_cast<std::uint32_t>(checked_add(q_rows, kv_rows));
    const std::uint32_t gdn_k_rows =
        static_cast<std::uint32_t>(checked_mul(header.gdn_key_heads, header.gdn_key_head_dim));
    const std::uint32_t gdn_v_rows =
        static_cast<std::uint32_t>(checked_mul(header.gdn_value_heads, header.gdn_value_head_dim));
    const std::uint32_t gdn_qk_rows = static_cast<std::uint32_t>(checked_mul(gdn_k_rows, 2));
    const std::uint32_t gdn_total_rows =
        static_cast<std::uint32_t>(checked_add(gdn_qk_rows, gdn_v_rows));
    const std::uint32_t mlp_rows = static_cast<std::uint32_t>(checked_mul(intermediate, 2));

    for (std::uint32_t layer = 0; layer < header.layer_count; ++layer) {
        (void)require_model_tensor(tensors, layer_name(layer, "input_layernorm.weight"),
                                   ModuleKind::TextCore, QType::BF16_CTRL, QuantLayout::Contiguous,
                                   SourceKind::InputLayernorm, layer, {hidden});
        (void)require_model_tensor(tensors, layer_name(layer, "post_attention_layernorm.weight"),
                                   ModuleKind::TextCore, QType::BF16_CTRL, QuantLayout::Contiguous,
                                   SourceKind::PostAttnLayernorm, layer, {hidden});

        if ((layer + 1) % header.full_attention_interval == 0) {
            const ParsedQ5090Tensor& qk = require_model_tensor(
                tensors, layer_name(layer, "attn_in.q4"), ModuleKind::TextCore, QType::Q4G64_F16S,
                QuantLayout::RowSplit, SourceKind::Other, layer, {attn_block_rows, hidden});
            require(qk.segment_count == 2, "q5090 TEXT full-attention q/k segment mismatch");
            (void)require_model_segment(qk, segments, layer_name(layer, "self_attn.q_proj.q"),
                                        SourceKind::AttnQ, layer, 0, q_rows);
            (void)require_model_segment(qk, segments, layer_name(layer, "self_attn.k_proj.weight"),
                                        SourceKind::AttnK, layer, q_rows, kv_rows);
            const ParsedQ5090Tensor& gatev = require_model_tensor(
                tensors, layer_name(layer, "attn_in.q5"), ModuleKind::TextCore, QType::Q5G64_F16S,
                QuantLayout::RowSplit, SourceKind::Other, layer, {attn_block_rows, hidden});
            require(gatev.segment_count == 2, "q5090 TEXT full-attention gate/v segment mismatch");
            (void)require_model_segment(gatev, segments, layer_name(layer, "self_attn.q_proj.gate"),
                                        SourceKind::AttnGate, layer, 0, q_rows);
            (void)require_model_segment(gatev, segments,
                                        layer_name(layer, "self_attn.v_proj.weight"),
                                        SourceKind::AttnV, layer, q_rows, kv_rows);
            (void)require_model_tensor(tensors, layer_name(layer, "self_attn.q_norm.weight"),
                                       ModuleKind::TextCore, QType::BF16_CTRL,
                                       QuantLayout::Contiguous, SourceKind::AttnQNorm, layer,
                                       {header.head_dim});
            (void)require_model_tensor(tensors, layer_name(layer, "self_attn.k_norm.weight"),
                                       ModuleKind::TextCore, QType::BF16_CTRL,
                                       QuantLayout::Contiguous, SourceKind::AttnKNorm, layer,
                                       {header.head_dim});
            (void)require_model_tensor(tensors, layer_name(layer, "self_attn.o_proj.weight"),
                                       ModuleKind::TextCore, QType::Q5G64_F16S,
                                       QuantLayout::RowSplit, SourceKind::AttnO, layer,
                                       {hidden, q_rows});
            require_group(1, qk, 2, static_cast<std::uint32_t>(checked_mul(attn_block_rows, 2)),
                          hidden, "q5090 TEXT full-attention fusion contract mismatch");
        } else {
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.A_log"),
                                       ModuleKind::TextCore, QType::FP32_CTRL,
                                       QuantLayout::Contiguous, SourceKind::GdnALog, layer,
                                       {header.gdn_value_heads});
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.dt_bias"),
                                       ModuleKind::TextCore, QType::FP32_CTRL,
                                       QuantLayout::Contiguous, SourceKind::GdnDtBias, layer,
                                       {header.gdn_value_heads});
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.conv1d.weight"),
                                       ModuleKind::TextCore, QType::BF16_CTRL,
                                       QuantLayout::Contiguous, SourceKind::GdnConv1d, layer,
                                       {gdn_total_rows, header.gdn_conv_width, 1});
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.in_proj_a.weight"),
                                       ModuleKind::TextCore, QType::BF16_CTRL,
                                       QuantLayout::Contiguous, SourceKind::GdnInProjA, layer,
                                       {header.gdn_value_heads, hidden});
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.in_proj_b.weight"),
                                       ModuleKind::TextCore, QType::BF16_CTRL,
                                       QuantLayout::Contiguous, SourceKind::GdnInProjB, layer,
                                       {header.gdn_value_heads, hidden});
            const ParsedQ5090Tensor& qk = require_model_tensor(
                tensors, layer_name(layer, "gdn_in.q4"), ModuleKind::TextCore, QType::Q4G64_F16S,
                QuantLayout::RowSplit, SourceKind::Other, layer, {gdn_qk_rows, hidden});
            require(qk.segment_count == 2, "q5090 TEXT GDN q/k segment mismatch");
            (void)require_model_segment(qk, segments,
                                        layer_name(layer, "linear_attn.in_proj_qkv.q"),
                                        SourceKind::GdnInProjQ, layer, 0, gdn_k_rows);
            (void)require_model_segment(qk, segments,
                                        layer_name(layer, "linear_attn.in_proj_qkv.k"),
                                        SourceKind::GdnInProjK, layer, gdn_k_rows, gdn_k_rows);
            const ParsedQ5090Tensor& value = require_model_tensor(
                tensors, layer_name(layer, "gdn_in.q5"), ModuleKind::TextCore, QType::Q5G64_F16S,
                QuantLayout::RowSplit, SourceKind::GdnInProjV, layer, {gdn_v_rows, hidden});
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.norm.weight"),
                                       ModuleKind::TextCore, QType::BF16_CTRL,
                                       QuantLayout::Contiguous, SourceKind::GdnNorm, layer,
                                       {header.gdn_value_head_dim});
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.in_proj_z.weight"),
                                       ModuleKind::TextCore, QType::Q5G64_F16S,
                                       QuantLayout::RowSplit, SourceKind::GdnInProjZ, layer,
                                       {gdn_v_rows, hidden});
            (void)require_model_tensor(tensors, layer_name(layer, "linear_attn.out_proj.weight"),
                                       ModuleKind::TextCore, QType::Q5G64_F16S,
                                       QuantLayout::RowSplit, SourceKind::GdnOutProj, layer,
                                       {hidden, gdn_v_rows});
            require(value.fusion_index == 1, "q5090 TEXT GDN value fusion index mismatch");
            require_group(2, qk, 2, gdn_total_rows, hidden,
                          "q5090 TEXT GDN fusion contract mismatch");
        }

        const ParsedQ5090Tensor& gateup = require_model_tensor(
            tensors, layer_name(layer, "mlp.gateup"), ModuleKind::TextCore, QType::Q4G64_F16S,
            QuantLayout::RowSplit, SourceKind::Other, layer, {mlp_rows, hidden});
        require(gateup.segment_count == 2, "q5090 TEXT MLP gate/up segment mismatch");
        (void)require_model_segment(gateup, segments, layer_name(layer, "mlp.gate_proj.weight"),
                                    SourceKind::MlpGate, layer, 0, intermediate);
        (void)require_model_segment(gateup, segments, layer_name(layer, "mlp.up_proj.weight"),
                                    SourceKind::MlpUp, layer, intermediate, intermediate);
        (void)require_model_tensor(tensors, layer_name(layer, "mlp.down_proj.weight"),
                                   ModuleKind::TextCore, QType::Q5G64_F16S, QuantLayout::RowSplit,
                                   SourceKind::MlpDown, layer, {hidden, intermediate});
        require_group(3, gateup, 1, mlp_rows, hidden, "q5090 TEXT MLP fusion contract mismatch");
    }

    const auto module_it = std::find_if(modules.begin(), modules.end(), [](const auto& module) {
        return module.module_kind == ModuleKind::MtpDraft;
    });
    if (module_it == modules.end()) { return; }
    require(module_it->tensor_index_count == 12, "q5090 MTP_DRAFT tensor count mismatch");

    const auto checked_u32 = [](std::uint64_t value) {
        require(value <= std::numeric_limits<std::uint32_t>::max(),
                "q5090 model shape exceeds uint32");
        return static_cast<std::uint32_t>(value);
    };
    const std::uint32_t attn_rows =
        checked_u32(checked_add(checked_mul(q_rows, 2), checked_mul(kv_rows, 2)));
    const std::uint32_t double_hidden       = checked_u32(checked_mul(hidden, 2));
    const std::uint32_t double_intermediate = checked_u32(checked_mul(intermediate, 2));

    (void)require_model_tensor(tensors, "mtp.fc.weight", ModuleKind::MtpDraft, QType::W8G32_F16S,
                               QuantLayout::RowSplit, SourceKind::MtpFc, kQ5090NoLayer,
                               {hidden, double_hidden});
    (void)require_model_tensor(tensors, "mtp.pre_fc_norm_embedding.weight", ModuleKind::MtpDraft,
                               QType::BF16_CTRL, QuantLayout::Contiguous,
                               SourceKind::MtpPreFcNormEmb, kQ5090NoLayer, {hidden});
    (void)require_model_tensor(tensors, "mtp.pre_fc_norm_hidden.weight", ModuleKind::MtpDraft,
                               QType::BF16_CTRL, QuantLayout::Contiguous,
                               SourceKind::MtpPreFcNormHid, kQ5090NoLayer, {hidden});
    (void)require_model_tensor(tensors, "mtp.layers.0.input_layernorm.weight", ModuleKind::MtpDraft,
                               QType::BF16_CTRL, QuantLayout::Contiguous,
                               SourceKind::InputLayernorm, 0, {hidden});
    const ParsedQ5090Tensor& attn = require_model_tensor(
        tensors, "mtp.layers.0.attn_in.w8", ModuleKind::MtpDraft, QType::W8G32_F16S,
        QuantLayout::RowSplit, SourceKind::Other, 0, {attn_rows, hidden});
    require(attn.segment_count == 4, "q5090 MTP attention segment count mismatch");
    (void)require_model_segment(attn, segments, "mtp.layers.0.self_attn.q_proj.q",
                                SourceKind::AttnQ, 0, 0, q_rows);
    (void)require_model_segment(attn, segments, "mtp.layers.0.self_attn.k_proj.weight",
                                SourceKind::AttnK, 0, q_rows, kv_rows);
    const std::uint32_t gate_begin = checked_add(q_rows, kv_rows);
    (void)require_model_segment(attn, segments, "mtp.layers.0.self_attn.q_proj.gate",
                                SourceKind::AttnGate, 0, gate_begin, q_rows);
    (void)require_model_segment(attn, segments, "mtp.layers.0.self_attn.v_proj.weight",
                                SourceKind::AttnV, 0, checked_add(gate_begin, q_rows), kv_rows);
    (void)require_model_tensor(tensors, "mtp.layers.0.self_attn.q_norm.weight",
                               ModuleKind::MtpDraft, QType::BF16_CTRL, QuantLayout::Contiguous,
                               SourceKind::AttnQNorm, 0, {header.head_dim});
    (void)require_model_tensor(tensors, "mtp.layers.0.self_attn.k_norm.weight",
                               ModuleKind::MtpDraft, QType::BF16_CTRL, QuantLayout::Contiguous,
                               SourceKind::AttnKNorm, 0, {header.head_dim});
    (void)require_model_tensor(tensors, "mtp.layers.0.self_attn.o_proj.weight",
                               ModuleKind::MtpDraft, QType::W8G32_F16S, QuantLayout::RowSplit,
                               SourceKind::AttnO, 0, {hidden, q_rows});
    (void)require_model_tensor(tensors, "mtp.layers.0.post_attention_layernorm.weight",
                               ModuleKind::MtpDraft, QType::BF16_CTRL, QuantLayout::Contiguous,
                               SourceKind::PostAttnLayernorm, 0, {hidden});
    const ParsedQ5090Tensor& gateup = require_model_tensor(
        tensors, "mtp.layers.0.mlp.gateup.w8", ModuleKind::MtpDraft, QType::W8G32_F16S,
        QuantLayout::RowSplit, SourceKind::Other, 0, {double_intermediate, hidden});
    require(gateup.segment_count == 2, "q5090 MTP gate/up segment count mismatch");
    (void)require_model_segment(gateup, segments, "mtp.layers.0.mlp.gate_proj.weight",
                                SourceKind::MlpGate, 0, 0, intermediate);
    (void)require_model_segment(gateup, segments, "mtp.layers.0.mlp.up_proj.weight",
                                SourceKind::MlpUp, 0, intermediate, intermediate);
    (void)require_model_tensor(tensors, "mtp.layers.0.mlp.down_proj.weight", ModuleKind::MtpDraft,
                               QType::W8G32_F16S, QuantLayout::RowSplit, SourceKind::MlpDown, 0,
                               {hidden, intermediate});
    (void)require_model_tensor(tensors, "mtp.norm.weight", ModuleKind::MtpDraft, QType::BF16_CTRL,
                               QuantLayout::Contiguous, SourceKind::MtpNorm, kQ5090NoLayer,
                               {hidden});

    const auto require_mtp_group = [&](std::uint32_t id, const ParsedQ5090Tensor& tensor,
                                       std::uint32_t total_n) {
        const auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& group) {
            return group.group_id == id && group.first_block_tensor_index ==
                                               static_cast<std::uint64_t>(&tensor - tensors.data());
        });
        require(it != groups.end() && it->block_count == 1 && it->source_layer == 0 &&
                    it->total_n == total_n && it->shared_k == hidden,
                "q5090 MTP fusion-group contract mismatch");
    };
    require_mtp_group(1, attn, attn_rows);
    require_mtp_group(3, gateup, double_intermediate);
}

void validate_draft_head(std::span<const std::byte> file, const ParsedQ5090Header& header,
                         const std::vector<ParsedQ5090Tensor>& tensors,
                         bool validate_idmap_values) {
    const ParsedQ5090Tensor* weights = nullptr;
    const ParsedQ5090Tensor* idmap   = nullptr;
    std::size_t weights_index        = 0;
    std::size_t idmap_index          = 0;
    std::size_t module_blocks        = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const ParsedQ5090Tensor& t = tensors[i];
        if (t.module_kind != ModuleKind::LmHeadDraft) { continue; }
        ++module_blocks;
        if (t.source_kind == static_cast<std::uint32_t>(SourceKind::LmHeadDraft)) {
            require(weights == nullptr, "q5090 duplicate LM_HEAD_DRAFT block");
            weights       = &t;
            weights_index = i;
        } else if (t.source_kind == static_cast<std::uint32_t>(SourceKind::LmHeadDraftIdmap)) {
            require(idmap == nullptr, "q5090 duplicate LM_HEAD_DRAFT_IDMAP block");
            idmap       = &t;
            idmap_index = i;
        }
    }
    const bool flag_present  = (header.flags & kLmHeadDraftPresentFlag) != 0;
    const bool draft_present = weights != nullptr;
    require(flag_present == draft_present,
            "q5090 LM_HEAD_DRAFT_PRESENT flag inconsistent with module");
    require((idmap != nullptr) == draft_present,
            "q5090 LM_HEAD_DRAFT and LM_HEAD_DRAFT_IDMAP must both be present or absent");
    if (!draft_present) { return; }
    require(module_blocks == 2, "q5090 LM_HEAD_DRAFT module must contain exactly two blocks");
    require(idmap_index == weights_index + 1,
            "q5090 LM_HEAD_DRAFT blocks are not in canonical order");
    require(weights->qtype == QType::Q4G64_F16S && weights->layout == QuantLayout::RowSplit,
            "q5090 LM_HEAD_DRAFT must be Q4G64 ROW_SPLIT");
    require(weights->name == "lm_head_draft" && weights->ndim == 2,
            "q5090 LM_HEAD_DRAFT name/rank is not canonical");
    require(idmap->qtype == QType::I32_CTRL && idmap->layout == QuantLayout::Contiguous &&
                idmap->ndim == 1,
            "q5090 LM_HEAD_DRAFT_IDMAP must be I32_CTRL CONTIGUOUS 1D");
    require(idmap->name == "lm_head_draft.idmap",
            "q5090 LM_HEAD_DRAFT_IDMAP name is not canonical");
    require(weights->source_layer == kQ5090NoLayer && idmap->source_layer == kQ5090NoLayer,
            "q5090 draft-head blocks must have no source layer");
    require(idmap->shape[0] == weights->shape[0],
            "q5090 LM_HEAD_DRAFT_IDMAP length must match draft-head row count");
    require(weights->shape[0] == kCanonicalDraftRows &&
                weights->shape[1] == kCanonicalDraftHidden &&
                idmap->shape[0] == kCanonicalDraftRows,
            "q5090 LM_HEAD_DRAFT fixed v4.1 shape mismatch");
    if (!validate_idmap_values) { return; }
    std::set<std::uint32_t> ids;
    for (std::uint32_t i = 0; i < idmap->shape[0]; ++i) {
        const std::uint32_t id = read_u32(file, idmap->payload_offset + 4ULL * i);
        require(id < kCanonicalTokenizerIds,
                "q5090 draft-head id-map contains non-tokenizer vocab id");
        require(ids.insert(id).second, "q5090 draft-head id-map contains duplicate id");
    }
}

} // namespace

std::uint64_t q5090_fnv1a64(std::string_view name) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char b : name) {
        h ^= b;
        h *= 0x100000001B3ULL;
    }
    return h;
}

ParsedQ5090Header parse_q5090_header(std::span<const std::byte> header, std::uint64_t file_size,
                                     const Q5090Expectations& expected) {
    return parse_header(header, file_size, expected);
}

ParsedQ5090File parse_q5090_catalog(std::span<const std::byte> metadata, std::uint64_t file_size,
                                    const Q5090Expectations& expected) {
    ParsedQ5090File parsed;
    parsed.header = parse_header(metadata, file_size, expected);
    require(metadata.size() == parsed.header.payload_offset,
            "q5090 catalog buffer must end at payload_offset");
    require_range(metadata, parsed.header.module_index_offset, parsed.header.module_index_bytes,
                  "module index");
    require_range(metadata, parsed.header.tensor_index_offset, parsed.header.tensor_index_bytes,
                  "tensor index");
    require_range(metadata, parsed.header.segment_index_offset, parsed.header.segment_index_bytes,
                  "segment index");
    require_range(metadata, parsed.header.fusion_group_index_offset,
                  parsed.header.fusion_group_index_bytes, "fusion group index");
    require_range(metadata, parsed.header.string_table_offset, parsed.header.string_table_bytes,
                  "string table");
    require_range(metadata, parsed.header.tokenizer_index_offset,
                  parsed.header.tokenizer_index_bytes, "tokenizer index");
    require_range(metadata, parsed.header.tokenizer_data_offset, parsed.header.tokenizer_data_bytes,
                  "tokenizer data");
    const std::uint64_t string_end =
        checked_add(parsed.header.string_table_offset, parsed.header.string_table_bytes);
    require_zero_range(metadata, string_end, parsed.header.tokenizer_index_offset - string_end,
                       "q5090 string-to-tokenizer padding nonzero");
    const std::uint64_t tokenizer_index_end =
        checked_add(parsed.header.tokenizer_index_offset, parsed.header.tokenizer_index_bytes);
    require_zero_range(metadata, tokenizer_index_end,
                       parsed.header.tokenizer_data_offset - tokenizer_index_end,
                       "q5090 tokenizer index padding nonzero");

    parsed.modules = parse_modules(metadata, parsed.header);
    require((parsed.header.flags & 0x17U) == module_flags(parsed.modules),
            "q5090 header module flags mismatch");
    parsed.tokenizer_records = parse_tokenizer(metadata, parsed.header, parsed.tokenizer);
    validate_tokenizer_semantics(parsed.tokenizer, parsed.header.vocab_size,
                                 expected.validate_model_contract);
    if (expected.validate_model_contract) { validate_tokenizer_identity(parsed.tokenizer_records); }
    parsed.tensors  = parse_tensors(metadata, parsed.header, parsed.modules, false);
    parsed.segments = parse_segments(metadata, parsed.header);
    validate_tensor_segments(parsed.tensors, parsed.segments);
    parsed.fusion_groups = parse_fusion_groups(metadata, parsed.header);
    validate_fusion_groups(parsed.tensors, parsed.fusion_groups);
    if (expected.validate_model_contract) {
        validate_qwen36_model_contract(parsed.header, parsed.modules, parsed.tensors,
                                       parsed.segments, parsed.fusion_groups);
    }
    validate_draft_head(metadata, parsed.header, parsed.tensors, false);
    return parsed;
}

ParsedQ5090File parse_q5090_file(std::span<const std::byte> file, const Q5090Expectations& expected,
                                 Q5090Progress* progress) {
    (void)progress;
    ParsedQ5090File parsed;
    parsed.header = parse_header(file, file.size(), expected);
    const std::uint64_t string_end =
        checked_add(parsed.header.string_table_offset, parsed.header.string_table_bytes);
    require_zero_range(file, string_end, parsed.header.tokenizer_index_offset - string_end,
                       "q5090 string-to-tokenizer padding nonzero");
    const std::uint64_t tokenizer_index_end =
        checked_add(parsed.header.tokenizer_index_offset, parsed.header.tokenizer_index_bytes);
    require_zero_range(file, tokenizer_index_end,
                       parsed.header.tokenizer_data_offset - tokenizer_index_end,
                       "q5090 tokenizer index padding nonzero");
    parsed.modules = parse_modules(file, parsed.header);
    require((parsed.header.flags & 0x17U) == module_flags(parsed.modules),
            "q5090 header module flags mismatch");
    parsed.tokenizer_records = parse_tokenizer(file, parsed.header, parsed.tokenizer);
    validate_tokenizer_semantics(parsed.tokenizer, parsed.header.vocab_size,
                                 expected.validate_model_contract);
    if (expected.validate_model_contract) { validate_tokenizer_identity(parsed.tokenizer_records); }
    parsed.tensors  = parse_tensors(file, parsed.header, parsed.modules, true);
    parsed.segments = parse_segments(file, parsed.header);
    validate_tensor_segments(parsed.tensors, parsed.segments);
    parsed.fusion_groups = parse_fusion_groups(file, parsed.header);
    validate_fusion_groups(parsed.tensors, parsed.fusion_groups);
    if (expected.validate_model_contract) {
        validate_qwen36_model_contract(parsed.header, parsed.modules, parsed.tensors,
                                       parsed.segments, parsed.fusion_groups);
    }
    validate_draft_head(file, parsed.header, parsed.tensors, true);
    return parsed;
}

} // namespace qus
