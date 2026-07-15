#include "ninfer/ops/vision_attention.h"
#include "ops/launcher/vision_attention.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kDim   = 72;
constexpr std::int32_t kHeads = 16;

struct Options {
    std::int32_t segments = 1;
    std::int32_t length   = 256;
    std::int32_t tile     = 0;
    int warmup            = 2;
    int repeat            = 20;
    int min_time_ms       = 100;
    bool descriptor       = false;
    bool control          = false;
    bool profile_once     = false;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::fprintf(stderr, "usage: ninfer_vision_attention_bench [--segments S] [--length L] "
                         "[--tile auto|16|32|64] [--descriptor] [--control] [--profile-once] "
                         "[--warmup N] [--repeat N] [--min-time-ms N]\n");
    std::exit(2);
}

std::int32_t parse_positive(const char* text, const char* flag) {
    errno      = 0;
    char* end  = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        usage(flag);
    }
    return static_cast<std::int32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--segments")) {
            if (++i == argc) { usage("--segments requires a value"); }
            opt.segments = parse_positive(argv[i], "invalid --segments");
        } else if (!std::strcmp(argv[i], "--length")) {
            if (++i == argc) { usage("--length requires a value"); }
            opt.length = parse_positive(argv[i], "invalid --length");
        } else if (!std::strcmp(argv[i], "--tile")) {
            if (++i == argc) { usage("--tile requires a value"); }
            if (!std::strcmp(argv[i], "auto")) {
                opt.tile = 0;
            } else {
                opt.tile = parse_positive(argv[i], "invalid --tile");
                if (opt.tile != 16 && opt.tile != 32 && opt.tile != 64) {
                    usage("--tile expects auto, 16, 32, or 64");
                }
            }
        } else if (!std::strcmp(argv[i], "--descriptor")) {
            opt.descriptor = true;
        } else if (!std::strcmp(argv[i], "--control")) {
            opt.control = true;
        } else if (!std::strcmp(argv[i], "--profile-once")) {
            opt.profile_once = true;
        } else if (!std::strcmp(argv[i], "--warmup")) {
            if (++i == argc) { usage("--warmup requires a value"); }
            opt.warmup = parse_positive(argv[i], "invalid --warmup");
        } else if (!std::strcmp(argv[i], "--repeat")) {
            if (++i == argc) { usage("--repeat requires a value"); }
            opt.repeat = parse_positive(argv[i], "invalid --repeat");
        } else if (!std::strcmp(argv[i], "--min-time-ms")) {
            if (++i == argc) { usage("--min-time-ms requires a value"); }
            opt.min_time_ms = parse_positive(argv[i], "invalid --min-time-ms");
        } else {
            usage("unknown argument");
        }
    }
    if (opt.descriptor && opt.tile != 0) { usage("--descriptor uses the fixed 64 tile"); }
    if (opt.descriptor && opt.control) { usage("--control is only valid for uniform segments"); }
    const std::int64_t patches = static_cast<std::int64_t>(opt.segments) * opt.length;
    if (patches > std::numeric_limits<std::int32_t>::max()) { usage("S*L exceeds int32"); }
    return opt;
}

template <int Tile>
__launch_bounds__(Tile * 2, 128 / Tile) __global__
    void vision_attention_payload_control_kernel(const uint4* q, const uint4* k, const uint4* v,
                                                 uint4* out, uint4* sink,
                                                 std::int32_t segment_length) {
    constexpr int kThreads           = Tile * 2;
    constexpr int kWarps             = kThreads / 32;
    constexpr int kVectorsPerHead    = kDim / 8;
    constexpr int kInputTokenStride  = 3 * kHeads * kVectorsPerHead;
    constexpr int kOutputTokenStride = kHeads * kVectorsPerHead;
    const int tiles_per_segment      = (segment_length + Tile - 1) / Tile;
    const int segment                = static_cast<int>(blockIdx.x) / tiles_per_segment;
    const int tile_in_segment        = static_cast<int>(blockIdx.x) - segment * tiles_per_segment;
    const int begin                  = segment * segment_length;
    const int q0                     = begin + tile_in_segment * Tile;
    const int remaining              = begin + segment_length - q0;
    const int q_rows                 = remaining < Tile ? remaining : Tile;
    const int head                   = static_cast<int>(blockIdx.y);
    const int tid                    = static_cast<int>(threadIdx.x);

    for (int item = tid; item < q_rows * kVectorsPerHead; item += kThreads) {
        const int row   = item / kVectorsPerHead;
        const int dvec  = item - row * kVectorsPerHead;
        const int token = q0 + row;
        out[token * kOutputTokenStride + head * kVectorsPerHead + dvec] =
            q[token * kInputTokenStride + head * kVectorsPerHead + dvec];
    }

    uint4 checksum{0u, 0u, 0u, 0u};
    for (int item = tid; item < segment_length * kVectorsPerHead; item += kThreads) {
        const int row   = item / kVectorsPerHead;
        const int dvec  = item - row * kVectorsPerHead;
        const int index = (begin + row) * kInputTokenStride + head * kVectorsPerHead + dvec;
        const uint4 kv  = k[index];
        const uint4 vv  = v[index];
        checksum.x ^= kv.x ^ vv.x;
        checksum.y ^= kv.y ^ vv.y;
        checksum.z ^= kv.z ^ vv.z;
        checksum.w ^= kv.w ^ vv.w;
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
        checksum.x ^= __shfl_xor_sync(0xffffffffu, checksum.x, offset);
        checksum.y ^= __shfl_xor_sync(0xffffffffu, checksum.y, offset);
        checksum.z ^= __shfl_xor_sync(0xffffffffu, checksum.z, offset);
        checksum.w ^= __shfl_xor_sync(0xffffffffu, checksum.w, offset);
    }
    if ((tid & 31) == 0) {
        const int block                 = static_cast<int>(blockIdx.x) * kHeads + head;
        sink[block * kWarps + tid / 32] = checksum;
    }
}

void launch_control(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t segment_length,
                    std::int32_t tile, Tensor& out, DBuf& sink, cudaStream_t stream) {
    const std::int32_t segments = q.ne[2] / segment_length;
    const std::int32_t q_tiles  = segments * ((segment_length + tile - 1) / tile);
    const dim3 grid(static_cast<unsigned>(q_tiles), kHeads, 1u);
#define NINFER_VISION_CONTROL_LAUNCH(TILE)                                                         \
    do {                                                                                           \
        constexpr int kThreads = (TILE) * 2;                                                       \
        constexpr int kSmem    = 3 * (TILE) * 128 * static_cast<int>(sizeof(std::uint16_t));       \
        vision_attention_payload_control_kernel<(TILE)><<<grid, kThreads, kSmem, stream>>>(        \
            static_cast<const uint4*>(q.data), static_cast<const uint4*>(k.data),                  \
            static_cast<const uint4*>(v.data), static_cast<uint4*>(out.data),                      \
            static_cast<uint4*>(sink.p), segment_length);                                          \
    } while (0)
    switch (tile) {
    case 16:
        NINFER_VISION_CONTROL_LAUNCH(16);
        break;
    case 32:
        NINFER_VISION_CONTROL_LAUNCH(32);
        break;
    case 64:
        NINFER_VISION_CONTROL_LAUNCH(64);
        break;
    }
#undef NINFER_VISION_CONTROL_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    const Options opt             = parse_options(argc, argv);
    const std::int32_t patches    = opt.segments * opt.length;
    const std::size_t token_elems = static_cast<std::size_t>(kDim) * kHeads;
    const std::size_t plane_elems = token_elems * static_cast<std::size_t>(patches);

    DBuf qkv = make_bf16(plane_elems * 3u);
    DBuf out = make_zeros(plane_elems * sizeof(std::uint16_t));
    Tensor q(qkv.p, DType::BF16, {kDim, kHeads, patches});
    q.nb[2]  = static_cast<std::int64_t>(token_elems * 3u * sizeof(std::uint16_t));
    Tensor k = q;
    Tensor v = q;
    k.data   = static_cast<unsigned char*>(qkv.p) + token_elems * sizeof(std::uint16_t);
    v.data   = static_cast<unsigned char*>(qkv.p) + token_elems * 2u * sizeof(std::uint16_t);
    Tensor output(out.p, DType::BF16, {kDim, kHeads, patches});

    std::vector<std::int32_t> cu(static_cast<std::size_t>(opt.segments) + 1u);
    for (std::int32_t segment = 0; segment <= opt.segments; ++segment) {
        cu[static_cast<std::size_t>(segment)] = segment * opt.length;
    }
    DBuf device_cu(cu.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpy(device_cu.p, cu.data(), device_cu.bytes, cudaMemcpyHostToDevice));
    Tensor cu_seqlens(device_cu.p, DType::I32, {static_cast<std::int32_t>(cu.size())});

    const std::int32_t descriptor_tiles =
        ops::vision_attention_scratch_tiles(patches, opt.segments);
    const std::size_t workspace_bytes =
        descriptor_tiles == 0
            ? 1u
            : static_cast<std::size_t>(descriptor_tiles) * 4u * sizeof(std::int32_t) + 255u;
    WorkspaceArena workspace(workspace_bytes);

    const std::int32_t tile =
        opt.descriptor
            ? 64
            : (opt.tile != 0 ? opt.tile : ops::detail::vision_attention_uniform_tile(opt.length));
    const std::int32_t q_tiles_per_segment = (opt.length + tile - 1) / tile;
    const std::int64_t active_ctas =
        static_cast<std::int64_t>(opt.segments) * q_tiles_per_segment * kHeads;
    const std::size_t control_sink_bytes =
        static_cast<std::size_t>(active_ctas) * (tile * 2 / 32) * sizeof(uint4);
    DBuf control_sink(std::max<std::size_t>(control_sink_bytes, 1u));
    auto launch = [&](cudaStream_t stream) {
        if (opt.control) {
            launch_control(q, k, v, opt.length, tile, output, control_sink, stream);
        } else if (opt.descriptor) {
            ops::vision_attention(q, k, v, cu_seqlens, workspace, output, stream);
        } else if (opt.tile != 0) {
            ops::detail::vision_attention_uniform_launch_with_tile(q, k, v, opt.length, opt.tile,
                                                                   output, stream);
        } else {
            ops::vision_attention(q, k, v, opt.length, output, stream);
        }
    };

    const std::int32_t descriptor_query_tiles =
        opt.segments == 1 ? (patches + 63) / 64 : descriptor_tiles;
    const std::int64_t launched_ctas =
        opt.descriptor ? static_cast<std::int64_t>(descriptor_query_tiles) * kHeads : active_ctas;
    const double mathematical_flops =
        4.0 * static_cast<double>(opt.segments) * opt.length * opt.length * kDim * kHeads;
    const double padded_length    = static_cast<double>(q_tiles_per_segment * tile);
    const double issued_mma_flops = 2.0 * static_cast<double>(opt.segments) * padded_length *
                                    padded_length * kHeads * (80 + kDim);
    const double control_payload_bytes =
        4.0 * static_cast<double>(opt.segments) * opt.length * kHeads * kDim +
        4.0 * static_cast<double>(opt.segments) * q_tiles_per_segment * opt.length * kHeads * kDim;

    if (opt.profile_once) {
        launch(nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        const char* kernel_regex = opt.control ? "vision_attention_payload_control_kernel"
                                               : "vision_attention_flash_kernel";
        std::printf("PROFILE_ONCE vision_attention S=%d L=%d P=%d mode=%s tile=%d "
                    "active_ctas=%lld launched_ctas=%lld mathematical_flops=%.0f "
                    "issued_mma_flops=%.0f control_payload_bytes=%.0f kernel_regex='%s'\n",
                    opt.segments, opt.length, patches,
                    opt.control      ? "control"
                    : opt.descriptor ? "descriptor"
                                     : "uniform",
                    tile, static_cast<long long>(active_ctas),
                    static_cast<long long>(launched_ctas), mathematical_flops, issued_mma_flops,
                    control_payload_bytes, kernel_regex);
        return 0;
    }

    const Result result = bench_loop(
        launch, opt.control ? control_payload_bytes : static_cast<double>(plane_elems) * 8.0,
        opt.warmup, opt.repeat, opt.min_time_ms);
    const double seconds = result.median_us * 1.0e-6;
    if (opt.control) {
        const double payload_gbps = control_payload_bytes / seconds / 1.0e9;
        std::printf("vision_attention S=%-3d L=%-6d P=%-6d mode=control    tile=%d "
                    "median=%9.2f us min=%9.2f us p95=%9.2f us "
                    "payload=%8.3f MB payload_rate=%8.2f GB/s active_ctas=%lld runs=%d inner=%d\n",
                    opt.segments, opt.length, patches, tile, result.median_us, result.min_us,
                    result.p95_us, control_payload_bytes / 1.0e6, payload_gbps,
                    static_cast<long long>(active_ctas), result.n_runs, result.inner_iters);
        return 0;
    }
    const double mathematical_tflops = mathematical_flops / seconds / 1.0e12;
    const double issued_mma_tflops   = issued_mma_flops / seconds / 1.0e12;
    std::printf("vision_attention S=%-3d L=%-6d P=%-6d mode=%-10s tile=%d median=%9.2f us "
                "min=%9.2f us p95=%9.2f us math=%7.2f TFLOP/s issued=%7.2f TFLOP/s "
                "useful/issued=%6.2f%% active_ctas=%lld launched_ctas=%lld runs=%d inner=%d\n",
                opt.segments, opt.length, patches,
                opt.control      ? "control"
                : opt.descriptor ? "descriptor"
                                 : "uniform",
                tile, result.median_us, result.min_us, result.p95_us, mathematical_tflops,
                issued_mma_tflops, mathematical_flops / issued_mma_flops * 100.0,
                static_cast<long long>(active_ctas), static_cast<long long>(launched_ctas),
                result.n_runs, result.inner_iters);
    return 0;
}
