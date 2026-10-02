// Route-boundary sweep for the plain `linear` shape tables (schedule_sweep.cuh).
//
// The 2026-09-17 catch-up rewrote every Q4/Q5/Q8 `linear` shape table and moved them out of one
// switch into per-shape files. Those bounds are upstream's sm_120 measurements; none of them had
// ever been swept on sm_86, and `linear` was the one route family with no schedule bench at all.
//
// Unlike the linear_pair / linear_add / linear_swiglu Ops, `linear` has no schedule enum: a shape
// table returns a plain launch pointer. The candidate set here is therefore the union of the
// launches the tables can return at a given quantization, and `routed_to` is recovered by matching
// the shape table's own pointer against that set -- which also catches a table naming a launch the
// bench does not sweep (printed as "(unswept)").
//
// Domains, all of which silently produce a wrong answer rather than throwing:
//   * gemv_* are one-column kernels (T=1 only).
//   * q4 ksplit_cN is masked to its compile-time capacity: handed more columns it computes N of
//     them at a constant cost and looks like it wins everywhere. max_cols = N.
//   * q5 ksplit_cN is an exact-T instantiation, same story.
//   * q8 K-split capacities above 64 do not fit sm_86's 49,152-byte static shared budget with four
//     K warps. The 88-column rung compiles and then faults the device on launch rather than
//     returning an error, so it is not offered here; no shipped table selects above 64 either.

#include "ninfer/ops/linear.h"

#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"
#include "ops/linear/q4/q4_simt_launch.cuh"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q5/q5_ksplit_launch.cuh"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"
#include "ops/linear/q8/q8_shapes.h"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Optional name filter (`--only sub[,sub...]`). A schedule that faults takes the process with it,
// so being able to run one candidate at a time is how a faulting one gets identified.
std::vector<std::string> g_only;

bool selected(const char* name) {
    if (g_only.empty()) { return true; }
    for (const std::string& part : g_only) {
        if (std::string(name).find(part) != std::string::npos) { return true; }
    }
    return false;
}

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
namespace detail = ninfer::ops::detail;

// A named launch plus the largest column count it actually computes.
template <class Launch>
struct Candidate {
    const char* name;
    Launch launch;
    std::int32_t max_cols;
};

// --- Q4 ----------------------------------------------------------------------------------------
//
// Field-for-field the instantiations the shape files compile, so this adds no new kernel image
// beyond the ksplit capacities a given shape does not itself register.

using ninfer::ops::Cache;
using detail::Q4FragmentPipeline;
using detail::Q4RowSplitMmaGemmSchedule;
using detail::Q4RowSplitSimtGemmSchedule;
using detail::Q4ScaleLoad;

using Q4SimtR4C4 = Q4RowSplitSimtGemmSchedule<4, 4, 8, 2, Cache::ca, 1>;
using Q4MmaR16C32 = Q4RowSplitMmaGemmSchedule<16, 32, 64, 16, 8, 2, 2, Q4FragmentPipeline::Serial,
                                              Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using Q4MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                              Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using Q4MmaR32C32Wide =
    Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 8, 4, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;
using Q4MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 2, 2, Q4FragmentPipeline::Serial,
                                              Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using Q4MmaR32C64S3 =
    Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;
using Q4MmaR64C64Tile =
    Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 16, 2, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

template <int K>
using Q4GemvR1W8 =
    detail::Q4RowSplitGemvSchedule<1, 8, 16, 1, detail::Q4GemvActivationAccess::Direct,
                                   detail::Q4GemvLaneMapping::PackedByte2,
                                   detail::Q4GemvDecodeMode::ScalarInteger,
                                   detail::Q4GemvCodeTransfer::SyncVector16,
                                   detail::Q4GemvScaleAccess::Scalar16Shuffle, Cache::ca, K / 64,
                                   1>;

template <int N, int K>
std::vector<Candidate<detail::Q4Launch>> q4_candidates() {
    return {
        {"gemv_r4_w1", detail::launch_q4_gemv_r4_w1_direct, 1},
        {"gemv_r1_w8", detail::launch_q4_gemv<Q4GemvR1W8<K>>, 1},
        {"simt_r4_c4", detail::launch_q4_simt<Q4SimtR4C4, true>, 0},
        {"simt_r8_c4", detail::launch_q4_simt_r8_c4, 0},
        {"simt_r8_c8", detail::launch_q4_simt_r8_c8, 0},
        {"ksplit_c4", detail::launch_q4_ksplit<N, K, 4>, 4},
        {"ksplit_c8", detail::launch_q4_ksplit<N, K, 8>, 8},
        {"ksplit_c16", detail::launch_q4_ksplit<N, K, 16>, 16},
        {"ksplit_c24", detail::launch_q4_ksplit<N, K, 24>, 24},
        {"ksplit_c32", detail::launch_q4_ksplit<N, K, 32>, 32},
        {"mma_r16_c32", detail::launch_q4_mma<Q4MmaR16C32>, 0},
        {"mma_r32_c32", detail::launch_q4_mma<Q4MmaR32C32>, 0},
        {"mma_r32_c32w", detail::launch_q4_mma<Q4MmaR32C32Wide>, 0},
        {"mma_r32_c64", detail::launch_q4_mma<Q4MmaR32C64>, 0},
        {"mma_r32_c64s3", detail::launch_q4_mma<Q4MmaR32C64S3>, 0},
        {"mma_r64_c32", detail::launch_q4_mma_r64_c32, 0},
        {"mma_r64_c48", detail::launch_q4_mma_r64_c48, 0},
        {"mma_r64_c56", detail::launch_q4_mma_r64_c56, 0},
        {"mma_r64_c64t", detail::launch_q4_mma<Q4MmaR64C64Tile>, 0},
        {"mma_r64_c72", detail::launch_q4_mma_r64_c72, 0},
        {"mma_r64_c80", detail::launch_q4_mma_r64_c80, 0},
        {"mma_r64_c96", detail::launch_q4_mma_r64_c96, 0},
        {"mma_r64_c112", detail::launch_q4_mma_r64_c112, 0},
        {"mma_r64_c120", detail::launch_q4_mma_r64_c120, 0},
        {"mma_r64_c128", detail::launch_q4_mma_r64_c128, 0},
    };
}

// --- Q5 ----------------------------------------------------------------------------------------

template <int K, int KWarps>
std::vector<Candidate<detail::Q5Launch>> q5_candidates() {
    return {
        {"split4_c1", K == 5120    ? detail::launch_q5_split4_c1_k5120
                      : K == 6144  ? detail::launch_q5_split4_c1_k6144
                                   : detail::launch_q5_split4_c1_k17408,
         1},
        {"ksplit_t2", detail::launch_q5_ksplit<K, 2, KWarps>, 2},
        {"ksplit_t3", detail::launch_q5_ksplit<K, 3, KWarps>, 3},
        {"ksplit_t4", detail::launch_q5_ksplit<K, 4, KWarps>, 4},
        {"ksplit_t5", detail::launch_q5_ksplit<K, 5, KWarps>, 5},
        {"ksplit_t6", detail::launch_q5_ksplit<K, 6, KWarps>, 6},
        {"ksplit_t8", detail::launch_q5_ksplit<K, 8, KWarps>, 8},
        {"ksplit_t12", detail::launch_q5_ksplit<K, 12, KWarps>, 12},
        {"simt_r8_c4", detail::launch_q5_simt_r8_c4, 0},
        {"simt_r8_c8", detail::launch_q5_simt_r8_c8, 0},
        {"mma_r64_c16", detail::launch_q5_mma_r64_c16, 0},
        {"mma_r64_c32s3", detail::launch_q5_mma_r64_c32_s3, 0},
        {"mma_r64_c64", detail::launch_q5_mma_r64_c64, 0},
        {"mma_r32_c128", detail::launch_q5_mma_r32_c128, 0},
        {"mma_r64_c128", detail::launch_q5_mma_r64_c128, 0},
    };
}

// --- Q8 ----------------------------------------------------------------------------------------
//
// Every Q8 shape table states its K-split ladder as (capacity, K warps, scale access, activation
// cache, weight cache, staging). The catch-up retuned all six fields per rung on sm_120, including
// the activation cache policy it changed wholesale in 028eb61e. Sweeping a whole cross product is
// not affordable, so each capacity is timed in the three combinations the shipped tables actually
// choose between on this card: shared scales with `ca` or `cg` activations, and -- at the
// capacities where staging choice is live -- runtime-active staging. Under NINFER_SM8X_COMPAT the
// sixteen- and eight-warp rungs the 5090 tables ask for do not fit 49,152 bytes of static shared
// memory, which is why four K warps is the baseline here and eight is swept only to 32 columns.

using detail::Q8KSplitActivationStage;
using detail::Q8KSplitScaleAccess;
using detail::Q8KSplitSchedule;

template <int Cap, Cache AC, Q8KSplitActivationStage S = Q8KSplitActivationStage::ActiveOnly,
          int KWarps = 4>
using Q8K = Q8KSplitSchedule<KWarps, Cap, 2, Q8KSplitScaleAccess::Shared, AC, Cache::cg, S>;

template <class Geometry>
std::vector<Candidate<detail::Q8Launch>> q8_candidates() {
    constexpr auto kRuntime = Q8KSplitActivationStage::RuntimeActive;
    return {
        {"simt_r8_c4", detail::launch_q8_simt_r8_c4, 0},
        {"simt_r8_c8", detail::launch_q8_simt_r8_c8, 0},
        {"k8_ca", detail::launch_q8_ksplit<Geometry, 8, Q8K<8, Cache::ca>>, 8},
        {"k8_cg", detail::launch_q8_ksplit<Geometry, 8, Q8K<8, Cache::cg>>, 8},
        {"k8_ca_rt", detail::launch_q8_ksplit<Geometry, 8, Q8K<8, Cache::ca, kRuntime>>, 8},
        {"k8_ca_w8", detail::launch_q8_ksplit<Geometry, 8, Q8K<8, Cache::ca, kRuntime, 8>>, 8},
        {"k16_ca", detail::launch_q8_ksplit<Geometry, 16, Q8K<16, Cache::ca>>, 16},
        {"k16_cg", detail::launch_q8_ksplit<Geometry, 16, Q8K<16, Cache::cg>>, 16},
        {"k16_ca_rt", detail::launch_q8_ksplit<Geometry, 16, Q8K<16, Cache::ca, kRuntime>>, 16},
        {"k16_ca_w8", detail::launch_q8_ksplit<Geometry, 16, Q8K<16, Cache::ca, kRuntime, 8>>, 16},
        {"k24_ca", detail::launch_q8_ksplit<Geometry, 24, Q8K<24, Cache::ca>>, 24},
        {"k24_cg", detail::launch_q8_ksplit<Geometry, 24, Q8K<24, Cache::cg>>, 24},
        {"k32_ca", detail::launch_q8_ksplit<Geometry, 32, Q8K<32, Cache::ca>>, 32},
        {"k32_cg", detail::launch_q8_ksplit<Geometry, 32, Q8K<32, Cache::cg>>, 32},
        {"k32_ca_rt", detail::launch_q8_ksplit<Geometry, 32, Q8K<32, Cache::ca, kRuntime>>, 32},
        {"k40_ca", detail::launch_q8_ksplit<Geometry, 40, Q8K<40, Cache::ca>>, 40},
        {"k40_cg", detail::launch_q8_ksplit<Geometry, 40, Q8K<40, Cache::cg>>, 40},
        {"k48_ca", detail::launch_q8_ksplit<Geometry, 48, Q8K<48, Cache::ca>>, 48},
        {"k48_cg", detail::launch_q8_ksplit<Geometry, 48, Q8K<48, Cache::cg>>, 48},
        {"k48_ca_rt", detail::launch_q8_ksplit<Geometry, 48, Q8K<48, Cache::ca, kRuntime>>, 48},
        {"k56_ca", detail::launch_q8_ksplit<Geometry, 56, Q8K<56, Cache::ca>>, 56},
        {"k64_ca", detail::launch_q8_ksplit<Geometry, 64, Q8K<64, Cache::ca>>, 64},
        {"k64_cg", detail::launch_q8_ksplit<Geometry, 64, Q8K<64, Cache::cg>>, 64},
        {"mma_r32_c64", detail::launch_q8_mma_r32_c64, 0},
        {"mma_r32_c96", detail::launch_q8_mma_r32_c96, 0},
        {"mma_r32_c128", detail::launch_q8_mma_r32_c128, 0},
        {"mma_r48_c64", detail::launch_q8_mma_r48_c64, 0},
        {"mma_r64_c96", detail::launch_q8_mma_r64_c96, 0},
        {"mma_r64_c128", detail::launch_q8_mma_r64_c128, 0},
        {"mma_r96_c96", detail::launch_q8_mma_r96_c96, 0},
        {"mma_r128_c64", detail::launch_q8_mma_r128_c64, 0},
        {"mma_r128_c80", detail::launch_q8_mma_r128_c80, 0},
    };
}

// --- the sweep ---------------------------------------------------------------------------------

template <class Launch>
void run(QType qtype, std::int32_t n, std::int32_t k,
         const std::vector<Candidate<Launch>>& candidates, Launch (*table)(std::int32_t),
         const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight packed =
        ninfer::bench::make_row_split_weight(qtype, n, k, k, {0x31, 0xa5, 0x3c00});
    ninfer::DeviceBuffer input(static_cast<std::size_t>(k) * max_tokens * 2);
    ninfer::DeviceBuffer output(static_cast<std::size_t>(n) * max_tokens * 2);

    std::vector<ninfer::bench::SweepEntry> schedules;
    schedules.reserve(candidates.size());
    for (const Candidate<Launch>& candidate : candidates) {
        if (!selected(candidate.name)) { continue; }
        const Launch launch = candidate.launch;
        schedules.push_back({candidate.name,
                             [&, launch](std::int32_t tokens, cudaStream_t stream) {
                                 Tensor x(input.p, DType::BF16, {k, tokens});
                                 Tensor out(output.p, DType::BF16, {n, tokens});
                                 launch(x, packed.weight, out, stream);
                             },
                             candidate.max_cols});
    }

    const std::string title = std::string(qtype == QType::Q4_G64_FP16   ? "q4"
                                          : qtype == QType::Q5_G64_FP16 ? "q5"
                                          : qtype == QType::Q6_G64_FP16 ? "q6"
                                                                        : "q8") +
                              " linear n=" + std::to_string(n) + " k=" + std::to_string(k);
    ninfer::bench::SweepOptions options = base;
    options.title                       = title.c_str();
    options.routed_name = [&candidates, table](std::int32_t tokens) -> const char* {
        Launch chosen = nullptr;
        try {
            chosen = table(tokens);
        } catch (const std::exception&) { return "(throws)"; }
        for (const Candidate<Launch>& candidate : candidates) {
            if (candidate.launch == chosen) { return candidate.name; }
        }
        return "(unswept)";
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor out(output.p, DType::BF16, {n, tokens});
        ninfer::ops::linear(x, packed.weight, out, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

struct Shape {
    const char* key;
    void (*sweep)(const ninfer::bench::SweepOptions&);
};

template <int N, int K>
void sweep_q4(const ninfer::bench::SweepOptions& base) {
    run<detail::Q4Launch>(QType::Q4_G64_FP16, N, K, q4_candidates<N, K>(),
                          +[](std::int32_t t) { return detail::select_q4_a16_launch(N, K, t); },
                          base);
}

template <int N, int K, int KWarps>
void sweep_q5(const ninfer::bench::SweepOptions& base) {
    run<detail::Q5Launch>(QType::Q5_G64_FP16, N, K, q5_candidates<K, KWarps>(),
                          +[](std::int32_t t) { return detail::select_q5_a16_launch(N, K, t); },
                          base);
}

// --- Q6 ----------------------------------------------------------------------------------------
//
// The per-row GEMV is exact-T (one instantiation per token count); one candidate dispatches on T
// so the sweep can offer it across its whole 1..4 domain.

void q6_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1: detail::launch_q6_gemv_t1(x, w, out, stream); return;
    case 2: detail::launch_q6_gemv_t2(x, w, out, stream); return;
    case 3: detail::launch_q6_gemv_t3(x, w, out, stream); return;
    default: detail::launch_q6_gemv_t4(x, w, out, stream); return;
    }
}

std::vector<Candidate<detail::Q6Launch>> q6_candidates() {
    return {
        {"gemv", q6_gemv, 4},
        {"simt_r8_c4", detail::launch_q6_simt_r8_c4, 0},
        {"mma_r64_c16_k128", detail::launch_q6_mma_r64_c16_k128, 0},
        {"mma_r64_c32_k128", detail::launch_q6_mma_r64_c32_k128, 0},
        {"mma_r64_c128", detail::launch_q6_mma_r64_c128, 0},
    };
}

template <int N, int K>
void sweep_q6(const ninfer::bench::SweepOptions& base) {
    run<detail::Q6Launch>(QType::Q6_G64_FP16, N, K, q6_candidates(),
                          +[](std::int32_t t) { return detail::select_q6_a16_launch(N, K, t); },
                          base);
}

template <class Geometry>
void sweep_q8(const ninfer::bench::SweepOptions& base) {
    run<detail::Q8Launch>(
        QType::Q8_G32_FP16, Geometry::kOutputRows, Geometry::kInputRows, q8_candidates<Geometry>(),
        +[](std::int32_t t) {
            return detail::select_q8_a16_launch(Geometry::kOutputRows, Geometry::kInputRows, t);
        },
        base);
}

const Shape kShapes[] = {
    {"q4:1024x5120", sweep_q4<1024, 5120>},
    {"q4:4096x5120", sweep_q4<4096, 5120>},
    {"q4:6144x5120", sweep_q4<6144, 5120>},
    {"q4:7168x5120", sweep_q4<7168, 5120>},
    {"q4:34816x5120", sweep_q4<34816, 5120>},
    {"q4:5120x6144", sweep_q4<5120, 6144>},
    {"q4:131072x5120", sweep_q4<131072, 5120>},
    {"q5:1024x5120", sweep_q5<1024, 5120, 4>},
    {"q5:6144x5120", sweep_q5<6144, 5120, 4>},
    {"q5:7168x5120", sweep_q5<7168, 5120, 4>},
    {"q5:5120x6144", sweep_q5<5120, 6144, 2>},
    {"q5:5120x17408", sweep_q5<5120, 17408, 2>},
    {"q6:248320x5120", sweep_q6<248320, 5120>},
    {"q8:5120x25600", sweep_q8<detail::Q8N5120K25600>},
    {"q8:2048x16384", sweep_q8<detail::Q8N2048K16384>},
    {"q8:5120x6144", sweep_q8<detail::Q8N5120K6144>},
    {"q8:5120x17408", sweep_q8<detail::Q8N5120K17408>},
    {"q8:6144x5120", sweep_q8<detail::Q8N6144K5120>},
    {"q8:34816x5120", sweep_q8<detail::Q8N34816K5120>},
    {"q8:14336x5120", sweep_q8<detail::Q8N14336K5120>},
    {"q8:5120x10240", sweep_q8<detail::Q8N5120K10240>},
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <shape> [--tokens T,...] [--repeat N] [--warmup N]"
                             " [--spread]\nshapes:\n", argv[0]);
        for (const Shape& shape : kShapes) { std::fprintf(stderr, "  %s\n", shape.key); }
        return 2;
    }
    const std::string key(argv[1]);
    --argc;
    ++argv;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != "--only") { continue; }
        const std::string value(argv[i + 1]);
        std::size_t start = 0;
        while (start <= value.size()) {
            const std::size_t comma = value.find(',', start);
            const std::string item  = value.substr(start, comma - start);
            if (!item.empty()) { g_only.push_back(item); }
            if (comma == std::string::npos) { break; }
            start = comma + 1;
        }
        for (int j = i; j + 2 <= argc; ++j) { argv[j] = argv[j + 2]; }
        argc -= 2;
        break;
    }

    ninfer::bench::SweepOptions options;
    options.tokens = {1,  2,  4,  8,  12,  16,  20,  24,  32,  40,  48,  56,  64, 80,
                      96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 640, 1024};
    if (!ninfer::bench::parse_sweep_args(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s <shape> [--tokens T,...] [--repeat N] [--warmup N]"
                             " [--spread]\n", argv[0]);
        return 2;
    }
    for (const Shape& shape : kShapes) {
        if (key == shape.key) {
            shape.sweep(options);
            return 0;
        }
    }
    std::fprintf(stderr, "unknown shape '%s'\n", key.c_str());
    return 2;
}
