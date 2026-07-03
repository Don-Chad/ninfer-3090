#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/core/weight_store_parser.h"
#include "qus/model/config.h"
#include "qus/model/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

struct Options {
    std::string weights = "out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus";
    int device          = 0;
    int warmup          = 1;
    int reps            = 3;
    bool parity         = false;
};

void usage(const char* argv0) {
    std::cout << "usage: " << argv0 << " [--weights <path>] [--device <id>] [--warmup <n>]"
              << " [--reps <n>] [--parity]\n";
}

Options parse_args(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(name) + " needs value"); }
            return argv[++i];
        };
        if (arg == "--weights") {
            out.weights = need_value("--weights");
        } else if (arg == "--device") {
            out.device = std::stoi(need_value("--device"));
        } else if (arg == "--warmup") {
            out.warmup = std::stoi(need_value("--warmup"));
        } else if (arg == "--reps") {
            out.reps = std::stoi(need_value("--reps"));
        } else if (arg == "--parity") {
            out.parity = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "qus_target_verify_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (out.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (out.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (out.reps <= 0) { throw std::invalid_argument("--reps must be positive"); }
    return out;
}

qus::Q5090Expectations expectations() {
    qus::Q5090Expectations expected;
    expected.layer_count             = qus::model::kCfg.n_layers;
    expected.hidden_size             = qus::model::kCfg.hidden;
    expected.intermediate_size       = qus::model::kCfg.intermediate;
    expected.vocab_size              = qus::model::kCfg.vocab;
    expected.num_attention_heads     = qus::model::kCfg.n_q;
    expected.num_key_value_heads     = qus::model::kCfg.n_kv;
    expected.head_dim                = qus::model::kCfg.head_dim;
    expected.gdn_key_heads           = qus::model::kCfg.gdn_k_heads;
    expected.gdn_value_heads         = qus::model::kCfg.gdn_v_heads;
    expected.gdn_key_head_dim        = qus::model::kCfg.gdn_k_dim;
    expected.gdn_value_head_dim      = qus::model::kCfg.gdn_v_dim;
    expected.gdn_conv_width          = qus::model::kCfg.gdn_conv_k;
    expected.full_attention_interval = qus::model::kCfg.full_interval;
    expected.max_position_embeddings = 262144;
    return expected;
}

qus::model::StepState make_step_state(qus::DeviceArena& arena, int window_cols,
                                      int prefill_chunk) {
    const int draft_cols = std::max(1, window_cols - 1);
    return qus::model::StepState{
        arena.alloc(qus::DType::I32, {1}),
        arena.alloc(qus::DType::I32, {1}),
        arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab, window_cols}),
        arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, window_cols}),
        arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, prefill_chunk}),
        arena.alloc(qus::DType::I32, {window_cols}),
        arena.alloc(qus::DType::I32, {draft_cols}),
        arena.alloc(qus::DType::I32, {window_cols}),
        arena.alloc(qus::DType::I32, {1}),
        arena.alloc(qus::DType::I32, {window_cols}),
        arena.alloc(qus::DType::I32, {window_cols}),
        arena.alloc(qus::DType::I32, {window_cols}),
        arena.alloc(qus::DType::I32, {1}),
        arena.alloc(qus::DType::I32, {1}),
        arena.alloc(qus::DType::I32, {1}),
        arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, 1}),
        arena.alloc(qus::DType::I64, {qus::model::kStepStatsCounters}),
    };
}

void copy_i32_scalar(int value, qus::Tensor& dst, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst.data, &value, sizeof(value), cudaMemcpyHostToDevice, stream));
}

void copy_i32_vector(const std::vector<int>& value, qus::Tensor& dst, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst.data, value.data(), value.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
}

void copy_bf16_column(const qus::Tensor& src, qus::Tensor& dst, int dst_col,
                      cudaStream_t stream) {
    const auto bytes = static_cast<std::size_t>(src.ne[0]) * sizeof(std::uint16_t);
    auto* dst_ptr    = static_cast<unsigned char*>(dst.data) +
                    static_cast<std::size_t>(dst_col) * bytes;
    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src.data, bytes, cudaMemcpyDeviceToDevice, stream));
}

void copy_i32_column(const qus::Tensor& src, qus::Tensor& dst, int dst_col,
                     cudaStream_t stream) {
    auto* dst_ptr = static_cast<unsigned char*>(dst.data) +
                    static_cast<std::size_t>(dst_col) * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src.data, sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
                               stream));
}

void copy_device_tensor(const qus::Tensor& src, const qus::Tensor& dst, cudaStream_t stream) {
    if (src.bytes() != dst.bytes()) { throw std::runtime_error("device copy size mismatch"); }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src.data, src.bytes(), cudaMemcpyDeviceToDevice, stream));
}

float bf16_to_f32(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float out             = 0.0f;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

std::vector<std::uint16_t> copy_bf16_to_host(const qus::Tensor& t) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, out.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    return out;
}

std::vector<std::int32_t> copy_i32_to_host(const qus::Tensor& t) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, out.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return out;
}

double max_abs_bf16_diff(const qus::Tensor& a, const qus::Tensor& b) {
    const std::vector<std::uint16_t> ah = copy_bf16_to_host(a);
    const std::vector<std::uint16_t> bh = copy_bf16_to_host(b);
    if (ah.size() != bh.size()) { throw std::runtime_error("parity tensor size mismatch"); }
    double max_abs = 0.0;
    for (std::size_t i = 0; i < ah.size(); ++i) {
        const double diff =
            std::abs(static_cast<double>(bf16_to_f32(ah[i])) - static_cast<double>(bf16_to_f32(bh[i])));
        max_abs = std::max(max_abs, diff);
    }
    return max_abs;
}

double mean_ms(const std::vector<float>& ms) {
    return std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
}

struct HiddenCaptureTap {
    static constexpr bool enabled = true;

    qus::Tensor* hidden = nullptr;
    int col             = 0;

    void operator()(qus::model::TapId id, int layer, qus::model::Phase, const qus::Tensor& x,
                    cudaStream_t stream) {
        if (id == qus::model::TapId::AfterFinalNorm && layer == -1) {
            copy_bf16_column(x, *hidden, col, stream);
        }
    }
};

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "target_verify_bench: " << e.what() << '\n';
        return 2;
    }

    const std::filesystem::path weights_path(options.weights);
    if (!std::filesystem::exists(weights_path)) {
        std::cout << "SKIP: weights file not present: " << weights_path << '\n';
        return 0;
    }

    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        qus::DeviceContext ctx(options.device);
        constexpr int kMaxVerifyT     = 6;
        constexpr int kMaxContext     = 128;
        constexpr int kPrefillChunk   = 128;
        const std::size_t weight_bytes =
            std::filesystem::file_size(weights_path) + 256ULL * kMiB + 451267584ULL;
        qus::DeviceArena weight_arena(weight_bytes);
        qus::DeviceArena cache_arena(3ULL * kGiB);
        qus::WorkspaceArena workspace(3ULL * kGiB);
        qus::WeightStore weights(expectations());
        qus::LoadOptions load_options;
        load_options.load_mtp = false;
        weights.load(weights_path.c_str(), weight_arena, ctx, load_options);

        qus::KVCache kv(cache_arena, qus::model::kCfg.n_full(), kMaxContext,
                        qus::model::kCfg.n_kv, qus::model::kCfg.head_dim);
        qus::GdnState state(cache_arena, qus::model::kCfg.n_gdn(), qus::model::kCfg.conv_dim,
                            qus::model::kCfg.gdn_conv_state_width, qus::model::kCfg.gdn_v_heads,
                            qus::model::kCfg.gdn_v_dim, qus::model::kCfg.gdn_k_dim, kMaxVerifyT);
        qus::model::StepState io = make_step_state(cache_arena, kMaxVerifyT, kPrefillChunk);
        qus::model::Qwen3_6_27B card(ctx, weights, workspace, kv, state, io, kPrefillChunk);

        qus::Tensor ids       = cache_arena.alloc(qus::DType::I32, {kMaxVerifyT});
        qus::Tensor positions = cache_arena.alloc(qus::DType::I32, {kMaxVerifyT});
        qus::Tensor seq_hidden =
            cache_arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, kMaxVerifyT});
        qus::Tensor seq_logits =
            cache_arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab, kMaxVerifyT});
        qus::Tensor seq_tokens = cache_arena.alloc(qus::DType::I32, {kMaxVerifyT});
        qus::Tensor verify_hidden_copy =
            cache_arena.alloc(qus::DType::BF16, {qus::model::kCfg.hidden, kMaxVerifyT});
        qus::Tensor verify_logits_copy =
            cache_arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab, kMaxVerifyT});
        qus::Tensor verify_tokens_copy = cache_arena.alloc(qus::DType::I32, {kMaxVerifyT});
        copy_i32_vector({1, 2, 3, 4, 5, 6}, ids, ctx.stream);
        copy_i32_vector({0, 1, 2, 3, 4, 5}, positions, ctx.stream);
        ctx.synchronize();

        auto reset_decode_state = [&] {
            kv.reset();
            state.reset(ctx.stream);
            workspace.reset();
            copy_i32_scalar(1, io.token, ctx.stream);
            copy_i32_scalar(0, io.pos, ctx.stream);
            ctx.synchronize();
        };
        auto reset_verify_state = [&] {
            kv.reset();
            state.reset(ctx.stream);
            workspace.reset();
            ctx.synchronize();
        };
        auto reset_to_prefix = [&](int prefix) {
            kv.reset();
            state.reset(ctx.stream);
            workspace.reset();
            if (prefix > 0) {
                std::vector<int> prefix_ids(static_cast<std::size_t>(prefix));
                for (int i = 0; i < prefix; ++i) { prefix_ids[static_cast<std::size_t>(i)] = 11 + i; }
                card.prefill(prefix_ids);
            } else {
                copy_i32_scalar(1, io.token, ctx.stream);
                copy_i32_scalar(0, io.pos, ctx.stream);
                ctx.synchronize();
            }
        };

        auto time_decode = [&] {
            reset_decode_state();
            qus::CudaEventTimer timer(ctx);
            timer.start();
            card.decode_step();
            return timer.stop_ms();
        };
        auto time_verify = [&](int T) {
            reset_verify_state();
            qus::Tensor ids_t       = ids.slice(0, 0, T);
            qus::Tensor positions_t = positions.slice(0, 0, T);
            qus::CudaEventTimer timer(ctx);
            timer.start();
            card.target_verify(ids_t, positions_t, 0);
            return timer.stop_ms();
        };

        for (int i = 0; i < options.warmup; ++i) {
            (void)time_decode();
            for (int T = 2; T <= kMaxVerifyT; ++T) { (void)time_verify(T); }
        }

        std::vector<float> decode_ms;
        decode_ms.reserve(static_cast<std::size_t>(options.reps));
        for (int i = 0; i < options.reps; ++i) { decode_ms.push_back(time_decode()); }
        const double decode_mean = mean_ms(decode_ms);

        std::cout << "target_verify_bench\n";
        std::cout << "weights," << weights_path << '\n';
        std::cout << "device," << ctx.props.name << '\n';
        std::cout << "warmup," << options.warmup << '\n';
        std::cout << "reps," << options.reps << '\n';
        std::cout << "decode_t1_ms," << decode_mean << '\n';
        std::cout << "T,verify_ms,ratio_vs_decode,epsilon\n";
        for (int T = 2; T <= kMaxVerifyT; ++T) {
            std::vector<float> verify_ms;
            verify_ms.reserve(static_cast<std::size_t>(options.reps));
            for (int i = 0; i < options.reps; ++i) { verify_ms.push_back(time_verify(T)); }
            const double verify_mean = mean_ms(verify_ms);
            const double ratio       = verify_mean / decode_mean;
            std::cout << T << ',' << verify_mean << ',' << ratio << ',' << (ratio - 1.0) << '\n';
        }

        if (options.parity) {
            std::cout << "parity\n";
            std::cout << "prefix,T,hidden_max_abs,logits_max_abs,argmax_mismatches\n";
            for (const int prefix : {0, 4}) {
                std::vector<int> absolute_positions(kMaxVerifyT);
                for (int i = 0; i < kMaxVerifyT; ++i) {
                    absolute_positions[static_cast<std::size_t>(i)] = prefix + i;
                }
                copy_i32_vector(absolute_positions, positions, ctx.stream);
                ctx.synchronize();

                for (int T = 1; T <= kMaxVerifyT; ++T) {
                    reset_to_prefix(prefix);
                    qus::Tensor ids_t       = ids.slice(0, 0, T);
                    qus::Tensor positions_t = positions.slice(0, 0, T);
                    card.target_verify(ids_t, positions_t, static_cast<std::uint32_t>(prefix));
                    copy_device_tensor(io.verify_hidden.slice(1, 0, T),
                                       verify_hidden_copy.slice(1, 0, T), ctx.stream);
                    copy_device_tensor(io.logits.slice(1, 0, T),
                                       verify_logits_copy.slice(1, 0, T), ctx.stream);
                    copy_device_tensor(io.target_tokens.slice(0, 0, T),
                                       verify_tokens_copy.slice(0, 0, T), ctx.stream);
                    ctx.synchronize();

                    reset_to_prefix(prefix);
                    for (int col = 0; col < T; ++col) {
                        copy_i32_scalar(col + 1, io.token, ctx.stream);
                        copy_i32_scalar(prefix + col, io.pos, ctx.stream);
                        ctx.synchronize();
                        HiddenCaptureTap tap{&seq_hidden, col};
                        card.decode_step(tap);
                        copy_bf16_column(io.logits, seq_logits, col, ctx.stream);
                        copy_i32_column(io.token, seq_tokens, col, ctx.stream);
                        ctx.synchronize();
                    }

                    const qus::Tensor verify_hidden = verify_hidden_copy.slice(1, 0, T);
                    const qus::Tensor verify_logits = verify_logits_copy.slice(1, 0, T);
                    const qus::Tensor replay_hidden = seq_hidden.slice(1, 0, T);
                    const qus::Tensor replay_logits = seq_logits.slice(1, 0, T);
                    const double hidden_diff = max_abs_bf16_diff(verify_hidden, replay_hidden);
                    const double logits_diff = max_abs_bf16_diff(verify_logits, replay_logits);
                    const std::vector<std::int32_t> verify_tokens =
                        copy_i32_to_host(verify_tokens_copy.slice(0, 0, T));
                    const std::vector<std::int32_t> replay_tokens =
                        copy_i32_to_host(seq_tokens.slice(0, 0, T));
                    int mismatches = 0;
                    for (int i = 0; i < T; ++i) {
                        if (verify_tokens[static_cast<std::size_t>(i)] !=
                            replay_tokens[static_cast<std::size_t>(i)]) {
                            ++mismatches;
                        }
                    }
                    std::cout << prefix << ',' << T << ',' << hidden_diff << ',' << logits_diff
                              << ',' << mismatches << '\n';
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "target_verify_bench: " << e.what() << '\n';
        return 1;
    }
}
