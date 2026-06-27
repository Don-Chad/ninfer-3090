#include "qus/core/arena.h"
#include "qus/core/device.h"
#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"
#include "qus/core/weight_store.h"
#include "qus/model/config.h"
#include "qus/model/model.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

int parse_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <weights.qus> <out-dir> <target-generated-index> <prompt ids...>\n";
        return 2;
    }

    try {
        const std::filesystem::path weights_path = argv[1];
        const std::filesystem::path out_dir      = argv[2];
        const int target                         = parse_int(argv[3], "target-generated-index");
        std::vector<int> prompt;
        for (int i = 4; i < argc; ++i) { prompt.push_back(parse_int(argv[i], "token-id")); }
        if (prompt.empty()) { throw std::invalid_argument("prompt must not be empty"); }
        std::filesystem::create_directories(out_dir);

        qus::DeviceContext ctx(0);
        const std::size_t weight_bytes =
            static_cast<std::size_t>(std::filesystem::file_size(weights_path)) +
            256ULL * 1024ULL * 1024ULL;
        qus::DeviceArena weight_arena(weight_bytes);
        qus::DeviceArena cache_arena(1024ULL * 1024ULL * 1024ULL);
        qus::WorkspaceArena work(2ULL * 1024ULL * 1024ULL * 1024ULL);
        qus::WeightStore store(expectations());
        store.load(weights_path.string().c_str(), weight_arena, ctx);

        const auto max_ctx =
            static_cast<std::uint32_t>(prompt.size() + static_cast<std::size_t>(target) + 1);
        qus::KVCache kv(cache_arena, qus::model::kCfg.n_full(), max_ctx, qus::model::kCfg.n_kv,
                        qus::model::kCfg.head_dim);
        qus::GdnState state(cache_arena, qus::model::kCfg.n_gdn(), qus::model::kCfg.conv_dim,
                            qus::model::kCfg.gdn_conv_state_width, qus::model::kCfg.gdn_v_heads,
                            qus::model::kCfg.gdn_v_dim, qus::model::kCfg.gdn_k_dim);
        qus::model::StepState io{cache_arena.alloc(qus::DType::I32, {1}),
                                 cache_arena.alloc(qus::DType::I32, {1}),
                                 cache_arena.alloc(qus::DType::BF16, {qus::model::kCfg.vocab, 1})};
        qus::model::Qwen3_6_27B card(ctx, store, work, kv, state, io);
        qus::model::FileTap tap(out_dir);

        kv.reset();
        state.reset(ctx.stream);
        if (target == 0) {
            card.prefill(prompt, tap);
        } else {
            card.prefill(prompt);
            for (int step = 1; step < target; ++step) { card.decode_step(); }
            card.decode_step(tap);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
