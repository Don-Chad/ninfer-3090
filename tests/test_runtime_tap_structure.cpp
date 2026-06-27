#include "qus/model/model.h"
#include "qus/runtime/engine.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string read_file(const char* path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error(std::string("failed to open ") + path); }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

int expect_present(const std::string& text, const char* needle, const char* message) {
    if (contains(text, needle)) { return 0; }
    std::cerr << message << ": missing `" << needle << "`\n";
    return 1;
}

struct MemoryTap {
    static constexpr bool enabled = true;

    void operator()(qus::model::TapId, int, qus::model::Phase, const qus::Tensor&, cudaStream_t) {}
};

} // namespace

static_assert(!qus::model::NullTap::enabled);
static_assert(qus::model::FileTap::enabled);
static_assert(requires(qus::model::Qwen3_6_27B& card, std::span<const int> ids, MemoryTap& tap) {
    card.prefill(ids, tap);
    card.decode_step(tap);
});

int main() {
    int failures = 0;

    const qus::EngineOptions positional{7, 4096};
    if (positional.device != 7 || positional.max_ctx != 4096 || positional.eos_token_id != -1) {
        std::cerr << "EngineOptions positional aggregate compatibility broke\n";
        ++failures;
    }

    qus::EngineOptions eos_options;
    eos_options.eos_token_id = 123;
    if (eos_options.eos_token_id != 123) {
        std::cerr << "EngineOptions eos_token_id setter path broke\n";
        ++failures;
    }

    try {
        qus::EngineOptions bad_options;
        bad_options.eos_token_id = -2;
        qus::Engine bad_engine(bad_options);
        (void)bad_engine;
        std::cerr << "Engine accepted an invalid eos_token_id\n";
        ++failures;
    } catch (const std::invalid_argument&) {}

    const std::string engine_h = read_file(QUS_SOURCE_DIR "/include/qus/runtime/engine.h");
    failures += expect_present(engine_h, "bool is_eos_token(int token) const noexcept",
                               "Engine must centralize EOS checks");

    const std::string engine = read_file(QUS_SOURCE_DIR "/src/runtime/engine.cpp");
    failures += expect_present(engine, "if (is_eos_token(token)) { return out; }",
                               "generate must stop after a prefill EOS token");
    failures += expect_present(engine, "if (is_eos_token(token)) { break; }",
                               "generate must stop after a decode EOS token");

    const std::string model_h = read_file(QUS_SOURCE_DIR "/include/qus/model/model.h");
    failures += expect_present(model_h, "class FileTap", "FileTap must be part of the parity API");
    failures +=
        expect_present(model_h, "template <class Tap>", "tap-enabled drivers must be generic");
    failures += expect_present(model_h, "prefill_erased(ids, &tap",
                               "generic prefill must route through the erased tap trampoline");
    failures += expect_present(model_h, "decode_step_erased(&tap",
                               "generic decode must route through the erased tap trampoline");

    const std::string model = read_file(QUS_SOURCE_DIR "/src/model/qwen3_6_27b.cpp");
    failures += expect_present(model, "template <class Tap>",
                               "tap hooks must be wired through template helpers");
    failures +=
        expect_present(model, "if constexpr (Tap::enabled)", "NullTap hooks must compile out");
    failures += expect_present(model, "failed to write", "FileTap must report short writes");
    failures += expect_present(model, "TapId::AfterEmbed", "tap hook must cover embeddings");
    failures += expect_present(model, "TapId::AfterMixer", "tap hook must cover mixers");
    failures += expect_present(model, "TapId::AfterMlp", "tap hook must cover MLP outputs");
    failures += expect_present(model, "TapId::AfterFinalNorm", "tap hook must cover final norm");
    failures += expect_present(model, "TapId::AfterLogits", "tap hook must cover logits");
    failures +=
        expect_present(model, "layer_%02d.f32", "FileTap must keep legacy layer dump files");

    const std::string layer_dump = read_file(QUS_SOURCE_DIR "/tools/parity/layer_dump.cpp");
    failures += expect_present(layer_dump, "qus::model::FileTap tap(out_dir)",
                               "layer dump must use FileTap");
    failures += expect_present(layer_dump, "card.prefill(prompt, tap)",
                               "layer dump must exercise tap-enabled prefill");
    failures += expect_present(layer_dump, "card.decode_step(tap)",
                               "layer dump must exercise tap-enabled decode");

    if (failures != 0) {
        std::cerr << "FAIL runtime EOS/FileTap structural invariants\n";
        return 1;
    }

    std::cout << "OK runtime EOS/FileTap structural invariants\n";
    return 0;
}
