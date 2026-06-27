#include <fstream>
#include <iostream>
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

int expect_absent(const std::string& text, const char* needle, const char* message) {
    if (!contains(text, needle)) { return 0; }
    std::cerr << message << ": found `" << needle << "`\n";
    return 1;
}

int expect_present(const std::string& text, const char* needle, const char* message) {
    if (contains(text, needle)) { return 0; }
    std::cerr << message << ": missing `" << needle << "`\n";
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    const std::string gdn_common = read_file(QUS_SOURCE_DIR "/src/kernels/kernel/gdn_common.cuh");
    failures += expect_absent(gdn_common, "fastmodulo", "unused fast modulo helper must stay gone");

    const std::string gdn_chunked =
        read_file(QUS_SOURCE_DIR "/src/kernels/kernel/gdn_chunked_common.cuh");
    failures += expect_absent(gdn_chunked, "kda", "dead KDA config field must stay gone");
    failures += expect_absent(gdn_chunked, "Aqk", "dead KDA qk product field must stay gone");
    failures += expect_absent(gdn_chunked, "k_or_kg", "dead KDA key alias must stay gone");

    const std::string launcher =
        read_file(QUS_SOURCE_DIR "/src/kernels/launcher/gated_delta_rule.h");
    failures += expect_absent(launcher, "gdn_cast_bf16_to_f32_launch",
                              "unused single-tensor GDN cast launcher must stay gone");
    const std::string cast = read_file(QUS_SOURCE_DIR "/src/kernels/kernel/gdn_cast.cuh");
    failures += expect_absent(cast, "gdn_cast_bf16_to_f32_kernel",
                              "unused single-tensor GDN cast kernel must stay gone");

    const std::string silu = read_file(QUS_SOURCE_DIR "/src/kernels/kernel/silu_and_mul.cuh");
    const std::string sigmoid =
        read_file(QUS_SOURCE_DIR "/src/kernels/kernel/sigmoid_gate_mul.cuh");
    const std::string rmsnorm = read_file(QUS_SOURCE_DIR "/src/kernels/kernel/rmsnorm.cuh");
    failures += expect_absent(silu, "__expf", "silu_and_mul must use expf");
    failures += expect_absent(sigmoid, "__expf", "sigmoid_gate_mul must use expf");
    failures += expect_absent(rmsnorm, "__expf", "rmsnorm gated SiLU must use expf");

    const std::string config = read_file(QUS_SOURCE_DIR "/include/qus/model/config.h");
    failures += expect_present(config, "gdn_conv_state_width = gdn_conv_k - 1",
                               "model config must name the K-1 conv state width");
    const std::string model = read_file(QUS_SOURCE_DIR "/src/model/qwen3_6_27b.cpp");
    failures += expect_absent(model, "conv_state_for_kernel",
                              "model must not slice an over-allocated conv state");
    failures += expect_present(model, "Tensor& conv_state = state_.conv",
                               "model must pass the K-1 conv state directly");

    const std::string standard = read_file(QUS_SOURCE_DIR "/docs/l1-op-test-standard.md");
    failures += expect_present(standard, "`linear_bf16`", "standard must record linear_bf16");
    failures += expect_present(standard, "`attention_bf16`", "standard must record attention_bf16");
    failures +=
        expect_present(standard, "`gdn_output_bf16`", "standard must record gdn_output_bf16");
    failures += expect_present(standard, "`gdn_state_fp32`", "standard must record gdn_state_fp32");

    if (failures != 0) {
        std::cerr << "FAIL hardening cleanup structural invariants\n";
        return 1;
    }

    std::cout << "OK hardening cleanup structural invariants\n";
    return 0;
}
