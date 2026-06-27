#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace {

std::string read_file(const char* path) {
    std::ifstream file(path);
    if (!file) { throw std::runtime_error(std::string("failed to open ") + path); }
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

bool is_identifier_char(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

bool contains_identifier(const std::string& text, const std::string& symbol) {
    std::size_t pos = 0;
    while ((pos = text.find(symbol, pos)) != std::string::npos) {
        const bool left_ok  = pos == 0 || !is_identifier_char(text[pos - 1]);
        const auto end      = pos + symbol.size();
        const bool right_ok = end == text.size() || !is_identifier_char(text[end]);
        if (left_ok && right_ok) { return true; }
        pos = end;
    }
    return false;
}

int require_absent_substring(const std::string& path, const std::string& text,
                             const std::string& symbol) {
    if (text.find(symbol) == std::string::npos) { return 0; }
    std::cerr << path << ": retired dense linear symbol still present: " << symbol << '\n';
    return 1;
}

int require_absent_identifier(const std::string& path, const std::string& text,
                              const std::string& symbol) {
    if (!contains_identifier(text, symbol)) { return 0; }
    std::cerr << path << ": retired dense linear symbol still present: " << symbol << '\n';
    return 1;
}

} // namespace

int main() {
    const std::string kernel_path =
        std::string(QUS_SOURCE_DIR) + "/src/kernels/kernel/linear_dense.cuh";
    const std::string launcher_path =
        std::string(QUS_SOURCE_DIR) + "/src/kernels/launcher/linear_dense.cu";
    const std::string kernel   = read_file(kernel_path.c_str());
    const std::string launcher = read_file(launcher_path.c_str());

    int failures = 0;
    for (const char* symbol : std::array{
             "linear_dense_gemm_bf16_full_kernel",
             "linear_dense_gemm_bf16_half_wide_full_kernel",
             "linear_dense_gemm_tf32_kernel",
             "linear_dense_gemm_warp_kernel",
             "dense_cp_async_",
             "dense_gemm_issue_bf16_full_tile",
             "kDenseGemvWarps",
         }) {
        failures += require_absent_substring(kernel_path, kernel, symbol);
    }
    for (const char* symbol : std::array{
             "linear_dense_gemm_bf16_full_kernel",
             "linear_dense_gemm_bf16_half_wide_full_kernel",
             "linear_dense_gemm_tf32_kernel",
             "linear_dense_gemm_warp_kernel",
             "is_full_bf16_tiled_gemm",
             "is_full_bf16_half_wide_gemm",
             "tiled_gemm_half_wide_grid_for",
             "gemm_grid_for",
         }) {
        failures += require_absent_identifier(launcher_path, launcher, symbol);
    }

    if (failures != 0) {
        std::cerr << "FAIL linear dense structure\n";
        return 1;
    }
    std::cout << "OK linear dense structure\n";
    return 0;
}
