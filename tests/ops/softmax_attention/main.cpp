#include <exception>
#include <iostream>
#include <string_view>

int run_softmax_attention_causal_cache_tests();
int run_softmax_attention_dflash2_tests();
int run_softmax_attention_int8_prompt_tests();
int run_softmax_attention_nvfp4_tests();
int run_softmax_attention_k8v4_tests();
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

namespace {

// Without this, any throw out of a suite leaves MSVC to terminate the process with
// STATUS_STACK_BUFFER_OVERRUN (0xC0000409) and no output at all -- which reads exactly like memory
// corruption in a CUDA kernel and is not. That misdiagnosis cost real time on the DFlash2 sweep:
// the actual fault was a std::vector index out of range in the host-side reference, and it took
// compute-sanitizer reporting zero device errors to rule the GPU out.
int run_guarded(const char* what, int (*suite)()) {
    try {
        return suite();
    } catch (const std::exception& error) {
        std::cerr << "softmax_attention: " << what << " threw: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "softmax_attention: " << what << " threw a non-std exception\n";
        return 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--dflash2-only")
        return run_guarded("dflash2", run_softmax_attention_dflash2_tests);
    if (argc == 2 && std::string_view(argv[1]) == "--int8-prompt-only")
        return run_guarded("int8 prompt", run_softmax_attention_int8_prompt_tests);
    if (argc == 2 && std::string_view(argv[1]) == "--nvfp4-only") {
        return run_guarded("nvfp4", run_softmax_attention_nvfp4_tests);
    }
    if (argc == 2 && std::string_view(argv[1]) == "--k8v4-only") {
        return run_guarded("k8v4", run_softmax_attention_k8v4_tests);
    }
    if (argc != 1) {
        std::cerr
            << "usage: ninfer_softmax_attention_test "
               "[--dflash2-only|--int8-prompt-only|--nvfp4-only|--k8v4-only]\n";
        return 2;
    }
    const int causal = run_guarded("causal cache", run_softmax_attention_causal_cache_tests);
    if (causal == 77) return 77;

    const int plain_and_packed = run_guarded("plain/packed", run_softmax_attention_plain_and_packed_tests);
    if (plain_and_packed == 77) return 77;

    const int context = run_guarded("context", run_softmax_attention_context_tests);
    if (context == 77) return 77;

    const int failures = causal + plain_and_packed + context;
    std::cout << (failures == 0 ? "softmax_attention: PASS\n" : "softmax_attention: FAIL\n");
    return failures == 0 ? 0 : 1;
}
