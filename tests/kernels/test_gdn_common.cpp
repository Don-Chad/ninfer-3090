// Host-only checks for the shared GDN kernel helpers. This intentionally tests
// the Qwen3.6 grouped value-head -> q/k-head mapping before the full recurrence
// kernels exist.
#include "kernels/kernel/gdn_common.cuh"

#include <iostream>

int main() {
    constexpr int H_qk = 16;
    constexpr int H_v  = 48;
    constexpr int G    = H_v / H_qk;

    const qus::kernels::head_map map = qus::kernels::head_map::of(H_qk, H_v);

    int failures = 0;
    for (int h_v = 0; h_v < H_v; ++h_v) {
        const int got      = map.qk_head(h_v);
        const int expected = h_v / G;
        if (got != expected) {
            std::cerr << "qk_head(" << h_v << ") got " << got << " expected " << expected << '\n';
            ++failures;
        }
    }

    for (int cta_h = 0; cta_h < H_v; ++cta_h) {
        const int got = map.cta_h_v(cta_h);
        if (got != cta_h) {
            std::cerr << "cta_h_v(" << cta_h << ") got " << got << " expected " << cta_h << '\n';
            ++failures;
        }
    }

    return failures == 0 ? 0 : 1;
}
