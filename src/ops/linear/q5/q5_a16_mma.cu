#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {
void launch_q5_a16_mma_r32_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T128>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T128>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t16_k64_wr16_wt8_s2_a2_b3(const Tensor& x, const Weight& w, Tensor& out,
                                                     cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T16K64Wr16Wt8S2A2B3>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t32_k64_wr16_wt16_s3_a3_b2(const Tensor& x, const Weight& w,
                                                      Tensor& out, cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T32K64Wr16Wt16S3A3B2>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t64_k64_wr32_wt32_s2_a2_b3_pingpong(const Tensor& x, const Weight& w,
                                                               Tensor& out, cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T64K64Wr32Wt32S2A2B3PingPong>(x, w, out,
                                                                                 stream);
}

void launch_q5_a16_mma_r64_t96_k128_s1_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T96K128S1A1>(x, w, out, stream);
}
} // namespace ninfer::ops::detail
