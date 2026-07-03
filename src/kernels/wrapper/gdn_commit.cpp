#include "qus/kernels/gdn_commit.h"

#include "kernels/launcher/gdn_commit.h"

#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {

void require_contiguous_nonnull(const Tensor& t, const char* label) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gdn_commit: ") + label + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gdn_commit: ") + label + " data must be non-null");
    }
}

void validate(const Tensor& conv_states, const Tensor& ssm_states, const Tensor& accepted) {
    if (conv_states.dtype != DType::BF16) {
        throw std::invalid_argument("gdn_commit: conv_states must be BF16");
    }
    if (ssm_states.dtype != DType::FP32) {
        throw std::invalid_argument("gdn_commit: ssm_states must be FP32");
    }
    if (accepted.dtype != DType::I32 || accepted.ne[0] != 1 || accepted.ne[1] != 1 ||
        accepted.ne[2] != 1 || accepted.ne[3] != 1) {
        throw std::invalid_argument("gdn_commit: accepted must be I32 [1]");
    }
    if (conv_states.ne[0] <= 0 || conv_states.ne[1] != 3 || conv_states.ne[2] <= 0 ||
        conv_states.ne[3] != 1) {
        throw std::invalid_argument("gdn_commit: conv_states must have shape [C,3,S]");
    }
    if (ssm_states.ne[0] <= 0 || ssm_states.ne[1] <= 0 || ssm_states.ne[2] <= 0 ||
        ssm_states.ne[3] != conv_states.ne[2]) {
        throw std::invalid_argument("gdn_commit: ssm_states must have shape [K,V,H,S]");
    }
    require_contiguous_nonnull(conv_states, "conv_states");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    require_contiguous_nonnull(accepted, "accepted");
}

} // namespace

void gdn_commit(Tensor& conv_states, Tensor& ssm_states, const Tensor& accepted,
                cudaStream_t stream) {
    validate(conv_states, ssm_states, accepted);
    detail::gdn_commit_launch(conv_states, ssm_states, accepted, stream);
}

} // namespace qus::kernels
