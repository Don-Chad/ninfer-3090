#include "qus/kernels/logits_mask.h"

#include "kernels/launcher/logits_mask.h"

#include <stdexcept>

namespace qus::kernels {

void mask_invalid_token_logits(Tensor& logits, std::int32_t valid_vocab, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || logits.ne[0] <= 0 || logits.ne[1] < 0 || logits.ne[2] != 1 ||
        logits.ne[3] != 1 || !logits.is_contiguous() || logits.data == nullptr) {
        throw std::invalid_argument(
            "mask_invalid_token_logits: logits must be contiguous non-null BF16 [vocab,T]");
    }
    if (valid_vocab <= 0 || valid_vocab > logits.ne[0]) {
        throw std::invalid_argument("mask_invalid_token_logits: valid_vocab outside logits rows");
    }
    detail::mask_invalid_token_logits_launch(logits, valid_vocab, stream);
}

} // namespace qus::kernels
