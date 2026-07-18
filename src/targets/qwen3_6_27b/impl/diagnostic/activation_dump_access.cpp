#include "targets/qwen3_6_27b/impl/diagnostic/activation_dump_access.h"

#include "targets/qwen3_6_27b/impl/variant.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b::detail {

void ActivationDumpAccess::attach(Package::Program& program, void* context,
                                  schedule::TextTapCallback text,
                                  schedule::VisionTapCallback vision) {
    if (context == nullptr || (text == nullptr && vision == nullptr)) {
        throw std::invalid_argument("activation dump attachment requires a context and callback");
    }
    Variant::attach_diagnostics(program, context, text, vision);
}

void ActivationDumpAccess::detach(Package::Program& program) noexcept {
    Variant::detach_diagnostics(program);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
