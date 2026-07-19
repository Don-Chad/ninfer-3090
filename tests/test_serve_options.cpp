#include "serve/serve_options.h"
#include "serve/translate.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");

    const ServeOptions disabled = parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse"});
    failures += check(!disabled.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");

    GenerationRequest request;
    request.max_tokens = 1;
    failures += check(to_request_options(request, defaults).execution.allow_prefix_reuse,
                      "default server policy did not reach Engine options");
    failures += check(!to_request_options(request, disabled).execution.allow_prefix_reuse,
                      "disabled server policy did not reach Engine options");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
