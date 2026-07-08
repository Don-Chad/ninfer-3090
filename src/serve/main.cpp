#include "qus/serve/generation_service.h"
#include "qus/serve/http_server.h"
#include "qus/serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

std::atomic<qus::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    qus::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    } else {
        out << bytes << " B";
    }
    return out.str();
}

std::string format_kv_dtype(qus::DType dtype) {
    switch (dtype) {
    case qus::DType::BF16:
        return "bf16";
    case qus::DType::I8:
        return "int8";
    default:
        return "unknown";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const qus::serve::ServeOptions options = qus::serve::parse_serve_options(argc, argv);
        if (options.help_requested) {
            std::cout << qus::serve::serve_usage_text(argv[0]);
            return 0;
        }

        using Clock = std::chrono::steady_clock;
        std::cerr << "qus-serve: loading model...\n";
        const auto load_start = Clock::now();
        qus::serve::GenerationService service(options);
        std::cerr << "qus-serve: model loaded in "
                  << std::chrono::duration<double>(Clock::now() - load_start).count() << " s\n";
        const qus::EngineMemoryStats memory = service.memory_stats();
        std::cerr << "qus-serve: kv cache dtype=" << format_kv_dtype(memory.kv_dtype) << " payload="
                  << format_bytes(static_cast<std::uint64_t>(memory.kv_cache_payload_bytes))
                  << " cache_used="
                  << format_bytes(static_cast<std::uint64_t>(memory.cache.used_bytes)) << " / "
                  << format_bytes(static_cast<std::uint64_t>(memory.cache.capacity_bytes)) << '\n';

        std::cerr << "qus-serve: warming up...\n";
        service.warmup();

        qus::serve::HttpServer server(service, options);
        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::cerr << "qus-serve: listening on http://" << options.host << ':' << options.port
                  << " (model id: " << options.model_id
                  << ", auth: " << (options.api_key.empty() ? "disabled" : "bearer") << ")\n";

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            std::cerr << "error: failed to bind " << options.host << ':' << options.port << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << qus::serve::serve_usage_text(argv[0]);
        return 1;
    }
}
