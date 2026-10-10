#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "product/version/version.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

// Time for in-flight error responses to flush once the Engine has latched, before the exit.
constexpr std::chrono::seconds kEngineFailureExitGrace{5};
constexpr int kEngineFailureExitStatus = 3;

// Called on the Engine worker thread after every host-side worker failure. A recovered failure is
// only logged. A latch leaves the process alive holding VRAM and answering 503 forever, so unless
// disabled it is logged FATAL and the process exits non-zero for a supervisor to restart; /health
// is already 503 during the grace period.
void handle_engine_fault(const ninfer::serve::OperationalLog& log,
                         const std::shared_ptr<spdlog::logger>& logger, bool exit_on_failure,
                         const ninfer::EngineFaultEvent& event) {
    // The exit is scheduled before anything that can throw: rendering or writing the event must
    // not be able to keep a latched process alive.
    static std::atomic<bool> exit_scheduled{false};
    const bool first_latch = event.latched && !exit_scheduled.exchange(true);
    if (first_latch && exit_on_failure) {
        try {
            std::thread([logger] {
                std::this_thread::sleep_for(kEngineFailureExitGrace);
                try {
                    logger->flush();
                } catch (...) {}
                std::_Exit(kEngineFailureExitStatus);
            }).detach();
        } catch (...) {
            // The thread could not be started. The exit is mandatory, so take it now without the
            // grace period rather than staying up latched.
            std::_Exit(kEngineFailureExitStatus);
        }
    }
    try {
        log.engine_fault(event);
        if (first_latch) {
            log.engine_fatal_exit(std::chrono::duration<double>(kEngineFailureExitGrace).count(),
                                  exit_on_failure);
        }
        logger->flush();
    } catch (...) {}
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "ninfer-serve " << ninfer::product::build_version() << '\n';
        return 0;
    }
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        // Answer 503 from here on rather than leaving the accepted connection silent. The socket
        // has been listenable since bind() either way; the difference is whether a caller arriving
        // during the ten seconds of weight loading gets a documented "still loading" or a hang.
        server.start_serving_during_startup();

        ninfer::serve::GenerationService service(
            options, startup_log.observer(),
            [&operational_log](const ninfer::ContextStoreWriteEvent& e) {
                operational_log.context_store_write(e);
            },
            [operational_log, logger, exit_on_failure = options.exit_on_engine_failure](
                const ninfer::EngineFaultEvent& event) {
                handle_engine_fault(operational_log, logger, exit_on_failure, event);
            });
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#ifdef SIGBREAK
        // Windows has no way to deliver SIGTERM to another process: TerminateProcess kills it
        // outright and the shutdown path -- which flushes the final partial throughput interval --
        // never runs. GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT) is the one graceful stop a parent
        // can request, and the CRT raises it as SIGBREAK.
        std::signal(SIGBREAK, handle_signal);
#endif

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
