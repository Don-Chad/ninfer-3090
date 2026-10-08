#pragma once

#include "serve/generation_service.h"
#include "serve/load_report.h"
#include "serve/openai_common.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "serve/serve_metrics.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

class HttpServer {
public:
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger);
    // Stops and joins the startup listener if it is still running. Without this, a failure between
    // start_serving_during_startup() and listen() -- the Engine throwing while loading weights, the
    // most likely failure there is -- would destroy a joinable std::thread and call std::terminate,
    // turning a clean diagnosable error into an abort.
    ~HttpServer();

    HttpServer(const HttpServer&)            = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    // Serve 503 while the Engine is still loading.
    //
    // bind() deliberately runs before the Engine is constructed, so a port clash fails in
    // milliseconds instead of after ten seconds of weight loading. The cost used to be a window --
    // measured at 0.27s to 9.5s on the 27B -- where the kernel accepted connections into the
    // backlog and nothing ever answered them: a TCP readiness probe called that "ready", and an
    // HTTP probe burned its whole timeout instead of failing fast.
    //
    // Calling this immediately after bind() starts the accept loop on a background thread with
    // every route answering 503 plus Retry-After. attach() then publishes the service and the same
    // loop begins serving normally, with no second bind and no handoff of the listening socket.
    void start_serving_during_startup();
    // Blocks until the background accept loop returns, which happens when stop() is called.
    // Mirrors listen()'s return: true when the loop exited cleanly.
    bool await_startup_listener();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    [[nodiscard]] std::shared_ptr<RequestLifecycle> begin_request(RequestLogContext context);

    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_load(const httplib::Request& req, httplib::Response& res) const;
    void handle_slots(const httplib::Request& req, httplib::Response& res) const;
    [[nodiscard]] ModelDescription model_description() const;
    void handle_props(const httplib::Request& req, httplib::Response& res) const;
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();

    // Written once by attach() on the main thread and read by request handlers on httplib's worker
    // threads, so the publication has to be ordered. Handlers only ever test readiness through
    // ready_; service_ itself is not read until ready_ has been observed true.
    GenerationService* service_ = nullptr;
    std::atomic<bool> ready_{false};
    std::thread startup_listener_;
    std::atomic<bool> startup_listener_result_{false};
    ServeOptions options_;
    std::string public_model_id_;
    // Written by attach() together with service_, before ready_ is published.
    LoadCapacity load_capacity_;
    std::chrono::steady_clock::time_point attached_at_;
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    JsonlRequestLog request_jsonl_;
    ServeMetrics metrics_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
};

} // namespace ninfer::serve
