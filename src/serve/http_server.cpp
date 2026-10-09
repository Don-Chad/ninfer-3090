#include "serve/http_server.h"
#include "serve/slot_files.h"

#include "product/version/version.h"
#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/props.h"
#include "serve/request_log.h"
#include "serve/slots_report.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.engine_recoveries != report.previous.engine_recoveries ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    if (path == "/v1/load") { return "load"; }
    if (path == "/slots" || path.starts_with("/slots/")) { return "slots"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

// Stamped on every response, including the loading 503, auth failures and malformed requests that
// never reach the pre-routing handler, so a caller can tell which build answered.
void set_version_header(httplib::Response& response) {
    response.set_header("X-NInfer-Version", std::string(ninfer::product::build_version()));
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    set_version_header(response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
    // cpp-httplib is thread-per-connection: a worker is held for a connection's whole
    // life, including the idle keep-alive window between requests. Sizing the pool to the
    // request-lifetime capacity alone lets idle pooled connections occupy every worker, at
    // which point a C8 server accepts and runs requests strictly one at a time. Base
    // workers cover the admissible request capacity; the headroom is grown on demand for
    // connections that are merely open, and those dynamic workers retire once idle.
    constexpr std::size_t kKeepAliveWorkerHeadroom = 64;
    const std::size_t request_workers =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests + 1;
    const std::size_t worker_limit = request_workers + kKeepAliveWorkerHeadroom;
    server_.new_task_queue         = [request_workers, worker_limit] {
        return new httplib::ThreadPool(request_workers, worker_limit, worker_limit);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    metrics_.record_rejection();
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    metrics_.record_done(outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    metrics_.record_failure();
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        set_version_header(res);
        if (!ready_.load(std::memory_order_acquire)) {
            // Runs for every route, including /health and OPTIONS, so a caller cannot tell "not
            // ready" apart from "unauthenticated" -- and skips the API-key check below, since a
            // loading-status response carries nothing worth protecting.
            ApiError error;
            error.status  = 503;
            error.type    = "service_unavailable";
            error.code    = "model_loading";
            error.message = "The model is still loading. Retry shortly.";
            // Weight load plus warmup measured ~10s on the 27B and longer on the 35B. Two seconds
            // is a polite poll interval rather than a promise about when readiness arrives.
            res.set_header("Retry-After", "2");
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"},
                                       {"version", ninfer::product::build_version()}}
                            .dump(),
                        "application/json");
    });
    server_.Get("/slots", [this](const httplib::Request& req, httplib::Response& res) {
        handle_slots(req, res);
    });
    server_.Post(R"(/slots/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_slot_action(req, res);
    });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        // Pre-routing answers 503 until attach(), so service_ is published here.
        res.set_header("Cache-Control", "no-store");
        res.set_content(metrics_.render(options_.max_concurrency, service_->runtime_stats(),
                                        service_->admitted_requests()),
                        "text/plain; version=0.0.4");
    });
    server_.Get("/v1/load", [this](const httplib::Request& req, httplib::Response& res) {
        handle_load(req, res);
    });
    server_.Get("/props", [this](const httplib::Request& req, httplib::Response& res) {
        handle_props(req, res);
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

// llama.cpp-shaped slot listing: one entry per private context-cache catalog cell. A cell an
// active request will publish into reports that request's prompt and reused tokens; a retained
// cell reports the session depth as both, with its session digest and restorable checkpoints,
// the slot file it is bound to, and how it has been used (see make_slots_report).
void HttpServer::handle_slots(const httplib::Request&, httplib::Response& res) const {
    const bool speculative = options_.speculative.backend != ninfer::SpeculativeBackend::None;
    res.set_header("Cache-Control", "no-store");
    res.set_content(make_slots_report(service_->slot_states(), options_.max_context, speculative),
                    "application/json");
}

// llama.cpp-shaped session persistence: POST /slots/{id}?action=save|restore|erase with
// {"filename": NAME} for save and restore and an optional {"if_digest": DIGEST} precondition on
// save and erase. Enabled only by --slot-save-path; names are confined to that directory.
void HttpServer::handle_slot_action(const httplib::Request& req, httplib::Response& res) {
    const auto fail = [&res](int status, std::string code, std::string message) {
        ApiError error;
        error.status  = status;
        error.type    = status >= 500 ? "server_error" : "invalid_request_error";
        error.code    = std::move(code);
        error.message = std::move(message);
        write_openai_error(res, error);
    };
    if (options_.slot_save_path.empty()) {
        fail(501, "slot_persistence_disabled",
             "this server was started without --slot-save-path; slot save/restore is disabled");
        return;
    }
    const std::string id_text = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(id_text);
    } catch (const std::exception&) {
        fail(400, "invalid_slot", "slot id is not a number");
        return;
    }
    // Range-checked before narrowing, so an id past 2^32 is refused rather than wrapped.
    const std::size_t slot_count = service_->slot_states().size();
    if (parsed >= slot_count) {
        fail(400, "invalid_slot",
             "slot " + id_text + " is outside this server's " + std::to_string(slot_count) +
                 " slots");
        return;
    }
    const auto slot = static_cast<std::uint32_t>(parsed);
    const std::string action = req.get_param_value("action");

    std::string filename;
    std::string if_digest;
    try {
        const nlohmann::json body =
            req.body.empty() ? nlohmann::json::object() : nlohmann::json::parse(req.body);
        if (!body.is_object()) { throw std::invalid_argument("body is not an object"); }
        filename  = body.value("filename", std::string());
        if_digest = body.value("if_digest", std::string());
    } catch (const std::exception&) {
        fail(400, "invalid_request",
             "request body must be a JSON object with string filename and if_digest");
        return;
    }

    try {
        if (action == "erase") {
            const std::uint32_t erased = service_->slot_erase(slot, if_digest);
            operational_log_.slot_erased(slot, erased);
            res.set_content(nlohmann::json{{"id_slot", slot}, {"n_erased", erased}}.dump(),
                            "application/json");
            return;
        }
        if (action != "save" && action != "restore") {
            fail(400, "invalid_action", "action must be save, restore, or erase");
            return;
        }
        const std::optional<std::string> sanitized = sanitize_slot_filename(filename);
        if (!sanitized) {
            fail(400, "invalid_filename",
                 "filename must be 1-" + std::to_string(kSlotFilenameMaxBytes) +
                     " characters of [A-Za-z0-9._-], must not start or end with a dot, and "
                     "must not name a device");
            return;
        }
        const std::string path = (options_.slot_save_path / *sanitized).string();
        if (action == "save") {
            const ninfer::SlotSaveResult saved = service_->slot_save(slot, path, if_digest);
            operational_log_.slot_saved(slot, *sanitized, saved);
            res.set_content(nlohmann::json{{"id_slot", slot},
                                           {"filename", *sanitized},
                                           {"n_saved", saved.tokens},
                                           {"n_written", saved.bytes},
                                           {"session_digest", saved.session_digest},
                                           {"timings", {{"save_ms", saved.seconds * 1000.0}}}}
                                .dump(),
                            "application/json");
        } else {
            const ninfer::SlotRestoreResult restored = service_->slot_restore(slot, path);
            operational_log_.slot_restored(slot, *sanitized, restored);
            res.set_content(
                nlohmann::json{{"id_slot", slot},
                               {"filename", *sanitized},
                               {"n_restored", restored.tokens},
                               {"n_read", restored.bytes},
                               {"session_digest", restored.session_digest},
                               {"timings", {{"restore_ms", restored.seconds * 1000.0}}}}
                    .dump(),
                "application/json");
        }
    } catch (const ninfer::RequestError& busy) {
        fail(409, "slot_busy", busy.what());
    } catch (const ninfer::SlotSessionMismatch& mismatch) {
        fail(409, "slot_session_mismatch", mismatch.what());
    } catch (const std::invalid_argument& rejected) {
        fail(400, "slot_" + action + "_failed", rejected.what());
    }
}

void HttpServer::handle_load(const httplib::Request&, httplib::Response& res) const {
    LoadSample sample;
    sample.uptime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - attached_at_).count();
    sample.admitted_requests = service_->admitted_requests();
    sample.stats             = service_->runtime_stats();
    sample.last_generation   = metrics_.last_generation_pace();
    res.set_header("Cache-Control", "no-store");
    res.set_content(make_load_report(load_capacity_, sample), "application/json");
}

void HttpServer::handle_props(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_props(options_, model_description(), service_->sampling_defaults()),
                    "application/json");
}

ModelDescription HttpServer::model_description() const {
    return ModelDescription{.id            = public_model_id_,
                            .max_model_len = options_.max_context,
                            .vision        = service_->engine_options().enable_vision};
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(model_description(), unix_time_now()), "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(model_description(), unix_time_now()), "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::start_serving_during_startup() {
    if (startup_listener_.joinable()) {
        throw std::logic_error("HTTP startup listener is already running");
    }

    // The readiness gate lives in register_routes()'s single pre-routing handler, ahead of the
    // API-key check -- not here, so starting this listener never replaces (and thereby drops) the
    // auth/request-ID middleware for the remainder of the process's life.
    startup_listener_ = std::thread([this] {
        // listen_after_bind() blocks here for the whole life of the server, spanning the switch
        // from 503 to serving. stop() is what ends it. It can also throw before ever reaching
        // that loop -- the task queue's thread pool spawns its worker threads here, and
        // std::thread's constructor throws std::system_error under resource exhaustion. An
        // exception escaping a thread function is std::terminate, so it is caught and folded into
        // the same false result a synchronous listen() failure already produces.
        bool result = false;
        try {
            result = server_.listen_after_bind();
        } catch (const std::exception&) {
        }
        startup_listener_result_.store(result, std::memory_order_release);
    });
}

bool HttpServer::await_startup_listener() {
    if (startup_listener_.joinable()) { startup_listener_.join(); }
    return startup_listener_result_.load(std::memory_order_acquire);
}

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_name);
    service_                       = &service;
    // memory_summary() takes the Engine execution lock; read it once here, never per /v1/load poll.
    const ninfer::MemorySummary memory = service.memory_summary();
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load, memory);
    load_capacity_ = make_load_capacity(public_model_id_, service.engine_options(), memory);
    attached_at_   = std::chrono::steady_clock::now();
    // Release: everything above must be visible to a handler that observes ready_ as true. This is
    // the only write, and handlers acquire it in the pre-routing guard before touching service_.
    ready_.store(true, std::memory_order_release);
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    try {
        if (options_.log_stats_interval_ms != 0) {
            stats_stopping_ = false;
            stats_thread_   = std::thread([this] { run_stats_reporter(); });
        }
        // When the startup listener is running, the accept loop is already live on its thread and
        // has been since bind(); calling listen_after_bind() again would try to accept on the same
        // socket from two threads. Wait for that loop instead.
        const bool result =
            startup_listener_.joinable() ? await_startup_listener() : server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        // Stop and join the startup listener before this exception unwinds past us. attach() has
        // already run by the time listen() can be called, so that thread is live against
        // service_; the caller (main.cpp) destroys the attached GenerationService, constructed
        // after this HttpServer, before this object -- leaving that thread dereferencing a
        // dangling pointer for however long stack unwinding takes if it is still running then.
        if (startup_listener_.joinable()) {
            server_.stop();
            startup_listener_.join();
        }
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

HttpServer::~HttpServer() {
    if (startup_listener_.joinable()) {
        server_.stop();
        startup_listener_.join();
    }
    stop_stats_reporter();
}

} // namespace ninfer::serve
