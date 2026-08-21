from __future__ import annotations

import http.client
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools.codex_gateway.gateway import (
    CLOUD_HOST,
    CLOUD_PORT,
    CLOUD_PREFIX,
    LOCAL_MODEL_SLUG,
    Gateway,
    GatewayConfig,
    GatewayServer,
    LocalResponseOwnership,
    default_launch_spec,
)


class DummyLease:
    def __init__(self, lifecycle: "DummyLifecycle") -> None:
        self.lifecycle = lifecycle
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.lifecycle.active -= 1

    def invalidate(self, reason: str) -> None:
        self.lifecycle.invalidations.append(reason)


class DummyLifecycle:
    def __init__(self) -> None:
        self.acquires = 0
        self.active = 0
        self.invalidations: list[str] = []

    def acquire(self) -> DummyLease:
        self.acquires += 1
        self.active += 1
        return DummyLease(self)

    def status(self) -> dict[str, object]:
        return {
            "state": "stopped",
            "owned_pid": None,
            "active_leases": self.active,
            "model": LOCAL_MODEL_SLUG,
        }


class RecordingUpstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), RecordingHandler)
        self.records: list[dict[str, object]] = []


class RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:
        pass

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        server = self.server
        assert isinstance(server, RecordingUpstream)
        server.records.append({
            "method": self.command,
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": body,
        })
        if self.headers.get("X-Test-SSE") == "1":
            response_id = self.headers.get("X-Test-Response-Id", "resp_stream_local")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            first = (
                'event: response.created\n'
                f'data: {{"type":"response.created","response":{{"id":"{response_id}"}}}}\n\n'
            ).encode()
            self.wfile.write(first)
            self.wfile.flush()
            time.sleep(0.6)
            self.wfile.write(b'event: response.completed\ndata: {"type":"response.completed"}\n\n')
            self.wfile.flush()
            self.close_connection = True
            return
        if self.headers.get("X-Test-Redirect") == "1":
            raw = b"redirect"
            self.send_response(307)
            self.send_header(
                "Location", self.headers.get("X-Test-Location", "https://evil.example/steal")
            )
        elif self.path.rstrip("/").endswith("/models"):
            if self.headers.get("If-None-Match") or self.headers.get("X-Test-Force-304"):
                self.send_response(304)
                self.send_header("ETag", '"cloud-old"')
                self.end_headers()
                return
            raw = b'{"models":[{"slug":"gpt-cloud","display_name":"Cloud"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("ETag", '"cloud-old"')
            self.send_header("Last-Modified", "Wed, 01 Jan 2020 00:00:00 GMT")
            self.send_header("Digest", "sha-256=stale")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Connection", "X-Stale-Validator")
            self.send_header("X-Stale-Validator", "must-not-pass")
        elif self.headers.get("X-Test-Response-Id"):
            raw = json.dumps({"id": self.headers["X-Test-Response-Id"], "object": "response"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            raw = body or b"upstream-ok"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    do_GET = _handle
    do_POST = _handle
    do_DELETE = _handle
    do_HEAD = _handle


@contextmanager
def running_server(server: ThreadingHTTPServer):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def gateway_fixture():
    cloud = RecordingUpstream()
    local = RecordingUpstream()
    lifecycle = DummyLifecycle()
    requested_destinations: list[tuple[str, str, int]] = []

    def connection_factory(kind: str, host: str, port: int, timeout: float):
        requested_destinations.append((kind, host, port))
        selected = cloud if kind == "cloud" else local
        return http.client.HTTPConnection("127.0.0.1", selected.server_port, timeout=timeout)

    config = GatewayConfig(port=0)
    gateway = Gateway(config, lifecycle, connection_factory=connection_factory)  # type: ignore[arg-type]
    server = GatewayServer((config.host, 0), gateway)
    with running_server(cloud), running_server(local), running_server(server):
        yield server, cloud, local, lifecycle, requested_destinations


def request(
    server: GatewayServer,
    method: str,
    path: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, list[tuple[str, str]], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    merged = dict(headers or {})
    if body:
        merged["Content-Length"] = str(len(body))
    connection.request(method, path, body=body, headers=merged)
    response = connection.getresponse()
    raw = response.read()
    result = response.status, response.getheaders(), raw
    connection.close()
    return result


def wait_for_no_active_lease(lifecycle: DummyLifecycle) -> None:
    deadline = time.monotonic() + 1
    while lifecycle.active and time.monotonic() < deadline:
        time.sleep(0.005)
    assert lifecycle.active == 0


def test_nonlocal_model_cloud_body_is_byte_exact_and_destination_is_fixed() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, destinations):
        body = b'{ "model" : "gpt-cloud", "input" : [1, 2, 3] }\n'
        status, _, raw = request(
            server, "POST", "/v1/responses?trace=yes", body,
            {
                "Authorization": "Bearer cloud-token",
                "Cookie": "session=cloud",
                "ChatGPT-Account-ID": "account-cloud",
                "x-oai-attestation": "attestation-cloud",
            },
        )

        assert status == 200
        assert raw == body
        assert not local.records
        assert lifecycle.acquires == 0
        assert cloud.records[0]["body"] == body
        assert cloud.records[0]["path"] == CLOUD_PREFIX + "/responses?trace=yes"
        headers = cloud.records[0]["headers"]
        assert isinstance(headers, dict)
        assert headers["authorization"] == "Bearer cloud-token"
        assert headers["cookie"] == "session=cloud"
        assert headers["chatgpt-account-id"] == "account-cloud"
        assert headers["x-oai-attestation"] == "attestation-cloud"
        assert destinations == [("cloud", CLOUD_HOST, CLOUD_PORT)]


def test_exact_local_slug_routes_local_and_strips_all_credentials() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, destinations):
        body = json.dumps({"model": LOCAL_MODEL_SLUG, "input": "hello"}).encode()
        status, _, raw = request(
            server, "POST", "/v1/responses", body,
            {
                "Authorization": "Bearer secret",
                "Proxy-Authorization": "Basic secret",
                "Cookie": "session=secret",
                "Cookie2": "legacy=secret",
                "X-Api-Key": "secret",
                "ChatGPT-Account-ID": "account-secret",
                "x-oai-attestation": "attestation-secret",
                "X-Request-Id": "safe",
                "Accept-Encoding": "gzip",
            },
        )

        assert status == 200
        assert raw == body
        assert not cloud.records
        assert lifecycle.acquires == 1
        wait_for_no_active_lease(lifecycle)
        headers = local.records[0]["headers"]
        assert isinstance(headers, dict)
        for name in (
            "authorization", "proxy-authorization", "cookie", "cookie2", "x-api-key",
            "chatgpt-account-id", "x-oai-attestation",
        ):
            assert name not in headers
        assert headers["x-request-id"] == "safe"
        assert headers["accept-encoding"] == "identity"
        assert destinations == [("local", "127.0.0.1", 8080)]


def test_exact_local_request_removes_only_hosted_web_search_tool() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        preserved_tools = [
            {"type": "function", "name": "probe", "parameters": {}},
            {"type": "custom", "name": "apply_patch", "format": {"type": "text"}},
            {"type": "namespace", "name": "github", "tools": []},
            {"type": "tool_search", "execution": "client"},
        ]
        body = json.dumps({
            "model": LOCAL_MODEL_SLUG,
            "input": "hello",
            "tools": [
                {"type": "web_search"},
                *preserved_tools,
                {"type": "web_search"},
            ],
            "tool_choice": "required",
        }).encode()

        status, _, raw = request(server, "POST", "/v1/responses", body)
        forwarded = json.loads(local.records[0]["body"])

        assert status == 200
        assert json.loads(raw) == forwarded
        assert forwarded["tools"] == preserved_tools
        assert forwarded["tool_choice"] == "required"
        assert not cloud.records
        assert lifecycle.acquires == 1


@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [
        ([{"type": "web_search"}], {"type": "web_search"}),
        ([{"type": "web_search"}], "required"),
        (None, {"type": "web_search"}),
    ],
)
def test_exact_local_request_fails_if_stripping_invalidates_tool_choice(
    tools: list[dict[str, str]] | None,
    tool_choice: object,
) -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        payload = {
            "model": LOCAL_MODEL_SLUG,
            "input": "hello",
            "tool_choice": tool_choice,
        }
        if tools is not None:
            payload["tools"] = tools
        body = json.dumps(payload).encode()
        status, _, raw = request(server, "POST", "/v1/responses", body)

        assert status == 400
        assert "unsupported hosted web_search" in json.loads(raw)["error"]["message"]
        assert not cloud.records
        assert not local.records
        assert lifecycle.acquires == 0


def test_cloud_hosted_web_search_request_body_remains_byte_exact() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        body = (
            b'{ "model" : "gpt-cloud", "tools" : '
            b'[{"type":"web_search"}], "tool_choice" : {"type":"web_search"} }\n'
        )
        status, _, raw = request(server, "POST", "/v1/responses", body)

        assert status == 200
        assert raw == body
        assert cloud.records[0]["body"] == body
        assert not local.records
        assert lifecycle.acquires == 0


def test_near_match_local_slug_stays_on_cloud() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        body = json.dumps({"model": LOCAL_MODEL_SLUG + "-typo", "input": "hello"}).encode()
        assert request(server, "POST", "/v1/responses", body)[0] == 200
        assert len(cloud.records) == 1
        assert not local.records
        assert lifecycle.acquires == 0


def test_exact_local_slug_on_other_model_bearing_post_still_routes_local() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        body = json.dumps({"model": LOCAL_MODEL_SLUG, "messages": []}).encode()
        assert request(server, "POST", "/v1/chat/completions", body)[0] == 200
        assert len(local.records) == 1
        assert not cloud.records
        assert lifecycle.acquires == 1


@pytest.mark.parametrize("path", ["/models", "/v1/models", CLOUD_PREFIX + "/models"])
def test_models_merges_one_vision_capable_local_entry(path: str) -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        status, _, raw = request(server, "GET", path)
        payload = json.loads(raw)
        local_entries = [item for item in payload["models"] if item.get("slug") == LOCAL_MODEL_SLUG]

        assert status == 200
        assert len(local_entries) == 1
        assert local_entries[0]["input_modalities"] == ["text", "image"]
        assert local_entries[0]["supports_image_detail_original"] is True
        assert local_entries[0]["apply_patch_tool_type"] == "freeform"
        # Codex 0.149 overloads this flag for deferred plugin tool_search. The
        # exact-local request sanitizer separately removes hosted web_search.
        assert local_entries[0]["supports_search_tool"] is True
        assert lifecycle.acquires == 0
        assert not local.records


def test_cloud_redirect_is_returned_without_following_or_replaying_post() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, destinations):
        body = b'{"model":"gpt-cloud","input":"once"}'
        status, headers, raw = request(
            server, "POST", "/v1/responses", body, {"X-Test-Redirect": "1"}
        )

        assert status == 307
        assert raw == b"redirect"
        assert "location" not in dict((key.lower(), value) for key, value in headers)
        assert len(cloud.records) == 1
        assert destinations == [("cloud", CLOUD_HOST, CLOUD_PORT)]
        assert not local.records
        assert lifecycle.acquires == 0


@pytest.mark.parametrize(
    "location",
    [
        "https://chatgpt.com/outside-codex",
        "/outside-codex",
        "https://chatgpt.com/backend-api/codex/%2e%2e/outside",
        "https://user@chatgpt.com/backend-api/codex/responses",
    ],
)
def test_unsafe_cloud_redirect_locations_are_stripped(location: str) -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        body = b'{"model":"gpt-cloud","input":"once"}'
        status, headers, _ = request(
            server,
            "POST",
            "/v1/responses",
            body,
            {"X-Test-Redirect": "1", "X-Test-Location": location},
        )
        assert status == 307
        assert "location" not in {name.lower() for name, _ in headers}
        assert len(cloud.records) == 1
        assert not local.records
        assert lifecycle.acquires == 0


def test_fixed_origin_in_prefix_redirect_location_is_preserved_without_following() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        body = b'{"model":"gpt-cloud","input":"once"}'
        location = "https://chatgpt.com/backend-api/codex/responses/resp_cloud"
        status, headers, _ = request(
            server,
            "POST",
            "/v1/responses",
            body,
            {"X-Test-Redirect": "1", "X-Test-Location": location},
        )
        response_headers = {name.lower(): value for name, value in headers}
        assert status == 307
        assert response_headers["location"] == location
        assert len(cloud.records) == 1
        assert not local.records
        assert lifecycle.acquires == 0


def test_absolute_target_and_path_traversal_are_rejected_before_upstream() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        assert request(server, "GET", "/v1/../outside")[0] == 400
        assert not cloud.records
        assert not local.records
        assert lifecycle.acquires == 0


@pytest.mark.parametrize(
    "path",
    [
        "/v1/%2e%2e/outside",
        "/v1/%252e%252e/outside",
        "/v1/safe%2f..%2foutside",
        "/v1/safe%5c..%5coutside",
        "/v1/safe\\..\\outside",
        "/v1/bad%2",
    ],
)
def test_encoded_or_backslash_traversal_is_rejected_before_upstream(path: str) -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        assert request(server, "GET", path)[0] == 400
        assert not cloud.records
        assert not local.records
        assert lifecycle.acquires == 0


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (b"not-json", {}),
        (b'{"input":"missing model"}', {}),
        (b"[]", {}),
        (b'{"model":"gpt-cloud"}', {"Content-Encoding": "gzip"}),
    ],
)
def test_undecidable_model_required_responses_fail_closed(
    body: bytes, headers: dict[str, str]
) -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        status, _, _ = request(server, "POST", "/v1/responses", body, headers)
        assert status == 400
        assert not cloud.records
        assert not local.records
        assert lifecycle.acquires == 0


def test_catalog_strips_request_conditionals_and_stale_response_validators() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        status, headers, raw = request(
            server,
            "GET",
            "/v1/models",
            headers={
                "If-None-Match": '"cloud-old"',
                "If-Modified-Since": "Wed, 01 Jan 2020 00:00:00 GMT",
            },
        )
        response_headers = {name.lower(): value for name, value in headers}
        upstream_headers = cloud.records[0]["headers"]

        assert status == 200
        assert any(item.get("slug") == LOCAL_MODEL_SLUG for item in json.loads(raw)["models"])
        assert isinstance(upstream_headers, dict)
        assert "if-none-match" not in upstream_headers
        assert "if-modified-since" not in upstream_headers
        for name in ("etag", "last-modified", "digest", "x-stale-validator"):
            assert name not in response_headers
        assert response_headers["cache-control"] == "no-store"
        assert upstream_headers["cache-control"] == "no-cache"
        assert not local.records
        assert lifecycle.acquires == 0


def test_unconditional_cloud_catalog_304_fails_instead_of_losing_local_model() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        status, _, raw = request(
            server, "GET", "/v1/models", headers={"X-Test-Force-304": "1"}
        )
        assert status == 502
        assert json.loads(raw)["error"]["type"] == "gateway_catalog_error"
        assert len(cloud.records) == 1
        assert not local.records
        assert lifecycle.acquires == 0


def test_connection_named_request_header_is_removed() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        body = b'{"model":"gpt-cloud","input":"safe"}'
        status, _, _ = request(
            server,
            "POST",
            "/v1/responses",
            body,
            headers={"Connection": "X-Remove-Me", "X-Remove-Me": "secret-hop"},
        )
        upstream_headers = cloud.records[0]["headers"]
        assert status == 200
        assert isinstance(upstream_headers, dict)
        assert "x-remove-me" not in upstream_headers


def test_nonstream_local_response_owns_all_model_less_state_routes() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        response_id = "resp_local_nonstream_1"
        body = json.dumps({"model": LOCAL_MODEL_SLUG, "input": "hello"}).encode()
        status, _, raw = request(
            server,
            "POST",
            "/v1/responses",
            body,
            headers={"X-Test-Response-Id": response_id},
        )
        assert status == 200
        assert json.loads(raw)["id"] == response_id

        routes = [
            ("GET", f"/v1/responses/{response_id}"),
            ("GET", f"/v1/responses/{response_id}/input_items"),
            ("POST", f"/v1/responses/{response_id}/cancel"),
            ("DELETE", f"/v1/responses/{response_id}"),
            # Deletion keeps a tombstone so this local ID can never leak cloud.
            ("GET", f"/v1/responses/{response_id}"),
        ]
        for method, path in routes:
            assert request(server, method, path)[0] == 200

        assert not cloud.records
        assert len(local.records) == 1 + len(routes)
        assert lifecycle.acquires == 1 + len(routes)


def test_streaming_local_response_is_incremental_and_owns_followup_route() -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        response_id = "resp_local_stream_1"
        body = json.dumps({"model": LOCAL_MODEL_SLUG, "input": "stream", "stream": True}).encode()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        started = time.monotonic()
        connection.request(
            "POST",
            "/v1/responses",
            body=body,
            headers={
                "Content-Length": str(len(body)),
                "X-Test-SSE": "1",
                "X-Test-Response-Id": response_id,
            },
        )
        response = connection.getresponse()
        first_event = b"".join(response.readline() for _ in range(3))
        first_latency = time.monotonic() - started
        rest = response.read()
        connection.close()

        assert response.status == 200
        assert response_id.encode() in first_event
        assert first_latency < 0.4
        assert b"response.completed" in rest
        assert request(server, "GET", f"/v1/responses/{response_id}")[0] == 200
        assert not cloud.records
        wait_for_no_active_lease(lifecycle)


def test_local_response_ownership_survives_gateway_restart(tmp_path) -> None:
    state = tmp_path / "owned-response-ids.txt"
    first = LocalResponseOwnership(state)
    first.remember("resp_persistent_local")
    second = LocalResponseOwnership(state)

    assert second.owns("resp_persistent_local")
    assert state.read_text(encoding="utf-8") == "resp_persistent_local\n"


def test_local_transport_failure_invalidates_owned_generation() -> None:
    class FailingConnection:
        def request(self, *args: object, **kwargs: object) -> None:
            raise TimeoutError("local transport wedged")

        def close(self) -> None:
            pass

    lifecycle = DummyLifecycle()
    config = GatewayConfig(port=0)

    def connection_factory(kind: str, host: str, port: int, timeout: float):
        assert kind == "local"
        return FailingConnection()

    gateway = Gateway(config, lifecycle, connection_factory=connection_factory)  # type: ignore[arg-type]
    server = GatewayServer((config.host, 0), gateway)
    body = json.dumps({"model": LOCAL_MODEL_SLUG, "input": "hello"}).encode()
    with running_server(server):
        status, _, _ = request(server, "POST", "/v1/responses", body)

    assert status == 502
    assert lifecycle.invalidations == ["local transport wedged"]
    assert lifecycle.active == 0


@pytest.mark.parametrize("routing_model", [LOCAL_MODEL_SLUG, "gpt-5.6-luna"])
def test_websocket_upgrade_is_declined_without_trusting_routing_hint(
    routing_model: str,
) -> None:
    with gateway_fixture() as (server, cloud, local, lifecycle, _):
        status, _, raw = request(
            server,
            "GET",
            "/v1/responses",
            headers={
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "x-codex-routing-hint": f"model={routing_model}",
            },
        )
        assert status == 426
        assert json.loads(raw)["error"]["type"] == "websocket_not_supported"
        assert not cloud.records
        assert not local.records
        assert lifecycle.acquires == 0


def test_default_engine_contract_is_exact_huihui_vision_mtp_on_port_8080(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HUIHUI_NINFER_EXE", "HUIHUI_NINFER_MODEL",
        "HUIHUI_STARTUP_TIMEOUT_SECONDS", "HUIHUI_IDLE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    spec = default_launch_spec()
    command = list(spec.command)

    assert spec.host == "127.0.0.1"
    assert spec.port == 8080
    assert command[1].endswith("models\\huihui_qwen3_8_27b_abliterated.ninfer")
    assert command[command.index("--port") + 1] == "8080"
    assert command[command.index("--max-context") + 1] == "65536"
    assert command[command.index("--kv-capacity") + 1] == "65536"
    assert command[command.index("--kv-dtype") + 1] == "rk8v4"
    assert command[command.index("--spec") + 1] == "mtp"
    assert command[command.index("--draft-tokens") + 1] == "3"
    assert "--lm-head-draft" in command
    assert "--context-lookup" in command
    assert command[command.index("--context-lookup-verify-tokens") + 1] == "4"
    assert "--vision" in command
