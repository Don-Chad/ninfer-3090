"""HTTP-only Codex model gateway for one exact local Huihui identity."""

from __future__ import annotations

import copy
import http.client
import json
import logging
import os
import re
import ssl
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import SplitResult, unquote, urlsplit

from .lifecycle import EngineLifecycle, LaunchSpec, LifecycleError


LOCAL_MODEL_SLUG = "huihui-qwen3.8-27b-abliterated"
CLOUD_HOST = "chatgpt.com"
CLOUD_PORT = 443
CLOUD_PREFIX = "/backend-api/codex"

HOP_BY_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host",
})
LOCAL_SECRET_HEADERS = frozenset({
    "authorization", "proxy-authorization", "cookie", "cookie2", "x-api-key",
    "chatgpt-account-id", "x-oai-attestation",
})
CATALOG_CONDITIONAL_HEADERS = frozenset({
    "if-match", "if-none-match", "if-modified-since", "if-unmodified-since", "if-range",
})
CATALOG_REQUEST_CACHE_HEADERS = frozenset({"cache-control", "pragma"})
REPRESENTATION_VALIDATORS = frozenset({
    "etag", "last-modified", "digest", "content-digest", "repr-digest", "content-md5",
})
CATALOG_RESPONSE_CACHE_HEADERS = frozenset({"cache-control", "expires", "age"})
RESPONSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,256}$")

LOCAL_MODEL_ENTRY = {
    "slug": LOCAL_MODEL_SLUG,
    "display_name": "Huihui Qwen3.8 27B Abliterated (NInfer RTX 3090)",
    "description": (
        "Exact pinned Huihui abliterated checkpoint with always-on image vision "
        "and optimized MTP on the local RTX 3090."
    ),
    "default_reasoning_level": "medium",
    "supported_reasoning_levels": [
        {"effort": "none", "description": "Disable thinking for direct answers and routine tool calls."},
        {"effort": "low", "description": "Brief focused reasoning."},
        {"effort": "medium", "description": "Normal Qwen3.8 reasoning."},
        {"effort": "xhigh", "description": "Extended deliberation and verification."},
    ],
    "shell_type": "shell_command",
    "visibility": "list",
    "supported_in_api": True,
    "priority": 100,
    "base_instructions": (
        "You are Codex, a coding agent working in the user's current workspace. "
        "Follow the user's request, inspect relevant files before changing them, "
        "preserve unrelated work, use tools when needed, and report verified results concisely."
    ),
    "supports_reasoning_summaries": False,
    "default_reasoning_summary": "none",
    "support_verbosity": False,
    "apply_patch_tool_type": "freeform",
    "truncation_policy": {"mode": "tokens", "limit": 8192},
    "supports_parallel_tool_calls": True,
    "experimental_supported_tools": [],
    "input_modalities": ["text", "image"],
    "supports_image_detail_original": True,
    # Codex 0.149 overloads this flag to enable deferred client tool_search;
    # false expands the entire plugin inventory into the prompt. The local
    # request sanitizer below removes injected hosted web_search before NInfer.
    "supports_search_tool": True,
    "use_responses_lite": False,
    "context_window": 32768,
    "effective_context_window_percent": 90,
}


@dataclass(frozen=True)
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8081
    local_host: str = "127.0.0.1"
    local_port: int = 8080
    upstream_timeout: float = 3600.0

    def __post_init__(self) -> None:
        if self.host != "127.0.0.1" or self.local_host != "127.0.0.1":
            raise ValueError("gateway and local engine must be loopback-only")
        if self.local_port != 8080:
            raise ValueError("the candidate local engine port is fixed at 8080")


def _safe_request_target(raw_target: str) -> SplitResult:
    parsed = urlsplit(raw_target)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        raise ValueError("absolute or malformed request target is forbidden")
    if "\\" in parsed.path:
        raise ValueError("path traversal is forbidden")
    decoded = parsed.path
    for _ in range(4):
        next_decoded = unquote(decoded, errors="strict")
        if next_decoded == decoded:
            break
        decoded = next_decoded
    else:
        raise ValueError("excessively encoded request path is forbidden")
    if "\\" in decoded or any(ord(character) < 0x20 for character in decoded):
        raise ValueError("path traversal is forbidden")
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise ValueError("path traversal is forbidden")
    if re.search(r"%(?![0-9A-Fa-f]{2})", parsed.path):
        raise ValueError("malformed percent encoding is forbidden")
    if decoded.count("/") != parsed.path.count("/"):
        # Encoded separators make origin-prefix validation ambiguous even when
        # the resulting segments are not literal dot segments.
        raise ValueError("encoded path separators are forbidden")
    return parsed


def _with_query(path: str, query: str) -> str:
    return path + (("?" + query) if query else "")


def _cloud_path(parsed: SplitResult) -> str:
    path = parsed.path
    if path == CLOUD_PREFIX or path.startswith(CLOUD_PREFIX + "/"):
        result = path
    else:
        if path == "/v1":
            path = "/"
        elif path.startswith("/v1/"):
            path = path[3:]
        result = CLOUD_PREFIX + (path if path.startswith("/") else "/" + path)
    if result != CLOUD_PREFIX and not result.startswith(CLOUD_PREFIX + "/"):
        raise ValueError("cloud target escaped the fixed Codex prefix")
    return _with_query(result, parsed.query)


def _required_model_from_body(body: bytes) -> str:
    if not body:
        raise ValueError("Responses request body is required")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Responses request body must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Responses request body must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Responses request requires a non-empty model")
    return model


def _optional_model_from_body(body: bytes) -> str | None:
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return model if isinstance(model, str) else None


def sanitize_exact_local_request(body: bytes) -> bytes:
    """Remove only the hosted search tool injected by built-in Codex 0.149.

    NInfer supports function, custom, namespace, and deferred ``tool_search``
    entries. Those and the request's tool-choice semantics are preserved.
    """
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("exact local request body must be a JSON object")
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "web_search":
        raise ValueError(
            "exact local request explicitly selects unsupported hosted web_search"
        )
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return body
    filtered = [
        tool
        for tool in tools
        if not (isinstance(tool, dict) and tool.get("type") == "web_search")
    ]
    removed = len(tools) - len(filtered)
    if not removed:
        return body

    if tool_choice == "required" and not filtered:
        raise ValueError(
            "exact local request requires a tool but only unsupported hosted web_search was supplied"
        )

    sanitized = dict(payload)
    sanitized["tools"] = filtered
    logging.info("removed %s hosted web_search tool(s) from exact local request", removed)
    return json.dumps(sanitized, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _endpoint_path(path: str) -> str:
    if path == CLOUD_PREFIX:
        path = "/"
    elif path.startswith(CLOUD_PREFIX + "/"):
        path = path[len(CLOUD_PREFIX):]
    if path == "/v1":
        return "/"
    if path.startswith("/v1/"):
        return path[3:]
    return path


def _model_required_endpoint(path: str) -> bool:
    return _endpoint_path(path).rstrip("/") in {"/responses", "/responses/input_tokens"}


def _local_path(parsed: SplitResult) -> str:
    endpoint = _endpoint_path(parsed.path)
    path = "/v1" if endpoint == "/" else "/v1" + endpoint
    return _with_query(path, parsed.query)


def _response_id_from_path(path: str) -> str | None:
    endpoint = _endpoint_path(path).rstrip("/")
    match = re.fullmatch(r"/responses/([^/]+)(?:/(?:input_items|cancel))?", endpoint)
    if match is None:
        return None
    response_id = unquote(match.group(1), errors="strict")
    return response_id if RESPONSE_ID_PATTERN.fullmatch(response_id) else None


class LocalResponseOwnership:
    """Persistent authority for model-less local response routes."""

    def __init__(self, state_path: Path | None = None) -> None:
        self._state_path = state_path
        self._ids: set[str] = set()
        self._lock = threading.Lock()
        if state_path is not None and state_path.is_file():
            for line in state_path.read_text(encoding="utf-8").splitlines():
                if RESPONSE_ID_PATTERN.fullmatch(line):
                    self._ids.add(line)

    def remember(self, response_id: str) -> None:
        if RESPONSE_ID_PATTERN.fullmatch(response_id) is None:
            return
        with self._lock:
            if response_id in self._ids:
                return
            if self._state_path is not None:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                with self._state_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(response_id + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            self._ids.add(response_id)

    def owns(self, response_id: str | None) -> bool:
        if response_id is None:
            return False
        with self._lock:
            return response_id in self._ids


def merge_local_model(raw: bytes) -> bytes:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("cloud model catalog is not an object")
    key = "models" if "models" in payload else "data" if "data" in payload else "models"
    models = payload.setdefault(key, [])
    if not isinstance(models, list):
        raise ValueError(f"cloud model catalog field {key!r} is not an array")
    models[:] = [
        item for item in models
        if not isinstance(item, dict)
        or (item.get("slug") != LOCAL_MODEL_SLUG and item.get("id") != LOCAL_MODEL_SLUG)
    ]
    models.append(copy.deepcopy(LOCAL_MODEL_ENTRY))
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        lifecycle: EngineLifecycle,
        *,
        connection_factory: Callable[[str, str, int, float], http.client.HTTPConnection] | None = None,
        ownership: LocalResponseOwnership | None = None,
    ) -> None:
        self.config = config
        self.lifecycle = lifecycle
        self._connection_factory = connection_factory or self._default_connection
        self._ownership = ownership or LocalResponseOwnership()

    @staticmethod
    def _default_connection(
        kind: str, host: str, port: int, timeout: float
    ) -> http.client.HTTPConnection:
        if kind == "cloud":
            context = ssl.create_default_context()
            return http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            parsed = _safe_request_target(handler.path)
            if parsed.path == "/gateway/health" and handler.command == "GET":
                self._send_json(handler, 200, {"status": "ok", **self.lifecycle.status()})
                return
            # Codex reuses one Responses WebSocket across model switches, while its
            # handshake routing hint describes only the model that opened the socket.
            # Tunneling by that hint could therefore send a later local Huihui prompt
            # to cloud. Reject every upgrade before reading frames; Codex then uses its
            # session-scoped HTTP fallback, whose model-bearing requests route safely.
            if (handler.headers.get("Upgrade") or "").lower() == "websocket":
                self._send_json(handler, 426, {
                    "error": {
                        "type": "websocket_not_supported",
                        "message": (
                            "This model gateway is HTTP-only; use the Codex "
                            "app-server/Remote channel directly."
                        ),
                    }
                })
                return
            if handler.headers.get("Transfer-Encoding"):
                self._send_json(handler, 400, {
                    "error": {
                        "type": "invalid_request",
                        "message": "chunked request bodies are unsupported",
                    }
                })
                return
            length_text = handler.headers.get("Content-Length", "0")
            try:
                length = int(length_text)
            except ValueError:
                raise ValueError("invalid Content-Length") from None
            if length < 0:
                raise ValueError("negative Content-Length")
            body = handler.rfile.read(length) if length else b""
            response_id = _response_id_from_path(parsed.path)
            local = self._ownership.owns(response_id)
            exact_local_model_request = False
            if handler.command == "POST" and _model_required_endpoint(parsed.path):
                content_encoding = (handler.headers.get("Content-Encoding") or "identity").lower()
                if content_encoding not in {"", "identity"}:
                    raise ValueError("compressed Responses request bodies are forbidden")
                exact_local_model_request = (
                    _required_model_from_body(body) == LOCAL_MODEL_SLUG
                )
                local = exact_local_model_request
            elif (
                handler.command == "POST"
                and _optional_model_from_body(body) == LOCAL_MODEL_SLUG
            ):
                exact_local_model_request = True
                local = True
            if exact_local_model_request:
                body = sanitize_exact_local_request(body)
            is_models = (
                handler.command == "GET"
                and parsed.path.rstrip("/")
                in {"/models", "/v1/models", CLOUD_PREFIX + "/models"}
            )
            if local:
                self._proxy_local(handler, parsed, body, response_id=response_id)
            else:
                self._proxy_cloud(handler, parsed, body, augment_models=is_models)
        except ValueError as error:
            self._send_json(handler, 400, {"error": {"type": "invalid_request", "message": str(error)}})
        except LifecycleError as error:
            self._send_json(handler, 503, {
                "error": {"type": "local_engine_unavailable", "message": str(error)}
            })
        except (OSError, http.client.HTTPException) as error:
            logging.exception("gateway upstream failure")
            self._send_json(handler, 502, {
                "error": {"type": "gateway_upstream_error", "message": str(error)}
            })

    def _proxy_local(
        self,
        handler: BaseHTTPRequestHandler,
        parsed: SplitResult,
        body: bytes,
        *,
        response_id: str | None,
    ) -> None:
        lease = self.lifecycle.acquire()
        try:
            path = _local_path(parsed)
            connection = self._connection_factory(
                "local", self.config.local_host, self.config.local_port, self.config.upstream_timeout
            )
            try:
                self._exchange(
                    handler, connection, path, body, local=True, augment_models=False
                )
            except (BrokenPipeError, ConnectionResetError):
                raise
            except (OSError, http.client.HTTPException) as error:
                lease.invalidate(str(error))
                raise
        finally:
            lease.close()

    def _proxy_cloud(
        self,
        handler: BaseHTTPRequestHandler,
        parsed: SplitResult,
        body: bytes,
        *,
        augment_models: bool,
    ) -> None:
        # The destination is not derived from request headers or request-target
        # authority. HTTPConnection performs no redirect following or retry.
        connection = self._connection_factory(
            "cloud", CLOUD_HOST, CLOUD_PORT, self.config.upstream_timeout
        )
        self._exchange(
            handler,
            connection,
            _cloud_path(parsed),
            body,
            local=False,
            augment_models=augment_models,
        )

    def _exchange(
        self,
        handler: BaseHTTPRequestHandler,
        connection: http.client.HTTPConnection,
        path: str,
        body: bytes,
        *,
        local: bool,
        augment_models: bool,
    ) -> int:
        headers: dict[str, str] = {}
        connection_tokens = {
            token.strip().lower()
            for token in (handler.headers.get("Connection") or "").split(",")
            if token.strip()
        }
        for name, value in handler.headers.items():
            lowered = name.lower()
            if (
                lowered in HOP_BY_HOP_HEADERS
                or lowered in connection_tokens
                or lowered == "content-length"
            ):
                continue
            if local and lowered in LOCAL_SECRET_HEADERS:
                continue
            if local and lowered == "accept-encoding":
                continue
            if augment_models and (
                lowered == "accept-encoding"
                or lowered in CATALOG_CONDITIONAL_HEADERS
                or lowered in CATALOG_REQUEST_CACHE_HEADERS
            ):
                continue
            headers[name] = value
        if augment_models:
            headers["Accept-Encoding"] = "identity"
            headers["Cache-Control"] = "no-cache"
        elif local:
            headers["Accept-Encoding"] = "identity"
        headers["Content-Length"] = str(len(body))

        # Exactly one upstream attempt. In particular, POST is never retried or
        # replayed after a transport error or a redirect response.
        try:
            connection.request(handler.command, path, body=body, headers=headers)
            response = connection.getresponse()
            if augment_models:
                raw = response.read()
                if response.status == 304:
                    self._send_json(handler, 502, {
                        "error": {
                            "type": "gateway_catalog_error",
                            "message": (
                                "cloud returned an unusable 304 for an "
                                "unconditional model catalog request"
                            ),
                        }
                    })
                    return 502
                if response.status == 200:
                    try:
                        raw = merge_local_model(raw)
                    except (ValueError, TypeError, json.JSONDecodeError) as error:
                        raise http.client.HTTPException(
                            f"invalid cloud model catalog: {error}"
                        ) from error
                self._send_response(
                    handler, response.status, response.reason, response.getheaders(), raw,
                    local=local, augmented=True,
                )
            elif local and not self._is_event_stream(response):
                raw = response.read()
                if 200 <= response.status < 300:
                    self._remember_response_ids(raw)
                self._send_response(
                    handler, response.status, response.reason, response.getheaders(), raw,
                    local=True, augmented=False,
                )
            else:
                self._stream_response(handler, response, local=local)
            return response.status
        finally:
            connection.close()

    @staticmethod
    def _response_headers(
        source: list[tuple[str, str]],
        *,
        local: bool,
        content_length: int | None,
        augmented: bool,
    ) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        connection_tokens = {
            token.strip().lower()
            for name, value in source
            if name.lower() == "connection"
            for token in value.split(",")
            if token.strip()
        }
        for name, value in source:
            lowered = name.lower()
            if (
                lowered in HOP_BY_HOP_HEADERS
                or lowered in connection_tokens
                or lowered == "content-length"
            ):
                continue
            if local and lowered == "set-cookie":
                continue
            if augmented and (
                lowered in REPRESENTATION_VALIDATORS
                or lowered in CATALOG_RESPONSE_CACHE_HEADERS
            ):
                continue
            if not local and lowered == "location" and not Gateway._safe_cloud_location(value):
                continue
            result.append((name, value))
        if content_length is not None:
            result.append(("Content-Length", str(content_length)))
        if augmented:
            result.append(("Cache-Control", "no-store"))
        result.append(("Connection", "close"))
        return result

    def _send_response(
        self,
        handler: BaseHTTPRequestHandler,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        body: bytes,
        *,
        local: bool,
        augmented: bool,
    ) -> None:
        handler.send_response(status, reason)
        for name, value in self._response_headers(
            headers, local=local, content_length=len(body), augmented=augmented
        ):
            handler.send_header(name, value)
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(body)
        handler.close_connection = True

    def _stream_response(
        self,
        handler: BaseHTTPRequestHandler,
        response: http.client.HTTPResponse,
        *,
        local: bool,
    ) -> None:
        handler.send_response(response.status, response.reason)
        for name, value in self._response_headers(
            response.getheaders(), local=local, content_length=None, augmented=False
        ):
            handler.send_header(name, value)
        handler.end_headers()
        if handler.command != "HEAD":
            if self._is_event_stream(response):
                while True:
                    line = response.readline()
                    if not line:
                        break
                    if local and line.startswith(b"data:"):
                        self._remember_response_ids(line[5:].strip())
                    handler.wfile.write(line)
                    handler.wfile.flush()
            else:
                read_incremental = getattr(response, "read1", response.read)
                while True:
                    chunk = read_incremental(65536)
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
        handler.close_connection = True

    @staticmethod
    def _is_event_stream(response: http.client.HTTPResponse) -> bool:
        content_type = response.getheader("Content-Type", "")
        return content_type.split(";", 1)[0].strip().lower() == "text/event-stream"

    def _remember_response_ids(self, raw: bytes) -> None:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        candidates = [payload]
        response = payload.get("response")
        if isinstance(response, dict):
            candidates.append(response)
        for candidate in candidates:
            response_id = candidate.get("id")
            if isinstance(response_id, str):
                self._ownership.remember(response_id)

    @staticmethod
    def _safe_cloud_location(value: str) -> bool:
        try:
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc:
                if parsed.scheme != "https" or parsed.hostname != CLOUD_HOST:
                    return False
                if parsed.username is not None or parsed.password is not None:
                    return False
                if parsed.port not in {None, CLOUD_PORT}:
                    return False
            _safe_request_target(_with_query(parsed.path, parsed.query))
        except ValueError:
            return False
        path = parsed.path
        if not path:
            return False
        return path == CLOUD_PREFIX or path.startswith(CLOUD_PREFIX + "/")

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, status: int, value: object) -> None:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(raw)
        handler.close_connection = True


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], gateway: Gateway) -> None:
        self.gateway = gateway
        super().__init__(address, GatewayHandler)


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ninfer-huihui-codex-gateway"

    def log_message(self, format: str, *args: object) -> None:
        logging.info("http " + format, *args)

    def _handle(self) -> None:
        server = self.server
        assert isinstance(server, GatewayServer)
        try:
            server.gateway.handle(self)
        except (BrokenPipeError, ConnectionResetError):
            logging.info("gateway client disconnected during %s %s", self.command, self.path)

    do_GET = _handle
    do_POST = _handle
    do_DELETE = _handle
    do_HEAD = _handle


def default_launch_spec() -> LaunchSpec:
    repository = Path(__file__).resolve().parents[2]
    root = repository.parent
    executable = Path(os.environ.get(
        "HUIHUI_NINFER_EXE",
        repository / "build-huihui-sm86-vcpkg" / "apps" / "ninfer-serve.exe",
    ))
    artifact = Path(os.environ.get(
        "HUIHUI_NINFER_MODEL",
        root / "models" / "huihui_qwen3_8_27b_abliterated.ninfer",
    ))
    local_port = 8080
    command = (
        str(executable), str(artifact),
        "--host", "127.0.0.1", "--port", str(local_port),
        "--max-context", "65536", "--kv-capacity", "65536",
        "--max-concurrency", "1", "--max-pending-requests", "16",
        "--prefill-chunk", "1024", "--kv-dtype", "rk8v4",
        "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
        "--context-lookup", "--context-lookup-verify-tokens", "4",
        "--vision", "--no-cuda-graph",
        "--request-log-jsonl", str(root / "logs" / "huihui-gateway.requests.jsonl"),
    )
    return LaunchSpec(
        command=command,
        cwd=executable.parent,
        required_files=(artifact,),
        port=local_port,
        startup_timeout=float(os.environ.get("HUIHUI_STARTUP_TIMEOUT_SECONDS", "180")),
        idle_timeout=float(os.environ.get("HUIHUI_IDLE_TIMEOUT_SECONDS", "600")),
        log_dir=root / "logs",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = default_launch_spec()
    config = GatewayConfig(
        port=int(os.environ.get("HUIHUI_GATEWAY_PORT", "8081")),
        local_port=spec.port,
    )
    lifecycle = EngineLifecycle(spec, LOCAL_MODEL_SLUG)
    ownership = LocalResponseOwnership(spec.log_dir / "huihui-local-response-ids.txt")
    server = GatewayServer(
        (config.host, config.port), Gateway(config, lifecycle, ownership=ownership)
    )
    logging.info("Huihui Codex gateway listening on http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("stopping Huihui Codex gateway")
    finally:
        server.server_close()
        lifecycle.close()
    return 0
