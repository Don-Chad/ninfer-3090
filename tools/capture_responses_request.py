"""One-shot localhost Responses request capture for compatibility diagnostics."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            # Installed Codex may probe the Responses WebSocket before its
            # documented HTTP fallback. Keep the one-shot capture alive for
            # the subsequent POST instead of consuming it with that probe.
            self.send_response(426)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(raw)
            message = json.dumps(
                {"error": {"type": "capture_complete", "message": "request captured"}}
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    while not args.output.exists():
        server.handle_request()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
