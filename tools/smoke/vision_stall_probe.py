"""Measures how long a text lane stalls while an image is encoded.

Streams one long text completion and, once it is producing tokens, sends an
image request on a second lane. Reports the inter-token gaps of the text
stream: with a KV-funded window the encode runs beside the decode and the gaps
stay small, with a weight-tail window the whole engine waits for it.
Stdlib only; reuses the deterministic gradient image of overlay_ab.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import threading
import time
import urllib.request


def gradient_bmp(width: int, height: int) -> bytes:
    """Uncompressed 24-bit BMP with a deterministic gradient."""
    row_bytes = (width * 3 + 3) & ~3
    image_bytes = row_bytes * height
    header = struct.pack(
        "<2sIHHIIiiHHIIiiII",
        b"BM", 54 + image_bytes, 0, 0, 54,
        40, width, height, 1, 24, 0, image_bytes, 2835, 2835, 0, 0,
    )
    rows = bytearray()
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row += bytes(((x * 7 + y * 3) % 256, (x + 2 * y) % 256, (255 - (x // 8 + y // 8)) % 256))
        row += b"\x00" * (row_bytes - len(row))
        rows += row
    return header + bytes(rows)


def post(base_url: str, payload: dict, timeout: float = 900.0):
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def stream_gaps(base_url: str, model: str, prompt: str, max_tokens: int,
                started_event: threading.Event, out: dict) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": True,
    }
    gaps = []
    previous = None
    tokens = 0
    with post(base_url, payload) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if not delta.get("content"):
                continue
            now = time.monotonic()
            tokens += 1
            if previous is not None:
                gaps.append(now - previous)
            previous = now
            if tokens == 8:
                started_event.set()
    out["tokens"] = tokens
    out["gaps"] = gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18099")
    parser.add_argument("--model", default="m")
    parser.add_argument("--label", default="probe")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--text-tokens", type=int, default=256)
    args = parser.parse_args()

    data_uri = "data:image/bmp;base64," + base64.b64encode(
        gradient_bmp(args.width, args.height)).decode()
    image_payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Describe the colors and structure of this image in one sentence."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        "max_tokens": 48,
        "temperature": 0,
        "enable_thinking": False,
    }

    started = threading.Event()
    text_out: dict = {}
    text_thread = threading.Thread(
        target=stream_gaps,
        args=(args.base_url, args.model,
              "Write a detailed essay about the history of cartography, in full sentences.",
              args.text_tokens, started, text_out))
    text_thread.start()
    if not started.wait(timeout=120):
        print(f"[{args.label}] text stream never started")
        text_thread.join()
        return 1

    image_started = time.monotonic()
    with post(args.base_url, image_payload) as response:
        image = json.loads(response.read())
    image_wall = time.monotonic() - image_started
    text_thread.join()

    gaps = sorted(text_out.get("gaps", []))
    if not gaps:
        print(f"[{args.label}] text stream produced no gaps")
        return 1
    worst = gaps[-1]
    median = gaps[len(gaps) // 2]
    content = image["choices"][0]["message"]["content"]
    print(f"[{args.label}] text_tokens={text_out['tokens']} "
          f"gap_max={worst * 1000:.0f}ms gap_p50={median * 1000:.0f}ms "
          f"gap_p99={gaps[int(len(gaps) * 0.99)] * 1000:.0f}ms")
    print(f"[{args.label}] image_wall={image_wall:.2f}s "
          f"prompt={image.get('usage', {}).get('prompt_tokens')}")
    print(f"[{args.label}] image_text={json.dumps(content, ensure_ascii=False)[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
