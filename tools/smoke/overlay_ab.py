"""Resident-vs-overlay A/B driver for the vision VRAM overlay.

Talks to a running ninfer-serve over the OpenAI chat route with one generated
test image, greedy sampling, and prints the exact completion text plus usage so
two runs can be diffed byte-for-byte. Stdlib only.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import struct
import sys
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


def chat(base_url: str, model: str, messages: list, max_tokens: int, timeout: float = 600.0) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
    }).encode()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    body["_wall_seconds"] = round(time.monotonic() - started, 3)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18099")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--label", default="run")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--concurrency-probe", action="store_true",
                        help="run two text requests in parallel with the image request")
    args = parser.parse_args()

    image = gradient_bmp(args.width, args.height)
    data_uri = "data:image/bmp;base64," + base64.b64encode(image).decode()
    image_messages = [{
        "role": "user",
        "content": [
            {"type": "text",
             "text": "Describe the colors and structure of this image in one sentence."},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }]

    def text_request(tag: str) -> dict:
        return chat(args.base_url, args.model,
                    [{"role": "user", "content": f"Count from 1 to 5. ({tag})"}], 32)

    results = {}
    if args.concurrency_probe:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            image_future = pool.submit(chat, args.base_url, args.model, image_messages, 64)
            text_a = pool.submit(text_request, "a")
            text_b = pool.submit(text_request, "b")
            results["image"] = image_future.result()
            results["text_a"] = text_a.result()
            results["text_b"] = text_b.result()
    else:
        results["image"] = chat(args.base_url, args.model, image_messages, 64)

    # Follow-up turn over the same image: prefix reuse must avoid a re-encode.
    followup = image_messages + [
        {"role": "assistant",
         "content": results["image"]["choices"][0]["message"]["content"]},
        {"role": "user", "content": "Now answer with exactly one word: what is the dominant hue?"},
    ]
    results["followup"] = chat(args.base_url, args.model, followup, 16)

    for name, body in results.items():
        message = body["choices"][0]["message"]
        usage = body.get("usage", {})
        print(f"[{args.label}:{name}] wall={body['_wall_seconds']}s "
              f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
        print(f"[{args.label}:{name}] text={json.dumps(message.get('content', ''), ensure_ascii=False)}")
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning:
            print(f"[{args.label}:{name}] reasoning={json.dumps(reasoning[:600], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
