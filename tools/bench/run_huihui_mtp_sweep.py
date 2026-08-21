"""Matched RTX 3090 MTP-depth sweep for the exact Huihui Qwen3.8 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_SLUG = "huihui-qwen3.8-27b-abliterated"
PROMPTS = (
    "Write a Python function that merges overlapping integer intervals. Explain complexity.",
    "Return JSON describing three planets with name, type, and one notable property.",
    "Explain why batching improves GPU inference throughput and when it hurts latency.",
    "An 80 euro item is discounted 15 percent, then taxed 21 percent. Calculate step by step.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serve",
        type=Path,
        default=ROOT / "build-huihui-sm86-vcpkg" / "apps" / "ninfer-serve.exe",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT.parent / "models" / "huihui_qwen3_8_27b_abliterated.ninfer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_results" /
        f"huihui_rtx3090_mtp_sweep_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument("--draft-depths", default="0,1,2,3,4,5")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--port", type=int, default=8093)
    return parser.parse_args()


def wait_ready(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                payload = json.load(response)
            if any(item.get("id") == MODEL_SLUG for item in payload.get("data", [])):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("exact Huihui model identity did not become ready")


def request(port: int, prompt: str, output_tokens: int) -> dict:
    body = json.dumps({
        "model": MODEL_SLUG,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": 57004,
        "enable_thinking": False,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def run_mode(args: argparse.Namespace, depth: int) -> dict:
    name = f"k{depth}"
    destination = args.output / name
    destination.mkdir(parents=True)
    request_log = destination / "requests.jsonl"
    command = [
        str(args.serve), str(args.artifact),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-context", "4096", "--kv-capacity", "4096",
        "--max-concurrency", "1", "--max-pending-requests", "4",
        "--prefill-chunk", "1024", "--kv-dtype", "rk8v4",
        "--greedy", "--no-prefix-reuse", "--vision", "--no-cuda-graph",
        "--request-log-jsonl", str(request_log),
    ]
    if depth:
        command += ["--spec", "mtp", "--draft-tokens", str(depth), "--lm-head-draft"]
    (destination / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (destination / "stdout.log").open("wb") as stdout, \
            (destination / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=args.serve.parent,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(process, args.port)
            responses = [request(args.port, prompt, args.output_tokens) for prompt in PROMPTS]
        finally:
            process.terminate()
            try:
                process.wait(20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(20)

    (destination / "responses.json").write_text(
        json.dumps(responses, indent=2), encoding="utf-8"
    )
    records = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    done = [record for record in records if record.get("event") == "request_done"]
    generated = sum(record["result"]["completion_tokens"] for record in done)
    committed_decode = generated - len(done)
    decode_seconds = sum(record["timings_seconds"]["decode"] for record in done)
    texts = [
        (response["choices"][0]["message"].get("reasoning_content") or "") + "\0" +
        (response["choices"][0]["message"].get("content") or "")
        for response in responses
    ]
    result = {
        "mode": name,
        "requests": len(done),
        "completion_tokens": generated,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": committed_decode / decode_seconds,
        "mean_ttft_ms": 1000 * sum(
            record["timings_seconds"]["ttft"] for record in done
        ) / len(done),
        "output_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
    }
    if depth:
        drafted = sum(record["speculative"]["drafted_tokens"] for record in done)
        accepted = sum(record["speculative"]["accepted_tokens"] for record in done)
        rounds = sum(record["speculative"]["rounds"] for record in done)
        result |= {
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_percent": 100 * accepted / drafted,
            "committed_tokens_per_round": committed_decode / rounds,
        }
    (destination / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if not args.serve.is_file() or not args.artifact.is_file():
        raise FileNotFoundError("serve binary or exact Huihui artifact is missing")
    depths = tuple(int(value) for value in args.draft_depths.split(","))
    if not depths or any(depth < 0 or depth > 5 for depth in depths):
        raise ValueError("draft depths must be between 0 and 5")
    args.output.mkdir(parents=True, exist_ok=False)
    results = [run_mode(args, depth) for depth in depths]
    baseline = next((result["output_sha256"] for result in results if result["mode"] == "k0"), None)
    summary = {
        "model": MODEL_SLUG,
        "artifact": str(args.artifact.resolve()),
        "results": results,
        "exact_output_match_to_k0": {
            result["mode"]: baseline is not None and result["output_sha256"] == baseline
            for result in results
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
