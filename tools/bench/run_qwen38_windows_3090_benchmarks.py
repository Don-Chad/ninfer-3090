"""C1 RTX 3090 prefill and 1K-generation benchmark configured by the companion BAT file."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(os.environ.get("NINFER_BENCH_SERVER", ROOT / "build-sm86-replayssm/apps/Release/ninfer-serve.exe"))
MODEL = Path(os.environ.get("NINFER_BENCH_MODEL", ROOT.parent / "qwen3_8_27b.ninfer"))
MAX_CONTEXT = int(os.environ.get("NINFER_BENCH_MAX_CONTEXT", "65536"))
OUTPUT_TOKENS = int(os.environ.get("NINFER_BENCH_OUTPUT_TOKENS", "1024"))
PREFILL_PROMPT_CHARACTERS = int(os.environ.get("NINFER_BENCH_PREFILL_CHARS", "28000"))
PORT = 8093
STARTUP_TIMEOUT_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 900
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = ROOT / "benchmark_results" / f"windows_3090_c1_{RUN_ID}"

PARAGRAPH = (
    "A reliable GPU inference service separates admission control, prompt ingestion, decode "
    "scheduling, memory accounting, observability, and failure recovery. It measures latency "
    "and throughput independently, bounds concurrency, avoids hidden cache reuse, and records "
    "enough metadata to reproduce every result. Engineers validate correctness before tuning "
    "kernels and preserve memory headroom for transient workspaces. "
)


def wait_ready(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("server health timeout")


def gpu_used_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def request(prompt: str, max_tokens: int) -> tuple[dict, float]:
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 57004,
        "reasoning_effort": "medium",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response), time.perf_counter() - started


def run_round(name: str, prompt: str, max_tokens: int) -> dict:
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True)
    request_log = out / "requests.jsonl"
    command = [
        str(SERVER), str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", str(MAX_CONTEXT), "--kv-capacity", str(MAX_CONTEXT),
        "--max-concurrency", "1", "--max-pending-requests", "4",
        "--pending-timeout-ms", "900000", "--prefill-chunk", "1024",
        "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3",
        "--lm-head-draft", "--greedy", "--no-prefix-reuse",
        "--request-log-jsonl", str(request_log),
    ]
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (out / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        stop = threading.Event()
        peak_mib = 0

        def poll_gpu() -> None:
            nonlocal peak_mib
            while not stop.wait(0.1):
                peak_mib = max(peak_mib, gpu_used_mib())

        try:
            wait_ready(process)
            peak_mib = gpu_used_mib()
            poller = threading.Thread(target=poll_gpu, daemon=True)
            poller.start()
            response, wall_seconds = request(prompt, max_tokens)
            stop.set()
            poller.join()
        finally:
            stop.set()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    (out / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    records = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    done = [record for record in records if record.get("event") == "request_done"]
    if len(done) != 1:
        raise RuntimeError(f"{name}: expected one completed request, found {len(done)}")
    record = done[0]
    generated = record["result"]["completion_tokens"]
    prompt_tokens = record["result"]["prompt_tokens"]
    computed_prefill = record["result"]["computed_prefill_tokens"]
    prefill_seconds = record["timings_seconds"]["prefill"]
    decode_seconds = record["timings_seconds"]["decode"]
    drafted = record["speculative"]["drafted_tokens"]
    accepted = record["speculative"]["accepted_tokens"]
    result = {
        "round": name,
        "max_context": MAX_CONTEXT,
        "prompt_tokens": prompt_tokens,
        "computed_prefill_tokens": computed_prefill,
        "completion_tokens": generated,
        "wall_seconds": wall_seconds,
        "prefill_tokens_per_second": computed_prefill / prefill_seconds,
        "end_to_end_output_tokens_per_second": generated / wall_seconds,
        "decode_tokens_per_second": generated / decode_seconds,
        "mtp_acceptance_percent": 100 * accepted / drafted if drafted else 0,
        "ttft_ms": 1000 * record["timings_seconds"]["ttft"],
        "peak_gpu_memory_mib": peak_mib,
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"{name}: invalid {key}: {value}")
    if name == "generation_1k" and generated != OUTPUT_TOKENS:
        raise RuntimeError(f"generation_1k: generated {generated} of {OUTPUT_TOKENS} required tokens")
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if MAX_CONTEXT < OUTPUT_TOKENS + 1024:
        raise ValueError("MAX_CONTEXT is too small for the requested output")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    configuration = {
        "server": str(SERVER), "model": str(MODEL), "max_context": MAX_CONTEXT,
        "output_tokens": OUTPUT_TOKENS, "prefill_prompt_characters": PREFILL_PROMPT_CHARACTERS,
        "mtp_draft_tokens": 3,
    }
    (OUTPUT_ROOT / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    generation_prompt = "Write a detailed technical guide to reliable local GPU inference. Continue until the requested output limit."
    prefill_prompt = (PARAGRAPH * (PREFILL_PROMPT_CHARACTERS // len(PARAGRAPH) + 2))[:PREFILL_PROMPT_CHARACTERS]
    results = [
        run_round("generation_1k", generation_prompt, OUTPUT_TOKENS),
        run_round("prefill", prefill_prompt, 16),
    ]
    (OUTPUT_ROOT / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    score_lines = ["round,prompt_tokens,completion_tokens,prefill_tok_s,e2e_output_tok_s,decode_tok_s,mtp_acceptance_pct,ttft_ms,peak_vram_mib"]
    for result in results:
        score_lines.append(
            f"{result['round']},{result['prompt_tokens']},{result['completion_tokens']},"
            f"{result['prefill_tokens_per_second']:.2f},{result['end_to_end_output_tokens_per_second']:.2f},"
            f"{result['decode_tokens_per_second']:.2f},{result['mtp_acceptance_percent']:.2f},"
            f"{result['ttft_ms']:.0f},{result['peak_gpu_memory_mib']}"
        )
    (OUTPUT_ROOT / "scores.csv").write_text("\n".join(score_lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Scores saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
