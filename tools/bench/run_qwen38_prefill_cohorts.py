"""RTX 3090 Qwen3.8 long-prefill cohort benchmark (edit constants below)."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "build-sm86-replayssm/apps/Release/ninfer-serve.exe"
MODEL = ROOT.parent / "qwen3_8_27b.ninfer"
OUTPUT_ROOT = ROOT / "benchmark_results/qwen3_8_prefill_cohorts_4362_chunk512"
COHORTS = (1, 2, 4, 8)
PORT = 8093
MAX_CONTEXT = 8192
PROMPT_CHARACTERS = 28000
OUTPUT_TOKENS = 16
STARTUP_TIMEOUT_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 180

PARAGRAPH = (
    "A reliable GPU inference service separates admission control, prompt ingestion, decode "
    "scheduling, memory accounting, observability, and failure recovery. It measures latency "
    "and throughput independently, bounds concurrency, avoids hidden cache reuse, and records "
    "enough metadata to reproduce every result. Engineers validate correctness before tuning "
    "kernels and preserve sufficient memory headroom for transient workspaces. "
)
# The server log is the authority for the exact tokenizer count. With the current Qwen3.8 template,
# this construction produces 4,362 prompt tokens per request.
PROMPT = (PARAGRAPH * ((PROMPT_CHARACTERS // len(PARAGRAPH)) + 2))[:PROMPT_CHARACTERS]


def wait_ready(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError("server health timeout")


def post_chat(index: int, barrier: threading.Barrier) -> dict:
    body = json.dumps(
        {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": PROMPT + f"\nRequest {index}: summarize the operational priorities."}],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "seed": 58000 + index,
            "reasoning_effort": "none",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        result = json.load(response)
    return {"wall_seconds": time.perf_counter() - started, "response": result}


def gpu_used_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def run_cohort(cohort: int) -> dict:
    out = OUTPUT_ROOT / f"c{cohort}"
    out.mkdir(parents=True, exist_ok=False)
    request_log = out / "requests.jsonl"
    command = [
        str(SERVER), str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", str(MAX_CONTEXT), "--kv-capacity", str(cohort * MAX_CONTEXT),
        "--max-concurrency", str(cohort), "--max-pending-requests", "16",
        "--pending-timeout-ms", "120000", "--prefill-chunk", "512", "--kv-dtype", "int8",
        "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft", "--greedy",
        "--no-prefix-reuse", "--request-log-jsonl", str(request_log),
    ]
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (out / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            wait_ready(process)
            peak_mib = gpu_used_mib()
            stop_poll = threading.Event()

            def poll_gpu() -> None:
                nonlocal peak_mib
                while not stop_poll.wait(0.05):
                    peak_mib = max(peak_mib, gpu_used_mib())

            poller = threading.Thread(target=poll_gpu, daemon=True)
            poller.start()
            barrier = threading.Barrier(cohort + 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=cohort) as executor:
                futures = [executor.submit(post_chat, index, barrier) for index in range(cohort)]
                barrier.wait()
                wave_started = time.perf_counter()
                clients = [future.result() for future in futures]
                makespan = time.perf_counter() - wave_started
            stop_poll.set()
            poller.join()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    (out / "responses.json").write_text(json.dumps(clients, indent=2), encoding="utf-8")
    records = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    done = [record for record in records if record.get("event") == "request_done"]
    prompt_tokens = sum(record["result"]["computed_prefill_tokens"] for record in done)
    max_prefill_seconds = max(record["timings_seconds"]["prefill"] for record in done)
    result = {
        "cohort": cohort,
        "requests": len(done),
        "prompt_tokens_per_request": [record["result"]["prompt_tokens"] for record in done],
        "total_computed_prefill_tokens": prompt_tokens,
        "wave_seconds": makespan,
        "aggregate_prefill_tokens_per_second": prompt_tokens / max_prefill_seconds,
        "end_to_end_prompt_tokens_per_second": prompt_tokens / makespan,
        "mean_server_prefill_ms": 1000 * sum(record["timings_seconds"]["prefill"] for record in done) / len(done),
        "mean_ttft_ms": 1000 * sum(record["timings_seconds"]["ttft"] for record in done) / len(done),
        "peak_gpu_memory_mib": peak_mib,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    results = [run_cohort(cohort) for cohort in COHORTS]
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
