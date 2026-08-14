"""Matched RTX 3090 ReplaySSM cohort sweep; edit constants below."""

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
OUTPUT_ROOT = ROOT / "benchmark_results/qwen3_8_replayssm_cohort_k3_1024"
COHORTS = (1, 2, 4, 8)
PORT = 8093
MAX_CONTEXT = 8192
BASE_KV_CAPACITY = 8192
OUTPUT_TOKENS = 1024
PROMPTS = (
    "Write a self-contained 700 to 900 word technical brief for an engineering team. Explain how to design a robust GPU inference batching service, covering architecture, backpressure, failure handling, testing, and latency-throughput tradeoffs. Do not include full source code.",
    "Write a detailed guide to optimizing CUDA inference on a memory-constrained consumer GPU, including measurement methodology.",
    "Analyze the benefits and risks of speculative decoding and explain how acceptance rate affects real end-to-end throughput.",
    "Create a technical proposal for a local OpenAI-compatible model server with bounded concurrency and predictable memory usage.",
    "Explain recurrent-state memory in hybrid transformer models and compare snapshotting with replay-based speculative transactions.",
    "Develop a practical benchmark plan for comparing inference runtimes fairly across latency, throughput, memory, and output quality.",
    "Write an implementation-oriented explanation of paged KV caches, prefix reuse, and concurrent decode scheduling.",
    "Propose a production checklist for releasing a native Windows CUDA inference runtime to non-expert users.",
)
STARTUP_TIMEOUT_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 300


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
            "messages": [{"role": "user", "content": PROMPTS[index]}],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "seed": 57004 + index,
            "reasoning_effort": "medium",
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
    kv_capacity = 16384 if cohort == 8 else BASE_KV_CAPACITY
    command = [
        str(SERVER), str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", str(MAX_CONTEXT), "--kv-capacity", str(kv_capacity),
        "--max-concurrency", str(cohort), "--prefill-chunk", "1024", "--kv-dtype", "int8",
        "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft", "--greedy",
        "--no-prefix-reuse", "--request-log-jsonl", str(request_log),
    ]
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (
        out / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(process)
            peak_mib = gpu_used_mib()
            stop_poll = threading.Event()

            def poll_gpu() -> None:
                nonlocal peak_mib
                while not stop_poll.wait(0.1):
                    peak_mib = max(peak_mib, gpu_used_mib())

            poller = threading.Thread(target=poll_gpu, daemon=True)
            poller.start()
            barrier = threading.Barrier(cohort + 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=cohort) as executor:
                futures = [executor.submit(post_chat, index, barrier) for index in range(cohort)]
                barrier.wait()
                started = time.perf_counter()
                clients = [future.result() for future in futures]
                makespan = time.perf_counter() - started
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
    generated = sum(record["result"]["completion_tokens"] for record in done)
    drafted = sum(record["speculative"]["drafted_tokens"] for record in done)
    accepted = sum(record["speculative"]["accepted_tokens"] for record in done)
    rounds = sum(record["speculative"]["rounds"] for record in done)
    result = {
        "cohort": cohort,
        "requests": len(done),
        "generated_tokens": generated,
        "wave_seconds": makespan,
        "end_to_end_tokens_per_second": generated / makespan,
        "decode_tokens_per_second": generated / max(r["timings_seconds"]["decode"] for r in done),
        "acceptance_percent": 100 * accepted / drafted,
        "committed_tokens_per_round": generated / rounds,
        "mean_ttft_ms": 1000 * sum(r["timings_seconds"]["ttft"] for r in done) / len(done),
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
