"""Fixed RTX 3090 Qwen3.8 concurrency benchmark (edit constants below)."""

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
OUTPUT_ROOT = ROOT / "benchmark_results/qwen3_8_round_12_replayssm_c8_k3_8k_shared"
CONCURRENCY_LEVELS = (8,)
PORT = 8093
MAX_CONTEXT = 8192
KV_CAPACITY = 8192
OUTPUT_TOKENS = 128
PROMPT = "Explain three practical ways to make a CUDA inference server faster, with technical detail."
STARTUP_TIMEOUT_SECONDS = 45
REQUEST_TIMEOUT_SECONDS = 120


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=2) as response:
        return json.load(response)


def post_chat(seed: int, barrier: threading.Barrier) -> dict:
    body = json.dumps(
        {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "seed": seed,
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
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def run_level(concurrency: int) -> dict:
    out = OUTPUT_ROOT / f"c{concurrency}"
    out.mkdir(parents=True, exist_ok=False)
    request_log = out / "requests.jsonl"
    command = [
        str(SERVER),
        str(MODEL),
        "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", str(MAX_CONTEXT), "--kv-capacity", str(KV_CAPACITY),
        "--max-concurrency", str(concurrency), "--prefill-chunk", "1024",
        "--kv-dtype", "int8", "--spec", "mtp", "--draft-tokens", "3",
        "--lm-head-draft", "--greedy", "--no-prefix-reuse",
        "--request-log-jsonl", str(request_log),
    ]
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (
        out / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup with status {process.returncode}")
                try:
                    get_json("/health")
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                raise TimeoutError("server health timeout")

            peak_mib = gpu_used_mib()
            stop_poll = threading.Event()

            def poll_gpu() -> None:
                nonlocal peak_mib
                while not stop_poll.wait(0.05):
                    peak_mib = max(peak_mib, gpu_used_mib())

            poller = threading.Thread(target=poll_gpu, daemon=True)
            poller.start()
            barrier = threading.Barrier(concurrency + 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(post_chat, 57004 + index, barrier) for index in range(concurrency)]
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

    records = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    done = [record for record in records if record.get("event") == "request_done"]
    generated = sum(record["result"]["completion_tokens"] for record in done)
    summary = {
        "concurrency": concurrency,
        "requests_completed": len(done),
        "generated_tokens": generated,
        "wave_makespan_seconds": makespan,
        "aggregate_end_to_end_tokens_per_second": generated / makespan,
        "mean_client_wall_seconds": sum(client["wall_seconds"] for client in clients) / len(clients),
        "mean_ttft_ms": 1000 * sum(record["timings_seconds"]["ttft"] for record in done) / len(done),
        "aggregate_decode_tokens_per_second": generated / max(record["timings_seconds"]["decode"] for record in done),
        "mtp_acceptance_percent": 100
        * sum(record["speculative"]["accepted_tokens"] for record in done)
        / sum(record["speculative"]["drafted_tokens"] for record in done),
        "peak_gpu_memory_mib": peak_mib,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    summaries = [run_level(concurrency) for concurrency in CONCURRENCY_LEVELS]
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
