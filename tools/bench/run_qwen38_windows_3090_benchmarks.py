"""C1/C2/C4/C8 RTX 3090 prefill and 1K-generation benchmark configured by the companion BAT."""

from __future__ import annotations

import concurrent.futures
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
COHORTS = tuple(int(value) for value in os.environ.get("NINFER_BENCH_COHORTS", "1,2,4,8").split(","))
KV_DTYPE = os.environ.get("NINFER_BENCH_KV_DTYPE", "int8")
PORT = 8093
STARTUP_TIMEOUT_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 900
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = ROOT / "benchmark_results" / f"windows_3090_c1_c8_{RUN_ID}"

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


def request(prompt: str, max_tokens: int, index: int, barrier: threading.Barrier) -> tuple[dict, float]:
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt + f"\nRequest number: {index + 1}."}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 57004 + index,
        "reasoning_effort": "medium",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response), time.perf_counter() - started


def run_round(name: str, prompt: str, max_tokens: int, cohort: int) -> dict:
    out = OUTPUT_ROOT / name / f"c{cohort}"
    out.mkdir(parents=True)
    request_log = out / "requests.jsonl"
    context_per_request = MAX_CONTEXT // cohort
    if cohort == 8:
        # MTP3 C8 needs the qualified 8K lane envelope on a 24 GB RTX 3090.
        context_per_request = min(context_per_request, 8192)
    kv_capacity = min(MAX_CONTEXT, context_per_request * cohort)
    if context_per_request < max_tokens + 1024:
        raise ValueError(f"C{cohort} context {context_per_request} is too small for this round")
    command = [
        str(SERVER), str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", str(context_per_request), "--kv-capacity", str(kv_capacity),
        "--max-concurrency", str(cohort), "--max-pending-requests", "8",
        "--pending-timeout-ms", "900000", "--prefill-chunk", "512",
        "--kv-dtype", KV_DTYPE, "--spec", "mtp", "--draft-tokens", "3",
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
            warmup_barrier = threading.Barrier(1)
            request("Warm up the inference path with a short response.", 32, -1, warmup_barrier)
            peak_mib = gpu_used_mib()
            poller = threading.Thread(target=poll_gpu, daemon=True)
            poller.start()
            barrier = threading.Barrier(cohort + 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=cohort) as executor:
                futures = [executor.submit(request, prompt, max_tokens, index, barrier) for index in range(cohort)]
                barrier.wait()
                wave_started = time.perf_counter()
                clients = [future.result() for future in futures]
                wall_seconds = time.perf_counter() - wave_started
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

    (out / "responses.json").write_text(json.dumps([client[0] for client in clients], indent=2), encoding="utf-8")
    records = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    done = [record for record in records if record.get("event") == "request_done"]
    if len(done) != cohort + 1:
        raise RuntimeError(
            f"{name} C{cohort}: expected one warmup plus {cohort} measured requests, found {len(done)}"
        )
    done = done[-cohort:]
    generated = sum(record["result"]["completion_tokens"] for record in done)
    prompt_tokens = sum(record["result"]["prompt_tokens"] for record in done)
    computed_prefill = sum(record["result"]["computed_prefill_tokens"] for record in done)
    prefill_seconds = sum(record["timings_seconds"]["prefill"] for record in done)
    decode_seconds = max(record["timings_seconds"]["decode"] for record in done)
    drafted = sum(record["speculative"]["drafted_tokens"] for record in done)
    accepted = sum(record["speculative"]["accepted_tokens"] for record in done)
    result = {
        "round": name,
        "cohort": cohort,
        "max_context_per_request": context_per_request,
        "shared_kv_capacity": kv_capacity,
        "prompt_tokens": prompt_tokens,
        "computed_prefill_tokens": computed_prefill,
        "completion_tokens": generated,
        "wall_seconds": wall_seconds,
        "prefill_tokens_per_second": computed_prefill / prefill_seconds,
        "end_to_end_output_tokens_per_second": generated / wall_seconds,
        "decode_tokens_per_second": generated / decode_seconds,
        "mtp_acceptance_percent": 100 * accepted / drafted if drafted else 0,
        "ttft_ms": 1000 * sum(record["timings_seconds"]["ttft"] for record in done) / cohort,
        "peak_gpu_memory_mib": peak_mib,
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"{name}: invalid {key}: {value}")
    if name == "generation_1k" and generated != cohort * OUTPUT_TOKENS:
        raise RuntimeError(
            f"generation_1k C{cohort}: generated {generated} of {cohort * OUTPUT_TOKENS} required tokens"
        )
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if MAX_CONTEXT < OUTPUT_TOKENS + 1024:
        raise ValueError("MAX_CONTEXT is too small for the requested output")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    configuration = {
        "server": str(SERVER), "model": str(MODEL), "max_context": MAX_CONTEXT,
        "output_tokens": OUTPUT_TOKENS, "prefill_prompt_characters": PREFILL_PROMPT_CHARACTERS,
        "mtp_draft_tokens": 3, "cohorts": COHORTS, "kv_dtype": KV_DTYPE,
    }
    (OUTPUT_ROOT / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    generation_prompt = "Write a detailed technical guide to reliable local GPU inference. Continue until the requested output limit."
    prefill_prompt = (PARAGRAPH * (PREFILL_PROMPT_CHARACTERS // len(PARAGRAPH) + 2))[:PREFILL_PROMPT_CHARACTERS]
    results = []
    for cohort in COHORTS:
        results.append(run_round("generation_1k", generation_prompt, OUTPUT_TOKENS, cohort))
    for cohort in COHORTS:
        results.append(run_round("prefill", prefill_prompt, 16, cohort))
    generation_by_cohort = {
        result["cohort"]: result for result in results if result["round"] == "generation_1k"
    }
    prefill_by_cohort = {
        result["cohort"]: result for result in results if result["round"] == "prefill"
    }
    headline_results = [
        {
            "cohort": cohort,
            "prompt_tokens_per_second": prefill_by_cohort[cohort]["prefill_tokens_per_second"],
            "decode_tokens_per_second": generation_by_cohort[cohort]["decode_tokens_per_second"],
        }
        for cohort in COHORTS
    ]
    (OUTPUT_ROOT / "scores.json").write_text(json.dumps(headline_results, indent=2), encoding="utf-8")
    score_lines = ["cohort,prompt_tok_s,decode_tok_s"]
    for result in headline_results:
        score_lines.append(
            f"C{result['cohort']},{result['prompt_tokens_per_second']:.2f},"
            f"{result['decode_tokens_per_second']:.2f}"
        )
    (OUTPUT_ROOT / "scores.csv").write_text("\n".join(score_lines) + "\n", encoding="utf-8")
    print("\nHeadline scores (server compute time):")
    print("Cohort   Prompt tok/s   Decode tok/s")
    for result in headline_results:
        print(
            f"C{result['cohort']:<7} "
            f"{result['prompt_tokens_per_second']:>12.2f} "
            f"{result['decode_tokens_per_second']:>14.2f}"
        )
    print(f"Scores saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
