"""Matched Qwen3.8 ordinary/K2/K3 benchmark with exact output comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "build-sm86-qwen38/apps/Release/ninfer-serve.exe"
MODEL = ROOT.parent / "qwen3_8_27b.ninfer"
OUTPUT_ROOT = ROOT / "benchmark_results/qwen3_8_round_06_mtp_k0_k2_k3"
PORT = 8093
MODES = (("k0", 0), ("k2", 2), ("k3", 3))
MAX_CONTEXT = 4096
OUTPUT_TOKENS = 256
PROMPTS = (
    "Write a Python function that merges overlapping integer intervals. Explain complexity.",
    "Return JSON describing three planets with name, type, and one notable property.",
    "Explain why batching can improve GPU inference throughput and when it hurts latency.",
    "A shop discounts an 80 euro item by 15 percent, then adds 21 percent VAT. Calculate the final price step by step.",
)
EXTRA_SERVER_ARGS: tuple[str, ...] = ()


def wait_ready(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError("server health timeout")


def request(prompt: str) -> dict:
    body = json.dumps(
        {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "seed": 57004,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def run_mode(name: str, draft_tokens: int) -> dict:
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True)
    request_log = out / "requests.jsonl"
    command = [
        str(SERVER), str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
        "--max-context", str(MAX_CONTEXT), "--kv-capacity", str(MAX_CONTEXT),
        "--max-concurrency", "1", "--prefill-chunk", "1024", "--kv-dtype", "int8",
        "--greedy", "--no-prefix-reuse", "--request-log-jsonl", str(request_log),
    ]
    if draft_tokens:
        command.extend(["--spec", "mtp", "--draft-tokens", str(draft_tokens), "--lm-head-draft"])
    command.extend(EXTRA_SERVER_ARGS)
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(process)
            responses = [request(prompt) for prompt in PROMPTS]
        finally:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)
    texts = []
    for response in responses:
        message = response["choices"][0]["message"]
        texts.append(
            (message.get("reasoning_content") or "") + "\0" + (message.get("content") or "")
        )
    (out / "responses.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")
    records = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    done = [record for record in records if record.get("event") == "request_done"]
    generated = sum(record["result"]["completion_tokens"] for record in done)
    decode_seconds = sum(record["timings_seconds"]["decode"] for record in done)
    result = {
        "mode": name,
        "requests": len(done),
        "generated_tokens": generated,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": generated / decode_seconds,
        "mean_ttft_ms": 1000 * sum(record["timings_seconds"]["ttft"] for record in done) / len(done),
        "output_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
    }
    if draft_tokens:
        drafted = sum(record["speculative"]["drafted_tokens"] for record in done)
        accepted = sum(record["speculative"]["accepted_tokens"] for record in done)
        rounds = sum(record["speculative"]["rounds"] for record in done)
        result.update(
            {
                "drafted_tokens": drafted,
                "accepted_tokens": accepted,
                "acceptance_percent": 100 * accepted / drafted,
                "committed_tokens_per_round": generated / rounds,
            }
        )
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    results = [run_mode(name, k) for name, k in MODES]
    baseline_hashes = results[0]["output_sha256"]
    exact = {result["mode"]: result["output_sha256"] == baseline_hashes for result in results}
    summary = {"results": results, "exact_output_match_to_k0": exact}
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
