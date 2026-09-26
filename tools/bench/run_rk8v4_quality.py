import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"G:\python\custom-kernel-3090\infer-qwen38-replayssm-sm86")
EXE = ROOT / "build-sm86-replayssm" / "apps" / "Release" / "ninfer.exe"
MODEL = Path(r"G:\python\custom-kernel-3090\qwen3_8_27b.ninfer")
OUTPUT_DIR = ROOT / "benchmark_results" / "20260815_rk8v4_quality" / "01_matched_1k"
MAX_CONTEXT = 4096
MAX_NEW = 1024
KV_MODES = ("int8", "rk8v4")

PROMPT = r"""You are reviewing a production algorithm. Design and implement a complete C++20 solution for offline dynamic connectivity in an undirected graph.

The input is a time-ordered list of operations:
- add(id, u, v): activate an edge with a unique edge instance id;
- remove(id): deactivate that exact edge instance;
- connected(u, v): report whether u and v are connected at that moment.

The graph permits parallel edges and self-loops. An id is added at most once and removed at most once, but an edge may remain active through the end. Vertex labels are 0..n-1. Use a segment tree over time and a rollback DSU. Do not use path compression.

Deliver all of the following:
1. compilable C++20 code with explicit operation and answer types;
2. a precise explanation of how each edge's active half-open interval is constructed;
3. the rollback invariant, including what is recorded for a union that changes nothing;
4. a proof that every query sees exactly the edges active at its time;
5. time and memory complexity in terms of operations q, vertices n, and edge lifetimes;
6. handling of empty input, self-loops, parallel edges, invalid remove ids, and still-active edges.

Treat invalid remove ids as input errors and show where the implementation rejects them. Keep the answer rigorous and avoid replacing code with pseudocode."""


def run_mode(mode: str) -> dict:
    command = [
        str(EXE),
        str(MODEL),
        "--prompt",
        PROMPT,
        "--max-context",
        str(MAX_CONTEXT),
        "--kv-capacity",
        str(MAX_CONTEXT),
        "--prefill-chunk",
        "1024",
        "--max-new",
        str(MAX_NEW),
        "--kv-dtype",
        mode,
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--lm-head-draft",
        "--greedy",
        "--no-thinking",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=300, check=False)
    (OUTPUT_DIR / f"{mode}.answer.md").write_text(completed.stdout, encoding="utf-8")
    (OUTPUT_DIR / f"{mode}.metrics.txt").write_text(completed.stderr, encoding="utf-8")
    return {
        "mode": mode,
        "returncode": completed.returncode,
        "answer_characters": len(completed.stdout),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "prompt.md").write_text(PROMPT + "\n", encoding="utf-8")
    results = [run_mode(mode) for mode in KV_MODES]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL),
        "max_context": MAX_CONTEXT,
        "max_new_tokens": MAX_NEW,
        "sampling": "greedy",
        "speculation": "MTP3 with ReplaySSM and CUDA Graphs",
        "results": results,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
