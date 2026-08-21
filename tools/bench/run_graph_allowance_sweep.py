"""Measure CUDA Graph memory consumption against KV capacity.

The engine plans a graph allowance at startup and aborts if capture exceeds it::

    error: CUDA Graph preparation consumed 875819008 bytes, exceeding the planned
    allowance of 12582912 bytes

The non-speculative allowance is a flat ``12 MiB * max_concurrency`` with no capacity term
(``layouts_impl.h``), while the number of captured graphs is
``len(ordinary_graph_profiles(capacity)) * max_concurrency`` (``variant.cpp``). This sweep measures
what capture actually costs so the allowance can be given a defensible formula instead of a
constant.

No profiler is involved. The CLI prints ``CUDA Graph memory <observed> / <allowance>`` itself, and
a failure names both figures in the error text, so a failed run is still a data point.

Two properties of the measurement drive the design:

**It is a device-wide delta, not this process's allocation.** ``prepare_graphs`` brackets capture
with two ``cudaMemGetInfo`` calls, so anything else resident on the GPU during that window lands in
the number. Repeats are therefore mandatory: a single reading per capacity cannot distinguish a
real curve from desktop VRAM drift. ``--repeats`` defaults to 3 and the report leads with spread.

**Capture is eager and happens at construction**, before any prompt is seen. Prompt length cannot
affect it. The prompt here is deliberately trivial and ``--max-new`` is 1: this measures startup,
so each point costs roughly one model load rather than a benchmark run.

Usage::

    export LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nvddi.inf_amd64_dabe455c4b5cf6f0:/usr/lib/wsl/lib
    uv run python -m tools.bench.run_graph_allowance_sweep

Edit the configuration constants below before running the sweep.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLI = ROOT / "build/apps/ninfer"
DEFAULT_MODEL = Path.home() / "ai/models/qwen3_8_27b.ninfer"

MIB = 1024 * 1024

# Mirrors Variant::ordinary_graph_profiles in src/targets/qwen3_6_27b/impl/variant.cpp. Kept here so
# the report can state the predicted graph count next to the measured cost; if the source table
# changes this must change with it, which is why the values are quoted rather than derived.
ORDINARY_PROFILE_ENDS = (127, 511, 2047, 4095, 8197, 16389, 32767)

# Capacities either side of every profile boundary, plus the four points already measured by hand.
# Boundaries are where the graph count steps, so they are where a count-driven model and a
# frontier-driven model disagree most.
DEFAULT_CAPACITIES = (
    2048, 4096, 8198, 16390,
    32768, 32832, 40960, 49152, 65536, 81920, 103424, 131072,
)

# Editable sweep configuration. Keep these values explicit so saved benchmark artifacts identify
# the exact workload without depending on shell history.
CLI = DEFAULT_CLI
MODEL = DEFAULT_MODEL
CAPACITIES = DEFAULT_CAPACITIES
REPEATS = 3
SPEC: str | None = None
DRAFT_TOKENS = 3
KV_DTYPE = "int8"
PREFILL_CHUNK = 640
OUT = ROOT / "out/graph_allowance"

SUMMARY = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9 _/-]*?)\s{2,}(.+?)\s*$")
# A bare "B" unit is not a formatting curiosity: the engine clamps a negative device-wide delta to
# zero, so "0 B" is the signature of another process freeing VRAM during capture. It must parse.
GRAPH_LINE = re.compile(r"CUDA Graph memory\s+([\d.]+)\s*([KMG]?i?B)\s*/\s*([\d.]+)\s*([KMG]?i?B)")
GRAPH_ERROR = re.compile(
    r"CUDA Graph preparation consumed (\d+) bytes, exceeding the planned allowance of (\d+) bytes"
)
UNITS = {"B": 1, "KiB": 1024, "MiB": MIB, "GiB": 1024 * MIB}


def ordinary_graph_count(capacity: int, concurrency: int = 1) -> int:
    """Number of ordinary decode graphs captured for a capacity.

    Mirrors graph_profiles_through(capacity - 1, ORDINARY_PROFILE_ENDS) exactly, including its
    early return when a range lands on the frontier.
    """
    max_frontier = capacity - 1
    profiles = 0
    begin = 0
    for preferred_end in ORDINARY_PROFILE_ENDS:
        if begin > max_frontier:
            break
        end = min(preferred_end, max_frontier)
        profiles += 1
        if end == max_frontier:
            return profiles * concurrency
        begin = end + 1
    if begin <= max_frontier:
        profiles += 1
    return profiles * concurrency


def parse_bytes(value: str, unit: str) -> int:
    return int(round(float(value) * UNITS[unit]))


def measure(cli: Path, model: Path, capacity: int, spec: str | None, draft_tokens: int,
            kv_dtype: str, prefill_chunk: int) -> dict:
    command = [
        str(cli), str(model),
        "--prompt", "hi",
        "--max-context", str(capacity),
        "--kv-capacity", str(capacity),
        "--kv-dtype", kv_dtype,
        "--prefill-chunk", str(prefill_chunk),
        "--max-new", "1",
        "--greedy", "--no-thinking",
    ]
    if spec:
        command += ["--spec", spec, "--draft-tokens", str(draft_tokens), "--lm-head-draft"]

    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - started

    observed = allowance = None
    # A run that aborts still reports both figures, in exact bytes rather than the summary's
    # two-decimal MiB, so failures are the more precise data points.
    error = GRAPH_ERROR.search(result.stderr)
    if error:
        observed, allowance = int(error.group(1)), int(error.group(2))
    else:
        line = GRAPH_LINE.search(result.stderr)
        if line:
            observed = parse_bytes(line.group(1), line.group(2))
            allowance = parse_bytes(line.group(3), line.group(4))

    return {
        "capacity": capacity,
        "returncode": result.returncode,
        "started": result.returncode == 0,
        "graph_blocked": error is not None,
        "observed_bytes": observed,
        "allowance_bytes": allowance,
        "predicted_graphs": ordinary_graph_count(capacity),
        "wall_s": round(wall, 2),
        "stderr_tail": result.stderr.strip().splitlines()[-1] if result.returncode else "",
        # A run that neither reports a figure nor aborts is the one case this sweep cannot
        # interpret, so keep its output rather than recording a silent None.
        "unparsed_stderr": None if observed is not None else result.stderr[-4000:],
    }


def main() -> int:
    if SPEC not in (None, "mtp"):
        raise SystemExit(f"unsupported speculative mode: {SPEC}")
    if REPEATS < 2:
        raise SystemExit("REPEATS must be at least 2 to separate signal from device-wide noise")
    if not CLI.exists():
        raise SystemExit(f"engine binary not found: {CLI}")
    if not MODEL.exists():
        raise SystemExit(f"model artifact not found: {MODEL}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(f"{'capacity':>9}  {'graphs':>6}  {'observed':>12}  {'allowance':>10}  {'per graph':>10}  "
          f"{'spread':>8}  result")
    for capacity in CAPACITIES:
        runs = [
            measure(CLI, MODEL, capacity, SPEC, DRAFT_TOKENS, KV_DTYPE, PREFILL_CHUNK)
            for _ in range(REPEATS)
        ]
        rows.extend(runs)

        seen = [r["observed_bytes"] for r in runs if r["observed_bytes"] is not None]
        graphs = runs[0]["predicted_graphs"]
        if not seen:
            print(f"{capacity:>9}  {graphs:>6}  {'—':>12}  {'—':>10}  {'—':>10}  {'—':>8}  "
                  f"no graph figure reported")
            continue

        median = statistics.median(seen)
        spread = max(seen) - min(seen)
        allowance = next(r["allowance_bytes"] for r in runs if r["allowance_bytes"] is not None)
        blocked = sum(1 for r in runs if r["graph_blocked"])
        verdict = "ok" if blocked == 0 else f"BLOCKED {blocked}/{len(runs)}"
        print(f"{capacity:>9}  {graphs:>6}  {median / MIB:>10.2f} MiB  {allowance / MIB:>8.2f} MiB  "
              f"{median / graphs / MIB:>8.2f} MiB  {spread / MIB:>6.2f} MiB  {verdict}")

    report = {
        "config": {
            "cli": str(CLI), "model": str(MODEL), "repeats": REPEATS,
            "spec": SPEC, "draft_tokens": DRAFT_TOKENS if SPEC else None,
            "kv_dtype": KV_DTYPE, "prefill_chunk": PREFILL_CHUNK,
        },
        "runs": rows,
    }
    label = SPEC or "ordinary"
    destination = OUT / f"graph-allowance-{label}-{KV_DTYPE}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
