"""Matched Qwen3.8 K2/K3/K4/K5 benchmark using the established sweep harness."""

from __future__ import annotations

import json

import run_qwen38_mtp_sweep as sweep


OUTPUT_ROOT = sweep.ROOT / "benchmark_results/qwen3_8_round_10_mtp_k2_k3_k4_k5_graph"
MODES = (("k2", 2), ("k3", 3), ("k4", 4), ("k5", 5))


def main() -> None:
    sweep.OUTPUT_ROOT = OUTPUT_ROOT
    sweep.EXTRA_SERVER_ARGS = ()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    results = [sweep.run_mode(name, depth) for name, depth in MODES]
    baseline_hashes = results[0]["output_sha256"]
    exact = {result["mode"]: result["output_sha256"] == baseline_hashes for result in results}
    summary = {"results": results, "exact_output_match_to_k2": exact}
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
