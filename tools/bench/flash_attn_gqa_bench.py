#!/usr/bin/env python3
"""FlashAttention reference bench for the Qwen3.6 GQA prompt-attention shape.

This is an optional performance oracle, not a project runtime dependency. It
measures the same causal append-prompt attention geometry as
bench/gqa_attention_bench.cu:

  q: [1, T, 24, 256]
  k/v: [1, context + T, 4, 256]
  scale: 0.0625

FlashAttention's bottom-right causal mask for seqlen_q != seqlen_k matches the
append prompt mask: query row i can see keys [0, context + i].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable


HEAD_DIM = 256
Q_HEADS = 24
KV_HEADS = 4
SCALE = 0.0625
DENSE_TC_PEAK_TFLOPS = 209.5


def parse_i32_list(text: str, flag: str) -> list[int]:
    values: list[int] = []
    for piece in text.split(","):
        if not piece:
            raise SystemExit(f"{flag} expects comma-separated integers")
        value = int(piece)
        if value < 0:
            raise SystemExit(f"{flag} expects non-negative integers")
        values.append(value)
    if not values:
        raise SystemExit(f"{flag} expects at least one value")
    return values


def pct(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


def bench_cuda_events(
    launch: Callable[[], object],
    torch_mod,
    warmup: int,
    repeat: int,
    min_time_ms: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        launch()
    torch_mod.cuda.synchronize()

    start = torch_mod.cuda.Event(enable_timing=True)
    stop = torch_mod.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(4):
        launch()
    stop.record()
    stop.synchronize()
    probe_us = max(start.elapsed_time(stop) * 1000.0 / 4.0, 1.0)
    inner_iters = max(1, min(1024, math.ceil(500.0 / probe_us)))

    samples: list[float] = []
    total_us = 0.0
    while len(samples) < repeat or total_us < min_time_ms * 1000.0:
        start.record()
        for _ in range(inner_iters):
            launch()
        stop.record()
        stop.synchronize()
        batch_us = start.elapsed_time(stop) * 1000.0
        samples.append(batch_us / inner_iters)
        total_us += batch_us
        if len(samples) > 100000:
            break

    sorted_samples = sorted(samples)
    return {
        "runs": len(samples),
        "inner_iters": inner_iters,
        "median_us": pct(sorted_samples, 0.50),
        "min_us": sorted_samples[0],
        "p95_us": pct(sorted_samples, 0.95),
        "mean_us": statistics.fmean(samples),
    }


def append_prompt_key_sum(tokens: int, context: int) -> float:
    return float(tokens) * float(context) + float(tokens) * float(tokens + 1) * 0.5


def useful_flops(tokens: int, context: int) -> float:
    return 4.0 * HEAD_DIM * Q_HEADS * append_prompt_key_sum(tokens, context)


def metrics_from_result(tokens: int, context: int, mode: str, result: dict[str, float | int]) -> dict:
    median_us = float(result["median_us"])
    key_sum = append_prompt_key_sum(tokens, context)
    flops = useful_flops(tokens, context)
    sec = median_us * 1.0e-6
    tflops = flops / sec / 1.0e12 if sec > 0.0 else 0.0
    return {
        "impl": "flash_attn",
        "mode": mode,
        "T": tokens,
        "context": context,
        "end_context": context + tokens,
        "ms": median_us * 1.0e-3,
        "tflops": tflops,
        "tflops_pct": tflops / DENSE_TC_PEAK_TFLOPS * 100.0,
        "avg_keys_per_query": key_sum / float(tokens),
        "ns_per_key_query": median_us * 1000.0 / key_sum,
        "us_per_token": median_us / float(tokens),
        "useful_flops": flops,
        "key_sum": key_sum,
        "median_ms": median_us * 1.0e-3,
        "min_ms": float(result["min_us"]) * 1.0e-3,
        "p95_ms": float(result["p95_us"]) * 1.0e-3,
        "mean_ms": float(result["mean_us"]) * 1.0e-3,
        "runs": int(result["runs"]),
        "inner_iters": int(result["inner_iters"]),
    }


def load_qus_csv(path: str) -> dict[tuple[int, int], dict[str, str]]:
    if not path:
        return {}
    out: dict[tuple[int, int], dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(int(row["T"]), int(row["context"]))] = row
    return out


def attach_qus_comparison(rows: list[dict], qus_rows: dict[tuple[int, int], dict[str, str]]) -> None:
    for row in rows:
        qus = qus_rows.get((int(row["T"]), int(row["context"])))
        if qus is None:
            continue
        qus_ms = float(qus["ms"])
        qus_tflops = float(qus["tflops"])
        row["qus_ms"] = qus_ms
        row["qus_tflops"] = qus_tflops
        row["flash_vs_qus_speedup"] = qus_ms / float(row["ms"]) if float(row["ms"]) > 0.0 else None
        row["flash_vs_qus_tflops_ratio"] = (
            float(row["tflops"]) / qus_tflops if qus_tflops > 0.0 else None
        )


def write_csv(path: str, rows: list[dict]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def write_json(path: str, rows: list[dict], metadata: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "flash_attn_gqa_append_prompt_bench",
                **metadata,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1024", help="comma-separated T values")
    parser.add_argument("--context", required=True, help="comma-separated context values")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--min-time-ms", type=int, default=500)
    parser.add_argument("--include-fill", action="store_true", help="also time cache K/V slice fill")
    parser.add_argument("--attention-only", action="store_true", help="only time FlashAttention")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--qus-csv", default="", help="optional qus_gqa_attention_bench CSV to merge")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.attention_only and args.include_fill:
        raise SystemExit("--attention-only and --include-fill are mutually exclusive")

    try:
        import torch
        from flash_attn import flash_attn_func
    except Exception as exc:
        print(
            "error: this optional bench requires CUDA torch and flash-attn in the active Python "
            f"environment: {exc}",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print("error: torch.cuda.is_available() is false", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokens_list = parse_i32_list(args.tokens, "--tokens")
    if any(t <= 0 for t in tokens_list):
        raise SystemExit("--tokens expects positive values")
    contexts = parse_i32_list(args.context, "--context")

    modes = ["attention"]
    if args.include_fill:
        modes.append("attention_with_cache_fill")
    elif not args.attention_only:
        modes.append("attention_with_cache_fill")

    rows: list[dict] = []
    for tokens in tokens_list:
        for context in contexts:
            end_context = context + tokens
            q = torch.empty((1, tokens, Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
            k_full = torch.empty((1, end_context, KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
            v_full = torch.empty((1, end_context, KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
            q.normal_(mean=0.0, std=0.5)
            k_full.normal_(mean=0.0, std=0.5)
            v_full.normal_(mean=0.0, std=0.5)

            def attention_launch():
                return flash_attn_func(q, k_full, v_full, dropout_p=0.0, softmax_scale=SCALE, causal=True)

            if "attention" in modes:
                result = bench_cuda_events(
                    attention_launch, torch, args.warmup, args.repeat, args.min_time_ms
                )
                row = metrics_from_result(tokens, context, "attention", result)
                rows.append(row)
                print(
                    f"flash_attn attention T={tokens:<6d} C={context:<7d} "
                    f"median={row['ms']:9.3f} ms useful={row['tflops']:9.2f} TFLOP/s "
                    f"tc={row['tflops_pct']:6.2f}% ns/key={row['ns_per_key_query']:6.3f}"
                )

            if "attention_with_cache_fill" in modes:
                k_cache = torch.empty_like(k_full)
                v_cache = torch.empty_like(v_full)
                k_cache[:, :context].normal_(mean=0.0, std=0.5)
                v_cache[:, :context].normal_(mean=0.0, std=0.5)
                k_new = k_full[:, context:end_context].contiguous()
                v_new = v_full[:, context:end_context].contiguous()

                def fill_and_attention_launch():
                    k_cache[:, context:end_context].copy_(k_new)
                    v_cache[:, context:end_context].copy_(v_new)
                    return flash_attn_func(
                        q,
                        k_cache[:, :end_context],
                        v_cache[:, :end_context],
                        dropout_p=0.0,
                        softmax_scale=SCALE,
                        causal=True,
                    )

                result = bench_cuda_events(
                    fill_and_attention_launch, torch, args.warmup, args.repeat, args.min_time_ms
                )
                row = metrics_from_result(tokens, context, "attention_with_cache_fill", result)
                rows.append(row)
                print(
                    f"flash_attn with-fill T={tokens:<6d} C={context:<7d} "
                    f"median={row['ms']:9.3f} ms useful={row['tflops']:9.2f} TFLOP/s "
                    f"tc={row['tflops_pct']:6.2f}% ns/key={row['ns_per_key_query']:6.3f}"
                )

    attach_qus_comparison(rows, load_qus_csv(args.qus_csv))
    metadata = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "head_dim": HEAD_DIM,
        "q_heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "scale": SCALE,
        "tc_peak_tflops": DENSE_TC_PEAK_TFLOPS,
        "causal_mask": "FlashAttention bottom-right causal mask for seqlen_q != seqlen_k",
        "timing": {"warmup": args.warmup, "repeat": args.repeat, "min_time_ms": args.min_time_ms},
    }
    write_csv(args.csv_out, rows)
    write_json(args.json_out, rows, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
