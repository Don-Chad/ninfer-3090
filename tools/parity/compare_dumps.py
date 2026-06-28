#!/usr/bin/env python3
"""Dump and compare per-stage prefill tensors across the three references.

Stages (one prefill forward): embed, layer_00..layer_63 (hidden state after each
decoder block, i.e. after the MLP residual add), final_norm, logits_last.

Usage:
  # dump the quantized oracle (q5090) stages
  compare_dumps.py dump-oracle --weights W --ids F --out DIR

  # compare two stage dirs (ref is the trusted baseline, e.g. bf16)
  compare_dumps.py compare --ref DIR_A --test DIR_B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def stage_names(n_layers: int = 64) -> list[str]:
    return ["embed"] + [f"layer_{i:02d}" for i in range(n_layers)] + ["final_norm", "logits_last"]


def cmd_dump_oracle(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(ROOT))
    from tools.parity.ref_model import RefModel

    if args.ids:
        ids = [int(x) for x in Path(args.ids).read_text().split()]
    else:
        ids = [int(x) for x in args.prompt.replace(",", " ").split()]

    model = RefModel(
        args.weights,
        device=args.device,
        resident="gpu" if args.device == "cuda" else "stream",
    )
    dumps: dict = {}
    with torch.inference_mode():
        model.forward(ids, 1, dumps=dumps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embed.npy", dumps["embed"].numpy())
    for i in range(64):
        np.save(out / f"layer_{i:02d}.npy", dumps[f"layer_{i}"].numpy())
    np.save(out / "final_norm.npy", dumps["final_norm"].numpy())
    np.save(out / "logits_last.npy", dumps["logits_last"].numpy())
    print(f"dumped oracle stages to {out}")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / na) if na else float("nan")


def cmd_compare(args: argparse.Namespace) -> None:
    ref = Path(args.ref)
    test = Path(args.test)
    print(f"{'stage':>14}  {'cosine':>10}  {'rel_err':>10}  {'max_abs':>11}  shape")
    first_bad = None
    for name in stage_names():
        fa, fb = ref / f"{name}.npy", test / f"{name}.npy"
        if not fa.exists() or not fb.exists():
            continue
        a = np.load(fa)
        b = np.load(fb)
        if a.shape != b.shape:
            # last-position align for [T,D] vs [T,D] mismatches; otherwise flatten
            n = min(a.reshape(-1).shape[0], b.reshape(-1).shape[0])
            a2, b2 = a.reshape(-1)[:n], b.reshape(-1)[:n]
        else:
            a2, b2 = a, b
        cos = _cos(a2, b2)
        rel = _rel(a2, b2)
        maxabs = float(np.max(np.abs(a2.astype(np.float64) - b2.astype(np.float64))))
        flag = "" if cos >= args.threshold else "  <-- DIVERGES"
        if cos < args.threshold and first_bad is None:
            first_bad = name
        print(f"{name:>14}  {cos:>10.6f}  {rel:>10.5f}  {maxabs:>11.4e}  {tuple(a.shape)}{flag}")
    if first_bad is not None:
        print(f"\nfirst stage below cosine {args.threshold}: {first_bad}")
    else:
        print(f"\nall stages >= cosine {args.threshold}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump-oracle")
    d.add_argument("--weights", default=str(ROOT / "out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus"))
    d.add_argument("--ids", default=None)
    d.add_argument("--prompt", default=None)
    d.add_argument("--out", required=True)
    d.add_argument("--device", default="cuda")
    d.set_defaults(func=cmd_dump_oracle)

    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--test", required=True)
    c.add_argument("--threshold", type=float, default=0.99)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
