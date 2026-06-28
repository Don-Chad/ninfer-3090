#!/usr/bin/env python3
"""Trusted bf16 reference for Qwen3.6-27B (text core), driven manually.

Why not model.generate(): the upstream Qwen3_5ForConditionalGeneration generate
path manages multimodal mRoPE / rope_deltas + GDN cache state during incremental
decode; with our text-only prompt that path produced a correct first token then
degraded into garbage. To get a trustworthy ground truth we control the compute
graph ourselves:

  * greedy decode is a no-cache full re-forward each step (every step is a fresh,
    correct prefill -> no cache/position-advance corruption), and
  * per-stage tensors are captured from a single prefill via output_hidden_states.

The model is ~54 GB in bf16 (> one 32 GB GPU) so we use accelerate device_map
"auto" with a CPU spill budget. This is slow but only needs to run once: dump the
per-stage tensors and reuse the dumps for repeated comparison.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def parse_ids(args) -> list[int]:
    if args.ids:
        text = Path(args.ids).read_text(encoding="utf-8")
    elif args.prompt:
        text = args.prompt
    else:
        raise SystemExit("pass --ids <file> or --prompt '<ids>'")
    return [int(t) for t in text.replace(",", " ").split()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16")
    ap.add_argument("--ids", default=None, help="path to a .ids file of prompt token ids")
    ap.add_argument("--prompt", default=None, help="comma/space separated prompt token ids")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--gpu-mem", default="26GiB")
    ap.add_argument("--cpu-mem", default="80GiB")
    ap.add_argument("--dump-dir", default=None, help="if set, dump per-stage prefill tensors here")
    ap.add_argument("--stop-token-ids", default="248046,248044")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    ids = parse_ids(args)
    stop = {int(x) for x in args.stop_token_ids.split(",") if x.strip()}
    print(f"prompt_len={len(ids)}", flush=True)

    t0 = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_mem, "cpu": args.cpu_mem},
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    print(f"load_s={time.perf_counter() - t0:.1f}", flush=True)

    try:
        first_device = next(model.parameters()).device
    except StopIteration:
        first_device = torch.device("cuda:0")

    # ---- per-stage dump from a single prefill (one correct forward) ----
    if args.dump_dir:
        out_dir = Path(args.dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inp = torch.tensor([ids], dtype=torch.long, device=first_device)
        with torch.inference_mode():
            res = model(input_ids=inp, use_cache=False, output_hidden_states=True)
        hs = res.hidden_states  # tuple: (embed, layer0_out, ..., layer63_out)

        def save(name, t):
            np.save(out_dir / f"{name}.npy", t.detach().float().cpu().numpy())

        save("embed", hs[0][0])
        for i in range(len(hs) - 1):
            save(f"layer_{i:02d}", hs[i + 1][0])
        save("logits_last", res.logits[0, -1])
        print(f"dumped {len(hs)} hidden stages + logits to {out_dir}", flush=True)

    # ---- meaningful-output confirmation: no-cache greedy ----
    seq = list(ids)
    gen: list[int] = []
    t1 = time.perf_counter()
    for step in range(args.max_new):
        inp = torch.tensor([seq], dtype=torch.long, device=first_device)
        with torch.inference_mode():
            out = model(input_ids=inp, use_cache=False)
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        seq.append(nxt)
        if nxt in stop:
            break
    print(f"gen_s={time.perf_counter() - t1:.1f} steps={len(gen)}", flush=True)
    print("HF_REF_TOKENS:", gen, flush=True)
    print("HF_REF_TEXT:", repr(tok.decode(gen)), flush=True)


if __name__ == "__main__":
    main()
