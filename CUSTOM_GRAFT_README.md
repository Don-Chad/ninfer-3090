# Direct KV Graft Injection — Build & Use

## Overview

Direct KV graft injection lets you bake product knowledge, brand guidelines, or
other static context into the model's KV cache at server startup. Grafted
requests start generating from that cached state with zero prompt tokens — the
graft never appears in message history and costs no prefill compute.

---

## 1. Build the prefill graft (phantom-kv)

Create the initial KV cache snapshot from a source JSON containing your content.
Run from the `phantom-kv` repo using the `.venv`:

```
.venv\Scripts\python -m phantom_kv.graft.cli build-prefill ^
  --model Qwen/Qwen3.8-27B ^
  --quant nf4 ^
  --source data/grafts/brand_example.json ^
  --out artifacts/grafts/brand_prefill.bin ^
  --verify
```

Produces `brand_prefill.bin` (safetensors) + `brand_prefill.json` (metadata
sidecar). The `--verify` flag round-trips the artifact to confirm splice logits
match a single-pass forward.

## 2. (Optional) Train with direct-KV

Fine-tune the cached state against downstream QA pairs to improve task accuracy.
Note: `--quant` is a **global flag** and must come before the subcommand.

```
.venv\Scripts\python -m phantom_kv.train.cli --quant nf4 train ^
  --arm kv ^
  --model Qwen/Qwen3.8-27B ^
  --targets data/train/targets_v2.jsonl ^
  --warm-graft artifacts/grafts/brand_prefill.bin ^
  --steps 400 ^
  --lr 1e-3 ^
  --micro-batch 1 ^
  --out-dir artifacts/train/brand_directkv
```

**Notes:**
- Use `--micro-batch 1` on a 24 GB GPU (e.g. RTX 3090). The 27B NF4 model
  uses ~15 GB VRAM; higher batch sizes OOM.
- Install `flash-linear-attention` and `triton-windows` for ~5-10x faster
  GDN kernels. Without them, PyTorch fallback works but is slow.
  ```
  pip install flash-linear-attention triton-windows
  ```

Then compile the trained checkpoint back to graft format:

```
.venv\Scripts\python -m phantom_kv.train.cli --quant nf4 compile ^
  --arm kv ^
  --ckpt artifacts/train/brand_directkv/ckpt_final.pt ^
  --model Qwen/Qwen3.8-27B ^
  --out artifacts/grafts/brand_trained.bin
```

## 3. Start ninfer-serve with the graft

```
build-ninja\apps\ninfer-serve.exe ^
  --model path\to\brand_q38_nf4.bin ^
  --graft brand=artifacts\grafts\brand_trained.bin ^
  --port 8080
```

`--graft NAME=PATH` is repeatable — load multiple grafts side by side. Each
direct-KV or softprompt-KV graft is injected into a pinned shared-prefix slot
at startup. Prefill-KV grafts use token replay instead.

## 4. Select the graft per request

Add `"graft": "NAME"` to any OpenAI-compatible chat completion request:

```json
{
  "model": "qwen3.8-27b",
  "messages": [
    {"role": "user", "content": "What's our return policy?"}
  ],
  "graft": "brand"
}
```

The model responds as though the brand knowledge preceded the user's message.
No system turn is needed — the graft already holds one.

---

## What happens under the hood

### Startup (inject_direct_graft)

1. K/V tensors are written into the paged KV cache via `kv_cache_append`
   (16 attention layers, BF16).
2. GDN conv and recurrent state are transposed and uploaded into a state image
   (48 linear layers, conv BF16, recurrent FP32).
3. The shared-prefix slot is set to **Pinned** role (never evicted) and
   registered in `graft_prefix_slots` by name.

### Request admission (inspect_admission → inspect_lane)

1. Frontend sets `prompt.graft_name` and `prompt.graft_frontier` from the
   graft metadata.
2. Resource manager calls `inspect_admission` with no external source.
3. Graft bypass activates: looks up the pinned slot, creates a temporary
   `SharedPrefixHandle`, synthesizes a `SharedStablePrefix` checkpoint.
4. `inspect_lane` receives `is_graft=true`, skips token-identity matching
   (grafts have no token sequence), and sets `ReusePath::SharedStablePrefix`
   with `reuse_base = graft_frontier`.
5. Generation starts from the graft's frontier position — no prefill needed.

### Key properties

- **Zero prompt token cost.** The graft's slots don't consume any of the
  request's context budget.
- **Pinned lifetime.** The slot is never evicted by cache pressure; it lives
  for the duration of the server process.
- **Multiple grafts.** Each occupies its own shared-prefix slot. Requests
  select one by name or use none.
- **Hybrid model support.** Works on Qwen3.8-27B's mixed architecture
  (48 linear-attention + 16 full-attention layers). Both the KV cache and
  the GDN state (conv + recurrent) are injected.

---

## File format

A graft artifact is two files side by side:

| File                 | Contents                                              |
|----------------------|-------------------------------------------------------|
| `name.bin`           | Safetensors container: `k`, `v` (BF16), `conv` (BF16), `rec` (FP32), optionally `replay_ids` (I64) |
| `name.json`          | Metadata sidecar: model_id, layer_types, dimensions, sha256 over the safetensors payload |

The `--graft` flag points to the `.bin`; ninfer finds the `.json` beside it.

## Building ninfer (Windows)

From the ninfer-3090 repo root:

```
build_merge.bat
```

This sources `vcvars64.bat` (MSVC 14.44), pins CUDA 12.8, runs CMake
configure + ninja build. Output binaries land in `build-ninja/apps/`.

For a rebuild after code changes (no reconfigure needed):

```
build_only.bat
```

---

## Appendix: v1 prefill dataset commands

The exact commands used to build and train the v1 compliance graft from
`data/grafts/v1_prefill.json` on Qwen3.8-27B NF4. All paths relative to
the `phantom-kv` repo root.

```bash
# 1. Build prefill graft (already done — produces v1_q38_nf4.bin, 132 slots)
.venv/Scripts/python -m phantom_kv.graft.cli build-prefill \
  --model Qwen/Qwen3.8-27B \
  --quant nf4 \
  --source data/grafts/v1_prefill.json \
  --out artifacts/grafts/v1_q38_nf4.bin \
  --verify

# 2. Train direct-KV (112 targets: 63 ce / 29 sup / 20 kl)
.venv/Scripts/python -m phantom_kv.train.cli --quant nf4 train \
  --arm kv \
  --model Qwen/Qwen3.8-27B \
  --targets data/train/targets_v2.jsonl \
  --warm-graft artifacts/grafts/v1_q38_nf4.bin \
  --steps 400 \
  --lr 1e-3 \
  --micro-batch 1 \
  --out-dir artifacts/train/v1_directkv

# 3. Compile trained checkpoint to graft
.venv/Scripts/python -m phantom_kv.train.cli --quant nf4 compile \
  --arm kv \
  --ckpt artifacts/train/v1_directkv/ckpt_final.pt \
  --model Qwen/Qwen3.8-27B \
  --out artifacts/grafts/v1_q38_nf4_trained.bin

# 4. Serve with ninfer (from ninfer-3090 repo root)
build-ninja/apps/ninfer-serve.exe \
  --model <path-to-artifact>/brand_q38_nf4.bin \
  --graft v1=<phantom-kv>/artifacts/grafts/v1_q38_nf4_trained.bin
```

### v1 training results (2026-09-23)

- **400 steps**, micro-batch 1, lr 1e-3, RTX 3090 24 GB
- CE loss: 0.94 → 0.20, KL loss: 0.17 → 0.03, sup loss: 0.00 throughout
- Anchor (L2 drift): ~5e-4 at final step — minimal parameter drift
- Checkpoint: `artifacts/train/v1_directkv/ckpt_final.pt`
- Compiled graft: `artifacts/grafts/v1_q38_nf4_trained.bin` (sha `2746e98b1712`)
- Compile verification: bitwise tensor equality, zero logit diff

## Appendix: v1 Qwen3.6-35B-A3B commands

Same v1 prefill dataset (`data/grafts/v1_prefill.json`) and targets
(`data/train/targets_v2.jsonl`), replicated on the Qwen3.6-35B-A3B MoE model
(`qwen3_5_moe_text`, 40 layers: 10 full_attention + 30 GDN linear_attention,
256 experts ~3B active, 2 KV heads, head_dim 256).

```bash
# 1. Build prefill graft (produces v1_q36_35b_nf4.bin, 128 slots)
.venv/Scripts/python -m phantom_kv.graft.cli build-prefill \
  --model Qwen/Qwen3.6-35B-A3B \
  --quant nf4 \
  --source data/grafts/v1_prefill.json \
  --out artifacts/grafts/v1_q36_35b_nf4.bin \
  --verify

# 2. Train direct-KV (same 112 targets)
.venv/Scripts/python -m phantom_kv.train.cli --quant nf4 train \
  --arm kv \
  --model Qwen/Qwen3.6-35B-A3B \
  --targets data/train/targets_v2.jsonl \
  --warm-graft artifacts/grafts/v1_q36_35b_nf4.bin \
  --steps 400 \
  --lr 1e-3 \
  --micro-batch 1 \
  --out-dir artifacts/train/v1_q36_35b_directkv

# 3. Compile trained checkpoint to graft
.venv/Scripts/python -m phantom_kv.train.cli --quant nf4 compile \
  --arm kv \
  --ckpt artifacts/train/v1_q36_35b_directkv/ckpt_final.pt \
  --model Qwen/Qwen3.6-35B-A3B \
  --out artifacts/grafts/v1_q36_35b_nf4_trained.bin

# 4. Serve with ninfer (from ninfer-3090 repo root)
build-ninja/apps/ninfer-serve.exe \
  --model <path-to-artifact>/qwen3_6_35b_a3b.ninfer \
  --graft v1=<phantom-kv>/artifacts/grafts/v1_q36_35b_nf4_trained.bin
```

### v1 35B training results (2026-09-24)

- **400 steps**, micro-batch 1, lr 1e-3, RTX 3090 24 GB
- CE loss: ~2.07 → ~0.63, KL loss: ~0.40 → ~0.20, sup loss: 0.00 on most steps
- Anchor (L2 drift): ~3.3e-4 at final step — minimal parameter drift
- Gradient flow proof: PASS (k, v, rec, conv all nonzero)
- Checkpoint: `artifacts/train/v1_q36_35b_directkv/ckpt_final.pt`
- Compiled graft: `artifacts/grafts/v1_q36_35b_nf4_trained.bin` (sha `e0121206cca8`)
- Compile verification: bitwise tensor equality, zero logit diff

## run.bat graft integration

`scripts/run.bat` now auto-loads trained grafts at server startup. Each model
key maps to a graft filename; the script looks in the sibling `phantom-kv`
repo's `artifacts/grafts/` directory by default.

| Model key           | Graft file                      |
|---------------------|---------------------------------|
| `qwen38-27b`        | `v1_q38_nf4_trained.bin`        |
| `qwen36-35b-a3b`    | `v1_q36_35b_nf4_trained.bin`    |

Overrides:
- `NINFER_GRAFT_DIR=<path>` — custom graft directory
- `NINFER_GRAFTS=off` — disable graft loading entirely

If the graft file doesn't exist, the server starts without it (no error).

### Dependencies

```
pip install flash-linear-attention triton-windows
```

`flash-linear-attention` provides fused GDN kernels (~5-10x faster than the
PyTorch reference fallback in transformers). Requires `triton-windows` on
Windows. `causal-conv1d` is optional — its source build fails on Windows
(bare_metal_version NameError) but the conv fallback is cheap.
