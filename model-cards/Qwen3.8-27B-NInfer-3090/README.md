---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - conversational
  - cuda
  - rtx-3090
---

# Qwen3.8-27B for NInfer on the RTX 3090

This model card is the version-controlled source for
[WarlaxZ/Qwen3.8-27B-NInfer-3090](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NInfer-3090).

The repository contains [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to the
native `.ninfer` artifact format by the RTX 3090 fork of NInfer,
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090). It is the fork's default
27B model (`download-model qwen38-27b`). It is not a Transformers checkpoint, Safetensors
distribution or GGUF file, and it needs this fork's executables: upstream NInfer and older fork
builds refuse it at startup because they lack its 4-bit MLP down route.

## What is different from the upstream artifact

Same model, same container, a different weight encoding. Upstream's
[neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) rounds each
64-weight group to the nearest code with scale = absmax / qmax. This artifact uses the
`grouped_search` encoder:

- each group's FP16 scale is searched over 50 candidates, choosing the one with the least
  importance-weighted squared error, where importance is the mean squared input activation per
  channel (an imatrix);
- scales may be negative, which lets a group use the full code range (−8 for Q4, −16 for Q5);
- the layout spends fewer bytes per decoded token: a 4-bit token embedding, a 6-bit output head,
  and 4-bit attention/GDN output projections and MLP down projections in layers 36–63.
  Everything else keeps upstream's formats (Q4 query/key/gate/up, Q5 value/z/gate/down/output).

Vision, MTP, DFlash2 and the 131,072-row proposal head are carried over with upstream's encoding.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b.ninfer` |
| Size | 18,982,458,624 bytes (17.68 GiB); upstream's is 20,437,521,664 |
| SHA-256 | `7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706` |
| Container version | 3 |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `qwen3.8-27b` |
| Components | text, vision, MTP, DFlash2, proposal head |
| Stored objects | 1,190 (1,184 tensors and 6 resources) |

```bash
printf '%s  %s\n' \
  '7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706' \
  'qwen3_8_27b.ninfer' | sha256sum --check
```

The repository also holds `qwen3_8_27b.ninfer.conversion.json` (the converter's report: every
object, its sources, format and method) and `qwen3_8_27b.imatrix.safetensors`, the importance
matrix the encoder used, so the artifact can be rebuilt.

## Measurements

RTX 3090 (24 GB), CUDA 12.8, Windows. "Upstream" is `neroued/Qwen3.8-27B-NInfer` at
`1cbd84e7`, run with the same binaries. Quality was measured at fork commit `66a05b72`; speed with
the small-T Q6 head kernel that followed it (`36b798c9`), which this artifact needs to be faster
under speculative decoding.

**Quality.** `ninfer-perplexity --quick --kv-dtype int8`:

| Corpus | Upstream | This artifact | Change |
|---|---:|---:|---:|
| `perplexity-1m` (inherited) | 4.3428 | 4.3023 | −0.93% |
| `perplexity-heldout-2026-09` (wikipedia en/zh, arXiv, GitHub, own code, chat) | 4.3396 | 4.3178 | −0.50% |

KL divergence against Unsloth's Q8_0 GGUF on the held-out corpus (8 × 4096 tokens per stream,
llama.cpp b11316 with the NInfer weights exported bit-exactly into Q8_0 blocks):

| Weights | Mean KLD |
|---|---:|
| Upstream encoding and layout | 0.0376 |
| This layout with a 6-bit embedding | 0.0304 |
| This artifact (4-bit embedding) | 0.0318 |
| Unsloth UD-Q4_K_XL (llama.cpp formats, 17.56 GB of text weights) | 0.0156 |

The 4-bit embedding costs Chinese Wikipedia (0.0794 → 0.0859) and chat (0.0355 → 0.0373) against
a 6-bit one; English, arXiv and code streams are unchanged to the fourth decimal.

Unsloth's UD-Q4_K_XL is still clearly better per byte. Their formats are not NInfer's, and
about half of the gap is their per-tensor allocation, which needs kernel routes NInfer does not
have yet.

**Speed.** `ninfer_bench`, rk4v4 KV, 8,192-token context, two interleaved rounds of three
repetitions each:

| Workload | Upstream | This artifact | Change |
|---|---:|---:|---:|
| Plain decode, tg128 | 47.1 tok/s | 49.2 tok/s | +4.5% |
| Prefill, pp4096 | 1,713 tok/s | 1,727 tok/s | +0.8% |
| MTP3 + draft head, pp2048+tg256 decode | 147.4 tok/s | 151.5 tok/s | +2.8% |
| DFlash2 K=7 + draft head, pp2048+tg256 decode | 251.1 tok/s | 261.2 tok/s | +4.0% |

The bench corpus is unusually easy to draft (DFlash2 accepts 7.3 tokens per round on it), so the
speculative rows were also measured on 24 varied chat prompts through `ninfer-serve` (greedy,
thinking off, 512 tokens each, one stream):

| Profile | Upstream | This artifact | Tokens per round |
|---|---:|---:|---|
| DFlash2 K=7 + draft head | 125.9 tok/s | 133.5 tok/s (+6.1%) | 3.53 → 3.57 |
| MTP3 + draft head | 111.5 tok/s | 115.4 tok/s (+3.5%) | 2.87 → 2.84 |

MTP acceptance is about 1-1.5% lower than upstream's, so its gain is smaller than DFlash2's.
Before the small-T Q6 kernel, the 6-bit head's 8-token verify took 2.56 ms against 1.58 ms for
upstream's 8-bit head and cancelled DFlash2's gain (−0.6% on the bench); with it the head reads
at about 850 GB/s from 3 to 8 tokens.

**Memory.** Started back to back through `run.bat qwen38-27b` on the same desktop. The earlier
`run.bat` transcoded upstream's embedding to 4 bits at load, so the saving is smaller than the file
size difference:

| Profile | Upstream weights | This artifact | KV that fitted |
|---|---:|---:|---|
| DFlash2 (172,032 requested) | 17.7 GiB | 17.0 GiB | 150,720 → 172,224 tokens |
| MTP3, two lanes, 262,144 | 15.7 GiB | 15.3 GiB | both full; free 822 MiB → 1.19 GiB |

With the desktop holding 1.6 GiB, the DFlash2 profile starts at 188,416 tokens with 721 MiB free
and is refused at 196,608, so `run.bat` now defaults to 188,416. `run.bat` no longer passes
`--embedding-q4` or `--lm-head-q6`, which this artifact does not need; the upstream file therefore
no longer fits the default profiles.

**Smoke tests.** These all completed coherently:

- both `run.bat` profiles served a chat request;
- DFlash2, MTP3 and plain text generation;
- `image_chart` (`NIFER VISION 731；3；左侧`, the expected answer);
- `image_natural` (mailbox 24, sun on the right) with `--vision --vision-residency overlay`.

## Run it

Build [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090), then:

```text
scripts\download-model.bat qwen38-27b
scripts\run.bat qwen38-27b
```

or on Linux `scripts/download-model.sh qwen38-27b` and `scripts/run.sh qwen38-27b`. Direct CLI use:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 --kv-dtype rk4v4 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft
```

DFlash2 needs `--lm-head-draft`. Its candidate top-k reads either the proposal head or the full
head, and the top-k kernel has no 6-bit route, so DFlash2 with the full 6-bit head is refused.
`--embedding-q4` and `--lm-head-q6` are accepted and do nothing here, because both tensors are
already stored in those formats.

## Provenance

| Field | Value |
|---|---|
| Source | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| DFlash2 source | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Importance matrix | `imatrix_unsloth.gguf` from `unsloth/Qwen3.8-27B-GGUF` @ `4ca72078` (Apache-2.0), converted with `python -m tools.convert.imatrix` |
| Recipe | `qwen3_8_27b` in `tools/convert/official_recipes.py` |
| Converter | `python -m tools.convert --model Qwen3.8-27B --recipe qwen3_8_27b --source imatrix=qwen3_8_27b.imatrix.safetensors --source dflash2=Qwen3.8-27B-DFlash2 --components text,vision,mtp,dflash2 --proposal --name qwen3.8-27b` |

[`artifact-manifest.json`](artifact-manifest.json) has the object inventory and every source hash.

## Limits

- Vision, MTP, DFlash2 and proposal-head weights use upstream's round-to-nearest encoding. The
  imatrix covers the text layers only, and the embedding and head are searched without weights.
- The imatrix is Unsloth's general calibration set, not one tuned for this engine's workloads.
- NInfer executes on one CUDA device, or splits layers across up to eight GPUs on Linux. It does
  not do CPU offload, and it does not execute generated tool calls.

## License

Apache License 2.0, matching [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), the DFlash2
companion weights and Unsloth's importance matrix.
