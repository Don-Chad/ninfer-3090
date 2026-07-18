# Performance

This page records the representative NInfer CLI measurements used in the project README. The source
run was captured on 2026-07-18 in
`profiles/bench/qwen36_mtp_cli_repeat5_20260718`. Raw profiler output remains local; the complete
aggregated results and test conditions are preserved here.

## Test conditions

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Driver reported by the run | 591.86 |
| Prompt source | `examples/cli/messages/long_8k.json` |
| NInfer prepared prompt | 7,678 tokens |
| Context capacity | 16,384 tokens |
| Generated tokens | 512 |
| KV cache | BF16 |
| Sampling | greedy |
| NInfer prefill chunk | 1,024 |
| CUDA Graph | enabled |
| MTP draft window | 0 or 3 |
| NInfer MTP proposal head | optimized when MTP=3 |
| Warm-up | one run per configuration, excluded |
| Measurement | five runs per configuration |
| Reported statistic | mean plus sample standard deviation |
| llama.cpp build | `b201-b369ae383` |

The NInfer and llama.cpp runs used the same source prompt and 512-token generation budget, with each
engine applying its own chat path and reporting its own prefill/decode phase rate. Configurations
were alternated between MTP off and on to reduce ordering bias.

## Full results

| Model | Engine | MTP | Prefill mean ± SD | Decode mean ± SD | MTP acceptance | Accepted length |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | llama.cpp | 0 | 2,966.58 ± 4.51 tok/s | 60.60 ± 0.14 tok/s | — | — |
| Qwen3.6-27B | llama.cpp | 3 | 2,646.64 ± 30.47 tok/s | 140.38 ± 0.70 tok/s | not reported | not reported |
| Qwen3.6-27B | NInfer | 0 | 2,841.46 ± 22.44 tok/s | 66.46 ± 0.08 tok/s | — | — |
| Qwen3.6-27B | NInfer | 3 | 2,829.62 ± 6.65 tok/s | 167.22 ± 1.07 tok/s | 86.62% | 3.60 tok/round |
| Qwen3.6-35B-A3B | llama.cpp | 0 | 6,272.86 ± 268.07 tok/s | 197.80 ± 1.13 tok/s | — | — |
| Qwen3.6-35B-A3B | llama.cpp | 3 | 5,476.82 ± 312.99 tok/s | 285.88 ± 5.28 tok/s | not reported | not reported |
| Qwen3.6-35B-A3B | NInfer | 0 | 13,594.36 ± 71.12 tok/s | 244.52 ± 0.21 tok/s | — | — |
| Qwen3.6-35B-A3B | NInfer | 3 | 13,411.71 ± 111.89 tok/s | 436.32 ± 1.58 tok/s | 81.53% | 3.45 tok/round |

## Comparisons

| Model | Comparison | Prefill ratio | Decode ratio |
|---|---|---:|---:|
| Qwen3.6-27B | NInfer MTP=3 / NInfer MTP=0 | 1.00x | 2.52x |
| Qwen3.6-35B-A3B | NInfer MTP=3 / NInfer MTP=0 | 0.99x | 1.78x |
| Qwen3.6-27B | NInfer / llama.cpp at MTP=3 | 1.07x | 1.19x |
| Qwen3.6-35B-A3B | NInfer / llama.cpp at MTP=3 | 2.45x | 1.53x |

Ratios are ratios of the five-run means before display rounding.

## Artifact boundary

This is an end-to-end configuration comparison, not an identical-format kernel comparison.

NInfer used the published native artifacts:

- `qwen3_6_27b.ninfer`, 17,495,365,888 bytes;
- `qwen3_6_35b_a3b.ninfer`, 22,373,184,256 bytes.

llama.cpp used:

- `Qwen3.6-27B-Q4_K_M-mtp.gguf`;
- `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

The artifacts use different quantization and storage recipes. The comparison therefore describes
the measured complete-engine combinations and must not be interpreted as an isolated runtime or
kernel result.

## NInfer command

The measured MTP configuration for either registered artifact was:

```bash
./build/apps/ninfer "$ARTIFACT" \
  --messages examples/cli/messages/long_8k.json \
  --device 0 \
  --max-context 16384 \
  --prefill-chunk 1024 \
  --kv-dtype bf16 \
  --max-new 512 \
  --greedy \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

The MTP=0 control replaced the final two flags with `--mtp-draft-tokens 0`.

## llama.cpp command

The corresponding llama.cpp MTP configuration was:

```bash
PROMPT=$(jq -r '.[0].content' examples/cli/messages/long_8k.json)

llama-cli \
  -m "$GGUF" \
  --prompt "$PROMPT" \
  --conversation \
  --single-turn \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --ctx-size 16384 \
  --n-predict 512 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers all \
  --split-mode none \
  --main-gpu 0 \
  --flash-attn on \
  --cache-type-k bf16 \
  --cache-type-v bf16 \
  --temperature 0 \
  --seed 0 \
  --no-display-prompt \
  --color off \
  --perf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  --spec-draft-type-k bf16 \
  --spec-draft-type-v bf16
```

The MTP=0 control omitted the five `--spec-*` options.
