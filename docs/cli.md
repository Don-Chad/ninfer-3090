# NInfer CLI

`build/apps/ninfer` runs one request against one v3 `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

The examples use Qwen3.8-27B NVFP4 with FP8 KV storage.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype int8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting NInfer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort. `--greedy` selects exact argmax
decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be
combined with `--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as
reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw
control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only
  the selected speculative backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI uses root-only context mode, so it does not reserve an extra Device
  checkpoint StateImage or capture a continuation that no later request could consume.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and does not
accelerate Vision encode. The default speculative and Vision settings produce the smallest resident
profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP, the 35B-A3B DFlash or the Qwen3.8-27B
DFlash2 backend with one to fifteen draft positions. Both masked-draft backends may be combined
with `--vision`.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The MTP draft count is a trade on what the output looks like. Each round verifies K+1 columns and
runs K draft-head steps whether or not the drafts survive, so a larger K pays only where the head
keeps guessing right. Measured on the RTX 3090, Qwen3.8-27B, `rk4v4` KV, `--lm-head-draft
--lm-head-q6 --embedding-q4 --gdn-state-fp16`, greedy, 512 generated tokens, one stream (decode
tok/s; tokens emitted per round in parentheses):

| `--draft-tokens` | edit code (copies the prompt) | write new code | explain a concept | short story |
|---:|---:|---:|---:|---:|
| (none) | 47.3 | 47.5 | 47.5 | 47.6 |
| **3** | 141.9 (3.9) | 118.2 (3.3) | 102.1 (2.8) | 77.1 (2.1) |
| 5 | 177.7 (5.7) | 124.8 (4.0) | 95.6 (3.1) | 75.1 (2.4) |
| 7 | 226.0 (7.4) | 130.6 (4.3) | 103.9 (3.4) | 67.7 (2.2) |
| 9 | 226.4 (8.8) | 129.9 (5.0) | 91.3 (3.5) | 56.3 (2.2) |
| 11 | 244.4 (10.4) | | | |
| 15 | 262.6 (12.5) | 114.7 (5.4) | 75.1 (3.5) | 47.6 (2.2) |

Three stays the default: it is best or within 2% on prose, and larger counts lose up to 38% there.
When the output mostly reproduces the input -- refactoring, renaming, applying an edit and
returning the whole file -- the draft head predicts it almost perfectly and every extra position is
nearly free, so 11 to 15 is up to 1.85x faster than three. For a coding assistant that mostly
writes new code, seven is about 10% faster than three. `--lookup-ngram` adds nothing on top of MTP
there: the head already copies (K=15 with and without `--lookup-ngram 8`: 262.4 and 262.6 tok/s,
the same tokens per round). For comparison, DFlash2 at its default seven is still faster at one
stream on the same runs (268.5 tok/s editing code, 115.7 explaining); MTP is the backend that fits
the full context and a second lane. Draft counts of eight and above add a second CUDA Graph
topology class on the 27B (its wide verify moves between the prompt and chunked attention routes),
which reserves about 64 MiB more per lane.

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype int8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 7`, optionally with `--lm-head-draft` and `--vision`; this is what
the launchers pass. DFlash2 accepts every draft count from 1 through 15. Seven is the checkpoint
recommendation and the best mean on this card after the v0.10.0 small-T kernels: swept on a
reasoning, a code and a summarisation prompt, K = 3/4/5/6/7/8/9/10/12 averaged
128.0/146.0/159.0/169.6/**172.3**/163.7/163.3/160.2/159.2 tok/s. The best K still depends on the
workload — the reasoning prompt kept improving to K=12 while summarisation fell sharply there — so
a deployment serving one kind of work should sweep its own
([performance](performance.md#choosing-a-speculative-backend-by-concurrency-rtx-3090-qwen38-27b)).

An earlier sweep, before the small-T kernels made wide verify rounds cheap, favoured four. It was
measured on the RTX 3090, 27B, INT8 KV, greedy, generating 256 tokens of ordinary prose, mean of
three runs, and is kept for the record; it no longer describes the current build:

| `--draft-tokens` | decode | vs no speculation |
|---:|---:|---:|
| (none) | 37.7 tok/s | — |
| 1 | 49.7 | +31.7% |
| 2 | 56.4 | +49.5% |
| 3 | 58.4 | +54.9% |
| **4** | **59.1** | **+56.5%** |
| 5 | 57.2 | +51.5% |
| 6 | 48.4 | +28.3% |
| 7 | 48.2 | +27.7% |
| 8 | 47.6 | +26.2% |
| 10 | 42.4 | +12.4% |
| 12 | 40.6 | +7.6% |

Seven costs 18.4% against four, which is the same fact as four being 22.6% faster than seven — only
the denominator differs. There is a distinct cliff between five and six — 57.2 to 48.4 —
which looks like a block-geometry boundary rather than an acceptance effect, since acceptance is
still rising there. `--lm-head-draft` is within noise of unset for DFlash2 at every count and can
be left off.

For reference on the same measurement, MTP3 with the draft head reaches 62.4 tok/s (+65.3%), so
MTP remains the faster backend on text — but by 5% at DFlash2's best draft count, not the 36% that
seven implies.
Both `groupwise-int` and `nvfp4` artifacts use the same Engine route, including CUDA Graph,
concurrent requests, sampling penalties, and prefix reuse. An artifact without the companion
weights reports a missing DFlash2 component when selected. Vision, MTP and DFlash follow the same
rule: their weights are required only when that component is enabled at startup.

Only one speculative backend can be enabled per Engine. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with
the optimized proposal head. DFlash accepts one to fifteen draft tokens; seven forms the measured
block length eight, while fifteen uses the maximum supported block length sixteen.

## Common options

The table lists executable defaults. The examples above select FP8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see the [README](../README.md#several-gpus-pipeline-stages---devices-ab)); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|nvfp4\|k8v4` | KV-cache storage. `rk8v4` is opt-in RotorQuant and `rk4v4` opt-in Lloyd-Max 4-bit keys; all seven are accepted on this fork's sm_86/sm_89 targets | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--lookup-ngram N` | context-lookup drafting alongside `--spec`: the last `N` tokens are matched against the sequence so far and what followed is proposed; exact, since verification rejects a wrong guess | `0` (off) |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay` | `overlay` keeps the Vision tower host-pinned and borrows device memory per image from the evictable text weight tail (no resident Vision cost; needs CUDA VMM) | `resident` |
| `--vision-max-merged N` | merged-token budget of one media item; larger media downscales at preprocessing | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture
and rendered prompt mode. The current official models use:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ninfer --help` for the exact option contract.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at
the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep;
the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto`
uses CUDA's scheduling heuristic, not an automatic performance benchmark.

```bash
NINFER_CUDA_SYNC=blocking ./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --prompt "Hello"
```

The engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. The practical allocation
on one RTX 3090 depends on the selected artifact, media workload, output budget, and KV-cache type.
Use `--kv-dtype int8` for the recommended large-context quality profile; BF16 is also available.
The artifact describes its model configuration and weight representations; `--kv-dtype`
independently selects runtime KV storage. All seven
formats — `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `k8v4`, `nvfp4` — are accepted on SM86; measured size,
decode speed and perplexity for each are in
[`docs/config-calculator.html`](config-calculator.html). The Blackwell-only
`mma.sync...kind::f8f6f4` restriction applies to FP8/NVFP4 *weights and activations*, not to KV
storage, which is why this paragraph used to say `fp8` KV was rejected here.

`rk8v4` is the best all-round choice: rotated INT8 keys with a packed signed int4 value plane,
about 23% smaller than INT8 for about 0.082% perplexity, and the flattest decode curve of any
format measured. `nvfp4` buys the most context — 45% smaller than INT8 — at about 13% of decode
speed at a 32K cache depth. `rk4v4` keeps `rk8v4`'s values and stores keys as 4-bit Lloyd-Max
indices: 31% smaller than `rk8v4`, within 3% of `nvfp4`'s size, better perplexity than `nvfp4`
(+0.21% against INT8) and `rk8v4`'s decode speed, so it is the choice when context is the limit.
`fp8` and `k8v4` are each beaten by `rk8v4` on size, speed and quality together, so neither has a
niche. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.
