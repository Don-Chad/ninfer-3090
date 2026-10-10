# NInfer CLI

`build/apps/ninfer` runs one request against one v3 `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

The examples use the Qwen3.8-27B artifact that `download-model qwen38-27b` writes to
`models/qwen3_8_27b.ninfer`, with INT8 KV storage. [Common options](#common-options) lists
every flag and its default; `./build/apps/ninfer --help` is the exact option contract.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
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
./build/apps/ninfer models/qwen3_8_27b.ninfer \
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

## Constrained output

`--grammar-file FILE` constrains the answer with a GBNF grammar whose entry rule is `root`:

```bash
printf 'root ::= "yes" | "no"\n' > answer.gbnf
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Is 17 prime?" --no-thinking --grammar-file answer.gbnf --max-new 64
```

`--choice TEXT` selects a literal candidate; repeat the flag to supply the candidate set.
`--regex PATTERN` constrains the complete answer to a regular expression:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Classify this review: the service was excellent." --no-thinking \
  --choice positive --choice neutral --choice negative

./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return ticket TASK-0042." --no-thinking --regex '(BUG|TASK)-[0-9]{4}'
```

Candidates preserve exact text; `--choice ''` explicitly permits empty content. `--regex ''`
permits only empty content. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice)
for supported regex syntax.

`--json-object` constrains the answer to a JSON object. `--json-schema-file FILE` applies a JSON
Schema. All output constraint options are mutually exclusive:

```bash
printf '%s\n' '{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}' > answer.schema.json
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return the answer to 6 times 7 as JSON." --no-thinking \
  --json-schema-file answer.schema.json --max-new 64
```

GBNF supports recursive rules, Unicode character classes and repetition. All constraint modes work with
ordinary decoding, MTP, DFlash and DFlash2. Thinking may precede the constrained answer; an output
limit or cancellation can leave it incomplete. JSON modes can accompany tools supplied in messages:
the answer is either JSON or a tool-call sequence. GBNF, choice and regex require no active tools.
Constraints reject custom stops and `--raw-output`. JSON uses compact separators and declared
property order. Describe the desired content in the prompt; the schema is not added to it automatically. See the
[supported schema subset](maintainer/constrained-decoding.md#42-json-与-schema-的执行合同).

## Thinking and reasoning

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking` or
`--reasoning-effort none`, which both disable thinking, but it can be combined with any other
`--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
The inserted suffix is never truncated, and a request is never rejected for lacking room for it.
When the output capacity left after the budget cannot hold the complete tokenizer-derived control
suffix plus one post-close model token, Engine lowers the effective budget so both fit. When the
whole capacity is no larger than that, the budget cannot be enforced, so thinking runs to the
`--max-new` limit and may end inside the reasoning. The run summary prints `effective thinking
budget` when it differs from the requested one, and nothing extra when the budget is unenforced; the
exact rule is in [Chat Completions](serving.md#openai-chat-completions).

Normal output sends the inserted guidance to stderr as reasoning. `--print-token-ids` includes the
inserted IDs, while `--raw-output` preserves the raw control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

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

Run message files from the repository root when they contain repository-relative media paths. Image
and video input needs `--vision`:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
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

## Common options

The table lists executable defaults. The examples above select INT8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see the [README](../README.md#several-gpus-pipeline-stages---devices-ab)); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|nvfp4\|k8v4` | KV-cache storage; see [Context and memory](#context-and-memory). `rk8v4` is opt-in RotorQuant and `rk4v4` opt-in Lloyd-Max 4-bit keys; all seven are accepted on this fork's sm_86/sm_89 targets | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend; see [Speculative decoding](#speculative-decoding) | off |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 | unset |
| `--lm-head-draft` | optimized proposal head; implied by `--spec`, accepted for compatibility | on with `--spec` |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--no-prefill-a8` | return full prefill tiles to their A16 routes, which is how the integer routes are measured on a whole request | integer routes on |
| `--lm-head-q4\|--lm-head-q6` | narrower language-model-head matrix: a speed- or memory-for-quality trade (DFlash and DFlash2 refuse it) | off |
| `--embedding-q4\|--embedding-q6` | narrower token-embedding matrix, freeing weight memory for context | off |
| `--mtp-experts-q4` | transcode the 35B-A3B's MTP draft-layer experts to Q4/Q6 at load; rejected on the dense 27B | off |
| `--gdn-state-fp16` | FP16 recurrent GDN state, halving each state image | off |
| `--mlp-a8-decode` | integer-activation MLP at decode | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay` | `overlay` keeps the Vision tower host-pinned and borrows device memory per image from free KV pages, or the evictable text weight tail when those fall short (no resident Vision cost; needs CUDA VMM) | `resident` |
| `--vision-max-merged N` | merged-token budget of one media item; larger media downscales at preprocessing | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding, independent of the thinking options | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |
| `--stop-token-id N`, `--stop TEXT`, `--reasoning-stop TEXT` | add a stop token ID, a stop string matched in the answer content, or a stop string matched in the reasoning; each may be repeated | none |
| `--grammar-file FILE`, `--json-object`, `--json-schema-file FILE`, `--choice TEXT`, `--regex PATTERN` | constrain the answer; mutually exclusive, see [Constrained output](#constrained-output) | none |
| `--raw-output` | expose the frontend's raw output stream | off |
| `--print-token-ids` | include generated token IDs in diagnostics | off |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | diagnostic verbosity on stderr | `info` |

The `--lm-head-*`, `--embedding-*`, `--mtp-experts-q4`, `--gdn-state-fp16` and `--mlp-a8-decode`
flags are speed- or memory-for-quality trades. The measured cost of each is in
[quality trades](maintainer/quality-trade-experiments.md).

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

## Speculative decoding

Speculative decoding is disabled by default. Select MTP, the 35B-A3B DFlash or the Qwen3.8-27B
DFlash2 backend with one to fifteen draft positions. Only one backend can be enabled per Engine, and
Every backend uses the optimized proposal head: `--spec` implies it, and `--lm-head-draft` is still accepted but
requires a selected backend. Both
masked-draft backends may be combined with `--vision`:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype int8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 7`, optionally with `--vision`; this is what
the launchers pass. DFlash2 accepts every draft count from 1 through 15. Both `groupwise-int` and
`nvfp4` artifacts use the same Engine route, including CUDA Graph, concurrent requests, sampling
penalties, and prefix reuse. An artifact without the companion weights reports a missing DFlash2
component when selected. Vision, MTP and DFlash follow the same rule: their weights are required
only when that component is enabled at startup.

### Choosing a draft count

The draft count is a trade on what the output looks like. Each round verifies K+1 columns and runs
K draft-head steps whether or not the drafts survive, so a larger K pays only where the head keeps
guessing right.

- **MTP:** three is the default. It is best or within 2% on prose, and larger counts lose up to 38%
  there. When the output mostly reproduces the input -- refactoring, renaming, applying an edit and
  returning the whole file -- 11 to 15 is up to 1.85x faster than three, and for a coding assistant
  that mostly writes new code seven is about 10% faster. The former MTP-only context lookup
  (`--lookup-ngram`) added nothing on top of MTP there, because the head already copies; it has
  been removed, and n-gram copy drafting, which verifies a copy wider than the neural window, is
  being built to replace it.
- **DFlash2:** seven is the checkpoint recommendation and the best mean on this card. The best K
  still depends on the workload, so a deployment serving one kind of work should sweep its own.
  The optimized proposal head, which `--spec` now always enables, measured within noise of the full head for
  DFlash2 at every count.
- **DFlash:** seven forms the measured block length eight; fifteen uses the maximum supported block
  length sixteen.

The tables and sweeps are in [performance](performance.md#choosing-the-draft-count-rtx-3090-qwen38-27b).
The published [performance results](performance.md) use MTP with three draft tokens and DFlash with
seven, both with the optimized proposal head. Draft counts of eight and above add a second CUDA Graph
topology class on the 27B, which reserves about 64 MiB more per lane.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at
the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep;
the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto`
uses CUDA's scheduling heuristic, not an automatic performance benchmark.

```bash
NINFER_CUDA_SYNC=blocking ./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. The practical allocation
on one RTX 3090 depends on the selected artifact, media workload, output budget, and KV-cache type.
The artifact describes its model configuration and weight representations; `--kv-dtype`
independently selects runtime KV storage.

### KV storage

All seven formats — `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `k8v4`, `nvfp4` — are accepted on SM86;
measured size, decode speed and perplexity for each are in
[`docs/config-calculator.html`](config-calculator.html). The Blackwell-only
`mma.sync...kind::f8f6f4` restriction applies to FP8/NVFP4 *weights and activations*, not to KV
storage.

- `bf16` is the executable default.
- `int8` is the quality reference: use it when quality matters more than context.
- `rk8v4` is the best all-round choice: rotated INT8 keys with a packed signed int4 value plane,
  about 23% smaller than INT8 for about 0.082% perplexity, and the flattest decode curve of any
  format measured.
- `rk4v4` keeps `rk8v4`'s values and stores keys as 4-bit Lloyd-Max indices: 31% smaller than
  `rk8v4`, within 3% of `nvfp4`'s size, better perplexity than `nvfp4` (+0.21% against INT8) and
  `rk8v4`'s decode speed, so it is the choice when context is the limit. The launchers use it.
- `nvfp4` buys the most context, 45% smaller than INT8, at about 13% of decode speed at a 32K
  cache depth.
- `fp8` and `k8v4` are each beaten by `rk8v4` on size, speed and quality together, so neither has a
  niche.

### Capacity

The prepared prompt must fit `--max-context`; generation stops at the remaining context capacity
when necessary. `--kv-capacity N` controls the shared physical Main Text KV pool independently and
is rounded up to the 64-token page size. `--kv-capacity auto` loads the selected weights, measures
the remaining GPU memory, and directly chooses the largest legal page capacity for the complete
enabled runtime layout. This includes the selected speculative backend, fixed sequence state,
unified workspace, and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows `--max-context`; the distinction matters
primarily to a concurrent Engine or server.

### Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only the selected
  speculative backend;
- the optimized proposal head is loaded whenever a speculative backend is selected;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input;
- the one-request CLI disables cross-request history and Host context backing, so it does not
  reserve an extra Device checkpoint StateImage or retain continuations.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and do
not accelerate Vision encode. The default speculative and Vision settings produce the smallest
resident profile.

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
