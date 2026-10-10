# NInfer-3090

[![Release](https://img.shields.io/github/v/release/ashalliants/ninfer-3090?label=release&color=2a78d6)](https://github.com/ashalliants/ninfer-3090/releases/latest)
[![License](https://img.shields.io/github/license/ashalliants/ninfer-3090?color=2a78d6)](LICENSE)
[![Host checks](https://github.com/ashalliants/ninfer-3090/actions/workflows/host-checks.yml/badge.svg)](https://github.com/ashalliants/ninfer-3090/actions/workflows/host-checks.yml)

![NInfer-3090 throughput on one RTX 3090](docs/assets/perf-banner.svg)

NInfer-3090 is a specialized C++20/CUDA inference engine for **Qwen3.8-27B** and Qwen3.6 on one
24 GB NVIDIA GeForce RTX 3090, or split as a pipeline across several GPUs on Linux. The native SM86
runtime loads the official groupwise `.ninfer` artifacts, serves OpenAI- and Anthropic-compatible
APIs, and supports paged KV, compatible-prefix reuse, structured JSON output, CUDA Graphs, MTP and
DFlash2 speculative decoding, vision, reasoning-effort control, ReplaySSM state transactions, and
concurrent cohorts through **C8**.

**One RTX 3090, Linux launcher defaults (`rk4v4` KV):**

| | Qwen3.8-27B (dense) | Qwen3.6-35B-A3B (MoE) |
|---|---:|---:|
| **Max context** | **262,144** tokens (native) | **262,144** tokens (native), shared by 3 lanes |
| **Prefill**, 4K prompt | **3,185 tok/s** | **8,848 tok/s** |
| **Decode**, one stream | **140 tok/s** (DFlash2) | **295 tok/s** (MTP3 + draft head) |
| **Decode**, all lanes | **477 tok/s** at C8 (MTP3) | **383 tok/s** at C6 (MTP3) |

The 27B speed rows were re-measured on 2026-10-02 for the 3090 conversion `download-model` now
fetches, in one sitting with upstream's artifact on the same binaries. Decode is
`tools/bench/run_chat_decode.py` on syv-ai's eight thinking-off prompts (1,024 tokens, INT8 KV,
greedy), where upstream's artifact gives 127 tok/s at C1 and 482 at C8; prefill is `ninfer_bench`
pp4096 on the cuBLAS route the launcher enables, `rk4v4` KV (upstream 3,196, a tie; +0.156%
perplexity for the route). The earlier 187 and 523 came from runs whose workload was not recorded
and that this harness does not reproduce on either artifact. The 35B prefill was measured the same day with its launcher settings (cuBLAS route, chunk 4096, `rk4v4`; 5,470 tok/s on the previous default-route chunk 512); its decode rows were not re-measured.
The 27B's 262,144 was measured with MTP3 and the draft head; the default DFlash2 profile reaches it
on a headless card by extrapolation and steps down if refused. Sources and conditions are in
[Performance on an RTX 3090](#performance-on-an-rtx-3090).

![Qwen3.8-27B: upstream artifact vs this fork's 3090 conversion](docs/assets/qwen38-artifact-gains.svg)

The default 27B is this fork's own conversion ([model card](model-cards/Qwen3.8-27B-NInfer-3090/README.md)):
an importance-weighted encoder with signed scales and fewer bytes per decoded token, for a smaller
file, output closer to full precision, and faster decode on the same card.

The goal is the most rippin' Qwen inference stack for the 3000 series. It is a community project
maintained on a best-effort basis: issues and PRs are very welcome, but support and feature
requests are not guaranteed.

> **Sponsored by [NeverMetered](https://nevermetered.com/?ref=ninfer-3090)** — hosted Qwen3.8-27B for coding agents,
> priced by parallel streams instead of tokens. No per-token bills, no 5-hour windows and no weekly
> caps: point Claude Code, Codex, Cursor, Cline or anything else that speaks the OpenAI or
> Anthropic API at it and let your agents run all day. No 3090 of your own, or want your agents off
> your desktop's GPU? Plans start with a day pass, and every verified account gets a few free
> requests a day. Thank you to NeverMetered for supporting this project.

**New in v0.14.2:** long prompts no longer have to block short ones. `--max-prefill-lanes N` reads
several prompts in at once (default 2 with three or more `--max-concurrency` lanes; a short request behind a 3,000-token prompt: 2.5 s to 1.5 s to first word),
`GET /slots` reports each retained conversation's reuse, and `--host-cache-max-mib` caps the
RAM cache on shared machines. The rest are opt-in. See the [v0.14.2 release notes](RELEASE_NOTES_0.14.2.md).

**v0.14.1:** a stability fix for `ninfer-serve`. A large or unplannable request no longer fails
every running request (`internal error generation`) or clears the context cache; only that request
fails, and a latched engine now exits so a supervisor can restart it. See the
[v0.14.1 release notes](RELEASE_NOTES_0.14.1.md).

**v0.14.0: `--auto-host-cache`.** One flag now sizes the RAM tier of the context cache from
the machine, so a server that exists to serve long agent conversations keeps them cached after they
leave the GPU instead of re-reading them from scratch. It replaces four hand-tuned flags and respects
a container's memory limit. Also: vision together with a multi-GPU `--devices` split, a
`download-model` step that fetches the `godmode` graft when you have access, a repetition guard in
the tuned launcher profiles, and a compiler cache that cuts a full rebuild from over ten minutes to
about one and a half. 5 merged PRs — see the [v0.14.0 release notes](RELEASE_NOTES_0.14.0.md) for what
each does and what was not verified.

Previous: [v0.13.0](RELEASE_NOTES_0.13.0.md) (a smaller, closer-to-Q8 Qwen3.8-27B, per-request
`thinking_budget`), [v0.12.0](RELEASE_NOTES_0.12.0.md) (real multi-GPU, structured JSON output, a
self-healing engine), [v0.11.0](RELEASE_NOTES_0.11.0.md) (prefill roughly 2x faster, DFlash2 the recommended
decode backend), [v0.10.0](RELEASE_NOTES_0.10.0.md) (tensor-core small-T kernels: MTP3 decode 1.5x
at C1 and 1.7x at C8), [v0.9.1](RELEASE_NOTES_0.9.1.md), [v0.9.0](RELEASE_NOTES_0.9.0.md).

> **Model files are v3.** A v2 `.ninfer` from an earlier release is refused at load. Upgrade it
> instead of re-downloading — see [Models](#models).

## Contents

- [Quick start](#quick-start) — download, fetch a model, run
- [Building from source](#building-from-source)
- [Features](#features) and [serving APIs](#serving-apis)
- [Models](#models)
- [Performance on an RTX 3090](#performance-on-an-rtx-3090)
- [Choosing a KV format](#choosing-a-kv-format)
- [Several GPUs](#several-gpus-pipeline-stages---devices-ab)
- [Current limits](#current-limits)
- [Upstream, contributors and contributing](#upstream)

## Quick start

**You do not need to build anything.** Prebuilt archives for Windows x64 and Linux x64 are on
[the latest release](https://github.com/ashalliants/ninfer-3090/releases/latest), each with
`ninfer-serve`, the CLI, the benchmark tool, every launcher and, on Windows, the DLLs. Both
platforms need an RTX 3090 or 3090 Ti and a recent NVIDIA driver.

**Windows 11** — download `ninfer-rtx3090-windows-x64-*.zip`, unzip it, and from that folder:

```powershell
.\download-model.bat qwen38-27b          # downloads qwen3_8_27b.ninfer (~19 GB, resumable)
.\run.bat qwen38-27b                      # serves on 127.0.0.1:8080, one user, 188,416 tokens, DFlash2
```

Double-clicking either file asks which model instead; Qwen3.8-27B is the recommended choice. The
[MoE alternative](#qwen36-35b-a3b), `qwen36-35b-a3b`, serves more lanes.

**Linux** — download `ninfer-rtx3090-linux-x64-*.tar.gz`, and:

```bash
tar -xzf ninfer-rtx3090-linux-x64-*.tar.gz && cd ninfer-rtx3090-linux-x64-*/
./download-model.sh qwen38-27b           # downloads qwen3_8_27b.ninfer (~19 GB, resumable)
./run.sh qwen38-27b                      # one user, 262,144 tokens (headless), DFlash2 + draft head, vision
```

Then point your harness at `http://127.0.0.1:8080/v1`. It is an OpenAI-compatible endpoint (the
Anthropic Messages API is served too), so anything that speaks `/v1/chat/completions` works; leave
the API key blank.

The downloaders fetch `qwen38-27b`, `qwen36-35b-a3b` or `qwen36-27b`. Each pins a HuggingFace
revision, stages under a revision-scoped name so a resume can only ever continue the same artifact,
and verifies size and SHA-256 before promoting it.

The platform guides cover GPU checks, Docker, native builds and model mounts:
[Linux](docs/rtx-3090-linux.md) · [Windows](docs/rtx-3090-windows.md).

### Launchers and profiles

| Command (`run.bat` / `run.sh`) | Best for |
|---|---|
| `run qwen38-27b` | **Recommended.** Qwen3.8-27B, one user, DFlash2, cuBLAS prefill; 188K on Windows, 262K headless |
| `NINFER_SPEC=mtp` + `run qwen38-27b` | Qwen3.8-27B at the full 262K with two lanes, MTP3 instead of DFlash2 |
| `run qwen36-35b-a3b` | Qwen3.6-35B-A3B, MTP3, cuBLAS prefill, vision; 213K with two lanes on Windows, 262K with three on Linux |
| `run qwen38-27b int8` | Reference profile: one user, INT8 KV (the quality default), 64K context |
| `run qwen38-27b c8` | Reference profile: eight lanes at 8K, highest aggregate throughput |

Every default profile serves images (vision in overlay residency, which costs about 10 MiB of
device reservation), uses `rk4v4` KV, and backs the context cache with 8 GiB of pinned host RAM
(`--host-context-mib 8192`): one budget that retained conversation state, KV pages and the snapshots
of paused requests share once they leave the card. It is host memory, not GPU memory, and Windows
now pins all of it too (earlier releases clamped it there to about 4.6 GiB).

**Overrides**, from the environment: `NINFER_HOST`, `NINFER_PORT`, `NINFER_MODEL`, `NINFER_SERVER`,
`NINFER_CHAT_TEMPLATE` for every profile, plus `NINFER_CONTEXT`, `NINFER_CONCURRENCY`,
`NINFER_KV_CAPACITY`, `NINFER_KV_DTYPE`, `NINFER_SPEC`, `NINFER_DRAFT_TOKENS`, `NINFER_PREFILL_CHUNK`,
`NINFER_VISION`, `NINFER_HOST_CONTEXT_MIB` (8192), `NINFER_MIN_P` (0.03) and `NINFER_PRESENCE_PENALTY` (0.5) for the
default ones; the last two are the loop guard, and `default` leaves the registered sampling preset in force. Other serving flags are opt-in overrides too: `NINFER_AUTO_HOST_CACHE=on`, `NINFER_MAX_OUTPUT_TOKENS`, `NINFER_LOOKUP_NGRAM`, `NINFER_MLP_A8_DECODE=on` and `NINFER_CONTEXT_STORE` (listed with their caps at the top of `run.sh` / `run.bat`). The launchers bind `127.0.0.1`;
`NINFER_HOST=0.0.0.0` exposes the server to the LAN, **unauthenticated**. On Windows,
`set NINFER_SPEC=mtp && run.bat qwen38-27b`; on Linux, `NINFER_SPEC=mtp ./run.sh qwen38-27b`.

**Custom chat templates.** `NINFER_CHAT_TEMPLATE` points the launcher at a local Jinja file, which
overrides the artifact's built-in template for that run — handy for a model-specific fixed template
instead of the maintained one. It maps straight to `ninfer-serve`'s own `--chat-template FILE`
(`docs/serving.md`), so it works the same way if you drive the server binary directly, e.g. from an
unpacked release archive without the launcher:

```bash
# Linux, release archive root (add whatever other serving flags you'd normally pass)
./ninfer-serve models/qwen3_8_27b.ninfer --host 127.0.0.1 --port 8080 \
  --chat-template my-template.jinja
```

```powershell
# Windows, release archive root (add whatever other serving flags you'd normally pass)
ninfer-serve.exe models\qwen3_8_27b.ninfer --host 127.0.0.1 --port 8080 `
  --chat-template my-template.jinja
```

The template engine (`third_party/llama-jinja`) is from the same family llama.cpp uses, so templates
written for llama.cpp — including third-party "fixed" chat templates distributed for specific
models — generally work unmodified. Changes to the file take effect on the next restart.

**Lanes share one KV pool.** `--kv-capacity` is the pool and `--max-context` the per-request cap,
and the launchers set both to the profile's context. Any one request can use the full context, but
the lanes' requests together hold at most that many tokens at a time, so two users can't each keep a
200K conversation resident. Requests take KV pages as they advance rather than reserving them up
front; when the pool runs short the server first drops retained cache entries, then pauses the
younger request (saving a snapshot to host memory, or replaying its tokens later) and resumes it
once there is room. A lane adds only its fixed state, not a second pool.

**If startup refuses** for lack of GPU memory, the default profiles step down on their own — an
eighth of the context at a time, up to five times — and say what they did. An explicit
`NINFER_CONTEXT`, `NINFER_PREFILL_CHUNK`, `NINFER_HOST_CONTEXT_MIB` or `NINFER_KV_CAPACITY` is
honoured as given and fails loudly; `NINFER_FALLBACK=off` turns the step-down off. By hand, drop a
lane or a context rung first, then speculation (worth 992 MiB on the 35B-A3B), and vision last.

**To size a profile that is not listed, open [`docs/config-calculator.html`](docs/config-calculator.html)
in a browser.** It is a single self-contained file with no network access that takes a model, a
KV format and a context length and tells you whether it fits in 24 GiB, the largest context you
could run instead, and what the choice costs in decode speed and perplexity. Every constant in it
is measured on an RTX 3090 against this fork.

The measurement history behind every default is in
[launcher profiles](docs/maintainer/launcher-profiles.md).

### Qwen3.8-27B

The recommended model. A dense model rather than an MoE, so slower per token but more predictable:

```powershell
.\download-model.bat qwen38-27b          # downloads qwen3_8_27b.ninfer (~19 GB, resumable)
.\run.bat qwen38-27b                      # one user, 188,416 tokens, DFlash2, cuBLAS prefill, rk4v4, vision
```

```bash
./download-model.sh qwen38-27b
./run.sh qwen38-27b                       # one user, 262,144 tokens (headless), same flags
```

The default is the fast profile — about 1.7x the previous prefill and 1.39x the decode:

```
--spec dflash2 --draft-tokens 7 --lm-head-draft --prefill-cublas --prefill-chunk 4096
--kv-dtype rk4v4 --gdn-state-fp16 --vision --vision-residency overlay
```

`NINFER_SPEC=mtp` swaps to `--spec mtp --draft-tokens 3` and `--prefill-chunk 2048`, which runs
the full 262,144 tokens with two lanes sharing the pool:

| Profile | lanes | context | KV | measured beside a desktop |
|---|---|---|---|---|
| **`tuned`** (default, DFlash2), Windows | 1 | 188,416 | rk4v4 | starts with 721 MiB spare; 196,608 refused |
| `tuned` (default, DFlash2), Linux | 1 | 262,144 | rk4v4 | headless extrapolation; steps down if refused |
| `NINFER_SPEC=mtp` | 2 | 262,144 | rk4v4 | 23.4 of 24.5 GiB used |
| `int8` | 1 | 65,536 | int8 | 2.85 GiB left unused |

DFlash2 takes one lane because its advantage is largest at one stream (+38.6% decode at C1, +31.6%
at C2), and its draft weights use most of the ~1.45 GB that the full context needs beside a
desktop. The 27B is tighter than the 35B-A3B because of the model, not the
tuning: 16 full-attention layers × 4 KV heads × 256 head_dim is **3.2× the KV per token** of the
35B-A3B's 10 × 2 × 256. The linear memory model behind these contexts is in
[launcher profiles](docs/maintainer/launcher-profiles.md#qwen38-27b-tuned).

**Fast prefill costs context.** The `tuned` profile's cuBLAS prefill at chunk 4096 is part of every
context figure above. Its runtime reservation against leaner settings, measured 2026-10-02
(`ninfer_bench`, DFlash2 K=7, `rk4v4`, 4K prompt; about 17 KiB of KV per token):

| Prefill setting | runtime reservation | prefill | context it costs |
|---|---:|---:|---:|
| **cuBLAS, chunk 4096 (launcher default)** | **1,536 MiB** | **3,246 tok/s** | — |
| default route, chunk 4096 | 1,285 MiB | 1,900 tok/s | cuBLAS itself: ~250 MiB, ~15K tokens |
| cuBLAS, chunk 2048 | 1,298 MiB | 2,975 tok/s | ~240 MiB, ~14K tokens |
| default route, chunk 1024 | 797 MiB | 1,773 tok/s | whole setup: ~740 MiB, ~44K tokens |

On the 27B the speed comes from the cuBLAS route itself — a larger chunk on the default route barely
helps — so the launcher trades about 44K tokens of context for 83% faster prefill, which pays for
itself in agent sessions that ingest long prompts. `NINFER_PREFILL_CHUNK=2048` keeps 92% of the
speed for ~14K tokens back. (The 35B-A3B makes the same trade differently: there the chunk size,
not the route, carries most of the gain; see [its section](#qwen36-35b-a3b).)

The default Qwen3.8-27B artifact stores its token embedding as Q4 and its head as Q6, so the
`--embedding-q4` and `--lm-head-q6` load-time transcodes the upstream file needed are not passed.
`--gdn-state-fp16` (FP16 recurrent state, -72 MiB per device state slot) is measured free on
quality. The 27B's StateImage is
74.5 MiB with the FP16 state; retained ones beyond the card's slots, KV pages and paused requests'
snapshots share the **8 GiB of pinned host RAM** that `--host-context-mib 8192` sets aside — host,
not device. Lower it with `NINFER_HOST_CONTEXT_MIB` if the box is short on RAM (`0` keeps the cache
on the card).

### Qwen3.6-35B-A3B

The MoE alternative, for more lanes. The launchers run `rk4v4` KV (twice INT8's context per GiB for
+0.21% perplexity), MTP3 speculation plus the draft head, vision, and since 2026-10-02 the cuBLAS
prefill route at chunk 4096 (8,848 tok/s on a 4K prompt, against 5,470 on the old default route at
chunk 512). That route's larger runtime reservation costs context beside a desktop:

| Profile (`rk4v4`, MTP3 + draft, vision) | lanes | context | starts beside a desktop |
|---|---|---|---|
| **Linux — default** | 3 | 262,144 | not re-measured with the cuBLAS route; steps down if refused |
| **Windows — default** | 2 | 212,992 | yes, 220 MiB free (229,376 refused, 27 MB short) |
| `NINFER_PREFILL_CHUNK=1024` on Windows | 2 | 262,144 | yes, 138 MiB free; 7,140 tok/s prefill, default route |
| `NINFER_CONCURRENCY=4` | 4 | 262,144 | no, 48 MB short; a headless card should fit |

Measured 2026-09-24. With `rk8v4` the same profile needed the ~1.5 GiB a desktop holds, so Windows
ran one user at 147,456 tokens; `rk4v4` stores the same context in 31% less memory.

## Building from source

Only needed if you are changing the code — the release archives are prebuilt for `sm_86`.

`scripts/build.ps1` (Windows) and `scripts/build.sh` (Linux/WSL) pin the toolchain this project
needs and fail with a message naming the real cause when one is missing. Three things are not the
defaults on a typical machine: MSVC 14.4x from **VS 2022 BuildTools** (CUDA 12.8 rejects VS 2026's
14.50), **CUDA 12.8** forced through `CUDACXX`, and the **Ninja** generator. Both builds need CUDA
12.8 or newer and CMake 3.28 or newer; Linux uses GCC 13 with system packages or the pinned vcpkg
manifest.

```powershell
.\scripts\build.ps1                  # configure + build into build-ninja
.\scripts\build.ps1 -Test            # ... and run the test suite
.\scripts\build.ps1 -Package         # ... and build the release archive
```

```bash
./scripts/build.sh --test --package
```

On Bazzite and other distributions the Dockerfile is the shortest path:
`docker build --tag ninfer-3090:sm86 .`. NixOS is supported through `flake.nix`. The
[Linux](docs/rtx-3090-linux.md) and [Windows](docs/rtx-3090-windows.md) guides have the details.

## Features

- Native SM86 CLI and server applications for Linux and Windows, with prebuilt release archives and
  launchers that pick measured flags.
- OpenAI Chat Completions, Responses, and Anthropic Messages APIs.
- Structured JSON output (`response_format` / `text.format` / `output_config.format`), enforced
  during sampling under every speculative backend.
- MTP3 and DFlash2 speculative decoding, with ReplaySSM keeping speculation inside 24 GB.
- Seven KV-cache formats, from `bf16` to the 4-bit-key `rk4v4` that reaches the native 262K context.
- Image understanding on Qwen3.8-27B and Qwen3.6-35B-A3B, with overlay residency that keeps the
  vision tower off the device between images.
- Prefix reuse and a host-tier context cache for repeated or shared prompts, backed by one pinned
  Host budget (`--host-context-mib`) that `--auto-host-cache` can size from the machine's free RAM.
- `none`, `low`, `medium`, and `xhigh` reasoning effort on Qwen3.8.
- Concurrent cohorts of one to eight requests.
- Layer-pipeline execution across several GPUs on Linux, verified on real 2x RTX 3090 and 2x RTX
  A4000 hardware.
- Per-request prompt grafts: named phantom-KV prefixes injected by token replay or direct KV
  injection, including under MTP and DFlash2 speculation.

## Serving APIs

The server supports:

- OpenAI Chat Completions;
- OpenAI Responses Core with streaming and local continuation state;
- Anthropic Messages;
- constrained output: JSON mode, JSON Schema, GBNF grammar, choice and regex, plus strict and
  forced tool calls ([docs](docs/serving.md#output-constraints));
- compatible-prefix reuse of retained conversations, with a checkpoint at the stable boundary before
  each response, so a client that resends the last reply rewritten (its reasoning dropped, say)
  recomputes only from there; editing an earlier message may fall back to an earlier checkpoint or
  a full prefill;
- retained conversations survive a restart in a local context store (`--context-store`),
  optionally copied to an S3-compatible bucket ([docs](docs/serving.md#context-store));
- Prometheus metrics at `GET /metrics` ([docs](docs/serving.md#metrics)) and a read-only `GET /props`;
- prompt-rendered function tools and parsed tool calls (returned to the client, not executed);
- bounded pending-request admission and JSONL request logs.

See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

### Qwen3.8 reasoning effort

Qwen3.8-27B supports distinct reasoning-effort modes. `xhigh` injects the checkpoint's extended
deliberation instruction, asking it to validate assumptions and consider alternatives — a real
prompt-template change, not a sampling alias.

| Value | Qwen3.8 behavior |
|---|---|
| `none` | Disable thinking |
| `low` | Keep reasoning brief and focused |
| `medium` | Use normal Qwen3.8 thinking |
| `xhigh` | Use extended deliberation and verification |

OpenAI Chat Completions accepts a top-level `reasoning_effort` field:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Solve this carefully..."}],
  "reasoning_effort": "xhigh",
  "max_tokens": 4096
}
```

OpenAI Responses uses `"reasoning": {"effort": "xhigh"}`. Anthropic Messages uses
`"output_config": {"effort": "xhigh"}`. For the native CLI, pass
`--reasoning-effort low|medium|xhigh`, or `--no-thinking` to disable reasoning. Chat Completions
returns hidden reasoning separately as `message.reasoning_content`.

### Speculative decoding with DFlash2

Qwen3.8-27B artifacts carrying the DFlash2 companion weights support `--spec dflash2` with draft
counts 1..15 and either full or optimized proposal heads; [CLI usage](docs/cli.md) discusses the
draft count. **DFlash2 runs with `--vision`**, verified against the committed `image_chart`
fixture: target verification carries its own continuation RoPE position, so a multimodal row keeps
its per-sequence `rope_delta` through verification. On that fixture it accepts 85.7% of drafts
(7.00 tokens/round) and produces byte-identical output to the non-speculative vision run.
`--spec dflash` (v1) is a 35B-A3B backend and is refused by a 27B artifact.

### How cohort batching works

The C number is the maximum number of requests NInfer can run together. C1 favors one interactive
user; C8 can combine up to eight active requests into each GPU step for much higher total output.

Follow-up requests do not need to arrive at the same instant. When a running request finishes, the
next waiting request can join at a safe generation boundary. Empty or finished lanes are skipped,
so a C8 server also works normally with only one, two, or four active users.

This is deliberately more bounded than datacenter-style dynamic batching. The maximum number of
users and GPU memory are chosen when the server starts. In return, memory use stays predictable on
a 24 GB card and the server can reuse fast CUDA Graphs instead of rebuilding work continuously.

## Models

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| **Qwen3.8-27B** | [pinned v3 artifact](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NInfer-3090/tree/d47f2732d369acaec76dc44228f67c72b081df2f) | 17.68 GiB | **Recommended.** Validated at C1–C8 with ReplaySSM. This fork's [3090 conversion](model-cards/Qwen3.8-27B-NInfer-3090/README.md): imatrix-weighted encoder, 4-bit embedding, 6-bit head. Carries vision, MTP and the DFlash2 bundle for `--spec dflash2` |
| Qwen3.6-35B-A3B | [pinned v3 artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6) | 21.23 GiB | Carries the DFlash bundle for `--spec dflash` |
| Qwen3.6-27B | [pinned v3 artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer/tree/3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829) | 16.29 GiB | Supported with more runtime headroom |

`download-model qwen38-27b`, `qwen36-35b-a3b` or `qwen36-27b` (`.bat` on Windows, `.sh` on Linux)
fetches the pinned revision into `models/` and verifies it. **Optional bundles cost nothing in VRAM
unless selected**: the DFlash and DFlash2 weights are bound only with `--spec dflash`/`--spec
dflash2`.

**Upgrading a v2 file.** This release reads only the v3 `.ninfer` container; v1 and v2 files are
refused at load. Upgrade an existing official v2 file instead of re-downloading it:

```text
python tools/upgrade_ninfer_v2_to_v3.py models/qwen3_8_27b.ninfer models/qwen3_8_27b.v3.ninfer
```

The upgrade needs only the Python standard library, runs on Windows and Linux, keeps every weight
byte, and installs the maintained chat template ([weight conversion](docs/weight-conversion.md#upgrade-an-existing-v2-artifact)).
A load error about the container version means either an older executable reading a v3 file, or
this executable reading a v1/v2 file that needs the upgrade. Many of the published RTX 3090 figures
were measured against the v2 revisions (`18dfc887` for Qwen3.8-27B, `c8b8c1c0`/`560f227e` for
Qwen3.6-35B-A3B), whose weight bytes are unchanged in v3. Qwen3.8-27B figures published before
2026-10 used upstream's weights (`neroued/Qwen3.8-27B-NInfer` `1cbd84e7`, 19.03 GiB); the current
pin is a different encoding with fewer weight bytes per token and needs this fork's executables.

**Every downloader shipped here is pinned and verified, with one deliberate exception.** The Nix
app `download-qwen36-35b-v2` tracks upstream `main` to try a newer artifact than the measured one,
so it has no size or hash to check and cannot safely resume. It writes to its own filename, refuses
to resume, and is not the artifact the tests or the published figures use.

## Performance on an RTX 3090

Headline numbers; the full measurements, methods, rejected experiments and earlier baselines are in
[performance](docs/performance.md#rtx-3090-sm_86-findings--this-fork). Upstream's per-model pages
under `docs/performance/` were measured on an RTX 5090 and are not a statement about this fork.

### Qwen3.8-27B

**Decode by speculative backend and concurrency.** Aggregate decode tok/s through the serving
route, thinking off, greedy; measured 2026-09-19 on upstream's artifact with a workload that was not
recorded (the headline table above is the reproducible re-measurement):

| C | MTP3 | DFlash2 K=7 | change |
|---:|---:|---:|---:|
| 1 | 135.0 | **187.1** | +38.6% |
| 2 | 238.2 | **313.5** | +31.6% |
| 4 | 387.4 | **406.2** | +4.9% |
| 8 | **522.8** | does not fit | — |

**Prefill** reaches 2,989 tok/s at 4K with `--prefill-cublas --prefill-chunk 4096` (+0.156%
perplexity) and 1,649 tok/s on the default route. NInfer processes one long prefill at a time, so
cohort batching speeds up decode but not prompt ingestion.

**Against vLLM on the same card.** On the eight thinking-off prompts of
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) with MTP3 and INT8 KV,
NInfer decodes 118.9 tok/s at C1 and 477 at C8 with the default 27B artifact (2026-10-02;
upstream's artifact on the same binaries: 112.7 and 482), against the 111-124 and 407.3 that
project reports for patched vLLM. Their card is capped at 250 W against this one's 350 W and they report 5-8%
run-to-run spread, so C1 is parity and C8 a lead, measured on different machines. That comparison
predates DFlash2 and the cuBLAS prefill route.

An opt-in integer-activation MLP route for cohort decode (`--mlp-a8-decode`) adds about 1.3% at C8
for a small, oracle-bounded change in output;
[details](docs/performance.md#integer-activation-mlp-at-decode---mlp-a8-decode-opt-in).

### Qwen3.6-35B-A3B

Measured with the compact 20.84 GiB 35B-A3B artifact, a 4K shared INT8 paged KV pool, CUDA Graphs,
MTP3, greedy decoding, and no competing GPU workload:

| Concurrent requests | 128 output tokens each | Observed VRAM |
|---:|---:|---:|
| 1 | 162.7 aggregate tok/s | 22,427 MiB |
| 2 | 267.9 aggregate tok/s | 22,743 MiB |
| 4 | 366.2 aggregate tok/s | 23,377 MiB |
| 6 | 383.4 aggregate tok/s | 24,038 MiB |
| 8 | rejected at startup | about 503 MiB over the safe reservation limit |

Single-stream decode with the launcher's speculation (MTP3, draft head, `--mtp-experts-q4
--gdn-state-fp16`) measures **294.9 tok/s** in `ninfer_bench` tg512 with `rk8v4` KV, against 183.5
without speculation ([launcher profiles](docs/maintainer/launcher-profiles.md#qwen36-35b-a3b-tuned)).
No prefill figure for the 35B-A3B has been measured on a 3090 yet.

A longer 512-token-per-request check reached **286.8 tok/s at C1** and **399.1 aggregate tok/s at
C2**. Compatible-prefix reuse was validated end to end: a repeated 26-token prompt reused 24 tokens,
reducing measured prefill from 371 ms to 10 ms.

### Vision

- **Qwen3.8-27B**: a 1,920×1,080 image expanded to 2,074 prompt tokens and was read correctly, with
  MTP3, INT8 KV and a 32K context: 3.29 s TTFT, 98.1 tok/s decode, 96.7% MTP acceptance.
- **Qwen3.6-35B-A3B**: three 1,920×1,080 images, 2,081 prompt tokens each, were read correctly with
  3.76–4.07 s TTFT and about 159 tok/s decode. That v0.6 test ran without speculation at 32K,
  because vision was then resident on the device; the overlay residency the launchers now use is
  what lets vision run beside MTP3 at the full context.

The artifacts also declare multi-image and video support; these tests validated still images.

## Choosing a KV format

All seven SM86 KV formats, measured on Qwen3.8-27B. Size and perplexity are what most people weigh;
the decode column is the one that surprises, because the smallest formats are not the fastest.

| KV profile | Bytes/token | KV at 2,048 tokens | Perplexity | vs `bf16` | Decode at 32K depth |
|---|---:|---:|---:|---:|---:|
| `bf16` | 65,536 | 128.00 MiB | 4.342517 | — | 31.93 tok/s |
| `int8` | 33,792 | 66.00 MiB | 4.342425 | −0.0021% | **33.63 tok/s** |
| `fp8` | 33,024 | 64.50 MiB | 4.344724 | +0.0508% | 30.34 tok/s |
| `rk8v4` | 26,112 | 51.00 MiB | 4.346413 | +0.0897% | 33.17 tok/s |
| `k8v4` | 25,728 | 50.25 MiB | 4.347258 | +0.1092% | 28.90 tok/s |
| `nvfp4` | 18,432 | 36.00 MiB | 4.352201 | +0.2229% | 29.86 tok/s |
| `rk4v4` | **17,920** | **35.00 MiB** | see [below](#lloyd-max-4-bit-keys-rk4v4) | +0.214% vs `int8` | ≈ `rk8v4` |

Perplexity is `ninfer-perplexity` on the fixed `ninfer-ppl-1m-v1` corpus, `--quick`, context/stride
4096/2048, 261,167 scored tokens — the same corpus and window for every row. Decode is 128 timed
steps on top of a 32,768-token prefill, no speculation, and is the **mean of two independent runs**
because this card is power-capped at 315 W and the SM clock drifts 1,665–1,755 MHz with
temperature; single runs on the 27B vary by up to 5%. Attention re-reads the whole cache each step,
so a format's cost only shows at depth. The `rk4v4` row was measured in a later session on a newer
build, so it is stated against `int8` and `rk8v4` from that session rather than mixed into these
columns.

**Which to use:**

- **`int8`** is the quality default. Its perplexity is indistinguishable from `bf16` — within 0.0001,
  below this harness's own reproducibility — and it has the flattest decode curve on the 27B (−4.8%
  from 4K to 32K).
- **`rk4v4`** is what the launchers use, for context: 31% smaller than `rk8v4` at the same decode and
  prefill speed, for +0.214% perplexity over `int8`. It is the format that reaches the native
  262,144-token context on the 27B, even alongside MTP3 and the draft head.
- **`rk8v4`** sits between them: 23% smaller than `int8` for +0.09% perplexity, the flattest curve on
  the 35B (−9.5%), and on that model the fastest format at every depth measured.
- **`nvfp4`** is superseded by `rk4v4`, which is 3% smaller, has better perplexity, and decodes 19%
  faster on the 35B at 32K.
- **`fp8`** has slightly better perplexity than `rk8v4` (4.344724 against 4.346413, a real and
  reproducible 0.039%) for 26% more KV and 8.5% less decode speed at 32K — a genuine trade, and a
  bad one for almost everyone.
- **`k8v4`** is dominated: 1.5% smaller than `rk8v4`, with 13% less decode speed at depth, the worst
  falloff on the 35B (−25.9%), and slightly worse perplexity.

**These numbers were re-measured in September 2026.** `fp8`, `nvfp4` and `k8v4` had their perplexity
scored through a data race in the quantized attention kernels that corrupted every prefill output
column except the last. Greedy generation was unaffected, which is why it went unnoticed. With that
fixed, `nvfp4` improved by 0.154% and `fp8` by 0.057%, against a 0.019% drift on the formats the fix
did not touch, and all three now reproduce bit-identically run to run.

**Speculation shifts the trade.** Lower value precision lowers draft acceptance. Without
speculation, decode measured 39.07 tok/s on `int8` against 38.91 on `rk8v4`. With MTP3, C1 decode
fell about 5%, from 81.61 tok/s to 77.39, because acceptance dropped from 71.27% to 65.86%. That is
an accuracy effect on speculation rather than a slower kernel. Use `int8` when decode throughput
under speculation matters more than context.

Per-format numbers for the 35B-A3B and a fit calculator are in
[`docs/config-calculator.html`](docs/config-calculator.html). NVFP4 A4 and FP8 A8 *weight and
activation* kernels are a separate axis and need Blackwell; see [Current limits](#current-limits).

### RotorQuant (`rk8v4`)

Keys keep the rotated INT8 group-64 encoding; values are stored as signed 4-bit codes, two per byte,
over a **group-32** scale. Values use a finer group than keys because four bits resolve a group to
only 15 levels, so a single outlier would otherwise set the quantization step for 64 neighbours.
Halving the value group to 32 costs one extra FP16 scale per 64 dimensions and **halves the
perplexity penalty**, from +0.146% to +0.082%, without costing any context at the automatic-sizing
boundary — but recovers only about 15% of the acceptance loss, because acceptance turns on exact
token agreement rather than mean error. **Values are not rotated**: upstream's `kv_cache_append`
contract stores values from the represented BF16 source directly, and measurement showed that is
sufficient, so there is no inverse-rotation pass over the attention output.

### Lloyd-Max 4-bit keys (`rk4v4`)

`rk4v4` keeps `rk8v4`'s value plane and halves its keys: each rotated key dimension is a 4-bit
index into the 16-level Lloyd-Max quantizer for a Gaussian, with one FP16 scale per 64 dimensions.
That is the TurboQuant idea (rotate, then snap each coordinate to a fixed non-uniform codebook)
without its 1-bit residual stage, which measured no better.

Each codebook level is stored as a fixed INT8 code, so an expanded key is an ordinary rotated INT8
key and attention keeps `rk8v4`'s INT8 tensor-core QK path unchanged. Both attention kernels load
the packed keys into registers ahead of use and expand them with byte permutes straight into the
existing INT8 key tile, which adds no shared memory; that is what keeps it off the slow path the
`nvfp4` and `k8v4` kernels take (see [TODO.md](TODO.md), "KV decode falloff").

Measured together on one RTX 3090 (315 W cap), Qwen3.8-27B groupwise-int, 2026-09-23:

| KV profile | Bytes/token | Perplexity | vs `int8` | Decode 4K / 16K / 32K | Prefill at 32K |
|---|---:|---:|---:|---|---:|
| `int8` | 33,792 | 4.343155 | — | 44.6 / 43.1 / 41.5 tok/s | 1,363 tok/s |
| `rk8v4` | 26,112 | 4.347943 | +0.110% | 46.1 / 44.5 / 42.4 tok/s | 1,377 tok/s |
| **`rk4v4`** | **17,920** | 4.352432 | +0.214% | 45.1 / 43.7 / 41.9 tok/s | 1,361 tok/s |
| `nvfp4` | 18,432 | 4.353589 | +0.240% | 43.8 / 40.7 / 36.3 tok/s | 895 tok/s |

Perplexity is `ninfer-perplexity --quick` on `ninfer-ppl-1m-v1` (4096/2048). Decode is
`ninfer_bench -pg P,128`, no speculation, the mean of two interleaved runs for `rk8v4` and `rk4v4`;
the card slows about 0.4 tok/s per run as it heats, and against that drift `rk4v4` sits within ±1% of
`rk8v4` at every depth. `int8` and `nvfp4` are single runs taken last, so they read slightly low.
Qwen3.6-35B-A3B, same protocol: `rk4v4` decodes at 99.5-99.9% of `rk8v4` at 4K-32K (172.5 against
173.4 tok/s at 32K) and 19% faster than `nvfp4` (144.9). With MTP3 and the draft head on the 27B,
`rk4v4` is within about 0.5% of `rk8v4` at 32K; on four real prompts (greedy, 400 tokens) its draft
acceptance was 65.8% against `rk8v4`'s 67.2% and `int8`'s 63.2%. Across the attention op
benchmark's 18 decode/verify shapes it is at a geometric mean of 1.001x `rk8v4`'s time and 2-3x
faster than `nvfp4` at 32K.

**Context on Qwen3.8-27B**, measured with `--kv-capacity auto` and the standard 1 GiB headroom with a
desktop running (1.8 GiB in use):

| One request | `int8` | `rk8v4` | `rk4v4` |
|---|---:|---:|---:|
| no speculation | 171,648 tokens | 228,032 tokens | **262,144** (native maximum; 1.75 GiB still free) |
| MTP3 + draft head | — | 182,336 tokens | **262,144** (native maximum) |
| KV payload at 131,072 tokens | — | 3.19 GiB | **2.19 GiB** |

For a shared pool serving several lanes, the same memory holds 1.46x as many `rk4v4` tokens as
`rk8v4` tokens.

## Several GPUs: pipeline stages (`--devices A,B,...`)

`--devices 0,1` splits the model's layers into one pipeline stage per GPU. Each stage owns its
layers whole: weights, the KV cache of its attention layers, the recurrent state of its GDN layers
and the scratch it runs in. The point is memory. A model that does not fit one card, or a context
that does not, spreads across several, and every card's memory is usable for KV.

```
ninfer-serve model.ninfer --devices 0,1
ninfer model.ninfer --devices 0,1,2 --stage-layers 20,22,22 --prompt "..."
```

- **`--stage-layers A,B,...`** sets the layers per stage. Without it the split follows each
  device's free memory, so the first GPU, which also carries the embedding and head, takes fewer
  layers and the most KV cache fits on every card at once.
- **The first GPU also holds the embedding, the output head and the round state.** The last stage
  sends the residual back to it, one extra hop per forward pass.
- **It is a memory feature, not a speed feature.** The stages run in sequence and each reads only
  its own weights, so a single stream decodes about as fast as one GPU, minus the boundary hops.
- **Linux only for real multi-GPU.** Repeating one id (`--devices 0,0`) puts several stages on one
  card, saves no memory, and exercises the whole stage path; it is how the path is tested without a
  second GPU, and it works on Windows too.
- **Works with a split:** the context cache and prefix reuse (including pausing and replaying a
  request under KV pressure), CUDA graphs, MTP, DFlash/DFlash2 (the draft stays on the first device
  and the target layers it reads cross back to it), every KV format, and resident vision (the tower
  and its encode stay on the first device, so the first stage's layer budget carries it),
  `--vision-residency overlay` (its window borrows only the first device's memory), prompt grafts,
  and worker failure recovery.
- **Boundary transfers stage through pinned host memory.** Peer access is not needed, and no
  consumer PCIe pair measured so far offers it. Measured with `tools/tp_probe.cu` on rented 2x A4000
  and 2x 3090 PCIe boxes, a staged transfer took about 0.03 ms at a decode-sized payload and several
  milliseconds at a prefill-chunk-sized one, depending on the slot's link width.

Measured on two rented Linux boxes (Qwen3.6-27B, int8 KV, no peer access on either): on 2x RTX 3090
(PCIe 3.0 x16) greedy output is byte-identical to one card, decode is 48.5 tok/s against 46.9 on one
card (105.3 against 100.2 with MTP3), prefill is unchanged, and `--kv-capacity auto` resolves the
full 262,144-token context that one 24 GB card refuses. On 2x RTX A4000 the 27B runs at 262,144
tokens with 24.1 tok/s decode (52.6 with MTP3) and 825 tok/s prefill. The tables, the cases checked
and the design are in `docs/maintainer/pipeline-parallel-plan.md`.

## Current limits

- One model per process, on one GPU or split into pipeline stages with `--devices` (Linux). No
  tensor parallelism and no CPU/GPU weight offload. The pipeline on the current context engine has
  been verified only with stages sharing one GPU (`--devices 0,0`); distinct GPUs are unverified on
  this build.
- Concurrency is fixed at startup and limited to 1-8; the compact 35B-A3B fits C1-C6 at 4K and
  Qwen3.8-27B fits C8/8K with MTP3 through ReplaySSM.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- Tool calls are returned to the client but are not executed by NInfer.
- This fork targets `sm_86`. NVFP4 A4, FP8 A8 and TMA *weight and activation* kernels require
  Blackwell and are unavailable; FP8 and NVFP4 weights are admitted through their A16 dequantizing
  routes. KV-cache storage is a separate axis: all seven KV formats, including row-scaled FP8 E4M3,
  run on SM86.
- `sm_80` (Ampere GA100, e.g. an unlocked NVIDIA CMP 170HX) is an added but **unmeasured**
  compatibility target: it shares sm_86's instruction set and runs Qwen3.8-27B correctly at roughly
  3090-like speed with no route tables tuned for it yet.

## Upstream

NInfer-3090 is derived from [Neroued/ninfer](https://github.com/Neroued/ninfer). The upstream project
targets RTX 5090/`sm_120a`; this fork carries the Windows and Linux SM86 compatibility layer,
compact 35B artifact support, and RTX 3090-specific schedules and memory planning.

Upstream NInfer is a personal project its author develops out of interest. If you find it useful,
you can [support upstream on Ko-fi](https://ko-fi.com/neroued). Support is entirely voluntary. It is
not a purchase or investment and does not come with financial returns, promised services or
features, or a role in project decisions.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the complete, maintained credit list.

- [airtonix](https://github.com/airtonix) added Linux and Docker build and release support in
  [PR #1](https://github.com/Don-Chad/ninfer-3090/pull/1).
- [ColeWheatley](https://github.com/ColeWheatley) contributed SM86 runtime-count/GDN residency
  fixes, ECC diagnostics, and the GeForce-safe Docker fix in
  [PR #7](https://github.com/Don-Chad/ninfer-3090/pull/7).
- [justinlime](https://github.com/justinlime) added NixOS build support in
  [PR #5](https://github.com/Don-Chad/ninfer-3090/pull/5).
- [sry9681](https://github.com/sry9681) contributed the device-wide GPU-memory startup fix in
  [PR #6](https://github.com/Don-Chad/ninfer-3090/pull/6).
- [iamwavecut](https://github.com/iamwavecut) contributed the swscale destination-alignment
  JPEG safety fix in [PR #11](https://github.com/Don-Chad/ninfer-3090/pull/11).
- [nasedkinpv](https://github.com/nasedkinpv) contributed the tool-call parser crash fix in
  [PR #12](https://github.com/Don-Chad/ninfer-3090/pull/12).
- [wmehanna](https://github.com/wmehanna) contributed in-place system-turn rendering for
  Claude Code prefix reuse in [PR #13](https://github.com/Don-Chad/ninfer-3090/pull/13).
- [mgscreativa](https://github.com/mgscreativa) contributed the thinking-budget boundary fix, so a
  request near the end of its output window no longer fails with
  `thinking_budget_capacity_insufficient`, in
  [PR #156](https://github.com/ashalliants/ninfer-3090/pull/156) (continued and finished in this
  PR), and found and fixed the premature KV-loan race behind a `std::bad_alloc` on large
  vision prompts in [PR #138](https://github.com/ashalliants/ninfer-3090/pull/138); its
  KV-lending gate was kept in [PR #145](https://github.com/ashalliants/ninfer-3090/pull/145).

## Contributing

Please read the [Pull Request Policy](PR_POLICY.md) before opening an issue or pull request.
It explains how to keep changes focused and how to document correctness, performance, VRAM, and
compatibility evidence.

## License

Apache License 2.0. See [LICENSE](LICENSE).
