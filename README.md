# NInfer-3090

NInfer-3090 is a specialized C++20/CUDA inference engine for **Qwen3.8-27B** and Qwen3.6 on one
24 GB NVIDIA GeForce RTX 3090. Qwen3.8-27B is a first-class, tested target: the native SM86
runtime loads its official groupwise `.ninfer` artifact, serves OpenAI- and Anthropic-compatible
APIs, and supports paged KV, compatible-prefix reuse, CUDA Graphs, MTP speculative decoding,
reasoning-effort control, ReplaySSM state transactions, and concurrent cohorts through **C8**.

On an RTX 3090, Qwen3.8-27B supports a measured **171K-token INT8 context** with the standard
1 GiB safety headroom. The optional RotorQuant `rk8v4` cache raises this to **226K tokens with
1 GiB headroom**, or **247,872 tokens (about 248K)** in a tightly packed 300 MiB-headroom profile.
RotorQuant is an opt-in feature; INT8 remains the default because `rk8v4` is lossy.

Community project, maintained on a best-effort basis. Issues and PRs are very welcome, but support
and feature requests are not guaranteed.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable. DFlash is not part
of the recommended RTX 3090 path; use MTP speculative decoding.

## Start Qwen3.8 in three steps

1. Download and unzip the latest [Windows release](https://github.com/Don-Chad/ninfer-3090/releases/latest).
2. Double-click `download-qwen38.bat` to download the model. Interrupted downloads resume.
3. Double-click one launcher:

| Launcher | Best for |
|---|---|
| `run-qwen38-c1.bat` | One interactive user, lowest latency, up to 64K context |
| `run-qwen38-c8.bat` | Multiple users or agents, highest aggregate throughput, 8K context |
| `run-qwen38-vision.bat` | Qwen3.8 image understanding, one user, 32K context, MTP3 |
| `run-qwen36-35b-vision.bat` | Image understanding with Qwen3.6-35B-A3B, one user, 32K context |

The OpenAI-compatible API is then available at `http://127.0.0.1:8080/v1`. No Python environment,
CMake, or PowerShell commands are required for the release build.

## Qwen3.8-27B support and RTX 3090 results

Qwen3.8-27B is validated from one through eight simultaneous users. ReplaySSM cuts the memory cost
of speculative decoding, allowing the faster MTP3 mode to remain enabled at C8. The table below is
the new sustained test: every request generated 1,024 tokens with CUDA Graphs enabled.

The prompts were **29-34 input tokens** and the server's maximum context window was **8,192 tokens
per request**. Each measured sequence therefore reached roughly 1,053-1,058 tokens including its
generated output. This is a long-output/decode benchmark, not an 8K-prompt or long-prefill test.
C1-C4 used an 8,192-token shared KV pool; C8 used 16,384 tokens so all eight requested outputs
could be admitted simultaneously.

| Cohort | Total output | End-to-end throughput | Decode throughput | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 tokens | **70.19 tok/s** | **71.00 tok/s** | 61.13% | 149 ms | 19,641 MiB |
| C2 | 2,048 tokens | **89.43 tok/s** | **90.66 tok/s** | 59.66% | 262 ms | 20,022 MiB |
| C4 | 4,096 tokens | **97.89 tok/s** | **100.28 tok/s** | 59.63% | 538 ms | 20,641 MiB |
| C8 | 8,192 tokens | **161.28 tok/s** | **165.33 tok/s** | 56.84% | 1,215 ms | 22,138 MiB |

C1 is the responsive choice for a single user. C8 delivers **2.3x the total throughput** when
several requests are active. The C8 long-output test uses a 16K shared KV pool so all eight
1,024-token responses can be admitted together.

### Optional RotorQuant KV for longer context

`rk8v4` is an **experimental, opt-in** KV-cache mode for Qwen3.8-27B. INT8 remains the default and
the recommended quality setting. RotorQuant applies the same normalized transform to queries and
keys, rotates values before four-bit storage, and reverses the value transform after attention.
This reduces the V-cache footprint while keeping keys at eight bits.

Add `--kv-dtype rk8v4` to either `ninfer.exe` or `ninfer-serve.exe`. For example, from Command
Prompt:

```bat
ninfer-serve.exe qwen3_8_27b.ninfer --max-context 131072 --kv-capacity auto --max-concurrency 1 --prefill-chunk 1024 --kv-dtype rk8v4 --spec mtp --draft-tokens 3 --lm-head-draft
```

On the development RTX 3090, C1 with MTP and CUDA Graphs disabled fit a **226,560-token** logical
context with the normal 1 GiB automatic-sizing headroom; 226,624 was rejected. The comparable INT8
boundary was 171,648 tokens, so `rk8v4` increased measured allocatable context by about **32%**.
This is an allocation plus short-execution boundary, not a full 226K prefill quality claim.

A second explicit-reservation test reduced operational headroom to approximately 300 MiB.
`rk8v4` successfully started and generated at **247,872 tokens**, leaving **302.97 MiB** physically
free after startup. The next 64-token page boundary, 247,936, left only 274.58 MiB because it crossed
a CUDA allocation granularity. Therefore 247,872 is the measured maximum that retained at least
300 MiB—not 2x INT8 capacity, but about **44% more** than the 171,648-token INT8/1 GiB baseline.
This tight profile leaves little tolerance for other GPU users and is not the recommended default.

The matched 1,024-token hard coding test reduced 4K KV payload from 140.38 MiB to 106.35 MiB and
decoded at 84.07 tok/s versus 85.72 tok/s for INT8. Quality was not equivalent: the RotorQuant
answer introduced a faulty nested-rollback design that the INT8 answer avoided, and both answers
hit their output limit. Use `rk8v4` only when its context gain is worth task-specific quality
validation; do not use it as the default for correctness-sensitive work.

### Qwen3.8 vision

The same Qwen3.8 artifact supports images. Run `download-qwen38.bat` once, then double-click
`run-qwen38-vision.bat`. The tested RTX 3090 profile uses **one request, 32K maximum context,
INT8 KV, ReplaySSM, and MTP3**.

A 1,920×1,080 image expanded to 2,074 prompt tokens and was read correctly. Measured TTFT was
3.29 seconds, decode reached 98.1 tok/s, MTP acceptance was 96.7%, and startup retained 2.16 GiB
free VRAM. The artifact also declares multi-image and video support; this release test directly
validated a single image.

## Qwen3.6-35B-A3B RTX 3090 results

Measured with the compact 20.84 GiB 35B-A3B artifact, a 4K shared INT8 group-64 paged KV pool,
CUDA Graphs, MTP3, greedy decoding, and no competing GPU workload:

| Concurrent requests | 128 output tokens each | Observed VRAM |
|---:|---:|---:|
| 1 | 162.7 aggregate tok/s | 22,427 MiB |
| 2 | 267.9 aggregate tok/s | 22,743 MiB |
| 4 | 366.2 aggregate tok/s | 23,377 MiB |
| 6 | 383.4 aggregate tok/s | 24,038 MiB |
| 8 | rejected at startup | about 503 MiB over the safe reservation limit |

A longer 512-token-per-request check reached **286.8 tok/s at C1** and **399.1 aggregate tok/s at
C2**. These short-prompt measurements include request-level timing and are not directly comparable
to v0.3.1's 1,500-token adaptive prompt-lookup benchmark.

Compatible-prefix reuse was validated end to end: a repeated 26-token prompt reused 24 tokens,
reducing measured prefill from 371 ms to 10 ms.

### Qwen3.6-35B vision

The compact 35B artifact includes its vision encoder and accepts images through the same OpenAI-
compatible API. Run `download-qwen36-35b-vision.bat` once, then double-click
`run-qwen36-35b-vision.bat`.

The safe RTX 3090 profile is **one request, 32K maximum context, INT8 KV, vision enabled, and MTP
disabled**. A current v0.6 test processed three 1,920×1,080 images correctly. Each image expanded
to a 2,081-token prompt; engine TTFT was 3.76–4.07 seconds, decode was about 159 tok/s, and peak
VRAM was 23,944 MiB. This leaves little room for another GPU workload.

MTP is intentionally off for this profile. At 32K, speculative recurrent state would exceed the
3090 memory budget; KV compression alone does not recover enough memory. Text-only 35B profiles can
still use MTP3 as documented above.

## What the release gives you

- A native Windows server that runs Qwen3.8-27B on one RTX 3090.
- OpenAI Chat Completions, Responses, and Anthropic-compatible APIs.
- ReplaySSM and MTP3 for higher throughput without exceeding 24 GB VRAM.
- `low`, `medium`, and `xhigh` reasoning modes.
- Qwen3.8 image understanding with ReplaySSM and MTP3.
- Prefix reuse for faster repeated or shared prompts.
- Qwen3.6-35B image understanding with a guarded 32K profile.
- One-user and eight-user launchers with safe tested defaults.

## Supported artifacts

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B v1 | [pinned compact artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c) | 20.84 GiB | **Recommended for RTX 3090; text C1-C6 at 4K and vision C1 at 32K** |
| Qwen3.6-35B-A3B v2 | [current upstream artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 21.22 GiB | Reader supported by v0.5+; includes DFlash payload and is not the measured 3090 artifact |
| Qwen3.6-27B | [groupwise artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GiB | Supported with more runtime headroom |
| **Qwen3.8-27B** | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB | **Validated at C1, C2, C4 and C8/MTP3 with ReplaySSM** |

NInfer-3090 v0.5 and newer recognize both v1 and v2 container magic. The current 21.22 GiB v2
artifact contains additional DFlash weights and is not the artifact used for the published RTX
3090 concurrency results. The pinned compact v1 artifact keeps the measured model payload and
omits DFlash, providing the known 24 GB memory profile.

## Downloads and compatibility

The release requires Windows 11, an RTX 3090, and a recent NVIDIA driver. The archive already
contains the native applications and required DLLs.

For Qwen3.8, use `download-qwen38.bat` or download the
[official artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) manually and save it as
`models\qwen3_8_27b.ninfer` beside the launchers.

For Qwen3.6-35B-A3B, the smaller
[pinned container-v1 artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c)
is recommended on a 24 GB card. Releases v0.5 and newer also read the larger container-v2 file.
An `artifact magic is not NInfer version 1` message means the executable is outdated, not that the
current model download is necessarily corrupt.

Developers building from source can use Visual Studio 2022, CUDA 12.8 or newer, CMake, and vcpkg.
See the [Windows build guide](docs/rtx-3090-windows.md); ordinary release users do not need these
tools.

## Qwen3.8 reasoning effort

Qwen3.8-27B supports distinct reasoning-effort modes. `medium` uses the model's normal thinking
prompt. `xhigh` injects the checkpoint's extended deliberation instruction, asking it to validate
assumptions and consider alternatives. This is a real prompt-template change, not a sampling alias.

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
`--reasoning-effort low|medium|xhigh`; use `--no-thinking` instead of an effort to disable
reasoning. Chat Completions returns hidden reasoning separately as `message.reasoning_content`.

## Serving APIs

The server supports:

- OpenAI Chat Completions;
- OpenAI Responses Core with streaming and local continuation state;
- Anthropic Messages;
- compatible-prefix reuse;
- prompt-rendered function tools and parsed tool calls;
- bounded pending-request admission and JSONL request logs.

See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## How cohort batching works

The C number is the maximum number of requests NInfer can run together. C1 favors one interactive
user; C8 can combine up to eight active requests into each GPU step for much higher total output.

Follow-up requests do not need to arrive at the same instant. When a running request finishes, the
next waiting request can join at a safe generation boundary. Empty or finished lanes are skipped,
so a C8 server also works normally with only one, two, or four active users.

This is deliberately more bounded than datacenter-style dynamic batching. The maximum number of
users and GPU memory are chosen when the server starts. In return, memory use stays predictable on
a 24 GB card and the server can reuse fast CUDA Graphs instead of rebuilding work continuously.

## Current limits

- One process owns one model on one RTX 3090.
- Concurrency is fixed at startup and limited to 1-8 by the API; compact 35B fits C1-C6 and
  Qwen3.8-27B fits C8/8K with MTP3 through ReplaySSM.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- No multi-GPU execution or CPU/GPU weight offload.
- Tool calls are returned to the client but are not executed by NInfer.
- NVFP4 A4 and TMA kernels require Blackwell and are unavailable on SM86.
- The paged runtime exposes BF16, INT8, and experimental opt-in `rk8v4` KV. INT8 remains the
  quality-default path.

## Validation

The v0.5 release gate includes Qwen3.8 artifact loading/generation, materialization, request memory, admission policy,
paged KV, prefix append, speculative rounds, and SM86 W8 linear paths. The focused Windows gate
passed 11/11 tests. Benchmark logs remain local under `benchmark_results/` and are not included in
source releases.

## Upstream

NInfer-3090 is derived from [Neroued/ninfer](https://github.com/Neroued/ninfer). The upstream project
targets RTX 5090/`sm_120a`; this fork carries the Windows and SM86 compatibility layer, compact 35B
artifact support, and RTX 3090-specific schedules and memory planning.

## License

Apache License 2.0. See [LICENSE](LICENSE).
