# NInfer-3090

NInfer-3090 is a specialized C++20/CUDA inference engine for **Qwen3.8-27B** and Qwen3.6 on one
24 GB NVIDIA GeForce RTX 3090. Qwen3.8-27B is a first-class, tested target: the native SM86
runtime loads its official groupwise `.ninfer` artifact, serves OpenAI- and Anthropic-compatible
APIs, and supports paged KV, compatible-prefix reuse, CUDA Graphs, MTP speculative decoding,
reasoning-effort control, ReplaySSM state transactions, and concurrent cohorts through **C8**.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable. DFlash is not part
of the recommended RTX 3090 path; use MTP speculative decoding.

## Cohort batching, in plain language

Choose a cohort size at startup with `--max-concurrency`: C1 runs one request at a time, while C8
can keep up to eight requests active. NInfer combines every active, decode-ready request into one
compact GPU batch each generation round. Finished or temporarily non-ready requests are omitted,
so a C8 server can execute batch sizes from one through eight without wasting rows on empty slots.

New requests may arrive at any time. If a cohort lane and enough paged-KV/state capacity are free,
the scheduler admits the next prepared request at a safe round boundary; otherwise it waits in the
bounded pending queue configured by `--max-pending-requests`. A follow-up request is therefore able
to join the running server, but it does not interrupt or rebuild a GPU operation already in flight.

This differs from unrestricted dynamic or continuous batching: the active population can change,
but the maximum cohort, memory pools, workspace, and CUDA Graph batch shapes are fixed at startup.
That bounded design is less flexible than a large datacenter scheduler, but it gives predictable
VRAM use and lets an RTX 3090 reuse optimized CUDA Graphs for each reachable batch size.

## Qwen3.8-27B support and RTX 3090 results

Qwen3.8-27B is validated for the C1, C2, C4, and C8 cohorts. ReplaySSM records provisional GDN
updates and folds only accepted speculative tokens into persistent recurrent state. This replaces
the old per-depth state snapshots, saves several GiB at high concurrency, and makes C8/MTP3 fit on
the RTX 3090. Every row below completed real text generation over simultaneous 128-token requests
with CUDA Graphs and no competing GPU workload.

| Cohort | Validated profile | Whole-wave aggregate throughput | Mean TTFT | VRAM result |
|---:|---|---:|---:|---:|
| C1 | 4K, MTP3 | **61.82 tok/s** | **142 ms** | 19,500 MiB peak |
| C2 | 4K, MTP3 | **77.11 tok/s** | **262 ms** | 20,278 MiB peak |
| C4 | 4K, MTP3 | **80.20 tok/s** | **543 ms** | 21,811 MiB peak |
| C8, ReplaySSM | 8K, MTP3, 65,536-token shared KV | **114.88 tok/s** | **1,432 ms** | 23,745 MiB peak |

**Whole-wave aggregate throughput** is total completion tokens divided by elapsed time from the
simultaneous request launch until the last request completes. It includes prefill/TTFT, decode
ramp-up, and the tail where fewer requests remain active. This is the comparable headline number:
**61.82 tok/s at C1 versus 114.88 tok/s at C8**, an 86% aggregate throughput increase.

The original C8/MTP2 campaign sustained **179-211 aggregate tok/s during fully-active decode
intervals**. That telemetry excludes prefill, ramp-up, and drain time, so it must not be compared
directly with whole-wave throughput or interpreted as per-request speed. The ReplaySSM C8/MTP3
run measured 153.77 aggregate decode tok/s across the complete decode span.

A matched four-prompt C1 sweep measured ordinary decoding at 36.86 tok/s, MTP2 at 65.72 tok/s,
and MTP3 at 71.56 tok/s. MTP2 accepted 78.4% of drafted tokens; MTP3 accepted 67.7% but committed
3.04 tokens per round and remained 8.9% faster than MTP2. Greedy reasoning/output text was not
byte-identical across every execution route, so these throughput results do not establish exact
quality parity.

The headline ReplaySSM run reserved 65,536 paged-KV tokens: enough capacity for eight independent
8K sequences. A matched run with the established 8,192-token shared pool measured 114.73 tok/s and
21,818 MiB peak VRAM. The 0.13% throughput difference is noise; use the 8K shared pool for almost
2 GiB more headroom, or 65,536 when all eight requests must be able to grow to 8K simultaneously.
Use C1 when single-request latency matters and C2/C4 for a latency-throughput balance.

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

## Changes since v0.3.1

- shared paged BF16/INT8 KV storage;
- startup-bounded concurrent execution with decode-ready request compaction;
- batched ordinary and MTP decode, with C1-C6 validated for compact 35B at 4K;
- compatible-prefix KV reuse across requests;
- explicit shared `--kv-capacity` sizing and admission control;
- OpenAI Responses Core, Chat Completions, Anthropic Messages, streaming, function-call rendering,
  request logging, and throughput telemetry;
- direct compatibility with legacy compact v1 groupwise artifacts and current v2 artifacts;
- registered Qwen3.8-27B groupwise artifact support using W8 embedding/output endpoints;
- ReplaySSM speculative GDN state transactions, reducing recurrent-state memory at concurrency;
- Qwen3.8 reasoning-effort selection (`low`, `medium`, and `xhigh`);
- native Windows CUDA 13.x / MSVC support.

The legacy reader assigns only the two registered groupwise identities. A compact 35B artifact
without DFlash weights is accepted when DFlash is disabled and rejected explicitly if DFlash is
requested.

## Supported artifacts

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B v1 | [pinned compact artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c) | 20.84 GiB | **Recommended for RTX 3090; C1-C6 validated at 4K** |
| Qwen3.6-35B-A3B v2 | [current upstream artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 21.22 GiB | Reader supported by v0.5+; includes DFlash payload and is not the measured 3090 artifact |
| Qwen3.6-27B | [groupwise artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GiB | Supported with more runtime headroom |
| **Qwen3.8-27B** | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB | **Validated at C1, C2, C4 and C8/MTP3 with ReplaySSM** |

NInfer-3090 v0.5 and newer recognize both v1 and v2 container magic. The current 21.22 GiB v2
artifact contains additional DFlash weights and is not the artifact used for the published RTX
3090 concurrency results. The pinned compact v1 artifact keeps the measured model payload and
omits DFlash, providing the known 24 GB memory profile.

## Requirements

- NVIDIA GeForce RTX 3090 (`sm_86`);
- Windows 11 x64 (the release binary is native Windows);
- an NVIDIA driver compatible with CUDA 13.x;
- CMake 3.28 or newer and a C++20 compiler;
- CUDA Toolkit 12.8 or newer when building from source;
- FFmpeg and libcurl development/runtime dependencies.

## Build on Windows

Use Visual Studio 2022 and vcpkg:

```powershell
cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-windows --config Release --parallel
```

See [the Windows guide](docs/rtx-3090-windows.md) for complete setup instructions.

## Download the correct 35B artifact

For the validated RTX 3090 profiles, pin the compact **container-v1** revision. Do not omit
`--revision`: Hugging Face `main` now points to the larger container-v2/DFlash artifact.

```powershell
hf download neroued/Qwen3.6-35B-A3B-NInfer `
  qwen3_6_35b_a3b.ninfer `
  --revision c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c `
  --local-dir models

Get-FileHash .\models\qwen3_6_35b_a3b.ninfer -Algorithm SHA256
```

The expected v1 size is **20.84 GiB** and its SHA-256 is
`9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163`.

### What if I already downloaded container v2?

The current unpinned download is container v2, 21.22 GiB, SHA-256
`1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2`. NInfer-3090
**v0.5.0 or newer supports v2 parsing and binding**, but this larger DFlash-bearing artifact was
not used for the published 3090 memory/concurrency results. Prefer pinned v1 on a 24 GB card.

If the program reports `artifact magic is not NInfer version 1`, the executable is older than
the v1/v2 reader. Install the
[v0.5.0 Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.5.0-rtx3090)
or rebuild this branch. A v0.5 binary reports `artifact magic is not NInfer v1 or v2` for a truly
invalid file. Also check that a failed/interrupted download did not leave a small pointer or
partial file in place.

`.ninfer` artifacts contain quantized weights and frontend resources. They are not GGUF or
Transformers checkpoints.

For Qwen3.8-27B:

```powershell
hf download neroued/Qwen3.8-27B-NInfer `
  qwen3_8_27b.ninfer `
  --local-dir models
```

## Run the server

The explicit 4K KV capacity avoids reserving the extra 1 GiB safety margin used by `auto` and is
the validated compact-35B configuration:

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_6_35b_a3b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 4096 --kv-capacity 4096 --max-concurrency 4 `
  --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft
```

Use `--max-concurrency 2` when individual request latency matters more than aggregate throughput.
C6 provides the highest measured short-run aggregate throughput but leaves little VRAM headroom.

The recommended Qwen3.8 C8/8K shared-pool profile uses ReplaySSM with MTP3:

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 --max-concurrency 8 `
  --prefill-chunk 1024 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Set `--kv-capacity 65536` instead when every one of the eight requests must be able to occupy its
full 8K context simultaneously. That profile measured 114.88 aggregate tok/s and 23,745 MiB peak
VRAM; the 8K shared-pool profile measured 114.73 tok/s and 21,818 MiB.

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

## Current limits

- One process owns one model on one RTX 3090.
- Concurrency is fixed at startup and limited to 1-8 by the API; compact 35B fits C1-C6 and
  Qwen3.8-27B fits C8/8K with MTP3 through ReplaySSM.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- No multi-GPU execution or CPU/GPU weight offload.
- Tool calls are returned to the client but are not executed by NInfer.
- NVFP4 A4 and TMA kernels require Blackwell and are unavailable on SM86.
- The paged runtime exposes BF16 and INT8 KV. Legacy RotorQuant/KV4 paths are not ported to paged
  append, prefix reuse, batched decode, and provisional MTP state.

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
