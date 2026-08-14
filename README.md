# NInfer-3090

Run **Qwen3.8-27B**  and Qwen 3.6 35B A3B locally on one RTX 3090 with fully fused kernels, build from the ground up for the fastest inference especially for these models. Native Windows and Linux, OpenAI-compatible API, large context. 80 tokens per second and 265 tokens per second respectively.

NInfer-3090 is a small C++20/CUDA server tuned specifically for the 3090's 24 GB memory budget
and `sm_86` GPU. It supports streaming, MTP speculative decoding, CUDA Graphs, paged KV and prefix
caching. When several requests arrive together, it can also batch up to eight active generations.

## Quick start: Qwen3.8-27B API at C1/64K

Download and extract the
[v0.5.0 Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.5.0-rtx3090),
open PowerShell in that folder, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-qwen38-c1.ps1
```

That one command downloads the **pinned, correct** `qwen3_8_27b.ninfer` artifact, resumes an
interrupted download, verifies its SHA-256, and starts the server at `http://127.0.0.1:8080/v1`.
The model download is 16.96 GiB and is stored in `models\` beside the server.

Building from source? Use the same launcher from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-qwen38-c1.ps1
```

Test it from another PowerShell window:

```powershell
$body = @{
  model = 'qwen3.8-27b'
  messages = @(@{ role = 'user'; content = 'Write a haiku about local AI.' })
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions `
  -Method Post -ContentType 'application/json' -Body $body
```

The default is **C1 + 64K INT8 KV + MTP3**: the responsive profile for a single user, coding
agent, or desktop app. It has completed real 64K-capacity startup and generation on a 24 GB RTX
3090, including a 10K-token tool prompt and an 8K-token output. Avoid other heavy GPU workloads
while the server is running.

> Need throughput instead? The tested C8/8K profile is documented below. C8 uses MTP2 because
> MTP3 does not fit safely at that concurrency.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable, and DFlash is not
the recommended Qwen3.8 path on this GPU. Use MTP speculative decoding.

## Qwen3.8-27B support and RTX 3090 results

Qwen3.8-27B is validated for the C1, C2, C4, and C8 cohorts. C1, C2, and C4 use the matched 4K/MTP3
benchmark; the maximum-throughput C8 profile uses 8K/MTP2 because MTP3 does not fit safely at
C8/8K. Every row below completed real text generation on one RTX 3090 with CUDA Graphs and no
competing GPU workload.

| Cohort | Validated profile | Whole-wave aggregate throughput | Mean TTFT | VRAM result |
|---:|---|---:|---:|---:|
| C1 | 4K, MTP3 | **61.82 tok/s** | **142 ms** | 19,500 MiB peak |
| C2 | 4K, MTP3 | **77.11 tok/s** | **262 ms** | 20,278 MiB peak |
| C4 | 4K, MTP3 | **80.20 tok/s** | **543 ms** | 21,811 MiB peak |
| C8 | 8K, MTP2 | **105.30 tok/s** | not recorded | 774 MiB physically free |

**Whole-wave aggregate throughput** is total completion tokens divided by elapsed time from the
simultaneous request launch until the last request completes. It includes prefill/TTFT, decode
ramp-up, and the tail where fewer requests remain active. This is the comparable headline number:
**61.82 tok/s at C1 versus 105.30 tok/s at C8**, a 70% aggregate throughput increase.

The C8 run also sustained **179-211 aggregate tok/s during fully-active decode intervals**. This
is still aggregate across all eight requests, but it measures only telemetry intervals in which
all eight lanes are decoding. It excludes prefill, ramp-up, and drain time, so it must not be
compared directly with the 105.30 tok/s whole-wave result or interpreted as per-request speed.

A matched four-prompt C1 sweep measured ordinary decoding at 36.86 tok/s, MTP2 at 65.72 tok/s,
and MTP3 at 71.56 tok/s. MTP2 accepted 78.4% of drafted tokens; MTP3 accepted 67.7% but committed
3.04 tokens per round and remained 8.9% faster than MTP2. Greedy reasoning/output text was not
byte-identical across every execution route, so these throughput results do not establish exact
quality parity.

At C8/8K, MTP2 reached 90.8% MTP acceptance and left 388 MiB of planned slack. MTP3 is rejected
safely: it misses the reservation by 789 MiB with graphs and 101 MiB without graphs. Use C1 when
single-request latency matters, C2/C4 for a latency-throughput balance, and C8/MTP2 for maximum
aggregate throughput.

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
| **Qwen3.8-27B** | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB | **Validated at C1, C2, C4 (4K/MTP3) and C8 (8K/MTP2)** |

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

## Download model artifacts manually

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

For Qwen3.8-27B, pin the exact artifact revision used by the quick launcher:

```powershell
hf download neroued/Qwen3.8-27B-NInfer `
  qwen3_8_27b.ninfer `
  --revision 3526913004b1cf552cb57b88d6a5c6f5e4a89a70 `
  --local-dir models

Get-FileHash .\models\qwen3_8_27b.ninfer -Algorithm SHA256
```

Expected SHA-256:
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.

## Run the server

For the recommended interactive Qwen3.8 C1/64K API, use `scripts\run-qwen38-c1.ps1` as shown in
Quick start. The explicit command it runs is:

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 65536 --kv-capacity 65536 --max-concurrency 1 `
  --prefill-chunk 512 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

### Compact Qwen3.6-35B server

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

The validated Qwen3.8 C8/8K profile uses MTP2:

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 --max-concurrency 8 `
  --prefill-chunk 1024 --kv-dtype int8 `
  --spec mtp --draft-tokens 2 --lm-head-draft
```

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
  Qwen3.8-27B fits C8/8K with MTP2.
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
