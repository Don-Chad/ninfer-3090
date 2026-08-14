# NInfer-3090

NInfer-3090 is a specialized C++20/CUDA inference engine for **Qwen3.8-27B** and Qwen3.6 on one
24 GB NVIDIA GeForce RTX 3090. Qwen3.8-27B is a first-class, tested target: the native SM86
runtime loads its official groupwise `.ninfer` artifact, serves OpenAI- and Anthropic-compatible
APIs, and supports paged KV, compatible-prefix reuse, CUDA Graphs, MTP speculative decoding, and
concurrent cohorts through **C8**.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable. DFlash is not part
of the recommended RTX 3090 path; use MTP speculative decoding.

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
| Qwen3.6-35B-A3B | [compact groupwise artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 20.84 GiB | Recommended; C1-C6 validated at 4K |
| Qwen3.6-27B | [groupwise artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GiB | Supported with more runtime headroom |
| **Qwen3.8-27B** | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.96 GiB | **Validated at C1, C2, C4 (4K/MTP3) and C8 (8K/MTP2)** |

The current upstream 21.22 GiB 35B v2 artifact contains additional DFlash weights and does not fit
the tested RTX 3090 concurrent configuration. The compact artifact keeps the proven model weights
and omits that unused payload.

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

## Download the compact 35B model

```powershell
hf download neroued/Qwen3.6-35B-A3B-NInfer `
  qwen3_6_35b_a3b.ninfer `
  --local-dir models
```

The `.ninfer` artifact contains quantized weights and frontend resources. It is not a GGUF or
Transformers checkpoint.

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
