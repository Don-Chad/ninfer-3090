# NInfer-3090

NInfer-3090 is a specialized C++20/CUDA inference engine for Qwen3.6 on one 24 GB NVIDIA GeForce
RTX 3090. Version 0.4 adds upstream's paged KV cache, true concurrent MTP execution, compatible
prefix reuse, request admission, and expanded OpenAI/Anthropic serving while preserving support
for the compact v0.3.1 Qwen3.6-35B-A3B artifact.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 execution is unavailable. DFlash is not part
of the recommended RTX 3090 path; use MTP speculative decoding.

## RTX 3090 results

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
- native Windows CUDA 13.x / MSVC support.

The legacy reader assigns only the two registered groupwise identities. A compact 35B artifact
without DFlash weights is accepted when DFlash is disabled and rejected explicitly if DFlash is
requested.

## Supported artifacts

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B | [compact groupwise artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 20.84 GiB | Recommended; C1-C6 validated at 4K |
| Qwen3.6-27B | [groupwise artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GiB | Supported with more runtime headroom |

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
- Concurrency is fixed at startup and limited to 1-8 by the API; compact 35B currently fits C1-C6.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- No multi-GPU execution or CPU/GPU weight offload.
- Tool calls are returned to the client but are not executed by NInfer.
- NVFP4 A4 and TMA kernels require Blackwell and are unavailable on SM86.

## Validation

The v0.4 release gate includes artifact loading, materialization, request memory, admission policy,
paged KV, prefix append, speculative rounds, and SM86 W8 linear paths. The focused Windows gate
passed 11/11 tests. Benchmark logs remain local under `benchmark_results/` and are not included in
source releases.

## Upstream

NInfer-3090 is derived from [Neroued/ninfer](https://github.com/Neroued/ninfer). The upstream project
targets RTX 5090/`sm_120a`; this fork carries the Windows and SM86 compatibility layer, compact 35B
artifact support, and RTX 3090-specific schedules and memory planning.

## License

Apache License 2.0. See [LICENSE](LICENSE).
