# NInfer-3090 v0.5.0 for Windows

This native Windows release supports Qwen3.8-27B, Qwen3.6-27B, and the compact, text-only
Qwen3.6-35B-A3B v0.3.1 artifact. The runtime provides paged KV, concurrent request execution,
compatible-prefix reuse, bounded admission, ReplaySSM, reasoning-effort control, and
OpenAI/Anthropic serving APIs.

## Requirements

- Windows 11 x64;
- GeForce RTX 3090 with a recent NVIDIA driver;
- Microsoft Visual C++ 2022 runtime;
- a supported `.ninfer` model artifact.

The bundled applications and dependency DLLs are native Windows executables. Model artifacts are
not included in the release archive.

## Download the compatible Qwen3.6-35B artifact

The published RTX 3090 measurements use the compact 20.84 GiB container-v1 artifact. Pin its
revision because the Hugging Face repository's unpinned `main` file is now the larger 21.22 GiB
container-v2 artifact with DFlash weights:

```powershell
hf download neroued/Qwen3.6-35B-A3B-NInfer `
  qwen3_6_35b_a3b.ninfer `
  --revision c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c `
  --local-dir models

Get-FileHash .\models\qwen3_6_35b_a3b.ninfer -Algorithm SHA256
```

Expected SHA-256:
`9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163`.

The v0.5 runtime reader accepts both v1 and v2 containers. An error that says only
`artifact magic is not NInfer version 1` comes from an older executable; replace it with the
[v0.5.0 Windows release](https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.5.0-rtx3090).
Although v2 is readable, its DFlash-bearing payload is not the artifact used to qualify the 24 GB
3090 cohort profiles, so pinned v1 remains the recommended download.

## Run the concurrent server

Keep KV capacity explicit on a 24 GB card. Automatic sizing reserves an additional 1 GiB of
headroom and may reject an otherwise viable compact-35B configuration.

```powershell
.\ninfer-serve.exe models\qwen3_6_35b_a3b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 4096 --kv-capacity 4096 `
  --max-concurrency 4 --max-pending-requests 32 `
  --prefill-chunk 512 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Prefix reuse is enabled by default; `--no-prefix-reuse` disables it. The server exposes OpenAI
Responses, OpenAI Chat Completions, and Anthropic Messages-compatible endpoints. Run
`.\ninfer-serve.exe --help` for the complete option list.

The compact 35B artifact does not contain DFlash weights. Do not select `--spec dflash`; the
runtime reports the missing optional weights explicitly.

## Qwen3.8-27B C8/8K profile

ReplaySSM reduces speculative GDN state memory enough for the maximum-concurrency Qwen3.8 profile
to use MTP3:

```powershell
.\ninfer-serve.exe models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 `
  --max-concurrency 8 --max-pending-requests 32 `
  --prefill-chunk 1024 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

This 8,192-token shared-pool configuration measured 114.73 aggregate end-to-end tok/s for eight
simultaneous 128-token generations and peaked at 21,818 MiB. Set `--kv-capacity 65536` when all
eight requests need independent 8K capacity; that stronger reservation measured 114.88 tok/s and
23,745 MiB peak. Avoid competing GPU processes.

The cohort size is fixed at startup, but its active membership is not: every decode round compacts
all ready requests into one batch, while completed slots disappear. Follow-up requests wait in the
bounded pending queue and join at a safe round boundary when a lane and memory are available. This
is more predictable than unrestricted dynamic batching because maximum VRAM, workspace, and CUDA
Graph shapes are reserved in advance.

Qwen3.8 supports `low`, `medium`, and `xhigh` reasoning effort. For Chat Completions add the
top-level field `"reasoning_effort": "xhigh"`; Responses uses
`"reasoning": {"effort": "xhigh"}`. The CLI accepts
`--reasoning-effort low|medium|xhigh`.

The paged cache supports BF16 and INT8 storage. RotorQuant/KV4 from older contiguous-cache work is
not a supported v0.5 paged-cache mode.

## Measured 35B capacity

These results used the compact 20.84 GiB artifact on an otherwise idle RTX 3090:

| Workload | Concurrency | Aggregate decode | Peak VRAM |
|---|---:|---:|---:|
| 128 output tokens/request | 1 | 162.7 tok/s | 21.90 GiB |
| 128 output tokens/request | 2 | 267.9 tok/s | 22.21 GiB |
| 128 output tokens/request | 4 | 366.2 tok/s | 22.83 GiB |
| 128 output tokens/request | 6 | 383.4 tok/s | 23.47 GiB |
| 512 output tokens/request | 2 | 399.1 tok/s | within 24 GB |

Concurrency 8 was rejected by admission rather than overcommitting the GPU. Repeating a compatible
26-token prompt reused 24 prefix tokens and reduced measured prefill from 371 ms to 10 ms.

## Build from source

Use Visual Studio 2022, CUDA 12.8 or newer, CMake, and vcpkg:

```powershell
$vcpkgToolchain = 'C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake'

cmake -S . -B build-windows -G 'Visual Studio 17 2022' -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$vcpkgToolchain" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-windows --config Release --parallel
```

The source rejects unsupported CUDA architectures for this fork. CUDA 13 uses MSVC's conforming
preprocessor automatically.

## Release validation

The v0.5 Windows release gate rebuilt `ninfer.exe`, `ninfer-serve.exe`, and `ninfer_bench.exe`,
loaded the official Qwen3.8 artifact, generated coherent output, and completed C1-C4 plus C8/8K
serving checks. Focused tests cover artifact reading/materialization, request memory, admission,
paged KV, prefix append, speculative rounds, and the relevant SM86 W8 linear paths.
