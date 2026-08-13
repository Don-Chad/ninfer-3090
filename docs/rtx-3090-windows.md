# NInfer-3090 v0.4.0 for Windows

This native Windows release supports the Qwen3.6-27B artifact and the compact, text-only
Qwen3.6-35B-A3B v0.3.1 artifact. The runtime adds paged KV, concurrent request execution,
compatible-prefix reuse, bounded admission, and improved serving APIs.

## Requirements

- Windows 11 x64;
- GeForce RTX 3090 with a recent NVIDIA driver;
- Microsoft Visual C++ 2022 runtime;
- a supported `.ninfer` model artifact.

The bundled applications and dependency DLLs are native Windows executables. Model artifacts are
not included in the release archive.

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

The v0.4 Windows release gate rebuilt `ninfer.exe`, `ninfer-serve.exe`, and `ninfer_bench.exe` and
passed 11 focused tests covering artifact reading/materialization, request memory, admission,
paged KV, prefix append, speculative rounds, and the three relevant SM86 W8 linear paths.
