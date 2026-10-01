# NInfer-3090 for Windows

> Not the archive README. This is the checkout-side Windows guide, so the scripts it names live
> under `scripts/`. The README shipped inside the release archive is
> [`release-archive-windows.md`](release-archive-windows.md), and `VERSION` in the archive says
> which release you have.

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

## Recommended model: Qwen3.8-27B

`scripts\download-model.bat qwen38-27b` fetches the pinned Qwen3.8-27B artifact and
`scripts\run.bat qwen38-27b` serves it with the tuned profile (see the
[README](../README.md#quick-start)). The rest of this guide documents the Qwen3.6-35B-A3B MoE, which
serves more lanes.

## Download the compatible Qwen3.6-35B artifact

`scripts/download-model.bat qwen36-35b-a3b` fetches this pinned revision and is the simplest way to
get it. Pin the revision explicitly because the Hugging Face repository's unpinned `main` file can
move:

```powershell
hf download neroued/Qwen3.6-35B-A3B-NInfer `
  qwen3_6_35b_a3b.ninfer `
  --revision ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6 `
  --local-dir models

Get-FileHash .\models\qwen3_6_35b_a3b.ninfer -Algorithm SHA256
```

Expected SHA-256:
`3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84`.

This is the v3 container artifact, carrying the DFlash bundle needed for `--spec dflash`. The
runtime reads only v3. A v2 file from an earlier release (for example the `560f227e` pin) is refused
at load; upgrade it locally instead of downloading again:

```powershell
python tools\upgrade_ninfer_v2_to_v3.py models\qwen3_6_35b_a3b.ninfer models\qwen3_6_35b_a3b.v3.ninfer
```

**The DFlash bundle costs nothing in VRAM unless you ask for DFlash.** The published RTX 3090
concurrency measurements were taken against the smaller v2 `c8b8c1c0` artifact; loading with
`--spec` unset binds the same resident weights, and only `--spec dflash` maps the additional
bundle. The concurrency table below therefore still applies.

## Run the concurrent server

Keep KV capacity explicit on a 24 GB card. Automatic sizing reserves an additional 1 GiB of
headroom and may reject an otherwise viable compact-35B configuration.

```powershell
.\ninfer-serve.exe models\qwen3_6_35b_a3b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 4096 --kv-capacity 4096 `
  --max-concurrency 4 --max-pending-requests 32 --pending-timeout-ms 600000 `
  --prefill-chunk 512 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Prefix reuse is enabled by default; `--no-prefix-reuse` disables it. The server exposes OpenAI
Responses, OpenAI Chat Completions, and Anthropic Messages-compatible endpoints. Run
`.\ninfer-serve.exe --help` for the complete option list.

`--spec dflash` needs an artifact with the DFlash bundle, such as the pinned v3 artifact above;
without it startup refuses the selection explicitly.

## Qwen3.8-27B C8/8K profile

ReplaySSM reduces speculative GDN state memory enough for the maximum-concurrency Qwen3.8 profile
to use MTP3:

```powershell
.\ninfer-serve.exe models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 `
  --max-concurrency 8 --max-pending-requests 32 --pending-timeout-ms 600000 `
  --prefill-chunk 512 --kv-dtype int8 `
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

The paged cache supports seven KV formats — `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `k8v4` and `nvfp4`.
INT8 remains the
recommended default. On the development RTX 3090, `rk8v4` raises the measured automatic-sizing
boundary from 171,648 to 226,560 tokens at 1 GiB headroom, for 5.51 GiB of KV against INT8's
5.40 GiB.

Since the port onto the `kv_cache_append` Op, that context gain has a measured quality cost of
**+0.092% perplexity** (`ninfer-perplexity`, `ninfer-ppl-1m-v1` quick, 261,167 scored tokens:
4.342425 on INT8 against 4.346413 on rk8v4; both re-measured September 2026 and both
bit-reproducible run to run). Values are not rotated, so there is no
inverse-rotation pass over the attention output. Values do use a finer group than keys, 32 against
64, which halved that penalty from +0.146% at no cost in context.

Decode cost depends on speculation. Without it the packed value plane is within about 1% of INT8,
because halved value traffic and the added unpack nearly cancel. With MTP3, C1 decode falls about
5%, from a mean 81.61 tok/s across three runs to 77.39, because the lower value precision reduces
draft acceptance from 71.27% to 65.86%. The finer value group does not fix that: it recovers 44% of
the perplexity penalty but only about 15% of the acceptance loss, because acceptance turns on exact
token agreement rather than on mean error. Prefer `rk8v4` when context is the binding constraint,
and INT8 when decode throughput under speculation matters more.

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
