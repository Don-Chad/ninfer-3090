# NInfer RTX 3090

Run a genuinely capable **35B-class MoE model at 300–400+ output tok/s on a single RTX 3090** when
the workload gives speculative decoding useful structure. NInfer turns the still-formidable
24 GB GA102 card into a specialized local inference appliance for **Qwen3.6-35B-A3B**, with native
Windows and Ubuntu/WSL2 binaries, CUDA Graph decode, low-bit weights, INT8 KV cache, MTP
speculation, and adaptive prompt lookup.

This is not a generic model runner with a few GPU flags. It is a C++20/CUDA engine shaped around
this model and this GPU: quantized 35B-A3B weights fit beside a 4K-capable runtime, the MoE and
Gated DeltaNet paths use GA102 schedules, and speculative execution changes strategy automatically
between ordinary prompts and repetition-rich code.

## Headline RTX 3090 performance

| Qwen3.6-35B-A3B mode | Workload | Result |
|---|---|---:|
| **Adaptive hybrid K15/K8/K5 + MTP-3, best clean run** | 1,500-token repeated-code prompt + 1,500 generated tokens | **406.44 ± 0.79 tok/s** |
| **Adaptive hybrid, v0.3.0 release rerun** | same repetition-rich fixture and settings | **399.16 ± 2.33 tok/s** |
| **MTP-3, realistic two-copy code refactor** | 1,500-token code prompt + 1,500 generated tokens | **310.04 ± 0.95 tok/s** |
| **MTP-2** | canonical `tg128` fixture | **262.05 ± 1.62 tok/s** |
| **MTP-3** | canonical `tg128` fixture | **260.55 ± 0.92 tok/s** |
| **Ordinary decode** | canonical `tg128` fixture | **187.24 ± 0.27 tok/s** |
| **Prefill** | 1,500-token prompt through first token | **2,034.17 tok/s / 737.41 ms** |

The 406.44 tok/s best and 399.16 tok/s release rerun are real end-to-end decode measurements, but
they are intentionally labeled:
the input contains repeated Python code and gives prompt lookup useful continuations. It is not
presented as universal model throughput. The much less repetitive two-copy code fixture still
clears **310 tok/s** over a long 1,500-token generation. On ordinary non-repetitive text, use the
canonical MTP numbers as the conservative controlled reference.

The long hybrid benchmark also averages **752 ms from request start through prompt processing and
the first generated token** at 1,500 input tokens. Short-prompt TTFT measurements are listed in
[How performance is measured](#how-performance-is-measured).

## Why this release is special

- **A capable 35B-A3B model on one consumer GPU.** The 20.84 GiB mixed low-bit artifact and
  text-only memory plan leave enough room for a 4K INT8 KV cache and captured decode graphs.
- **The runtime chooses the speculative strategy.** MTP handles ordinary requests; a sub-0.1 ms
  input scan enables K15/K8/K5 target-only lookup only when the prompt contains enough reusable
  structure.
- **Fast long-form code, not only a 128-token microbenchmark.** The realistic two-copy code
  refactor reaches 310.04 tok/s across 1,500 generated tokens; the repetition-rich repository-edit
  fixture reaches 399–406 tok/s.
- **RTX 3090-native kernels.** GA102 schedules cover MoE decode, Gated DeltaNet projection and
  recurrent state, output/argmax work, and CUDA Graph families for ordinary, MTP, and lookup paths.
- **A complete local serving stack.** Native CLI, benchmark runner, and OpenAI-/Anthropic-compatible
  server are shipped for Windows and Ubuntu/WSL2.

## Download

Prebuilt packages are published on [GitHub Releases](https://github.com/Don-Chad/ninfer-3090/releases):

- Windows 11 x64: `ninfer`, `ninfer-serve`, benchmark tool, and required runtime DLLs.
- Ubuntu 24.04 / WSL2 x64: `ninfer`, `ninfer-serve`, and benchmark tool.

The model is distributed separately because the 35B-A3B artifact is approximately **20.84 GB**:

| Model | Download | Artifact size | SHA-256 |
|---|---|---:|---|
| Qwen3.6-35B-A3B | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | 20.84 GB | `9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163` |
| Qwen3.6-27B | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | 16.29 GB | `74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127` |

```powershell
hf download neroued/Qwen3.6-35B-A3B-NInfer `
  qwen3_6_35b_a3b.ninfer `
  --local-dir models
```

The `.ninfer` artifact already contains the quantized weights and frontend resources. It is not a
GGUF or Transformers checkpoint and does not require conversion after download.

## Recommended 35B-A3B command

### Windows

```powershell
.\ninfer.exe models\qwen3_6_35b_a3b.ninfer `
  --prompt "Write a Python function that merges overlapping intervals." `
  --max-context 4096 --prefill-chunk 128 --max-new 512 `
  --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft `
  --prompt-lookup-tokens 15 --prompt-lookup-min-match 4 `
  --prompt-lookup-auto --prompt-lookup-min-context 1000 `
  --text-only
```

### Ubuntu / WSL2

```bash
./ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a Python function that merges overlapping intervals." \
  --max-context 4096 --prefill-chunk 128 --max-new 512 \
  --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft \
  --prompt-lookup-tokens 15 --prompt-lookup-min-match 4 \
  --prompt-lookup-auto --prompt-lookup-min-context 1000 \
  --text-only
```

`--text-only` is required for the 35B-A3B model on a 24 GB RTX 3090. It leaves the model intact but
does not reserve the large vision workspace. Image and video requests are rejected in this mode.

## Inference modes

### Adaptive hybrid — recommended general mode

Adaptive hybrid combines MTP-3 with target-only prompt lookup:

```text
--mtp-draft-tokens 3 --lm-head-draft
--prompt-lookup-tokens 15 --prompt-lookup-min-match 4
--prompt-lookup-auto --prompt-lookup-min-context 1000
```

At the configured minimum context, the runtime classifies the request input using 15-token repeated
n-gram coverage. Prompt lookup activates at 25% coverage; generated boilerplate cannot switch an
ordinary request into lookup halfway through its answer. When active, lookup tries K=15, K=8, and
K=5 captured CUDA Graphs. A guarded lazy realignment path restores MTP after lookup stops.

This mode reached **406.44 ± 0.79 tok/s** in its best clean five-run measurement and
**399.16 ± 2.33 tok/s** in the final release rerun, while retaining MTP as the normal path for
non-repetitive contexts.

### MTP speculative decoding — recommended for ordinary text

MTP uses the model's trained proposal layer and verifies several draft tokens with the target model.

```text
--mtp-draft-tokens 2 --lm-head-draft
```

K=2 is the fastest controlled `tg128` setting at **262.05 tok/s**. K=3 reaches similar throughput
and is the hybrid default because it provides better fallback behavior when lookup is enabled.
Acceptance and round latency both matter: a larger K is not automatically faster.

### Forced prompt lookup — specialized mode

Omit `--prompt-lookup-auto` to force lookup attempts:

```text
--mtp-draft-tokens 3 --lm-head-draft
--prompt-lookup-tokens 15 --prompt-lookup-min-match 4
```

Use this only when the workload is known to contain substantial repetition, such as repository
edits, code transformations, templated output, or regenerating a document with local changes.
Forced K=15 can regress ordinary prose because the target still pays to verify 16 positions.

### Ordinary decode — reference and compatibility mode

```text
--mtp-draft-tokens 0
```

This disables MTP and prompt lookup. It is useful as a stable baseline, for acceptance debugging,
and for workloads where speculative execution is undesirable.

### CUDA Graphs and KV cache

CUDA Graphs are enabled by default and should remain enabled for production throughput.
`--no-cuda-graph` exists for diagnostics. For 35B-A3B on a 24 GB card, use `--kv-dtype int8`;
BF16 KV substantially reduces available context.

### Experimental rotated INT4 KV for long context

The recommended RotorQuant-style capacity mode is `--kv-dtype k8v4`: keys remain group-64 INT8,
values use packed group-64 INT4, and Q/K receive the same normalized Walsh-Hadamard rotation. This
keeps the attention-score path at eight bits while reducing the value-cache traffic and storage by
half. Rotation and quantization are fused into the existing prompt-fill and decode cache-write
kernels; there is no standalone rotation launch or intermediate rotated tensor.

```powershell
.\ninfer.exe models\qwen3_6_35b_a3b.ninfer `
  --prompt "Summarize the following repository..." `
  --max-context 131072 --prefill-chunk 1024 --max-new 512 `
  --kv-dtype k8v4 --mtp-draft-tokens 2 --lm-head-draft `
  --no-cuda-graph --text-only
```

Measured on the development RTX 3090:

| Workload | INT8 K/V | Rotated K8/V4 |
|---|---:|---:|
| repeated-code `pp1500+tg1500` | 399.16 ± 2.33 tok/s | 395.08 ± 0.78 tok/s |
| lookup acceptance | 85.19% | 83.26% |
| KV payload at context 3,072 | 34.03 MiB | 25.78 MiB |
| 65,536-token prefill | — | 3,117.28 tok/s |
| KV payload at 131,072 capacity | — | 0.977 GiB |

The 128K allocation uses 1.102 GiB for the full sequence arena and 20.825 GiB for model weights,
so it fits a 24 GiB RTX 3090 but should run without another CUDA workload. Use 120K or less when a
hard 22 GiB operational ceiling is required.

`--kv-dtype int4` stores two signed 4-bit K/V codes per byte with FP16 group-64 scales. Keys and
queries receive the same normalized 64-wide Walsh-Hadamard rotation before quantization, so their
unquantized dot product is preserved while isolated key outliers are spread across the group.
Values remain unrotated and are dequantized inside the fused attention kernel.

Use full K4/V4 only when maximum context capacity matters more than model fidelity:

```powershell
.\ninfer.exe models\qwen3_6_35b_a3b.ninfer `
  --prompt "Summarize the following repository..." `
  --max-context 131072 --prefill-chunk 1024 --max-new 512 `
  --kv-dtype int4 --mtp-draft-tokens 2 --lm-head-draft `
  --no-cuda-graph --text-only
```

On the development RTX 3090, a 131,072-token allocation reserves 21.73 GiB and a 65,536-token
prefill completed at 3,061.91 tok/s. A 180,224-token eager allocation also loaded and executed at
21.98 GiB reserved, but 131,072 is the safer practical ceiling when about 22 GiB is available.
Long-context CUDA Graphs require additional memory, so start with `--no-cuda-graph`.

INT4 is not the default speed route. On the controlled 4K `tg128` MTP benchmark, rotated INT4
reached 238.25 tok/s versus 261.10 tok/s for INT8 because its lower precision reduced proposal
acceptance. Keep `--kv-dtype int8` for the established short-context throughput result.

Direct packed-INT4 QK MMA was deliberately not selected for K8/V4: it accelerates the K4 key path,
whereas the measured quality/acceptance frontier selected eight-bit keys. Likewise, selective
high-precision sink heads were not enabled without a stable teacher-forced calibration signal;
free-running acceptance changes the generated sequence and is not a safe per-head calibration
objective.

## How performance is measured

All release-candidate headline measurements were produced locally on one NVIDIA GeForce RTX 3090
with the same 20.84 GB Qwen3.6-35B-A3B artifact, CUDA 13.2 runtime/driver, INT8 group-64 KV,
CUDA Graphs, text-only mode, and no simultaneous GPU workload.

### Canonical decode

The canonical comparison generates 128 timed tokens from the committed token fixture. It uses five
measured repetitions after five discarded warm-ups:

```powershell
.\ninfer_bench.exe `
  --weights models\qwen3_6_35b_a3b.ninfer `
  --corpus bench\fixtures\bench_corpus.ids `
  -n 128 -r 5 --warmup 5 --max-ctx 256 --prefill-chunk 128 `
  --kv-dtype int8 --mtp-draft-tokens 2 --lm-head-draft --text-only
```

`tg128` is deliberately short and controlled. It is useful for kernel and scheduling comparisons,
not a claim about every prompt.

### Long repeated-code hybrid

The hybrid result uses a fixed 1,500-token Python repository-edit prompt and generates exactly
1,500 timed output tokens. Five repetitions follow two warm-ups:

```powershell
.\ninfer_bench.exe `
  --weights models\qwen3_6_35b_a3b.ninfer `
  --corpus benchmark-results\prompt-lookup-code-long\python_repository_edit_1500.ids `
  -pg 1500,1500 -r 5 --warmup 2 --max-ctx 3072 --prefill-chunk 128 `
  --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft `
  --prompt-lookup-tokens 15 --prompt-lookup-min-match 4 `
  --prompt-lookup-auto --prompt-lookup-min-context 1000 --text-only
```

This run measured:

- **399.16 ± 2.33 output tok/s**
- **1,995.52 prefill tok/s**
- 85.19% aggregate lookup acceptance
- 13.44 mean output tokens per speculative round
- 36.13 ms mean speculative-round latency

The best clean run of the same fixed fixture and command measured **406.44 ± 0.79 output tok/s**.
The release headline shows both values so the peak is visible without disguising run-to-run
variation.

The benchmark excludes model loading and CUDA Graph construction. Prefill time includes prompt
execution through the first generated token; decode throughput covers the following fixed number
of generated tokens. Reports include every repetition, acceptance counts, memory usage, hardware,
runtime version, and the exact command.

### Time to first token

| Input length | Mean TTFT | Meaning |
|---:|---:|---|
| 128 tokens | **62.16 ms** | five-run release benchmark |
| 512 tokens | **242.25 ms** | five-run release benchmark |
| 1,500 tokens | **737.41 ms** | prefill-only release benchmark |
| 1,500 tokens + hybrid decode | **751.75 ms** | long combined benchmark |

Model loading is separate. The release TTFT run measured 5.42 seconds for cold engine construction,
including 4.28 seconds of host-to-device upload; subsequent requests use the resident model and
captured graphs.

## OpenAI- and Anthropic-compatible server

```powershell
.\ninfer-serve.exe models\qwen3_6_35b_a3b.ninfer `
  --model-id qwen3.6-35b-a3b `
  --host 127.0.0.1 --port 8080 `
  --max-context 4096 --prefill-chunk 128 --kv-dtype int8 `
  --mtp-draft-tokens 3 --lm-head-draft `
  --prompt-lookup-tokens 15 --prompt-lookup-min-match 4 `
  --prompt-lookup-auto --prompt-lookup-min-context 1000 `
  --text-only
```

The server exposes OpenAI chat/completions and Anthropic messages routes. See
[Serving](docs/serving.md) for request schemas, streaming, tool calls, and runtime logging.

## Build from source

Source builds require CMake 3.28+, a C++20 compiler, and CUDA Toolkit 13.0+.

### Ubuntu 24.04 / WSL2

Install FFmpeg 6 development libraries, curl, pkg-config, GCC 13, Ninja, and CUDA, then:

```bash
git clone https://github.com/Don-Chad/ninfer-3090.git
cd ninfer-3090
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Windows 11

Use Visual Studio 2022, CMake, CUDA Toolkit 13.0+, and vcpkg:

```powershell
cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build-windows --config Release --parallel
ctest --test-dir build-windows -C Release --output-on-failure
```

The Windows prebuilt package includes the CUDA runtime and dependency DLLs. Users of the prebuilt
package need a current compatible NVIDIA driver and the Microsoft Visual C++ 2022 runtime, not the
full CUDA Toolkit. The Ubuntu package dynamically uses the system CUDA runtime, FFmpeg, curl, and
standard C++ libraries.

## Scope and design

NInfer deliberately specializes for exact supported checkpoints instead of implementing a generic
model graph. The RTX 3090 route includes GA102-specific Q4/Q5/Q6/W8 schedules, small-T MoE kernels,
Gated DeltaNet projection and recurrent-state kernels, fused output work, CUDA Graph families, MTP,
prefix reuse, and adaptive prompt lookup.

The project supports one active request on one GPU. It is not a continuous-batching server and does
not load arbitrary GGUF or Safetensors models.

## Model quality

Speculative decoding does not replace target-model verification: returned tokens are licensed by
the target model under the selected sampling configuration. Prompt lookup drafts are likewise
verified before publication. Acceptance rate measures speed opportunity, not model capability.

Published model-card evaluations:

| Model | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B | 90.00% | 90.00% | 85.35% |
| Qwen3.6-27B | 86.67% | 93.33% | 86.87% |

See the model cards for evaluation configuration and limitations.

## Contributing

Issues and pull requests are welcome, especially for reproducible RTX 3090 kernel measurements,
additional real-world prompt-lookup fixtures, Windows/Ubuntu packaging fixes, and correctness
tests. Include the exact command, GPU state, artifact, and raw benchmark report with performance
claims.

## Project history

This RTX 3090 edition began as a port of
[Neroued/ninfer](https://github.com/Neroued/ninfer), whose original implementation targets the
RTX 5090. It now has its own native Windows/Ubuntu release workflow, GA102 kernel schedules,
35B-A3B memory profile, benchmark suite, MTP tuning, and adaptive hybrid decoding path.
