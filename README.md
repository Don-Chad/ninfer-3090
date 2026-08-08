# NInfer-3090

## State-of-the-art Qwen3.6-35B-A3B inference on a single RTX 3090

NInfer-3090 is a dedicated C++20/CUDA inference engine built to extract maximum single-batch performance from Qwen3.6-35B-A3B on one 24 GB NVIDIA GeForce RTX 3090.

It delivers **state-of-the-art single-GPU RTX 3090 inference performance**, reaching **399–406 output tokens per second on structured, repetition-rich workloads**, while retaining target-model verification of speculative tokens.

The release benchmark sustains **399.16 ± 2.33 tok/s over 1,500 generated tokens**. The best clean five-run measurement reaches **406.44 ± 0.79 tok/s**.

For less repetitive workloads, NInfer reaches **310.04 ± 0.95 tok/s** on a long code-refactoring workload and **262.05 ± 1.62 tok/s** on the controlled MTP benchmark.

The project is intentionally specialized. It does not try to be a universal model runner. It is optimized around the RTX 3090, supported Qwen checkpoints, single-request inference, and aggressive use of CUDA Graphs, MTP speculative decoding, adaptive prompt lookup, and low-bit memory formats.

---

## Why this project exists

General-purpose runtimes such as llama.cpp need to support a huge range of hardware, model architectures, quantization formats, and deployment scenarios.

NInfer-3090 takes the opposite approach.

It specializes around:

- **NVIDIA GA102 / RTX 3090**
- **Qwen3.6-35B-A3B**
- a small set of supported checkpoints
- **one active request per GPU**
- RTX 3090-specific kernels and scheduling
- low-latency local inference
- OpenAI- and Anthropic-compatible serving

That narrow scope makes optimizations possible that are difficult to justify in a general-purpose runtime.

The goal is simple:

> **Make a single 24 GB RTX 3090 a genuinely high-performance inference device for Qwen3.6-35B-A3B.**

---

## RTX 3090 performance

| Workload | NInfer-3090 |
|---|---:|
| Structured repository edit, best clean run | **406.44 ± 0.79 tok/s** |
| Structured repository edit, release rerun | **399.16 ± 2.33 tok/s** |
| Realistic 1,500-token code refactor | **310.04 ± 0.95 tok/s** |
| Controlled MTP decode | **262.05 ± 1.62 tok/s** |
| Controlled MTP-3 decode | **260.55 ± 0.92 tok/s** |
| Ordinary decode, no speculation | **187.24 ± 0.27 tok/s** |
| 1,500-token prefill | **2,034.17 tok/s** |

The 400 tok/s result is **not** presented as universal chat throughput.

It is the performance of NInfer's adaptive structured-output path when the prompt contains enough reusable information to construct long candidate continuations.

Every candidate continuation is still verified by the target model before publication.

That distinction is central to NInfer:

> **It exploits structure when structure exists, while retaining the target model as the final authority over generated tokens.**

---

## Performance leadership on RTX 3090

NInfer-3090 sets the current state of the art for single-batch Qwen3.6-35B-A3B inference on an RTX 3090.

Public implementations of the same model on a single RTX 3090 typically report generation speeds in the **120–150 tok/s** range, while highly optimized alternative implementations can exceed 200 tok/s.

NInfer reaches:

- **262 tok/s** on controlled MTP decode
- **310 tok/s** over a long, less repetitive code-refactoring workload
- **399–406 tok/s** on structured workloads that activate adaptive prompt reuse

At **399.16 tok/s**, the release structured-output benchmark is roughly **2.7x a ~149 tok/s llama.cpp result** and more than **3x several common RTX 3090 Qwen3.6-35B-A3B configurations**.

These are not claimed to be perfectly identical benchmark fixtures. Quantization formats, prompts, runtime settings, and implementation details differ.

For that reason, NInfer publishes the benchmark methodology and exact reproduction commands below.

---

## What you get

### Qwen3.6-35B-A3B on one 24 GB GPU

The NInfer Qwen3.6-35B-A3B artifact is approximately **20.84 GiB**.

The standard text-only memory plan leaves enough headroom for:

- model weights
- CUDA Graph decode
- a practical 4K INT8 KV cache
- speculative decoding state

### Fast ordinary generation

For normal text generation, MTP speculative decoding reaches approximately:

```text
262.05 ± 1.62 tok/s
```

This is the conservative controlled reference for ordinary workloads.

### Very fast structured output

Repository edits, code transformations, templated responses, and similar workloads often contain long continuations that can be predicted from the input context.

NInfer can detect this automatically and switch to target-verified prompt lookup.

Measured results:

```text
Realistic code refactor:        310.04 ± 0.95 tok/s
Structured repository edit:     399.16 ± 2.33 tok/s
Best clean structured run:      406.44 ± 0.79 tok/s
```

### Low time to first token

| Input length | Mean TTFT |
|---:|---:|
| 128 tokens | **62.16 ms** |
| 512 tokens | **242.25 ms** |
| 1,500 tokens | **737.41 ms** |
| 1,500 tokens + hybrid decode | **751.75 ms** |

Model loading is measured separately.

Once the model is resident, subsequent requests reuse the loaded model and captured CUDA Graphs.

### Local API serving

`ninfer-serve` provides:

- OpenAI-compatible chat/completions routes
- Anthropic-compatible messages routes
- streaming
- tool-call support
- Windows binaries
- Ubuntu / WSL2 binaries

---

# Quick start

## 1. Download NInfer

Prebuilt releases are published at:

```text
https://github.com/Don-Chad/ninfer-3090/releases
```

Available packages:

| Platform | Package contents |
|---|---|
| Windows 11 x64 | `ninfer`, `ninfer-serve`, benchmark tool, runtime DLLs |
| Ubuntu 24.04 / WSL2 x64 | `ninfer`, `ninfer-serve`, benchmark tool |

### Windows requirements

The prebuilt Windows package includes the required CUDA runtime DLLs.

You still need:

- a compatible NVIDIA driver
- Microsoft Visual C++ 2022 runtime

You do **not** need the full CUDA Toolkit just to run the prebuilt package.

### Ubuntu / WSL2 requirements

The Ubuntu package dynamically uses the system CUDA runtime, FFmpeg, curl, and standard C++ libraries.

---

## 2. Download the model

The model is distributed separately from the runtime.

| Model | Artifact size | Download |
|---|---:|---|
| Qwen3.6-35B-A3B | 20.84 GB | `neroued/Qwen3.6-35B-A3B-NInfer` |
| Qwen3.6-27B | 16.29 GB | `neroued/Qwen3.6-27B-NInfer` |

### Qwen3.6-35B-A3B

```powershell
hf download neroued/Qwen3.6-35B-A3B-NInfer `
  qwen3_6_35b_a3b.ninfer `
  --local-dir models
```

The `.ninfer` artifact already contains the quantized weights and frontend resources.

It is not a GGUF file or Transformers checkpoint and does not require conversion after download.

### SHA-256

```text
Qwen3.6-35B-A3B
9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163

Qwen3.6-27B
74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127
```

---

## 3. Run the model

### Windows

```powershell
.\ninfer.exe models\qwen3_6_35b_a3b.ninfer `
  --prompt "Write a Python function that merges overlapping intervals." `
  --max-context 4096 `
  --prefill-chunk 128 `
  --max-new 512 `
  --kv-dtype int8 `
  --mtp-draft-tokens 3 `
  --lm-head-draft `
  --prompt-lookup-tokens 15 `
  --prompt-lookup-min-match 4 `
  --prompt-lookup-auto `
  --prompt-lookup-min-context 1000 `
  --text-only
```

### Ubuntu / WSL2

```bash
./ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a Python function that merges overlapping intervals." \
  --max-context 4096 \
  --prefill-chunk 128 \
  --max-new 512 \
  --kv-dtype int8 \
  --mtp-draft-tokens 3 \
  --lm-head-draft \
  --prompt-lookup-tokens 15 \
  --prompt-lookup-min-match 4 \
  --prompt-lookup-auto \
  --prompt-lookup-min-context 1000 \
  --text-only
```

For Qwen3.6-35B-A3B on a 24 GB RTX 3090, `--text-only` is required.

It leaves the text model intact but does not reserve the large vision workspace.

Image and video requests are rejected in this mode.

---

# Recommended configurations

## Adaptive hybrid

**Recommended default for mixed workloads**

```text
--mtp-draft-tokens 3 --lm-head-draft
--prompt-lookup-tokens 15 --prompt-lookup-min-match 4
--prompt-lookup-auto --prompt-lookup-min-context 1000
```

Adaptive hybrid combines:

1. MTP speculative decoding for ordinary text
2. prompt lookup for inputs with substantial reusable structure

At the configured minimum context, the runtime scans repeated 15-token n-grams.

Prompt lookup activates when repeated coverage crosses the configured threshold.

When active, the runtime tries K=15, K=8, and K=5 captured CUDA Graphs.

When lookup stops being useful, the runtime can return to MTP.

Measured result on the structured repository-edit fixture:

```text
Best clean run:       406.44 ± 0.79 tok/s
Release rerun:        399.16 ± 2.33 tok/s
```

---

## MTP speculative decoding

**Recommended for ordinary text**

```text
--mtp-draft-tokens 2 --lm-head-draft
```

MTP uses the model's trained proposal layer to draft candidate tokens, then verifies them with the target model.

Measured controlled throughput:

```text
262.05 ± 1.62 tok/s
```

K=2 is the fastest controlled `tg128` setting.

K=3 reaches similar throughput and is used by the adaptive hybrid mode because it behaves better when switching between MTP and lookup.

More draft tokens do not automatically mean higher throughput. Acceptance rate and verification latency both matter.

---

## Forced prompt lookup

**Recommended only when you already know the workload is highly repetitive**

Remove:

```text
--prompt-lookup-auto
```

and use:

```text
--mtp-draft-tokens 3 --lm-head-draft
--prompt-lookup-tokens 15 --prompt-lookup-min-match 4
```

Good candidates include:

- repository edits
- code transformations
- templated output
- document regeneration
- structured text with large repeated regions

Do not force this mode for ordinary prose.

Large speculative lookup windows can become slower when the prompt does not contain enough reusable structure.

---

## Ordinary decode

Use:

```text
--mtp-draft-tokens 0
```

This disables MTP and prompt lookup.

Measured controlled throughput:

```text
187.24 ± 0.27 tok/s
```

This mode is mainly useful for:

- baseline measurements
- acceptance debugging
- compatibility testing
- workloads where speculative execution is undesirable

---

# Why NInfer is fast

NInfer-3090 combines several layers of optimization.

## RTX 3090-native kernels

The GA102 path includes specialized schedules for:

- Q4/Q5/Q6/W8 weight execution
- small-T MoE decode
- Gated DeltaNet projection
- recurrent state
- fused output work
- argmax
- ordinary CUDA Graph decode
- MTP CUDA Graphs
- prompt-lookup CUDA Graphs

NInfer does not execute a generic model graph intended to support arbitrary architectures.

---

## CUDA Graph decode

CUDA Graphs are enabled by default and should generally stay enabled.

They reduce launch overhead and allow the decode path to execute as a pre-captured graph.

Disable only for diagnostics:

```text
--no-cuda-graph
```

---

## MTP speculative decoding

MTP uses the model's own trained proposal layer to generate candidate tokens.

Those candidates are verified against the target model before they are emitted.

This increases effective decode throughput without allowing an independent draft model to publish unverified output.

---

## Adaptive prompt lookup

Structured prompts frequently contain output that can be reused directly from the context.

Examples:

- source-code edits
- repository modifications
- JSON transformations
- configuration rewrites
- documents where most content stays unchanged

NInfer scans the request input for reusable token sequences.

When there is enough structure, it can propose longer continuations directly from the prompt.

Those candidate tokens are still verified by the target model before publication.

The adaptive mode matters because lookup is not free.

For normal prose, forcing a large verification window can be slower than MTP.

---

# KV cache and long context

After loading a 20.84 GiB model onto a 24 GB GPU, memory headroom is limited.

KV-cache format therefore has a major effect on usable context length.

---

## INT8 KV

**Recommended for normal short-context performance**

```text
--kv-dtype int8
```

This is the established 4K performance configuration.

BF16 KV consumes substantially more memory and reduces available context.

---

## K8/V4 KV

**Recommended for long-context capacity**

```text
--kv-dtype k8v4
```

In this mode:

- keys remain group-64 INT8
- values use packed group-64 INT4
- Q/K receive the same normalized Walsh-Hadamard rotation
- cache quantization and rotation are fused into the existing kernels

Measured results:

| Workload | INT8 K/V | K8/V4 |
|---|---:|---:|
| repeated-code `pp1500+tg1500` | 399.16 ± 2.33 tok/s | 395.08 ± 0.78 tok/s |
| lookup acceptance | 85.19% | 83.26% |
| KV payload at context 3,072 | 34.03 MiB | 25.78 MiB |
| 65,536-token prefill | n/a | 3,117.28 tok/s |
| KV payload at 131,072 capacity | n/a | 0.977 GiB |

A 131,072-token allocation uses approximately:

```text
20.825 GiB model weights
 1.102 GiB sequence arena
```

This fits on a 24 GiB RTX 3090, but leaves little room for another CUDA workload.

If you want roughly a 22 GiB operational ceiling, use around **120K context or less**.

For very large contexts, start with:

```text
--no-cuda-graph
```

because long-context CUDA Graphs require additional memory.

### Example

```powershell
.\ninfer.exe models\qwen3_6_35b_a3b.ninfer `
  --prompt "Summarize the following repository..." `
  --max-context 131072 `
  --prefill-chunk 1024 `
  --max-new 512 `
  --kv-dtype k8v4 `
  --mtp-draft-tokens 2 `
  --lm-head-draft `
  --no-cuda-graph `
  --text-only
```

---

## Rotated K8/V4

**Quality-first option for short and moderate contexts**

```text
--kv-dtype rk8v4
```

This additionally rotates values before INT4 quantization and applies the inverse transform after attention.

Measured on canonical `tg128`:

| KV mode | Acceptance | Throughput |
|---|---:|---:|
| K8/V4 | 60.53% | 240.35 tok/s |
| RK8/V4 | 75.49% | **274.50 tok/s** |
| INT8 | 70.75% | 262.05 tok/s |

In three deterministic 96-token quality prompts:

- two matched INT8 byte-for-byte
- one remained factually correct but diverged in wording

This is evidence of improved fidelity relative to plain K8/V4.

It is **not** a claim of universally lossless four-bit inference.

Large-prefill performance is lower with full value rotation:

```text
65,536-token prefill:

rk8v4: 1,103.66 tok/s
k8v4:  3,117.28 tok/s
```

Use:

- `rk8v4` for quality-first short/moderate contexts
- `k8v4` for long-context capacity and maximum prefill speed

---

## Full INT4 KV

Use:

```text
--kv-dtype int4
```

Full K4/V4 is intended for cases where maximum context capacity matters more than fidelity.

Measured:

```text
131,072-token allocation: ~21.73 GiB reserved
180,224-token eager allocation: ~21.98 GiB reserved
```

The 180K allocation loaded and executed, but 131K is the safer practical operating point when around 22 GiB is available.

On the controlled 4K MTP benchmark:

```text
INT4: 238.25 tok/s
INT8: 261.10 tok/s
```

INT4 is therefore a capacity option, not the default throughput configuration.

---

# API server

Start the server with:

```powershell
.\ninfer-serve.exe models\qwen3_6_35b_a3b.ninfer `
  --model-id qwen3.6-35b-a3b `
  --host 127.0.0.1 `
  --port 8080 `
  --max-context 4096 `
  --prefill-chunk 128 `
  --kv-dtype int8 `
  --mtp-draft-tokens 3 `
  --lm-head-draft `
  --prompt-lookup-tokens 15 `
  --prompt-lookup-min-match 4 `
  --prompt-lookup-auto `
  --prompt-lookup-min-context 1000 `
  --text-only
```

The server exposes:

- OpenAI-compatible chat/completions routes
- Anthropic-compatible messages routes
- streaming
- tool calls
- runtime logging

See:

```text
docs/serving.md
```

for request schemas and additional details.

---

# Benchmark methodology

All headline release-candidate measurements were produced locally on one NVIDIA GeForce RTX 3090.

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| Model | Qwen3.6-35B-A3B |
| Artifact | 20.84 GB NInfer artifact |
| CUDA | 13.2 runtime / driver |
| KV cache | INT8 group-64 |
| CUDA Graphs | enabled |
| Mode | text-only |
| Concurrent GPU workload | none |

Benchmark reports include:

- all measured repetitions
- acceptance counts
- memory usage
- hardware details
- runtime version
- exact command

Model loading and CUDA Graph construction are excluded from decode throughput.

Prefill time includes prompt execution through the first generated token.

Decode throughput covers the configured number of generated tokens.

---

## Canonical decode benchmark

The canonical comparison generates 128 timed tokens.

Five measured repetitions follow five discarded warmups.

```powershell
.\ninfer_bench.exe `
  --weights models\qwen3_6_35b_a3b.ninfer `
  --corpus bench\fixtures\bench_corpus.ids `
  -n 128 `
  -r 5 `
  --warmup 5 `
  --max-ctx 256 `
  --prefill-chunk 128 `
  --kv-dtype int8 `
  --mtp-draft-tokens 2 `
  --lm-head-draft `
  --text-only
```

Measured:

```text
262.05 ± 1.62 tok/s
```

`tg128` is deliberately short and controlled.

It is useful for comparing kernel and decode configurations. It is not intended to represent every real-world prompt.

---

## Long structured-output benchmark

The adaptive hybrid benchmark uses:

- 1,500 input tokens
- 1,500 generated tokens
- a fixed Python repository-edit fixture
- five measured repetitions
- two warmups

```powershell
.\ninfer_bench.exe `
  --weights models\qwen3_6_35b_a3b.ninfer `
  --corpus benchmark-results\prompt-lookup-code-long\python_repository_edit_1500.ids `
  -pg 1500,1500 `
  -r 5 `
  --warmup 2 `
  --max-ctx 3072 `
  --prefill-chunk 128 `
  --kv-dtype int8 `
  --mtp-draft-tokens 3 `
  --lm-head-draft `
  --prompt-lookup-tokens 15 `
  --prompt-lookup-min-match 4 `
  --prompt-lookup-auto `
  --prompt-lookup-min-context 1000 `
  --text-only
```

Release rerun:

```text
399.16 ± 2.33 output tok/s
1,995.52 prefill tok/s
85.19% aggregate lookup acceptance
13.44 mean output tokens per speculative round
36.13 ms mean speculative-round latency
```

Best clean five-run measurement of the same fixture:

```text
406.44 ± 0.79 tok/s
```

Both numbers are reported deliberately so the peak result is visible without hiding run-to-run variation.

---

# Time to first token

| Input length | Mean TTFT | Notes |
|---:|---:|---|
| 128 tokens | **62.16 ms** | five-run release benchmark |
| 512 tokens | **242.25 ms** | five-run release benchmark |
| 1,500 tokens | **737.41 ms** | prefill-only benchmark |
| 1,500 + hybrid decode | **751.75 ms** | long combined benchmark |

Cold engine construction is separate.

The measured cold load was approximately:

```text
5.42 seconds total
4.28 seconds host-to-device upload
```

Once loaded, later requests reuse the resident model and captured graphs.

---

# Model quality and verification

NInfer's speculative paths do not simply substitute output from a smaller draft model.

MTP proposals are verified by the target model.

Prompt-lookup proposals are also verified by the target model before publication.

Acceptance rate therefore measures how much speculative work can be reused. It is not a model-capability score.

Published model-card evaluations include:

| Model | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B | 90.00% | 90.00% | 85.35% |
| Qwen3.6-27B | 86.67% | 93.33% | 86.87% |

See the upstream model cards for evaluation configuration and limitations.

---

# Build from source

Source builds require:

- CMake 3.28+
- C++20 compiler
- CUDA Toolkit 13.0+

---

## Ubuntu 24.04 / WSL2

Install:

- FFmpeg 6 development libraries
- curl
- pkg-config
- GCC 13
- Ninja
- CUDA

Then:

```bash
git clone https://github.com/Don-Chad/ninfer-3090.git
cd ninfer-3090

cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel

ctest \
  --test-dir build \
  --output-on-failure
```

---

## Windows 11

Use:

- Visual Studio 2022
- CMake
- CUDA Toolkit 13.0+
- vcpkg

```powershell
cmake -S . -B build-windows `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake

cmake --build build-windows `
  --config Release `
  --parallel

ctest `
  --test-dir build-windows `
  -C Release `
  --output-on-failure
```

---

# Scope and limitations

NInfer-3090 is intentionally not a replacement for a general-purpose inference framework.

| Capability | Support |
|---|---|
| Single RTX 3090 | **Primary target** |
| Qwen3.6-35B-A3B | **Supported** |
| Qwen3.6-27B | Supported |
| One active request per GPU | Yes |
| Continuous batching | No |
| Arbitrary GGUF models | No |
| Arbitrary Safetensors checkpoints | No |
| OpenAI-compatible serving | Yes |
| Anthropic-compatible serving | Yes |
| Windows | Yes |
| Ubuntu / WSL2 | Yes |

This specialization is part of the design.

If you need:

- broad model compatibility
- arbitrary architectures
- large-scale continuous batching
- many concurrent users per GPU

a general-purpose inference framework is likely the better choice.

If you want maximum single-request Qwen3.6 performance from one RTX 3090, NInfer is built for that narrower problem.

---

# Project history

NInfer-3090 began as a port of:

```text
https://github.com/Neroued/ninfer
```

The original implementation targets the RTX 5090.

The RTX 3090 edition now has its own:

- GA102 kernel schedules
- 35B-A3B memory profile
- Windows and Ubuntu release workflow
- benchmark suite
- MTP tuning
- adaptive hybrid decoding
- long-context KV modes
- serving binaries

---

# Contributing

Issues and pull requests are welcome.

Especially useful contributions include:

- reproducible RTX 3090 performance measurements
- additional real-world structured-output fixtures
- llama.cpp / vLLM / TensorRT-LLM comparisons
- Windows packaging fixes
- Ubuntu / WSL2 packaging fixes
- correctness tests
- alternative model support

For performance claims, please include:

1. exact model artifact
2. exact command
3. GPU and power state
4. runtime / build version
5. context length
6. generation length
7. warmup count
8. raw benchmark output

Reproducible numbers are much more useful than isolated screenshots.

---
