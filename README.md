# NInfer for RTX 3090

> Fast, single-GPU Qwen3.6 inference on a 24 GiB RTX 3090, for native Windows and Linux/WSL2.

NInfer runs the supported Qwen3.6 models locally through a command-line application or an
OpenAI-/Anthropic-compatible HTTP server. The fastest verified result so far is **252.83 +/- 0.49
tok/s** for Qwen3.6-35B-A3B using text-only mode and MTP-1 on one RTX 3090.

You do not have to compile the project yourself when a matching prebuilt package is available.
Download a Windows or Linux archive from [GitHub Releases](https://github.com/Don-Chad/ninfer-3090/releases),
download the model artifact separately, and run the included CLI or server. Release archives are
published separately from Git source; if the Releases page does not yet contain a package, use the
[source-build instructions](#build) below.

The phrase "from-scratch C++/CUDA inference engine" describes how NInfer itself is implemented; it
does not mean every user must build it from source.

## About this port

This repository ports the original [Neroued/ninfer](https://github.com/Neroued/ninfer) RTX 5090
implementation to the RTX 3090 (`sm_86`). It adds native Windows support while retaining a shared
Linux/WSL2 codebase, and tunes the inference kernels for GA102. Qwen3.6-27B and the text-only
Qwen3.6-35B-A3B configuration are verified targets; see the
[35B-A3B RTX 3090 report](docs/rtx-3090-35b-a3b.md), [WSL2 tuning report](docs/rtx-3090-wsl.md), and
[native Windows guide](docs/rtx-3090-windows.md). A separate
[ordinary-inference analysis](docs/rtx-3090-normal-inference.md) covers MTP-disabled decode, and
[`dist/`](dist/README.md) explains the verified Windows and Linux release bundles.

This port adds and verifies:

- native Windows 11 support using MSVC, CUDA, and vcpkg, alongside the Linux/WSL2 build;
- Windows memory-mapped and asynchronous direct artifact loading, Winsock support, and correct
  FFmpeg/curl/zlib runtime packaging;
- RTX 3090-specific Q4 and Gated DeltaNet kernel dispatch selected with CUDA-event operator
  harnesses;
- a 35B-A3B text-only engine mode that omits the otherwise prohibitive vision workspace, plus a
  GA102-tuned Q6 K=2048 verification-head schedule;
- CUDA Graph ordinary decode and an RTX 3090-tuned MTP-3 speculative path;
- reproducible benchmarks, full cross-platform tests, release packaging, and real
  `Qwen3.6-27B` generation from the 16.29 GiB NInfer artifact.

NInfer is a from-scratch C++/CUDA inference engine for exact Qwen3.6 checkpoints on a single
NVIDIA GPU. The implementation is deliberately specialized rather than a general model runtime.

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | NInfer artifact | Size | SHA-256 |
|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `qwen3_6_35b_a3b.ninfer` | 22,373,184,256 bytes (20.84 GiB) | `9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163` |

## Performance

### RTX 3090 port compared with the original RTX 5090 results

The RTX 3090 results are listed first. They show how much of the original RTX 5090 throughput this
24 GiB GA102 port currently retains:

| Model and decode mode | RTX 3090 port | Original RTX 5090 | RTX 3090 share |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B, no MTP | **179.09 +/- 0.30 tok/s** | **271.1 tok/s** | **66.1%** |
| Qwen3.6-35B-A3B, fastest measured speculative path | **252.83 +/- 0.49 tok/s** (MTP-1) | **542.8-661.2 tok/s** (MTP-3) | **38.2-46.6%** |
| Qwen3.6-27B, no MTP | **38.04 tok/s** | **77.6 tok/s** | **49.0%** |
| Qwen3.6-27B, fastest controlled MTP-3 result | **66.70 tok/s** | **158.7-189.1 tok/s** | **35.3-42.0%** |

These percentages are orientation, not an apples-to-apples GPU benchmark. The 3090 tg128 and
controlled port runs use short fixed contexts, while the original 5090 figures below come from
long-prompt serving fixtures whose acceptance rate and workload vary. The exact configurations and
raw reports are linked below.

On native Windows, Qwen3.6-35B-A3B in explicit `--text-only` mode measured **179.09 +/- 0.30
tok/s** without MTP and **252.83 +/- 0.49 tok/s** with MTP-1 and the optimized proposal head.
These are five-run tg128 means after two warm-ups, using CUDA Graphs, INT8 KV, a 256-token engine
capacity, and the unmodified 20.84 GiB mixed Q4/Q5/Q6/W8 artifact. MTP-1 acceptance was 84.06%.
Text-only mode reduces stable workspace capacity from about 1.90 GiB to 15.05 MiB and rejects
image/video requests; it does not alter or requantize model weights. See the
[full reproduction report](docs/rtx-3090-35b-a3b.md).

On native Windows, the complete Qwen3.6-27B configuration—CUDA Graphs, BF16 KV, MTP-3, and the
optimized proposal head—measures approximately **60–64 decode tok/s** on the RTX 3090. The latest
quick run measured **59.96 ± 1.29 tok/s**, 60.74% draft acceptance, and 2.822 mean tokens per
speculative round; the longer controlled run measured **64.23 tok/s**. Ordinary MTP-disabled
decode measured **38.04 tok/s** in the controlled Windows run.

On the verified RTX 3090 / WSL2 system, Qwen3.6-27B with INT8 KV and MTP-3 measured **1,029.56
prefill tok/s** at 512 input tokens and **66.70 decode tok/s** over 128 output tokens. MTP-disabled
decode measured **35.28 tok/s**. The full draft-window sweep, kernel measurements, and memory
figures are in the [3090 report](docs/rtx-3090-wsl.md).

The figures below are upstream RTX 5090 reference results and are not measurements from this port.

Serving performance was measured on an RTX 5090 with INT8 group-64 KV cache, CUDA Graphs, a 1,024-
token prefill chunk, and a maximum context of 262,144 tokens. Each reported fixture uses five fixed
seeds after one warm-up. The two registered targets are reported independently and are not
cross-target comparisons.

**Qwen3.6-35B-A3B**

- MTP0 at a 7,680-token prompt: **15,544.3 prefill tok/s** and **271.1 decode tok/s**.
- MTP0 at a 260,096-token prompt: **5,157.1 prefill tok/s** and **188.2 decode tok/s**.
- MTP3 long reasoning: **542.8–634.3 decode tok/s** with **73.0–82.7% acceptance**.
- MTP3 structured output: **661.2 decode tok/s**, **87.2% acceptance**, and **3.62 tokens/round**.

**Qwen3.6-27B**

- MTP0 at a 7,680-token prompt: **3,218.1 prefill tok/s** and **77.6 decode tok/s**.
- MTP0 at a 260,096-token prompt: **1,614.8 prefill tok/s** and **54.8 decode tok/s**.
- MTP3 long reasoning: **158.7–174.2 decode tok/s** with **73.3–79.9% acceptance**.
- MTP3 structured output: **189.1 decode tok/s**, **88.9% acceptance**, and **3.67 tokens/round**.

See [Performance](docs/performance.md) for the full methodology, variability, reproduction command,
and per-fixture results.

## Evaluation

Capability scores from the published model cards, measured through NInfer's OpenAI-compatible
serving route with thinking enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample
per problem):

| Model | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| [Qwen3.6-27B](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% |
| [Qwen3.6-35B-A3B](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% |

These are single-sample results under that NInfer evaluation profile, not pass@k. See each model
card for correct/total counts and the full evaluation notes.

## Requirements

This port currently requires either:

- 64-bit Linux/WSL2 with GCC 13 and CUDA 13.0 or newer; or
- 64-bit Windows with Visual Studio 2022, CUDA 13.0 or newer, CMake 3.28 or newer, and vcpkg;
- NVIDIA GeForce RTX 3090 (`sm_86`);
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config` on Linux;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja for the Linux commands below, or the Visual Studio 2022 generator on Windows.

The build rejects CUDA architectures other than `86`. Verified binary bundles can be produced from
the native Windows and WSL build trees with `scripts/package-release.ps1`; see
[`dist/README.md`](dist/README.md). The model artifact is intentionally distributed separately.

The checked-in `vcpkg.json` installs the Windows FFmpeg, zlib, curl, and pkgconf dependencies. See
the [native Windows guide](docs/rtx-3090-windows.md) for the exact configure, test, and benchmark
commands.

## Build from source

```bash
git clone https://github.com/Don-Chad/ninfer-3090.git
cd ninfer-3090

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Download a model

Use the Hugging Face CLI to download either registered artifact:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

The `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

## Run the CLI

For the fastest verified RTX 3090 configuration:

```powershell
build-windows\apps\Release\ninfer.exe models\qwen3_6_35b_a3b.ninfer `
  --prompt "Explain speculative decoding in three sentences." `
  --max-context 256 --prefill-chunk 128 --max-new 128 --kv-dtype int8 `
  --mtp-draft-tokens 1 --lm-head-draft --text-only
```

`--text-only` is required for the 35B-A3B artifact on a 24 GiB RTX 3090. It rejects image/video
input and avoids reserving vision scratch memory; normal text model behavior is unchanged.

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
MTP statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --model-id qwen3.6-27b \
  --max-context 16384 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements Anthropic Messages, streaming, token counting, multimodal input, and
function-tool request/response translation. See [HTTP serving](docs/serving.md).

## Capabilities

Both registered artifacts support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- greedy, temperature, top-k, top-p, min-p, and presence/frequency-penalty sampling;
- compatible-prefix reuse;
- OpenAI Chat Completions and Anthropic Messages, including streaming and usage accounting;
- prompt-rendered function tools and parsed tool calls.

## Current limits

- Qwen3.6-35B-A3B requires `--text-only` on a 24 GiB RTX 3090; multimodal mode does not fit with
  the full artifact resident.
- Execution is specialized for one RTX 3090 and one CUDA device.
- One Engine owns one resident sequence and runs one active request at a time.
- Continuous batching, multi-GPU execution, CPU/GPU offload, and distributed serving are not
  implemented.
- Context capacity is configurable up to the registered models' native 262,144-token limit, subject
  to GPU memory and KV-cache configuration.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [RTX 3090 / WSL2 port](docs/rtx-3090-wsl.md)
- [RTX 3090 ordinary inference](docs/rtx-3090-normal-inference.md)
- [Windows and Linux release bundles](dist/README.md)
- [CLI examples](examples/cli/)

## Contributing

Pull requests are welcome. This is a public repository and its
[Pull requests page](https://github.com/Don-Chad/ninfer-3090/pulls) is open to contributions.
Please describe the target platform, CUDA version, validation performed, and any measured
performance impact. Keep RTX 3090 kernel changes backed by numerical tests and reproducible
benchmarks.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), which are also distributed
under Apache-2.0. Vendored dependencies retain their own license files under `third_party/`.
