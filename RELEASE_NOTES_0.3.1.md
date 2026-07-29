# NInfer RTX 3090 v0.3.1 — 35B-class inference at 300–400+ tok/s

NInfer v0.3.1 turns one RTX 3090 into a highly specialized local inference engine for
Qwen3.6-35B-A3B. The entire 20.84 GiB quantized model, INT8 KV cache, CUDA Graphs, and speculative
runtime fit on the 24 GB card.

This release is built for useful local work rather than a single synthetic headline:

| Workload | Result |
|---|---:|
| Repetition-rich Python repository edit, best clean run | **406.44 ± 0.79 output tok/s** |
| Same 1,500-input/1,500-output fixture, release rerun | **399.16 ± 2.33 output tok/s** |
| Realistic two-copy code-refactor prompt, 1,500 input + 1,500 output | **310.04 ± 0.95 output tok/s** |
| Canonical `tg128`, MTP-2 | **262.05 ± 1.62 output tok/s** |
| Canonical `tg128`, no speculation | **187.24 ± 0.27 output tok/s** |

The 310 tok/s result matters: its prompt contains only two copies of the code rather than the
heavy repetition of the 400 tok/s showcase. Dynamic mode correctly keeps that workload on MTP-3.
The 399–406 tok/s result demonstrates the additional headroom available for repository edits,
templated code, regeneration, and other workloads with reusable local continuations.

## What we added

- **Adaptive hybrid speculation.** A one-time rolling-hash scan classifies long inputs in under
  0.1 ms. Ordinary requests remain on MTP-3; sufficiently repetitive requests activate target-only
  prompt lookup.
- **Adaptive K15/K8/K5 lookup windows.** The runtime verifies the longest useful continuation,
  then falls back to smaller captured windows when the available match is shorter.
- **Lazy MTP realignment.** After lookup misses, MTP state is restored only when needed, reducing
  ordinary fallback work.
- **GA102 kernel work.** RTX 3090 schedules cover MoE decode, Gated DeltaNet projection and
  recurrent-state work, output/argmax operations, and graph overhead.
- **Cross-platform graph memory planning.** Windows and CUDA 13.0 Linux graph-capture footprints
  are both validated within the 24 GB card.
- **Native release packages.** Windows 11 and Ubuntu 24.04/WSL2 builds include the CLI,
  OpenAI-/Anthropic-compatible server, benchmark runner, runtime dependencies, and checksums.

## Recommended flags

For general 35B-A3B text inference on a 24 GB RTX 3090, start with adaptive hybrid mode:

```powershell
.\ninfer.exe models\qwen3_6_35b_a3b.ninfer `
  --prompt "Refactor this Python module and return the complete implementation." `
  --max-context 4096 --prefill-chunk 128 --max-new 1500 `
  --kv-dtype int8 --text-only `
  --mtp-draft-tokens 3 --lm-head-draft `
  --prompt-lookup-tokens 15 --prompt-lookup-min-match 4 `
  --prompt-lookup-auto --prompt-lookup-min-context 1000
```

The same flags work with `ninfer-serve`:

```powershell
.\ninfer-serve.exe models\qwen3_6_35b_a3b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 4096 --prefill-chunk 128 `
  --kv-dtype int8 --text-only `
  --mtp-draft-tokens 3 --lm-head-draft `
  --prompt-lookup-tokens 15 --prompt-lookup-min-match 4 `
  --prompt-lookup-auto --prompt-lookup-min-context 1000
```

What the flags do:

- `--mtp-draft-tokens 3` enables the model's trained MTP proposal layer with three draft positions.
- `--lm-head-draft` selects the optimized proposal head required by the fast MTP path.
- `--prompt-lookup-tokens 15` makes up to 15 tokens from a matching local continuation available
  for target-model verification. The adaptive runtime can use captured K15, K8, or K5 windows.
- `--prompt-lookup-min-match 4` requires at least a four-token repeated suffix before lookup can
  propose a continuation.
- `--prompt-lookup-auto` keeps prompt lookup off for inputs that do not contain enough reusable
  structure. This is the recommended general-purpose setting.
- `--prompt-lookup-min-context 1000` delays automatic classification until the input is long enough
  for repetition coverage to be meaningful.
- `--kv-dtype int8` is the supported memory-efficient KV configuration for 35B-A3B on a 24 GB card.
- `--text-only` avoids the vision-workspace reservation and is required for the 35B-A3B release
  configuration on an RTX 3090.

For ordinary prose or short prompts, MTP-2 is the conservative controlled-performance mode:

```text
--kv-dtype int8 --text-only --mtp-draft-tokens 2 --lm-head-draft
```

Omitting `--prompt-lookup-auto` forces lookup attempts and should be reserved for workloads known
to contain substantial repetition. K15 is not free: the target model verifies 16 positions, so
forced lookup can be slower on non-repetitive text.

## Model scope

The published RTX 3090 release, its packages, tuning, memory limits, and headline measurements are
built and validated around **Qwen3.6-35B-A3B**. The 35B-A3B `.ninfer` artifact is downloaded
separately and is the recommended model for these binaries.

The open-source tree also contains the complete **Qwen3.6-27B** target, converter, reference
implementation, tests, and build integration. Developers can build the project from source and run
the 27B `.ninfer` artifact with the same CLI or server. The 27B path has less of the new
35B-A3B-specific GA102 acceleration and is not covered by the 300–400+ tok/s release claims.

## Latency

Warm resident time to first token, including prompt execution through the first generated token:

| Input | TTFT |
|---:|---:|
| 128 tokens | **62.16 ms** |
| 512 tokens | **242.25 ms** |
| 1,500 tokens | **737.41 ms** |

## How the numbers were measured

All throughput figures are output tokens divided by decode time, measured on one RTX 3090 using the
same Qwen3.6-35B-A3B artifact, CUDA Graphs, INT8 group-64 KV, text-only mode, five measured
repetitions after warm-up, and no competing GPU workload. Model loading and graph construction are
excluded. The long code tests generate a fixed 1,500 output tokens; `tg128` is a shorter controlled
kernel/scheduling comparison.

The 400 tok/s workload is deliberately labeled repetition-rich and is not a universal model-speed
claim. The 310 tok/s two-copy code workload and canonical figures provide more conservative points
of comparison.

Download the 20.84 GiB model separately from
[Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer). See the repository README
for exact commands, inference modes, server usage, memory requirements, and full caveats.

This RTX 3090 edition began as a port of Neroued's RTX 5090 NInfer implementation and has since
grown its own GA102 kernels, memory profile, Windows/Ubuntu release workflow, and adaptive
speculative runtime.
