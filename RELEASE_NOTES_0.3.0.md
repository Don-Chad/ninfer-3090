# NInfer RTX 3090 v0.3.0

The first standalone RTX 3090 release centered on Qwen3.6-35B-A3B.

## Highlights

- Native Windows 11 and Ubuntu 24.04 / WSL2 `sm_86` binaries.
- Adaptive hybrid decoding: MTP-3 for ordinary requests, with K15/K8/K5 prompt lookup selected
  automatically for sufficiently repetitive long inputs.
- Best clean repeated-code result: **406.44 ± 0.79 output tok/s**.
- Final release rerun on the same 1,500-input/1,500-output fixture:
  **399.16 ± 2.33 output tok/s**.
- Canonical `tg128`: **262.05 ± 1.62 tok/s** with MTP-2,
  **260.55 ± 0.92 tok/s** with MTP-3, and **187.24 ± 0.27 tok/s** without speculation.
- Warm resident TTFT: **62.16 ms** at 128 input tokens, **242.25 ms** at 512, and
  **737.41 ms** at 1,500.

Throughput numbers are output tokens per decode second, measured over five repetitions after
warm-up on one RTX 3090 with CUDA Graphs, INT8 group-64 KV, text-only mode, and no competing GPU
work. The 406/399 tok/s results use a deliberately repetitive Python repository-edit fixture and
are not presented as universal throughput. Model loading and graph construction are excluded from
TTFT; TTFT includes prompt execution through the first generated token.

The 20.84 GiB Qwen3.6-35B-A3B `.ninfer` artifact is downloaded separately from
[Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer).

See the repository README for mode selection, exact reproduction commands, server usage, memory
requirements, and complete benchmark caveats.
