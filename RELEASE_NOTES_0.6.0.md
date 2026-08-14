# NInfer-3090 v0.6.0

This release brings ReplaySSM, C8/MTP3, reasoning-effort control, and the official Qwen3.8-27B
groupwise NInfer artifact to a single 24 GB RTX 3090.

## Highlights

- registers `qwen3.8-27b/groupwise-int` as target `qwen3_8_27b`;
- binds its W8 token embedding and output head while reusing the qualified 27B execution package;
- validates text generation, MTP2/MTP3, paged INT8 KV, CUDA Graphs, and concurrent serving;
- uses ReplaySSM to reduce speculative recurrent-state memory;
- validates C1-C4 at 4K and C8 at 8K with MTP3;
- supports Qwen3.8 `low`, `medium`, and `xhigh` reasoning effort;
- includes reproducible Qwen3.8 concurrency and MTP sweep tools.

## Recommended C8 command

```powershell
.\ninfer-serve.exe models\qwen3_8_27b.ninfer `
  --max-context 8192 --kv-capacity 8192 --max-concurrency 8 `
  --prefill-chunk 1024 --kv-dtype int8 `
  --spec mtp --draft-tokens 3 --lm-head-draft
```

C8/8K MTP3 measured 114.73 aggregate end-to-end tok/s with an 8,192-token shared KV pool and
21,818 MiB peak VRAM. A 65,536-token pool also fits, measuring 114.88 tok/s and 23,745 MiB peak.

The official artifact SHA-256 used for validation is
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.

## Limitations

- Qwen3.8 validation in this release is text-only.
- Greedy outputs from ordinary, MTP2, and MTP3 execution were coherent but not byte-identical on
  every benchmark prompt; the performance sweep is not a claim of exact quality parity.
- Paged KV supports BF16 and INT8. RotorQuant/KV4 is not exposed by this runtime.
- NVFP4 and TMA remain unavailable on SM86.
