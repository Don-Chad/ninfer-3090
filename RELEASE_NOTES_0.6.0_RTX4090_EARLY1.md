# NInfer v0.6.0 RTX 4090 early1

This is an early compatibility build for RTX 4090 (`sm_89`) users. It carries the latest
Qwen3.8-27B ReplaySSM/MTP release path without claiming Ada-specific optimization.

## Included

- Native `sm_89` Windows server, CLI, and benchmark binaries.
- Qwen3.8-27B artifact support, ReplaySSM, MTP speculative decoding, INT8 KV, bounded concurrency,
  and OpenAI-compatible serving from the current v0.6 development line.
- Separate SM89 1,024-output cohort benchmark harness for C1/C2/C4/C8.

## Qualification boundary

Compilation, linking, native cubin inspection, and CLI startup passed. On-device RTX 4090 inference
and benchmark qualification are pending because the available RunPod API key returned HTTP 401.
Treat this archive as a test candidate until the recorded cohort sweep passes.

