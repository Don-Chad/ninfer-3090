# NInfer v0.6.0 RTX 4090 early1

This is an early compatibility build for RTX 4090 (`sm_89`) users. It carries the latest
Qwen3.8-27B ReplaySSM/MTP release path without claiming Ada-specific optimization.

## Included

- Native `sm_89` Windows server, CLI, and benchmark binaries.
- Qwen3.8-27B artifact support, ReplaySSM, MTP speculative decoding, INT8 KV, bounded concurrency,
  and OpenAI-compatible serving from the current v0.6 development line.
- Separate SM89 1,024-output cohort benchmark harness for C1/C2/C4/C8.

## Qualification boundary

Compilation, linking, native cubin inspection, and on-device RTX 4090 inference passed. The recorded
1,024-token cohort sweep completed C1/C2/C4/C8 at 102.13, 162.46, 193.49, and 299.82 aggregate
tok/s respectively, with peak VRAM from 18,250 MiB to 20,708 MiB. See `README.md` in the archive
for the full table and reproduction command.

This remains an early compatibility release. It uses the proven pre-Hopper schedules and has not
yet received RTX 4090-specific performance tuning.
