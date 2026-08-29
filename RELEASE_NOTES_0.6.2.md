# NInfer-3090 v0.6.2

v0.6.2 brings the qualified upstream semantic merge to the RTX 3090 release line. It preserves
SM86-safe execution while adding the current upstream scheduling, context-cache, serving, and
artifact improvements.

## Benefits

- **Fast local Qwen3.8-27B:** the Windows RTX 3090 benchmark measured 79.88 decode tok/s at C1,
  scaling to 184.86 aggregate tok/s at C8 with MTP3.
- **More capable serving:** updated resource scheduling, context-cache ownership, typed instruction
  turns, tool/API improvements, and vision-pipeline work from upstream.
- **RTX 3090-safe defaults:** Blackwell-only FP8/NVFP4 execution is fenced on SM86 rather than
  silently taking an unsupported kernel path.
- **Native binaries for Windows and Linux:** both packages include the CLI, server, download helper,
  safe C1/C8 launchers, checksums, and platform-specific guide.

## Validation

- Windows: Visual Studio 2022, CUDA 13.2, SM86 Release build.
- Windows CTest: 94/94 tests passed; five optional real-artifact checks reported their defined
  skip status because their artifacts were not present locally.
- Real Qwen3.8-27B Windows inference: three MTP-off and three MTP3 1,024-token runs completed.
  MTP3 averaged 64.21 tok/s in the CLI end-to-end profile, with 53.40% draft acceptance and
  5.50 GiB free after startup.
- Dedicated Windows server benchmark: C1/C2/C4/C8 generation and prefill campaign completed with
  no competing GPU workload.

## Contributors

- [iamwavecut](https://github.com/iamwavecut) - semantic upstream merge and resource/context work,
  [PR #14](https://github.com/Don-Chad/ninfer-3090/pull/14).
- [airtonix](https://github.com/airtonix) - Linux and Docker release support.
- [nasedkinpv](https://github.com/nasedkinpv) - tool-call parser fix.
- [wmehanna](https://github.com/wmehanna) - prefix-stable system turns.

## Notes

- The C1-C8 server values are server-compute metrics; they are not interchangeable with CLI
  end-to-end measurements.
- `rk8v4` remains opt-in and lossy. INT8 is the default quality-oriented KV mode.
