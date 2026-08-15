# NInfer early RTX 4090 support

Run Qwen3.8-27B locally on a 24 GB RTX 4090 with the same ReplaySSM and MTP path used by the
RTX 3090 release. This early build targets Ada `sm_89` natively. It prioritizes first-run
compatibility; RTX 4090-specific tuning comes after runtime qualification.

## Status

- Native Windows `sm_89` server, CLI, and benchmark binaries compile and link.
- The binaries contain native `sm_89` cubins rather than relying on PTX fallback.
- Blackwell-only NVFP4/TMA kernels remain disabled. The proven pre-Hopper compatibility schedules
  are retained for this first build.
- On-device generation passed on an RTX 4090 using CUDA 12.8. Every request in the C1/C2/C4/C8
  sweep completed exactly 1,024 output tokens.
- This is compatibility-qualified, not Ada-optimized. Treat the numbers below as an early baseline.

| Cohort | Total output | End-to-end | Decode | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 | 102.13 tok/s | 103.35 tok/s | 45.31% | 112 ms | 18,250 MiB |
| C2 | 2,048 | 162.46 tok/s | 165.74 tok/s | 53.13% | 160 ms | 18,562 MiB |
| C4 | 4,096 | 193.49 tok/s | 198.76 tok/s | 56.91% | 295 ms | 19,184 MiB |
| C8 | 8,192 | 299.82 tok/s | 315.09 tok/s | 53.16% | 644 ms | 20,708 MiB |

Test setup: Vast.ai RTX 4090, CUDA 12.8.93, Qwen3.8-27B INT8 artifact, INT8 KV,
ReplaySSM/MTP3 with LM-head draft, greedy decoding, and prefix reuse disabled. C1-C4 used an 8K
KV pool; C8 used 16K. Results include complete request-wave time and were collected on 2026-08-15.

## Build

```powershell
cmake -S . -B build-sm89 -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=G:/python/custom-kernel-3090/.vcpkg-local/scripts/buildsystems/vcpkg.cmake `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DNINFER_BUILD_APPS=ON `
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-sm89 --config Release --parallel $env:NUMBER_OF_PROCESSORS
```

CUDA 12.8 or newer is required. Keep `CMAKE_CUDA_ARCHITECTURES=89`; a combined 86/89 fat binary is
not part of this early package.

## Reproduce the RTX 4090 qualification

Use `tools/bench/run_qwen38_replayssm_sm89_cohort_sweep.py`. It launches separate C1, C2, C4, and C8
server processes, sends 1,024 output tokens per request, polls peak GPU memory, and stores every
command, response, server log, request log, and summary under a dedicated SM89 result directory.

Run it on Linux with:

```bash
NINFER_MODEL=/workspace/qwen3_8_27b.ninfer \
  uv run tools/bench/run_qwen38_replayssm_sm89_cohort_sweep.py
```

The result table must report total output, end-to-end tok/s, decode tok/s, MTP acceptance, mean
TTFT, and peak VRAM for C1/C2/C4/C8. Each request must complete exactly 1,024 output tokens.
