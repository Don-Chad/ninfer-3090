# NInfer early RTX 4090 support

Run Qwen3.8-27B locally on a 24 GB RTX 4090 with the same ReplaySSM and MTP path used by the
RTX 3090 release. This early build targets Ada `sm_89` natively. It prioritizes first-run
compatibility; RTX 4090-specific tuning comes after runtime qualification.

## Status

- Native Windows `sm_89` server, CLI, and benchmark binaries compile and link.
- The binaries contain native `sm_89` cubins rather than relying on PTX fallback.
- Blackwell-only NVFP4/TMA kernels remain disabled. The proven pre-Hopper compatibility schedules
  are retained for this first build.
- On-device RTX 4090 inference and performance are not yet qualified because the RunPod credential
  available during preparation was rejected with HTTP 401.

Do not publish performance claims from this package until the full cohort sweep below passes on an
otherwise idle RTX 4090.

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

## Required RTX 4090 qualification

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
