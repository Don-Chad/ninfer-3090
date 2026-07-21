# RTX 3090 / WSL2 port

This report records the CUDA 12.8 measurements for the `sm_86` port on one GeForce RTX 3090. The
current source accepts CUDA 12.8 or newer. The verified product
targets are Qwen3.6-27B and text-only Qwen3.6-35B-A3B. The 35B artifact requires `--text-only` to
omit vision workspace on a 24 GiB card. A quick WSL tg32 MTP-1 check measured 248.09 +/- 2.32
tok/s; native Windows remains the primary controlled result. See
[the full 35B-A3B report](rtx-3090-35b-a3b.md).

## Verified environment

- Windows 11 with WSL2 and Ubuntu 24.04
- GeForce RTX 3090 (24 GB, compute capability 8.6)
- NVIDIA Windows driver exposing CUDA 13.2 to WSL
- CUDA Toolkit 12.8 in WSL
- CMake 3.28, Ninja, GCC 13, and FFmpeg 6 development libraries

Check the GPU and compiler before configuring:

```bash
nvidia-smi
nvcc --version
cmake --version
```

The build intentionally rejects architectures other than `86` and CUDA toolkits older than 12.8.

## Build in WSL

From this checkout on the Windows `G:` drive:

```bash
cd /mnt/g/python/custom-kernel-3090/ninfer

cmake -S . -B build-sm86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON \
  -DNINFER_BUILD_BENCHMARKS=ON

cmake --build build-sm86 --parallel "$(nproc)"
```

The tested artifact can be downloaded with the current Hugging Face CLI:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models \
  --max-workers 1

sha256sum models/qwen3_6_27b.ninfer
```

Expected size: 16.29 GiB. Expected SHA-256:
`74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127`.

## Recommended 3090 command

MTP-3 is the measured optimum. INT8 group-64 KV and a 2,048-token capacity keep this qualification
run comfortably inside 24 GB:

```bash
./build-sm86/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 2048 \
  --max-new 128 \
  --kv-dtype int8 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

## Reproduce the benchmark

```bash
./build-sm86/bench/ninfer_bench \
  --weights models/qwen3_6_27b.ninfer \
  -p 512 -n 128 -r 3 --warmup 1 \
  --max-ctx 2048 --prefill-chunk 512 --kv-dtype int8 \
  --mtp-draft-tokens 3 --lm-head-draft \
  -o json --output-file profiles/bench/rtx3090_sm86_optimized_mtp3.json
```

Measured on the verified machine:

| Configuration | Prefill | Decode | Draft acceptance |
|---|---:|---:|---:|
| MTP disabled | 950.44 tok/s | 35.28 tok/s | n/a |
| MTP-3, initial sm86 port | 988.31 tok/s | 61.98 tok/s | 65.1% |
| MTP-3, GA102-tuned | 1,029.56 tok/s | 66.70 tok/s | 65.1% |

The optimized MTP-3 run used 16.28 GiB for weights, 0.79 GiB for sequence capacity, 0.07 GiB
for KV payload, and reserved 1.90 GiB of reusable workspace capacity. Peak measured workspace in
decode was 1.16 MiB.

The complete optimized draft-window sweep was:

| Draft window | Decode | Acceptance | Mean tokens per round |
|---:|---:|---:|---:|
| 1 | 52.18 tok/s | 85.5% | 1.855 |
| 2 | 59.22 tok/s | 69.8% | 2.396 |
| 3 | **66.70 tok/s** | 65.1% | 2.953 |
| 4 | 48.66 tok/s | 54.5% | 3.179 |
| 5 | 48.74 tok/s | 50.9% | 3.543 |

K=4 and K=5 are slower because verification reaches the five-token low-bit route boundary. Do not
increase the draft window on this 3090 configuration merely to increase accepted tokens per round.

## GA102-specific kernel decisions

- The Q4 34816x5120 gate/up T=2..4 route uses an explicit four-row CTA schedule. At T=4 it
  measured 200.70 microseconds versus 208.90 microseconds for the original eight-row CTA. A
  sixteen-row CTA measured 261.12 microseconds.
- The 27B BF16 GDN T=4 route remains `small-split10`: 12.29 microseconds versus 17.41 for split-8,
  28.67 for split-4, 51.20 for split-2, and 97.28 for unsplit.
- At T=512, GDN split-8 remains best: 26.62 microseconds versus 30.72 for split-4, 53.25 for
  split-2, and 96.26 for unsplit.
- Cooperative launch residency is bounded using the CUDA 12.8 `sm_86` register counts and the RTX
  3090's 82 SMs. This avoids the cooperative-launch-too-large failure seen with the upstream
  occupancy assumptions.

Nsight Systems under this WSL/driver combination captured CUDA API calls but did not emit GPU
kernel activity records, even with eager decode and graph-node tracing. The tuning decisions above
therefore use the in-tree CUDA-event cold-cache operator harnesses and end-to-end Engine benchmark.

## Validation

The cross-language packer golden test needs a local CPU-only Python environment:

```bash
uv venv .venv-tests
uv pip install --python .venv-tests/bin/python torch numpy safetensors \
  --index-url https://download.pytorch.org/whl/cpu
cmake -S . -B build-sm86 -DPython3_EXECUTABLE="$PWD/.venv-tests/bin/python"
cmake --build build-sm86 --parallel "$(nproc)"
```

Run the focused kernel checks:

```bash
./build-sm86/tests/ninfer_q4_linear_plan_test
./build-sm86/tests/ninfer_q4_linear_dispatch_test
./build-sm86/tests/ninfer_q4_linear_candidate_test
./build-sm86/tests/ninfer_linear_swiglu_plan_test
./build-sm86/tests/ninfer_linear_test
./build-sm86/tests/ninfer_gdn_gating_proj_plan_test
./build-sm86/tests/ninfer_gdn_gating_proj_test
```

The real-artifact integration test is opt-in:

```bash
NINFER_QWEN3_6_27B_WEIGHTS="$PWD/models/qwen3_6_27b.ninfer" \
  ./build-sm86/tests/ninfer_qwen3_6_27b_prefix_real_test
```
