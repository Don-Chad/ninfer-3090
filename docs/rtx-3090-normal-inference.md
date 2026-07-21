# RTX 3090 ordinary inference

This report isolates Qwen3.6-27B ordinary autoregressive decode with MTP disabled. It does not
count speculative tokens or use the draft head.

## Recommended configuration

CUDA Graph decode is the important runtime optimization and is enabled by default. BF16 KV was
slightly faster at short context on the tested card:

```powershell
build-windows\apps\Release\ninfer.exe models\qwen3_6_27b.ninfer `
  --prompt 'Explain memory bandwidth in three sentences.' `
  --max-context 2048 --max-new 128 --kv-dtype bf16 `
  --mtp-draft-tokens 0
```

Use INT8 KV when its lower memory footprint is more valuable, especially for long context. Do not
pass `--no-cuda-graph` for normal serving.

## Controlled Windows results

Each row used tg128, five measured repetitions after one warm-up, a 2,048-token capacity, and MTP
disabled.

| KV cache | Decode path | Decode | Standard deviation |
|---|---|---:|---:|
| BF16 | CUDA Graph | **38.04 tok/s** | 0.40 tok/s |
| INT8-G64 | CUDA Graph | 37.62 tok/s | 0.08 tok/s |
| BF16 | eager | 35.66 tok/s | 0.23 tok/s |
| INT8-G64 | eager | 35.63 tok/s | 0.11 tok/s |

CUDA Graphs improved BF16 ordinary decode by 6.7%. The equivalent WSL CUDA 12.8 BF16 Graph run
measured 36.89 +/- 1.61 tok/s.

## Operator evidence

The cold-cache harness measured a roughly 839-842 GB/s copy ceiling. Important one-token kernels
on native Windows measured:

| Operation | Median | Useful bandwidth |
|---|---:|---:|
| fused Q4 MLP gate/up + SwiGLU | 119.81 us | 790.80 GB/s |
| fused Q5 MLP down + residual | 87.04 us | 672.64 GB/s |
| Q6 full language-model head | 1,473.54 us | 674.42 GB/s |
| GDN Q4/Q5 input projection | 65.54 us | n/a |
| attention Q4/Q5 input projection | 88.06 us | n/a |
| recurrent Gated Delta Rule | 9.17 us | 689.75 GB/s |

The fused Q4 MLP path is already close to the measured memory ceiling. The Q5 down path and Q6
head have local headroom, but their theoretical savings are only a few milliseconds per token.
Four tested Q5 alternatives were rejected: 8-warp with staged activation, 8-warp direct activation,
16-warp triple buffering, and 32-warp double buffering. A direct-plane prototype was also rejected.
All were slower than the retained 16-warp double-buffered kernel; no experimental route remains in
the production source.

The next material step would be a larger architectural pass that fuses adjacent normalization,
projection, convolution/gating, and residual work across the 64 layers. That is a higher-risk
change to numerical boundaries and CUDA Graph topology, not an unclaimed dispatch switch. The
current port therefore keeps the verified kernels and documents the measured ceiling instead of
publishing a synthetic speedup.

Nsight Compute 2026.1 was present, but Windows performance counters were disabled by the driver
policy (`ERR_NVGPUCTRPERM`). The measurements above use CUDA-event operator harnesses and the full
Engine benchmark.
