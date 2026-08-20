# Vision VRAM Overlay — Measurement Ledger

All numbers measured on AI Farm unless noted. GPU2 = RTX 3090 24576 MiB (prod ninfer-3090),
GPU1 = RTX 4060 Ti 16 GiB (build/test), GPU0 = RTX 3090 (other services).

## Baseline (2026-08-20, prod container, commit 5a22c86a config)

Startup line:

```
KV capacity explicit resolved=120000 tokens pages=1875/1875 runtime=5.44 GiB
free-after-weights=5.47 GiB free-after-startup=111.38 MiB headroom=0.00 MiB
slack=35.09 MiB graphs=8.00 MiB/86.00 MiB
weights load: 16.95 GiB in 5.749 s (disk → device)
```

- KV bytes/token = 5.44 GiB / 120000 = **47.5 KiB/token** (pages 1875 → 64 tokens/page, 2.97 MiB/page)
- Host RAM: 62 GiB total, ~25 GiB available → pinned budget OK
- Single-stream decode ≈ 23.5 tok/s greedy short-ctx (MTP up to ~51–61 tok/s sampled);
  long prefill ≈ 819 tok/s (chunk 1024)

## Artifact group sizes (exact, from tools.artifact.inspect on the prod artifact)

| Group | Bytes | Notes |
|---|---|---|
| `vision/*` total | 282.0 MiB (333 objects) | host-pinned in overlay mode |
| vision layer (each of 27) | 8.53 MiB | ping-pong slot size |
| vision non-layer (patch+pos+merger) | 51.83 MiB | prelude/merger staging |
| `text/token_embedding` | 1288.3 MiB | evict rank 2 |
| `text/output_head` (lm_head) | 1288.3 MiB | evict rank 1 (arena end) |
| `text/draft_head` | 340.5 MiB | evict rank 3 |
| `mtp/*` | 430.4 MiB | evict rank 4 |
| text decoder block (each of ~62) | ~217 MiB | evict rank 5+ (tail blocks) |

- Evictable ladder capacity (lm_head+embed+draft+mtp) = **3.35 GiB** before touching text blocks
- Streamed window staging = 2×8.53 + 51.83 ≈ **69 MiB** → typical evict ≈ 80 MiB (5 × 16 MiB chunks)
- Overlay reclaim target = 282 MiB resident vision weights ≈ **+5.9k KV tokens** at 47.5 KiB/token

## VMM spike (Task 3, GPU1 sm_89, chunk = 64 MiB, granularity = 2 MiB)

```
test: ninfer_vmm_graph_remap_test PASSED (0.74 s)
evict (2 chunks unmap+map+setaccess)  = 595 µs
restore (2 chunks unmap+map+setaccess) = 240 µs
captured graph launched correctly after remap-home + mirror re-upload
```

Per-map cost ≈ 150–300 µs → even a 3.35 GiB full-ladder evict ≈ 25–50 ms of driver calls.

## Design facts confirmed in source

- Vision workspace is served from the SHARED text workspace envelope at request-plan time
  (`request_plan_impl.h:139`) — there is no separate resident vision workspace to reclaim;
  the resident cost of `--vision` is the 282 MiB of weights.
- Vision encode output lives in the per-request transient region (startup-frozen capacity);
  unchanged by the overlay.
- KV `Automatic` capacity mode derives from free-after-weights bytes (`registry.cpp`), so
  overlay mode raises automatic KV capacity with no config change.
- The engine has ONE GPU worker thread and serial GPU units (concurrent-inference-architecture
  §2.6/§7): the overlay window runs inside the existing prefill-begin unit; no new locking is
  required for any `--max-concurrency` — waiting requests keep their KV by construction.

## Implementation-phase measurements (2026-08-20)

- `ninfer_vmm_graph_remap_test` (GPU1, 64 MiB chunks): evict maps 502 µs, restore maps 206 µs.
- `ninfer_evictable_weight_pool_test` (GPU1, 16 MiB chunks, 2-chunk transaction over a 96 MiB
  arena): evict 0.94 ms, restore (remap + 24 MiB mirror H2D + sync) 12.8 ms.
- Full farm test suite on GPU1 (sm_89 build): 77/85 pass; the 8 failures are pre-existing
  `NVFP4 A4 execution requires an sm_120a GPU` capability rejections (nvfp4/gdn-nvfp4 suites),
  unrelated to this branch.
- Latent upstream bug found and fixed en route: materializer direct-I/O staging slots relied on
  `cudaMallocHost` returning 4096-aligned pointers; any small prior pinned allocation (e.g. the
  new pinned weight block) broke direct reads. Slots now align explicitly.
