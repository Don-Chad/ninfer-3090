# Multi-GPU: layer pipeline stages

`--devices A,B,...` splits the model's layers into one pipeline stage per GPU. Each stage owns its
layers whole: weights, the KV cache of its attention layers, the recurrent state of its GDN layers,
and the scratch it runs in. The point is memory: a model that does not fit one card, or a context
that does not, spreads across several with every card's memory usable for KV.

This replaces the earlier expert-offload split, which moved only each layer's MLP to the second card
and kept everything stateful on the first. That design capped KV room at whatever the first card had
left; whole-layer stages have no such asymmetry, and they cross a device boundary once per stage
rather than twice per layer.

## What it is and is not

**It is a memory feature.** The stages run in sequence: while one computes, the others wait.
Decode is weight-bandwidth-bound and each stage reads only its own weights, so a single stream is
about as fast as one GPU, minus the boundary hops. Splitting the batch into micro-batches would make
each stage read its weights once per micro-batch and win nothing.

**It is not tensor parallelism.** No layer is divided across devices, so no collective runs inside a
layer.

**Multi-GPU is a Linux feature.** Repeating one device id (`--devices 0,0`) exercises the whole
stage path on a single card and is accepted everywhere, which is how the path is tested without a
second GPU. Distinct ids are refused on Windows.

## How it works

- **`StagePlan`** (`core/stage_plan.h`) says which contiguous layers each stage owns. `--stage-layers
  30,34` sets the counts. Without it `default_stage_layers` (`models/qwen3_5/load.cpp`) sizes each
  layer from the artifact and calls `solve_stage_plan`, an exact memory-balancing solver (maximum
  page groups every stage can hold, then least-full stage), verified against a brute-force
  partition oracle. It works from the bytes free on each device, the stored KV page-group and
  per-slot GDN state bytes of each layer (from the engine's KV storage format and slot count), and
  rough constants for what a stage needs beyond weights (context, workspace, graphs; more on rank
  0 for the head), so it is a good default rather than exact; if it finds nothing fits
  it deals the layers evenly and lets the Program's planning report the device that runs out.
- **Placement.** `bind_text` places every layer's weights on its stage's device
  (`Bindings::place`, recorded for rank 0 as well so an object shared between a rank-0 layer and a
  later stage's layer is refused as a conflicting placement). The embedding, final norm and head stay
  on rank 0.
- **The head stays on rank 0.** The last stage sends the residual back; rank 0 then runs the final
  norm, head and sampling exactly as on one GPU. That costs one extra ~20 us hop per forward pass and
  keeps the round buffers, prefill buffers and every other head-side structure where they are.
- **Per-rank state.** `DeviceKVPagePool` is one page-group allocator over planes bound to several
  ranks' backings, so admission, prefix reuse and the context-cache policy never see how many devices
  hold the cache. Block tables are one host shadow with a device copy per attention-bearing rank
  (`KVExecutionTablePool`); GDN state is one `LinearAttentionStatePool` shard per stage
  (`StateImageDevicePool`); the host image of a state keeps the whole model's layers at their global
  offsets. Data movement takes `RankStreams`, so each plane's copy goes on its own rank's stream, and a
  single stream given to a multi-rank object fails instead of using the wrong device.
- **Capacity.** Each further device has its own affine reservation curve
  (`RankCapacityCurve`); the resolved page-group count is the smallest any device allows, and a
  failure names the device that could not hold the plan. Ranks sharing one physical device split its
  free memory.
- **`StageLink`** (`core/stage_link.h`) moves the residual between stages, and the small control
  tensors (positions, KV rows, slot indices, valid columns) the layers read from rank 0, through a
  ring of pinned-host slots. The copies are ordinary graph nodes. Decode captures the whole pass into
  one multi-device graph, so `StageLink` tracks, per event, which capture made its last record: a wait
  inside a capture may only target an event recorded in that capture, and an eager wait may not target
  one whose last record was in a capture (`cudaErrorInvalidValue`, even after the graph has
  launched).
- **`TextContext::run_staged`** packs the control tensors on rank 0, sends each stage its block up
  front, runs stage 0, and for each later stage rebinds the context's device, stream and workspace
  (`ScopedDeviceRank`, `ScopedArenaRank`), receives the residual and control block, runs the stage's
  layers and passes the residual on.

## What is not covered yet

- **DFlash/DFlash2** are refused when the model spans more than one device. DFlash reads
  layer outputs from several depths (its feature taps) into rank 0 buffers, which from a later stage
  are another device's memory; they need to cross the stage boundaries first.
- **Vision works, entirely on rank 0.** The tower, its encode workspace and the handoff live on the
  first device, and the visual columns are scattered into the residual before the stage loop, so a
  stage sees an ordinary residual; the multimodal `[3,T]` rope positions ride the control block.
  A resident tower is added to rank 0's share in `default_stage_layers`. `--vision-residency overlay`
  keeps its single-device meaning: the window is borrowed from rank 0's evictable tail (embedding,
  head, MTP) or rank 0's free KV granules (`lendable_kv_end_bytes` counts rank 0 planes only; a
  lent granule removes those page ids from admission on every rank). Only exercised with ranks
  sharing one device so far; distinct GPUs are unverified.
- **Prefill does not overlap stages.** A prefill chunk runs through the stages in turn and the
  engine synchronizes after each chunk, so at any moment one stage is busy. Overlapping stages needs
  micro-chunks inside a chunk (later stages start on micro-chunk 0 while stage 0 runs micro-chunk 1);
  that changes the chunk shapes the kernels see, and its benefit can only be measured on real cards.
- **MTP works** across stages: the replay records and their fold are per state shard, on the shard's
  own device, and the draft layer and head stay on rank 0. **The context cache works**: its
  transactions copy each rank's planes and state shards on that rank's streams.

## Verification

- `ninfer_multi_gpu_test`, three suites in one executable (ranks share device 0, the staged protocol
  forced): the stage link (byte integrity across sizes and ring reuse, a two-hop chain, payload
  halves through separate graphs, the slot-reuse dependency read off the graph, eager use of a slot
  before and after a capture), the multi-rank KV pool and block-table copies against the single-rank
  pool as the oracle in both plane orders, and the stage solver against a brute-force partition
  enumeration.
- `ninfer_kv_capacity_test`: the per-device curve, including a device without KV and errors that
  name the device.
- `ninfer_qwen3_5_stages_real_test` (set `NINFER_TEST_ARTIFACT`): greedy output with `--devices 0,0`
  and `0,0,0` equals `--device 0` byte for byte, since the layers run the same kernels on the same
  data. Rows: graphs and eager, forced pinned-host transport, an uneven split, MTP, and a
  continuation that reuses the context cache. The 27B on the RTX 3090 passes every row.
- `ninfer_qwen3_5_loading_real_test`: the default split covers the model, gives the head stage
  fewer layers, and gives a device with twice the memory more.

Ranks sharing a device cannot show a wrong-device pointer: a stale pointer into "another rank's"
memory still works there. That class of bug needs a real second card.

## Measured on two real GPUs (2026-09-21)

Rented Linux boxes, CUDA 12.8 build, Qwen3.6-27B groupwise-int (the pinned Hugging Face artifact),
int8 KV. Neither box has peer access, so every boundary transfer goes through pinned host memory.

| | 2x RTX 3090 (PCIe 3.0 x16 each) | 2x RTX A4000 (x8 + x16, cross-NUMA) |
|---|---|---|
| `ninfer_qwen3_5_stages_real_test` | every row byte-identical to one device: 2 stages, 3 stages (devices 0,1,0), graphs and eager, forced staged transport, uneven split, MTP, prefix reuse | the model does not fit one 16 GB card, so every row is compared with the first: identical across cut points, graphs and eager, staged transport, MTP, prefix reuse |
| CLI, prefill chunk 1024, splits 20/44, 44/20, 32/32 and the default | output identical to one device | output identical across splits, and identical to the 3090 output |
| Decode, plain | 46.9 tok/s on one card, 48.5 split | 24.1 tok/s |
| Decode, MTP3 | 100.2 tok/s on one card, 105.3 split | 52.6 tok/s |
| Prefill | 1.56k tok/s on one card and split | 825 tok/s |
| Weights per card (default split) | 15.3 GiB on one card, 8.35 GiB on the first split card | 7.71 GiB |
| `--kv-capacity auto` at `--max-context 262144` | one card: refused (minimum reservation 9.2 GB, 8.76 GB free after weights); split: 262,144 tokens, 11.2 GiB free afterwards | 262,144 tokens, 3.9 GiB free afterwards |

A `ninfer-serve` check on the 3090 pair (four lanes, concurrent and repeated chat requests, prefix
cache on) gave the same output split as on one card for every request run alone, run in parallel
once, and repeated; a second concurrent round differed on one request, and one card differs from
itself the same way, because lane batching changes the arithmetic.

What this says: the split costs nothing measurable at decode on PCIe 3.0 x16 (one 20 KB residual
crossing per stage boundary per token is about 25 us against a ~21 ms step), and it is what makes the
27B's full 262,144-token context fit at all. It does not speed anything up: prefill is the same and
decode is within noise. The A4000 pair runs the 27B with its full context and its decode rate matches
the plan's estimate for a pipeline on 448 GB/s cards.

One defect turned up only on the 48-SM A4000: the GDN gating projection's route table was tuned on 82
SMs, and its cooperative schedules need their whole grid resident, so the default `--prefill-chunk 1024`
failed at startup. The covering route now falls through to the first later route that fits the device
(`ninfer_gdn_gating_proj_test` checks it against an independent grid model at 30 to 84 SMs and
that the tuned 82-SM routes are unchanged). Its performance on the A4000 has not been tuned.

Launch note for rentals: the CUDA base images carry a `compat` `libcuda.so.1` that GeForce cards
refuse (`cudaErrorCompatNotSupportedOnDevice`); put `/usr/lib/x86_64-linux-gnu` first in
`LD_LIBRARY_PATH`. Ubuntu 22.04 needs CMake 3.28+, FFmpeg 6 development packages and libcurl 7.85+
built or installed separately.

## Tensor parallelism: status

Not built. What was established, so the next attempt starts from evidence rather than from the plan:

- **Transport** (`tools/tp_probe.cu`, rented boxes, 2026-09-21): neither a 2x RTX 3090 box (one slot
  PCIe 3.0 x4), a second 2x RTX 3090 box with both slots x16, nor a 2x RTX A4000 box had peer access
  or NVLink. The x16 pair measured 15 us (host-mapped kernel) to 24 us (staged, in a graph) for one
  column, 33-38 us for four, 75-78 us for sixteen, 130 us for thirty-two and 3.4 ms for a 20 MB
  prefill chunk, half the x4 box's time at every size. An all-reduce of one decode column
  (20 KB, FP32) costs 17-33 us whether done by a host-mapped kernel, a staged pinned-host copy or
  NCCL; at 640 KB it is 190-440 us and at 20 MB 5-13 ms. The transports are within about 20% of each
  other at every size, so a first TP should reuse `StageLink`-style staged copies (they already
  capture into graphs) and a local reduce, not write spin-wait kernels.
- **Where it pays.** Projected against this pipeline on two A4000s: 1.6-1.7x at one decode column,
  about 1.5x at four, roughly break-even at sixteen, a loss at thirty-two (MTP3 with eight lanes).
  TP prefill is bound by the link (~840 ms of all-reduce per 1,024-token chunk), so prefill speed
  belongs to the micro-chunk overlap above, not TP. Link width is per machine and decides
  everything; run the probe on the target box first. On the x16 3090 pair the 21 ms decode step
  would become about 16 ms with MLP-only TP (64 collectives at ~27 us against half the MLP bytes)
  and about 14 ms with full TP (128 collectives), i.e. roughly 1.3x and 1.5x at one column.
- **Smallest useful first step: MLP-only TP on the dense 27B** (about 64% of the weight bytes;
  projected ~1.35x decode). Each rank keeps attention and GDN whole and replicated (weights and KV),
  and takes half the MLP: `gate_up` split by rows (gate rows then up rows, so the SwiGLU pairing
  holds) and `down` split along K. A rank's down projection can reuse `linear_add`: rank 0 adds its
  partial into the real residual, the other rank into zeros, and one BF16 exchange plus an add gives
  every rank the same new residual (one extra rounding against the single-device path, so the oracle
  is a tolerance on logits and perplexity, not byte equality).
- **What it needs before any speed can be claimed:** loader-time slicing of row-split quantized
  parents (contiguous row ranges for `gate_up`, per-plane column ranges for `down`, group-aligned);
  the `linear_swiglu` Q4 kernel at 17,408 rows and `linear_add` Q5 at K=8,704, each registered for
  those exact shapes and qualified against the FP64 oracle and tuned on the target card (routes
  inherited from the full-width shapes are hypotheses); per-rank replicas of the KV pool and state
  pool with fan-out in the transactions; and the exchange itself. Expect weeks, and the decisive
  measurement is the probe on the machine that will run it.
