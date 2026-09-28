# Single-GPU serving performance

> **Read the hardware label before the numbers.** This index and the per-model pages it links to
> carry measurements from two different GPUs, and they must not be read against each other:
>
> | section | hardware | measured by |
> |---|---|---|
> | [RTX 3090 (`sm_86`) findings](#rtx-3090-sm_86-findings-this-fork) below | RTX 3090, `sm_86`, CUDA 12.8 | this fork |
> | [Vision residency on RTX 3090](performance/qwen3.8-27b.md#vision-residency-on-rtx-3090-groupwise-int-sm_86) | RTX 3090, `sm_86` | this fork |
> | Every per-model page under `performance/` | **RTX 5090, `sm_120a`, CUDA 13.1** | upstream |
>
> The upstream campaign is kept because it is the only corpus-scale evidence published for these
> artifact profiles, and all of its tested revisions are reachable in this fork's history. It is
> **not** a statement about how this fork performs on an RTX 3090. The two parts answer different
> questions: upstream's tables characterise the artifact profiles, and this fork's measurements
> characterise `sm_86` kernel behaviour — where upstream's inherited route boundaries were wrong by
> 12–41%.

## RTX 3090 (`sm_86`) findings — this fork

**DFlash2 measured on this fork (RTX 3090, Qwen3.8-27B groupwise-int, `--kv-dtype int8`,
`--draft-tokens 7`, greedy, 96 new tokens).** Text: 20.0% acceptance, 2.38 tok/round.
Vision (`--vision`, the committed `image_chart` fixture): 85.7% acceptance, 7.00 tok/round,
and byte-identical output to the non-speculative vision run. These are single-prompt smoke
numbers, not a campaign; the [Vision residency](performance/qwen3.8-27b.md#vision-residency-on-rtx-3090-groupwise-int-sm_86)
and per-model pages remain the measured corpus results.

**Speculative decoding is not bit-identical to non-speculative decoding here, it is not required
to be, and every configuration is nonetheless deterministic in itself.** Measured 2026-09-09 by
hashing the generated text of 23 configurations x 3 repetitions
(`scripts/sweeps/dflash2-draft-tokens-realtext.ps1`, `content_sha256`):

- **every configuration reproduced its own output exactly**, 23 of 23, three runs each;
- the width-1 greedy path produced a hash matched by no speculative configuration;
- **DFlash2 and MTP do not match each other**, and DFlash2's output varies with the draft count —
  eight distinct outputs across k = 1..12.

Verification evaluates k+1 columns in one pass while plain decode evaluates one, so the reductions
run in a different order and a near-tie argmax can flip; a different draft count is a different
width and so a different order again. That is why bit-identity to greedy is not a target here: it
would require computing the accepted column with the width-1 kernel on every round, which is the
work speculation exists to avoid. It costs nothing in quality — swapping an MMA tile for a
different reduction order leaves perplexity bit-identical to twelve significant figures — so what
is guaranteed is per-configuration determinism, not cross-configuration equality.

*An earlier version of this section claimed DFlash2 and MTP "produce the same output as each other".
That was inferred from one prompt and is wrong; the hashes above are the measurement.*

**Acceptance on realistic text, which the committed corpus cannot measure.** 27B DFlash2, INT8 KV,
greedy, 256 generated tokens of the model's own prose, medians of three. `bench/fixtures/bench_corpus.ids`
reports a flat 100% at every draft count because it is a curated bank tiled to length; these are the
numbers to quote instead:

| `--draft-tokens` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acceptance | 80.9% | 67.6% | 51.7% | 49.1% | 42.1% | 33.1% | 28.2% | 31.0% | 23.2% | 19.5% |
| tok/round | 1.81 | 2.35 | 2.55 | 2.95 | 3.07 | 2.97 | 2.93 | 3.45 | 3.27 | 3.27 |
| decode tok/s | 47.3 | 55.2 | **58.6** | 58.3 | 57.2 | 51.0 | 50.5 | 47.5 | 39.7 | 39.3 |

Acceptance falls steadily as the window widens, from 80.9% at k=1 to 19.5% at k=12, with one
uptick against the trend at k=7 -> 8 (28.2% -> 31.0%); tokens-per-round plateaus near 3.0 from
k=4 — which is the real reason to recommend four rather than more. MTP3 reaches 58.3 tok/s at 56.0%
acceptance, and MTP3 with `--lm-head-draft` is the fastest configuration measured at 62.9 tok/s.

**Route boundaries are measured here, not inherited.** Two maintainer benches time every
schedule of an Op at the same column count, cold, so a boundary can be chosen from data rather
than from whichever table happened to be live:
`bench/ops/q4_q5_attn_input_schedule_bench.cu` and
`bench/ops/q4_linear_swiglu_schedule_bench.cu`. Both take an explicit per-schedule domain,
because a kernel handed more columns than it supports either corrupts memory (q4_q5
parent_split_fixed past 12) or silently does one column's work at a constant cost and appears to
win everywhere (q4 swiglu gemv_pair, registered for a single column). Cold is the production
regime for these: the Q4/Q5 attention projection alone streams ~35 MB of weights per call
against the 3090's 6 MB of L2.

The Q4/Q5 attention-input table had been dead code since the catch-up -- `resolve_plan` was a
hardcoded chain of upstream's sm_120 boundaries -- and routing through the measured table
exposed a `switch` fallthrough that ran a second kernel over the first. Public-Op cost at T=16
fell 502.8 -> 226.3 us and at T=32 500.7 -> 217.1 us. Q4 SwiGLU's SmallTTiled bound returned to
24, where the tiled kernel and the 40-wide pair tile actually cross (6% at T=25, 14% at T=32).

Concurrent decode extents are what the sm_86 kernel routes are selected for. A decode round covers
`concurrency x (draft window + 1)` token columns, and above the single-token point the cost of a
route is set by its padded tile width rather than by the live column count. Selecting the narrowest
tile that still covers each extent is worth 40% at C4 and C8; C1, whose four columns already sit on
the exact-T routes, is unchanged.

**The upstream DFlash2 corpus numbers are still not reproduced here.** The upstream catch-up
brought the DFlash2 speculative backend (`--spec dflash2 --draft-tokens 7`, Qwen3.8-27B only), and
upstream publishes DFlash2 corpus numbers for their own hardware. Those are deliberately not
added to the findings in this section: only single-prompt smoke numbers exist on a 3090 so
far, and pasting another architecture's corpus results beside them would read as agreement that
has not been measured. DFlash2 rows will be added here once measured on a 3090.

### Handing prefill GEMMs to cuBLAS (`--prefill-cublas`, opt-in)

**Result.** Prompt processing is 43-83% faster again, for +0.156% perplexity. The weights are
materialised as int8 with one scale per row and the GEMM is handed to cuBLAS, which runs this card's
shapes about twice as fast as this fork's own integer mainloop can. Measured on one RTX 3090,
Qwen3.8-27B groupwise-int, `--kv-dtype int8`, both arms on the same build and card in one session:

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| shipped integer-activation routes | 1,666.6 | 1,634.4 | 1,503.9 | 1,240.8 |
| **`--prefill-cublas --prefill-chunk 4096`** | **2,379.8** | **2,988.7** | **2,609.0** | **1,904.1** |
| change | +42.8% | +82.9% | +73.5% | +53.5% |

Decode is untouched (45.1 against 45.7 tok/s, the route is gated to wide token counts), and the
workspace grows from 155.6 MiB to 661.2 MiB.

**It is off by default because it is a quality trade, not a free win.** cuBLAS reduces over the
whole of K, so it cannot see a scale per 64 columns: the weight carries one scale per row and the
activations one per token. Perplexity on `ninfer-ppl-1m-v1` (quick preset, kv int8) against 4.343155
for the integer route:

| | perplexity | change |
|---|---:|---:|
| MLP and out_proj on the route | 4.346284 | +0.072% |
| plus the attention and GDN input projections | 4.349944 | +0.156% |

The second step is `--no-prefill-cublas-projections` to decline. For scale, this fork has accepted a
trade at +0.082% and rejected one at +0.69%.

**Two things had to be true for the quality to land there.** The weight side is cheap because int8
buys four more bits per code while only losing per-group scale granularity: measured on the real
artifact it costs 9.7e-3 to 1.1e-2 relative L2 (the median per projection kind; the worst single
tensor is 1.5e-2), against the 9e-3 to 2.0e-2 this fork already accepts from per-group activation
quantisation (`tools/w4_row_scale_error.cpp`, using the route's own row scale with channel
equalisation off). An earlier figure of 8e-3 to 1.0e-2 came from a clipping search the route cannot
perform, and was the best case rather than the route. The activation side was
*not* cheap until the channels were equalised -- activation outliers concentrate in a few input
channels, so a token's absmax is set by those and everything else is quantised against far too
large a step. Folding a per-channel scale out of the activations and into the weights is exact
(`X[j,t]/s[j]` with `W[i,j]*s[j]` leaves the product unchanged) and took the cost from +0.474% to
+0.072%.

**Chunk size is the knob.** The dequantise pass is weight-sized and the GEMM token-sized, so the
route wants a large `--prefill-chunk` and the two settings belong together. The trade-off table,
including what each chunk costs a concurrent decode lane, is in
`src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.h`.

### Integer activations for every registered prefill projection

**Result.** Prompt processing is 22-29% faster than the previous state at every context length:
the attention and GDN projections now reach the s8 tensor cores that only the MLP pair used, and
the schedule they share stopped re-streaming the weight matrix once per 128 tokens. Measured on one RTX 3090 at 315 W, Qwen3.8-27B groupwise-int, `--kv-dtype int8`,
`ninfer_bench -r 3 --warmup 1`, all three arms on the same build and card:

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| `--no-prefill-a8` (every projection A16) | 1,008.5 | 995.4 | 946.9 | 836.4 |
| previous state (integer MLP only, 128-token tile) | 1,303.2 | 1,277.5 | 1,195.1 | 1,022.9 |
| every registered projection, 128-token tile | 1,529.8 | 1,493.3 | 1,382.3 | 1,153.3 |
| **every registered projection, widened tile** | **1,684.9** | **1,648.5** | **1,521.8** | **1,242.7** |
| change against the previous state | +29.3% | +29.0% | +27.3% | +21.5% |

Three routes were added, in the order their Nsight share justified — a 4,096-token prefill spends
749 ms of 3,280 ms in the GDN input projection, 346 ms in the two output projections and 209 ms in
the attention input projection:

| route | shape | per-Op, T=1024 | end to end, pp4096 |
|---|---|---|---|
| attention o_proj and GDN out_proj | Q5 [5120,6144] LinearAdd | 1,097.7 → 827.4 us | +3.0% |
| GDN input projection | Q4 [4096,5120] + Q5 [12288,5120] | — | +10.1% |
| attention input projection | Q4 + Q5 [7168,5120] | — | +1.6% |

They share one schedule (`src/ops/common/rowsplit_a8_mma.cuh`): the 128x128 tile the MLP routes
measured, with the codec and the output mapping as template parameters. Both input projections are
two parents over one activation, so each quantises its input once for all its launches — O(K*T)
against the GEMMs' O(N*K*T).

**Quality.** Quick corpus, same build: 4.342982 with `--no-prefill-a8` against **4.343155** with
every integer route on, **+0.004%** — inside run-to-run noise, and inside the +0.05% this fork
requires before a lossy route is default-on. Relative L2 against the FP64 oracle is 0.010-0.020
across every destination range, against the 0.04 allowance A8 activation compute is held to.
Decode is untouched (45.8 against 45.6 tok/s at tg128): the routes admit only full 128-column
prefill tiles, so decode and partial chunks stay on A16.

**The token tile, and why it was hiding.** Ablating the probe at the gate_up shape, T=512, gives
the decomposition that explains the whole schedule -- each row removes one cost and keeps the MMA
count identical:

| | us | |
|---|---:|---|
| complete kernel | 2,068 | |
| minus the activation and weight scale reads | 1,918 | the rescale's loads are ~7% |
| minus the per-group rescale entirely | 1,856 | the rescale is ~10% |
| minus the A-fragment shared reads | 1,423 | **assembling A is ~21%** |
| minus the B-fragment reads as well | 1,424 | B is free: one 128-bit load per n-tile |
| minus the MMAs, keeping every load | 1,377 | |
| minus the MMAs *and* every shared read | 1,378 | **streaming alone is 67% of the kernel** |

So these kernels were never compute-bound, and what generates the traffic is the token tile: a
block covering BN tokens re-streams the entire weight matrix once per column block, eight times
over at the production chunk of 1,024. Widening it is worth more than everything else attempted:

| tile (rows x tokens) | us at T=1024 | TOP/s |
|---|---:|---:|
| 128 x 128 | 4,265 | 85.6 |
| 128 x 256 | 3,702 | 98.6 |
| **64 x 512** | **3,060** | **119.3** |

A grid swizzle to let L2 serve the repeated passes measured within 0.5% of nothing: the passes have
to be removed, not cached. gate_up is the exception that proves the rule -- its gate/up pairing
fixes the row tile at 128 and so caps it at 256 tokens, where the tile it gains does not pay for
the operand assembly it would lose, so it keeps its own kernel.

**What is still missing, measured against a tuned kernel rather than against a microbenchmark.**
`tools/int8_gemm_reference_probe.cu` runs cuBLAS's own int8 GEMM (IMMA, TN, s32 accumulate) over
these shapes on this card. It solves a strictly easier problem -- 8-bit weights, no unpack, no
per-group scale, no fused epilogue -- and it reads *twice* the weight bytes our int4 codes do, so
it is an upper bound rather than a like-for-like rival. At T=1024:

| shape | cuBLAS int8 | this fork's W4A8 | ratio |
|---|---:|---:|---:|
| gate_up 34816x5120 | 237.5 TOP/s | 95.5 | 2.5x |
| down 5120x17408 | 176.1 TOP/s | 100.1 | 1.8x |
| out_proj 5120x6144 | 176.7 TOP/s | 93.3 | 1.9x |

That reading also corrects the ablation above: cuBLAS finishes the entire gate_up GEMM in 1,537 us,
less than the 1,711 us our schedule spends on streaming *alone* at the same shape and tile. The
streaming is not a hardware floor, it is our streaming -- a 2-stage pipeline over 16 warps with the
A fragments assembled from per-lane 2-byte loads. Everything cheap has now been tried against it:
pipeline depth 2 -> 4 is worth 3.9%, interleaving the B loads with the MMAs -1.9%, occupancy 6%,
grid swizzle 0.5%, operand layout ~17% (and that one needs a repack this engine does not allow).
What is left is not a knob but a mainloop: register-level double buffering, `ldmatrix`, and a
swizzled shared layout, which is what CUTLASS-class kernels -- and the Marlin kernel the vLLM
stacks use -- are built out of.

**What did not pay, so nobody re-walks it.** `TODO.md`'s long-standing explanation for these
kernels running at ~30% of the INT8 ceiling — 124 registers holding the SM to 16 of 48 warps — is
not what costs the time. `tools/w4a8_rowsplit_probe.cu` measures the alternatives at the gate_up
shape, T=512:

- **Doubling occupancy buys 6%.** A 64x128 tile compiles to 55 registers and genuinely runs 2
  blocks per SM (32 of 48 warps, confirmed with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`):
  1,890 → 1,785 us, against the 40% `ncu` estimated.
- **Smaller tiles past that lose badly**: 128x64 at 63 registers measures 2,844 us, 1.5x worse.
- **Layout is worth nothing without a repack.** Streaming the weights with `cp.async` in their
  RowSplit order, packed nibbles in shared and the scales as an async ring — everything the
  fragment-order probe won except permuting the weights — measures 1,726 us against production's
  1,701. `tools/w4a8_real_weight_probe.cu` puts the fully repacked layout at 1,566 us with
  per-group scales, so the whole layout lever is ~8%, and a permuted copy of the 27B's two MLP
  matrices is ~9.7 GB on a 24 GB card. The 1,415 us headline needs per-token activation scales,
  whose relative L2 reaches 12.9% on outlier-heavy inputs against 0.9-2.0% per group.

### Small-T tensor-core kernels for verify and cohort decode

**Result.** Qwen3.8-27B MTP3 decode is 1.5x faster at C1 and 1.7x at C8 than v0.9.1, with
perplexity bit-identical. Measured before/after on one rented RTX 3090 in one session: Linux,
CUDA 12.8, 350 W, INT8 KV, MTP3 with the optimized draft head, CUDA Graphs, greedy unless noted.

| workload | v0.9.1 | small-T kernels |
|---|---:|---:|
| thinking-off chat, C1 (their 8 prompts, C × 1000 / mean TPOT) | 74.6 tok/s | **113.2 tok/s** |
| thinking-off chat, C1, temperature 0.7 | 72.6 tok/s | **112.8 tok/s** |
| thinking-off chat, C8 | 268.6 tok/s | **460.4 tok/s** |
| reasoning cohort C1 / C2 / C4 / C8 decode | 63 / 89 / 137 / 221 | **93 / 168 / 271 / 376** |
| `ninfer_bench` plain / MTP3 | 39.98 / 53.96 | **47.15 / 85.29** |
| perplexity, quick corpus | 4.342425 | 4.342425 |

The thinking-off prompts are
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)'s
`bench/prompts_real.jsonl`. That project serves the same model on the same card through patched
vLLM, and reports 111-124 tok/s at C1 and 407.3 at C8 with the same metric. Their card is capped
at 250 W and they report 5-8% run-to-run spread, so C1 is parity and C8 a lead, measured on
different machines.

**Why the round cost too much.** An MTP3 verify is four token columns and a C8 cohort round is 32.
The profile attributing an MTP3 round against a plain decode step
(`scripts/sweeps/decode-step-profile.sh 27b-decode 27b-decode-mtp3`) had the round at 38.2 ms
against a 24.8 ms step, about 1.5x. vLLM/Marlin pays 1.14x for the same round on this card. The
extra time sat in four families: Q4 gate_up, the Q5 residual projections, and the GDN and
attention input projections. At four columns they ran either SIMT split kernels, whose cost grows
with every column, or a Q4 small-T MMA that was instruction-bound. The round is now 25.7 ms
against a 22.3 ms step, 1.15x.

**What changed**, cold single-kernel medians in us, from the schedule benches
(`bench/ops/*_schedule_bench.cu`):

| Op (27B shape) | T | v0.9.1 route | now |
|---|---:|---:|---:|
| Q4 gate_up, 34816 × 5120 | 4 | 200.7 | **120.8** |
| | 32 | 444.4 (c40 tile) | **~205** |
| Q5 down, 5120 × 17408 | 4 | 99.3 (split2) | **82.9** |
| | 32 | 289.8 (best MMA tile) | **144.4** |
| Q5 out, 5120 × 6144 | 4 | 38.9 (split2) | **34.8** |
| | 32 | 105.5 (best MMA tile) | **55.3** |
| GDN input, Q4 4096 + Q5 12288 rows | 4 | 100.4 (independent) | **78.8** |
| | 32 | 263.2 (grouped c32) | **153.6** |
| attention input, Q4 7168 + Q5 7168 rows | 4 | 90.1 (parent split) | **68.6** |
| | 32 | 191.5 (r32/c32) | **135.2** |

- **Bank conflicts.** The Q4 small-T MMA staged its code rows 256 B apart, an 8-way shared-memory
  bank conflict on every A load. Padding the rows was worth 9.6% at C1 on its own.
- **Permuted k order.** The MMA's k slots may be any permutation of a group, provided A and B agree.
  Giving each lane sixteen contiguous k makes its A operand one 64-bit code load, decoded to bf16
  by a magic-bias `byte_perm` with no int-to-float, and its B operand two 128-bit loads with no
  `ldmatrix`.
- **A Q5 counterpart** (`q5_small_t_mma.cuh`) folds the high-bit plane into the same decode. It
  replaces split2 from T=3 and the MMA tiles to 32 columns, for the residual projections, GDN
  value/z and attention gate/value.
- **Wide extents share activation slabs** (`ops/common/small_t_layout.cuh`). At 32 columns and one
  16-row tile per CTA, gate_up re-read ~713 MB of staged activations per call against 89 MB of
  weights. Two to four row tiles per CTA now share each staged slab, and at 24 columns each warp
  feeds one B fragment to two tiles.
- **The fused GDN conv projection is gone.** Batch-1 widths 1-3 and 5-6 of the GDN conv
  forms ran a fused SIMT projection that lost at every width: 95.2 against 80.9 us at T=1, and
  162.8 against 85.0 at T=5. At T=5 that was 4.8 ms of a four-draft-token round. Removing it made
  plain decode 5% faster, and four or five draft tokens stopped costing more than they return.

**Measured and rejected**, so nobody re-runs them:

| idea | result |
|---|---|
| split-K across CTAs for the 5120-row Q5 shapes | 0-2.5% at narrow T; not worth a workspace |
| magic-bias decode alone, before the k permutation | neutral in situ |
| 2- and 3-stage cp.async rings for the narrow Q4 tile | slower; occupancy is the latency hiding |
| deeper rings through dynamic shared memory (to 4 stages, 99 KB) | slower at every shape: occupancy loss outweighs the hidden latency |
| two tiles per warp at 16 and 32 columns | slower (kept only at 24, where it wins 14%) |
| `cp.async ... L2::256B` prefetch hint on weight loads | neutral |
| a 40-65K-row prefix of the frequency-sorted draft head | acceptance 62.7% -> 56.5% at 40,960 rows; net 4% slower |
| drafting with the full output head | +2% acceptance for 4 ms more per round |
| programmatic dependent launch to hide kernel ramp | needs sm_90; compiled out on `sm_86` |

What is left at C1 is mostly ramp-up and drain at the ~500 kernel boundaries of a round (the
Q5 residual kernels reach 68-82% of their streaming floor, gate_up 89%).

**What is left at C8 is not tensor-core rate, though this page said so for a cycle.** The claim was
that gate_up at T=32 runs at about 68% of the card's measured bf16 MMA peak -- 205 us against a
139 us tensor floor -- which reads as a kernel most of the way to saturating its tensor cores.
Counters disagree. `ncu` on the T=32 gate_up kernel, Windows RTX 3090 at 315 W:

| | BF16 small-T | int8 small-T |
|---|---:|---:|
| tensor pipe active | **38.3%** | 21.0% |
| L1/TEX throughput | 36.4% | 73.5% |
| DRAM throughput | 39.4% | 42.7% |
| warp occupancy | 30.9% | 45.4% |

Nothing is near saturation in the BF16 column: tensor, L1 and DRAM all sit within three points of
each other around 38%, at 31% occupancy, which is the signature of a latency-bound kernel rather
than one limited by any unit. The 68% figure compares against a computed floor, not against a
measured pipe, and the two do not agree.

That distinction decided a real experiment. An int8 tensor-core route for this Op was built on the
strength of the old reading, since s8 MMA is about 4.7x the bf16 rate on this hardware. It
delivered 4-6%, not 4.7x, and the right column above says why: it did exactly what a denser MMA
should, halving tensor-pipe pressure from 38.3% to 21.0%, but that pipe was never the constraint,
and the cost of getting there landed on L1 at 73.5%. See
[the quality-trade notes](maintainer/quality-trade-experiments.md) and
`src/ops/linear/q4/q4_small_t_mma_i8.cuh`. Work aimed at C8 should target operand movement and
occupancy; a wider tile that puts more work in flight is the lever the counters actually point at.

**Reproduce.** Build both trees with `-DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON` and run
each harness against both `ninfer-serve` binaries on one card, interleaved in one sitting:
`tools/bench/run_qwen38_replayssm_cohort_sweep.py` for the cohort, and for the thinking-off numbers

```bash
python tools/bench/run_chat_decode.py --model models/qwen3_8_27b.ninfer \
  --prompts prompts_real.jsonl --concurrency 1 --reps 2 --out chat-decode \
  --arm base=/path/to/old/ninfer-serve --arm new=./build/apps/ninfer-serve
```

with the eight prompts of syv-ai/qwen38-27b-rtx3090's `bench/prompts_real.jsonl` (that file is
theirs and is not vendored here). `--concurrency 8` gives the C8 row. Both report decode as
C × 1000 / mean TPOT from the server's own request log.

### Recommended configurations (RTX 3090, Qwen3.8-27B)

Every flag below is measured elsewhere in this file or in
[quality-trade-experiments.md](maintainer/quality-trade-experiments.md). Nothing here is a guess.

**Fastest at one stream, when context beyond ~130K is not needed:**

```
--spec dflash2 --draft-tokens 7 --lm-head-draft --prefill-cublas --prefill-chunk 4096 --kv-dtype rk8v4 --embedding-q4 --gdn-state-fp16
```

Prefill about 1.7x and decode about 1.39x against the previous defaults. Costs +0.156% perplexity
from the cuBLAS route, +0.083% from rk8v4, and nothing measurable from the other two.

**Longest context, still fast:**

```
--spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --prefill-chunk 2048 --kv-dtype rk8v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16
```

**200,000 tokens of context, verified by loading it.** MTP rather than DFlash2 because DFlash2's
draft weights and its refusal of `--lm-head-q6` cost about 65K tokens between them -- the same
configuration on DFlash2 loads at 130K and fails at 150K (needing 4.79 GB against 4.46 free) and at
200K (6.09 GB against 4.45). Chunk 2048 rather than 4096 because the larger chunk costs ~300 MiB of
workspace for its last 4% of prefill, and at this context that is the binding constraint.

**Why these flags and not others:**

| flag | what it buys | what it costs |
|---|---|---|
| `--kv-dtype rk8v4` | 23% less KV than int8 — 26,112 B/token against 33,792 | +0.083% perplexity, ~1% decode |
| `--embedding-q4` | +24.3K tokens of context | nothing measurable (−0.062%, inside noise) |
| `--lm-head-q6` | +12.9K tokens of context | +0.012%; **incompatible with DFlash/DFlash2** |
| `--gdn-state-fp16` | halves the host state image, modest C8 gain | nothing measurable |
| `--prefill-cublas` | prefill 1.63x–1.83x | +0.156% perplexity, ~500 MiB workspace |
| `--lm-head-q4` | +24.3K tokens | **+0.69%** — larger than every KV format; not recommended |
| `--kv-dtype nvfp4` | 45% less KV than int8 | +0.36% perplexity, 12% slower decode |

**With vision**, add `--vision --vision-residency overlay`: the overlay residency keeps the tower in
host memory and borrows device memory per image, which is what preserves the context budget above.
Note that `--lm-head-q4` was silently skipped under overlay vision before 2026-09-14; `--lm-head-q6`
and `--embedding-q4` transcode in the materializer and are captured from the final bytes, so they
compose with it correctly.

### Choosing a speculative backend by concurrency (RTX 3090, Qwen3.8-27B)

**DFlash2 at K=7 is faster than MTP3 wherever it fits, and it stops fitting at C8.** Aggregate
decode tok/s through the serving route, thinking off, greedy, decode time taken from the server's
request log rather than a wall clock:

| C | MTP3 | DFlash2 K=7 | change | tokens/round, MTP3 / DFlash2 |
|---:|---:|---:|---:|---|
| 1 | 135.0 | **187.1** | +38.6% | 3.51 / 5.57 |
| 2 | 238.2 | **313.5** | +31.6% | 3.51 / 5.82 |
| 4 | 387.4 | **406.2** | +4.9% | 3.54 / 5.63 |
| 8 | **522.8** | does not fit | — | 3.53 / — |

The lead shrinks with concurrency because speculation pays by making a multi-column round cost about
one sweep of the weights, and batching already amortises that sweep across lanes, so the baseline
catches up. Acceptance itself does not degrade: tokens per round holds flat at every level.

What stops it at C8 is memory. The DFlash2 artifact carries the draft model — 18.3 GiB of weights
against 16.7 — and the runtime reservation then needs 4.64 GB where 3.78 GB remains. A 4096-token KV
still needs 4.24 GB; only 2048 fits, which is too little for eight streams to do useful work.

**K is workload-dependent.** Swept on realistic generation, mean of a reasoning, a code and a
summarisation prompt: K = 3/4/5/6/7/8/9/10/12 gives 128.0/146.0/159.0/169.6/**172.3**/163.7/163.3/
160.2/159.2 tok/s. The peak is 7, matching the published shape `E = (1 - a^(K+1))/(1 - a)` and the
5–8 band reported for EAGLE-class drafters. But the mean hides real spread — the reasoning prompt
keeps improving to K=12 (204.5) while summarisation collapses there (119.6) — so a deployment
serving one kind of work should sweep its own K. A synthetic corpus is no guide: `bench_corpus.ids`
picks K=15, which loses on real generation.

**DFlash2 also forfeits `--lm-head-q4/q6`**, which DFlash's candidate top-k cannot read
(`src/models/qwen3_5/load/storage_trades.cpp`). Between the extra weights and the lost head trade it
costs about 65K tokens of context on this card -- 130K against 200K, both verified by loading -- so
the choice between it and MTP3 is a speed/context trade, not a free upgrade.

### Choosing a KV format (RTX 3090, Qwen3.8-27B)

All seven SM86 KV formats, measured on Qwen3.8-27B.

| KV profile | Bytes/token | KV at 2,048 tokens | Perplexity | vs `bf16` | Decode at 32K depth |
|---|---:|---:|---:|---:|---:|
| `bf16` | 65,536 | 128.00 MiB | 4.343225 | — | 32.50 tok/s |
| `int8` | 33,792 | 66.00 MiB | 4.343263 | +0.0009% | **33.86 tok/s** |
| `fp8` | 33,024 | 64.50 MiB | 4.347181 | +0.0911% | 30.13 tok/s |
| `rk8v4` | 26,112 | 51.00 MiB | 4.346811 | +0.0826% | 33.54 tok/s |
| `k8v4` | 25,728 | 50.25 MiB | 4.347596 | +0.1006% | 28.61 tok/s |
| `nvfp4` | **18,432** | **36.00 MiB** | 4.358924 | +0.3615% | 29.62 tok/s |
| `rk4v4` | 17,920 | 35.00 MiB | 4.352432 | +0.214% vs `int8` | ≈ `rk8v4` |

Perplexity is `ninfer-perplexity` on the fixed `ninfer-ppl-1m-v1` corpus, `--quick`, context/stride
4096/2048, 261,167 scored tokens. Decode is 128 timed steps on top of a 32,768-token prefill, no
speculation; attention re-reads the whole cache each step, so a format's cost only shows at depth.
The `rk4v4` row comes from a later session and build (2026-09-23), where `int8` measured 4.343155,
`rk8v4` 4.347943 and `nvfp4` 4.353589, and `rk4v4` decoded within ±1% of `rk8v4` at 4K-32K; its
figures are stated against those rather than this table's. See the README's
[`rk4v4` section](../README.md#lloyd-max-4-bit-keys-rk4v4).
See [the README](../README.md#choosing-a-kv-format) for the fuller writeup and recommendations.

**Not built: `rk2v4-e8` (2-bit E8 root keys).** The sibling RTX 4090 forks ship an `rk2v4-e8` mode:
keys as one E8 root index, a 4-bit log radius and a 4-bit axis correction per 8 dimensions, about
216 B per head and token here against `rk4v4`'s 280. Simulated on this build by passing `int8`'s
rotated keys through that fork's encoder and decoder (`absmax/7` G64 scale) and storing the decoded
keys as INT8, with values left at INT8, the same quick protocol measured **4.476942 against
4.343155 (+3.08%)**, with every domain between +2.2% and +3.9% (2026-09-29). That is the keys
alone, about fourteen times `rk4v4`'s whole +0.214%, and it agrees with the rule `rk4v4` found that
any 3-bit key coding costs +0.7% or more. `rk4v4` already reaches the model's native 262,144-token
context on the 27B, so the extra headroom does not pay for the quality.

## Published coverage

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded historical measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

Each cell links to the relevant result section. "Not published" describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | Not published |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. The Qwen3.8 NVFP4 saturation reports retain configuration and
values but no tested Git revision; the model page records that provenance limitation.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 DFlash2 outcomes](performance/qwen3.8-27b.md#dflash2-completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts), with a [README summary](../README.md#evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
