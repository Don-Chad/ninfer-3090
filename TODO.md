# TODO

## Decode: two proven wins waiting on the attention work, 2026-09-19

Both are measured, both are large, and both are deliberately parked until INT8 PV lands, because
that change touches the same rounds and should be proven first.

**1. RESOLVED 2026-09-19: DFlash2 is a single-stream optimisation only, and the sweep stays on MTP3.** Measured at `-pg 4096,256`,
kv int8, on one 3090:

  | config | decode tok/s | acceptance |
  |---|---:|---:|
  | plain, no speculation | 45.1 | - |
  | **MTP3 -- what the scripts and published numbers use** | 144.9 | 0.96 |
  | MTP5 | 182.7 | 0.95 |
  | DFlash2 K=7 | 249.3 | 0.95 |
  | **DFlash2 K=15** | **326.2** | 0.87 |

  The mechanism, which explains the size of it: a 16-column verify round takes 20.9 ms against a
  22.2 ms one-column plain step, because decode is memory bound at ~87% of roofline and both are
  one sweep of the 15.9 GiB of weights. **Extra columns are nearly free; acceptance is the only
  thing that converts them into tokens.** It also says where it stops -- at 0.868 per-token
  acceptance the expected run saturates at 1/(1-0.868) = 7.6 tokens, which is why K=13 -> 15 moved
  only 319 -> 326.

  **Confirming it on real generation changed the answer, and this is why the corpus was not
  enough.** Three prompts -- a reasoning question, a code question, a summarisation -- greedy, 400
  tokens, mean tok/s: MTP3 124.3, MTP5 145.9, **DFlash2 K=7 172.3**, DFlash2 K=15 156.3. The
  synthetic corpus says K=15 (326 against 249 for K=7) because it continues itself and acceptance
  stays high; on real generation the extra columns stop being accepted and are paid for anyway, so
  **K=15 loses to K=7**. The 2.25x above is a corpus artefact; the real number is 1.39x.

  **The cohort measurement, done correctly.** Aggregate decode tok/s through the serve path,
  thinking off, decode time read from the server's request log rather than a wall clock:

  | C | MTP3 | DFlash2 K=7 | change | tokens/round |
  |---:|---:|---:|---:|---|
  | 1 | 135.0 | 187.1 | +38.6% | 3.51 / 5.57 |
  | 2 | 238.2 | 313.5 | +31.6% | 3.51 / 5.82 |
  | 4 | 387.4 | 406.2 | +4.9% | 3.54 / 5.63 |
  | 8 | 522.8 | **does not fit** | - | - |

  DFlash2 is faster at every level it can run, and the lead shrinks with concurrency exactly as the
  mechanism predicts -- speculation pays because a multi-column round costs about one sweep of the
  weights, and batching already amortises that sweep, so the baseline catches up. Acceptance itself
  does not degrade: tokens per round holds flat across all levels.

  What stops it is memory, not throughput. The draft weights are 18.3 GiB against 16.7; at C8 the
  runtime reservation needs 4.64 GB against 3.78 GB free, a 4096-token KV still needs 4.24 GB, and
  only 2048 fits -- too little for eight streams. **The sweep includes C8, so it stays on MTP3; a
  C1-C4 deployment should use DFlash2.**

  **An earlier version of this measurement said the opposite and it was wrong twice over**: it left
  thinking mode on while the CLI comparisons had it off, which changes the generated text and so the
  acceptance rate, and it timed wall clock per request, folding in HTTP, tokenisation, prefill and
  queueing. Together those reported 98 tok/s where decode was actually 196, and made the serve path
  look broken. It is not: whole-request host-exposed time is ~22 ms on a 2.2 s request, queue wait
  ~1 ms, and serve decode matches the CLI to within 0.5% once the two are configured alike.

**3. Adaptive draft window: the signal is free, acting on it is not.** Worth writing down because
the idea is obvious and the obstacle is not.

  The engine already reads `accepted_drafts` and `licensed_counts` back to the host every round, so a
  running per-sequence acceptance estimate costs nothing, and the optimal K for a given acceptance
  has a closed form. What is not free is *acting* on it: within a captured graph the round executes
  K+1 columns whatever the live extent, because the tail is masked by `target_valid_columns` rather
  than skipped. Shrinking the extent therefore saves no time at all.

  Over-drafting does cost real time, which is what makes this tempting. On a low-acceptance workload
  K = 3/7/12 measures 58.9/53.2/46.1 tok/s -- **28% between K=3 and K=12** -- but the only thing that
  moved there was the *captured* K, not the extent.

  So adaptation means several captured graph profiles selected per round. Profiles are already keyed
  on (batch size, frontier bucket); adding a K dimension multiplies the captured set against a fixed
  graph allowance (128 MiB at `-p 512 -n 8`, 288 MiB with more shapes). And at C>1 the lanes share
  one graph, so they must agree on a K -- the max wastes the low-acceptance lanes, the min wastes the
  high -- which erodes the benefit exactly as concurrency rises.

  **Prize, honestly**: perfect *per-prompt* selection is only +1.5% over fixed K=7 on the three
  prompts swept, because 7 is already optimal for two of them. Per-*round* adaptation should be worth
  more, since acceptance varies within a generation and the low-acceptance spread is 28%, but that is
  an extrapolation rather than a measurement.

  **Cheapest version worth trying first**: two profiles, not N -- capture at K=3 and K=7, switch on
  one acceptance threshold. The curve is flat in the middle, so two points take most of the spread
  for one extra graph and one bit of per-sequence state.

**2. Copy drafting on the DFlash2 path -- in progress.**
The MTP-only `--lookup-ngram` is removed: it verified at the head's own width, so it could never
widen a round, and measured equal to off. Its replacement is n-gram copy drafting (indexed
proposer and tool-result de-numbering ported from Infernix), which pays only by verifying a copy
*wider* than the neural window -- at K=7 acceptance is 0.95, which caps a perfect same-width drafter
at **+19%**, so the width is the prize, not the source of the drafts.

## Open after the 2026-09-17 upstream catch-up (v3 artifacts, `src/models/qwen3_5`)

The fork now sits on Neroued/ninfer `f76e19c0`. Everything below this section predates that merge;
what it says about kernels and measurements still holds except where this section contradicts it.

- [x] **The plain `linear` Q4/Q5/Q8 shape tables are swept and retuned on sm_86.** They were
      upstream's RTX 5090 sweep and **every one of the eighteen shapes measured here was wrong**,
      by between 1.2x and 3.8x at some width. `bench/ops/linear_schedule_bench.cu` is the sweep --
      the fourth user of `bench/ops/schedule_sweep.cuh`, and the first for an Op whose tables return
      a plain launch pointer rather than a schedule enum, so `routed_to` is recovered by matching
      the shape table's own pointer against the candidate set. Cold, L2 flushed, median of 11 and
      then a second independent run at median of 21-31 with min..p95; a band moved only where both
      runs agreed on the sign and the margin cleared the spread. Best speedup per shape, public
      `linear()` before vs after:

      | shape | best | shape | best | shape | best |
      |---|---|---|---|---|---|
      | q4 131072x5120 | **3.84x** (T=12) | q8 34816x5120 | **2.72x** (T=56) | q4 1024x5120 | 1.95x (T=24) |
      | q8 6144x5120 | 1.85x (T=56) | q4 5120x6144 | 1.72x (T=128) | q5 5120x6144 | 1.67x (T=160) |
      | q4 7168x5120 | 1.66x (T=12) | q4 34816x5120 | 1.63x (T=12) | q4 6144x5120 | 1.61x (T=12) |
      | q8 2048x16384 | 1.60x (T=128) | q5 5120x17408 | 1.60x (T=160) | q8 5120x10240 | 1.53x (T=64) |
      | q8 5120x6144 | 1.53x (T=64) | q5 7168x5120 | 1.51x (T=112) | q8 14336x5120 | 1.46x (T=160) |
      | q8 5120x25600 | 1.44x (T=96) | q5 1024x5120 | 1.42x (T=1024) | q8 5120x17408 | 1.39x (T=32) |
      | q4 4096x5120 | 1.37x (T=256) | q5 6144x5120 | 1.30x (T=112) | | |

      Four corrections recur and are worth knowing before touching any other inherited table:

      * **The draft head was the worst route in the registry.** `q4 131072x5120` sent everything
        above eight columns to the 128-wide tile; at the widths a DFlash2 or MTP round actually
        uses that is 1.7-3.8x the right tile. Nothing about it is specific to the draft head --
        it is what a single wide tile costs when the extent does not fill it.
      * **`ca` beats `cg` for staged activations on sm_86.** Upstream's 028eb61e moved the Q8
        K-split activation loads to `cg`; measured here that is backwards by 11-19% at every
        capacity that was flipped. The L1 `cg` bypasses is where the staged slab wants to live.
      * **Eight K warps fit sm_86's 49,152 bytes up to a 32-column tile**, so `Q8KSplitSm8xFourWarpSchedule`'s
        blanket four-warp fallback gave away 20-26% on the narrow Q8 rungs of the tall shapes.
        It is still the right fallback above 32 columns.
      * **Above the capacity ladder, 64-row MMA tiles beat 32-row ones and beat wide K-split rungs.**
        This is the same "too few CTAs" story §2c records for the GDN projection, and it accounts
        for most of the 33..192 band on every shape of both quantizations.

      Each shape file now records what moved, by how much, and which bands are upstream's value
      kept because it measured best here. `tests/ops/linear/test_q{4,5,8}_a16.cpp` gained the new
      route boundaries.
- [ ] **Not yet swept on this card: the Q6, BF16, FP8-A16 and NVFP4-A16 `linear` shape tables**, and
      the Q4/Q5 `1152`-family Vision shapes. `linear_schedule_bench.cu` has no candidate set for
      them yet; adding one is the 40-line file its header promises, and given that eighteen of
      eighteen swept shapes moved, the prior on these is not good. The Q8 vocabulary crossover
      (`248320x5120`) is deliberately excluded -- it is this fork's own measurement and the catch-up
      left it alone.
- [x] **The new Q5 `linear_add` tail route past 513 columns is right on this card too, and is kept.**
      `MmaResidualR64C128Tail` splits a wide extent into whole 512-column waves plus a narrow tail
      routed through the normal table, on the argument that a trailing mostly-empty 128-wide tile is
      billed as a full wave. Swept with `bench/ops/q5_linear_add_schedule_bench.cu`, cold, median of
      11 with min..p95, public Op (the composite) against `mma_r64_c128` (the plain wide launch),
      both k (us):

      | T | 513 | 576 | 640 | 704 | 768 | 896 | 1024 | 1152 | 1280 | 1536 |
      |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
      | k=6144 composite | 610 | 751 | 828 | 877 | 869 | 1070 | 1149 | 1402 | 1440 | 1708 |
      | k=6144 plain | 864 | 865 | 812 | 940 | 897 | 1098 | 1194 | 1404 | 1490 | 1762 |
      | k=17408 composite | 1635 | 2004 | 2209 | 2358 | 2369 | 2862 | 3144 | 3786 | 3912 | 4682 |
      | k=17408 plain | 2330 | 2378 | 2189 | 2585 | 2478 | 2998 | 3239 | 3717 | 4036 | 4804 |

      **1.42x at 513 columns** on both k, 1.1-1.2x through 704, and 1.0-1.05x beyond. The two points
      where the composite is 1-2% behind (640 and k=17408's 1152) are inside the spread. This is the
      one inherited table in this pass that measured right as it shipped; do not re-sweep it.
- [x] **The new dense (5120-row) `linear_add` Q8 and Q4 tables are swept; k=6144 is retuned and
      k=17408 turns out to be right as it shipped, for a reason nobody had written down.** The Q8
      table was two entries -- K-split capacity to 64 columns, then the grouped split-K for
      everything above -- where the same Op's 2048-row tables are thirty-three. At **k=6144** the
      grouped route is about 2x slower than a plain MMA tile at every width it covers: +74% at
      T=80, +135% at T=192, +110% at T=256, +123% at T=1024; and the capacity route's ceiling is 32
      columns, not 64. At **k=17408** the same tiles measure 1.6-2.5x faster and **every one of
      them is numerically wrong**, so that table is unchanged. The Q4 dense table repeats the
      plain-`linear` finding at the same geometry -- the 32-row tiles lose 21-76% from 65 columns
      up, and 9..16 wants the capacity-24 rung (+43% at T=12). `bench/ops/dense_linear_add_schedule_bench.cu`
      is the sweep. **k=6144's T=128 was mis-routed** and is fixed: re-measured twice with the
      bench's row-tile predicate corrected, `mma_r32_c128` is 170-173 us there and `mma_r64_c64`
      150, so 128 is now its own entry; 112..124 keep the shipped order (154 vs 160, 162 vs 163).
- [x] **The Q8 tiled path at K=17408 was not wrong; it was rounding the dequantized weight, and the
      tiles now have the option not to.** The failure was real and reproducible -- index 297,
      actual -33.25 against reference -33.5091, at every width from 33 up, identically for C64,
      C96, C112 and C128 -- but the criterion it broke is `relative_l2`, at 1.15 of the limit, with
      the gross bound never above 0.56. That is the shape of a precision budget, not of a dropped
      K-tile or a wrapped offset, either of which lands on the gross bound first.

      **Root cause.** `src/ops/linear/q8/q8_rowsplit_gemm_mma.cuh`, `dequant_w`:
      `__floats2bfloat162_rn(q0 * scale, q1 * scale)` stores `round_bf16(code * scale)` as the MMA
      operand. The code needs eight significand bits and an FP16 group scale up to eleven; BF16 has
      eight, so up to eleven bits of every weight are discarded. That error does not average out
      over K on this weight population -- it grows about linearly in K while the dot product grows
      like sqrt(K) -- so the relative error grows like sqrt(K): 0.57 of the criterion at k=6144,
      1.15 at k=17408. Every other Q8 route decodes exactly instead (`q8_ksplit_bf16_pair_from_s8`
      holds the bare code, which BF16 represents exactly, and `fmaf`s the FP32 group partial by the
      scale), which is why the K-split and grouped routes sit at 0.43 at the same K and why the
      shipped table was right. **Confirmed** by rounding the oracle's weights to BF16: the tiles
      fall from 1.15 to 0.42 and the K-split routes rise from 0.43 to 1.15, an exact mirror.

      **Fix.** `EXACT_GROUP_SCALE_`, a new schedule parameter, defaults off, so no existing
      instantiation changes by a bit. Set, the tile carries the bare code, accumulates one Q8G32
      group (two m16n8k16 steps) into an FP32 partial and `fmaf`s it into the accumulator by the
      row's group scale, exactly as the K-split family does. The k=17408 dense `linear_add` table
      takes the exact tiles and measures 0.42 across every width, continuous with the K-split
      routes below it -- no step at the route boundary.

      **Cost.** The exact body holds an FP32 group partial and this tile's row scales, so ptxas
      wants more registers; `with_min_blocks` pins the three tiles that would otherwise lose a
      resident block (r32_c64, r32_c96, r64_c64) back to their default twin's occupancy. Nothing
      spills, and all 42 default instantiations keep their exact register and stack counts
      (`cuobjdump -res-usage`, before against after), so no other Op that shares this header moves.

      **Per-op, retuned** (sm_86, cold, median of 15, `dense_linear_add_schedule_bench`, us, new
      route against the grouped split-K it replaces): T=65 398 vs 681 (1.71x), T=128 445 vs 829
      (1.86x), T=192 578 vs 1317 (2.28x), T=224 801 vs 1419 (1.77x), T=256 887 vs 1551 (1.75x),
      T=512 1755 vs 3227 (1.84x), T=1024 3648 vs 6913 (1.89x). The capacity route's ceiling drops
      from 64 to 40, where the two cross.
- [ ] **The 1.7-2.3x above is worth nothing on today's artifacts, because none of them quantize the
      dense down projection to Q8 -- and the earlier write-up of it as "the 27B MLP
      down-projection's Q8 shape" was wrong about that.** Reading the manifests: in
      `qwen3_8_27b_dflash2.v3.ninfer` all sixty-four `text/layers/*/mlp/down` are `q5_g64_fp16` at
      [5120, 17408]; the only `q8_g32_fp16` tensors at that shape are `mtp/layer/mlp/down` and the
      five `dflash2/layers/*/mlp/down`. `qwen3_6_27b.v3.ninfer` has exactly one (the MTP layer) and
      `qwen3_6_27b_nvfp4.v3.ninfer`'s text tower is NVFP4. The draft heads run at draft widths, and
      every width up to 40 keeps the K-split capacity rung this change did not touch.

      Measured accordingly and it is a wash, which is the right answer rather than a disappointing
      one: `run_interleaved_ab.py`, arm order swapped every repetition, paired medians of 4,
      `ninfer_bench -pg 2048,128 -r 2 --warmup 1 --kv-dtype int8 --max-ctx 4096` on
      `qwen3_8_27b_dflash2.v3.ninfer`, both arms one commit apart in `src/ops/linear_add/q8` only:

      | metric | median | min..max | positive |
      |---|---:|---|---:|
      | 27B prefill (pp2048) | -0.22% | -0.89%..+0.17% | 2/4 |
      | 27B dense decode (tg128) | +0.16% | -0.24%..+0.36% | 3/4 |

      Decode is the control here by construction: T=1 resolves to the same K-split capacity route in
      both arms, so anything it shows is drift, and it shows 0.16%.

      **So the win is banked, not spent.** It is claimed the moment a conversion puts the dense down
      projection in Q8 -- which the fork's `q8_linear_add_admits` has always been ready for and the
      route table now serves correctly. Before quoting the 1.7-2.3x anywhere, say which artifact it
      would apply to.
- [ ] **Two more dense k=6144 widths are mis-routed, and the pattern says there are more between
      them.** Measured 2026-09-18 alongside the T=128 fix, same conditions (us, shipped route vs
      best): **T=320** `mma_r64_c128` 434 against `mma_r128_c80` 343 (**-21%**) and `mma_r64_c64`
      367; **T=448** `mma_r64_c128` 552 against `mma_r64_c112` 514 (-7%). Both are widths that are
      *not* a multiple of 128, so the 128-wide tile pays for a half-empty trailing column tile --
      the same effect the Q5 composite note describes one level up, and the same reason T=384, 512,
      768 and 1024 (all multiples of 128 or close) keep `r64_c128` as the winner. Left alone rather
      than fitted to two points: 272, 288, 304, 336..368, 400..432 and 464..496 were never sampled,
      and a band table built from two measurements is how the T=128 miss got there in the first
      place. Sweep the 257..512 range at a 16-column stride before touching it.
- [x] **The same BF16 dequantization is live in plain `linear`, it is within budget at every shape
      the artifact carries, and the suite can now see it.** The tiles were measured, not assumed,
      and the answer is different from `linear_add`'s: the worst Q8 A16 route spends **0.75** of the
      relative-L2 criterion, not 1.15, and **no route changed**. `::with_exact_group_scale` stays
      unset in `linear`.

      **The coverage gap was real and is closed.** `tests/ops/linear/linear_test_common.cpp`
      compared `Comparison::Sampled`: thirty-two sampled rows against **thirty-two sampled columns**
      per invocation, at most 1,024 elements no matter how wide the call. It is now
      `Comparison::SampledRows` -- the same rows against *every* column, up to 32,768 elements --
      and it is *cheaper* than what it replaced, because the oracle is evaluated once per shape at
      the widest invocation and sliced by the narrower ones instead of being recomputed per call.
      `ctest -R linear -j2` is 63 s for twenty-five tests and every qtype in it stays green.

      **Sampling columns was not, in the end, what hid this.** Widening it moved the k=17408 tile
      ratio from a scattered 0.60-0.66 to a flat 0.61; the step at the route boundary -- 0.42 on the
      K-split rung, 0.61 on the tile above it -- was in the sampled numbers all along. Nobody had
      read them, because nothing failed. The widening is still the right change: it is what makes
      the step legible as a step rather than as scatter, and it is what a *future* decode error will
      trip on.

      The other half of the change is `ActivationSigns`, because the size of this particular error
      turns out to depend on the fixture's activation as much as on the kernel. `linear`'s A16
      activation is centered on zero, which lets a per-weight error cancel along K at the same rate
      as the dot product it perturbs; `linear_add`'s is all-positive, which does not. Every Q8 shape
      the artifact carries is now run both ways. They agree within 0.02 of the criterion on nine of
      the ten; on [14336, 5120] the one-sign fixture reads 0.75 where the centered one reads 0.60.

      **Measured** (sm_86, `NINFER_OP_REPORT_STATS=1`, worst relative-L2 ratio over T=1..1024, every
      column; "K-split" is the rung below the first tiled band, "tile" the bands above it, and
      "excess" is the tile's own contribution, sqrt(tile^2 - ksplit^2), all in units of the 2^-8
      criterion):

      | shape | tensors | K-split | tile | excess |
      |---|---|---:|---:|---:|
      | [14336, 5120] | `mtp/layer/attention/query_key_gate_value` | 0.43 | **0.75** | 0.61 |
      | [34816, 5120] | 6x `mlp/gate_up` | 0.43 | 0.69 | 0.53 |
      | [6144, 5120] | 5x dflash2 `attention/query_key_value` | 0.43 | 0.64 | 0.47 |
      | [248320, 5120] | `text/output_head`, `text/token_embedding` | 0.46 | 0.63 | 0.43 |
      | [4608, 4608] | `vision/merger/fc1` | 0.43 | 0.61 | 0.44 |
      | [5120, 25600] | `dflash2/feature_projection` | 0.43 | 0.61 | 0.42 |
      | [5120, 17408] | 6x `mlp/down` | 0.44 | 0.60 | 0.40 |
      | [5120, 10240] | `mtp/input_projection` | 0.43 | 0.57 | 0.36 |
      | [5120, 6144] | `mtp/layer/attention/output` | 0.43 | 0.52 | 0.29 |
      | [5120, 4608] | `vision/merger/fc2` | 0.44 | 0.53 | 0.29 |

      Worst gross ratio across these shapes: 0.52. The centered fixture is within 0.02 of these everywhere
      except [14336, 5120], where it reads 0.60 against the biased 0.75.

      **The excess does not grow with K**, which is the part of `linear_add`'s write-up that does
      *not* generalize. It is 0.40 at K=17408 and 0.61 at K=5120 -- if anything the wrong way round.
      An offline model of the same rounding (random int8 codes, the fixture's four 1.0625*2^e
      scales, K from 5,120 to 51,200, both sign conventions) puts it at 0.40-0.43 and flat. So the
      1.15 that `linear_add` measured at k=17408 is a resonance between *that* fixture's hashed code
      pattern and its activation, not a K law. **Do not carry 1.15 to a third Op; re-measure.**

      **So the entry closes with the routes as they shipped.** Nothing changed, so there was nothing
      to weigh against DFlash2 or MTP acceptance -- and it would have measured nothing anyway: at
      `--draft-tokens 3` or 4 every one of these tensors runs at T<=5 per decode step, which every
      table routes to the K-split rung that decodes exactly. The tiles are reached only by prefill.
      What is
      worth knowing is that [14336, 5120] leaves only a quarter of the criterion in hand, and that
      `text/output_head` is the one Q8 tensor on the target's own output path, where an error is not
      absorbed by draft verification. If either moves, `::with_exact_group_scale` plus a
      `with_min_blocks` is built and waiting; price the throughput before taking it.
- [ ] **Trap to know before extending a schedule sweep.** Three Q8 `linear_add` launches -- decode,
      exact-T split-K, medium split-K -- hardcode `kRows = 2048` (`q8_linear_add_gemm_splitk.cu`),
      so at 5120 rows they compute the first 2,048 and return. The first dense sweep read that as a
      2-8x win and it was entirely fictional. What caught it was the memory floor, not the code: a
      5120x17408 Q8 weight is 89 MB and cannot be streamed in the 19.5 us those kernels appeared to
      take. `schedule_sweep.cuh` documents the same failure for column domains; this is the
      shape-domain version, and the bench now excludes them explicitly. **Sanity-check any new
      schedule-sweep winner against bytes / 854 GB/s before believing it.**
- [ ] **Not yet swept: the wider Q8 K-split cross product.** The sweep offered each capacity with
      `ca`/`cg` activations, and eight K warps only at capacities 8 and 16. Eight warps also fit at
      24 and 32 and won wherever it was offered, and staging (`ActiveOnly` / `RuntimeActive` /
      `PaddedZero`) was only sampled. Three shapes still show their shipped route beating every
      swept candidate at T=17..24, which is the signature of a rung the sweep did not offer.
- [x] **End to end, the catch-up plus this retune is a wash on the published workloads -- no
      regression from the merge, and no visible gain from the retune.** Pre-merge tip `42a24f21`
      (reading the v2 artifacts) against the retuned branch (reading the v3 ones, identical weight
      bytes), interleaved by `tools/bench/run_interleaved_ab.py` with the arm order swapped every
      repetition, paired medians of 4 repetitions, `ninfer_bench -pg 2048,128 -r 2 --warmup 1
      --kv-dtype int8 --max-ctx 4096`. Post-merge relative to pre-merge:

      | configuration | median | min..max | positive |
      |---|---:|---|---:|
      | 27B plain decode | -0.05% | -0.66%..+0.72% | 2/4 |
      | 27B MTP3 decode | +0.25% | +0.12%..+0.95% | 4/4 |
      | 27B DFlash2 k=3 decode | +0.16% | -0.08%..+0.25% | 3/4 |
      | 27B DFlash2 k=4 decode | +0.03% | -0.09%..+0.11% | 2/4 |
      | 35B-A3B plain decode | +0.22% | -0.87%..+0.24% | 3/4 |
      | 27B prefill (pp2048) | -0.14% | -0.58%..+0.25% | 2/4 |
      | 35B-A3B prefill (pp2048) | +0.95% | -0.21%..+1.63% | 3/4 |

      Every one of those is inside this box's own 3-5% between-process spread. **The merge cost
      nothing measurable** -- which is the question that mattered, since the catch-up replaced a
      great many kernels.
- [x] **Quality is unchanged, to the digits the docs publish.** `ninfer-perplexity` on the quick
      corpus (`ninfer-ppl-1m-v1`, 4,096/2,048 context/stride, INT8 KV, 261,167 tokens over 124
      windows), both arms, 27B DFlash2:

      | domain | pre-merge (v2) | post-merge + retune (v3) |
      |---|---:|---:|
      | chinese_reference | 5.003578 | 5.003578 |
      | english_long_form | 6.892655 | 6.892655 |
      | english_reference | 6.191690 | 6.191690 |
      | ninfer_code | 1.652672 | 1.652672 |
      | **overall** | **4.342425** | **4.342425** |

      Identical in every domain, and `docs/performance.md` publishes 4.342425 for exactly this
      configuration. The catch-up replaced a great many kernels and this retune moved thirty-odd
      route bands; neither changed a digit.

- [ ] **Why the retune does not show here, and what would show it.** `ops::linear` is not on the
      27B or 35B hot path at these shapes. The dense FFN calls it **only on the MTP branch**
      (`src/models/qwen3_5/execution/ffn.cpp`); the main path uses fused `linear_swiglu` and
      `linear_add`, whose tables the catch-up left at this fork's own sm_86 values. What plain
      `linear` does carry at decode is the vocabulary head (`248320x5120`), which is deliberately
      outside this sweep, and the Vision `1152`-family, which is not exercised by these runs. So
      the 1.2-3.8x the sweep measured is real at the widths it measured and simply is not reached
      by `-pg 2048,128` on either model.

      Three things would make it visible, in increasing order of effort: a **DFlash2 run at a
      larger draft count or a serving cohort**, where the Q4 draft head (`131072x5120`) leaves the
      capacity ladder and enters the 12..64 band that was 1.7-3.8x wrong; an **MTP configuration**,
      which is the one path that routes the dense FFN through plain `linear`; and the **35B-A3B
      prefill**, the only figure above that is positive at all (+0.95%) and the one whose shapes
      (`2048x16384`, `5120x25600`, `5120x10240`) this sweep moved most. None of these has been
      measured yet; the first is cheap and is the obvious next step.
- [ ] **Re-measure the RTX 3090 context-cost presets.** They were re-keyed to v3 prefill signatures
      without re-running `ninfer_context_cost_bench`; the coefficients are the 2026-09-14 fits. The
      27B signatures are the ones this box reports for the upgraded artifacts, so a converted-from-
      source artifact with different bindings will fall back to generic coefficients.
- [ ] **A load-time transcode changes nothing in the prefill signature** (it is keyed on the stored
      format), but `--mlp-a8-decode`-style execution choices are not in the signature either. If a
      future preset needs to distinguish them, the signature is the place to say so.
- [x] **The layer pipeline split on real GPUs.** Run 2026-09-21 on rented 2x RTX 3090 and 2x RTX
      A4000 (results in `docs/maintainer/pipeline-parallel-plan.md`): the stage test passes with
      distinct devices (`NINFER_TEST_DEVICE_IDS=0,1`), the split output equals one card's, and the
      A4000 pair runs the 27B at its full context. Still open: heterogeneous pairs (SM count and
      compute-capability rejection), NVLink or peer access, and more than two devices, none of which
      has been on hardware.
- [ ] **A runtime race harness for the boundary transfer has no power on Windows.** Holding the
      destination stream, with a spinning kernel or a blocking host function, also stops the driver
      submitting the source stream's copies, so the source stalls whether or not the fence is
      there (measured both ways). The graph assertion replaces it. On Linux the same test could
      also be run as a real race; worth doing if this ever runs there.
- [ ] **Pipeline stages on distinct GPUs are unverified on the current context engine.** The
      re-port (DFlash/DFlash2 feature layers carried across stage boundaries, per-rank context
      transactions, pause/replay) has only been exercised with `--devices 0,0`; resident vision
      likewise. Overlay vision is parked.
      The default split's per-stage
      overheads are constants, not measured (`default_stage_layers`);
      prefill does not overlap stages (micro-chunk wavefront); tensor parallelism is unbuilt.
      `docs/maintainer/pipeline-parallel-plan.md` is the design of record.

State as of 2026-09-09. Four passes: a profiling pass that closed six items and refuted five of its
own hypotheses, a measurement-hygiene pass that closed three more, a counter pass that put a *cause*
under the four biggest performance entries, and a kernel pass that shipped **two** speedups and
reverted a third. **46 items closed, 13 open.**
If you are picking this up on different hardware, read "Handing this off to another machine" below
before anything else.

Released: **v0.9.0-rtx3090** (Windows + Linux). Full suite **126/126** on this box.

**Four speedups shipped this cycle, and the two biggest came from one measurement: `Mem Pipes Busy`
at 81% while DRAM sat at 50%.** Widening the Q5 and Q4 GEMV consume loops from one code byte per
lane to a 4-byte word — eight weights instead of two, a 4x cut in memory-pipe instructions for
identical arithmetic — is worth **+3.90%** and **+1.95%** on the 27B dense decode, about **+5.9%
compounded**, each 6 of 6 paired repetitions. The q4 SwiGLU pair now runs at **85.3% of DRAM**,
which is the ceiling this file exists to reach; the q5 GEMV became **compute-bound at 73%**, a
different problem than any other entry here. A third widening of the same kind on the q5
dequantization arithmetic — the half2 bit-trick plus a per-group scale hoist, both exact — is
**+2.67%**, and `sparse_moe_d2`'s branch-free sorting network is **+3.15% on the 35B**.

**End to end at depth 4,096 the dense decode went 37.44 → 40.11 tok/s: 69.0% → 73.9% of the
854.2 GB/s the card can actually read.** That is +7.13% throughput and **+4.9 points of the
ceiling**, from the thirteen points §2c identified. Fully closing the remaining 26.1 would be
54.3 tok/s; nothing here suggests that is reachable, and the number exists to stop the next person
quoting 936.2 GB/s. See §2c.

**Combined with the small-T kernels, measured 2026-09-13 once both branches were merged: the same
dense decode is +20.85% over master**, paired median of 6 repetitions, 6/6 positive, spread
+20.25%..+21.58% — about 37.0 → 44.6 tok/s on this box (Windows, 315 W, clocks *unlocked*, so these
absolutes sit above the 1,500 MHz-locked numbers above and only the ratio is comparable). Arms were
two `ninfer_bench` binaries in one directory, interleaved by `run_interleaved_ab.py` with the arm
order swapped every repetition. The same run reports **+48.26%** on an MTP3 round (6/6,
+48.13%..+48.42%) — **do not quote that one**: it runs on `bench/fixtures/bench_corpus.ids`, the
fixture already shown to overstate MTP rounds. The dense figure is the honest headline.

**Two speedups have shipped on the GDN input projection, both on the same boundary, and the second
one came from noticing the first had measured the wrong kernel.** Adding C8 and C16 tiles to the GDN input
projection was worth **+5.7% on the C8 serving profile** and +5.3% on DFlash2 at four draft tokens.
Then instantiating the existing split-K direct kernel past its `T <= 6` throw made *it* the winner at
width 8 — **+2.85% end to end, 6 of 6 paired repetitions, against a control that drifted 0.09%.**
That route table — `{1,6}` independent / `{7,7}` c8 / `{8,8}` independent / `{9,16}` c16 — **is
gone, superseded on 2026-09-13 by the small-T kernels.** Both changes were developed on separate
branches, so neither could be measured against the other until they were merged; in one binary
`small_t_mma` wins at every width 1..32, and at the contested width 8 it costs 82.9 µs against
split4's 167.9 and the c8 tile's 232.4, spreads disjoint. The whole 1..32 range is one route now.
split4-at-8 was a real win over what it replaced and it still reproduces — it is simply beaten 2x.

**One kernel experiment was built, measured and reverted**: splitting the MoE's nine-warp D3 block
into three-warp path blocks, which raised theoretical occupancy to 100% and *dropped* achieved from
67.7% to 52.6%. That closed the MoE entry as a dead end — all four of its candidate fixes are now
measured negatives.

**A shipped artifact does not load on this card.** `models/qwen3_6_27b_nvfp4.ninfer` fails in
runtime planning with `nvfp4 linear_swiglu A16 is registered only through T=16`, on a three-word
prompt — sm_86 forces NVFP4 weights onto the A16 route, and the A16 route has no width above 16.
The only test that reaches it skips unless an env var this file omitted until last cycle is set,
which is why "126/126" and a completely unusable configuration coexisted. `docs/cli.md` uses an
NVFP4 artifact as its worked example. **The obvious fix — project then `silu_mul`, as the W4A4
baseline does — was built, makes the artifact load, and is ~0.8% off because it rounds the
projection to BF16 before the SiLU; reverted.** See §1.

**The second speedup came from a single-warp kernel nobody had looked at.**
`sparse_moe_d2_warp_kernel` picks the top 8 of 257 experts in `<<<1, 32>>>` — one warp, one SM of
82 — and `nsys` put it at **8.45% of the 35B's decode kernel time**, third of the four MoE stages.
Its stall breakdown (`wait` 37.7%, `branch_resolving` 15.3%, `long_scoreboard` only 4.1%) said
divergence and exposed latency, not memory. Replacing the divergent insertion sort with a
branch-free 19-compare network took it **13.43 → 8.32 µs, −38%**, worth **+3.15% end to end on the
35B, 6 of 6 paired reps against a control that drifted 0.00%.** The `wait` half remains, and §2c now
carries a costed design for it — worth ~2%, and *not* the naive 8-warp split, which would
parallelise the cheaper half of the routine and measure as almost nothing. See §2c.

**Read this next: almost nothing here is bandwidth-bound, and this file spent a cycle assuming
otherwise.** `ncu` counters on five kernels, all on one instrument with clocks locked, say the same
thing — the kernels that matter reach 20-50% of DRAM while saturating something else or nothing at
all:

| kernel | DRAM % | highest pipe | achieved occupancy | eligible warps/sched | actually limited by |
|---|---:|---|---:|---:|---|
| `rowsplit_grouped_mma` C8 (GDN, w8) | **27.7** | 54.3 L1/TEX | **24.5** of 50 | **0.55** | too few warps exist |
| `q4a8_swiglu` (prefill MLP) | **20.6** | 61.1 L1/TEX | 33.3 of 33.3 | 0.78 | 124 registers/thread |
| `q5_rowsplit_gemv` (dense decode) | 50.1 → 54.3 | 86.6 → **36.7** L1/TEX | 57.6 | — | **fixed: +3.90%** |
| `sparse_moe_d3` | 43.2 | 34.7 L1/TEX | 61.1 of 93.8 | 1.36 | nothing — latency |
| `sparse_moe_d4` | 47.1 | 42.4 L1/TEX | 81.4 of 93.8 | 1.16 | nothing — latency |

Three consequences worth having before touching any kernel. **A "% of achievable bandwidth" figure
computed as bytes ÷ nsys duration told us the wrong thing four times**; the MoE at "40-45% against
contiguous kernels at 78-82%" is really 43-47% against 50% on one instrument, so the gap is a few
points, not thirty-five. **Narrowing a tile cannot fix a parallelism shortfall** — total warp-work
is `(rows/WM) x (BN/WN)`, which `BM` does not appear in and which narrowing `BN` actively *reduces*;
that is why three separate tile experiments lost. And **the fixes are now specific**: split-K for
the GDN projection (a working one exists for the W8 variant of the same Op), a register cut for the
prefill MLP, and wider per-lane decode in the Q5 GEMV consume loop.

**The five biggest opportunities are all now located rather than suspected**, and three of them
turned out to be the opposite of what this file assumed:

- The **MoE expert gather** is latency-bound, and it is now **closed as a dead end** — all four
  candidate fixes measured, none worked. 31.1 of 32 threads per warp are active, so there is no
  divergence to fix; neither tiling nor batching helps, because the kernel is 1.17 machine-fulls of
  work and an MoE with per-token routing does not amortise its routed weight read across a cohort
  (8 tokens touch ~57 distinct experts, not 8); and **splitting the nine-warp block made it worse**,
  raising theoretical occupancy to 100% while dropping achieved from 67.7% to 52.6%. The 35B's ~51%
  of achievable is what top-8-of-256 costs, not a bug.
- The **KV decode falloff** is 3.21x the per-key cost in the decode attention kernel — 5.89 ns
  against int8's 1.83 ns — and it belongs to the three formats that stage dequantized tiles in
  shared memory. `KeyBlock` was not the difference.
- **Prefill's MLP GEMMs** at 31% of INT8 peak are not limited by memory (compute-bound over their
  own memory floor by 4.3x), tile shape, or dequantization (1-3% of the call). The counters arrived
  and the answer is **occupancy**: 124 registers per thread hold the SM to 16 of 48 warps whatever
  the block shape, `ncu` puts 40% on fixing it, and DRAM at 20.6% means the intensity it costs to
  cut registers is affordable.
- **Eight lanes** buy 3.83x, and the gap is a kernel that launches **0.26 machine-fulls of warps**.
  Two narrower tiles were built for it and both lost, necessarily: `BM` is absent from the warp
  count and narrowing `BN` reduces it. **Split-K is the lever** — confirmed by shipping the cheap
  half of it (+2.85% at the C8 cohort's own extent, from a split-K *SIMT* kernel that was already
  in the tree and throwing above `T = 6`). The remaining and larger prize is a split-K **MMA**
  kernel, which keeps the tensor-core efficiency the direct path lacks; the W8 variant of the same
  Op has routed narrow widths through one all along.
- The **315 W power cap costs nothing** — +35 W buys +3% SM clock and 0% throughput, because memory
  clock never leaves 9,501 MHz at either limit.

**The three measurement-debt items that needed no GPU are closed, and two of them were misdiagnosed
in this very file.** `tools/bench/run_interleaved_ab.py` is the committed interleaved A/B harness
every comparison here has been hand-rolling. `Get-FileHash` was blamed for an empty hash column and
is innocent — it exists here and works; what it does on a missing path is raise a *non-terminating*
error and return nothing, which no `try/catch` catches. And the 24 GiB host-RAM guard is now shared
by every sweep via `scripts/sweeps/host-memory.ps1`, sized from the artifact instead of hardcoded.

**A third piece of tooling looked broken this pass and also was not.** `ncu`'s default kernel replay
mirrors the device working set into *host* RAM, and 21 GB of resident weights plus the artifact does
not fit beside `vmmemWSL` — it dies as `bad allocation` with an empty report, which reads like an
`ncu` bug. `--replay-mode application` needs no backup at all and is now `admin-profile.ps1`'s
default. That is the third time this cycle a tool was called broken and turned out to be a
permission, a shadowed copy, or a resource limit; check those three before believing the fourth.

Two pieces of tooling were believed broken and were not: `compute-sanitizer` (two copies installed,
and the older one reports `0 errors` without executing the binary) and `ncu` (a permission, not a
missing file). Both now work; see §3.

Every route table in the tree has been measured on sm_86 and every one of them has a schedule
bench. What remains is four kernel problems, some measurement debt, and two items needing hardware
or artifacts this box does not have.

---

## Split-K inside the CTA is the answer to the GDN kernel, and NOT to the prefill MLP

Written 2026-09-10 after reading both kernels, because the same shape resolves the GDN parallelism
problem *and* the prefill MLP register problem, and neither entry says so from the other's side.

**They are NOT the same problem, and an earlier version of this section said they were. Correcting
it here because acting on it would waste hours.** `rowsplit_grouped_mma_kernel` reaches 24.5%
achieved occupancy because total warp-work is `(rows/WM) x (BN/WN)` = 1,024 warps, 0.26
machine-fulls — too few warps *in the grid*. `q4a8_swiglu` reaches 33.3% because 124 registers per
thread hold the SM to one block — too many registers *per thread*. Split-K addresses the first and
makes the second **worse**.

**Why it makes the prefill MLP worse.** A K-split warp still owns its whole output tile, just less
of K, so its accumulator count is unchanged; what changes is that there are more warps per block.
Accumulators per *block* are fixed by the block tile (128 x 128 outputs = one float each) while the
non-accumulator overhead scales with thread count, and at 124 registers with 32 accumulators that
overhead is **92 registers per thread**:

| threads | acc/thread | total/thread | registers/block | blocks/SM | warps/SM |
|---:|---:|---:|---:|---:|---:|
| 512 (today) | 32 | 124 | 63,488 | 1 | **16** |
| 768 | 21 | 113 | 86,784 | **0 — does not fit** | — |
| 1024 | 16 | 108 | 110,592 | **0** | — |

So more warps per block is not available at any width: 768 threads already exceeds the register
file. The earlier claim that "768 threads gives 24 warps at <=85 registers" was arithmetic done on
the accumulators alone and ignored the overhead term, which dominates.

**The prefill MLP's real lever is therefore the 92 registers that are not accumulators** — `af`
is 16, `bf` is 16, the two `Stage`s ~14, addressing the rest. Reaching 2 blocks per SM at 512
threads needs <=64 total, i.e. overhead down from 92 to 32, which means fewer staged fragments and
shallower prefetch — a direct trade against the latency hiding those exist to provide. That is a
real tuning problem with a measurable knob, and it is a different one from split-K.

**Split-K remains the answer for the GDN kernel**, where the constraint is the opposite: only 1,024
warps exist in the whole grid, and at 48 registers per thread there is register headroom to spend on
more of them.

**The template is `src/ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu`**, in the tree, for the W8
variant of the same Op, and its route table has sent widths 2..96 through it all along.
`KSplits` warps cover disjoint K ranges, reduce through a shared arena, and the CTA takes 16 rows
instead of 64 — `(12,288 / 16) x 8` = 6,144 warps, 1.56 machine-fulls against the Q4/Q5 tile's 0.26.

**What makes the Q4/Q5 port harder than a copy, having read it:**

- `rowsplit_grouped_mma_kernel` takes **2 or 4 `RowSplitGroupedMmaJob`s** and a
  `RowSplitGroupedMmaCodec` that is `Mixed` for this Op — q4 for the qk rows, q5 for value_z — so
  the K-slice bookkeeping has to be per-codec (`HB` is 1 for Q4 and 8 for Q5).
- It carries **five shared arrays** (`As`, `Bs[S]`, `Cr[S]`, `Hr[S]`, `Sr[S]`) across an `S`-deep
  pipeline; the reduction arena has to fit *beside* them under the 48 KB static cap, and that cap is
  already what forces `BK = 64` (`static_assert(GPB == 1)`).
- The epilogue writes straight to `job.out` with no accumulator buffer, so a K-split needs the
  cross-warp reduction inserted before it rather than bolted after.
- `Jobs == 2 || Jobs == 4` is static_asserted, so the reduction has to be written once and work for
  both.

**So there is no cheap first step on the prefill MLP, and the GDN kernel is where split-K goes.**
Build it there directly, against the W8 template, and accept that the reduction pattern is being
validated on the harder of the two kernels rather than the easier one. The prefill MLP wants a
separate effort aimed at its overhead registers.

**And measure it against the right null.** Forcing occupancy on this kernel the cheap way —
`__launch_bounds__(kThreads, 2)` — cost **69.5%** (1,148.63 to 350.48 tok/s) because the compiler
spilled to reach 64 registers. A K-split that lowers the *natural* register count is a different
thing entirely, but it shares the failure mode: check the compiler's register count and spill
traffic before believing any throughput number.

---

## Handing this off to another machine

Written 2026-09-09 for an agent picking this up on different hardware. The section after this one
("Start here if you are new to this box") is about *this* box; read this one first, because it says
which of that still applies.

### What every number in this file is, and is not

All of it is one machine: **RTX 3090, `sm_86`, 24,576 MiB, CUDA 12.8, MSVC 14.44.35207, Windows 11,
board power capped at 315 W.** Three measured ceilings are used throughout and none of them is a
datasheet figure:

| ceiling | value | note |
|---|---:|---|
| achievable read bandwidth | **854.2 GB/s** | not the advertised 936.1; every roofline here divides by this |
| INT8 MMA | **314.8 TOPS** | `tools/tensor_core_rate_probe.cu` |
| BF16 MMA, f32 accumulate | **67.6 TFLOPS** | same probe |

**Re-measure those three first on new hardware.** Every percentage in §2c is a ratio against them,
and carrying them across cards would repeat exactly the mistake this fork found in upstream's
inherited tables.

**The 315 W cap does not need reproducing.** Measured both ways: at 350 W decode is 37.15 tok/s
against 37.16-37.55 at 315 W, and prefill 1,228.5 against 1,204-1,255 — so +35 W buys +3% SM clock
and 0% throughput. Memory clock never leaves 9,501 MHz at either limit, which is why. See §3.

### What definitely does not transfer

**Every route table in the tree was measured on `sm_86` and is expected to be wrong elsewhere.**
That is not a caveat, it is the main finding of the last two cycles: upstream's tables were
inherited from `sm_120` and were wrong here by up to 52.8%. The tables, each with its own schedule
bench under `bench/ops/`:

| route table | bench | boundaries |
|---|---|---|
| `src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp` | `q4_linear_swiglu_schedule_bench` | 1 / 2..24 / 25..40 / 41..48 / 49..∞ |
| `src/ops/linear_add/q5/q5_linear_add_plan.cpp` | `q5_linear_add_schedule_bench` | 1 / 2..10 / 11..16 / 17..24 / 25.. / 104.. / 128..∞ |
| `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp` | `q4_q5_attn_input_schedule_bench` | see file |
| `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp` | `q4_q5_gdn_input_schedule_bench` | 1..6 / 7..8 / 9..16 / 17..32 / 33..64 / 65..∞ |
| `src/ops/linear_pair/w8/…` (`w8_pair`) | `w8_pair_schedule_bench` | see file |
| DFlash2 w8 | `w8_dflash2_schedule_bench` | see file |

Every one of those benches takes `--tokens`, `--repeat`, `--warmup` and **`--spread`**. Use
`--spread`: it prints `median min..p95` per cell, and a boundary is only decidable when the winner's
p95 clears the runner-up's min. A boundary in §5 went unresolved for a cycle because the harness
printed medians only.

Two rules learned the hard way about those tables, both with worked numbers in the files:

- **Narrowing a tile helps when padding is the cost and hurts when bandwidth is.** Adding C8/C16 to
  the GDN input projection won 9-11% and shipped; the identical change to `q5_linear_add` lost 6-32%,
  because that kernel is already at 24% of its weight-streaming floor and a narrower tile removes no
  work while halving the warps hiding the read.
- **Cold-flush margins overstate the win.** These benches flush 256 MiB through L2 per repetition,
  which is the right default, but a margin measured that way is not a speedup — one 7.5% bench
  margin was 2.4% in situ. §3 has the entry; confirm anything under ~5% with a profile.

### What must be re-established before any of this is reproducible

1. **Model artifacts.** Paths, sizes and which-is-which are in the next section. They are ~18-23 GB
   each and are not in the repo. The real-model tests take them by environment variable; the
   sweeps take `NINFER_MODEL_DIR`.
2. **Build environment.** Windows needs the VS 2022 BuildTools `vcvars64.bat` shell; the recipe is
   in the next section. `cmake --build --target A B` **silently builds only A** on this generator —
   pass one `--target` per name or you will measure a stale binary. That mistake produced a
   "verified" result against a binary four hours older than its source.
3. **GPU performance counters, if you want `ncu`.** On Windows they are administrator-only and
   `ncu` fails with `ERR_NVGPUCTRPERM`. Running from an elevated shell satisfies it per-run with
   nothing persistent and no reboot; `scripts/sweeps/admin-profile.ps1` refuses to start unelevated
   and collects everything that needs it in one pass. The persistent alternative is
   `HKLM\SOFTWARE\NVIDIA Corporation\Global\NvTweak\RmProfilingAdminOnly = 0` plus a reboot.
4. **Clock stability.** See below — this is the single biggest threat to reproducing any A/B here.

### Measurement hygiene, which is where the time actually goes

**This card drifts 3-5% between processes, and that is larger than most effects in this file.** Two
comparisons this cycle came out with opposite-signed drift (+2.9% then −3.8%) on code paths that had
not changed. Consequences, all of them load-bearing:

- **Interleave A/B arms within each repetition**, not all of one then all of the other. Build both
  binaries, alternate them inside the loop, and report the *paired* median. `scripts/` has no
  committed harness for this; the pattern is three lines of shell and it is not optional.
- **Keep a control** — a configuration whose code path did not change between the two arms — and
  normalise against it. In the GDN tile A/B, draft counts 4 and 5 stay on an unchanged route and
  calibrated the drift; without them the result was unreadable.
- **`nvidia-smi -lgc <clock>`** would remove this entirely and needs an elevated shell. It is the
  cheapest measurement improvement available and it is still not done — §3 carries it.
- Two identical-looking runs differing by <3% have measured nothing.

Three more that each cost hours here:

- **`nsys` needs `--cuda-graph-trace node`.** Decode replays a captured CUDA graph and the default
  records the replay as one opaque entity, which reported 99.3% GPU idle. The tell was 807 launches
  over 128 steps of a 64-layer model.
- **`ninfer_bench` has no concurrency option.** Multi-lane numbers come from
  `tools/bench/run_serve_concurrency.py`. An earlier entry claimed otherwise and sent the work the
  wrong way.
- **Never measure speculation on `bench/fixtures/bench_corpus.ids`.** It is 65,536 tokens over 682
  distinct ids with 98.4% of bigrams repeated, and DFlash2 reports *exactly 100% acceptance at every
  draft count from 1 to 12* on it. Any acceptance or tokens-per-round figure from that path is a
  statement about the fixture. Use the serving path on generated prose; `scripts/sweeps/dflash2-draft-tokens-realtext.ps1`
  is the pattern.

And one reassurance, because it saves a whole class of worry: **MMA tile choice does not perturb
perplexity.** Routing width 1024 in `q5_linear_add` from `C128` to `C64` moved the score rate
568.9 → 535.8 tok/s, so a different kernel demonstrably ran, and perplexity came back bit-identical
to twelve figures. Route changes are not a quality risk.

### Tooling traps specific to Windows, all hit this cycle

- **`compute-sanitizer` has two installed copies and one lies.** The 2024.1.0 copy under
  `CUDA/v12.4/` prints its banner and `ERROR SUMMARY: 0 errors`, exits 0, and **never executes the
  binary** — a clean report having checked nothing. The v12.8 copy works, and is what
  `compute-sanitizer` on `PATH` resolves to. Never call it by an absolute v12.4 path.
- **`-k 'regex:a|b'` in PowerShell** is parsed as a pipeline before `ncu` sees it. One
  `--kernel-name` pattern per invocation.
- **`apps/ninfer` writes its logs UTF-16LE.** `grep` finds nothing in them; decode first.
- **`Get-FileHash` is unavailable** under the `powershell` on this box, which silently breaks the
  hashing step of `dflash2-draft-tokens-realtext.ps1` (non-fatal, one error per iteration).
- **Copying a built `.exe` out of `build-ninja/apps/`** breaks DLL resolution: exit 127 with an
  empty log, which reads exactly like a model failure. Keep A/B binaries in that directory under
  different names.
- **Heredocs through this agent harness mangle `\n` and `\v`** inside Python strings. A `\v` in a
  path became a literal vertical tab in committed Markdown. Write files with a file-write tool
  rather than a heredoc when the content contains backslash escapes.

### Work in flight that does not transfer

- **Two `ncu` profiles are outstanding on this box** and a new machine simply re-runs
  `scripts/sweeps/admin-profile.ps1 -SkipPower` there: section 2 (the contiguous-kernel baseline the
  MoE's 40-45% figures are compared against, which failed first time on the PowerShell `|` bug) and
  section 3 (the prefill MLP GEMMs, where memory, tile shape and dequantization are all already
  ruled out and only issue rate / shared-memory feeding / occupancy remain). §2c has both.
- **The perplexity drift bisect is started, and it found two transitions rather than one**:
  −0.0092% then −0.0101%, summing to the recorded −0.0193%. So it is two bisects, not one, and
  both must run `--first-parent` because the commits inside the upstream catch-up merge declare
  `CMAKE_CUDA_ARCHITECTURES=120a` and cannot be built on this card. That cuts the range from 477
  commits to 96. §3 has the measured endpoints and the three eliminated candidates.
- **`ninfer_qwen3_6_27b_score_real_test`** and its siblings need `NINFER_*_WEIGHTS` set or they skip.
  A skip is not a pass, and under `compute-sanitizer` a skip surfaces as
  *"Target application terminated before first instrumented API call"*.

### The fourteen open items, and the next concrete action for each

Ordered by expected value, not by section.

| item | § | next action | needs |
|---|---|---|---|
| MoE expert gather is latency-bound | 2c | **nothing actionable left.** All four candidates measured: geometry no, batching no, block split *worse*, over-fetch real but 1.83x on a 47%-utilised pipe | closed as a negative |
| Prefill MLP GEMMs at ~30% of INT8 peak | 2c | run `admin-profile.ps1` section 3 and read issue rate vs shared-memory feeding vs occupancy | one elevated run |
| Eight lanes buy 3.6x (4.05x with MTP3 since the small-T kernels) | 2c | **not tensor rate** -- `ncu` puts the T=32 gate_up kernel at 38% tensor pipe, 36% L1 and 39% DRAM at 31% occupancy, i.e. latency-bound. A wider tile that puts more work in flight is the untried lever; the 5120-row Q5 shapes still want split-K at T=32 (0-2.5% at narrow T) | kernel work |
| KV decode falloff, 3.21x per-key on fp8 | 2c | attack the per-key dequant-into-shared cost; `rk8v4` proves the floor is reachable without a shared arena | kernel work |
| Eight lanes buy 3.6x (3.83x since the GDN tiles) | 2c | cheap half shipped (+2.85% at width 8); what is left is a split-K **MMA** kernel, modelled on `w8_gdn_input_gemm_splitk.cu` | kernel work |
| KV decode falloff, 3.21x per-key on fp8 | 2c | **counters in**: nothing saturated in either format, and *both* reach only 33% of their theoretical occupancy. The arena is a 1.33x term against a 2.4x gap — measure the grid before touching it | kernel work |
| DFlash2 5→6 cliff | 2c/3 | remainder after the GDN tiles is the grouped kernel sitting at 27% of its own weight-streaming floor | same kernel as the KV item |
| Routed prefill pipeline-depth threshold 7/4 | 2c | **swept**: a 1.33x change in the constant moves prefill 0.06-0.26%, the same as the drift between two identical builds. Safe to leave | closed |
| `w8_pair` medium discards its schedule on sm_86 | 3 | **done**: C48 and C64 instantiated, 13-17% over the chunked loop on the routed `{33,64}` band | closed |
| sm_86 fallback constants chosen to fit | 2c | sweep the alternatives at the eleven `NINFER_SM8X_COMPAT` sites | GPU time only |
| Cold-flush margins overstate wins | 3 | **answered, and the tidy rule is wrong**: three boundaries checked in situ came out −0.78%, +2.85% and +13.59% against +5.9%, +13.2% and +8.0% cold. A narrow margin is a reason to check, not a predictor of direction | closed |
| Perplexity drift 0.019% | 3 | **closed**: `f3f6c724` turned int8 prefill activations on by default (−0.0092%) and `1c12516e` did the rest (−0.0101%). Eight measurements. Only the sub-mechanism inside `1c12516e` is open, and it is one build | closed |
| Speculative decoding not bit-identical to greedy | 3 | decide whether it should be; the divergence is a reduction-order effect in k+1-column verification and MTP reproduces it, so it predates DFlash2 | judgement, not measurement |
| DFlash2 corpus acceptance on real text | 3 | bake a diverse corpus with `make_bench_corpus.py --source-text`, or extend the real-text sweep to report acceptance | a local HF tokenizer, which this box lacks |
| `27b_load_plan` DFlash2 binding matrix | 3 | **half of it can run now**: both groupwise artifacts are on this disk (the "old" one is `models/qwen3_8_27b.ninfer`, SHA-verified). The two NVFP4 ones were never published and would have to be converted locally | artifacts nobody has |
| DFlash2 + pipeline stages | 2 | built and identical to one device with `--devices 0,0`; needs a second GPU to check distinct devices | a second GPU |

Two of those fourteen are hard-blocked on things no amount of work here provides (a second GPU, and
artifacts that no longer exist). One is a judgement call rather than a measurement. The remaining
eleven are all actionable, and four of them are kernel work on two closely related problems:
narrow-extent weight streaming in `q5_linear_add`/`gdn_input`, and dequant-into-shared in the
quantized attention kernels. **The three items that needed no GPU are done** — the A/B harness, the
hash column and the memory guard — so everything left needs either the card or a decision.

---

## Start here if you are new to this box

Written as a handoff. Everything below is state you cannot recover by reading the code or the git
log, and getting it wrong costs hours.

### The models on disk, and which is which

`C:\Ninefer-3090\models\` — 111 GiB total, and only ~38 GiB free on `C:`, so **check free space
before downloading anything**. Nothing here is in git; `models/` is gitignored.

| file | bytes | what it is |
|---|---:|---|
| `qwen3_8_27b.ninfer` | 18,210,531,328 | Qwen3.**8**-27B dense. The default 27B for benchmarks. |
| `qwen3_8_27b_dflash2.ninfer` | 20,437,336,576 | Same model **with the DFlash2 bundle**. Only artifact that can run `--spec dflash2`. |
| `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 | Qwen3.6-35B-A3B MoE at revision `560f227e`, **with DFlash**. The recommended one. |
| `qwen3_6_35b_a3b_v1_no_dflash.ninfer` | 22,373,184,256 | The superseded `c8b8c1c0` revision. **Renamed from a `.pre-dflash.bak` suffix** — the loader rejects anything not ending `.ninfer`, which is why it could not be benchmarked until it was renamed. Kept only to re-run the DFlash-residency comparison; **20.84 GiB reclaimable** if that is not needed again. |
| `qwen3_6_27b.ninfer` | 17,495,365,888 | Qwen3.**6**-27B at `faaa0c14`. A **different model family** from `qwen3_8_27b` — having one does not satisfy tests wanting the other. |
| `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 | NVFP4-*weight* variant of the above. Note this is a weight profile, unrelated to `--kv-dtype nvfp4`. |

The `qwen3_8_` versus `qwen3_6_` prefix is the single easiest thing to get wrong here, and the
failure mode is a test skipping rather than erroring.

### Environment for the real-model tests

```
NINFER_QWEN3_8_27B_WEIGHTS=C:\Ninefer-3090\models\qwen3_8_27b.ninfer
NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS=C:\Ninefer-3090\models\qwen3_8_27b_dflash2.ninfer
NINFER_QWEN3_6_35B_A3B_WEIGHTS=C:\Ninefer-3090\models\qwen3_6_35b_a3b.ninfer
NINFER_QWEN3_6_27B_WEIGHTS=C:\Ninefer-3090\models\qwen3_6_27b.ninfer
NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=C:\Ninefer-3090\models\qwen3_6_27b_nvfp4.ninfer
NINFER_QWEN3_8_27B_OLD_WEIGHTS=C:\Ninefer-3090\models\qwen3_8_27b.ninfer
NINFER_REAL_TEST_MAX_CONTEXT=8192      # only needed for the 35B
```

`NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` was missing from this block for a cycle, and it is the only
thing that was keeping `27b_load_plan` skipping — the artifact has been on the disk all along.

**Setting it costs a different test, so the suite is 125/126 with this block and 126/126 without
it.** `ninfer_qwen3_6_27b_prefix_real_test` passes with the variable unset and dies with
`FATAL: nvfp4 linear_swiglu A16 is registered only through T=16` with it set — verified both ways
on 2026-09-09, exit 0 against exit 1, same binary. So "126/126" in this file means the old env
block; quote the count with the block it was measured under. The defect is §1's new entry.

**Free VRAM used to make these fail outright, not skip.** Skip (exit 77) here is reserved for a
missing artifact env var; a VRAM shortfall instead threw during `Engine` construction and failed
the test. The specific cause was the pinned host-KV allocation being charged against the card (§6,
#45), and with that clamped the six real-model tests pass on an idle box and under `ctest -j2`.
That closes the one source of VRAM-caused failure this fix addresses -- it does not mean free VRAM
no longer affects these tests at all: the model-weights allocation itself can still fail under real
memory pressure from another process. Still worth closing GPU clients before a *measurement* run,
where free VRAM also changes the automatic sizing.

### Recently landed, and what is still owed on it

- **#32 — `docs/config-calculator.html` — merged, but not finished.** Three of the six confirmed
  defects were fixed before it landed and **two were not**; the file itself says so, carrying a
  `KNOWN GAP (see TODO.md)` comment where speculative memory should be modelled. See §2b. So the
  page is live on master, README and `docs/cli.md` link to it, and **it still undercounts any
  speculative configuration by roughly 170 MiB.** Until then, treat its speculative rows as
  optimistic.
- **A second agent works in this checkout.** On 2026-09-09 a *locked* worktree at
  `.claude/worktrees/moonlit-scribbling-sundae` was on `sync/neroued-catchup-20260908` with build
  logs minutes old. **Run `git worktree list` before assuming a branch is free**, `git fetch`
  before pushing, and do not delete a worktree you did not create even when it is holding 46 GB.
- **Do not merge a stack of dependent PRs in one pass.** Seven were opened stacked, each based on
  the one below; merging them bottom-up with `--delete-branch` deleted each base out from under
  the PR above it, and GitHub *closed* three rather than retargeting them. Only the first reached
  master. The recovery is to retarget each branch at `master` and merge one at a time, which works
  because a branch that is a prefix of the chain diffs to exactly its own commit once its
  predecessors have landed.

### Disk, because it stopped a build dead

`C:` filled to **zero bytes free** mid-session and a build failed with
`fatal error C1085: Cannot write compiler generated file ... No space left on device`. Three
things were holding it: the other agent's worktree (46 GB), `build-linux` (60 GB), and
`scripts/models/qwen3_8_27b.ninfer` — a **19 GB byte-identical duplicate** of
`qwen3_8_27b_dflash2.ninfer`, verified by SHA-256, sitting under the downloaders' default output
path and *misnamed*, so anything trusting that path would silently load the DFlash2 artifact.
The duplicate is deleted. Two WSL crash dumps in `%TEMP%\wsl-crashes` were another 12 GB.

### Windows tooling hides Linux breakage, and did

`scripts/run-qwen36-35b-a3b-c1-maxctx.sh` and `scripts/run-qwen38-c1-maxctx.sh` — the two launchers
README recommends as *the* Linux entry point, both shipped in the Linux release archive — had CRLF
committed and **would not parse on Linux at all**:

```
run-qwen36-35b-a3b-c1-maxctx.sh: line 84: syntax error near unexpected token `$'in\r''
```

Inside a `case` block the stray CR joins the `in` token. Those two files are the only ones with a
`case`, which is why only they were fatal while the tree looked healthy. Fixed in #34, with the
policy pinned in `.gitattributes` and a guard added to `check-linux-scripts.sh`.

**The lesson is the part worth keeping.** Every tool on this box reported the tree clean: Git Bash
strips CR on read so `bash -n` passed, `check-linux-scripts.sh` passed, and Git Bash's `grep`
translates CR away so searching for them returned nothing — a loop using it called all 26 shell
scripts clean while two were 141 and 114 CRLF pairs deep. `core.autocrlf=input` was already set and
did not help, because the blobs predate it. It was visible only from **real Linux bash under WSL**,
or by reading bytes with .NET. When something is claimed to work on Linux and has only been checked
here, run it under `wsl -e bash` before believing it.

### Measurement scripts

`scripts/sweeps/` holds the PowerShell that produced every number in §2b, README's KV table and the
calculator. They are committed because §2b asks for re-measurement and reconstructing them is an
hour of work. Read `scripts/sweeps/README.md` first — it records which of them answer which
question, and which one produced a wrong answer and why.

---

## 0. Next up, in this order

Every item in the previous version of this list is now closed. What follows is what a fresh agent
should do first, and the ordering is by expected value rather than by section. The full
seventeen-item table with what each one needs is in "Handing this off to another machine" above.

**Where the open work actually is: §2c (13 items), §3 (1), §2 (1, blocked) and §1 (1).** §2c grew
by five in the 2026-09-10 pass — the five screens in §2c-vii — while §3 shrank from eight to one.
Sections 1, 2b, 4, 5 and 6 are fully closed and say so in their headings — they are kept because
the reasoning is what stops the same investigation being repeated, not because anything is owed
there. The trailing unnumbered subsections after §7 are methodology notes with no items; the one on
editing this file with a script is worth reading before you edit this file with a script.

1. **Screen the decode kernels for the L1/TEX-versus-DRAM gap** (§2c-vii). Three of this cycle's
   six speedups came from that one signature, the fix is an already-merged code pattern, and the
   rest of the kernel family has never been checked. One `ncu` pass, two metrics. This is first
   because it is the only item on the list whose *fix* is already written.
2. **Attribute the missing 26% of decode bandwidth to kernels** (§2c-vii), in the same `ncu` pass,
   with the amplification screen alongside it. Everything below this line is a guess about where
   the shortfall lives until that table exists — including the guesses in §2c.
3. **Run `scripts/sweeps/admin-profile.ps1 -SkipPower` from an elevated shell.** Two `ncu` profiles
   are outstanding and both are cheap: the prefill MLP GEMMs (§2c — memory, tile shape and
   dequantization are all already ruled out, so only issue rate, shared-memory feeding and
   occupancy remain, and those need counters) and the contiguous-kernel baseline the MoE gather's
   40-45% figures are compared against. This is minutes of GPU time and it unblocks the largest
   compute-side item in the file.
4. **Lock clocks for measurement — and always in a `try`/`finally` pair** (§3, elevated):

       nvidia-smi -lgc 1500        # before
       nvidia-smi -rgc             # in a finally / trap, ALWAYS

   The card drifts 3-5% between processes, which is larger than most effects here — it forced
   every A/B this cycle to interleave its two binaries within each repetition and carry an
   unchanged control. It is the cheapest improvement to every measurement in this repository.
   **See "Leave this card as you found it" below before you run it**: an unpaired `-lgc` is a
   persistent change to someone else's hardware, and this file recommended one for weeks.
5. **Two related kernel problems, and they are the real prizes** (§2c). Both are narrow-extent
   weight streaming and both have a measured target:
   - `q5_linear_add`'s `mma_r64_c16` sits at **24%** of what its weight costs to stream once, flat
     from T=4 to T=16. Closing that is most of the eight-lane gap. **Not** by narrowing the tile —
     that was built and lost 6-32%.
   - the quantized attention kernels pay **3.21x** int8's per-key cost (5.89 ns against 1.83 ns),
     which is the whole KV decode falloff. `rk8v4` reaches the flattest curve of all six formats
     with no shared arena at all, so the floor is demonstrably reachable.
6. **Split the shared W8 path out of the MoE's 9-warp block** (§2c). The source already names this
   — `sparse_moe_d3_path_tiled_kernel` exists for it and says so — and it is what the 44.6%/58.6%
   "no eligible warp" reading measures. Do not chase launch geometry or batching first: both were
   measured this cycle and neither helps, because the kernel is 1.17 machine-fulls of work and an
   MoE does not amortise its routed weight read across a cohort.
7. **Sweep the routed prefill pipeline-depth constant through the product** (§2c). `7/4` is
   upstream's RTX 5090 number and an RTX 5090 has 16x this card's L2. Sweeping the constant is the
   method the source comment endorses over the operator fixture, which disagrees with the server by
   6x. GPU time only, no new code.
8. **Confirm `FixedD` is what moved perplexity inside `1c12516e`** (§3) — the one loose end left
   by the drift bisect, which is otherwise **done**: the 0.019% is `f3f6c724` (int8 activations on
   by default, −0.0092%) plus `1c12516e` (−0.0101%). One build with the `rmsnorm.cuh` `FixedD`
   specializations forced off, then re-score. Everything else in that entry is closed.

Two of the sixteen open items are hard-blocked — one on a second GPU (§2), one on NVFP4 artifacts
that were never published rather than deleted (§3, checked upstream 2026-09-10). Everything else is
actionable, and **two of the five new screens need no GPU at all**: the register decomposition is
one build with `-Xptxas -v`, and the parallelism arithmetic is pen and paper. Do the arithmetic
first — it predicts which of the other screens can pay off on which kernel.

**Keep `investigate/small-t-upstream` until the next catch-up.** It is merged, but it is the clean
record of how upstream's small-T was adopted and what had to be fixed (`19c7617c` and its
parents). The next merge from `neroued/master` will touch the same subsystem.

---

## 1. Correctness and coverage — closed except the NVFP4 loadability decision

- [ ] **The NVFP4 27B artifact cannot start at all on sm_86. It is a shipped, documented
      configuration and it is completely broken on this card.** Found 2026-09-09 while confirming
      an unrelated change against the full suite; not a regression — nothing this cycle touched
      `nvfp4_linear_swiglu`.

      ```
      $ ninfer.exe models\qwen3_6_27b_nvfp4.ninfer --prompt "Hello there friend" --max-new 4
      starting engine
      error: startup failed | planning runtime | 1.25 ms
      error: nvfp4 linear_swiglu A16 is registered only through T=16
      ```

      **It throws during runtime planning, before a single token is processed**, so this is not "long
      prompts fail" — the artifact cannot be loaded. Reproduced with a 3-word prompt.

      **Why, and it is a two-line collision between two deliberate decisions.**
      `src/targets/qwen3_6_27b/impl/variant.cpp:49` forces NVFP4 weights onto the A16 route on this
      architecture, with a comment that is entirely reasonable:

      > *sm_86 has no FP8 or NVFP4 tensor-core path. Both quantized weight formats are admitted only
      > through their A16 routes, which dequantize the stored codes to BF16 before the MMA.*

      And `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:35` registers that A16 route
      only to T=16, throwing above it. `AllowA4` handles any width (fused to 128, then TMA or a
      split post-pass) — but `AllowA4` is exactly what sm_86 cannot use. **So the policy sm_86 is
      forced onto is the one that has no wide route, and planning a default 1,024-token prefill
      chunk hits the throw immediately.** `nvfp4_gdn_snapshot_plan.cpp:40` carries the same T=16
      bound, so fixing only the SwiGLU would move the failure rather than remove it.

      **Why it went unnoticed, which is the part worth keeping.** The only test that reaches it is
      `27b_prefix_real`, and that test skips unless `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` is set — a
      variable this file's env block omitted until last cycle, when it was added *because* it
      unblocked `27b_load_plan`. Verified both ways on one binary: unset, exit 0; set, exit 1. So
      the suite reads 126/126 on the old block and 125/126 on the current one, and the artifact has
      been unusable on this card the whole time. **A skip is not a pass**, and this is the second
      time this cycle that a missing env var was hiding something rather than merely reducing
      coverage.

      **`docs/cli.md` documents this configuration as a primary example** — an NVFP4 weight
      artifact is the model argument in four of its invocations, and line 208 states that
      "both `groupwise-int` and `nvfp4` artifacts use the same Engine route, including CUDA Graph".
      On sm_86 that is not true of any prompt.

      **The cheap fix was built, measured and reverted — it is ~0.8% off and fails the Op
      criterion.** Worth recording in full, because it is the fix anyone would reach for first and
      it does not work.

      `linear` under `A16Only` already accepts any T (`nvfp4_dispatch.cpp` returns the A16 route
      unconditionally), so the fused kernel's T=16 bound is a limit on *fusion*, not on the maths.
      That suggests routing wide A16 exactly as `LinearW4A4Post` already routes wide W4A4: project
      into a buffer with `linear`, then apply `silu_mul`. Implemented as `LinearA16Post` — about
      thirty lines, no new kernel, sharing the existing baseline block with the policy threaded
      through instead of a hardcoded `AllowA4`.

      **It makes the artifact load.** The three-word prompt that previously died in runtime planning
      ran to completion, and a 98-token prompt prefilled at 28.7 tok/s and decoded at 13.7.
      `27b_prefix_real` went from exit 1 to exit 0. Two workspace details had to be fixed on the way
      and both are worth knowing: the A16 linear reports a *zero-byte* requirement, and zero is not
      representable — `alloc_bytes(0)` throws "arena allocation must be nonzero" and a
      `WorkspaceArena` over an empty span throws "borrowed DeviceArena storage must be non-empty",
      so the request has to be floored at one alignment unit the way
      `variant.cpp`'s `kMinimumLeafWorkspaceBytes` does.

      **And then it fails accuracy at every new width.** `ninfer_linear_swiglu_nvfp4_test` with the
      A16 cases extended past 16 (`{1, 4, 8, 16, 17, 128, 256, 1024}`):

      ```
      LinearSwiGLU NVFP4_A16 T=17   eager: reduction criterion failed at index 11 actual=4.59375 reference=4.55671
      LinearSwiGLU NVFP4_A16 T=128  eager: reduction criterion failed at index 11 actual=4.59375 reference=4.55671
      LinearSwiGLU NVFP4_A16 T=256  eager: reduction criterion failed at index 11 actual=4.59375 reference=4.55671
      LinearSwiGLU NVFP4_A16 T=1024 eager: reduction criterion failed at index 11 actual=4.59375 reference=4.55671
      ```

      T ≤ 16 passes; every width on the new route fails, at the same index, with the same value. That
      is systematic, not noise, and the cause is structural: **the post route rounds the gate and up
      projections to BF16 before the SiLU, where the fused kernel keeps them in registers at FP32.**
      One BF16 rounding of an intermediate is ~0.4% of relative precision and the observed error is
      0.81%, so this is the design's inherent accuracy, not a coding slip. **Reverted** — a route
      that cannot pass its own Op criterion is not a fix, and widening the criterion to admit it
      would be exactly the mistake §4 was written about.

      So the ways out, now with one eliminated by measurement:

      1. ~~Linear-then-`silu_mul` under A16~~ — **built and reverted, ~0.8% error, see above.**
      2. **An FP32 intermediate.** The correct and fast version of (1): have the projection land in
         FP32 and apply SiLU from FP32. Blocked on two things that are both real work —
         `silu_mul` is BF16-only (`src/ops/wrapper/silu_mul.cpp:13` rejects anything else), and the
         NVFP4 A16 `linear` would have to emit an FP32 output. A fused
         `silu_mul_f32_to_bf16(gate_f32, up_f32, out_bf16)` plus an FP32 output path is the shape.
      3. **Chunk the fused A16 kernel over 16-column slices**, the way
         `w8_pair_splitk_medium_launch` does under `NINFER_SM8X_COMPAT`. Numerics identical to the
         passing T ≤ 16 path by construction, no new kernel — but it re-reads the whole MLP weight
         per chunk, so a 1,024-token chunk streams the ~89 MB gate_up 64 times. Correct and
         unusably slow; mentioned only so nobody proposes it as the easy answer.
      4. ~~Reject the artifact at load with a clear message~~ — **the message half is done
         (2026-09-10); the artifact still does not load.** Failing in `planning runtime` with
         "registered only through T=16" gave an operator nothing to act on, and working out why took
         a full-suite run plus a bisect of two deliberate decisions. Both throw sites now say what
         is wrong, why it is architectural, and what to do instead:

         ```
         error: startup failed | planning runtime | 40.3 ms
         error: nvfp4 linear_swiglu: the A16 route is registered only through T=16, and on sm_86
         A16 is the only policy available for NVFP4 weights (no NVFP4 tensor-core path), so this
         artifact cannot serve a prefill chunk wider than 16 columns on this architecture. See
         TODO.md section 1. Use a groupwise-int artifact here, or build for sm_89+ where AllowA4
         handles any width.
         ```

         `nvfp4_gdn_snapshot_plan.cpp` got the same treatment even though it is normally unreachable
         (the SwiGLU throws first), because a reader who lands there has fixed the SwiGLU and needs
         to know this is the next wall.

         **This is deliberately not a load-time rejection, and the item stays open because of it.**
         Whether sm_86 should refuse the artifact outright, or gain the FP32-intermediate route in
         (2), is a product call about a documented configuration rather than a bug fix — it is not
         mine to make. Making the existing failure legible removes nothing, since nothing worked
         before.
      5. Either way, **`docs/cli.md` must stop using an NVFP4 artifact as its worked example on a
         page whose own banner is about this fork's sm_86 target**, and the model table in this file
         should say the artifact does not load here.

      **`nvfp4_gdn_snapshot_plan.cpp:40` carries the same T=16 bound and was never reached**, because
      the SwiGLU throws first. Whichever of (2) or (4) is done must handle it too, or the failure
      simply moves one Op along.

      One measurement note: the extended A16 cases are the coverage that catches all of this, and
      they are *not* in the tree — the committed test stops at 16, which is why a route that cannot
      run on sm_86 at all had a passing Op test. Re-add `{17, 128, 256, 1024}` alongside any fix.

      Do not "fix" this by unsetting the env var. That restores 126/126 and re-hides a broken
      shipped configuration, which is how it survived this long.

      *Not yet checked, and deliberately not claimed:* whether the Qwen3.8 NVFP4 artifact
      `docs/cli.md` actually names is affected. Only `qwen3_6_27b` defines `kNvfp4TextPolicy`, that
      artifact is not on this disk, and there is no `qwen3_8_27b` target directory — so which target
      serves it, and whether that target forces A16Only, is unverified.

### 1.1 `attn_input_proj` grossly wrong at `W8 DFlash2 A16 T=112 graph phase=1` — closed

- [x] **A race in the test harness, not in any kernel.** Closed by #44, 2026-09-09.

      `DeviceBuffer::copy_from_host` uses `cudaMemcpy`, and for a host-to-device copy out of
      *pageable* memory CUDA documents that the call returns once the source has been staged
      for DMA, "but the DMA to final destination may not have completed". That trailing DMA
      completes on the legacy stream, and every stream `DeviceContext` owns is created with
      `cudaStreamNonBlocking`, which is exempt from the legacy stream's implicit ordering. So
      "stage the inputs, then launch on the engine stream" is a race, and it is the sequence
      every Op test's setup uses.

      A faithful replica of exactly what the failing case runs — a 1,146,880-byte pageable
      H2D, three `cudaMemsetAsync` on a non-blocking stream, then a captured graph launched on
      it — read stale bytes in **146 of 200 iterations**, up to 17.8% of the buffer, on an
      otherwise idle GPU. With `cudaStreamSynchronize(nullptr)` after the copy: 0 of 200.
      Copies at or below ~256 KiB never raced; every size above it did.

      Every property of the reported failure follows: only the graph-replay phase, because it
      is the only phase that re-uploads its activation; q, k *and* value at once, because they
      share one input; never in isolation, because an idle box lands the DMA in time; and the
      input verifying clean afterwards, because by then the DMA has long finished.

      `tests/test_device_buffer_visibility.cu` is the reproducer kept as a regression test. It
      reads stale data in roughly 90 of every 100 iterations against an unsynchronized
      `DeviceBuffer`.

      **The part worth keeping.** The window is narrow enough that instrumentation closes it.
      Two probes were built and both were useless for opposite reasons: a D2H read-back on the
      same stream, ordered ahead of the kernel, made the race vanish entirely (the copy engine
      serializes the in-flight H2D behind it), and an early draft of the regression test
      cleared its counter with `DeviceBuffer::fill` between the copy and the stream work —
      one synchronous runtime call in the gap, and 0 races out of 90 where there should have
      been 84. When measuring a race, check that your instrument still shows it on a case you
      know is broken.

### 1.2 Real-model tests — all six now run and pass

- [x] **Re-run on 2026-09-09, idle GPU, serially and under `ctest -j2`.** Full suite **126/126**.

      | test | before | now |
      |---|---|---|
      | `27b_prefix_real` | failed (host-KV pin) | **passes** |
      | `27b_score_real` | failed (`a repeated score window inherited prior State/KV`) | **passes** |
      | `27b_dflash2_real` | failed (host-KV pin) | **passes** |
      | `27b_load_plan` | skipped | **passes** with `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` set |
      | `35b_a3b_real` | failed (host-KV pin) | **passes** |
      | `35b_a3b_dflash_real` | failed (host-KV pin) | **passes** |

      Three separate defects were behind those, and none of them was the model:

      1. **The pinned host KV cache could not be reserved beside any shipped model** (#45, and
         see §6). On Windows a pinned host allocation is charged against VRAM, so the 8 GiB
         default failed with the card full and tens of GiB of system RAM free.
      2. **`score_real` was a real correctness defect** — fp8/nvfp4/k8v4 attention corrupting
         every prefill output column but the last (#49, and see §5).
      3. **`load_plan` crashed silently** rather than reporting, because it had no top-level
         catch (#46), and the environment variable naming its second artifact was missing from
         the handoff block above.

      `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` is the one environment variable the handoff block at
      the top of this file was missing; the artifact has been on disk all along.

      **`27b_load_plan` now pins a hardware bound rather than skipping.** Its NVFP4 case plans
      a 1,024-token prefill chunk, and on sm_86 that cannot exist: NVFP4 weights need A4
      execution, which is sm_100a/sm_120a only, so the policy degrades to `A16Only`, and NVFP4
      A16 `linear_swiglu` is registered through T=16 and no further. The test asserts the
      refusal and its message, so a wider registration would fail it rather than pass silently.

- [x] **The maximum-configuration judgement call, resolved.** The tests pin layouts sized for a
      card this box does not have, and `NINFER_REAL_TEST_MAX_CONTEXT` already lowers the ceiling
      the 35B's `exercise_maximum_configuration` asks for. With the host-KV fix in place the
      *base* engine now fits without closing anything, so the remaining question — whether the
      base config should also honour the env var — is moot: it starts. Left as it is.

### 1.3 `docs/config-calculator.html` undercounts startup memory for speculative configurations — closed

- [x] Closed 2026-09-09 by the same work as §2b's first entry, which has the measured terms, the
      engine cross-checks and the scale of what was missing. The short version: the page reused the
      12 MiB no-speculation CUDA graph allowance for every mode and added no extra cache for any of
      them, and the entry's own guess at the fix — "a full port of `mtp_graph_profiles` /
      `dflash_graph_profiles` / `graph_topology_allowance` into the page's JS, nontrivial and
      piecewise" — turned out not to be needed. Measuring the engine's reported terms at three
      contexts per mode gave five constants per mode, one of which is a two-step function of
      context, and those reproduce the engine to 0.044% on MTP3 and exactly on DFlash-3.

      Worth keeping: the entry descoped this as "out of scope for a docs PR" and was right that
      porting the layout logic would have been. It was wrong that the alternative needed "fresh
      `ninfer_bench` measurements of the actual per-mode graph allowance" as though that were the
      expensive path — the whole matrix is ten minutes of load-and-report, because the engine
      already prints every term.

---

## 2. Genuinely blocked, and what by

This section once read "blocked on hardware or artifacts" and lumped three items together. Two of
those three were not blocked at all, and both have since been closed by going and looking. **Exactly
one open item needs hardware this box does not have.** Check before assuming an entry here is
unreachable — this section has a poor record of being right about that.

### Needs a second GPU — one item, and it is the only one

- [ ] **DFlash2 + pipeline stages.** Feature layers past the first stage now cross back to rank 0
      on their own link, and `--devices 0,0` output matches one device. Checking distinct devices
      needs a second card (`nvidia-smi` reports exactly one device here).

### Needs an artifact we do not have — nothing is left here

Both entries are closed. The suggestion one of them carried — "check whether
`neroued/Qwen3.6-35B-A3B-NInfer` publishes one at another revision before treating it as out of
reach" — was the right instinct and paid off: it did. Kept rather than deleted because the pattern
recurs, and because the follow-on work below only exists now that they are gone.

- [x] **`--spec dflash` (v1) on 35B-A3B** — resolved 2026-09-08. Revision `560f227e` (now `main`)
      carries the DFlash bundle; `c8b8c1c0` predates it. `scripts/download-qwen36-35b-a3b.{sh,bat}`
      and `flake.nix` are pinned to it as of #31. Loading with `--spec dflash --draft-tokens 3`
      reports 21,448,523,264 bytes of weights against 21,038,469,632 with `--spec` unset — exactly
      the 410,053,632-byte on-disk difference between the two revisions, so **carrying DFlash costs
      nothing resident unless it is selected**, and the published 24 GB profiles still apply.
- [x] **The Qwen3.6 27B artifact** — downloaded and pinned at `faaa0c14` (#31), so the four tests
      in §1.2 that wanted it are no longer artifact-blocked.

### Unblocked, and now owed work — closed

- [x] **Re-run the six real-model tests now that both artifacts exist** (§1.2). Done 2026-09-09.
      All six run and pass. `27b_score_real` was indeed hiding a genuine defect rather than an
      environmental skip, and it was worse than the message suggested: not State/KV inheritance
      but a data race corrupting every prefill output column but the last, in three of the six KV
      formats (#49, §5). The other four were failing on the pinned host-KV allocation (#45, §6),
      and `27b_load_plan` needed one environment variable that the handoff block had never listed.

---

## 2b. `docs/config-calculator.html` is advertised as authoritative and is not yet correct — closed

Added by #32, **which merged with two of its six confirmed defects still open**, one of them only partly. README and
`docs/cli.md` link to the page, so those two are live rather than theoretical. Read the checkboxes
rather than assuming a merged PR means a finished page; the file itself agrees, carrying a
`KNOWN GAP (see TODO.md)` comment where speculative memory should be modelled.

Closed before it landed: KV page rounding (with regression tests), sub-4,096 decode (now labelled a lower bound held at the
shallowest measurement rather than claimed as interpolated), and per-row weight-artifact labelling.
The doc contradictions were fixed separately in #33. The evidence for each is kept below so nobody
has to re-derive it.

- [x] **Speculative modes undercount memory by roughly 170 MiB, plus a per-token term.** Closed
      2026-09-09. Every term is now measured per mode by
      `scripts/sweeps/speculative-memory-terms.ps1`, and the undercount was far larger than 170 MiB
      at anything but the mildest mode and depth.

      | model | spec | graph | seq fixed | seq/token | kv ratio | kv extra/token |
      |---|---|---:|---:|---:|---:|---:|
      | 27B | none | 12 MiB | 158.7 MiB | 1/16 | 1 | 0 |
      | 27B | mtp3 (+head) | 86 MiB | 167.6 MiB | 1/8 | 1.06299 | 0 |
      | 27B | dflash2-7 (+head) | **288/384 MiB** | 266.2 MiB | 1/16 | 1 | 0 |
      | 35B | none | 12 MiB | 67.3 MiB | 1/16 | 1 | 0 |
      | 35B | mtp3 (+head) | 86 MiB | 72.6 MiB | 1/8 | 1.10078 | 0 |
      | 35B | dflash-3 (+head) | 160 MiB | 184.2 MiB | 1/8 | 1 | **4,096 B** |

      **The "one constant multiplier" in the old entry only ever fitted MTP.** MTP buffers extra
      draft tokens *in the KV format*, so its cost is multiplicative and identical across formats
      — +6.2988% per token on the 27B, +10.078% on the 35B, both confirmed against nvfp4 to within
      page rounding. DFlash v1 instead reserves a fixed scratch region per token, measured at
      **exactly 4,096 bytes** and format-independent: 32 MiB at 8,192 tokens, 64 MiB at 16,384,
      128 MiB at 32,768. Read as a ratio that is +38.8% on `int8` and +71.1% on `nvfp4`, which is
      what made it look format-dependent. DFlash2 costs nothing per token and pays entirely in a
      graph allowance that is itself a step in context: 288 MiB to 8,192 tokens, 384 MiB beyond.

      **Cross-checked against the engine's own refusal arithmetic, which the old entry noted had
      only ever been done for `--spec` unset:**

      | configuration | engine | page | error |
      |---|---:|---:|---:|
      | 27B int8 262,144, mtp3+head | 9,838,038,272 | 9,842,380,571 | +0.044% |
      | 35B int8 262,144, dflash-3 | 4,312,377,600 | 4,312,377,600 | **exact** |

      For scale, the model the page used before this: 611 MiB short on the first, and **1,289 MiB
      short (−31%)** on the second. "Roughly 170 MiB" was measured on MTP3 at a 40,960 context —
      the mildest combination of mode and depth in the matrix.

      User-visible effect: the page reported 262,144 tokens as the largest fitting context for
      *every* speculation mode on the 35B. DFlash-3 actually tops out at 240,192.

- [x] **KV is allocated in 64-token pages; the page charges exact tokens.** Fixed on #32. `KV page groups
      4096 / 4096` at a 262,144 context is 64 tokens per group. A context just past a page boundary
      reserves a whole further page, so both the memory figure and the largest-context result
      should round to a page. Every context measured so far happens to be page-aligned, which is
      why this never showed up in the validation against the engine's own refusal message.

- [x] **Sub-4,096 decode is reported at the wrong depth.** Fixed on #32, by labelling rather than by measuring — it now reads "lower bound, held at the 4,096-token measurement". Measuring 1,024 and 2,048 would still be better and is cheap. `decodeAtDepth` returns the 4,096-token
      measurement for every context from 256 up, and the UI then labels it `interpolated` when no
      interpolation happened. The honest fix is to measure: 1,024 and 2,048 are cheap, and short
      contexts are exactly where a casual user starts.

- [x] **The model selector does not name the artifact each row was measured against.** Fixed on #32. `27b` means
      `qwen3_8_27b.ninfer` specifically; the runtime has separate load plans and device capacities
      per weight profile, so `weightsBytes` is not transferable to, say, the NVFP4-weight variant.
      Either add the weight-profile dimension or label each row with its artifact.

- [x] **The regression tests exist but nothing runs them.** Closed. #47 put
      `docs/config-calculator.test.mjs` in a GitHub workflow — it needs `node`, which no other test
      here does, so CTest was the wrong home for it. It now also covers the speculative path, which
      the old entry correctly said it did not: three golden cases against the engine's own refusal
      figures (no speculation, MTP3+head, DFlash-3), a check that DFlash's extra KV is a fixed
      4,096 bytes per token rather than a ratio, a check that DFlash2's graph allowance steps with
      context, and an invariant that no speculative mode may cost less than no speculation in
      either fixed or per-token memory — which is precisely the failure the old model had.

      `docs/config-calculator.render.test.mjs` is new and covers what none of that did: `render()`
      itself, over all 360 model x speculation x KV x context combinations, plus an assertion that
      the reported largest-fitting context actually responds to the speculation mode. Most of the
      page's code lives in `render()`, and a wrong property name there shows up only as a blank
      page in a browser.

- [x] **Active docs still contradict the six-format claim.** Fixed in #33, across four places. #32 fixed README's `Current limits`
      and `docs/perplexity.md`, but README's *opening* summary still says the FP8 E4M3 KV profile
      "is not" admitted on SM86, and `docs/cli.md` and `docs/serving.md` were not touched. All six
      formats are measured and working; the Blackwell restriction applies to FP8/NVFP4 *weights and
      activations*, not KV storage. Fix them together so the claim is consistent everywhere.

---

## 2c. Performance left on the table

Ordered by size of the prize. The first three entries were rewritten on 2026-09-09, when the
roofline finally acquired a denominator; read them before the rest.

- [x] **Dense decode runs at 65-66% of the card's memory bandwidth.** Closed by #50 and #51 — the
      observation was right that headroom exists, and wrong in both of its terms.

      *936.2 GB/s is not a ceiling anything reaches.* It is the advertised 384-bit GDDR6X at
      19.5 Gbps. `tools/hbm_bandwidth_probe.cu` was already in the tree and had never been
      pointed at this card. 4 GiB working set (683x L2), best of five: read 854.2 GB/s (91.3% of
      advertised), memset-write 863.3, kernel write 824.1, D2D copy 813.2. Decode streams
      weights and KV *in*, so 854.2 GB/s is its ceiling.

      *Resident is not read.* `weights_capacity_bytes` counts the vision tower, MTP and DFlash
      when unselected, the draft head with speculation off, and all but one row of the token
      embedding — 2.47 GB of the 27B that never moves per token.
      `tools/decode_byte_accounting.py` counts what a token actually touches, from the
      artifact's own object directory: **15.743 GB** on the 27B.

      | model | kv | depth | tok/s | achieved | % achievable |
      |---|---|---:|---:|---:|---:|
      | 27B dense | int8 | 4,096 | 37.44 | 595 GB/s | 69.6% |
      | 27B dense | int8 | 32,768 | 33.56 | 565 GB/s | 66.2% |

      So the dense path sits around two-thirds of what the card can deliver, not 66% of an
      unreachable number. Real headroom, roughly thirteen points.

      **Three points of that headroom are now taken.** Re-measured 2026-09-10 after the two GEMV
      consume-loop widenings, same workload (`-pg 4096,128`, int8 KV, three repetitions, clocks
      locked at 1,500 MHz), using the same 15.743 GB/token accounting:

      | | tok/s | achieved | % achievable |
      |---|---:|---:|---:|
      | start of cycle | 37.44 | 589.4 GB/s | 69.0% |
      | after the two consume-loop widenings | 39.19 ± 0.03 | 617.0 GB/s | 72.2% |
      | **after the dequant bit-trick** | **40.11 ± 0.07** | **631.5 GB/s** | **73.9%** |

      **+7.13% throughput and +4.9 points of the ceiling on the dense decode**, from removing a
      memory-pipe bottleneck that this file had spent a cycle mistaking for bandwidth, and then the
      arithmetic bottleneck that removing it exposed. The ceiling
      itself is 54.3 tok/s if the remaining 27.8 points were ever fully closed, which is the honest
      shape of what is left. Note this depth-4,096 figure (+4.67%) is slightly below the +5.9%
      compounded from the two paired A/Bs; those measured prose generation through the CLI and this
      measures `ninfer_bench` at depth. Both are real and the workloads differ.

- [x] **Nobody knows where the MoE sits at all, because the accounting does not exist.** Closed by
      #50. It exists now, and it changes which entry in this section matters most.

      An A3B MoE reads 1.943 GB of dense text weight per token plus 8 of 256 experts per layer —
      0.580 GB out of 18.556 GB of experts resident — for **2.522 GB per token**. Against the
      854.2 GB/s read ceiling:

      | model | kv | depth | tok/s | achieved | % achievable |
      |---|---|---:|---:|---:|---:|
      | 35B-A3B | int8 | 4,096 | 169.55 | 435 GB/s | 50.9% |
      | 35B-A3B | int8 | 16,384 | 163.39 | 440 GB/s | 51.6% |
      | 35B-A3B | int8 | 32,768 | 153.62 | 441 GB/s | 51.6% |

      **Flat to within a point across an eightfold change in cache depth**, which is a far
      cleaner signal than the dense model's gentle decay: whatever limits the MoE is not the
      cache and not the depth.

      This reverses the ordering this section opened with. The dense path has about thirteen
      points of headroom; **the MoE has about thirty**, on the model the release recommends. A
      gather of 8 scattered expert blocks per layer is the obvious suspect and is where the
      profile should start. `scripts/sweeps/decode-step-profile.ps1` (#51) is the tooling.

- [x] **Profile one decode step.** Done 2026-09-09 (#53), and it says the shortfall is inside the
      kernels rather than between them — on both models, but for different reasons.

      | workload | launches | GPU busy | wall | idle |
      |---|---:|---:|---:|---:|
      | 27B dense decode | 83,623 | 3,336 ms | 3,500 ms | 4.7% |
      | 35B-A3B decode | 57,770 | 717 ms | 849 ms | **15.5%** |
      | 27B dense prefill | 87,110 | 6,512 ms | 6,675 ms | 2.4% |
      | 35B-A3B prefill | 60,715 | 1,362 ms | 1,481 ms | 8.1% |

      Dividing the read set into *busy* time alone: 27B 601 GB/s (70.7% of achievable), 35B
      450 GB/s (52.7%). So closing every launch gap buys 4.7% on the dense path and **15.5%** on
      the MoE, and the rest is kernel efficiency.

      *These are the corrected figures.* #53 reported 4.5% and 12.6%, computed by summing each
      kernel's duration rather than taking the union of their `[start, end)` intervals. Decode
      launches across streams and concurrent kernels overlap, so summing double-counts the overlap
      and understates idle — found by CodePulse review on #51 and fixed in #60. The dense path
      barely moved (1.0% of overlap); the MoE's idle went 12.6% to 15.5%, which makes its launch
      overhead a fifth of its total shortfall rather than a quarter of it. Any future change to
      this script's arithmetic should be checked against synthetic overlapping intervals, which is
      how #60 verified the merge-sweep.

      **On the MoE the shortfall has an address.** Bytes attributed from the artifact inventory,
      instance counts fixing the mapping (129 rounds, 40 text layers, 8 of 256 experts):

      | kernel | bytes | time | achieved | % achievable |
      |---|---:|---:|---:|---:|
      | `sparse_moe_d3` (routed gate_up) | 8.91 MB | 23.4 µs | 381 GB/s | **44.6%** |
      | `sparse_moe_d4` (routed down) | 5.58 MB | 16.4 µs | 340 GB/s | **39.9%** |
      | `w8_k2048_decode` (gdn qkv_z) | 26.74 MB | 38.2 µs | 700 GB/s | 81.9% |
      | `q6_rowsplit_gemm_simt` (output head) | 397.31 MB | 595.6 µs | 667 GB/s | 78.1% |

      The expert kernels reach 40-45% of what the card can read while *contiguous* weight kernels
      on the same step in the same model reach 78-82%. The four `sparse_moe` stages are 38% of
      decode busy time. That is the gather of eight scattered expert blocks per layer, measured
      rather than suspected, and it is the single largest identified opportunity in this file.

      The 27B is a different shape: 86% of its busy time is four GEMV kernels that *are* the weight
      streaming, so its remaining thirty points are inside those.

      **Which 2026-09 speedups reach the 35B, checked 2026-09-13 rather than assumed.** The two
      models share a runtime but almost no kernels, and the difference is codec: the 35B's
      attention input, GDN input and linear_add are all `W8G32_F16S`, its output head is Q6, and
      only its MoE experts are Q4/Q5. So:

      | change | 35B? | why |
      |---|---|---|
      | `sparse_moe` branch-free top-8 network (#88) | **yes** | +3.15% measured on the 35B itself |
      | `--gdn-state-fp16` (#91) | **yes** | shared `layouts_impl.h`; ~+2.8% median, 3 of 4 pairs |
      | small-T Q4/Q5 MMA kernels (#89) | **no** | 35B projections are W8; a different plan table |
      | q5 GEMV / q4 SwiGLU GEMV widenings (#88) | **no** | 27B dense-path kernels; 35B head is Q6 |
      | `--mlp-a8-decode` (#90) | **no** | 27B dense post-mixer only |
      | `--lm-head-q4` (#91) | **no** | gated in the 27B loader on a W8 head; 35B loader has no such path |

      - [x] **Tensor cores for the MoE experts: retired 2026-09-13 on measurement, before any
        kernel was written.** This entry used to point at `src/ops/sparse_moe/` containing no
        `mma.sync` at all while its experts are Q4/Q5 at 2..46 tokens -- the exact shape #89 solved
        for dense Q4/Q5 -- and call it the port worth scoping. It is not, and the counter that
        decides it is the one nobody had read:

        | kernel | FMA pipe | tensor pipe | DRAM (elapsed) |
        |---|---:|---:|---:|
        | `d3_nine_warp` (T=1) | 31.25% | **0%** | 47.4% |
        | `d4_nine_warp` (T=1) | 25.08% | **0%** | 51.6% |
        | `d3_path_tiled` (T~4) | 41.86% | **0%** | **71.25%** |
        | `d4_token` (T~4) | 35.05% | **0%** | 49.18% |

        **The FMA pipe never gets above ~42%, and DRAM is the high-water mark everywhere.** A
        tensor core raises the compute ceiling; there is no compute ceiling being hit. Measured
        arithmetic intensity is ~5.7-6.5 MAC/byte against a machine balance of ~21 on the FP32 FMA
        pipe, so these kernels are memory-bound *by a factor of three on the pipe they already
        use*. Moving to int8 tensor cores would push the balance to ~83 MAC/byte and make the
        mismatch worse, while adding activation quantisation error the way `--mlp-a8-decode` does.

        The structural reason, which is worth keeping because it applies to any MoE: top-8 of 256
        experts means average tokens per routed expert is `8T/256 = T/32`, so it is **1.0 at T=32
        and 1.4 at T=46**. Each expert's weights are read once and used once. There is no reuse
        for a tensor core to exploit -- the MMA's N dimension would sit ~1/8 filled across the
        whole small-T range. The one exception is the *shared* expert, which every token visits,
        and that is a single warp of nine.

        `d3_path_tiled` at 71-73% of achievable DRAM is already close to the 78-82% the contiguous
        kernels reach. The remaining headroom on this Op is the 8-of-256 gather, and it is not a
        kernel-math problem.

      **Do not "port small-T to the 35B" as stated.** The 35B's narrow-width projections already
      route to `SplitKMmaDirect` over 2..96 (W8 tables), which is the same split-K direct family
      that won width 8 on the Q4/Q5 side before small-T beat it. The real gap is that
      `src/ops/sparse_moe/` contains **no tensor-core code at all** -- `mma.sync` appears zero
      times in the whole directory -- while its experts are Q4G64/Q5G64 at 2..46 tokens, which is
      the exact shape #89 solved for dense Q4/Q5. That is the port worth scoping, and it is a new
      kernel family, not a routing change. Read the d3/d4 stall table in §2c first: d3 is memory
      latency and d4 is barrier-bound, so one kernel family will not fix both.

      *Two instrument bugs on the way, both of which produced confident nonsense.* `-pg 4096,128`
      captures prefill and decode together and prefill dominates — the 27B's top kernel came back
      as 256 launches of `q4a8_swiglu`, which is 4 prefill chunks x 64 layers. Then decode reported
      **99.3% GPU idle** from 807 launches over 128 steps of a 64-layer model, about six kernels
      per step, because nsys records a CUDA graph replay as one opaque entity unless given
      `--cuda-graph-trace=node`. Both reasons are written into the script.

- [x] **The routed prefill pipeline-depth threshold is measured on this card and 7/4 is not a
      risk. Closed 2026-09-09** — sweeping the constant through the product moves prefill by less
      than the run's own drift, and the likely reason is that no threshold in [1.5, 2.0] can change
      the decision at any prompt length anyone runs.

      `scripts/sweeps/moe-prefill-pipeline-depth.ps1`, 35B, int8 KV, `-p 1024,2048,4096,8192 -r 3`,
      clocks locked at 1,500 MHz, source patched and rebuilt per ratio, with 7/4 repeated last as
      the drift control the script's own header demands:

      | prompt | 1.50 | 1.5625 | 1.625 | **1.75** | 2.00 | 1.75 again | spread |
      |---:|---:|---:|---:|---:|---:|---:|---:|
      | 1,024 | 5,895.6 | 5,880.9 | 5,880.6 | 5,882.7 | 5,892.4 | 5,891.97 | **0.26%** |
      | 2,048 | 5,845.2 | 5,854.2 | 5,846.6 | 5,847.7 | 5,849.2 | 5,853.23 | **0.15%** |
      | 4,096 | 5,706.4 | 5,705.0 | 5,706.0 | 5,707.8 | 5,701.4 | 5,712.63 | **0.11%** |
      | 8,192 | 5,424.9 | 5,425.4 | 5,427.3 | 5,423.7 | 5,425.0 | 5,426.66 | **0.06%** |

      **The two 7/4 readings agree to 0.05-0.16%, and the between-ratio spread is 0.06-0.26%** — so
      the effect of a 1.33x change in the constant is the same size as the drift between two
      identical builds. There is nothing here to tune, and nothing to fear: the concern was that
      upstream's RTX 5090 number could be wrong on a card with a sixteenth of the L2, and across
      the whole range it does not matter.

      **Why it cannot matter is worth more than the table, and it is a caveat on this measurement
      too.** The decision is `jobs * Den < Num * touched` (line 327) where `jobs` is the work-item
      count and `touched` the number of experts with work, so the constant is compared against
      *jobs per touched expert*. With `kExpertBM = 64` and top-8-of-256, average tokens per expert
      is `T * 8 / 256`, so jobs per expert is `ceil(T / 2048)`: **~1.0 at T=1,024 and 2,048, ~2.0 at
      4,096, ~4.0 at 8,192.** Every threshold in [1.5, 2.0] therefore puts 1,024/2,048 on Spread and
      4,096/8,192 on Packed — **the branch never flipped between arms**, which is the simplest
      explanation for four identical columns.

      So this is a null result on the *risk* and a null *measurement* on the crossing point. The
      honest position: 7/4 is safe to leave, and anyone who wants the actual crossing must first
      make the branch move. Routing variance means jobs-per-expert is not exactly integral, so a
      real test needs the decision printed rather than inferred — instrument `route_job_count[0]`
      and `[1]` and the chosen `GateUpRoute` per prompt length, and look for a shape where the ratio
      lands strictly inside [1.5, 2.0]. On this geometry that is roughly `T` between 1,536 and
      2,048, i.e. a narrow band of short prompts, which is also why it is not worth chasing. `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:314-315`
      (`kGateUpDeepJobsNum`/`Den`, currently 7/4) picks between a 2-stage ("Spread") and a 6-stage
      ("Packed") pipeline for the narrow routed gate/up kernel based on jobs-per-expert crossing
      this ratio. The comment above it is explicit that the crossing point was measured on
      upstream's server, with a fixture that walks the ratio continuously and disagrees with the
      operator microbenchmark by about 6x depending on L2 warmth. That crossing depends on L2 size,
      and an RTX 5090 has 16x the L2 of this box's RTX 3090 (96 MB vs. 6 MB) — there is no reason to
      expect 7/4 is where *this* card's Spread/Packed tradeoff actually flips. Wrong constant,
      wrong pipeline depth selected, not wrong output: `kExpertStages`/`kGateUpNarrowStages` still
      produce correct results either way, so this is a performance risk, not a correctness one.
      Re-measuring needs the same round-robin fixture the comment describes, built and run on this
      RTX 3090; no sweep script here currently drives sparse MoE prefill directly
      (`decode-step-profile.ps1` and the `scripts/sweeps/*.ps1` sweeps are all decode-only). Do not
      guess a replacement constant without that measurement.

- [ ] **Eight lanes buy 3.6x, not 8x, because no kernel amortises a weight read across 2-10 rows.**
      **2026-09-11, largely addressed:** the small-T tensor-core kernels (`q4_small_t_mma.cuh`,
      `q5_small_t_mma.cuh`, `ops/common/small_t_layout.cuh`) now serve every 1-32-column decode
      extent of the Q4/Q5 GEMMs, and the SIMT families in the table below have no decode
      instances left. MTP3 reasoning cohort on a Linux RTX 3090: C1 92.9, C8 376.2 tok/s (4.05x),
      from 63.3 and 221.5. What remains is gate_up at T=32 running at ~68% of the bf16 MMA peak, and
      the 5120-row Q5 shapes having too few CTAs at T=32. See
      [docs/performance.md](docs/performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode).
      The measurements below are the pre-small-T profile and stay for the record.
      Measured 2026-09-09 on the 27B, int8 KV, no speculation, `--decode-tokens 512
      --max-context 8192`, via `tools/bench/run_serve_concurrency.py --suite decode-saturation`:

      | C | tok/s | vs C1 | batch | ms/round |
      |---|---|---|---|---|
      | 1 | 36.8 | 1.00x | 1.00 | 27.20 |
      | 2 | 61.1 | 1.66x | 2.00 | 32.73 |
      | 4 | 101.0 | 2.75x | 4.00 | 39.61 |
      | 8 | 132.9 | 3.62x | 8.00 | 60.18 |

      Those C8 figures predate the narrow GDN tiles of §3, which lifted C8 to **140.8 tok/s, 3.83x**
      without moving C1/2/4 (widths 1, 2 and 4 stay on the independent route). Everything below is
      still measured against the 3.62x profile, and the gap it describes is unchanged in kind.

      Batching itself works: `batch` is exactly C at every point and `row_rounds = C x rounds`, so
      one round serves the whole cohort and the 15.9 GiB weight read is amortised across it as
      intended. The loss is entirely in what a round costs. Against the 20.0 ms weight-bandwidth
      floor (15.9 GiB at the measured 854.2 GB/s), a C1 round spends 7.2 ms on everything that is
      not the weight read, and each added row costs 4.7 ms — 65% of that whole budget.

      **What it is not.** nsys at node-level graph tracing over a clean 150-round steady window
      (`profiles/conc-prof`, overhead negligible: 27.45 ms/round against 27.20 unprofiled) puts the
      GPU 95.2% busy at C1 and 97.6% busy at C8, so launch gaps explain none of it. The cohort is
      ~850 tokens deep on int8 KV, far too little traffic to matter, and host time stayed at
      0.3-0.5%.

      **What it is.** C1 and C8 run almost disjoint kernel sets. At C1, 89% of the round is GEMV
      specialisations (`q5_rowsplit_gemv`, `q4_linear_swiglu_gemv_pair`). At C8 those drop to zero
      instances and the round redistributes:

      | kernel | C1 ms/round | C8 ms/round |
      |---|---|---|
      | `q4_small_t_mma` | 0.00 | 15.45 |
      | `rowsplit_grouped_mma` | 0.00 | 13.78 |
      | `q5_rowsplit_gemm_simt_split2` (2 variants) | 0.00 | 17.10 |
      | `q5_rowsplit_gemv` (4 variants) | 13.61 | 0.09 |
      | `q4_linear_swiglu_gemv_pair` | 7.59 | 0.05 |
      | **all kernels** | **26.18** | **56.29** |

      36% of the C8 round runs in SIMT kernels that use no tensor cores and had zero instances at
      C1. That looks like a routing bug and **it is not one** — checked, so nobody re-checks it.
      Every dominant family was swept against its own schedule bench at T=1..16 and in each case
      the routed schedule is the fastest kernel that exists at T=8: `q5_linear_add` picks
      `split2_exact` (74.8 us at k=6144) over `mma_r64_c16` (101.4); `q4_q5_attn_input` picks
      `parent_split_fixed` (169.0) over `grouped_r32_c32_s4` (202.8); `q4_swiglu` picks
      `small_t_tiled`. The `{{2, 10}, Split2ExactResidual}` band in
      `src/ops/linear_add/q5/q5_linear_add_plan.cpp` carries no measured comment while
      every band from 11 upward does, but it turns out to be right anyway.

      The gap is kernel coverage, not dispatch. In the T=2..10 band the choice is between two bad
      shapes, and the crossover at 11 is simply where they cross:

      - `split2_exact` **grows with T** — 41.0 / 56.3 / 74.8 / 93.2 us at T=4/6/8/10 (k=6144),
        about +8.7 us per row. It barely amortises the weight read at all.
      - `mma_r64_c16` is **flat** — 101.4 us from T=4 all the way to T=16 — but flat at 4x the
        25.3 us that weight (21.6 MB at 854.2 GB/s) should cost to stream once. The T=1 GEMV
        reaches 41.0 us, or 62% of that floor.

      So the prize is a narrow-extent MMA kernel that keeps `mma_r64_c16`'s flatness at the GEMV's
      fraction of bandwidth. Perfect amortisation would hold the C8 round at C1's 26.18 ms and give
      305 tok/s instead of 132.9; the realistic share of that is whatever closes the 4x. Worth it:
      C8 is the profile the 27B release recommends for multi-user serving, and all three dominant
      families sit at 1.9-2.5x their T=1 cost for 8x the rows.

      **The narrow-extent MMA kernel was built and it is worse. Measured 2026-09-09.** Added an
      `R64C8` to `q5_linear_add` — `BlockCols` 8 with `WarpCols` 8, so one warp column, 4 warps and
      128 threads against C16's 8 and 256 — on exactly the reasoning above. It loses to `c16` at
      every width tested (us, medians of 21-31, k=6144):

      | T | 2 | 4 | 6 | 8 | 10 | 16 |
      |---|---:|---:|---:|---:|---:|---:|
      | `c8` | 131.1 | 116.7 | 115.7 | 108.5 | 133.1 | 126.0 |
      | `c16` | 114.7 | 103.4 | 102.4 | 102.4 | 103.4 | 95.2 |
      | `split2_exact` | 36.9 | 44.0 | 56.3 | 76.8 | 96.3 | — |

      6-32% worse than `c16`, and still far behind `split2_exact` below 11. Same story at k=17408.
      Reverted; the route table is unchanged and the schedule is not registered. The numbers are
      recorded in `src/ops/linear_add/q5/q5_linear_add_plan.cpp` beside the table so it is not
      retried.

      **Why it fails here and works elsewhere is the transferable part.** The same change succeeds
      on the GDN input projection — C8 and C16 beat C32 by 9-11% there, and that shipped. The
      difference is what dominates each kernel. The GDN projection's cost is padded MMA work, so
      cutting `BN` cuts real work. `q5_linear_add` is already weight-read-bound: `c16` sits at 24%
      of the 25.3 µs its weight costs to stream once, so cutting `BN` removes no work and halves
      the warps available to hide the read. **Narrowing a tile helps when padding is the cost and
      hurts when bandwidth is** — and this Op is the second kind.

      So the 4x between `c16` and the weight-streaming floor is real and still worth having, but it
      is not a tile-width problem and the prize stated above is not reachable that way. What
      `rk8v4` proves in §2c's KV entry applies here too: something reaches a much larger fraction of
      the floor at narrow extents, so the ceiling is not the obstacle — but the mechanism is not
      tile geometry.

      **Part of this is now attributed.** §3's DFlash2 cliff entry chased the same kernel from the
      other direction and landed on `rowsplit_grouped_mma_kernel`, which is 13.78 ms of the 56.29 ms
      C8 round here. It is the GDN input projection, and at C8 it runs the `{{7, 32}}` route from
      `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp` — a **32-wide** MMA tile whose cost is
      set by padded width, so eight live columns pay for thirty-two. The per-instance cost confirms
      it: 0.300 ms at width 8 against 0.311 ms at width 7, near-identical where a live-token-scaled
      kernel would differ by an eighth. So a `GroupedMixedMmaR64C8`/`R64C16` is wanted by both
      entries, and neither the wider direct route nor a routing change gets it — that was measured
      and refused, see §3.

      **The 35B MoE is now measured on the same curve, and it scales worse: 3.22x at eight lanes.**
      164.4 / 273.2 / 400.0 / 529.7 tok/s at C1/2/4/8, same harness and settings. That is the
      opposite of what "the MoE is work-starved at one token, so a cohort should fill the machine"
      predicts, and the reason is worth carrying here: **a dense model amortises its weight read
      across the cohort, an MoE with per-token routing does not.** Each token brings its own top-8
      of 256, so the expected distinct routed experts per round goes 8.0 / 15.8 / 30.5 / **57.4** —
      7.18x the routed weight bytes for 8x the tokens. Only the shared expert and the dense layers
      amortise at all. §3's MoE entry has the full working.

      So the 3.6x here and the 3.22x there have different causes and want different fixes, which is
      worth knowing before anyone treats "batched decode should be nearly free" as a general rule.
      On the dense 27B it nearly should be, and the gap is the narrow-extent kernels below. On the
      MoE it should not be, and no kernel work changes that.

      Two notes on how this entry used to read. The numbers it quoted (C1 78.71 vs C8 250.26 tok/s)
      came from README's cohort table, which is end-to-end **with MTP3 enabled** — tokens per round
      is roughly `1 + 3 x acceptance` rather than 1, so dividing them into a bandwidth figure gives
      nonsense (144% of peak at C1, done naively). And it named two tools that cannot do this:
      `ninfer_bench` has no concurrency option at all, and `NINFER_OP_REPORT_STATS` is the Op-test
      *accuracy* reporter, not a timing one. The serving harness plus nsys is the path, and reaching
      it needed the Windows shutdown fix in #70 — the campaign aborted on a throughput/`request_done`
      reconciliation that was correct and unmeetable.

- [x] **Prefill has no roofline either.** It has one now (2026-09-09), and the naive version of it
      is a trap worth documenting.

      Prefill is compute-bound: a 1,024-token chunk does 2 x 25.02e9 x 1024 = 51.2 TFLOP of GEMM
      against 15.74 GB of weight reads, an arithmetic intensity above 3,000 FLOP/byte. So the
      denominator is MMA throughput, and `tools/tensor_core_rate_probe.cu` — already in the tree,
      never pointed at this card — measures what it actually is:

      | MMA shape | measured |
      |---|---:|
      | BF16 m16n8k16, f32 accumulate | **67.6 TFLOPS** |
      | FP16 m16n8k16, f16 accumulate | 148.6 TFLOPS |
      | INT8 m16n8k32 | **314.8 TOPS** |
      | INT8 m16n8k16 | 302.4 TOPS |
      | INT4 m16n8k64 | 601.3 TOPS |

      **There is no single denominator, and using one gives whatever answer you want.** The 27B's
      prefill mixes paths: the MLP GEMMs quantize activations to INT8 (`q4a8_swiglu_kernel`,
      `q5a8_add_kernel`, fed by `quantize_activations`) while the GDN projections stay BF16
      (`rowsplit_grouped_mma_kernel`, `q5_rowsplit_gemm_mma_kernel`). Time splits 54/46 between
      them. Taking `2 x params` over the whole model and dividing by one ceiling gives **96.0% of
      peak if you assume BF16** and **20.6% if you assume INT8** — for the same measurement. The
      first number is very nearly what this entry was going to claim.

      Per kernel, against the ceiling each one actually uses (from #53's capture, 4,096-token
      prefill, parameter counts from the artifact):

      | kernel | GFLOP/token | path | achieved | % of that ceiling |
      |---|---:|---|---:|---:|
      | `q4a8_swiglu` (mlp gate_up) | 22.82 | int8 | 97.4 T/s | **30.9%** |
      | `q5a8_add` (mlp down) | 11.41 | int8 | 89.4 T/s | **28.4%** |
      | `rowsplit_grouped_mma` (gdn value_z, query_key) | 8.05 | bf16 | 34.7 T/s | 51.4% |
      | `q5_rowsplit_gemm_mma` (gdn output) | 3.02 | bf16 | 36.8 T/s | 54.4% |

      So prefill is *not* near its ceiling. The MLP GEMMs, which are 68% of the FLOPs, run at
      about **30% of the card's INT8 tensor-core rate**, and the BF16 projections at about 52% of
      the BF16 rate. A well-tuned large GEMM on Ampere usually reaches 60-80% of a pure-MMA
      microbenchmark, so there is real room here — plausibly more than in decode.

      One caveat kept deliberately: these chunks are 1,024 tokens, which is skinny for a GEMM whose
      other dimensions are 34,816 x 5,120, and the percentage would look different at a larger
      prefill chunk. Sweeping `--prefill-chunk` against this ceiling is the obvious next step and
      has not been done.

- [x] **Vision now has numbers. Closed 2026-09-09**, and the headline is that the encoder is not
      the expensive part. `scripts/sweeps/vision-encode-throughput.ps1`; 27B, int8 KV, one image per
      request, two measured repetitions per point. `apps/ninfer` reports the Vision stage separately
      from text prefill, so encode cost is read directly rather than subtracted out.

      | side | image tokens | encode (resident) | tok/ms | encode (overlay) | overlay penalty | text prefill | prefill / encode |
      |---:|---:|---:|---:|---:|---:|---:|---:|
      | 224 | 85 | 15.1 ms | 5.65 | 36.5 ms | +143% | 206 ms | 13.7x |
      | 448 | 217 | 27.2 ms | 7.98 | 50.3 ms | +85% | 316 ms | 11.6x |
      | 672 | 462 | 56.0 ms | **8.26** | 85.0 ms | +52% | 568 ms | 10.2x |
      | 896 | 805 | 112.0 ms | 7.19 | 128.0 ms | +14% | 964 ms | 8.6x |
      | 1120 | 1246 | 181.5 ms | 6.87 | 228.0 ms | +26% | 1,100 ms | 6.1x |
      | 1344 | 1785 | 300.0 ms | 5.95 | 342.0 ms | +14% | 1,700 ms | 5.7x |
      | 1568 | 2422 | 475.0 ms | 5.10 | 513.0 ms | +8% | 2,100 ms | 4.4x |

      **Prefilling the image tokens costs 4.4-13.7x what encoding them does.** That is the number
      to know before optimising anything here: at 1024px a single image is ~173 ms of encode against
      ~945 ms of text prefill, so the Vision tower is ~15% of time-to-first-token and the other 85%
      is the ordinary prefill path already covered by §2c's prefill entries. Vision needs no special
      optimisation attention until that changes.

      **Encode throughput peaks at 672px and falls 38% by 1568px** — 8.26 down to 5.10 tok/ms.
      Against the 672px peak, encode time is superlinear in tokens: 1.01x at 672, 1.21x at 1120,
      1.40x at 1344, **1.64x at 1568**. That is the shape attention inside the tower predicts, and
      it means very large images are charged twice — more tokens, each more expensive. Small images
      are also inefficient (1.48x at 224px) but for the opposite reason: 85 tokens does not fill the
      machine.

      **Overlay residency costs a roughly fixed ~30 ms per request, not a proportional one, and
      saves 250.7 MiB.** Runtime reservation is 848.0 MiB resident against 597.3 MiB overlay. The
      penalty is +143% at 224px and **+8% at 1568px** because the tower crosses PCIe once per
      request whatever the image size. So overlay is close to free on large images and expensive on
      small ones — the opposite of the intuition that a bigger image costs more to overlay. The
      release launchers default to overlay, which these numbers support for the image sizes anyone
      actually sends.

      **Encode is exactly linear in image count**, and the overlay penalty is paid once per request
      rather than once per image. Overlay, 1024px fixtures:

      | images | tokens | encode | tok/ms | vs 1 image |
      |---:|---:|---:|---:|---:|
      | 1 | 1,045 | 173.0 ms | 6.04 | 1.00x |
      | 2 | 2,071 | 352.0 ms | 5.88 | **2.03x** |
      | 4 | 4,123 | 691.5 ms | 5.96 | **4.00x** |

      So batching images into one request is neither better nor worse per token than sending them
      separately, except that it amortises the one-off overlay crossing. Nothing here needs a
      per-image cache.

      One measurement trap the script documents: **resolutions are not free-form.** The
      preprocessor snaps to a multiple of the merged patch size, so nearby resolutions can produce
      identical token counts and a naive sweep shows plateaus that look like measurement error. The
      sides above are multiples of 112 for that reason, and the token count is printed beside every
      row so a plateau is visible rather than mysterious.

- [x] **DFlash2 makes the 27B *slower*, and no document says so.** Wrong, and closed 2026-09-09.
      On text DFlash2 is a **win at every draft count from 1 to 12**. The entry's own premise came
      from a benchmark fixture that cannot measure drafting at all.

      Measured through the serving path on the model's own generated prose — 27B DFlash2 artifact,
      INT8 KV, greedy, 256 generated tokens, mean of three runs, spread ≤0.2 tok/s except one row:

      | `--draft-tokens` | decode | vs none |
      |---:|---:|---:|
      | (none) | 37.7 | — |
      | 1 | 49.7 | +31.7% |
      | 2 | 56.4 | +49.5% |
      | 3 | 58.4 | +54.9% |
      | **4** | **59.1** | **+56.5%** |
      | 5 | 57.2 | +51.5% |
      | 6 | 48.4 | +28.3% |
      | 7 | 48.2 | +27.7% |
      | 8 | 47.6 | +26.2% |
      | 10 | 42.4 | +12.4% |
      | 12 | 40.6 | +7.6% |

      Three things follow.

      **The shipped recommendation is the wrong draft count.** `docs/cli.md` said seven; four is
      **22.6% faster**, and `docs/cli.md` now says four and carries this table. The entry guessed
      there might be "a crossover below 7 where DFlash2 turns positive" — there is a crossover at
      four, but DFlash2 was never negative to begin with.

      **MTP is still ahead, by far less than seven implied.** MTP3 with the draft head reaches
      62.4 tok/s on the same measurement, so the gap is 5% at DFlash2's best count rather than the
      36% this entry recorded.

      **`--lm-head-draft` does nothing for DFlash2.** Within noise of unset at every count.

      There is a clean cliff between five and six, 57.2 to 48.4, that looks like a block-geometry
      boundary rather than acceptance — worth a look if anyone wants the last few percent.

      *Why the old number was wrong, which matters more than the number.* It was measured on
      `bench/fixtures/bench_corpus.ids`, which holds 65,536 tokens drawn from **682 distinct ids**
      with **98.4% of bigrams repeated**, because it is a curated bank tiled to length. Both the
      corpus manifest and the generator's docstring asserted that "repetition fills length only and
      does not bias throughput" — true for prefill and plain decode, and false for anything that
      drafts. Swept through `ninfer_bench` this corpus reports **100% acceptance at every draft
      count from 1 to 12**, with decode rising monotonically to 159 tok/s because each round emits
      k+1 free tokens. Both claims are now corrected in place, and
      `scripts/sweeps/dflash2-draft-tokens-realtext.ps1` is the sweep that does not use it.

- [ ] **The three KV formats with the worst decode falloff are exactly the three that stage
      dequantized tiles in shared memory — and the falloff is 3.21x the per-key cost.** Located
      2026-09-09; the mechanism is below. #49 fixed the non-determinism that first drew attention to
      the same three formats, and the falloff did not move — but they are siblings rather than a
      coincidence, because the missing barrier sat around exactly that shared staging. Re-measured after it, and after the
      split-tier change in #52:

      | model | int8 | rk8v4 | fp8 | k8v4 | nvfp4 |
      |---|---:|---:|---:|---:|---:|
      | 27B dense | −10.4% | −7.9% | −12.5% | −15.6% | −18.0% |
      | 35B-A3B | −9.4% | −7.2% | −21.3% | −26.0% | −21.4% |

      The original table, kept because the ratios are what matter and they are unchanged:

      | format | 27B | 35B | deterministic (§5) |
      |---|---:|---:|---|
      | `int8` | −6.3% | −10.9% | yes |
      | `rk8v4` | −6.2% | −9.5% | yes |
      | `bf16` | −11.5% | −14.9% | mostly |
      | `fp8` | −14.6% | −21.0% | **no** |
      | `nvfp4` | −15.7% | −21.1% | **no** |
      | `k8v4` | −17.9% | −25.2% | **no** |

      `rk8v4` is the control that kills the obvious explanation: it packs 4-bit values exactly as
      `k8v4` does, and it has the flattest curve of all six. So this is not the cost of unpacking.
      **If `nvfp4` decoded on `rk8v4`'s curve it would be the outright best format** — 45% smaller
      than INT8 with no speed penalty — rather than the compromise it currently is.

      Read the code rather than inferring from the table, and two things came back. One is a
      likely cause; the other kills the tidy theory that produced this entry.

      **Cause, and a one-line experiment.** fp8, k8v4 and nvfp4 are the only formats calling
      `causal_small_t_quantized_active_splits` (`small_t.cuh:122-130`), which at decode past 8,198
      keys asks for `SmallTMaximumSplits` (85). But the *host* capacity function
      (`small_t.cu:49-53`) grants that bump to `Fp8E4M3Row256` **only**, and the device then clamps
      to whatever the host allocated (`small_t.cuh:129`). At a 32,768-token decode the default tier
      is `div_up(32768, 480) = 69`, so:

      | format | device policy asks | host grants | actually runs |
      |---|---:|---:|---:|
      | `fp8` | 85 | 85 | 85 |
      | `nvfp4` | 85 | 69 | **69** |
      | `k8v4` | 85 | 69 | **69** |

      **That experiment was run (#52) and it refutes the hypothesis.** Extending the grant to
      `Nvfp4Group16` and `Fp8KeyNvfp4Value` made them *slower* on the 35B -- -3.1%/-2.5%/-2.3% at
      4,096/16,384/32,768 against an fp8 control that moved 0.3% -- and removing the bump from fp8
      as well gained 0.4%/0.5%/1.2%. More splits at depth is worse for every quantized format, the
      host's default tier was the better number, and the special case is now gone from both sides.
      The falloff is unchanged by it and still wants an explanation.

      **Located and quantified 2026-09-09, and the sub-hypothesis above is wrong.** nsys with
      node-level graph tracing on the 27B, `-pg P,128` at P = 4,096 and 32,768, filtered to the
      decode attention kernels (`small_t`, 4,225 launches in both runs so per-launch figures are
      directly comparable) and away from the prefill `prompt` kernels that otherwise dominate a
      `-pg` capture:

      | format | 4,096 keys | 32,768 keys | growth for 8x the depth |
      |---|---:|---:|---:|
      | `int8` | 62.65 µs/launch | 115.24 µs/launch | **1.84x** |
      | `fp8` | 87.06 µs/launch | 255.96 µs/launch | **2.94x** |

      So the falloff is in the decode attention kernel, it is not subtle, and it separates into two
      distinct components:

      - a **fixed** part — fp8 already costs 1.39x int8 at 4,096 keys, +24.4 µs per launch before
        depth enters;
      - a **per-key** part — marginal cost per additional key is **1.83 ns for int8 against 5.89 ns
        for fp8, a 3.21x ratio**. This is the falloff. It is per-key work, which is what
        dequantizing every key block into shared memory looks like, and it is why the gap widens
        with depth (1.39x at 4K keys, 2.22x at 32K).

      Both formats scale far better than the 8x a serial-in-keys kernel would give, because
      split-K absorbs depth — so this is not a parallelism failure, it is a cost-per-key one.

      **`KeyBlock` is not the difference.** This entry guessed fp8 was "paying something else —
      plausibly `KeyBlock` pinned to 32 rather than the int8 path's 64". Read the dispatch:
      `small_t.cu`'s final `else` branch, which is the one decode takes (`TokenTile <= 3`), launches
      `<8, 2, 32, false>` — **`KeyBlock = 32` for int8 too**. The 64-wide case appears only under
      `TokenTile >= 6`, which is a prefill width. There is no 32-against-64 asymmetry at decode to
      explain anything.

      **What does separate the six formats is `DynamicArena`.** int8, rk8v4 and bf16 all decode with
      `DynamicArena = false` and zero dynamic shared memory. fp8, k8v4 and nvfp4 all decode with
      `DynamicArena = true`, staging dequantized tiles in a shared arena:

      | format | dynamic shared per block | blocks/SM that allows | 27B falloff | 35B falloff |
      |---|---:|---:|---:|---:|
      | `int8` / `rk8v4` / `bf16` | **0** | unbounded by shared | −10.4 / −7.9 / — | −9.4 / −7.2 / — |
      | `nvfp4` | 24 KiB (`3 x 32 x 256`) | 4 | −18.0% | −21.4% |
      | `k8v4` | 44 KiB (`11 x 32 x 256 / 2`) | 2 | −15.6% | −26.0% |
      | `fp8` | 48 KiB (`6 x 32 x 256`) | 2 | −12.5% | −21.3% |

      That is the **same three-way split as §5's non-determinism**, and it means the correlation this
      entry opened by calling "a coincidence" is not one. Both symptoms come from the same
      architectural choice — dequantize-into-shared — because #49's missing barrier sat between
      `cp_wait<0>()` and `dequant_k_tile()`, i.e. around exactly that staging. The non-determinism
      was not causally upstream of the falloff (fixing it moved nothing), but the two are siblings
      rather than coincidence, which is a more useful thing to know.

      **Occupancy alone does not order them, so it is necessary and not sufficient.** nvfp4 uses
      half fp8's dynamic bytes and gets twice the blocks per SM, yet has the *worst* 27B falloff of
      the six. So block count is not the whole mechanism; the per-key dequantization work is the
      part that matches.

      **The `ncu` counters this entry asked for are collected (2026-09-10), and they add a fact the
      entry does not have: both formats reach only a third of their own theoretical occupancy.**
      27B, `-pg 8192,64`, clocks locked, kernel replay, six launches of the `small_t` decode
      attention kernels, medians:

      | | `int8` (`..._i8_tiled`) | `fp8` (`..._fp8_tiled`) |
      |---|---:|---:|
      | duration | 32.62 µs | **78.40 µs** |
      | DRAM throughput | 26.16% | 13.43% |
      | L1/TEX throughput | 25.61% | 14.34% |
      | Compute (SM) throughput | 16.61% | 16.07% |
      | **theoretical occupancy** | **66.66%** | **50.00%** |
      | **achieved occupancy** | **21.98%** | **16.51%** |
      | stall `long_scoreboard` | 33.02% | 30.98% |
      | stall `wait` | 14.76% | **21.95%** |
      | stall `barrier` | 16.76% | 8.56% |

      Three readings, in order of how much they change the entry.

      **Nothing is saturated in either format.** The highest utilisation anywhere in that table is
      26%. So the falloff is not bandwidth, not L1, and not compute — it is latency, the same
      diagnosis §2c reached for the MoE gather and the narrow-extent kernels, and it means the
      framing of "3.21x the per-key cost" as a *cost* is misleading: the kernel is not doing 3.21x
      the work, it is waiting 3.21x as long.

      **The dynamic shared arena costs exactly what this entry predicted, and it is the smaller
      half.** fp8's theoretical occupancy is 50.00% against int8's 66.66%, which is the 2-blocks-per-SM
      against unbounded that the arena table above derives. But that is a **1.33x** ratio against a
      **2.4x** duration ratio, so the arena explains well under half of the gap. Removing it — the
      `rk8v4`-shaped fix this entry proposes — should therefore be expected to recover part of the
      falloff, not all of it, and anyone who ships it should predict ~1.3x rather than 3.2x.

      **And the new fact: achieved occupancy is 33% of theoretical in *both* formats** — 21.98 of
      66.66, and 16.51 of 50.00, the same ratio to two figures. The grid explains it, and the grid
      is identical in both:

      | | `int8` | `fp8` |
      |---|---:|---:|
      | grid | **(4, 65) = 260 blocks** | **(4, 65) = 260 blocks** |
      | block | 256 threads (8 warps) | 256 threads (8 warps) |
      | **Block Limit Registers** | **4** | **4** |
      | Block Limit Shared Mem | 8.50 | 7.50 |
      | Block Limit Warps / SM | 6 / 16 | 6 / 16 |
      | theoretical warps per SM | 32 | 24 |
      | achieved warps per SM | 10.54 | 7.93 |

      **260 blocks against a capacity of 4 x 82 = 328 is 0.79 machine-fulls.** The kernel cannot fill
      the card even once, in either format, which is the same shape as §2c's GDN entry and the MoE
      gather — and at 32-78 µs a kernel that never fills the machine spends much of its life in ramp
      and drain, which is what drags achieved warps to a third of theoretical.

      **Two things here do not match what this entry has been assuming, and both want reconciling
      before any kernel work.** First, **`Block Limit Registers` is 4 in both formats while
      `Block Limit Shared Mem` is 7.5-8.5** — so registers bind at or before shared memory, and the
      dynamic arena is *not* the binding term for `int8` at all. The arena table above derives
      2 blocks/SM for fp8 from 48 KiB of dynamic shared; ncu reports 7.5. Those disagree, and the
      arena story rests on the derivation rather than on a counter. Second, fp8's theoretical warps
      (24) is 3 blocks' worth where int8's (32) is 4, so *something* costs fp8 a block — but with
      both showing `Block Limit Registers = 4` it is not obvious from these counters which term it
      is.

      So the ordering changes: **the grid is the bigger and better-evidenced problem, it is
      format-independent, and it is shared by `int8` — the format everyone considers the good one.**
      More key-splits would add blocks and cost nothing but a wider reduction. Settle the
      register-versus-shared question with `--section Occupancy --section LaunchStats` on all six
      formats before spending effort on removing the arena, which on this evidence buys the smaller
      half of a gap it may not even be causing.

      What this is worth: **if `nvfp4` decoded on `rk8v4`'s curve it would be the outright best
      format** — 45% smaller than INT8 with no speed penalty — rather than the compromise it is.
      The target is the 3.21x per-key ratio, and `rk8v4` proves it is achievable, because it packs
      4-bit values exactly as `k8v4` does and yet has the flattest curve of all six without a
      shared arena at all.

      **What this entry originally claimed, wrongly — now settled.** It read the §5
      non-determinism and this falloff as one root cause — "a split reduction whose order varies, or an atomic
      accumulation". There are **no atomics anywhere** in `src/ops/softmax_attention/`, and both
      reducers accumulate over an identical ordered `for (split = 0; split < active_splits; ++split)`
      loop, which is deterministic given the same split count. So the correlation across six
      formats is real and still wants explaining, but the mechanism proposed for it is not the one.
      Treat §5 as open on its own terms.

      One thing found on the way that is worth its own look: the fp8 partial kernel returns early
      for `split >= active_split_count` **without writing neutral values**
      (`small_t_fp8.cuh:148`), so untouched splits keep whatever the workspace arena last held.
      That is safe only while the partial kernel and the reducer compute the same
      `active_split_count`. They do today — both clamp to the same launch capacity — but it is an
      invariant held by coincidence of two separate call sites rather than by construction, and the
      host/device asymmetry above shows those sites already disagree about intent.

- [ ] **The sm_86 fallback constants were chosen to fit, not measured.** `NINFER_SM8X_COMPAT`
      guards eleven sites, and most encode a real hardware limit rather than a missing feature —
      sm_86's 49,152-byte static shared-memory cap forces `KWarps` from 8 to 4 in
      `w8_config.h:59`, halves the K tile in `w8_linear_swiglu_gemm_mma.cu:120`, drops
      `kLastExactCols` from 48 to 32 in `w8_linear_add_gemm_splitk.cu:19`, and so on. The comments
      are honest that these are what fits. None of them record a measurement showing the chosen
      value is the *best* one that fits, and the alternatives were never swept on this hardware.
      Lower confidence than the two above, but it is untouched ground across every W8 Op.

### Small-T decode kernels landed, and what they left (2026-09-11/12)

PR #89 (`perf/small-t-mma`) put every 1-32-column Q4/Q5 decode GEMM on tensor-core small-T kernels
and deleted the fused GDN projection-epilogue conv. MTP3 decode is 1.5x faster at C1 and 1.7x at
C8 with perplexity bit-identical; the numbers, the per-Op table and the eight measured-and-rejected
ideas live in [docs/performance.md](docs/performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode).

**Read those absolute microseconds with care.** They were measured on a rented Linux RTX 3090 at a
350 W cap, whose probes read 892.8 GB/s and 81.8 TFLOPS BF16 -- not this box's 854.2 GB/s and 67.6
TFLOPS. Ratios transfer; microseconds do not. Re-measure the three ceilings before quoting any of
it here, exactly as "Handing this off to another machine" says.

What is left, in order of size:

- [x] **C8 is NOT tensor-rate bound -- this entry was wrong, and acting on it cost a kernel.**
      It read: gate_up at T=32 runs at ~68% of the bf16 MMA peak (205 us against a 139 us tensor
      floor), so the only large lever left is int8 tensor cores. Both halves are now measured and
      the first one does not hold. `ncu` on the T=32 gate_up kernel, this box:

      | | BF16 small-T | int8 small-T |
      |---|---:|---:|
      | tensor pipe active | **38.3%** | 21.0% |
      | L1/TEX throughput | 36.4% | 73.5% |
      | DRAM throughput | 39.4% | 42.7% |
      | warp occupancy | 30.9% | 45.4% |

      Tensor, L1 and DRAM all sit within three points of each other around 38%, at 31% occupancy:
      nothing is saturated, which is a latency-bound kernel, not a tensor-rate-bound one. The 68%
      compared against a computed floor rather than a measured pipe and the two disagree.

      The int8 route was built anyway and is the evidence: it halved tensor-pipe pressure exactly as
      a 4.7x-denser MMA should (38.3% -> 21.0%) and returned **4-6%**, because the pipe it relieved
      was never the constraint and the cost landed on L1 at 73.5%. It ships behind `--mlp-a8-decode`
      (docs/maintainer/quality-trade-experiments.md) since it is also a quality trade.

      **What the counters point at instead:** operand movement and occupancy. The largest single win
      in that whole exercise was deleting a redundant copy of the staged activation slab, and the
      second was dropping a runtime column count so the staging loop could fold (below).

      **Three things are already checked, so do not spend the time again.**

      1. **Occupancy cannot be bought by raising the launch-bound ceiling.** 31% occupancy at two
         blocks per SM reads as an invitation; it is not one. The ceiling is a template parameter
         (`Q4SwiGluSmallTTile::kMinBlocks`) and was swept. Capping registers spills, cold medians in
         us with the count of instantiations ptxas reports spilling:

         | min blocks | T=16 | T=24 | T=32 | spilling |
         |---|---:|---:|---:|---:|
         | 2 (shipping) | 161.8 | 175.1 | 205.8 | 0 |
         | 3 | 162.8 | **321.5** | 227.3 | 4 |
         | 4 | 161.8 | **543.7** | 281.6 | 4 |

         At 96 registers this kernel needs them. More warps in flight has to come from a kernel that
         *needs* fewer registers -- that redesign is the genuinely untried lever, not this knob.

      2. **The shipped small-T kernels have no redundant activation staging.** `q4_small_t_mma.cuh`
         and `q5_small_t_mma.cuh` both stage CTA-wide (`item = tid; item += kThreads`), so each
         element is staged once and shared across row tiles exactly as `small_t_layout.cuh` intends.
         The redundancy worth deleting was in the int8 kernel's own first draft, which staged
         per-warp. There is nothing to find here.

      3. **cp.async ring depth loses at every width** on this shape, which the L1 and latency
         readings above explain: there is no DRAM latency to hide. Numbers in
         `q4_small_t_mma_i8.cuh`.

- [x] **A width that fills its tile does not need a runtime column count.** Every small-T call site
      passed `MaskedColumns=true`, so the staging loop's trip count and the inner loop's bounds test
      could never fold, even at exact widths -- and a C8 MTP3 cohort is exactly thirty-two columns.
      Giving gate_up an unmasked instantiation is worth a median +4% at 16 columns, +3-6.5% at 24
      (positive in all six runs) and +0.5-2.9% at 32, bit-identical.

      Two things stopped it being universal, both worth knowing before trying it elsewhere. Eight
      columns *spills* unmasked -- ptxas reports 16 bytes of cumulative stack -- because the
      compile-time count lets the staging loop unroll past the 42-register ceiling that the KWarps 8
      schedule's `__launch_bounds__(256, 6)` imposes; wider tiles run KWarps < 8 at two blocks per
      SM and unroll with registers to spare. And the same change **loses** on the GDN and attention
      query/key projections, which share the template: attention by 4.1% at thirty-two columns in
      three identical runs, GDN by 11%. Both were reverted.

      **Why they lose is an issue-efficiency effect, not a resource limit.** `ncu` on the attention
      query/key kernel at T=32, exact against masked:

      | | exact | masked (shipping) |
      |---|---:|---:|
      | issue active | 26.0% | **29.5%** |
      | warps active | 22.6% | **24.6%** |
      | tensor pipe | 32.1% | 33.4% |
      | L1/TEX | 30.5% | 31.7% |
      | DRAM | 25.2% | 27.2% |

      Registers (94 against 95), shared memory and spills are all unchanged, and both builds have
      the same theoretical occupancy -- two blocks per SM either way. What differs is that the exact
      build issues an instruction 13.5% less often and keeps 8% fewer warps resident. Nothing is
      above 34%, so this kernel is deeply latency-bound and its performance turns on instruction
      scheduling rather than on any unit's throughput; a fully unrolled staging loop appears to
      burst its copies and then stall where the runtime-bounded loop interleaves them. That is
      consistent with gate_up reacting the other way -- it runs a different epilogue and row policy
      at a higher 31% occupancy -- but the mechanism behind the sign flip is inferred from the issue
      counters, not proven.

- [ ] **C1 is mostly kernel-boundary ramp and drain.** A round is ~500 kernels; the Q5 residual
      projections reach 68-82% of their streaming floor and gate_up 89%, and the shortfall scales
      with how small the matrix is, which is the signature of per-launch ramp rather than of any
      one kernel. Programmatic dependent launch is the fix and needs sm_90, so on `sm_86` the only
      route is fusing work into fewer kernels.

- [x] **Two quality trades, measured 2026-09-12 and wired to CLI flags** on branch
      `perf/quality-trades` (off #89): `--lm-head-q4` (int4 vocabulary head, requantized in place at
      load) and `--gdn-state-fp16` (FP16 GDN recurrent-state storage). Full numbers, method, and the
      verdict per trade are in `docs/maintainer/quality-trade-experiments.md`. Short version:
      `--gdn-state-fp16` is a clean win (free within measurement noise, +2.0% real-text C8, halves
      the host state image) and `--lm-head-q4` is a narrower one (+0.69% perplexity, altered greedy
      output from ~char 210 onward, for +3.2% real-text C8 and ~0% C1). Both default off.

      **The naive C1 expectation (~3% for the head) did not hold on real text**, and the first
      MTP3-decode read (+41%, via `ninfer_bench`'s default corpus) was a measurement artifact of
      `bench/fixtures/bench_corpus.ids`'s 98.4%-repeated-bigram content inflating acceptance —
      exactly the pitfall this file already names elsewhere. `tools/bench/run_chat_decode.py` on
      real prompts is what actually settled it; that script gained an `env=K=V` arm field so
      env-gated trades (or anything else) can be A/B'd on one binary without losing the
      interleaving this card's 3-5% process drift requires.

### The narrow-extent kernels are parallelism-starved, and that explains four separate entries

Found 2026-09-09 with `ncu` counters on the schedule benches (`admin-profile.ps1` section 5). This
is the cause behind "24% of its weight-streaming floor", "27% of its own weight-streaming floor",
the DFlash2 5→6 cliff and most of the eight-lane gap. **Four entries below were chasing one
mechanism, and it is not tile geometry.**

- [x] **Width 8 now routes to the split-K direct path and the C8 cohort's extent is 2.85% faster.
      Shipped 2026-09-09. SUPERSEDED 2026-09-13 — the route is gone; read this entry for the
      parallelism diagnosis, not for the routing decision.** The small-T kernels beat split4 at
      width 8 by 2.03x (82.9 µs against 167.9, disjoint spreads) once a merge put both in one
      binary. Everything below was measured on a branch with no small-T kernel, so it compares
      split4 against the grouped tiles only — which is why a result this size went unnoticed for
      four days. The diagnosis stands and the kernel is kept as the bench's control; the
      `{8,8}` band does not. The parallelism diagnosis below is what produced it: the fix was not a
      new kernel but ten template instantiations of one that already existed and had been dismissed
      on a measurement of a different kernel. Result first, then the diagnosis that found it.

      `launch_q5_split4_exact` threw above T=6, so `launch_q5` fell back to `simt_r8_c8` at exactly
      width 7 — and that is where the schedule bench's `independent` column jumped 188.4 → 330.8 µs
      for one extra column. Instantiating split4 to 8 (cold, medians of 31, `--spread`, clocks
      locked):

      | T | independent (split4) | c8, previously routed here | independent before (simt_r8_c8) |
      |---|---:|---:|---:|
      | 6 | 207.9 (205.8..209.9) | 243.7 | 188.4 |
      | 7 | **228.4** (227.3..229.4) | 242.7 (240.6..245.8) | 330.8 |
      | 8 | **208.9** (207.9..209.9) | 240.6 (236.5..243.7) | 319.5 |
      | 9 | 572.4 | 527.4 | 486.4 |

      Cold, that is +5.9% at width 7 and +13.2% at width 8 over the routed `c8` tile, spreads
      disjoint at both, and 31-35% over what the direct route used to cost.

      **Only one of the two survived in situ, and that is the part worth carrying.** Measured end to
      end with `tools/bench/run_interleaved_ab.py` — two binaries alternating inside each
      repetition, clocks locked at 1,500 MHz, six repetitions, draft count 4 (width 5, unchanged
      route in both arms) as a drift control:

      | config | paired median | positive | control drift |
      |---|---:|---:|---:|
      | k=6, width 7 | **−0.78%** | 0 / 6 | +0.00% |
      | k=7, width 8 | **+2.85%** | **6 / 6** | +0.09% |

      Both identical to three figures across all six repetitions, with the control at 0.00-0.09% —
      the cleanest A/B in this file, and the clock lock is why. So the route table is
      `{1,6}` independent / `{7,7}` c8 / `{8,8}` independent / `{9,16}` c16. **The single-width band
      at 8 is not tidy and it is what the measurement supports**: width 8 is the C8 serving cohort,
      the profile the 27B release recommends, where this kernel is 13.8 ms of a 56.3 ms round — so a
      13% kernel win landing as ~2.9% of the round is the arithmetic working out exactly.

      Width 9 collapses to 572.4 and sets the upper bound. It is a register cliff, not a geometric
      one: split4 carries `__launch_bounds__(128, 10)`, capping it at 51 registers per thread, and
      `acc[kTt]` costs one register per column. Widening past 8 needs `MIN_BLOCKS` relaxed first,
      which is affordable — the kernel has 12.5 machine-fulls of blocks and does not need ten per SM
      — but is a separate measurement.

      `ninfer_gdn_input_proj_test` covers widths 7 and 8 already, so the reroute is under test on
      both sides of the new boundary; the affected suite is 18/18.

      **The remaining prize is unchanged and larger**: a split-K *MMA* kernel, which would keep the
      tensor-core efficiency the direct path lacks. See the diagnosis below.

- [ ] **`rowsplit_grouped_mma_kernel` reaches a quarter of the card because it only ever launches a
      quarter of a machine-full of warps. The fix is split-K, and a working one already exists in
      the same directory for the W8 variant of the same Op.**

      Counters at width 8, `q4_q5_gdn_input_schedule_bench`, clocks locked at 1,500 MHz:

      | | C8 `<64,8,64,16,8>` | C16 `<64,16,64,16,8>` |
      |---|---:|---:|
      | duration | 243.55 µs | 261.47 µs |
      | grid x block | 256 x 128 | 256 x 256 |
      | DRAM throughput | **27.71%** | **25.39%** |
      | L1/TEX throughput | 54.26% | 61.51% |
      | Mem pipes busy | 47.83% | 52.93% |
      | SM throughput | 47.83% | 52.93% |
      | theoretical occupancy | 50% | 83.33% |
      | **achieved occupancy** | **24.52%** | **45.90%** |
      | eligible warps per scheduler | **0.55** | 0.96 |
      | registers per thread | 48 | 46 |

      **Nothing is saturated.** The highest utilisation of any pipe is 54-62%, DRAM is at a quarter,
      and there is barely half an eligible warp per scheduler. So "weight-read-bound at 27% of the
      floor" was the wrong diagnosis: it is not read-bound at all, it is starved.

      **And the shortfall is arithmetic, not mysterious.** Total warp-work for this kernel is
      `(rows / WM) x (BN / WN)`. The GDN input projection is 4,096 q4 rows plus 12,288 q5 rows, so
      with `WM = 16`:

      | schedule | warps | of the card's 3,936 | predicted occupancy | measured |
      |---|---:|---:|---:|---:|
      | C8 | 1,024 | 26.0% | 26.0% | **24.52%** |
      | C16 | 2,048 | 52.0% | 52.0% | **45.90%** |

      Predicted from block counts alone, matched to within 1.5 and 6 points. The kernel is **0.26
      machine-fulls** at C8. It cannot occupy the card, so it cannot hide latency, so it reaches a
      quarter of DRAM — and every symptom follows.

      **`BM` does not appear in that formula, which is why every tile experiment failed and had to.**
      Narrowing `BN` 32→8 removed 75% of the padded MMA work and bought 11%, because it also
      *halved* the warps (`BN / WN`); the `R64C8` built for `q5_linear_add` lost 6-32% for the same
      reason. Halving `BM` does not help either: it halves warps per block and doubles the block
      count, leaving the product unchanged. **No block-tile shape reaches more parallelism.** That
      retires "the prize is a narrow-extent MMA kernel that keeps `mma_r64_c16`'s flatness at the
      GEMV's fraction of bandwidth" — there is no such tile.

      **It also explains the flatness itself.** `mma_r64_c16` is flat at 101.4 µs from T=4 to T=16
      because the grid does not depend on T at all; T only changes how much of each padded tile is
      live. A kernel whose cost is set by a fixed, too-small grid is flat by construction.

      **Split-K is the only lever, and the route tables have been quietly voting for it already.**
      `q5_linear_add` picks `Split2ExactResidual` over `mma_r64_c16` for all of T=2..10 (74.8 µs
      against 101.4 at k=6,144). That was recorded as "the right choice between the kernels that
      exist, not a good one" — it is better than that. It is the higher-parallelism kernel winning,
      and nobody knew that was why.

      **The template is `src/ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu`, and the W8 route
      table has been doing this the whole time:**

      ```
      {1, 1},        DecodeR8Direct
      {2, 96},       SplitKMmaDirect      <-- narrow widths go to split-K
      {97, kAnyCols} MmaR64C128
      ```

      against the Q4/Q5 table for the same Op, which has **no split-K entry anywhere**:

      ```
      {1,6} independent / {7,8} c8 / {9,16} c16 / {17,32} c32 / {33,64} c64 / {65,inf} c128
      ```

      The W8 kernel splits K *inside* the CTA — `KSplits` warps cover disjoint K ranges and reduce
      through shared memory, so no global reduction pass is needed — and takes 16 rows per CTA
      rather than 64. Its parallelism at `KSplits=4, NGroups=2` is `(12,288 / 16) x 8` = **6,144
      warps, 1.56 machine-fulls**, against the Q4/Q5 grouped tile's 1,024. Carried across to the
      Q4/Q5 shape: `(16,384 / 16) x 8` = **8,192 warps, 2.08 machine-fulls**.

      So the work is a `Q4Q5GdnInputScheduleId::SplitKMma` modelled on the W8 kernel, routed over
      roughly `{7, 32}` where the grouped tiles are starved, benched with the existing
      `q4_q5_gdn_input_schedule_bench`. **Two entries pay for it** — the DFlash2 5→6 cliff and the
      eight-lane C8 cohort, where this kernel is 13.78 ms of a 56.29 ms round.

      One thing to check before building, because it is the one way this could fail to transfer: the
      W8 kernel's shared-memory reduction is sized for `padded_k = 2048`, and the Q4/Q5 projection
      reduces over 5,120. `KSplits x kTileK` must divide the reduction depth, and 5,120 / 64 = 80
      tiles divides by both 2 and 4, so the arithmetic works; the shared arena is the thing to size.

      **There is a much cheaper experiment to run first, and the reason it has never been run is
      that the measurement which appeared to rule it out tested a different kernel.** The route
      table's own winner below width 7 is *already* a split-K kernel — `launch_q5` sends T=2..6 to
      `q5_rowsplit_gemm_simt_split4_kernel`, one block per output row with four warps splitting K
      and reducing through `s_part[4][kTt]`. Its parallelism dwarfs everything else here:

      | route | geometry | blocks | warps | machine-fulls |
      |---|---|---:|---:|---:|
      | `launch_q5_split4_exact` (T=2..6) | one block per row x 4 warps | 12,288 | **49,152** | **12.49** |
      | `launch_q5_gemv` (T=1) | 12,288/16 x 16 warps | 768 | 12,288 | 3.12 |
      | `launch_q5_simt_r8_c8` (T=7..15) | 12,288/8 x 8 warps | 1,536 | 12,288 | 3.12 |
      | `GroupedMixedMmaR64C8` (T=7..32) | (4,096+12,288)/64 x 4 warps | 256 | 1,024 | 0.26 |

      **`launch_q5_split4_exact` throws above T=6**, and `launch_q5` therefore drops to
      `simt_r8_c8` at exactly width 7 — which is exactly where the `independent` column falls off a
      cliff in the schedule bench: **188.4 µs at T=6 to 330.8 at T=7**, +142 µs for one extra
      column. Nothing else changes at that boundary (the q4 half switched to R8C8 back at T=4), so
      that +142 µs *is* the split4 → r8c8 switch.

      So when this cycle measured "extending the direct route to `{{1, 15}}`" and recorded
      −3.6% at width 7 and −23.8% at width 9, **it was measuring `simt_r8_c8`, not split4** — the
      plan's own comment says so ("`launch_q4` and `launch_q5` both carry a dedicated R8C8 route for
      5..15"). The high-parallelism kernel was never tested above 6. If split4 held anything like
      its T=6 cost at T=7-8 it would land near 190-240 µs against `c8`'s 236-238, i.e. competitive
      to better, **and it would remove the DFlash2 5→6 cliff outright**, because that cliff *is*
      this boundary.

      `q5_rowsplit_gemm_simt_split4_kernel` is already generic in its column count — `acc[kTt]`,
      `s_part[4][kTt]`, and a `tt` loop that reads `x + tt * kStride` — so widening it is adding
      template instantiations to `launch_q5_split4_exact`, not writing a kernel. **One constraint
      to respect:** it carries `__launch_bounds__(128, 10)`, which caps it at 51 registers per
      thread, and `acc[kTt]` grows one register per column. Past roughly T=10 that will spill to
      local memory and look like a loss for the wrong reason. Relax `MIN_BLOCKS` to 6 when widening
      — at 12.5 machine-fulls of blocks this kernel does not need ten blocks per SM to fill the
      machine, so trading occupancy for registers is close to free here.

      **Do this before the MMA split-K kernel.** It is a handful of template instantiations against
      a new kernel, it is measurable with the existing bench, and it tests the same hypothesis.

      *Why the parallelism argument is not the whole story, so nobody over-reads it.* `simt_r8_c8`
      has 3.12 machine-fulls against the grouped tile's 0.26 and still **loses** at width 8 (319.5
      against 236.5), because a SIMT kernel does far more work per weight than a tensor-core one.
      So warps alone do not win: the target is a kernel with the MMA's per-warp efficiency *and*
      split-K's parallelism, which is precisely why the answer is split-K on the **MMA** path and
      not simply "use the SIMT route".

- [x] **Dense decode's GEMV kernels were limited by memory-pipe *instruction issue*, not
      bandwidth. Widening the consume loop shipped 2026-09-10: +3.90% on the 27B, L1/TEX 87% → 37%.**

      The consume loop read **one** code byte per lane per group and so spent four memory-pipe
      instructions to produce two weights — a 1-byte `tn`, a 1-byte `th`, a 2-byte broadcast `tsc`
      and a 4-byte `x`. Sixteen groups per tile is 64 instructions on the consume side against
      three on the staging side, so consume outweighed staging roughly 20 to 1, and that is what
      `Mem Pipes Busy` was measuring.

      Now each lane takes a **4-byte code word — 8 nibbles, 8 weights** — so eight lanes cover a
      group's 32 code bytes and 32 lanes cover four groups per step. Per step: one 4-byte `tn`, one
      1-byte `th`, one 2-byte `tsc`, one 16-byte `x`. **Four instructions per four groups against
      sixteen: a 4x cut, for identical arithmetic.**

      | kernel | duration | DRAM % | L1/TEX % | Mem Pipes Busy % |
      |---|---:|---:|---:|---:|
      | `<5120, 6144>` | 46.18 → **42.59** (−7.8%) | 50.10 → 54.34 | 86.60 → **36.71** | 81.04 → **33.64** |
      | `<5120, 17408>` | 116.29 → **108.35** (−6.8%) | 55.80 → 59.77 | 89.85 → **35.77** | 85.59 → **33.81** |
      | `<12288, 5120>` | 88.74 → **80.58** (−9.2%) | 51.75 → 58.88 | 90.76 → **39.94** | 83.09 → **36.71** |

      **L1/TEX throughput collapses from 87-91% to 36-40% — the predicted 4x — and DRAM utilisation
      goes *up*.** That is the whole point: the kernel stops being pipe-limited and gets closer to
      actually streaming its weights. It is still not DRAM-bound at 54-59%, so there is more here.

      End to end, `run_interleaved_ab.py`, clocks locked, six paired repetitions:

      | model | paired median | positive | range |
      |---|---:|---:|---|
      | 27B dense | **+3.90%** | **6 / 6** | +3.21% to +4.31% |
      | 35B MoE | −0.34% | 2 / 6 | −1.72% to +2.00% |

      The 35B showing nothing is the expected control-by-shape: its decode is dominated by the
      `sparse_moe` expert kernels, not by this GEMV. **There is no true `--control` for this one and
      that is stated rather than hidden** — the change is inside `q5_rowsplit_gemv`, which every
      text model reaches, so no configuration has an identical code path in both arms. The
      substitute is the drift floor the two earlier A/Bs established on this box: with
      `nvidia-smi -lgc 1500` their controls moved 0.00% and +0.09% over six pairs, against a 3.9%
      effect that the kernel counters predicted independently.

      **Why 3.9% end to end and 6.8-9.2% per kernel.** §2c's decode profile has 86% of the 27B's
      *busy* time in four GEMV kernels, but only some of those four are `q5_rowsplit_gemv` — the
      rest are `q4_linear_swiglu_gemv_pair` and friends, which have the same byte-granular shape
      and were not touched. **So the same widening applied to the q4 SwiGLU GEMV pair is the
      obvious follow-up**, and it should be worth something similar.

      Correctness: every Op that reaches this GEMV passes — `linear_q5_a16`, `linear_add_q5_a16`,
      `gdn_input_proj`, `attn_input_proj`, `linear_swiglu_q4_a16` — and the packing convention is
      unchanged (code byte b holds the weights at 2b and 2b+1; high-plane byte h bit j carries bit
      4 of the weight at 8h+j), verified on the host to cover all 1,024 weights of a tile exactly
      once. The new code index works out to `step * 32 + lane`, so the shared reads stay
      conflict-free.

      *The file header claimed the opposite and was wrong.* It said "all weight reads are fully
      coalesced 128-bit loads, so the kernel runs DRAM-bound instead of L1/LSU- or latency-bound".
      True of the global staging, false of the kernel: DRAM was at 50% while the pipe was at 81%.
      Corrected in place.

- [x] **The q4 SwiGLU GEMV pair got the same widening: +1.95% on the 27B, shipped 2026-09-10.**
      It was the worse of the two kernels — one byte of gate codes, one byte of up codes and two
      2-byte scale broadcasts produced four weights, so **five** memory-pipe instructions per group.
      Now five per *four* groups. Q4 has no high plane, so the mapping is simpler than the Q5 case:
      code byte b holds the weights at 2b and 2b+1, so weight k0+j is nibble (j & 1) of byte
      (j >> 1). Host-verified to cover all 1,024 weights of a tile exactly once, code index again
      `step * 32 + lane` so the shared reads stay conflict-free.

      Measured against the q5 widening as its baseline, so the gain is attributable to this change
      alone — same harness, clocks locked, six paired repetitions:

      | model | paired median | positive | range |
      |---|---:|---:|---|
      | 27B dense | **+1.95%** | **6 / 6** | +1.81% to +2.33% |
      | 35B MoE | +0.31% | 4 / 6 | −1.22% to +1.42% |

      **So the two widenings together are worth about +5.9% compounded on the 27B dense decode**,
      which is the largest single-model gain this file records. The 35B is unmoved by either, as
      expected: its decode is the `sparse_moe` expert kernels.

- [x] **Re-profiled after both widenings, and the constraint moved — differently for each kernel.
      One is now at the bandwidth ceiling; the other became compute-bound.** Measured 2026-09-10,
      `ncu`, clocks locked, six launches each, medians:

      | | `q4_linear_swiglu_gemv_pair` | `q5_rowsplit_gemv` |
      |---|---:|---:|
      | **DRAM throughput** | **85.26%** | 55.83% |
      | L1/TEX throughput | 42.10% | 36.84% |
      | **Compute (SM) throughput** | 76.15% | **73.31%** |
      | stall: `wait` | **17.84%** | 5.88% |
      | stall: `long_scoreboard` | 6.53% | 10.37% |
      | stall: `barrier` | 3.39% | 6.39% |

      **The q4 SwiGLU pair is essentially done: 85.26% of DRAM.** That is the state this whole file
      is aiming at — a weight-streaming kernel actually limited by the memory it has to read. There
      is no meaningful headroom left in it, and anyone who "optimises" it further will be trading
      against bandwidth.

      **The q5 GEMV is now compute-bound, not memory-bound: SM throughput 73.31% against DRAM
      55.83%.** Removing the memory-pipe bottleneck promoted the dequantization arithmetic to the
      constraint — the per-weight `sign_extend`, int-to-float conversion and `fmaf` chain. That is a
      different kind of problem from everything else in this file and wants a different fix: fewer
      or cheaper operations per weight (a `__dp4a`-style integer dot product, or the s8 tensor cores
      the groupwise-int prefill path already uses), not better memory scheduling.

      Note the stall totals are now *low* on both — the largest single reason is 17.84% — where the
      pre-widening kernel had 81-86% of a pipe busy. Neither kernel is latency-starved any more,
      which retires the `kStages` pipeline-depth candidate this entry named before profiling: at
      these utilisations there is nothing for a deeper pipeline to hide.

- [x] **The q5 GEMV's dequantization arithmetic was the last identified decode constraint. Fixed
      2026-09-10 with the in-tree half2 bit-trick plus a per-group scale hoist: +2.67% on the 27B.**

      Both halves are exact, not approximations, which is what made this safe to ship:

      - **The bit-trick** is `Q5SimtDecodeAtom::decode_eight`'s, already used by
        `q5_rowsplit_gemm_simt_split4_kernel`. half `0x6400` is 1024.0, whose mantissa LSB at that
        exponent is exactly 1.0, so OR-ing a nibble into the mantissa *adds* it; inverting the high
        bit and subtracting 1040.0 yields `nibble - 16 * high_bit`, which is precisely what
        `sign_extend<5>(nibble | high << 4)` computes. **Verified on the host for all 32 five-bit
        codes.** It replaces a per-weight sign-extend and int-to-float convert with one `hsub2` and
        one `__half22float2` per *pair*.
      - **The scale hoist** turns `sum_j((q_j * scale) * x_j)` into `sum_j(q_j * x_j) * scale`,
        removing seven of every eight multiplies. It rounds *less*, not more — the scale is applied
        to one accumulated value instead of eight products — and the scale is already per-group, so
        nothing is being approximated.

      | model | paired median | positive | range |
      |---|---:|---:|---|
      | 27B dense | **+2.67%** | **6 / 6** | +2.28% to +2.80% |
      | 35B MoE | −0.65% | 2 / 6 | −2.56% to +2.00% |

      *One correction to what this entry originally proposed.* It named `__dp4a` as the first
      direction. **That is wrong for this kernel**: `__dp4a` is int8 x int8, and this is an A16 path
      where the activations are BF16 — using it would mean quantizing `x`, which is a different
      policy (`AllowA8Int`) and a different numeric contract, not an optimisation of this one. The
      technique that does apply was already in the tree for the same codec.

- [ ] **What is left after five speedups: the q5 GEMV now stalls on memory *latency*, and the
      obvious cure was tried and does nothing.** The dense decode is at **73.9% of the 854.2 GB/s
      the card can read**, up from 69.0%. The q4 SwiGLU pair is at 85.3% of DRAM and effectively
      finished. The q5 GEMV was re-profiled after the dequant fix:

      | metric | after widening | after dequant fix |
      |---|---:|---:|
      | DRAM throughput | 55.83% | **66.67%** |
      | L1/TEX throughput | 36.84% | 46.47% |
      | Compute (SM) throughput | 73.31% | 68.54% |
      | **stall: `long_scoreboard`** | 10.37% | **27.26%** |
      | stall: `wait` | 5.88% | 8.13% |

      Both earlier constraints are gone — DRAM is up eleven points, compute is down five — and what
      surfaced is **global-memory latency**: `long_scoreboard` went 10.37% → 27.26%. With the
      consume loop cheap, the kernel drains tiles faster than a two-stage `cp.async` pipeline
      prefetches them.

      **So `kStages = 3` was the obvious cure, and it is worth exactly nothing. Measured
      2026-09-10: 40.12 tok/s against 40.11 at `kStages = 2`**, inside the ±0.04-0.07 run spread, on
      the same clock-locked `-pg 4096,128`. Reverted, because three stages are not free — 42,496 B
      of shared against 31,744 costs a block per SM, so it buys a regression in occupancy for no
      throughput. Four stages do not fit under the 48 KB static cap while `kStageX` is on.

      That is the third plausible mechanism this cycle to be built, measured and thrown away (the
      MoE D3 block split and split4-at-width-7 were the others), and the pattern is worth naming:
      **a stall percentage says where the warps are waiting, not that removing the wait will make
      the kernel faster.** Something else — most likely the block count, since these are
      narrow-extent kernels and §2c's parallelism entry applies here too — takes up the slack.

      The remaining 26.1 points to the ceiling are worth 54.3 tok/s if ever fully closed, which is
      the honest shape of what is left. Nothing in this file suggests that is reachable; the point
      of the number is to stop the next person quoting 936.2 GB/s. Next probe, if anyone wants it:
      the launch geometry, not the pipeline — `kRowsPerBlock` and the resulting waves per SM, read
      against the `(rows/WM) x (BN/WN)` arithmetic that explained the GDN kernel.

      This is the missing explanation for "the dense path sits around two-thirds of what the card
      can deliver". `q5_rowsplit_gemv` on the 27B decode, from the same elevated run:

      | | |
      |---|---:|
      | Memory throughput | 456.39 GB/s |
      | DRAM throughput | **50.10%** |
      | **Mem pipes busy** | **81.04%** |
      | **L1/TEX throughput** | **86.60%** |
      | Mem busy (subsystem) | 40.24% |
      | L2 throughput | 22.72% |
      | achieved occupancy | 57.64% |

      `ncu`'s own note: *"utilizing greater than 80.0% of the available compute or memory
      performance... work will likely need to be shifted from the most utilized to another unit."*
      The most-utilised unit is the **L1/TEX path at 86.6%**, while DRAM delivers half of what it
      could. **This kernel is not weight-streaming-bound. It is issue-bound on the memory pipe.**

      **Where the instructions go, read out of the source.** Q5 is stored as three separate planes
      per group of 64 weights — `codes` 32 B, `high` 8 B, `scales` 2 B
      (`q5_rowsplit_storage.cuh`). The staging side is already good: `q5_gemv_issue_tile` moves a
      16-group tile in three `cp.async`s. The **consume** side is where the cost is. Per group, per
      warp, `q5_gemv_consume_tile` issues:

      | access | width | bytes moved |
      |---|---|---:|
      | `tsc[tg]` | 2 B, broadcast to 32 lanes | 2 |
      | `tn[tg * 32 + lane]` | **1 B per lane** | 32 |
      | `th[tg * 8 + (lane >> 2)]` | **1 B**, 4-way broadcast | 8 |
      | `x2[k0 >> 1]` | 4 B per lane | 128 |

      **Four memory-pipe instructions per group to produce two weights per lane** — 16 groups per
      tile is 64 instructions on the consume side against three on the staging side, so the consume
      loop outweighs staging roughly 20 to 1. That is what 81% Mem-Pipes-Busy at 50% DRAM looks
      like.

      The fix needs **no artifact change**, which is the good news — the planes stay exactly as they
      are on disk. Have each lane decode eight weights per group instead of two, reading four bytes
      of `codes` at a time: `Q5SimtDecodeAtom::decode_eight` **already exists in the tree** and
      takes precisely `(uint32 packed, uint8 high_bits, uint16 scale_bits)`. Handling four groups
      per iteration across 32 lanes gives 256 weights for about four memory instructions instead of
      sixteen — a ~4x cut in memory-pipe instructions on a kernel family that is **86% of the 27B's
      decode busy time**.

      Two caveats before building. The 4-byte-per-lane read of `tn` is stride-4 across lanes, which
      is conflict-free in shared memory, but the `th` and `tsc` mappings need re-deriving and the
      swizzle must be re-checked. And `Mem Pipes Busy` on Ampere counts the shared-memory pipe
      alongside the global LSU, so the split between staging and consume above is *derived from the
      source, not measured*: confirm it with `l1tex__data_pipe_lsu_wavefronts_mem_shared` against
      `l1tex__data_pipe_lsu_wavefronts_mem_global` before committing to the retile.

- [x] **`sparse_moe_d2_warp_kernel` cost 8.45% of the 35B's decode kernel time on one warp of one
      SM. Both halves are now gone, and the second did not need warps.** The branch-free sort
      shipped 2026-09-09 for **+3.15%**; the branch-free *merge* shipped 2026-09-13 for **+2.39%**.
      Read the 2026-09-13 note at the end before the 8-warp design below -- that design is now
      mostly unnecessary.

      **Shipped 2026-09-09: a branch-free sorting network.** The stall breakdown below put 15.25% of
      d2's active warp cycles in `branch_resolving`, all of it the `while`-loop insertion sort over
      each lane's eight candidates — its trip count is data-dependent, so lanes diverge and the warp
      pays every path (22.19 of 32 threads active). Replaced with Batcher's odd-even merge sort for
      eight elements: 19 compare-exchanges in six dependency levels, every index a compile-time
      constant, emitted as predicated selects. Semantics are identical because
      `sparse_moe_ranked_better` is a total order over distinct expert ids, and the network was
      verified exhaustively on the host — **all 40,320 permutations, plus 20,000 tie-heavy cases
      under the (value, id) order.**

      | | before | after | change |
      |---|---:|---:|---:|
      | `sparse_moe_d2_warp` | 13.43 µs/inst | **8.32 µs/inst** | **−38.0%** |
      | total decode kernel time | 820.3 ms | **795.8 ms** | **−2.99%** |
      | `d1` / `d3` / `d4` (untouched) | | | −2.0% / +0.5% / −0.7% |

      End to end, `run_interleaved_ab.py`, clocks locked, six paired repetitions, **the dense 27B as
      control because it has no `sparse_moe` path at all**: **+3.15% paired median, 6 of 6 pairs
      positive, +3.29% normalised, control drift +0.00%.** That matches the −2.99% kernel-time
      prediction, which is the check worth having — the two instruments agree.

      It beat the estimate (~1% was predicted from `branch_resolving` alone) because the network is
      also shorter in dependent depth than the insertion sort's worst case, six levels against
      seven, so it took some of the `wait` term too. Note the control here is chosen better than the
      D3 experiment's: a model with none of the changed code, rather than a downstream kernel in the
      same fused pipeline, which moved 6% on L2 state alone.

      **What is left: the `wait` term — but "more warps" as this entry first described it would
      miss most of it, and that is worth reading before building anything.** Counting the routine's
      own ops per lane:

      | | ops |
      |---|---:|
      | local sort (the Batcher network) | 19 compare-exchanges |
      | merge: shuffles (5 xor steps x 8) | 40 |
      | merge: bitonic restores (5 x 3 stages x 4) | 60 |
      | **merge total vs sort** | **5.3x** |

      **The merge is the dominant half, and an 8-warp split parallelises the sort — the smaller
      half.** Splitting 256 experts over 8 warps at one score per lane removes the per-lane sort
      from the critical path, but each warp still has to produce a sorted top-8 and *something* must
      still merge eight of those, so the naive version keeps the expensive part serial and would
      measure as very little. That is the D3 mistake in a different costume: a plausible mechanism
      that does not touch the dominant term.

      **The design that does work needs both halves**, and the second is the cheap one:

      1. 8 warps, one score per lane, each warp reducing its 32 candidates to a sorted top-8 —
         a bitonic sort of 32 across lanes, 15 branch-free shuffle compare-exchanges, all 8 warps
         concurrent so the latency interleaves.
      2. **Merge 8 runs in 3 xor steps, not 5.** `sparse_moe_merge_ranked_runs` hardcodes
         `for (partner = 1; partner < 32; partner <<= 1)` because today it merges 32 runs. Eight
         runs need `log2(8) = 3`. That alone is 100 ops down to 60, **−40% of the dominant term**,
         and it is a template parameter on the existing verified routine rather than new logic.

      Critical-path ops: **119 today (19 + 100) against ~60** (the 30 for step 1 being spread over
      eight concurrent warps). So call it half the addressable work.

      **And the addressable work is not all 8.32 µs.** The smallest real kernel in the same nsys
      capture, `rmsnorm_cta_bf16`, is **2.23 µs/instance**, which is about the floor for a small
      launch on this box. So ~6.1 µs is addressable, halving it saves ~3 µs, and 3 µs x 5,160
      instances is ~15.5 ms of 795.8 — **roughly 2% end to end, not the 5-6% this entry claimed
      before the branch-free sort landed.** Still worth having, and worth knowing it is 2% before
      committing to a bitonic-32 that needs its own exhaustive verification.

      **2026-09-13: the 2% arrived without the bitonic-32, because the op count above was right
      about *which* half dominates and wrong about *why*.** It modelled the merge's 60 bitonic
      restores as work to spread across warps. They were branches. The restore stage was written
      `if (!better) { swap }` -- a data-dependent branch per exchange -- and
      `sparse_moe_ranked_better` used short-circuit `||` / `&&`, which nvcc is free to lower to
      branches too. On a kernel with one warp nothing covers a branch while it resolves. That is
      precisely the lesson the 2026-09-09 sort fix applied to the *sort*; the merge that consumes
      the sort's output was never given it.

      Decomposed first, because the D3 work had just shown a kernel-level win need not reach the
      model. Three probes on d2 itself, each verified to have actually relinked:

      | probe | d2 time | reads as |
      |---|---:|---|
      | empty kernel (launch + teardown) | 1.47 us | launch floor |
      | + read and consume every score, no selection | 3.26 us | loads cost 1.79 us |
      | full selection (shipped before) | 11.04 us | **selection arithmetic 7.78 us** |
      | full selection, branch-free merge | **5.73 us** | **selection arithmetic 2.47 us** |

      **The selection arithmetic got 3.1x faster**, from making 60 exchanges and one comparator
      select instead of branch, with no change to the algorithm, the network, the order, or the
      warp count. Register spills were ruled out on the way (0 local loads, 0 local stores, 46
      registers), which is what pointed at branches.

      **Checked that it converts before believing it.** A `clock64` spin of +12.8 us inside d2
      cost **-6.89%** end to end (4/4 pairs negative), against ~-9.7% if d2 were fully serial -- so
      d2 is ~71% on the critical path, about 0.54% of throughput per us per instance. The
      branch-free merge saves 5.3 us, predicting **+2.85%**. Measured on `master`, paired median of
      10: **+2.39%** (9/10 positive). An earlier run on another branch gave +2.53% (10/10). Three
      independent estimates within half a point of each other.

      **Output is bit-identical**, which it has to be -- the same comparisons in the same total
      order, evaluated unconditionally. Verified rather than argued: greedy decode on the 35B
      produces the same text from both builds, and `ninfer_sparse_moe_route_network_test` (all
      40,320 permutations and 6,561 tie patterns, which exercises the changed comparator) and
      `ninfer_sparse_moe_test` are green.

      **What is left is small and should not be chased hard.** 2.47 us of selection arithmetic
      remains, about 1.3% end to end at the measured sensitivity even if it went to zero, and the
      8-warp bitonic-32 above would take only part of that while needing its own exhaustive
      verification. The same fix also reaches prefill and small-T, which call the same
      `sparse_moe_select_top8_warp`; those were not measured separately.

      Two measurement traps from this session, both of which silently produced wrong numbers:
      (1) **the first attempt at a slowdown probe repeated the selection N times and measured
      identically to 0.01 us at N=1 and N=4** -- nvcc removed the redundant identical stores; a
      `clock64` spin is what cannot be elided. (2) **A `ninja` still running from another process
      left `build.log` locked, the build failed, and the old binary was measured as if it were
      new.** Compare the executable's mtime before and after every build, and refuse to measure if
      it did not change.

      One thing that makes step 1 cheaper than it looks: lanes 8..31 of the merging warp can hold
      runs of `{-inf, INT_MAX}`, which are trivially descending and can never be selected, so the
      *existing* 5-step routine works unmodified as a correctness fallback while the 3-step variant
      is being validated against it.

      Found incidentally in the MoE counter capture, where `ncu` timed it at 15.52 µs and the
      arithmetic in this entry's first draft argued that had to be inflated. **It was inflated by
      15%, not by 3x.** `nsys`, node-level graph tracing, 35B / int8 / `-n 128`, clocks locked:

      | stage | total | % of kernel time | µs/instance | instances |
      |---|---:|---:|---:|---:|
      | `sparse_moe_d3_nine_warp` | 134.89 ms | 16.44% | 26.14 | 5,160 |
      | `sparse_moe_d4_nine_warp` | 104.22 ms | 12.70% | 21.83 | 4,773 |
      | **`sparse_moe_d2_warp`** | **69.30 ms** | **8.45%** | **13.43** | 5,160 |
      | `sparse_moe_d1` | 19.32 ms | 2.36% | 3.74 | 5,160 |
      | all four | 335.88 ms | **40.9%** | | |

      5,160 instances is 129 rounds x 40 text layers, so the mapping is exact. **d2 is the third
      largest MoE stage, ahead of d1 by 3.6x, and it launches `<<<1, 32>>>` — one warp, on one SM of
      82, 2.09% achieved occupancy.** The earlier reasoning that got this wrong divided #53's "38% of
      busy time" across the stages and found no room for it; the four stages are 40.9% of *kernel*
      time here, and d1 is far cheaper than assumed, which is where the room was.

      **Why it is this slow is NOT yet established, and the obvious answer does not add up.**
      13.43 µs at 1,500 MHz is ~20,000 cycles to pick the top 8 of 257 floats from about 1 KB. The
      selection is not badly written: `sparse_moe_route.cuh` gives each lane 8 scores, insertion
      sorts them locally, then runs five xor-merge steps, each 8 independent shuffles followed by a
      3-stage bitonic restore. Costing that out — shuffles pipeline, the bitonic stages are
      dependent at ~20-30 cycles each, the insertion sort is a few hundred cycles even divergent —
      lands somewhere near **1,000 cycles, under a microsecond.** The measurement is twenty times
      that, so the instruction mix does not explain it and "a single warp has nothing to interleave"
      is at best a partial answer.

      What the counters do say (from the MoE capture, so `ncu`'s inflated 15.52 µs duration but
      trustworthy ratios): **no eligible warp on 78.56% of cycles**, avg. active threads per warp
      **22.19 of 32**, L2 hit rate 81.67%, DRAM throughput 0.60%. So the one warp is stalled for
      four cycles in five, and it diverges — but which stall dominates is exactly what has not been
      measured.

      **The stall breakdown was collected rather than guessed, and it settles which fix is right.**
      `ncu --metrics smsp__warp_issue_stalled_*_per_warp_active.pct` on `sparse_moe_d2`, three
      launches, clocks locked — the `per_warp_active` family works without PC sampling, which is
      what the MoE capture's missing `smsp__pcsamp_sample_count` had blocked:

      | stall reason | % of active warp cycles |
      |---|---:|
      | **`wait`** — dependent fixed-latency instructions | **37.68** |
      | **`branch_resolving`** — the divergent insertion sort | **15.25** |
      | `imc_miss` | 7.69 |
      | `short_scoreboard` | 4.15 |
      | **`long_scoreboard`** — global memory latency | **4.09** |
      | `barrier`, `lg_throttle` | 0.00 |

      **Memory latency is 4% and is therefore not the problem** — which kills the most plausible of
      the three candidates. Staging `scores` through shared memory or fusing into d1's tail would
      buy essentially nothing. `wait` plus `branch_resolving` are **52.9% between them**, and both
      are what a single warp running a dependent, divergent chain looks like with nothing to
      interleave.

      So the fix is the one the counters point at, and it is two changes to the same routine:

      1. **More warps in the block**, to cover the 37.68% `wait`. One block still means one SM — the
         win is interleaving *inside* it. A block-scoped selection with 8 warps (each taking 32 of
         the 256 experts, one score per lane, a local top-8, then one cross-warp merge through
         shared memory) both shortens the dependent chain and gives the scheduler seven other warps
         to issue from.
      2. **A branch-free sorting network** for the per-lane sort, to remove the 15.25%
         `branch_resolving`. The current code is a `while`-loop insertion sort over 8 elements,
         which is where the 22.19-of-32 active threads comes from.

      On the ~1,000-cycle instruction cost that arithmetic gives, 4-5 µs is the target, i.e.
      **5-6% of the 35B's decode time** — larger than anything else left in this file and far
      cheaper than the split-K MMA kernel.

      **Two cautions before writing it.** The ranking is a total order over distinct expert ids and
      `sparse_moe_route.cuh` says so, so any correct selection must produce the same set *and the
      same order*; `ninfer_sparse_moe_test` at T=1 is the check. And
      `sparse_moe_select_top8_warp` is shared with the prefill and small-T paths
      (`sparse_moe_prefill_kernels.cu:153`, `sparse_moe_small_t_kernels.cu:103`), which call it per
      token from a wide launch and are *not* latency-starved — so add a block-scoped variant
      alongside it rather than replacing it.

      That the mechanism was measured rather than assumed matters here specifically: the D3 block
      split on this same page was built on an equally plausible story, measured, and came out
      *worse*. This time the losing candidate was eliminated before any kernel was written.

      Two cautions. The ranking is a total order over distinct expert ids and the file says so, so
      any correct selection must produce the same set *and the same order* — `ninfer_sparse_moe_test`
      at T=1 is the check. And `sparse_moe_select_top8_warp` is shared with the prefill and small-T
      paths (`sparse_moe_prefill_kernels.cu:153`, `sparse_moe_small_t_kernels.cu:103`), which call it
      per token from a wider launch and are *not* latency-starved; a block-scoped variant should be
      added alongside it rather than replacing it.

### Found by this cycle's profiling, and not previously on this list

The two entries below are the largest identified speed opportunities in the repository. Both come
out of #53's profile and #50's byte accounting, both have a measured gap against a measured
ceiling, and neither has had any optimisation attempted.

- [x] **The MoE expert gather is latency-bound, and there is nothing left to do about it. Closed
      2026-09-09** after all four candidate fixes were measured and none worked — the last of them,
      splitting the nine-warp block, made it actively worse. The 35B's ~51% of achievable is what
      top-8-of-256 routing costs on a card whose per-layer MoE work is a single machine-full, and
      this entry is kept because the four negatives are what stop the investigation being repeated. `ncu` counters collected
      2026-09-09 via `scripts/sweeps/admin-profile.ps1` (elevated; the counters are
      administrator-only on Windows and that was the whole blocker), 35B, int8 KV, decode, clocks
      locked at 1,500 MHz, `--graph-profiling node`:

      | | `d3_nine_warp` (gate_up) | `d4_nine_warp` (down) |
      |---|---|---|
      | duration | 28.45 µs | 25.18 µs |
      | memory throughput | 425.9 GB/s (**49.9%** of achievable) | 487.5 GB/s (**57.1%**) |
      | DRAM throughput | 46.79% | 53.56% |
      | compute (SM) throughput | 35.69% | 36.55% |
      | **avg. active threads per warp** | **31.11 / 32** | **29.06 / 32** |
      | L1/TEX hit rate | 31.67% | 87.58% |
      | L2 hit rate | 5.89% | 15.96% |
      | **schedulers with no eligible warp** | **44.57%** | **58.63%** |
      | eligible warps per scheduler | 1.62 (of 8.28 active) | 1.13 (of 10.25 active) |
      | achieved / theoretical occupancy | 60.16% / 93.75% | 78.91% / 93.75% |
      | **block limit: registers** | **5** | **5** |
      | block limit: shared memory | 12 | 7 |
      | registers per thread | 36 | 40 |
      | waves per SM | **1.25** | 5.00 |

      This entry listed three candidate causes. The counters settle all three.

      **Address divergence — ruled out.** 31.11 and 29.06 active threads per warp out of 32. The
      8-of-256 expert selection happens at *block* granularity, so lanes within a warp still walk
      one expert's weights contiguously. There is no lane divergence to fix, and that was the
      leading hypothesis.

      **L2 behaviour — real, inherent, and not a bug.** 5.89% and 15.96% hit rates are close to
      zero reuse, but an expert's weights are read once per token and there is nothing to hit. No
      tuning recovers this; it is what top-8-of-256 costs.

      **Too little work in flight — confirmed, and this is the answer.** Neither DRAM (46.79/53.56%)
      nor SM (35.69/36.55%) is anywhere near saturated, which is the textbook signature. The
      schedulers say it directly: **no warp is eligible to issue 44.6% of cycles on d3 and 58.6% on
      d4**, with only 1.1-1.6 eligible warps per scheduler against 8-10 resident. Warps are there
      and stalled, not absent.

      **The binding constraint is the 9-warp block, and it is structural.** `Block Limit Warps` is
      **also 5** — the same as `Block Limit Registers` — so registers are tied with the warp limit,
      not binding beyond it, and cutting them buys nothing on its own. sm_86 allows 48 warps per
      SM; a 9-warp block gives `48 / 9 = 5` blocks and 45 resident warps, which is exactly the
      93.75% theoretical the counters report.

      Nine warps is not a tuning choice. In both kernels warps 0..7 each take one of the top-8
      routed experts through `RoutedCodec`, and **warp 8 takes the shared expert** through
      `W8Codec` — `kTopK + 1`. Eight warps would have nowhere to put the shared expert. So the
      6.25% of theoretical occupancy lost to 45-of-48 warps is the price of top-8-plus-shared, and
      it is the small part anyway.

      **What actually costs d3 is the wave tail, and the arithmetic matches to within 2 points.**
      512 blocks (one per `kIntermediate` column) over 82 SMs at 5 blocks each is 410 slots: one
      full wave of 410, then a second wave only 102 blocks deep, i.e. **25% full**. Average
      occupancy across the two waves is `(410 + 102) / (2 x 410) = 62.4%` against a measured
      **60.16%**. d3's occupancy shortfall is that tail and essentially nothing else. d4 launches
      2,048 blocks (5.00 waves), where a tail of the same absolute size is a fifth as costly, and it
      duly reaches 78.91%.

      **And there is a load imbalance inside the block that the source already names.** The shared
      W8 path is heavier than a routed Q4 path, but the block cannot retire until all nine warps
      finish, so eight completed routed warps sit holding registers and warp slots while warp 8
      drains. That is the 44.6% / 58.6% "no eligible warp" directly, and it is not a guess:
      `sparse_moe_d3_path_tiled_kernel` in the same file exists for this reason and says so —
      *"Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
      heavier shared W8 path from holding eight completed routed warps resident."* That kernel is
      written for the multi-token path and takes a `tokens` argument; single-token decode does not
      use it.

      So the order to try things, revised — and the first item is a refutation of the obvious fix.

      1. **Launch geometry does not help, and no block shape does.** d3's 512 blocks over 82 SMs at
         5 blocks each is 410 slots, which reads like a fixable tiling problem. It is not: total
         work is 512 columns x 9 warp-paths = **4,608 warp-units against the card's 3,936 warp
         slots**, so the kernel is only **1.17 machine-fulls of work** and every partitioning gives
         two waves. Efficiency `N / (slots x ceil(N/slots))` comes out at 62.4% for 9-warp blocks,
         58.5% for 3-warp, 58.5% for 1-warp. There is no scheduling trick that makes a 1.17-full
         kernel efficient. **d3 is not badly written; it is too small for this card at one token.**

      2. **Batching does not rescue it either — measured, and this refuted a prediction.** If the
         gather were merely work-starved, a cohort should fill the machine and the MoE should scale
         *better* with concurrency than a dense model. Measured 2026-09-09, mtp0, int8 KV, 512
         decode tokens, both models through `run_serve_concurrency.py`:

         | C | 35B MoE tok/s | vs C1 | 27B dense tok/s | vs C1 |
         |---|---|---|---|---|
         | 1 | 164.4 | 1.00x | 36.8 | 1.00x |
         | 2 | 273.2 | 1.66x | 61.1 | 1.66x |
         | 4 | 400.0 | 2.43x | 101.0 | 2.75x |
         | 8 | 529.7 | **3.22x** | 132.9 | **3.62x** |

         The MoE scales *worse*, and the reason is structural rather than a kernel defect. **A dense
         model amortises its weight read across the cohort — the same bytes serve every lane — but
         an MoE with per-token routing does not.** Each token brings its own top-8 of 256, so the
         expected number of distinct routed experts a round must read grows almost linearly with
         the cohort: 8.0 at C1, 15.8 at C2, 30.5 at C4, **57.4 at C8** — `7.18x` the routed weight
         bytes for `8x` the tokens. Only the shared expert and the dense layers amortise. The
         marginal cost per added row bears it out: 1.29 ms/row on a 6.08 ms base for the MoE (21%
         of base) against 4.71 ms/row on 27.20 ms for the dense model (17%).

         This is worth stating plainly because it is the opposite of the usual intuition about
         batching, and it means **the 35B's ~51% of achievable is not a bug to be fixed by
         batching or by tiling.** It is what top-8-of-256 routing costs on a card whose per-layer
         MoE work is a single machine-full.

      3. **Split the shared path out of the block — built, measured, and it is worse. Closed
         2026-09-09.** This was the last standing candidate and the entry argued it was exempt from
         item 1 ("this does not add work to the machine ... it removes a serialisation inside a
         block that is already resident"). That argument is wrong, and item 1 is what beats it.

         Wired `sparse_moe_d3_path_tiled_kernel` into single-token decode at `PathsPerBlock = 3` —
         which is what the multi-token small-T path already routes *every* token count 2..46 to —
         behind a plan schedule so both geometries ran from one build. At one token the two kernels
         write an identical activation layout (`(0 x (kTopK+1) + path) x kIntermediate + j` is
         `warp x kIntermediate + j`), and on sm_86 `NINFER_SM8X_COMPAT` compiles every `pdl::` call
         to nothing, so this was a pure launch-geometry A/B with no numeric or ordering difference.
         `ninfer_sparse_moe_test` passes on both arms, so the reroute was correct.

         35B / int8 / decode, clocks locked at 1,500 MHz, `ncu` kernel replay, medians of 3:

         | | `nine_warp` (512x1, 288 thr) | `paths3` (512x3, 96 thr) |
         |---|---:|---:|
         | duration | **29.63 µs** | **30.75 µs** |
         | block limit registers / warps | 5 / 5 | 16 / 16 |
         | theoretical warps per SM | 45 | **48** |
         | theoretical occupancy | 93.75% | **100%** |
         | **achieved occupancy** | **67.65%** | **52.60%** |
         | no eligible warp | **43.2%** | **53.4%** |
         | DRAM throughput | 50.6% | 41.3% |

         **It delivers exactly the higher *theoretical* occupancy it was designed for — 93.75% to
         100% — and a materially lower *achieved* one, and the scheduler stall it was meant to
         remove gets worse.** That is the hypothesis inverted, not merely unmet.

         The cause is the wave tail, and it is item 1's arithmetic: three-warp blocks need
         **1,536 blocks against a 16-per-SM capacity of 1,312**, so two waves at **58.5%**
         efficiency, against 512 blocks over 410 slots at **62.4%**. Item 1 already computed those
         two numbers and this entry set them aside on the grounds that no work was being added. The
         mistake was treating "resident" as free: spreading a kernel that is only ~1.17
         machine-fulls over three times as many blocks costs more in tail than the intra-block
         serialisation costs in stall. **Removing a serialisation does not help a kernel that is
         smaller than the machine.**

         Reverted — the plan carries no schedule and the default is unchanged — with the numbers
         recorded above `sparse_moe_d3_nine_warp_kernel` so it is not retried. The precondition for
         ever revisiting it is making the kernel bigger than the machine, which for an MoE at one
         token nothing in items 1 and 2 offers.

         **One measurement lesson, and it is the transferable part.** `d4` was captured alongside as
         an unchanged control and **moved 6.14% between the two arms** (DRAM 46.9% → 52.4%) —
         larger than the d3 effect being measured. A downstream kernel in the same fused pipeline is
         **not a clean control**: d3's different access pattern leaves different L2 state behind for
         d4. So the duration column above is inside its own control's drift and cannot carry this
         result on its own; the occupancy and scheduler counters, which are not drift-sensitive in
         the same way and which moved by 15 and 10 points, are what settle it. Pick controls that do
         not share a cache with the thing under test.

      4. **Over-fetch — measured with `dram__bytes_read.sum`, and it is real but smaller than the
         section reading suggested.** Collected 2026-09-09, `admin-profile.ps1` section 4, four
         launches each, kernel replay:

         | | d3 | d4 |
         |---|---:|---:|
         | `dram__bytes_read.sum` | **11.16 MB** (11.16/11.16/11.17/11.16) | **~10.2 MB** (10.24/10.27/11.14/10.14) |
         | useful weight bytes (#50) | 8.91 MB | 5.58 MB |
         | over-fetch | **1.25x** | **1.83x** |
         | previous estimate from the section reading | 1.36x | 2.20x |

         So d4 does move nearly twice the bytes it needs, but 1.83x rather than 2.20x, and this
         entry's "worth more than anything else here" no longer holds — 1.83x of 5.58 MB is 4.6 MB
         of waste on a kernel whose DRAM pipe is only 47% utilised, so the bytes are not what the
         kernel is waiting for. **Asking for the metric rather than trusting the section was the
         right call**: `MemoryWorkloadAnalysis_Tables` produced *"No metrics to show"* in both the
         application-replay and kernel-replay captures, so the 12.12/12.27 MB figures came from a
         section that had not actually reported and should not have been quoted.

         **What the sector counters show instead is an L1 amplification, and it is d4's alone.**

         | | d3 | d4 |
         |---|---:|---:|
         | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 528,384 | **2,560,000** |
         | that as bytes (32 B/sector) | 16.9 MB | **81.9 MB** |
         | `lts__t_sectors_srcunit_tex_op_read.sum` as bytes | 11.6 MB | 8.3 MB |
         | DRAM read | 11.16 MB | 10.2 MB |
         | L1/TEX hit rate | 31.66% | **87.60%** |

         d4 issues **81.9 MB of L1 load traffic to move 10.2 MB from DRAM — 8x** — and its 87.6% L1
         hit rate is what absorbs it. That is the activation vector: `act` is 9 x 512 floats, and
         every one of d4's **2,048 blocks** re-reads it (`act + warp * kIntermediate`) for each of
         its nine warps, which is `2,048 x 9 x 512 x 4 B` = 37.7 MB of the total on its own. It is
         not currently the binding constraint — d4's L1/TEX pipe is at 42.4%, below its 47.1% DRAM
         — so this is not a fix to make today. It is worth knowing before anyone reads d4's 87.6% L1
         hit rate as evidence of good locality: it is evidence of a re-read that L1 happens to
         forgive.

         **Re-profiled 2026-09-13 on merged master, and the stall breakdown changes the target.**
         The numbers above all reproduce exactly -- d4 still issues 2,560,000 load sectors (81.9 MB)
         to move 10.27 MB, still 8x -- but nobody had looked at *why* the warps are stalled, only
         that they are. Per issue-active cycle, average warps stalled:

         | stall reason | d3 | d4 |
         |---|---:|---:|
         | `barrier` | 0.57 | **10.05** |
         | `long_scoreboard` (global latency) | **7.54** | 5.47 |
         | `wait` | 0.75 | 1.41 |
         | `lg_throttle` | 0.14 | 1.54 |

         **d3 and d4 are not the same problem and should stop being treated as one.** d3 is global
         memory latency, 7.54 of its ~9.3 stall budget -- the activation-reuse and prefetch family
         of fixes is aimed correctly there. d4 spends **over half its stall budget waiting at
         `__syncthreads()`**, which is a synchronisation-pattern problem inside the 9-warp block,
         not a memory problem. That is worth stating plainly because items 1-3 above were all
         closed as measured negatives and all three attacked *occupancy* -- launch geometry,
         batching, block splitting. None of them attacked the barrier pattern, so "all four
         candidate fixes are measured negatives" overstates what was actually ruled out.

         **I raised a caution here on 2026-09-13 and then tested it, and the caution was wrong.**
         It said that "d4's DRAM pipe is only 47% utilised, so the wasted bytes are not what the
         kernel is waiting for" was the same argument #90 overturned on the 27B, so low pipe
         utilisation should not be what retires the over-fetch. Fair as an argument, and false
         here. `Rows` is already a template parameter on `sparse_moe_d4_nine_warp_kernel` and each
         row re-reads the same activation slab, so raising it amortises exactly this over-fetch.
         Swept it:

         | Rows | d4 time | L1 load sectors |
         |---:|---:|---:|
         | 1 (shipped) | 19.97 us | 2,629,632 |
         | 2 | 19.78 us | 1,431,552 |
         | 4 | 20.38 us | 832,512 |

         **A 3.2x cut in L1 traffic bought nothing.** Rows=4 is slower -- it drops the grid from
         2,048 blocks to 512 and loses more in latency hiding than it saves in traffic. Rows=2
         measured +0.26% end to end (paired median of 6, min -0.02%, crossing zero), which is
         noise. Reverted. The original conclusion in this entry stands on its own evidence now,
         not just on the DRAM percentage: the kernel waits on the scattered *weight* gather, and
         the activation re-read is not on that dependency chain. #90's lesson does not transfer,
         because there the redundant movement *was* the operand the kernel waited for.

         Denominator trap, since this bit me while re-measuring: `dram__throughput` reads **100%**
         on both kernels with `.avg.pct_of_peak_sustained_active` and 47.40 / 51.63% with
         `.avg.pct_of_peak_sustained_elapsed`. The elapsed form is the one that matches the
         numbers in this file. Quote the wrong one and these kernels look bandwidth-saturated.

         **The 8-of-256 gather is not the remaining headroom either, and d3 is now done.**
         Measured 2026-09-13, two results.

         The access pattern is already perfect. On d3, `dram__sectors_read` x 32 B equals
         `dram__bytes_read` exactly -- 348,932 sectors, 11.17 MB -- so every sector fetched is
         fully used. There is no coalescing loss, no wasted traffic and no weight-layout win
         available; L2 hit is 6.51%, which is what compulsory reads of weights used once look
         like. The entire gap to the 13.1 us bandwidth floor is latency hiding.

         And the latency-hiding lever works exactly as designed while buying nothing. d3's 9-warp
         block strands 3 of 48 warps and caps the grid at 1.25 waves, which is why its SMs sat idle
         26% of elapsed. Folding the shared expert's K range across the eight routed warps makes
         the block 8 warps:

         | | nine_warp | eight_warp |
         |---|---:|---:|
         | warps per SM | 45 of 48 | 48 of 48 |
         | blocks per SM | 5 | 6 (410 -> 492 slots) |
         | waves per SM | 1.25 | 1.04 |
         | achieved occupancy | 77.13% | **87.08%** |
         | duration | 23.87 us | **21.82 us** |

         End to end that is **-0.03% at C1 and +0.11% at MTP3**, both crossing zero over six paired
         repetitions. Under replay the four MoE kernels total 57.05 -> 54.97 us, so the time is
         real; it does not convert because these kernels overlap under PDL, and d3 finishing 0.8 us
         earlier does not let d4 start 0.8 us earlier. Reverted.

         **This is the sharpest form of the "a stall percentage is not a speedup" rule in this
         file: a faster kernel is not a faster model, and ncu replay cannot tell you which one you
         have -- only the end-to-end paired A/B can.** Three d3 geometries are now measured
         negatives (3-warp split, Rows>1, 8-warp fold). Stop tuning d3.

         What is left on this Op is `sparse_moe_d2_warp_kernel`: ~8.3 us of the ~57 us the four
         kernels cost, running `<<<1, 32>>>` -- one warp of one SM, 81 idle, and d3 cannot start
         without its ids. That is a parallel-selection problem, not a memory one.

- [x] **The shared-expert warp was the straggler, and widening its consume loop is worth +6.47%
      on the 35B. Shipped 2026-09-13.** Found by taking the stall breakdown above and then asking
      what warp 8 actually runs. Both d3 and d4 give warps 0-7 the routed experts on an
      eight-values-per-lane consume loop, and warp 8 the shared expert on a much narrower one:
      `dot_fp32_rows` sent W8 to `load_pair` (two values per lane, and `lane < kGroupK / 2` is
      **16 of 32 lanes idle** at kGroupK=32), and `dot_two_rows` sent it to `load_one`, one value
      per lane across 64 trips. Identical arithmetic, 4-8x the loop trips, each carrying its own
      dependent load.

      In d3 that only lengthens the block. In d4 it is worse than that: d4 reduces nine warps
      through one `__syncthreads()`, so the eight fast warps *wait* for the slow one, which is
      what the 10.05 barrier figure was. Giving W8 a `load_eight` and one generalised lane mapping
      (a group needs `kGroupK/8` lanes, so a warp covers `32/(kGroupK/8)` groups per pass -- 4 for
      Q5/Q6 at 64, 8 for W8 at 32, which compiles to the previous code for the Q5/Q6 case):

      | | before | after |
      |---|---:|---:|
      | d4 `barrier` stall | 10.05 | **2.13** |
      | d4 duration | 26.78 us | **21.38 us** |
      | d3 duration | 30.94 us | **24.90 us** |
      | d3 `long_scoreboard` | 7.54 | 8.76 |

      End to end on the 35B, paired median of 6, 6/6 positive: **+6.47% at C1** (+3.93..+6.74%)
      and **+13.43% with MTP3** (+13.21..+13.64%) -- MTP3 is larger because the small-T kernels
      (`d3_path_tiled`, `d4_token`) call the same two helpers, so they were fixed at the same
      time. **Do not quote the MTP3 number**: it runs on `bench/fixtures/bench_corpus.ids`. The
      27B is untouched at 46.58 against 46.83 tok/s, inside its +-0.21 spread, because it is dense
      and calls no `sparse_moe` kernel at all.

      Note what this was *not*. The three fixes closed above as measured negatives all attacked
      occupancy -- launch geometry, batching, block splitting -- and this attacked neither
      occupancy nor the 8x L1 over-fetch, which is still there and still not worth chasing
      (d3's `long_scoreboard` even rose slightly; the win is balance, not bytes). It is #88's
      consume-loop widening, applied to the one warp in the block that never got it.

      Registers are *not* on that list, which is the correction: an earlier draft of this entry
      claimed `Block Limit Registers = 5` was the constraint and that 32 registers per thread would
      give 7 blocks per SM. At 288 threads it would give 7 by the register rule, but the warp rule
      caps it at 5 regardless, so the change would measure as exactly nothing. Read both limits
      before believing either.

      **What is left after all that is item 4 alone — a real but 1.83x over-fetch on d4 — and it is
      not worth much**, because d4's DRAM pipe is only 47% utilised, so the wasted bytes are not
      what the kernel is waiting for. Items 1, 2 and 3 are all now closed as measured negatives:
      launch geometry cannot help, batching cannot help, and splitting the block makes it worse.
      Certainly not the 15-20% "close half the gap to the contiguous kernels" this entry
      used to promise. That framing compared an 8-of-256 gather against kernels that stream
      contiguous weights, and items 1 and 2 are why that comparison was never going to close.

      **The contiguous reference is collected, and it dismantles the comparison this entry was
      built on.** `q5_rowsplit_gemv` on the 27B decode, same elevated run, same clock lock, same
      instrument — which is the point, because every "40-45% versus 78-82%" figure in this file
      divides byte attributions by nsys durations rather than reading one tool:

      | kernel | dur | SM % | L1/TEX % | L2 % | DRAM % | saturated? |
      |---|---:|---:|---:|---:|---:|---|
      | `sparse_moe_d3_nine_warp` | 28.51 µs | 34.3 | 34.7 | 19.9 | **43.2** | **nothing** |
      | `sparse_moe_d4_nine_warp` | 24.13 µs | 37.7 | 42.4 | 24.1 | **47.1** | **nothing** |
      | `q5_rowsplit_gemv` (contiguous) | 46.18 µs | 81.0 | **86.6** | 22.7 | 50.1 | L1/TEX |

      **On one instrument the two are 43-47% against 50% of DRAM peak, not 45% against 80% of
      anything.** The contiguous kernel is not streaming weights at four-fifths of the card's
      bandwidth; it is at *half* of DRAM peak with its **L1/TEX path at 86.6%**. The difference
      between it and the MoE is not how much bandwidth each reaches — it is that the contiguous
      kernel has saturated a pipe and therefore has no headroom, while the MoE has saturated
      **nothing**: its highest utilisation of any pipe is 47%.

      That is the cleanest confirmation of item 2's "latency, not bandwidth" that this entry has,
      and it arrives with the corollary that **the gap to be closed is much smaller than the file
      has been claiming.** The percentages in #53's table are still useful as bytes-per-second, but
      "the expert kernels reach 40-45% of what the card can read while contiguous weight kernels
      reach 78-82%" should not be read as 35 points of available headroom. It is not.

      *One tooling note kept, because the failure looked like the tool's fault.* An earlier attempt
      at this reference passed `-k 'regex:a|b'` and PowerShell parsed the `|` as a pipe before `ncu`
      saw it, producing an empty report. Fixed to a single `--kernel-name` pattern.

- [ ] **Prefill's MLP GEMMs run at ~30% of the card's INT8 tensor-core rate. The occupancy
      explanation below is wrong, and the retile it asks for is measured and does not pay.**

      **Measured 2026-09-18** with `tools/w4a8_rowsplit_probe.cu`, which parameterises the warp
      tile, at the gate_up shape and T=512. Occupancy was raised for real, not argued about:

      | warp tile | block tile | registers | blocks/SM | warps of 48 | us |
      |---|---|---:|---:|---:|---:|
      | 2x4 (production shape) | 128x128 | 108 | 1 | 16 | 1,890 |
      | 1x4 | 64x128 | **55** | **2** | **32** | **1,785** |
      | 2x2 | 128x64 | 63 | 2 | 32 | 2,844 |

      Doubling resident warps is worth **6%**, not the 40% `ncu` estimated, and cutting the tile
      further to reach the same occupancy costs 50% because each thread re-reads more weight.
      `cudaOccupancyMaxActiveBlocksPerMultiprocessor` confirms the block counts, so this is not a
      launch that failed to get the occupancy it asked for. **Registers are not the lever.**

      The same probe rules out the other half of the theory: keeping the artifact's RowSplit
      weights and winning everything else the fragment-order probe won — `cp.async` on all four
      planes, packed nibbles in shared, the weight scales as an async ring so nothing waits on a
      synchronous global read — measures **1,726 us against production's 1,701**, i.e. nothing.
      `tools/w4a8_real_weight_probe.cu` puts the fully repacked layout at 1,566 us with per-group
      scales (the 1,415 us figure needs per-token scales, at 12.9% relative L2 on outlier-heavy
      inputs), so **the entire layout lever is ~8%** and it requires a repack `AGENTS.md` forbids
      and ~9.7 GB a 24 GB card does not have.

      **And the reason none of it moved is that the kernel was not compute-bound.** Ablating the
      probe (same shape, T=512) removes one cost at a time while keeping the MMA count identical:
      2,068 us complete, 1,918 without the scale reads, 1,856 without the per-group rescale, 1,423
      without the A-fragment shared reads, 1,424 without B's as well -- and **1,378 us with the
      MMAs removed but every load kept, against 1,378 us with the MMAs and every shared read
      removed.** Two thirds of the time is the global-to-shared streaming path, at ~28% of DRAM
      peak, so it is cp.async latency rather than bandwidth, occupancy, or the tensor cores.

      What generates that traffic is the token tile: a block covering BN tokens re-streams the whole
      weight matrix once per column block, eight times over at the production chunk of 1,024. At
      T=1024, 128x128 measures 4,265 us, 128x256 3,702, and **64x512 3,060 (+39%)**; a grid swizzle
      to let L2 serve the repeats is worth within 0.5% of nothing. That is shipped -- see
      `docs/performance.md` -- and it is what this entry should have been about.

      **And the size of what is left is now measured, not guessed.**
      `tools/int8_gemm_reference_probe.cu` runs cuBLAS's int8 GEMM over these shapes on this card:
      237.5 TOP/s at gate_up against our 95.5, 176.1 at mlp/down against 100.1, 176.7 at the mixer
      out_proj against 93.3 -- 1.8x to 2.5x, while reading twice the weight bytes our int4 codes do
      and with no unpack, no per-group scale and no fused epilogue to pay for. cuBLAS finishes
      gate_up in 1,537 us, less than the 1,711 us our schedule spends streaming alone, so the
      streaming is ours rather than the hardware's.

      **And the deficit is specific to the integer path, which the bf16 control settles.** The same
      probe runs cuBLAS's bf16 GEMM over the same shapes: 66.4 TFLOP/s at gate_up against this
      fork's A16 route at 59.4, so the 16-bit path is at **89% of a tuned kernel** while the integer
      path is at **40%** of one. This fork does not write slow GEMMs; it writes one slow *integer*
      GEMM, because that mainloop does two things extra -- unpacking 4-bit codes and applying a
      per-group scale -- and under MMAs four times faster than bf16 those stop hiding.

      **Every knob this structure has is now priced** (gate_up, T=1024, against 3,113 us for the
      shipped 64x512 schedule):

      | knob | worth |
      |---|---:|
      | token tile 128 -> 512 (shipped) | **+39%** |
      | occupancy, 16 -> 32 warps | +6% |
      | cp.async pipeline depth 2 -> 4 | +3.9% |
      | grid swizzle for L2 reuse | +0.5% |
      | L2 `evict_first` on the weight stream (Marlin's hint) | -0.8% |
      | interleaving the B loads with the MMAs | -1.9% |
      | operand layout in MMA-fragment order | ~17%, behind a repack |

      Nothing left is worth more than a few percent, so **stop tuning this kernel**.

      **What Marlin has instead, read from `IST-DASLab/marlin` and summarised so the next person
      does not have to:** four cp.async stages rather than two; register fragments double-buffered
      across k-steps (`frag_a[2]`, `frag_b_quant[2]`, indexed `k % 2`); `ldmatrix` (`ldsm4`) for the
      fragments rather than per-lane loads; the quantised operand held *packed* in registers and
      dequantised immediately before its MMA with LOP3 bit tricks (~6-7 instructions per 8 values);
      an XOR-swizzled shared layout rather than padding; 256 threads (8 warps) rather than 512; and
      a striped split-K with an L2 global reduce. The one that makes the rest possible is that
      **its weights are permuted into fragment order before the kernel ever runs** -- which is why
      its fragment loads are single instructions and ours are four.

      **The other direction is priced too, and it is worse.** `tools/w4_dequant_cublas_probe.cu`
      materialises the weights as int8 and calls cuBLAS. It cannot be done once and kept -- int8
      weights for this model are ~24 GB against a 24 GB card, so neither VRAM nor a different
      on-disk format rescues it -- so it measures a per-chunk materialisation. At T=1024: gate_up
      324 us to dequantise plus 1,540 in cuBLAS against our 3,825 (2.05x), mlp/down 1.51x, out_proj
      1.60x. Deduct an epilogue pass over cuBLAS's int32 output that the probe does not charge for
      (~360 us on gate_up) and it is ~1.7x, bought with a coarser weight quantisation (one scale per
      row rather than per 64), per-token activation scales, and ~320 MB of scratch. Strictly worse
      than the layout change, which keeps 4-bit weights.

      **RESOLVED, by leaving the kernel alone and changing who runs the GEMM.** The hand-written
      line has a ceiling: with streaming perfectly hidden its compute path still costs 2,092 us, so
      the best case is ~1.36x the shipped kernel -- parity with the stacks this entry is chasing,
      not a lead. Handing the GEMM to cuBLAS instead is worth 1.7x end to end. Prefill went
      1,634 -> 2,989 tok/s at pp4096 on one card in one session (+83%), for +0.156% perplexity,
      behind `--prefill-cublas`. See `docs/performance.md` and
      `src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.h`, which carries the chunk trade-off table
      and the two rejected experiments (overlapping the dequantise with the GEMM; inverting the
      prefill loops).

      The kernel work below stands as the record of why that was the right move, and the layout
      work it produced is committed but parked: the panel-major layout is worth 12.5%, but only at
      a 128x128/256-thread tile the engine does not use, and it trades the decode path for the
      prefill path.

      **Marlin's design was then built and measured, and it does not pay.**
      `tools/w4a8_marlin_probe.cu` implements all of it at once over a permuted weight layout: the
      weight permuted within each group of 64 so one 8-byte shared load is a lane's whole A fragment
      for a row, LOP3 dequant whose natural output order *is* the MMA's required order, packed
      nibbles in shared, a cp.async ring of 3-4 stages, and one barrier per stage instead of two per
      group. Best configuration (128x128, 256 threads, 2 blocks/SM): **2,901 us / 125.8 TOP/s
      against the shipped 117.3 - 1.07x**, where the gate was 1,900 us / 192 TOP/s. Abandoned.

      **The decomposition is the useful part, because it relocates the problem twice over.** The
      kernel has two different bottlenecks at two different shapes, and no shape escapes both.
      Per-thread accumulator count is `BM*BN/THREADS`, and 64 fp32 accumulators is what a
      128-register budget allows once fragments, addresses and the scale ring also live there. So a
      tile wide enough to cut the streamed bytes forces 512 threads, and 512 threads is what wrecks
      the MMA issue rate:

      | ablation (gate_up, T=1024) | 128x128, 256 thr | 128x256, 512 thr |
      |---|---:|---:|
      | streaming alone, no MMAs, no shared reads | 2,729 us | 1,376 us |
      | MMAs alone, no streaming, no reads, no rescale | 1,444 us | 1,704 us (1,855 with barriers) |
      | full kernel | **2,901 us** | 2,966 us |
      | | memory bound | issue bound, parts additive |

      **The arithmetic this campaign spent itself attacking is not the gap.** At the better shape
      the entire per-group rescale -- 64 int-to-float converts plus 64 FMAs per thread per group,
      four ALU operations for every MMA -- is worth **44 us of 2,901** (int32 accumulate, rescaled
      once), and per-token activation scales are worth 42 us. Both sit inside the noise of a kernel
      on its memory floor. **Group-128 weight scales and per-token activation scales -- the two
      quality trades Marlin makes and this fork declines -- would buy ~1.5% here.** That closes the
      open question the previous revision of this entry left, and closes it against the trade.

      **What the bytes say.** At 128x128, 1,712 MB of activation re-reads plus 856 MB of weights in
      2,729 us is 941 GB/s: exactly this card's DRAM peak, so no L2 reuse at all -- although the
      786 KB activation tile is shared by all 164 concurrent blocks and ought to be L2-resident. At
      128x256 with 512 threads the same sum runs at 1,555 GB/s, 1.66x DRAM peak, so there the reuse
      *is* happening. The difference is memory-level parallelism, not bytes: the 256-thread shape
      cannot keep enough cp.async in flight to reach L2's rate. And byte-minimal shapes do not
      rescue it either -- 256x128 at 512 threads streams the least of any register-legal tile
      (1,712 MB) and runs 3,220 us, because 70 KB of shared drops it to one block per SM.

      **So two questions remain, and both are counter reads, not probes:** whether the 128x256
      shape is leaving ~3x of L2 bandwidth unclaimed, and why its streaming and its MMAs add rather
      than overlap. `dram__bytes`, `lts__t_sectors` and the issue-stall reasons answer both in one
      `ncu` session, which needs elevation on this box (ERR_NVGPUCTRPERM). Until someone has that
      data, **prefill kernel work on this fork is closed**: the shipped state is +21-29% over where
      this entry started, the remaining gap is localised to the memory path rather than the
      arithmetic, and further guessing is mispriced.

      The projections that had no integer route at all were the larger win and are done: see
      `docs/performance.md`, +13-17% prefill at every length.

      The original entry follows, kept because its ruled-out explanations are still ruled out.

- [ ] **(superseded, see above) Prefill's MLP GEMMs run at ~30% of the card's INT8 tensor-core
      rate, because 124 registers per thread hold the SM to 16 of 48 warps.** Cause located
      2026-09-09 with counters; the remaining work is a retile. Every other explanation has been
      measured and ruled out.

      Measured (#53 plus `tools/tensor_core_rate_probe.cu`): `q4a8_swiglu` reaches 97.4 T/s and
      `q5a8_add` 89.4 T/s against a measured **314.8 TOPS** INT8 ceiling, while the BF16 GDN
      projections reach 51-54% of the measured **67.6 TFLOPS** BF16 ceiling. A well-tuned large
      GEMM on Ampere usually reaches 60-80% of a pure-MMA microbenchmark.

      **It is not a skinny-tile artifact.** The worry was that 1,024-token chunks against GEMM
      dimensions of 34,816 x 5,120 are too narrow to fill the MMA pipeline, so the percentage would
      improve with a bigger chunk. Swept on the 27B, int8 KV, an 8,192-token prompt, three
      repetitions each:

      | `--prefill-chunk` | prefill tok/s | vs 1,024 |
      |---:|---:|---:|
      | 256 | 1,139.7 | −6.9% |
      | 512 | 1,192.1 | −2.7% |
      | 1,024 (default) | 1,224.5 | — |
      | 2,048 | 1,234.9 | +0.8% |
      | 4,096 | 1,237.0 | +1.0% |

      Throughput **plateaus by 2,048**, and quadrupling the chunk from the default buys 1.0%.
      Nothing there closes a gap from 30% to 60-80%, so the shortfall is in the kernels and they
      are the target. The MLP pair is 68% of prefill FLOPs, which makes this the largest
      compute-side opportunity in the file.

      **Two more explanations ruled out 2026-09-09, both analytically, which narrows this to the MMA
      pipeline itself.** Working from the chunk-640 profile — `q4a8_swiglu` takes 1,212.20 ms over
      512 calls, so 2.368 ms per call at the 27B's `34,816 x 5,120` MLP shape with 640 columns:

      | | |
      |---|---:|
      | work | 228.2 GOP |
      | weights read | 94.7 MB (4-bit codes plus one FP16 scale per 64) |
      | activations | 47.8 MB |
      | achieved | **96.4 TOPS = 31%** of the 314.8 TOPS ceiling |
      | compute floor at the ceiling | 0.725 ms |
      | memory floor at 854.2 GB/s | 0.167 ms |

      **Not memory.** The kernel is compute-bound over its own memory floor by **4.3x** at this
      shape, so no amount of bandwidth work touches it. Worth stating because "30% of peak" invites
      the assumption that something is starving.

      **Not dequantization.** Unpacking every one of the 178M 4-bit codes costs 0.020 ms at 2 int
      ops per code, 0.040 ms at 4, and 0.080 ms at 8 — **1%, 2% and 3% of the 2.368 ms call**,
      against the 3090's 17.8 T int-op/s CUDA-core rate. Even a pessimistic unpack cannot account
      for a 69% shortfall. This was the natural next hypothesis after the tile-shape one and it is
      also wrong.

      **The counters are in, and the answer is occupancy — specifically registers.** Collected
      2026-09-09 via `admin-profile.ps1` section 3, `ncu` kernel replay, clocks locked at 1,500 MHz,
      27B / int8 / `-p 4096`. `q4a8_swiglu_kernel`, grid `(272, 8)` x 512 threads:

      | | |
      |---|---:|
      | Block Limit Registers | **1** |
      | Block Limit Shared Mem | **1** |
      | Block Limit Warps | 3 |
      | Theoretical active warps per SM | **16 of 48** |
      | Theoretical / achieved occupancy | 33.33% / **33.32%** |
      | Active warps per scheduler | **4.00** of a hardware 12 |
      | Eligible warps per scheduler | **0.78** |
      | No eligible warp | 60.52% |
      | Issue slots busy / SM busy | 39.47% |
      | Registers per thread | **124** |
      | Tensor pipeline utilisation | 36.5% |
      | DRAM throughput | **20.56%** |
      | L1/TEX throughput | 61.12% |

      `ncu` states the conclusion itself — *"This kernel's theoretical occupancy (33.3%) is limited
      by the number of required registers, and the required amount of shared memory"* — and puts an
      estimated **40%** on fixing it. Achieved occupancy equals theoretical to two decimals, so the
      launch is already perfectly efficient *given* the occupancy; there is nothing to win in the
      grid. And the tensor pipe at 36.5% matches the 31%-of-peak measured end to end, so the two
      measurements agree: **the MMA pipeline is not slow, it is starved of issue opportunities.**

      That settles the three candidates this entry could not separate. Issue rate is a symptom —
      each scheduler issues once every 2.5 cycles *because* 60.5% of cycles have no eligible warp.
      Shared-memory feeding binds too, but secondarily. Occupancy is the cause.

      **The arithmetic says registers are the hard constraint and shared memory is the soft one.**
      Resident warps per SM is `65,536 / (regs_per_thread x 32)`, which at 124 registers is
      **16 warps — 33.3% — whatever block shape is chosen.** No launch geometry escapes it: a
      1,024-thread block at 124 registers does not fit on an SM at all. The targets:

      | registers/thread | warps/SM | occupancy |
      |---:|---:|---:|
      | 124 (today) | 16 | 33.3% |
      | 85 | 24 | 50.0% |
      | 64 | 32 | 66.7% |

      Shared memory is `kBM x kSRow + kBN x kSRow + (kBM + kBN) x 2` = **20,992 B** per block, and
      `Block Limit Shared Mem = 1` implies the default ~32 KiB dynamic carveout. A 48 KiB carveout
      admits 2 blocks and 100 KiB admits 4, which is one `cudaFuncSetAttribute` call — **but it
      would measure as exactly nothing on its own**, because registers still cap the SM at one
      block. Both have to move, and the register cut is the hard half. This is the same trap the
      MoE entry fell into with `Block Limit Registers = 5`: read both limits before believing
      either.

      **What makes a fix affordable is that DRAM sits at 20.56%** — a smaller per-thread
      accumulator tile costs arithmetic intensity and re-reads more weight, and on a kernel using a
      fifth of the card's bandwidth there is 4x of headroom to spend on that.

      **But the cheap way of buying the occupancy is catastrophic, and that is now measured.**
      `__launch_bounds__(kThreads, 2)` asks nvcc for two blocks per SM, which forces 64 or fewer
      registers per thread; the compiler reaches that by spilling. Tried it 2026-09-10, with the
      shared-memory carveout raised to `cudaSharedmemCarveoutMaxShared` so shared memory would not
      re-impose the one-block limit:

      | | prefill tok/s (`pp8192`, three reps, clocks locked) |
      |---|---:|
      | baseline, 1 block/SM at 124 registers | **1,148.63 ± 0.18** |
      | forced 2 blocks/SM at <=64 registers | **350.48 ± 0.01** |

      **A 69.5% regression — spilling costs 3.3x what doubling resident warps buys.** Correctness
      was unaffected (`ninfer_linear_swiglu_q4a8_int_test` passes either way), so this is purely a
      performance answer. Reverted.

      So **`ncu`'s "Est. Speedup 40%" is not reachable by forcing the register budget down**, and
      that is the useful part: the 40% is only available from a retile that lowers the *natural*
      register requirement. The registers are accounted for — `acc[2][4][4]` is 32,
      `af[2][2][4]` is 16, `bf[4][2][2]` is 16, the two `Stage`s are ~14, addressing is the rest —
      so the accumulator shape is the only term big enough to matter.

      **And that retile is coupled, which is why it is not a constant.** Resident warps per SM is
      `65,536 / (regs x 32)`, so 24 warps needs <=85 registers and 32 needs <=64. Reaching either
      means changing the warp tile, which fixes `kThreads`; `kThreads` and `kBN` together fix the
      warp grid (`4 warp_m x 4 warp_n` today); and `kBN` is what this kernel *claims*, through
      `q4a8_tokens_supported` returning `tokens >= kBN && tokens % kBN == 0`. Halving `kBN` to 64
      would widen the claimed domain to any multiple of 64 and change routing for widths this
      kernel does not serve today. Halving the accumulator alone lands near 100 registers, which is
      still one block at 512 threads and therefore still 16 warps — the block granularity eats it.

      **And 768 threads does not work either, which took a second pass to see.** Accumulators per
      *block* are fixed by the 128 x 128 tile while the 92 registers of non-accumulator overhead
      scale with thread count, so 768 threads needs 86,784 registers per block and 1,024 needs
      110,592 — both over the 65,536 file. More warps per block is unavailable at any width, and
      split-K makes it worse rather than better for exactly this reason (a K-split warp still owns
      its whole output tile). See the split-K section near the top, which says so and corrects an
      earlier claim of the opposite.

      **So the lever is the 92 overhead registers, not the accumulators and not more warps.**
      Reaching 2 blocks per SM at 512 threads needs <=64 total, i.e. overhead 92 -> 32: `af` is 16,
      `bf` is 16, the two `Stage`s ~14, addressing the rest. Cutting it means fewer staged fragments
      and shallower prefetch, traded directly against the latency hiding they provide — a tuning
      problem with a measurable knob rather than a rewrite. `q5a8_add`, whose shape and 28.4% figure
      are the same story, would follow it.

      One caveat kept: `Block Limit Warps = 3` and `Block Limit SM = 16` are both far from binding,
      so nothing here is about block count.

      Two small things fall out. The default chunk of 1,024 is 1.0% off the plateau, so 2,048 is
      free throughput *if* the extra workspace is affordable — worth checking against the memory
      model in `docs/config-calculator.html` before changing a default. And 256 costs 6.9%, which
      is worth knowing for anyone tempted to shrink the chunk to save memory.

- [ ] **DFlash2's cliff between five and six draft tokens is a GDN input-projection route
      boundary, and the fix is a narrower MMA tile.** Found 2026-09-09. Not the block geometry this
      entry guessed at.

      The cliff reproduces exactly — 27B DFlash2 artifact, greedy, `--max-new 256` on a prose
      prompt, four repetitions each within ±0.1 tok/s. Converting to round cost is what makes it
      readable, since acceptance and rate move together:

      | k | tok/s | tok/round | ms/round | step |
      |---|---|---|---|---|
      | 3 | 59.1 | 2.55 | 43.1 | — |
      | 4 | 59.5 | 2.95 | 49.6 | +6.4 |
      | 5 | 59.2 | 3.07 | 51.9 | +2.3 |
      | 6 | 50.0 | 2.97 | 59.4 | **+7.5** |
      | 7 | 48.2 | 2.93 | 60.8 | +1.4 |
      | 8 | 47.1 | 3.45 | 73.2 | +12.5 |

      **Accepted tokens per round is flat at ~3.0 from k=4 onward** (2.95, 3.07, 2.97, 2.93, 3.45),
      so nothing past four draft tokens pays for itself — every column beyond it is cost. That is
      the real justification for the recommendation of four, which `docs/cli.md` already gives.

      The 5→6 step itself is a schedule switch, confirmed by nsys per-round diff (83 rounds at k=5,
      86 at k=6, both deterministic): kernel time per round goes 58.22 → 66.21 ms, +7.99, matching
      the +7.5 from the throughput arithmetic. `q4_rowsplit_gemm_simt` (5.17 ms) and
      `q5_rowsplit_gemm_simt_split4` (6.43) vanish and `rowsplit_grouped_mma_kernel` (15.82)
      appears, on ~51 instances per round — the 27B's GDN layer count, not its 17 attention layers.
      That is `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp`:

      ```
      {{1, 6},  IndependentDirectFixed},
      {{7, 32}, GroupedMixedMmaR64C32},
      ```

      Verification width is k+1, so k=5 is the last count inside the direct route and k=6 is the
      first to cross into a **32-wide** MMA tile whose cost, as that file's own comment says, is set
      by padded width rather than live tokens. Width 7 pays for 32 columns.

      **The obvious fix does not work, and that is the useful part.** Both `launch_q4` and
      `launch_q5` accept T up to 15 with a dedicated R8C8 route for 5..15, so extending the direct
      band looks free. Measured `{{1, 15}}` / `{{16, 32}}` end to end and normalised against k=4/5
      (unchanged in both tables, and they calibrate a ~3% between-process drift): **−3.6% at width
      7, +1.3% at width 8, −23.8% at width 9, −13.2% at width 11.** The `{1,6}` boundary is right.
      Width 8 is the one place the direct path is competitive, which is the R8C8 tile fitting
      exactly; 9 needs two passes and collapses. Recorded in that file so it is not retried.

      **What to build instead: a `GroupedMixedMmaR64C8` or `R64C16`.** The grouped tile wins at
      width 7 *despite* padding 7 columns into 32 — so the MMA path beats the SIMT direct path by
      more than 4.6x of wasted width, and the prize is keeping that efficiency without the dead
      columns. Two independent measurements want the same kernel: this cliff, and §2c's 8-lane
      entry, where a C8 decode cohort spends **13.8 ms of a 56.3 ms round in this exact kernel at
      width 8** (0.300 ms per instance at width 8 against 0.311 at width 7 — near-identical, which
      is what padded-width-dominated cost looks like). Fixing it pays twice.

      One tooling gap was behind all of this: this Op had no schedule bench, which is why its
      boundary carried no measurement for so long. Written now, and it produced the two narrow
      tiles in the entry below — which take back a third of this cliff (k=6 goes from 47.9 to 51.2
      tok/s, +5.3% paired, 6/6 runs). The cliff is smaller but still there: the remaining gap is
      that the grouped kernel runs at 27% of its own weight-streaming floor at any width, and no
      tile choice changes that.

- [x] **The GDN input projection now has a schedule bench, and it bought two narrower tiles.**
      Closed 2026-09-09. This was the one route table here with no schedule bench, which is why its
      boundary carried no measurement while being implicated in two separate slowdowns.

      `bench/ops/q4_q5_gdn_input_schedule_bench.cu` plus a
      `q4_q5_gdn_input_execute_schedule` seam (the pattern `q4_linear_swiglu` and `w8_pair` already
      use: the real dispatch minus its plan-matches-problem check, so every schedule can be timed
      at every width). First result: **every existing boundary in that table is correct** — 6/7,
      32/33 and 64/65 all sit exactly where the curves cross.

      Second result: the table was missing two tiles. `GemmCfg` takes the tile width as `BN`, so
      `R64C8` and `R64C16` are instantiations, not new kernels. Measured, cold, medians of 31 with
      `--spread`, each winner's p95 below the runner-up's min:

      | T | independent | c8 | c16 | c32 |
      |---|---|---|---|---|
      | 6 | **188.4** | 240.6 | 247.8 | 290.8 |
      | 7 | 330.8 | **238.6** | 242.7 | 267.3 |
      | 8 | 319.5 | **236.5** | 243.7 | 267.3 |
      | 9 | 486.4 | 504.8 | **243.7** | 267.3 |
      | 16 | — | 506.9 | **242.7** | 267.3 |
      | 17 | — | 750.6 | 495.6 | **269.3** |
      | 32 | — | 999.4 | 491.5 | **264.2** |

      So c8 wins 7..8 by 10.7-11.5% and c16 wins 9..16 by 8.8-9.2%, each collapsing one column past
      its own width because a second pass costs a whole extra weight read. The table is now
      `{1,6} independent / {7,8} c8 / {9,16} c16 / {17,32} c32 / {33,64} c64 / {65,∞} c128`, and
      `tests/ops/test_gdn_input_proj.cpp` gained widths 8, 9 and 17 so every new boundary is
      exercised on both sides.

      End to end on DFlash2, interleaving the two binaries within each repetition so thermal drift
      lands on both arms — which it had to, because the card moved 3-5% between processes and that
      is larger than the effect:

      | k | width | base | c8/c16 | paired median | positive |
      |---|---|---|---|---|---|
      | 6 | 7 | 47.9 | 51.2 | **+5.3%** | 6/6 pairs |
      | 8 | 9 | 44.4 | 45.2 | **+3.2%** | 4/6 pairs |

      **Do not read the 11% as evidence that padding was the main cost.** Dropping `BN` from 32 to
      8 removes 75% of the padded MMA work and buys 11%, because this Op streams ~55 MB of weights
      (4,096x5,120 q4 plus 12,288x5,120 q5) whose 64.4 µs at 854.2 GB/s no tile choice changes. c8
      at 236.5 µs is **27% of that floor**. The padded work was a minor term all along, and the
      remaining 3.7x is what is actually worth chasing — the same conclusion §2c's 8-lane entry
      reaches about `mma_r64_c16`, from a different Op.

      **The C8 serving cohort — the other workload this kernel dominates, and the recommended
      multi-user profile — gains 5.7%.** Measured after the fact, same interleaving, 27B/mtp0/int8
      at `--concurrency 8 --decode-tokens 512`:

      | rep | base | c8 tile | |
      |---|---|---|---|
      | 1 | 133.8 | 140.8 | +5.2% |
      | 2 | 131.6 | 139.1 | +5.7% |
      | 3 | 131.0 | 142.8 | +9.0% |

      Positive 3 of 3 and **the two arms do not overlap at all** — base tops out at 133.8, tiles
      bottom out at 139.1. That is better than the "low single digits" this entry first predicted
      from `~25% of the round x ~11% of the kernel`, so something else improved alongside it; the
      prediction was a lower bound rather than an estimate.

      §2c's eight-lane curve moves with it: C1/2/4 are untouched (widths 1, 2 and 4 all stay on the
      independent route) so the curve is now **36.8 / 61.1 / 101.0 / 140.8 tok/s, or 3.83x at eight
      lanes** against the 3.62x measured before these tiles existed.

- [x] **`w8_pair` medium no longer discards its schedule on sm_86, and the two tiles that are
      actually routed here are 13-17% faster than the chunked loop they replaced. Closed
      2026-09-10.**
      `w8_pair_gemm_splitk.cu:133` is `(void)schedule` under `NINFER_SM8X_COMPAT`, so every
      schedule runs the same chunked loop, slicing T into `kLastExactT = 32` columns and calling the
      exact-T launcher repeatedly. PR #22 already removed the twelve now-identical route entries
      this made redundant, confirmed by identical 24.576 µs timings, so the *table* is honest.

      **The reason for the blanket discard, derived 2026-09-09.** `w8_rowsplit_medium_t_splitk_kernel`
      declares two static shared arrays — `code_shared[16][KSplits x 64]` and
      `b_shared[KSplits x NGroups][(TileCols / NGroups) x 64]` at 2 B — so each schedule's footprint
      follows from its own template arguments, against sm_86's **49,152-byte static shared cap**:

      | schedule | KSplits | NGroups | warps | shared | fits sm_86 |
      |---|---:|---:|---:|---:|---|
      | C48 `<48,4,2,3>` | 4 | 2 | 8 | 28.0 KiB | **yes** |
      | C64 `<64,4,2,2>` | 4 | 2 | 8 | 36.0 KiB | **yes** |
      | C80 `<80,4,2,1>` | 4 | 2 | 8 | 44.0 KiB | **yes** |
      | C88 `<88,4,1,1>` | 4 | 1 | 4 | **48.0 KiB** | no — exactly *at* the cap |
      | C96 `<96,4,1,1>` | 4 | 1 | 4 | 52.0 KiB | no |
      | C104 `<104,4,1,1>` | 4 | 1 | 4 | 56.0 KiB | no |
      | C112 `<112,4,1,1>` | 4 | 1 | 4 | 60.0 KiB | no |
      | C128 `<128,2,4,2>` | 2 | 4 | 8 | 34.0 KiB | **yes** |
      | C160 `<160,2,5,2>` | 2 | 5 | 10 | 42.0 KiB | **yes** |
      | C192 `<192,2,6,2>` | 2 | 6 | 12 | 50.0 KiB | no |

      So the five wide tiles are **compile errors on sm_86, not slow paths**, and instantiating the
      whole `switch` is what forced the discard. That is a much better-understood position than
      "nobody has written one": the sm_86-specific tiling is not a new kernel, it is **the five
      existing instantiations that already fit**, behind an `#if` that lets the other five fall
      through to the chunked loop.

      **They win, and the bench gives a clean in-run baseline for free.** Only C48 and C64 are ever
      routed on this card — `w8_pair_plan.cpp` sends `{33,48}` to C48 and `{49,64}` to C64, and from
      T=66 the concat kernels win outright — so those are the two instantiated. C80, C128 and C160
      also fit but are never asked for, and the remaining four exceed the cap. Because `medium_c128`
      and `medium_c192` are *not* instantiated, they still fall through to the chunked loop, which
      means the same bench run measures the tiles and their replacement side by side under identical
      conditions:

      | T | real tile | chunked loop (`medium_c128`) | gain | routed to |
      |---|---:|---:|---:|---|
      | 33 | **26.624** (c48) | 30.720 | **13.3%** | c48 |
      | 48 | **30.720** (c48) | 36.864 | **16.7%** | c48 |
      | 64 | **34.816** (c64) | 40.960 | **17.1%** | c64 |

      `bench/ops/w8_pair_schedule_bench.exe --k2048`, cold, medians of 9, clocks locked. The
      `public_op` column tracks the winner at every width, so the route table was already pointing
      at the right schedule — it just had no schedule to reach.

      The chunked fallback's cost is what the gain measures: it slices T into 32-column chunks and
      pays a whole extra weight pass per slice, so at T=48 it streams the weights twice where C48
      streams them once. §2c's parallelism caution does not bite here — `C48 <48,4,2,3>` is 8 warps
      per block, the same as the exact-T kernel the chunked loop calls — so this is one fewer weight
      pass rather than a geometry trade.

      **What this is not: a model-level speedup, and that is stated rather than glossed.** This is
      the k=2048 W8 pair, which is the 35B's DFlash draft-head path, and the band is T=33..64 — a
      wide speculative extent rather than anything a decode step or an ordinary prefill reaches. The
      13-17% is an Op-level number above the ~10% cold-flush threshold established in §3, so it
      should survive in situ, but **no end-to-end measurement was taken because no shipped profile
      obviously drives this band.** Anyone who finds one has a ready 13-17%.

---

### 2c-vii. Five screens nobody has run, and every kernel win this cycle came from one of them

Written 2026-09-10, at the end of a cycle that shipped six speedups. Every open item above names
**one kernel**. But all six wins came from applying the *same* three signatures to a kernel that
nobody had checked, and none of those signatures has ever been run across the kernel set as a
sweep. So the list is a list of kernels somebody happened to look at, and these five entries are
the screens that would tell you which kernels to look at next. All five are cheap; two need no GPU
at all, and one of those two should be done before any of the others.

- [ ] **Nobody has attributed the missing 26% of decode bandwidth to any kernel.** The dense decode
      is at **73.9% of the 854.2 GB/s this card can actually read**, and the analytic read-set is
      **15.743 GB/token**. Both numbers are for the whole step. There is no per-kernel table, so
      every remaining decision about what to optimise next is a guess about where the 26% lives —
      including the guess in the entry above that block count takes up the slack in the q5 GEMV.

      Build the table: `ncu --replay-mode application --metrics
      dram__bytes_read.sum,gpu__time_duration.sum` over one clock-locked `-pg 4096,128` decode step,
      then per kernel report measured bytes, duration, and implied GB/s against 854.2. Rank by
      **bytes x (1 - achieved/854.2)** — the absolute time each kernel is leaving on the floor —
      because ranking by percentage promotes tiny kernels and this is a bandwidth budget, not a
      league table. `scripts/sweeps/admin-profile.ps1` already collects `dram__bytes_read.sum` for
      the MoE section; this generalises that section to the whole step.

      Use `--replay-mode application`. Kernel replay mirrors the 21 GB device working set into host
      RAM and dies with `bad allocation` unless nothing else on the box is holding memory.

- [ ] **Bytes-read amplification has been measured for exactly one kernel, and it was 4.9x.** The
      metric is measured DRAM (or L1 sector) bytes divided by the kernel's *analytic* read-set. A
      decode kernel that streams weights once should sit at 1.0. `d4` came out at
      `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` = 2,560,000 sectors = **81.9 MB against
      8.3 MB actually read from L2** — it re-reads its working set roughly five times through L1.
      That was found by hand, on one kernel, chasing something else.

      Nobody has run it as a screen. Any kernel above 1.0 is doing avoidable work, and on a machine
      that is 26% short of its own read bandwidth an amplification factor is the most direct
      possible pointer at where the shortfall is. Same ncu pass as the entry above plus the two
      sector counters, so it is one profiling run for both. **Screen every decode kernel, rank by
      amplification x bytes, and treat anything above ~1.2 as a defect rather than a tuning
      opportunity** — re-reading a weight tile is not a trade-off, it is waste.

      **Collect `lts__t_sector_hit_rate` in the same pass.** Amplification says a kernel re-reads;
      L2 hit rate says whether the re-read is costing DRAM traffic or is absorbed. It also
      independently informs the routed prefill pipeline-depth item: `7/4` is upstream's RTX 5090
      constant and **an RTX 5090 has 16x this card's L2**, so a measured hit rate is the direct
      evidence for what that constant should be here, rather than sweeping it blind.

- [ ] **The signature that produced three of this cycle's six speedups has never been run as a
      screen.** It is: **L1/TEX throughput high while DRAM throughput is well below it.** That
      combination means the kernel is limited by how fast its *consume loop* retires bytes, not by
      how fast memory delivers them — the kernel is issuing far more load instructions than the
      bytes require, usually because each lane consumes one or two weights per trip.

      It was found three times and fixed three times with the same two code changes, both now
      shipped and available to copy:

      | kernel | before | after | fix |
      |---|---|---|---|
      | `q5_rowsplit_gemv` | L1/TEX 87%, DRAM 50% | L1/TEX 37% | widen the consume loop to 8 weights/lane |
      | `q4_linear_swiglu_gemv` | same shape | — | same widening, gate+up as 4-byte words |
      | `q5_rowsplit_gemv` (dequant) | Compute 73% | 68.5%, DRAM 66.7% | half2 bit-trick + per-group scale hoist |

      Together those moved dense decode **37.44 → 40.11 tok/s**. The remaining quantized GEMV and
      GEMV-shaped kernels have never been checked for the same signature, and the fix is a known,
      reviewed, already-merged pattern rather than a design problem. **This is the highest
      expected-value mechanical item on the list**: one ncu pass over the decode kernel set
      collecting `l1tex__throughput.avg.pct_of_peak_sustained_active` and
      `dram__throughput.avg.pct_of_peak_sustained_elapsed`, then rank by the *gap* between them.

      The half2 bit-trick is in `src/ops/linear/q5/q5_rowsplit_gemv.cuh`: half `0x6400` is 1024.0
      and its mantissa LSB is 1.0, so OR-ing a nibble into the mantissa and subtracting the bias
      `0x6410` (1040.0) yields `nibble - 16*high_bit`, which is `sign_extend<5>` for free. Any
      kernel dequantizing 4- or 5-bit codes can use it.

- [ ] **The parallelism arithmetic has been done for two kernels and it decided both of them, and
      it has never been tabulated for the rest. It costs no GPU time at all.** The quantity is total
      warp-work divided by what the card can hold resident — "machine-fulls". For a tiled GEMM it is
      `(rows / WM) x (BN / WN)`; **note `BM` does not appear**, which is why narrowing `BN` *reduces*
      parallelism and why the tile-narrowing experiment lost 6-32%. The denominator is 82 SMs times
      `65,536 / (regs x 32)` warps per SM.

      Two kernels have been through it and in both cases it was the answer:

      | kernel | warp-work | machine-fulls | what it explained |
      |---|---:|---:|---|
      | `rowsplit_grouped_mma_kernel` | 1,024 warps | **0.26** | 24.5% achieved occupancy; why split-K is the fix |
      | `sparse_moe_d3_path_tiled` | — | **1.17** | why launch geometry and batching both measured flat |

      Nothing else has a number. That matters right now, because the entry above ends by *guessing*
      that block count is what takes up the slack when `kStages = 3` removes a 27% memory-latency
      stall and buys nothing — and this table would settle that guess with pen and paper. Do it for
      every kernel in the decode step and for the prefill GEMMs, rank by machine-fulls, and treat
      anything under ~1.0 as latency-bound-by-construction: **no amount of stall-metric chasing will
      help a kernel that cannot fill the machine**, which is the trap that consumed three built-and-
      discarded experiments this cycle.

      It also predicts which of the four screens above will pay off on which kernel, so it is worth
      doing *first* even though it produces no measurement: a kernel at 0.3 machine-fulls does not
      need its consume loop widened, it needs more blocks.

- [ ] **Nobody knows the register decomposition of any kernel except one, and occupancy work has to
      be bought from the part nobody has measured.** `-Xptxas -v` appears nowhere in this file. The
      prefill MLP entry established the principle the hard way: `q4a8_swiglu`'s 124 registers are
      **32 accumulators and 92 of overhead**, the overhead term is what actually caps occupancy at
      one block per SM, and the fix therefore has to come out of staged fragments and prefetch depth
      rather than out of the accumulator array. That decomposition exists for that one kernel
      because I worked it out by hand after getting it wrong.

      It is available for every kernel at once, for the cost of one build: add `-Xptxas -v` and read
      registers, spill stores and spill loads per kernel out of the log. Then compute, per kernel,
      `65,536 / (regs x 32)` warps per SM and compare against what the kernel needs. **Zero GPU
      time.** Two things this would surface that nothing on the list currently can:

      1. **Spills already happening.** Forcing occupancy with `__launch_bounds__(kThreads, 2)` on
         the prefill MLP cost **69.5%** (1,148.63 → 350.48 tok/s) purely because the compiler spilled
         to reach 64 registers. Any kernel already spilling is paying that silently, and the log
         says so for free.
      2. **Which occupancy shortfalls are register-bound at all.** Both quantized KV decode
         attention kernels reach only **33% of their own theoretical occupancy** (21.98 of 66.66 for
         int8, 16.51 of 50.00 for fp8 — the same ratio to two figures). If their register counts are
         low, the shortfall is not register pressure and the arena/grid work named in that entry is
         the right target; if they are high, that entry is aimed at the wrong thing. Nobody can tell
         which today.

---

## 3. Measurement debt

- [x] **The interleaved A/B harness is committed. Closed 2026-09-09** as
      `tools/bench/run_interleaved_ab.py`, documented in `tools/bench/README.md`. The card drifts
      3-5% between processes, so measuring all of arm A then all of
      arm B is not a comparison — two runs this cycle came out with opposite-signed drift (+2.9%
      then −3.8%) on code paths that had not changed. The working pattern, hand-rolled three
      separate times this cycle:

      1. build both binaries and place them **inside `build-ninja/apps/`** under different names,
         because an executable copied elsewhere fails DLL resolution with exit 127 and an empty
         log, which reads exactly like a model failure;
      2. alternate the two inside one repetition loop, not one arm then the other;
      3. report the **paired** median — the median of per-repetition ratios — and how many pairs
         were positive, not the ratio of the two medians;
      4. keep a control configuration whose code path is identical in both arms, and normalise
         against it. In the GDN tile A/B, draft counts 4 and 5 stay on an unchanged route and are
         what made the result readable.

      All four properties are implemented, and the two that are easy to get wrong are *refused*
      rather than left to the caller. Both arms in different directories is a hard error naming the
      exit-127 DLL trap; a `--config` with no `{exe}` placeholder is a hard error, because it would
      run the same binary twice and report a comparison of nothing. A run where any sample produced
      no number exits 1 — a comparison with a hole in it is not a comparison.

      Two things it adds beyond the hand-rolled pattern. It **swaps which arm leads on alternate
      repetitions**, so the warmer second slot in each repetition lands on each arm equally often
      instead of accumulating on one side. And `--control` is divided out **per repetition before
      the median is taken**, not median-over-median, which is the same distinction the paired
      statistic itself rests on.

      Verified against stub binaries with a known answer — a planted +5.85% on the measured arm and
      a −0.17% control — which it recovers as `+5.85%` raw, `+6.02%` normalised, 4/4 pairs positive.
      The guard paths were tested too: different directories, a missing placeholder, and a metric
      regex that never matches all fail the way they are documented to.

      `nvidia-smi -lgc` reduces the need for this but does not remove it, and needs an elevated
      shell.

- [x] **The empty `content_sha256` column is fixed, and the diagnosis in this entry was wrong.**
      Closed 2026-09-09. This entry said
      `scripts/sweeps/dflash2-draft-tokens-realtext.ps1` "calls `Get-FileHash`, which does not exist
      under this box's `powershell`". **It does exist and it hashes correctly** — Windows PowerShell
      5.1.26100.9168, verified directly. So neither of the replacements this entry recommended
      (`certutil -hashfile`, `SHA256::Create()`) was needed, and either would have "fixed" the
      symptom without touching the cause.

      **What actually happens.** `Get-FileHash` on a path that does not exist raises a
      **non-terminating** error — from a `Resolve-Path` inside the cmdlet's own implementation — and
      returns *nothing at all*. `.Hash` on the nothing yields an empty string, and the error prints
      once per iteration. That is exactly the reported symptom, error-block count included, and the
      real defect is whatever left the stdout file missing rather than the hashing call.

      Worth keeping because it defeats the obvious defensive fix too: a bare `try/catch` around it
      does **not** help, since a non-terminating error never reaches `catch`. Confirmed by
      measurement, not reasoning — the first patch written for this was exactly that useless
      `try/catch`.

      The column now does `Test-Path` first, then `Get-FileHash -ErrorAction Stop` inside a
      `try/catch`, so a failure lands *in the CSV cell* (`ERR:no-stdout-file`,
      `ERR:<ExceptionType>`) instead of as console noise, and the sweep still finishes. The next
      run therefore reports why rather than needing this diagnosed again.

- [x] **Host memory pressure can silently invalidate any 35B measurement on this box, and did.**
      Guarded everywhere as of 2026-09-09; the mechanism below is unchanged and still worth reading.
      `vmmemWSL` holds up to **27 GiB of the 64 GiB** of host RAM while WSL is running, and the 35B
      artifact is 22.8 GB — so loading it contends directly with WSL's footprint, the machine pages,
      and timings become noise. A first attempt at the pipeline-depth sweep was abandoned for
      exactly this. `wsl --shutdown` reclaims it.

      **Every sweep now carries the guard**, factored into `scripts/sweeps/host-memory.ps1`
      alongside `model-dir.ps1` and dot-sourced by all twelve scripts that load a model
      (`decode-roofline.ps1` is exempt: it reads CSVs and never touches the GPU).

      The requirement is **sized from the artifact** rather than fixed, because a 27B-only sweep
      does not need what a 35B one does and refusing it at 20 GiB free would be a false alarm. The
      headroom term is the one that was actually validated: the original hand-written guard demanded
      24 GiB for the 21.22 GiB 35B artifact, i.e. 2.8 GiB above the file, so keeping that constant
      reproduces the old threshold *exactly* on the 35B and scales it everywhere else. Measured:
      19.8 GiB for the 27B, 21.8 for the 27B DFlash2, 24.0 for anything touching the 35B.

      **`admin-profile.ps1` asks for 28.0 GiB, and that is not padding.** `ncu`'s default kernel
      replay saves and restores the device memory a kernel could touch; with 21 GB of weights
      resident there is no room on the card, so it spills the backup to *host* RAM — ~42 GB
      alongside the artifact's own footprint, which does not fit beside `vmmemWSL`. It fails as
      `==WARNING== Backing up device memory in system memory` followed by
      `Unhandled C++ exception: bad allocation` and an **empty report**, which reads like an `ncu`
      bug and is not one. Hit on the first run of this cycle. `--replay-mode application` re-runs
      the application once per pass and needs no backup at all; it is now that script's default,
      with `-ReplayMode kernel` available for a model small enough that the backup fits on the card.

      `NINFER_SKIP_MEMORY_GUARD=1` overrides the check, for someone whose box is not this one. Related: this box's C:
      drive sits at 98% full (45 GB free) and hit *zero* bytes earlier in the cycle, which failed a
      build with `C1085 ... No space left on device`. Two artifacts were byte-identical duplicates
      (19 GB recovered) and WSL crash dumps held another 12 GB.

      **Reading today's numbers in light of this:** the paired interleaved A/Bs are robust to it,
      because both arms met the same conditions — the GDN tile results (+5.7% C8 serving, 3 of 3
      non-overlapping; +5.3% DFlash2 k=6, 6 of 6 pairs) and the route-boundary bench sweeps are
      safe. Single-shot end-to-end figures taken during the same window are the ones to treat as
      provisional and re-run under the guard.

- [x] **Every route boundary in the repository was decided on cold-flush margins, which overstate
      the win — and there is now a number for how much. Closed 2026-09-09.**

      Two boundaries of the same Op were taken from the cold bench to a paired, clock-locked,
      end-to-end A/B with a drift control (§2c's split4 entry has the full tables):

      | boundary | cold margin | in situ | positive |
      |---|---:|---:|---:|
      | GDN input width 7, split4 over c8 | **+5.9%** | **−0.78%** | 0 / 6 |
      | GDN input width 8, split4 over c8 | **+13.2%** | **+2.85%** | 6 / 6 |

      So on this Op the cold bench overstated by 4-5x, and **at width 7 it overstated enough to
      invert the sign** — on a margin that would previously have been called comfortable.

      **Then a third boundary was checked and it went the other way, which kills the tidy rule.**
      `q5_linear_add`'s `{2,10}`/`{11,16}` crossover was chosen on an 8% cold margin at T=10
      (split2 93.2 µs against c16 101.4). Re-measured in situ 2026-09-10 — DFlash2 at
      `--draft-tokens 9`, whose k+1 verification width is exactly T=10, against draft count 4 as a
      control, six paired repetitions, clocks locked. Moving the boundary so c16 takes T=10 is
      **−13.59%, 0 of 6 pairs positive, control +0.00%.** The shipped boundary is right, and the
      in-situ margin is **1.7x larger** than the cold one rather than smaller:

      | boundary | cold | in situ | outcome |
      |---|---:|---:|---|
      | GDN input width 7, split4 over c8 | +5.9% | **−0.78%** | sign inverted |
      | GDN input width 8, split4 over c8 | +13.2% | +2.85% | survived, 4.6x smaller |
      | `q5_linear_add` T=10, split2 over c16 | +8.0% | **+13.59%** | survived, **1.7x larger** |

      **So "cold overstates by 4-5x" was an overreach from one Op, and the honest rule is the one
      this entry started with: a narrow cold-flush margin is a reason to check, not a predictor of
      anything — including its direction.** The flush penalises whichever schedule makes more weight
      passes, so which way it biases depends on the pair being compared, not on the margin's size. A
      threshold on the margin alone cannot work; what the margin buys you is a priority order for
      which boundaries to re-measure.

      The method is now cheap enough that there is no excuse for skipping it —
      `tools/bench/run_interleaved_ab.py` plus `nvidia-smi -lgc 1500` produced control drift of
      **0.00-0.09%**, which is two orders of magnitude better than the 3-5% that forced all the
      hand-rolled interleaving in this file. Use it on any boundary whose cold margin is in single
      digits, and do not re-derive the "cold is the honest default" argument: cold is right for a
      first pass, it is just not a decision.

- [x] **One of the two narrow boundaries is checked and correct; the other is still owed.**
      The `{2,10}`/`{11,16}` q5 crossover at T=10 was re-measured in situ and the shipped choice
      wins by **13.59%** — see the table above. So it was never "presumed unsafe"; it is simply
      right, and by a wider margin in production than the bench suggested.

      Still owed: the q4 SwiGLU `{2,24}`/`{25,40}` crossover at T=25 (464.9 vs 436.2 cold, 6%). The
      method is now routine — find a workload that lands on the width, move the boundary one step,
      A/B the two builds with a control that stays inside both arms. The obstacle is the workload:
      TODO's q4-SwiGLU entry records that **every full prefill chunk goes to the integer-activation
      kernel and never reaches this table**, so T=25 is hit only by decode widths and a prompt's
      ragged tail chunk, and finding a repeatable driver for exactly 25 columns is the work.

- [x] **The mechanism, kept because it is the reasoning behind the threshold above.**
      `bench/ops/schedule_sweep.cuh` and its siblings call `measure_cold_launch`, which
      flushes 256 MiB through L2 before every repetition. That is the right default — these
      projections stream tens of MB against 6 MB of L2 and production is usually cold — but it is
      not the same as in situ, and the gap is not small. Measured on the q4 SwiGLU `{513,640}`
      decision (§5): the bench put c128 7.5% ahead of Materialized at T=576, while an nsys profile
      of the same schedules inside a real 600-token chunk put it 2.4% ahead, with **both** schedules
      slower per call in situ than in the bench (c128 4193 vs 3625 µs, Materialized 4295 vs 3927).
      The flush penalises Materialized's second weight pass more than a real prefill does.

      The sign was right in that case, and no boundary is known to be wrong. But every route
      comment in `src/ops/*/`*`_plan.cpp` quotes cold-flush microseconds, several of them at
      margins under 5%, and those margins are not speedups. Worth doing: take the boundaries whose
      measured margin is under ~5% — the `{2,10}`/`{11,16}` q5 crossover at T=10 (93.2 vs 101.4,
      8%) and the q4 SwiGLU `{2,24}`/`{25,40}` crossover at T=25 (464.9 vs 436.2, 6%) are the
      obvious two — and check each in situ with a profile rather than the bench. The method is in
      §5's entry; it is one nsys run per boundary.

      Do not "fix" the flush. Cold is the honest default for a first pass and warm numbers pick
      different winners, which is documented at the top of `schedule_sweep.cuh`. The point is that
      a narrow cold-flush margin is a reason to profile, not a decision.

- [x] **The 315 W power cap costs nothing measurable. Closed 2026-09-09.** Measured both ways with
      `scripts/sweeps/power-and-clocks.ps1`, and then at 350 W from an elevated shell:

      | | 315 W (cap) | 350 W (default) |
      |---|---|---|
      | decode `tg1024` | 37.16-37.55 tok/s | **37.15 tok/s** |
      | prefill `pp8192` | 1,204-1,255 tok/s | **1,228.5 tok/s** |
      | decode SM clock | 1,500-1,515 MHz median | 1,545 MHz median |
      | prefill SM clock | 1,635-1,650 MHz median | 1,680 MHz median |
      | memory clock | 9,501 MHz, every sample | 9,501 MHz, every sample |
      | board power | 314 W median | 342-349 W median |
      | `sw_power_cap` active | 161-168 of 162-168 | **161 of 163** |
      | thermal throttle | 0 samples | 0 samples |

      **+35 W (+11% of budget) buys +3% SM clock and 0% throughput, in both phases.** Decode is
      identical to within 0.4%; prefill is inside its own run-to-run spread. So the cap can stay
      where it is, and every number in this file stands without an asterisk.

      Why it costs nothing is the useful part. The memory clock **never leaves 9,501 MHz** — the
      full 19 Gbps spec — at either power limit, in any sample of any run. Decode is
      bandwidth-bound, so SM clock is not what it is waiting for; prefill is the phase where SM
      clock could matter and it is the phase the cap squeezes least (1,635 vs 1,680 MHz, 2.8%).
      Nothing was ever thermally throttled either, at any temperature reached, up to 80 °C.

      Note the cap is still *binding* at 350 W — `sw_power_cap` is active in 161 of 163 busy
      samples there too — so this is not "the cap stopped mattering", it is "this workload does not
      convert board power into throughput". Lifting to the 400 W maximum would not change that
      conclusion for decode, which cannot go faster than its memory clock allows.

      The compute ratios were self-cancelling as predicted: prefill's "~30% of INT8 MMA peak"
      divides a capped measurement by a ceiling probe run under the same cap.

      Also corrects the figures this entry originally carried. It claimed the SM clock "swings
      1,665-1,755 MHz"; steady state is 1,500-1,515 on decode and 1,635-1,650 on prefill at 315 W.
      Individual samples do reach 1,905-1,935 MHz, but only in the first sample or two before the
      cap clamps a cold card — the first draft of the sampling script reported
      `Measure-Object -Average` under a column labelled "median", which is exactly how a boost
      spike gets quoted as a steady clock, so it now takes a real median and says why.

      **Still worth doing, and not blocked on anything: `nvidia-smi -lgc` for measurement runs.**
      The 3-5% between-process spread is now the largest source of noise in every end-to-end
      comparison here. It forced the DFlash2 tile A/B to interleave the two binaries within each
      repetition rather than run one build then the other, and it made an earlier comparison need a
      control column to be readable at all. That is a measurement-hygiene fix, not a performance
      one, and it is independent of the power limit.

- [x] **`compute-sanitizer` and `ncu` both work. Closed 2026-09-09 — this entry was wrong about
      both, and the way it was wrong is worth keeping.**

      **`compute-sanitizer`: there are two copies installed and one of them lies.**

      | copy | version | result on `ninfer_gdn_input_proj_test.exe` |
      |---|---|---|
      | `CUDA\v12.4\compute-sanitizer\` | 2024.1.0 | **exit 0, "ERROR SUMMARY: 0 errors", and the test never ran** |
      | `CUDA\v12.8\compute-sanitizer\` | current | runs to completion, real test output, 0 errors |

      The v12.4 copy produces two lines of output — the banner and a clean error summary — and exits
      0 without executing a single line of the binary. **That is a false pass, which is worse than
      the failure this entry described**: a sanitizer run that reports success having checked
      nothing. The v12.8 copy runs everything. `compute-sanitizer` on `PATH` resolves to
      `CUDA\v12.8\bin\compute-sanitizer.bat`, which is the good one, so an unqualified invocation is
      fine — but an absolute path to the v12.4 directory is not, and that is presumably how this
      went wrong.

      Verified working on v12.8: `memcheck` and `initcheck` both run `ninfer_gdn_input_proj_test`
      and `ninfer_linear_swiglu_fp8_test` to completion, 0 errors. **`initcheck` also runs clean on
      `ninfer_softmax_attention_test`, which is the Op #49 fixed** — the exact investigation this
      entry says had to be done by hand instead.

      Size is not the problem either. The 598 MB
      `ninfer_qwen3_6_27b_score_real_test.exe` this entry named launches fine; it skips because
      `NINFER_QWEN3_6_27B_WEIGHTS` is unset, and compute-sanitizer then says *"Target application
      terminated before first instrumented API call"* — which is a consequence of the test making
      no CUDA calls, not a launch failure. That message is easy to read as one.

      **`ncu`: present, and the blocker was a permission, not a missing file.** The entry says
      "`ncu.exe` is also missing from the Nsight Compute 2025.1.0 install directory". True as
      written and misleading: the launcher is `ncu.bat` at the top of that directory, with the
      binary under `target\windows-desktop-win7-x64\`. It runs. What actually stopped it is
      `ERR_NVGPUCTRPERM` — **GPU performance counters are administrator-only by default on
      Windows** — which is a permission, fixable per-run by an elevated shell with nothing
      persistent and no reboot. `scripts\sweeps\admin-profile.ps1` exists for that and produced
      §2c's MoE counters on the first try.

      The lesson for the next investigation: when a tool appears not to work, check whether a
      second copy is shadowing it and whether the failure is a permission. Neither of the two
      things this entry called broken was broken.

- [x] **The perplexity harness's 0.019% drift is two commits, both now named.** `f3f6c724` turned
      the int8-activation prefill route on by default (−0.0092%) and `1c12516e`, the upstream
      catch-up, did the rest (−0.0101%). Eight measurements, 2026-09-10. The original entry follows,
      because the eliminations in it are what made eight measurements enough.

      **Transition 1 = `f3f6c724`, "register the integer-activation route as
      `LinearPolicy::AllowA8Int`".** The mechanism is not a guess — the bracket lands on exactly the
      commit whose diff removes the opt-in:

      | commit | | perplexity |
      |---|---|---:|
      | `9d84659c` | before the cluster | 4.34326255065906 |
      | `4f0be008` | A8 route present, still env-gated | 4.34326255065906 |
      | **`f3f6c724`** | **env gate removed** | **4.34286437475351** |
      | `c41b29dd` | follow-up fix | 4.34286437475351 |

      `0b0b098d` had added the route behind `NINFER_W4A8_PREFILL=1`, off by default, and `4f0be008`
      extended it — both measure the old value to twelve figures, which is the env gate being
      provably inert. `f3f6c724` deletes `q4a8_swiglu_enabled()` and makes `Q4G64_F16S` and
      `Q5G64_F16S` return `AllowA8Int` unconditionally under `NINFER_SM8X_COMPAT`. So the drift is
      **int8 activations on the 27B MLP prefill GEMMs**, which the commit itself documents as "0.9%
      relative L2 on that projection ... for about twice the prefill GEMM rate".

      **And the sign is the point.** Perplexity went *down*, 4.343263 → 4.342864. Quantizing
      activations to int8 did not improve the model; a 0.9% relative perturbation moves the fifth
      decimal, and which way it moves is a coin flip. That is the honest reading of every number in
      this range, and it is why the drift is a **floor on comparison, not a regression**: any change
      that perturbs the prefill GEMMs at the 1%-relative level can move perplexity by ~0.01%
      in either direction.

      **Transition 2 = `1c12516e`, the upstream catch-up (#16)**, pinned by the table further down.
      Its sub-mechanism is the one thing still open, and it is narrow: `rmsnorm.cuh`'s new `FixedD`
      template parameter, which makes the reduction trip count a compile-time constant and lets the
      compiler re-associate the sum-of-squares. Two other suspects are eliminated by proof, not by
      elimination-of-the-rest — the int8 KV codec refactor is bit-identical by arithmetic, and route
      boundaries cannot move perplexity at all. Confirming `FixedD` is one build with the
      specializations forced off, and it is the only loose end.

      **What this buys.** Perplexity comparisons in this repository now have a known floor and a
      known cause. `k8v4`'s −0.008% from #49, reported as "no change" because it sat under the
      drift, sits under a drift that is now explained — and the standing rule is that a quality
      claim smaller than ~0.01% needs the same build on both sides, not just the same corpus.

      ---

      **Everything below is the original entry, kept because its eliminations are what made eight
      measurements enough. Read it as history: the drift is no longer unexplained.**

      **The perplexity harness drifts 0.019% from the published figures and nobody knew why.**
      Re-measuring all six KV formats (#63) moved the three formats #49 does not touch by
      -0.009% to -0.019%: `int8` 4.343263 to 4.342425, `bf16` 4.343225 to 4.342517, `rk8v4`
      4.346811 to 4.346413. Same corpus, same window, same 261,167 scored tokens, and those code
      paths are byte-identical run to run. So something else differs between whenever the published
      numbers were taken and now — a build change, or a harness detail.

      It is small and it is systematic, which is exactly why it should be named: it is the floor on
      every perplexity comparison in this repository, and `k8v4`'s -0.008% "improvement" from #49
      sits underneath it and was reported as no change for that reason. Bisecting it would make
      future quality claims sharper by an order of magnitude.

      **The six speedups shipped 2026-09-09/10 do not move it, which is worth having on record.**
      `int8` at the head of `perf/gdn-split4-width8` scores **4.342425372232802** on 261,167 tokens
      — identical to the 4.342425 recorded below, to every figure quoted. That includes the q5 GEMV
      dequant change, which **reorders float accumulation** by hoisting the group scale out of the
      inner loop; the claim made for it was "it rounds less, not more", and this is that claim
      measured rather than argued. So the whole batch is quality-neutral, and §3's "tile geometry
      does not perturb perplexity" now extends to accumulation order within a group.

      **Three candidates eliminated 2026-09-09, and the current value is exactly reproducible.**
      `int8` re-runs to **4.342425** at HEAD — identical to twelve significant figures, on 261,167
      scored tokens, in 465 s. So the drift is a real, repeatable difference between two builds and
      not measurement noise.

      **Not the artifact.** `qwen3_8_27b.ninfer` has an mtime of 29 August 14:06, and the old
      figures were published in `838c8b5d` on 29 August. The weights have not been reconverted
      since before the old measurement, which was the tidiest available explanation for a small
      systematic shift across every format at once.

      **Not the scoring set.** The token count is 261,167 in both, so the harness is windowing and
      selecting exactly the same tokens. Whatever changed, changed arithmetic rather than what is
      being averaged.

      **Not kernel or tile selection — and this one is worth keeping.** The natural theory was that
      route-table changes (the upstream catch-up rewrote several bands) select a different MMA tile,
      whose reduction runs in a different order, moving the score in the fifth decimal. Tested
      directly: routed width 1024 in `q5_linear_add` from `MmaResidualR64C128` to `R64C64`, which
      the harness exercises on every one of its four 1,024-wide prefill chunks, and re-ran. The
      score rate moved 568.9 to 535.8 tok/s, so a genuinely different kernel ran — and perplexity
      came back **bit-identical at 4.342425**. Two different MMA tiles over the same weights produce
      the same score to twelve figures. **Tile geometry does not perturb perplexity at all**, so no
      route change anywhere in this repository can be the cause of this drift, and route changes
      need not be treated as a quality risk.

      What is left is a genuine numerics change in some kernel between 29 August and 9 September.
      Note the three shifts are *not* uniform (−0.0193%, −0.0163%, −0.0092% for `int8`, `bf16`,
      `rk8v4`), so whatever it is does not simply offset every score by a constant.

      **The bisect was started 2026-09-10 and it has already falsified this entry's central
      assumption, which is that there is one change to find.** There are at least two. Measured
      endpoints and the first interior point, all `int8`, all 261,167 scored tokens:

      | commit | date | perplexity |
      |---|---|---:|
      | `838c8b5d` (old, `git bisect good`) | 2026-08-29 | 4.343262550659062 |
      | `66378f06` | 2026-09-03 | **4.342864374753507** |
      | `HEAD` (new, `git bisect bad`) | 2026-09-10 | 4.342425372232802 |

      `66378f06` is a **third value**, not either endpoint, so the interval contains two distinct
      transitions: `838c8b5d → 66378f06` is −0.0092% and `66378f06 → HEAD` is −0.0101%. They sum
      to −0.0193%, which is exactly the total `int8` drift recorded above — so this is not a third
      measurement wobbling around one step, it is two steps of almost equal size that happen to add
      up to the published gap. **A plain `git bisect` cannot find two transitions**; it will converge
      on whichever one it happens to bracket and report it as the cause, and the next person will
      then be unable to reproduce the full 0.019% from that one commit. Each half has to be bisected
      separately.

      **`--first-parent` is right for the later half and WRONG for the earlier half, and an earlier
      version of this entry said to use it for both.** For the later half it is right: the naive walk
      descends into the upstream catch-up branch, where the code predates the sm_86 port and
      `00f02055` refuses outright with *"NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got
      '86'"*. Those commits cannot be built on this card at all — not `skip`-able noise, a region the
      bisect must not enter — and restricting to first-parent both avoids them and cuts 477 commits
      to **96**, at the right granularity: a merge names the PR that moved the number.

      **For the earlier half it walks off the path entirely, because `838c8b5d` is not on the
      first-parent chain.** `838c8b5d` is a commit *inside* the `feat/w4a8-prefill` branch, and
      master at that moment did not have the perplexity harness at all. Proof, in the order that
      makes it obvious:

          git merge-base --is-ancestor 838c8b5d 12ca6cbd    # false
          git merge-base --is-ancestor 838c8b5d b4ca75a9^2  # true  -- the merged branch
          git ls-tree -r --name-only 12ca6cbd | grep perplex # nothing at all

      So `838c8b5d` enters `b4ca75a9` through its **second** parent. `git rev-list --first-parent
      838c8b5d..b4ca75a9` therefore lists master-line commits that are not on any path from
      `838c8b5d`, and the bisect walked straight onto one: at `12ca6cbd` the build dies with
      **`ninja: error: unknown target 'ninfer-perplexity'`** because the harness does not exist
      there. That is not a broken commit to `skip`; it is a commit the range should never have
      contained.

      **The rule, which is worth more than either bisect.** Before trusting any range, check that
      the good commit is actually an ancestor along the chain you intend to walk, and check the thing
      you are measuring exists at the candidate:

          git merge-base --is-ancestor <good> <candidate>
          git show <candidate>:apps/CMakeLists.txt | grep ninfer-perplexity

      **And do not read commit order off `--date=short`, which prints the *author* date.**
      `838c8b5d` is authored 2026-08-29 and committed 2026-09-01. Every apparent date inversion in
      this range is that, and it is what made the first-parent list look plausible.

      The real earlier-half range is `838c8b5d..b4ca75a9` following the actual DAG: **34 commits**,
      all of them ours and all sm_86-capable.

      **Build only the target you run.** `cmake --build build-ninja` builds the test binaries too,
      and at `de5fc15a` `tests/ops/softmax_attention/causal_cache.cpp` fails to compile under MSVC
      (`error C3493: 'order' cannot be implicitly captured`) for reasons that have nothing to do
      with perplexity. Passing `--target ninfer-perplexity` avoids skipping a perfectly measurable
      commit, and cuts each step's build time as well. `scripts/sweeps/ppl-bisect-step.ps1` does
      this.

      **A commit that does not touch `src/`, `include/` or `apps/` cannot move the number, so filter
      before you measure.** Of the 25 first-parent commits in the later half, 16 are scripts, docs or
      release commits and 6 more touch only `src/serve` (which `ninfer-perplexity` does not link),
      `src/core/nvtx.h` or a chat template. That leaves 3 real candidates out of 25. In the earlier
      half it is starker: 34 commits, of which the entire W4A8 prototype series is `src=0`. The
      filter is one line and it turned a 5-step bisect into a 2-step one:

      ```
      git log --first-parent --format='%h|%s' A..B | while IFS='|' read h s; do
        printf '%s src=%s %s
' "$h" "$(git diff --name-only $h^ $h -- 'src/*' 'include/*' 'apps/*' | wc -l)" "$s"
      done
      ```

      **Transition 2 is `1c12516e`, "Upstream catch-up: DFlash2 unblocked, measured sm_86 route
      boundaries, upstream small-T adopted (#16)" — pinned to one commit, −0.0101%.** Measured
      2026-09-10, all `int8`, all 261,167 scored tokens:

      | commit | date | perplexity | |
      |---|---|---:|---|
      | `64403ada` dual-GPU graph mode (#15) | 2026-09-07 | 4.34286437475351 | mid |
      | `1c12516e` upstream catch-up (#16) | 2026-09-08 | **4.3424253722328** | **new** |
      | `de5fc15a` (#34) | 2026-09-08 | 4.3424253722328 | new |

      `1c12516e`'s first parent *is* `64403ada`, so there is nothing between them: the transition is
      that merge. Note `64403ada` rewrote 47 source files including every dense causal-cache
      attention kernel and moved the score **not at all** — more evidence for §3's finding that
      kernel and tile refactors do not perturb perplexity.

      **The mechanism inside `1c12516e` is not yet isolated, and two of the three obvious suspects
      are already ruled out by reading.** 153 files under `src/ops/` changed. Of them:

      - **`src/ops/kv_cache/int8_g64_codec.cuh` is not it**, despite being the int8 KV codec. The
        change is a pure addition (42 lines, 0 deletions) of f16 loaders, and
        `prompt_i8.cuh` switching to them is provably bit-identical: the old form multiplied an
        int8 code by an f16 scale in `half2`, the new form in FP32 before a single rounding, and an
        8-bit code times an 11-bit scale needs 19 bits, exact in FP32's 24. Both round exactly once
        to the same value. Upstream also verified it empirically, byte-identical `OP_ERROR_STATS`.
      - **The route-boundary half of the commit is not it either**, by §3's own tile-geometry
        result above.
      - **`src/ops/kernel/rmsnorm.cuh` is the leading candidate.** It gained a `FixedD` template
        parameter that replaces the runtime `d` with a compile-time constant. No arithmetic was
        edited — but a constant trip count is exactly what lets the compiler fully unroll and
        **re-associate the sum-of-squares reduction**, which changes rounding. RMSNorm runs on every
        layer of every token, and it is format-independent, which is the only kind of mechanism that
        can explain `bf16` KV moving (−0.0163%) alongside `int8` and `rk8v4`. Confirm it by
        building `1c12516e` with the `FixedD` specializations forced off and re-scoring.

      **Transition 1 is at or before `b4ca75a9`, and it is one of two clusters.** Both `04e22c3f`
      (2026-09-04, upstream catch-up) and `b4ca75a9` (`feat/w4a8-prefill`) already read
      4.34286437475351, the mid value, and `838c8b5d` *is* a genuine ancestor of `b4ca75a9`, so the
      bracket is sound even though the first-parent list was not. Walking the real 34-commit range
      with the source filter leaves two candidate clusters and nothing else:

      1. **The integer-activation prefill route for the 27B MLP** — `0b0b098d` (opt-in for
         `gate_up`), `4f0be008` (extended to `mlp/down`), `f3f6c724` (registered as
         `LinearPolicy::AllowA8Int`), `c41b29dd` (keyed on registered shapes). This quantizes
         *activations* to int8 on the prefill MLP GEMMs, which is the harness's hot path on every
         one of its 124 windows. The accuracy claim made for it was **"accuracy-neutral"** and
         pinned against a relative error — which is exactly the claim that permits a −0.0092%
         perplexity move without anyone noticing. This is the a priori favourite.
      2. **`91e78ab1`, "keep the sm_86 W8 linear_add single-column tail on a K-correct kernel"** —
         and this one is not a tuning change, it is a **fixed out-of-bounds read**. On sm_86 both W8
         `linear_add` split-K launchers hand a leftover single column to a kernel that bakes
         `kDecodeK = 6144`; on the `K = 4096` shape that tail *read 2,048 BF16 past the end of `x`
         and indexed the weights with the wrong row stride*. Fixing it necessarily changes results,
         and being sm_86-only it explains why our numbers moved when upstream's published ones did
         not.

      **The branch name is misleading, and the source filter is what shows it.** Every W4A8 GEMM
      commit in that branch — `dbdf207e`, `90af65cd`, `9d84659c` and the rest — is `src=0`: they are
      prototypes under `tools/`, and none of that code reached the engine. So "the W4A8 prefill
      merge" did not put W4A8 in the prefill path. What it actually shipped into `src/` was the A8
      *activation* route and the `linear_add` tail fix.

      **`c41b29dd` measures 4.34286437475351, the mid value — which eliminates the out-of-bounds
      fix.** `91e78ab1` is *newer* than `c41b29dd`, so the transition had already happened by then.
      That is worth stating plainly because `91e78ab1` was the more alarming of the two candidates:
      a kernel reading 2,048 BF16 past the end of its input is a real defect, it was really there,
      and it is really fixed — but **it is not what moved perplexity**, and the 27B scoring path
      evidently never hit the `K = 4096` single-column tail.

      **So transition 1 is at or before `c41b29dd`, and the next test is `9d84659c`** — the commit
      immediately preceding the A8 cluster. Everything between it and the cluster is `src=0`, so it
      splits the remaining candidates cleanly:

      | `9d84659c` reads | conclusion |
      |---|---|
      | old (4.343263) | the A8 activation route is the cause — one of `0b0b098d`, `4f0be008`, `f3f6c724`, `c41b29dd` |
      | mid (4.342864) | the cause is earlier, and the only real candidate left is **`2ba10da2`, "perf(kv): give rk8v4 values a 32-value group"** (`src=12`) |

      That second branch would be a surprise worth chasing: `2ba10da2` is an *rk8v4* change and we
      are measuring *int8*, so it could only move this number by touching shared codec code. Note
      the arithmetic coincidence that makes it plausible anyway — transition 1 is −0.0092% and
      **rk8v4’s entire recorded drift is also −0.0092%**. If `2ba10da2` turns out to be the
      commit, that is one change explaining both, and the non-uniformity this entry opens with stops
      being mysterious.

      **A warning for whoever runs those two.** `scripts/sweeps/ppl-bisect-step.ps1` classifies
      against a single threshold, which is correct for transition 2 (mid and old both count as
      "good") and **wrong for transition 1**, where the two values to separate are 4.343263 and
      4.342864 and the printed verdict line will say "OLD" for both. Read the printed perplexity,
      not the verdict.

- [ ] **`27b_load_plan`'s DFlash2 binding matrix needs four artifacts. Two are on this disk today
      — one of them misidentified all along — and the other two were never published.** Checked
      upstream 2026-09-10, which is what this entry told the next person to do.

      `huggingface.co/neroued/Qwen3.8-27B-NInfer` has four commits and has only ever published one
      `.ninfer` file:

      | commit | date | |
      |---|---|---|
      | `dc370fb6295a` | 2026-09-06 | Update artifact with DFlash2 companion weights |
      | `18dfc887423f` | 2026-08-19 | docs(eval): publish qwen3.8 groupwise-int results |
      | `3526913004b1` | 2026-08-14 | Add Qwen3.8-27B NInfer artifact |
      | `6925b5541b49` | 2026-08-06 | initial commit |

      **`qwen3_8_27b_nvfp4.ninfer` returns 404 at the pinned revision and appears in no commit's
      file list.** So the two NVFP4 artifacts are not "deleted", they were never published, and no
      amount of revision archaeology will produce them.

      **The two groupwise ones are both here.** `qwen3_8_27b.ninfer` at the pre-DFlash2 revision
      `18dfc887` is 18,210,531,328 bytes with SHA-256
      `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` — **byte-identical to
      `models/qwen3_8_27b.ninfer` on this box**, hash verified, not inferred from the size. So the
      artifact this entry has been calling missing has been sitting in `models/` the whole time
      under its plain name, because `dc370fb6` later *replaced* the same filename upstream with the
      DFlash2 build, which is our `qwen3_8_27b_dflash2.ninfer`. That is the second time this cycle
      an artifact was "missing" only from the env block; see the note under "Environment for the
      real-model tests".

      | the matrix wants | status |
      |---|---|
      | `NINFER_QWEN3_8_27B_OLD_WEIGHTS` | **have it** — `models/qwen3_8_27b.ninfer`, SHA-verified as rev `18dfc887` |
      | `NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS` | **have it** — `models/qwen3_8_27b_dflash2.ninfer` |
      | `NINFER_QWEN3_8_27B_NVFP4_OLD_WEIGHTS` | **never published** |
      | `NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS` | **never published** |

      So this is no longer "artifacts nobody has". It is **half a matrix that can run once the env
      block names the old artifact**, and a second half that would have to be **converted locally**
      — `tests/convert/qwen3_8_27b/test_nvfp4_inventory.py` says a Qwen3.8 NVFP4 recipe exists in
      this repo, so that is a conversion job rather than a download, and it needs the source
      weights and disk for two more ~18 GB artifacts.

      **Worth weighing against §1 before doing it:** NVFP4 weights cannot be loaded on sm_86 at all,
      so a locally converted Qwen3.8 NVFP4 artifact would exercise the binding matrix's *planning*
      path and nothing else on this card.

      *Fixing the env block is the free half and is done below.*

- [x] **DFlash2 acceptance on realistic text — measured 2026-09-09, and it is nothing like the
      committed corpus's answer.** `scripts/sweeps/dflash2-draft-tokens-realtext.ps1` now parses the
      acceptance metrics the CLI has always printed, so this needed no new instrument and no local
      tokenizer. 27B DFlash2 artifact, INT8 KV, greedy, 256 generated tokens on the model's own
      prose, medians of three:

      | k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
      |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
      | acceptance % | **80.9** | 67.6 | 51.7 | 49.1 | 42.1 | 33.1 | 28.2 | 31.0 | 23.2 | **19.5** |
      | tok/round | 1.81 | 2.35 | 2.55 | 2.95 | 3.07 | 2.97 | 2.93 | 3.45 | 3.27 | 3.27 |
      | decode tok/s | 47.3 | 55.2 | **58.6** | 58.3 | 57.2 | 51.0 | 50.5 | 47.5 | 39.7 | 39.3 |

      **Acceptance falls from 80.9% to 19.5% as the draft window widens**, monotonically except at
      k=7 -> 8, where it rises 28.2% -> 31.0% before resuming the decline. That is the shape a draft
      head on real prose should have and is the opposite of the committed corpus's flat 100% at
      every count. So `bench_corpus.ids` was not merely optimistic, it was reporting
      the fixture; these are the numbers to quote.

      **Tokens-per-round plateaus around 3.0 from k=4**, which is the real justification for
      recommending four: everything past it buys ~0.1 tokens per round for a wider verification
      pass. Throughput peaks at k=3-4 (58.6 / 58.3), consistent with the recommendation in
      `docs/cli.md`.

      MTP3 reaches 58.3 tok/s at **56.0%** acceptance and 2.67 tok/round, and `mtp3 --lm-head-draft`
      is the fastest configuration measured at **62.9 tok/s**. `--lm-head-draft` remains within noise
      for DFlash2 at every count, as previously recorded.

- [x] **DFlash2 corpus numbers on sm_86 — the throughput half, kept for the reasoning.**
      Only single-prompt smoke numbers exist (text 20.0% / 2.38 tok-per-round, vision 85.7% /
      7.00). `docs/performance.md` deliberately does not reproduce upstream's tables because they
      are sm_120 — see the provenance banner there.

      **Do not produce these with `ninfer_bench` on the committed corpus.** #65 established that
      `bench/fixtures/bench_corpus.ids` is 65,536 tokens over 682 distinct ids with 98.4% of
      bigrams repeated, and DFlash2 reports *exactly 100% acceptance at every draft count from 1
      to 12* on it. Any acceptance or tok-per-round figure from that path is a statement about the
      fixture. The text throughput question is answered (#65, via the serving path on generated
      prose), but **acceptance and tokens-per-round are still unmeasured on realistic text**, and
      that is what this entry now wants. Either bake a diverse corpus with
      `make_bench_corpus.py --source-text` — `eval/corpora/perplexity-1m/` is real wikitext and
      pg19 prose sitting right there, though it needs a local HF tokenizer that this box does not
      have — or extend the real-text sweep to report acceptance. Vision acceptance is unaffected by
      any of this: it is measured on the committed image fixture, not the token corpus.
- [x] **Speculative decoding is not bit-identical to greedy, it should not be required to be, and
      the premise this entry rested on is false. Decided 2026-09-09, with evidence.**

      The entry said "DFlash2 and MTP produce byte-identical output *to each other*". **They do
      not.** Hashing every run's generated text across 23 configurations x 3 repetitions:

      | | |
      |---|---|
      | every configuration self-deterministic | **yes — 23 of 23, one hash across three runs each** |
      | `none` (width-1 greedy) | its own unique hash, matched by nothing |
      | `mtp3` / `mtp3+head` | one hash, distinct from **every** DFlash2 configuration |
      | DFlash2 across draft counts | **8 distinct hashes** over k = 1..12 |

      So the divergence is real, but it is not one divergence: output depends on the backend *and*
      on the draft count. "Bit-identical to greedy" was never a single target — there are eight
      different DFlash2 outputs to choose between before the question even reaches MTP.

      **The decision: no, and it is architecturally incompatible with the feature.** The speedup is
      that verification evaluates k+1 columns in one pass. That pass's reduction order for the
      accepted column is not the width-1 GEMV's, and the only way to make it so is to compute the
      accepted column with the width-1 kernel on every round — which is exactly the work
      speculation exists to avoid. Requiring bit-identity would mean pinning one reduction order
      across every verification width, a far stronger constraint than "lossless".

      **And this class of difference is already measured to be quality-neutral in this repository.**
      §3's perplexity entry swapped `q5_linear_add` from `MmaResidualR64C128` to `R64C64` — a
      genuinely different MMA tile and reduction order, confirmed by a 6% change in score rate —
      and perplexity came back **bit-identical at 4.342425 to twelve significant figures**. So a
      changed reduction order does not move quality; it flips an argmax only at a near-tie, where
      the two candidates are by construction near-equally probable.

      **What should be guaranteed instead, and now is:**

      1. **self-determinism** — a configuration reproduces its own output run to run. Measured
         above, 23 of 23. This is the property that actually matters for reproducing a bug report,
         and until now nobody had checked it.
      2. **quality parity** — unaffected by reduction order, per the perplexity result.
      3. **documented divergence** — `docs/performance.md` says so, and now says it correctly.

      A test asserting bit-identity to greedy would fail permanently by design, which is worse than
      no test. A test asserting self-determinism and cross-run stability would be worth having, and
      `dflash2-draft-tokens-realtext.ps1`'s `content_sha256` column is the mechanism for it.
- [x] **q4 SwiGLU `{513,640}` settled: it goes to c128, and the alternation is gone.** The band
      stayed on `Materialized` because the measurement could not separate the two — over four runs
      Materialized won T=576 three times, while c128 there ranged 4147–4959 µs (19.6% spread) and
      the margins outside the outlying run were ±2%.

      The blocker was the instrument, not the card. `ColdTiming` had carried `min_us` and `p95_us`
      all along and the sweep harness threw both away, printing only the median, so the spread that
      made the decision impossible was never visible in the table. Added `--spread` to
      `bench/ops/schedule_sweep.cuh` and to the q4 SwiGLU bench (which has its own arg loop), then
      re-measured at 31–51 repetitions on an idle card. c128 wins all three widths in four
      independent runs — 1.3–2.5% at 513, 1.2–7.5% at 576, 7.3–11.2% at 640 — and its *fastest*
      sample beats Materialized's fastest everywhere, by 7–9% at 640 with no overlap at all. Sign
      consistent 12 times out of 12. `{49,512}`, `{513,640}` and `{641,∞}` collapse into one
      `{49,∞}` c128 route and the table drops from seven entries to five.

      **Two things worth more than the route change itself.**

      *The bench overstates margins.* Confirmed in situ with nsys on a single 600-token chunk:
      Materialized costs 269.73 ms in `q4_rowsplit_gemm_mma` plus 5.16 ms in
      `silu_and_mul_dim0_split` across 64 layers = 274.89 ms, against 268.32 ms for the c128 pair
      kernel. c128 still wins, but by **2.4%, not 7.5%** — and both are slower per call in situ
      than in the bench (4193 vs 3625 µs for c128; 4295 vs 3927 for Materialized). The bench
      flushes L2 before every repetition and a real prefill does not arrive with a cold cache, so
      the flush penalises Materialized's second pass more than production does. **This applies to
      every band in every one of these route tables**, all of which were decided on cold-flush
      numbers alone. Nothing is known to be mis-routed because of it — the sign was right here —
      but no margin in those comments should be quoted as a speedup.

      *End-to-end it is invisible, and that is expected.* `pp600` measured 886.0–888.8 tok/s with
      c128 against 888.5–892.1 on Materialized: inside the run-to-run spread, because 6.6 ms of a
      ~660 ms prefill sits under the ±3% that unrelated kernels move between runs.

      One trap on the way: `--prefill-chunk 640` looks like the obvious way to exercise this band
      and does not touch it. `q4a8_swiglu` claims any width with `tokens >= 128 && tokens % 128 ==
      0`, and `--prefill-chunk` is required to be a multiple of 128, so **every full prefill chunk
      goes to the integer-activation kernel and never reaches this table at all.** The `{49,∞}`
      route is reached only by decode widths and by a prompt's ragged tail chunk. That is why
      `{513,640}` was simultaneously close and inconsequential for so long. The first end-to-end
      comparison here was run at chunk 640 and measured nothing, twice, before the profile showed
      `q4a8_swiglu_kernel` where `q4_linear_swiglu` was assumed to be.

---

## 4. Test-criterion calibration — closed

The §5 audit floored every BF16 gross bound at two rounding steps (#20). Two criteria were left
out because the BF16 floor argument was thought not to reach them. Measured on 2026-09-09 with
`NINFER_OP_REPORT_STATS=1` over the full case matrix, it reaches both — and neither was quite the
shape the entries described.

- [x] **`sparse_moe` sits at 0.92 of its limit** with `gross_relative_to_max_reference = 0.0`.
      Closed. The Op is not uniformly inaccurate; its gross error **steps at a route boundary**:

      | codec | T | max_abs | max_reference | BF16 steps |
      |---|---|---:|---:|---:|
      | q4+q5 | 1..46 | 4.85e-4 | 0.1778 | 0.70 |
      | q4+q5 | 47..4097 | 2.36e-3 | 0.1778 | **3.40** |
      | q4+q6 | 1..46 | 4.88e-4 | 0.1766 | 0.71 |
      | q4+q6 | 47..768 | 2.51e-3 | 0.1766 | **3.64** |
      | w8+w8 | 1..19 | 9.18e-4 | 0.2718 | 0.87 |
      | w8+w8 | 20..768 | 3.69e-3 | 0.2718 | **3.47** |

      So "the largest observed BF16 error in the tree — 3.64 steps against 0.09–1.53 everywhere
      else" is **one route above a T boundary**, and below that boundary this Op sits at 0.70–0.87
      like everything else. The question was "either that kernel is genuinely less accurate than
      every other BF16 Op, or its criterion measures something different"; the answer is that the
      Op has two routes with a 5x accuracy spread and the criterion was sized for the worse one.
      A different accumulation order over more terms is a property of the algorithm, not a defect.

      The real fragility was the bound's *shape*, not its size. `gross_absolute` alone at 4.0e-3
      does not scale with the data, so the same relative accuracy on a case whose max_reference is
      0.5 rather than 0.27 would produce ~6.8e-3 and fail spuriously. Six rounding steps of
      max_reference covers the measured 3.64 with headroom and still bounds a genuinely wrong
      element hundreds of times more tightly. Worst case now sits at **0.36** of its limit.
      `relative_l2` is untouched at 1.2e-2 against a measured 1.118e-2 — it is the criterion that
      constrains accuracy, and #20's convention leaves it alone.

- [x] **`gated_delta_net`'s state criterion sits at 0.87** and compares an FP32 output. Closed,
      and the reason it was excluded turns out not to matter: the error's *source* is BF16 anyway.

      | path | max_abs | max_reference | BF16 steps |
      |---|---:|---:|---:|
      | decode / small-T / batch update | 1.2e-8 .. 3.5e-8 | ~0.09 | **0.00** |
      | exact chunk / chunk-tail / two-chunk | 4.6e-4 .. 8.1e-4 | ~0.23 | 0.53–0.88 |

      The non-chunked paths are exact to eight decimal places. Everything above 1e-4 comes from the
      chunked recurrence, where the carried state crosses a BF16 intermediate at each chunk
      boundary — and just under one rounding step of the state's own magnitude is exactly what one
      such round-trip costs. The FP32 output dtype was the wrong thing to reason from.

      That makes the hand-picked 3.9e-3 the same mistake #20 was written to fix: it is about 1.0
      rounding step against an observed 0.88, so the bound and the error were the same quantity
      with 12% between them. It now uses `kBf16GrossRelativeFloor` like every other criterion the
      audit touched, and the worst case sits at **0.44**.

      `relative_l2` stays at 2.7e-3 against a measured 2.582e-3 — 0.92–0.96, which is tight, but
      loosening it would stop the chunked recurrence being checked at all.
---

## 5. Reproducibility — closed

- [x] **fp8, k8v4 and nvfp4 causal attention are not run-to-run deterministic.** Closed by #49,
      2026-09-09, and it was not a tolerance curiosity — it was a data race corrupting output.

      Five quantized kernels called `dequant_k_tile()` immediately after `cp_wait<0>()` with no
      `__syncthreads()` between them. `cp_wait<0>()` retires only the *calling thread's*
      cp.async group, while `dequant_k_tile()` has every thread walk the whole tile, so each
      thread reads bytes another thread issued. The prologue in the same files pairs the two
      correctly and the lambda's own comment states the requirement; only the steady-state loop
      omitted it, and it omitted it in exactly the three storage families named here. bf16 and
      int8 pair them everywhere and were always byte-identical.

      What it cost, on the 27B through `EnginePurpose::CausalScoring`: `score_tokens` called
      twice on the same window disagreed on **all 1024** logprobs — median 0.2, mean 3.2, worst
      ~26. A token at -0.0013 in one run was -22.61 in the next: p about 1 becoming p about
      1e-10. Total logprob over a 200-token window ranged from -352 to -588 across five calls in
      one process, and again across processes. int8 and bf16 returned the identical checksum
      every time.

      **Why it hid for so long.** Greedy decode reads only the last column's hidden state.
      Prefill's other columns — the ones scoring and perplexity consume — are discarded there,
      so generation looked perfect and stayed byte-identical while the same kernels were
      producing garbage for every other position.

      After the fix, `ninfer_softmax_attention_test` run three times from one binary is
      byte-identical (15 differing `OP_ERROR_STATS` lines before), and `27b_score_real` reports
      `max_overlap_error=0` for fp8, nvfp4, k8v4 and int8 alike.

      *Ruled out on the way, so it is not re-investigated:* the `split >= active_split_count`
      early return in the small_t partial kernels skips `write_neutral()`, which §2c flags as
      safe only by coincidence. Writing the merge identity there changed nothing — 15 differing
      lines before and after. It is a latent hazard, not this defect.

- [x] **The published perplexity figures for fp8, k8v4 and nvfp4 were measured through the race
      above.** Re-measured 2026-09-09, twice each, with int8 and bf16 as controls whose code paths
      #49 does not touch.

      | format | published | re-measured | delta |
      |---|---:|---:|---:|
      | `int8` (control) | 4.343263 | 4.342425 | −0.019% |
      | `bf16` (control) | 4.343225 | 4.342517 | −0.016% |
      | `rk8v4` (control) | 4.346811 | 4.346413 | −0.009% |
      | `fp8` | 4.347181 | **4.344724** | **−0.057%** |
      | `k8v4` | 4.347596 | 4.347258 | −0.008% |
      | `nvfp4` | 4.358924 | **4.352201** | **−0.154%** |

      **All four re-measured formats are now bit-identical between passes** — `fp8` returned
      4.344723843631465 twice over 261,167 scored tokens. That is #49's determinism claim proven at
      the level that matters, not just at the Op-test level.

      The three formats #49 does not touch all drifted −0.009% to −0.019%, which is a harness or
      build offset against whenever the published numbers were taken, and is the floor on any claim
      from this comparison. `fp8` moved three times that and `nvfp4` eight to seventeen times it:
      real. `k8v4` at −0.008% sits *inside* the control band, so it did not measurably improve
      despite being one of the patched kernels.

      README, `docs/config-calculator.html` and `docs/rtx-3090-windows.md` all carry the new
      figures.

- [x] **fp8 and k8v4 are dominated on all three axes and nothing says so structurally.** Settled
      2026-09-09, and the premise was half wrong once the numbers were redone.

      **`fp8` is no longer dominated on all three axes.** Its re-measured perplexity, 4.344724,
      *beats* `rk8v4`'s 4.346413 — a reproducible 0.039%, and both come from the same run so the
      control drift cancels. It is still larger (33,024 B/token against 26,112) and still slower at
      depth (30.34 against 33.17 tok/s at 32K), so what it now offers is a genuine trade: 26% more
      KV memory and 8.5% of decode speed for 0.039% better quality. A bad trade for almost anyone,
      but a trade rather than a strict loss, and README says exactly that instead of "no niche".

      **`k8v4` is dominated, and comfortably.** 1.5% smaller than `rk8v4`, for 13% less decode
      speed at depth, the worst falloff of any format on the 35B (−25.9%), and slightly worse
      perplexity. No configuration makes that 1.5% worth having.

      Left fully available in `--help` either way: upstream parity is worth more than steering, and
      the README table now states the trade precisely enough that nobody needs steering.

- [x] **The speculative decode sweep measures acceptance, not depth, and cannot be read like the
      non-speculative one.** Cause found and named, 2026-09-09. The entry blamed the corpus for
      being "repetitive on a stretch" and prescribed "a corpus with realistic diversity, or many
      more repetitions". More repetitions would not have helped: the fixture is *structurally*
      unable to measure acceptance.

      `bench/fixtures/bench_corpus.ids` is 65,536 tokens drawn from **682 distinct token ids**,
      with **98.4% of its bigrams repeated**, because it is a curated bank rotated and tiled to
      length. A draft head predicts that perfectly. Swept through `ninfer_bench`, DFlash2 reports
      **exactly 100% acceptance at every draft count from 1 through 12**, and decode climbs
      monotonically from 37.6 to 159.7 tok/s because each round emits k+1 tokens for free. Not
      several rows at 100% — all of them.

      Worse, both the corpus manifest and `tools/bench/make_bench_corpus.py` asserted in writing
      that "repetition fills length only and does not bias throughput". That is correct for
      prefill and plain decode, which are token-count and bandwidth bound, and wrong for every
      speculative measurement. Both are corrected in place, and they were the reason this was read
      as a sampling problem rather than a fixture that cannot answer the question.

      The replacement is `scripts/sweeps/dflash2-draft-tokens-realtext.ps1`: the serving path on
      the model's own generated prose, greedy so each configuration produces identical text and the
      comparison is speed on the same output. That is what produced the DFlash2 result in §2c.
      Anything speculative measured through `ninfer_bench` on the committed corpus should be
      treated as void. The memory columns from such runs remain sound — they are read at load and
      do not depend on content.

---

## 6. Operational — closed

- [x] **The pinned host-KV default is 8 GiB regardless of host RAM.** Closed by #45, 2026-09-09,
      and the framing was wrong: host RAM was never the constraint.

      On Windows/WDDM a pinned host allocation is mapped into the GPU's address space and
      charged against the card. Measured on this 24,576 MiB 3090, allocating N MiB on the device
      and then finding the largest pin that succeeds:

      | device resident | VRAM free | largest pin |
      |---:|---:|---:|
      | 15,360 MiB | 7,972 MiB | 8,192 MiB |
      | 17,408 MiB | 5,924 MiB | 6,656 MiB |
      | 19,456 MiB | 3,876 MiB | 3,840 MiB |
      | 21,504 MiB | 1,828 MiB | 2,816 MiB |
      | 22,528 MiB |   804 MiB | 1,536 MiB |

      Resident-device plus pinned-host lands within a few hundred MiB of the card's capacity
      every time. The failure is `cudaErrorAlreadyMapped`, not out-of-memory, and #25's
      diagnostic read it as "this is system RAM, not VRAM" — exactly backwards, which is what
      sent the investigation the wrong way for an hour.

      **Backing off does not work, and this is the part to remember.** One failed
      `cudaMallocHost` poisons every later one in the process. With 2,852 MiB free, 1,024 MiB
      succeeded twice; then a deliberate 8,192 MiB failure made 1,024, 256 and even **64 MiB**
      fail with the same error, and `cudaGetLastError` did not clear it. A halving retry loop was
      written and abandoned on that evidence. The size is now clamped before the first attempt,
      to free VRAM less 1 GiB and then halved, Windows only.

      **Still owed, and its own entry at the end of this section.**

- [x] **The unpinned downloaders cannot verify anything they fetch.** Closed by #48, 2026-09-09.
      `download-qwen38-27b.{sh,bat}` had in fact already pinned `18dfc887` in their URL — they
      simply verified nothing and resumed onto the final path. They now stage under a
      revision-scoped name and check size and SHA-256, matching `download-qwen36-27b`.
      `flake.nix` had been tracking `main` for the same model, so `nix run` and the shell script
      could fetch different artifacts; it is pinned to match. The local artifact every published
      27B number was measured against hashes to the pinned revision, so the pin also records
      which bytes those numbers describe. One downloader stays unverifiable by design —
      `download-qwen36-35b-v2` tracks upstream `main`, which is the whole point of it — and
      README now says so where people choose.

- [x] **`package-release-rtx4090-early1.ps1` has no Linux counterpart.** Closed by #47. The
      counterpart is written and the `windows_only` exemption list is deleted rather than left
      empty. The same PR made that loop accumulate its misses instead of exiting at the first,
      and added an executable-bit check read from the git index rather than the filesystem —
      which immediately found four scripts committed at 644, including
      `scripts/package-release-v090.sh`, the current release's own Linux packager.

- [x] **Binaries embed their build directory.** Half fixed, and the other half measured as not
      fixable. Closed 2026-09-09.

      `-ffile-prefix-map=${PROJECT_SOURCE_DIR}=.` is now set for C, C++ and (via `-Xcompiler`) the
      host half of CUDA translation units on GCC/Clang, which covers the Linux binaries and their
      ~200 occurrences of `/home/ash/ninfer-rel/src/...`. Verified under real Linux g++ on this
      box: `/tmp/ftest/sub/a.cpp` becomes `./sub/a.cpp` and the absolute prefix leaves `strings`
      entirely.

      **MSVC has no working equivalent and that is measured, not assumed.** It takes `__FILE__`
      from the path as written on the command line and CMake writes absolute ones; the usual
      suggestion, the undocumented `/d1trimfile:`, had *no effect* on `__FILE__` when tested
      against 14.44.35207 with an absolute source path. The Windows binaries keep their ~466
      occurrences. Device-side `__FILE__` from nvcc's own frontend is not covered either -- there
      is no documented flag for it.

- [x] **The shipped `-maxctx` launchers ask for a host-KV pin they cannot have. Closed 2026-09-09 —
      by documenting it, because the behaviour is right and the flag is not.**

      All four launchers pass `--host-kv-mib 8192` and they already agree with each other, so the
      "make them agree" half of this entry was moot. What they do *not* do is agree across
      platforms, and nothing said so. The clamp is `#if defined(_WIN32)` only:

      | launcher | platform | free after startup | pinned host KV |
      |---|---|---:|---:|
      | `run-qwen38-c1-maxctx.sh` | Linux | — | **8,192 MiB**, honoured in full |
      | `run-qwen36-35b-a3b-c1-maxctx.sh` | Linux | — | **8,192 MiB**, honoured in full |
      | `run-qwen38-c1-maxctx.bat` | Windows | 1.59 GiB | **302 MiB** |
      | `run-qwen36-35b-a3b-c1-maxctx.bat` | Windows | 184-344 MiB | **0 — none at all** |

      `clamp_host_kv_reservation_bytes` takes `(free VRAM − 1 GiB) / 2`, so the 35B maxctx profile
      falls entirely under the 1 GiB floor and the flag is a **complete no-op** there. On the 27B it
      delivers 3.7% of what it asks for.

      **The three options this entry offered were all based on a wrong premise, which is why none of
      them was right.** It assumed the pin competes with context — "both deliberately fill the card
      with KV". It does not. The clamp reads `cudaMemGetInfo` *after* the KV cache is allocated
      (`program_impl.h:1030`), so the pin takes from the slack that is left over and **costs no
      context at all**. A realistic figure would be wrong on the next machine, dropping the flag
      changes nothing because the default is also 8192, and `--no-prefix-reuse` would trade away
      something that is currently free.

      So the behaviour needs no change and the flag needs no correction — it is right on Linux and
      harmless on Windows. What was wrong is that a reader had no way to know any of that. Each of
      the four launchers now carries the measured table above and states plainly, on the Windows
      side, not to read "8192" as a description of the machine; and on the Linux side, that the same
      flag really does pin 8 GiB of host RAM there and why the platforms differ from identical
      arguments.

---

## 7. Closed this cycle, and what it taught

Kept because the reasoning is what stops the same investigation being repeated.

| # | item | the useful part |
|---|---|---|
| #17 | release scripts could not cut a release | `--package` configured `NINFER_BUILD_BENCHMARKS=OFF` while every packager requires `bench/ninfer_bench`, so it failed *after* the whole tree had built |
| #18 | greedy finiteness guard | **five** routes, not the one recorded; and span-wide, not terminal-only — a matched column whose logits are all NaN matched by accident, and a terminal-only guard licensed it |
| #19 | `docs/performance.md` provenance | all ten "tested revisions" are upstream commits; every hardware line says RTX 5090 except the vision section, so "every figure here is measured on sm_86" was false |
| #20 | BF16 gross-error floor | the bound and the error were the same quantity — kernels are accurate to ~1 rounding step, so a bound of that size measures BF16, not the kernel. `2.0 * kBf16UnitRoundoff` was already the house convention in five files |
| #22 | `w8_pair` k=2048, **up to 52.8%** | under `NINFER_SM8X_COMPAT` all twelve `DualSplitKMedium` schedules are **one kernel**, so eight routes could never win. A live table whose *distinctions* are dead |
| #23 | unrouted schedules visible | pins the set **by name**, not by count — a change stranding one schedule while un-stranding another keeps the count identical |
| #24 | shipped launchers | one shipped `HOST=0.0.0.0` (unauthenticated, LAN-wide), one hardcoded an absolute path from this machine, one could never find its own server binary in the release layout |
| #25 | pinned-host diagnostics | `cudaMallocHost` failing says "out of memory" and points entirely at the GPU; it is **system RAM** |
| #26 | 503 during startup | `bind()` before the Engine is deliberate (fast port-clash failure) but left a 10 s window accepting TCP with nothing answering — a `tcpSocket` probe called that ready |
| #27 | q4 SwiGLU Materialized, **up to 23%** | upstream alternated Materialized/c128 three times across one contiguous range; the winner does not flip back and forth, and it did not |
| #28 | q4_q5 "never wins" | the old claim covered T≤208 only; extended to 4096 it holds, and `pair_c64` is a *slower twin* of `mixed_r32_c64_s3`, never ahead |
| #29 | repo housekeeping | worktrees removed, dead files deleted, `repro/` ignored rather than binned, PR #12 closed as the record |
| #30 | DFlash2 attention sweep | the fixture sized the cache table by the **batch**, but `table_rows` are indices *into* the table; B=1 addressing row 7 indexed a one-element vector, unchecked |
| — | §7 `prompt_i8` dedupe | already landed with the small-T adoption; the entry was simply stale |
| #37 | master did not compile | a lambda introduced in #30's review commit could not see `order`; nothing had rebuilt that TU, so every "125/125" since was measured against a binary the tree could no longer produce |
| #44 | the `T=112` graph-replay failure | not a kernel. `cudaMemcpy` out of pageable memory returns before the DMA lands, and the DMA rides the legacy stream that `cudaStreamNonBlocking` is exempt from |
| #45 | the pinned host-KV default | on WDDM a pinned host allocation is charged against **VRAM**; #25's diagnostic asserted the opposite. And a failed `cudaMallocHost` poisons every later one, so back-off is impossible |
| #46 | four real-model tests died as `0xc0000409` | no top-level catch, so `what()` never printed. `e06d7363` in a debugger is a C++ throw, not corruption |
| #47 | the Linux guard did not guard | an exemption list, a loop that exited at its first complaint, and an `-x` check that cannot see a mode-644 file on Windows or WSL — which had let the current release's own Linux packager sit at 644 |
| #48 | the 27B downloaders | already pinned, verified nothing; `flake.nix` disagreed with the shell scripts about which revision to fetch |
| #49 | fp8/nvfp4/k8v4 non-determinism | a missing `__syncthreads()` between `cp_wait<0>()` and a whole-tile shared-memory read. Corrupted every prefill output column but the last, which is why generation looked perfect |
| #52 | `SmallTMaximumSplits` | section 2c proposed extending the bump to nvfp4 and k8v4; measured, it made them 2.3-3.1% slower, and removing it from fp8 too gained 0.4-1.2%. The host was right and the device policy was wrong |
| #53 | where the MoE's bandwidth goes | the expert gather runs at 40-45% of achievable while contiguous weight kernels on the same step reach 78-82% |
| #50, #51 | the decode roofline | both terms were wrong: the ceiling is measured at 854 GB/s not 936, and the numerator is the read set not the resident set. Gave the MoE its first denominator |
| — | `--vision-residency overlay` + DFlash2 | **was never blocked** — it runs on this one 3090 and always could have. See below |

### `--vision-residency overlay` + DFlash2 — verified working, 2026-09-08

Listed for weeks as needing hardware. It does not. On this single 3090, with
`qwen3_8_27b_dflash2.ninfer`, `--vision --vision-residency overlay --spec dflash2 --draft-tokens 7`:

- starts cleanly — 18.0 GiB of weights, runtime 1.03 GiB, **2.30 GiB still free**, ready in 9.0 s;
- answers four *different* images in sequence (2.4–2.7 s each), describing each correctly and
  reading its embedded label with the index incrementing 00 → 01 → 02 → 03;
- the server is still healthy afterwards and the log carries no `ERROR`, `FATAL` or eviction line.

The worry in the old entry — overlay borrows device memory per image from the evictable
text-weight tail while DFlash2 holds its own weight bundle, so the eviction ladder is untested —
is exactly what the four-image sequence exercises, because the borrow-and-release has to happen
more than once. It holds.

**The lesson is about the list, not the feature**: "never run" had drifted into "cannot be run".
Try it before writing it off; this took one command.

**A correlation is not a mechanism, and this file said so twice before it mattered.** §2c noted
that the three formats with the worst decode falloff were exactly the three that were not
run-to-run deterministic, and wisely added "treat §5 as open on its own terms". #49 fixed the
non-determinism completely and the falloff did not move at all. Two real defects sharing a
population is not one defect.

**Check that your instrument still fires on a case you know is broken.** Two probes were built
for the `T=112` race and both were useless in opposite directions. A D2H read-back on the same
stream, ordered ahead of the kernel, made the race vanish entirely — the copy engine serialises
the in-flight H2D behind it. An early draft of the regression test cleared its counter with
`DeviceBuffer::fill` between the copy and the stream work, and that one synchronous runtime call
in the gap took 84 races out of 90 down to zero. Both looked like clean results.

**The bug you can see is the one that does not matter.** fp8 attention was corrupting every
prefill output column except the last, by up to 26 in logprob, for as long as anyone has been
measuring perplexity with it — and greedy generation stayed byte-identical throughout, because
greedy reads only the last column. `27b_score_real` was the only thing in the tree that looked,
and it had been skipping for want of an artifact that was sitting on the disk.

**"Resident" and "read" are different numbers and only one of them is a denominator.** Dividing
throughput into `weights_capacity_bytes` overstated the dense path by six points and returned
418% of peak on the MoE. The MoE figure had been in this file for a cycle, correctly labelled as
a non-result, and the fix was arithmetic over the artifact rather than any measurement.

Two from earlier cycles, still true:

**Fix the class, not the flagged line.** #18 was reported as one route and was five. #24 was
reported as one launcher and was four problems across six. Chasing the class is also what found the
`kv_cache_append` contract lines and the `AGENTS.md` wrong-target claim that nobody had flagged.

**A route table can be live and still wrong.** `q4_q5` had a tuned table nobody read beside a
hardcoded chain. `w8_pair` k=2048 had a table that *was* read, but whose distinctions did not exist
on this hardware. So `grep resolve_plan` is necessary and not sufficient — also grep the launchers
for `NINFER_SM8X_COMPAT`, and confirm in the sweep, where identical schedules print *identical*
times.

---

---

## This card is power-capped, which sets the floor on every measurement here

`nvidia-smi` reports the power limit as **315 W against a 350 W default** (400 W maximum), and
reports throttle reason `0x4` — `SwPowerCap` — continuously through every sweep. The SM clock
swings **1,665–1,755 MHz** as a result while the memory clock stays fixed at 9,501 MHz. It looks
deliberate rather than accidental, so it has been left alone; changing it needs an elevated shell
anyway.

**This is why within-run stddev lies.** `ninfer_bench -r 3` reports ±0.02–0.5 tok/s, which looks
like a tight measurement. The spread *between processes* on the 27B is 3–5%: int8 at a
4,096-token depth measured 37.44 tok/s in one run and 35.45 in another, on identical code, an hour
apart. The clock drifts with temperature across runs and no amount of repetition inside one
process sees it.

Three consequences worth internalising before quoting any number in this file:

- **Every performance comparison needs a control** — a configuration whose code path the change
  does not touch, measured in the same run. The `SmallTMaximumSplits` work (#52) is the worked
  example: the 35B's fp8 control held to 0.3% and made a 2.5% effect readable, while the 27B's
  control moved ±3% and made that model's numbers worthless. Without the control, six numbers
  looked like a result and three of them were noise.
- **Do not compare across sessions.** Interleave the variants you are comparing inside one sitting,
  or accept a 5% floor.
- **The 35B is the quieter instrument.** Its controls repeatedly held to ≤0.3% where the 27B's
  moved 3–5%, so a small effect should be measured there first.

Also worth someone's attention: 315 W is 90% of this card's default TDP. Decode is memory-bound and
the memory clock is not throttling, so the cost may be small — but it has never been measured.
`nvidia-smi -pl 350` and a re-run of `kv-decode-vs-depth.ps1` would answer it, and
`nvidia-smi -lgc <clock>` would collapse the 3–5% spread for measurement runs. Both need elevation.

**Both are also persistent changes to the machine, so read the next section first.** The 315 W cap
is this host's own setting, not a default and not ours to keep: if you raise it to measure, restore
it to **315** in a `finally`, and never leave a session having changed it.

### Leave this card as you found it

Every `nvidia-smi` write is a change to hardware someone else is using, and it survives the process
that made it. **This file recommended an unpaired `nvidia-smi -lgc 1500` in seven places and never
once mentioned `-rgc`**, which is how a measurement convenience becomes a machine left clocked at
1500 MHz. Two rules, and they are not negotiable:

- **Pair every write with its restore, in a `finally` or `trap`, so an error or a Ctrl-C still
  restores it.** `-lgc <n>` pairs with `-rgc`. `-pl <n>` pairs with `-pl 315` on this box — check
  `nvidia-smi --query-gpu=power.limit --format=csv` first and restore what you actually found, not
  what this file says.
- **Verify afterwards, because a failed restore is silent — and verify it the right way, because
  `-lgc` is invisible to every `nvidia-smi` report field on this driver.** Checked 2026-09-10 with
  a lock actually held: `nvidia-smi -q -d PERFORMANCE` still said
  `Applications Clocks Setting : Not Active`, no other Clocks Event Reason flipped, and
  `--query-gpu=clocks.applications.graphics` answered *"Requested functionality has been
  deprecated"*. That field tracks `-ac`-style **applications** clocks, not a `-lgc` **lock**, so
  reading it as an all-clear is how a stale lock goes unnoticed for days.

  The only signature is behavioural: **an idle GPU with locked clocks sits at the locked floor
  instead of dropping to idle.**

      nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader

  A few seconds after the last kernel, a clean RTX 3090 reads about **210 MHz** against a 2100 MHz
  maximum. If it sits pinned near the value you locked (1395–1500 MHz), the lock is still on — run
  `nvidia-smi -rgc`. For the power limit, `--query-gpu=power.limit` *is* reliable: compare it to
  what you recorded before you started.

`scripts/sweeps/admin-profile.ps1` is the worked example: it locks, profiles, and restores in a
`finally`, and it takes `-SkipPower` so a profiling run never has to touch the power limit at all.
Prefer `-SkipPower` — the power question above is interesting, but it is not worth a persistent
change on someone's workstation to answer as a side effect of something else.

---

## Measuring memory and capacity, because two of these were dead ends

**`ninfer_bench` cannot tell you the automatic-sizing context.** Running it with `-n 1` and no
`-p` and letting `--kv-capacity` default resolves `max_context` to **3**: auto-sizing follows the
*workload*, not the card. It looks like a plausible answer in the CSV and is nothing of the sort.
Use the serving path instead — `apps/ninfer --prompt hi --max-new 1 --kv-capacity auto` prints
`KV capacity` in its summary — and note the answer moves with free VRAM, so it is a property of
the box at that moment, not of the format.

**Never derive a per-token cost from a single context length.** `sequence_capacity_bytes` minus
`kv_payload_bytes` divided by one context looks exactly like 4,063.5 B/token on the 27B. Measured
at 8K/16K/32K/64K it resolves to `166,438,656 + 0.0625 × ctx` — a *fixed* 158.7 MiB block plus a
sixteenth of a byte per token. The single-point reading understates memory by 127 MiB at 8K and
overstates it by 349 MiB at 131K. Two points distinguish the shapes; four confirm it.

**`sequence_capacity_bytes` contains `kv_payload_bytes`.** They are not additive. Summing the two
columns double-counts the whole cache.

**The engine will check your arithmetic for you.** Ask for a context that does not fit and the
refusal names the exact requirement: `minimum Engine runtime reservation requires 9197389568 bytes
in addition to 1073741824 bytes of automatic headroom`. That figure is
`kv_per_token × ctx + fixed sequence block + 0.0625 × ctx + workspace + graph allowance`, to the
byte, and `--kv-capacity auto` holds back a further 1 GiB on top. Cheaper than any probe, and it
is the number the runtime actually applies.

---

## Guards that exit early can silently disable everything after them

`scripts/check-linux-scripts.sh` had been failing on master for some time, at a counterpart rule
near the top, so none of its downloader coverage below had been running — which is how the curl
fixture in it went stale unnoticed. A check that exits non-zero *is* visible in CI, but only as
"the check failed", and the failing line was unrelated to the part that mattered. When a guard
script grows sections, either make it accumulate failures and report them all at the end, or be
suspicious when the first failure is something cosmetic: everything downstream is then untested,
not passing. Same shape as the dead route table in the note above — the thing that looked live
was not running.

---

## Editing this file with a script, because I deleted two sections doing it

`s.index('

---', from)` looked like a safe way to find the end of an entry. There is no `---`
between §5 and §6, so it matched the one after §7 and silently removed both sections. It shipped in
#65 and was only noticed when a later edit could not find `## 6. Operational`; recovering it meant
`git show 17423c11:TODO.md`.

If you script an edit to this file, anchor the *end* of a replacement on the next thing you can
name — a heading, or the next `- [ ]` — never on a separator, and check `grep -c '^## '` before and
after. The same applies to `cmake --build --target A B`, which builds only `A` here and produced a
"verified" test result measured against a stale binary in the same session.

---

## Build notes, because they cost hours

**Never run two builds against `build-ninja` at once.** Concurrent ninja produces "Permission
denied" on object files, `cmake --build` reporting success with the executable never relinked, and
phantom hangs. **Always compare the test binary's mtime against the sources before believing a test
result.** A build that looks hung is usually just slow: `nvcc` idles at ~0.1 s CPU while its `cicc`
child works, and `small_t_fp8.cu` legitimately burns 170+ s in `cicc`. Check `cicc`, not `nvcc`.

**A running test binary breaks the next link**, with the same `LNK1104: cannot open file` signature
as concurrent ninja but a different cause. This bit three times in one session, twice from a
background `ctest` still holding a binary while the next build started — including once from a
background job that session had started itself and then rebased underneath. Two habits:

- Before building: `Get-Process ninja,cmake,cicc,ninfer_*` — the `ninfer_*` half is the one people
  forget.
- **Never leave a background build or ctest running while switching branches or rebasing.** Two
  tests "failed" that way and hung as processes; they passed in isolation and the suite was clean.

**Check the build's exit code separately from the test's.** The stale-binary trap is silent: the
build fails, the test runs the *previous* binary, and reports a pass. Every "verified" claim in this
file was checked that way after being caught out by it.

**And delete the log before every build, because the trap has a second form that survives the
first check.** If you write the exit code into a log — `cmake --build ... > build.log 2>&1` then
`echo BUILD_EXIT=%ERRORLEVEL% >> build.log` — and the build is *killed* rather than failed, the
line is never appended and the previous run's `BUILD_EXIT=0` is still sitting in the file. Checking
the exit code then reads a success that belongs to a different build. This cost three separate
false "verified" readings in one session before it was spotted. `rm -f build.log` first, every
time, and treat a *missing* `BUILD_EXIT` line as a failure rather than as no news.

**Never truncate a build pipeline.** `cmake --build ... | Select-Object -First N` returns while
`ninja` keeps running detached, and the next build collides with it. Redirect the whole build to a
file and grep the file.

**Buffered stdout lies about where a crash happened.** The last flushed line is not the last line
executed. Add explicit `<< std::flush` markers around candidate regions. `cdb` does not capture the
debuggee's **stderr** — put diagnostics on stdout.

## Debugging note

A scriptable console debugger is installed: `%LOCALAPPDATA%\Microsoft\WindowsApps\cdbX64.exe` (from
the Store WinDbg package — there is no plain `cdb.exe`, and the Windows Kits `Debuggers\x64`
directory holds only DLLs). `compute-sanitizer` sees device memory only, so `ERROR SUMMARY: 0
errors` on a process that still dies is positive evidence of a **host-side** fault — switch to cdb
at that point. The DFlash2 sweep (#30) is the worked example: zero device errors, and the fault was
a `std::vector` index out of range in the host reference. See `windows-cdb-debugger-available` in
memory for the invocations.
