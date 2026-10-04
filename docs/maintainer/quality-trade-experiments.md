# Quality trades: narrower vocabulary matrices, FP16 GDN state, integer-activation MLP decode

Seven speed- or memory-for-quality trades, all **implemented, measured, and wired to CLI flags**:
`--lm-head-q4`, `--lm-head-q6`, `--embedding-q4`, `--embedding-q6`, `--mtp-experts-q4`,
`--gdn-state-fp16` and `--mlp-a8-decode` on
`ninfer`, `ninfer-serve`, and `ninfer-perplexity`. All default off. The two new vocabulary trades
are memory trades first: see [Vocabulary transcoding](#vocabulary-transcoding---lm-head-q6-and---embedding-q4)
for why `--embedding-q4 --lm-head-q6` is worth about 37K tokens of context for no measurable
perplexity change. The [`sm_86` findings](../performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode)
explain why the first two were the next levers: a decode round is bandwidth-bound at C1, and both
of them buy bytes.

C8 was believed to be tensor-rate bound when `--mlp-a8-decode` was built, which is what motivated
it. Counters since say otherwise -- tensor, L1 and DRAM all sit near 38% at 31% occupancy, so the
round is latency-bound and operand movement, not MMA rate, is the limit. That is why the int8 route
returned 4-6% rather than the 4.7x its instruction rate suggests; the correction and its numbers are
in [performance.md](../performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode).

**Verdict, ahead of the detail below:** `--gdn-state-fp16` is a clean win — free within measurement
noise on quality, a modest real C8 speedup, and it halves the host state image. Keep it enabled
whenever a serving profile can spare the output-fidelity risk (which measures as none so far).
`--lm-head-q4` is a narrower win — a real ~3% C8 gain, no measurable C1 gain on real text, at a
quality cost (+0.69% perplexity, visibly different greedy output within the first ~50 tokens) larger
than every KV format in [the KV table](../performance.md#choosing-a-kv-format-rtx-3090-qwen38-27b).
It is kept as an opt-in flag because the evidence supports it winning on C8, but it should not be
recommended as a default-on suggestion the way `--gdn-state-fp16` could be.

The reference stack this fork is chased against (syv-ai/qwen38-27b-rtx3090) runs both — it lists a
"calibrated int4 lm_head" and "fp16 state" in its optimized configuration.

## What each one does

### `--lm-head-q4` — int4 vocabulary head

The 27B's output head is `W8G32_F16S`, 248320 x 5120: **1.27 GB read for every decode step and
every verify round**, about 1.5 ms of a 25.7 ms C1 round and 2.7 ms of a 58 ms C8 round.
The artifact stays W8; the weights arena reserves the `Q4G64_F16S` encoding and the materializer
writes it (`DeviceTranscode::W8G32ToQ4G64`, see [Vocabulary transcoding](#vocabulary-transcoding---lm-head-q6-and---embedding-q4)),
halving that read **and freeing 644 MiB of VRAM for KV**.

- Each 64-k group takes the fp16 scale, out of 25 clipping ratios of `absmax / 7` in [0.70, 1.18],
  that minimises the group's squared reconstruction error. Plain `absmax / 7` is one candidate, so
  the result is never worse than round-to-nearest.
- `q4_dispatch` gained `n = 248320` routes, and `launch_q4_small_t_rows` serves any K = 5120 matrix
  whose rows are a whole number of CTAs (rows travel in the store, not the geometry).

Until 2026-09-14 this flag requantized the head in place on the GPU after upload and left the W8
planes' upper halves allocated, so it freed nothing, and it was silently skipped with overlay
vision because the eviction mirror held W8 bytes. Transcoding in the materializer fixes both: the
mirror is captured from the final bytes. It is **rejected** with DFlash/DFlash2, whose
`linear_topk` reads the W8 head directly, and on the 35B-A3B, whose artifact already stores a Q6
head.

### `--gdn-state-fp16` — FP16 recurrent state

The GDN recurrent state is 48 layers x 48 heads x 128 x 128 FP32 = **147 MiB per slot**, read and
written every round. At C8 that is ~3.6 GB of a 58 ms round; it is also the size of the host state
image that `--host-state-slots` pins.

The recurrence still computes in FP32 — FP16 only rounds what is *stored* between steps.
`LinearAttentionStatePoolSpec::recurrent_dtype` carries the choice; the four recurrent kernels
(direct, batch update, record, replay fold) are templated on the storage type and their launchers
dispatch on the tensor's dtype; the chunked prefill kernels keep their FP32 interface and the
wrapper stages FP16 through an FP32 workspace around them.

## Measured 2026-09-12, this box

Windows, RTX 3090, `sm_86`, 315 W cap, CUDA 12.8, int8 KV, Qwen3.8-27B, MTP3 with the optimized
draft head unless noted. This box's ceilings are **854.2 GB/s** and **67.6 TFLOPS BF16** — lower
than the rented Linux 3090 the small-T kernel numbers in `docs/performance.md` used (892.8 GB/s,
81.8 TFLOPS at 350 W), so absolute tok/s here is not comparable to that doc; the ratios below are.

Four arms off one build — base, `q4head`, `fp16state`, `both` — toggled by CLI flag (measured before
they were wired, via the env vars this doc used to describe; behaviour is identical either way,
since the flag only changes how `StartupFeatures` gets set).

**Perplexity** (`ninfer-perplexity --quick`, `ninfer-ppl-1m-v1`, context/stride 4096/2048, 261,167
tokens):

| arm | overall PPL | vs base |
|---|---:|---:|
| base | 4.342425 | — |
| `q4head` | 4.372320 | **+0.688%** |
| `fp16state` | 4.342315 | −0.003% (noise) |
| `both` | 4.372435 | +0.691% |

`q4head`'s cost is real and larger than every KV format in the KV table (nvfp4 KV is +0.36%, the
largest there). `fp16state`'s perplexity is unchanged, as expected — see the caveat below on why
that alone doesn't clear it.

**GDN state drift** (`ninfer_gdn_state_fp16_test`, synthetic 4096-token decode, FP16 vs FP32 state
through the real Op): width 1 (decode) worst relative error **0.339%**, quarterly means flat at
~0.31-0.32%; width 4 (MTP3) worst **0.219%**, quarterly means flat at ~0.19-0.21%. Both comfortably
under the 2% fail threshold and not trending upward — this is the evidence perplexity can't give.

**Q4 head requantization** (then `ninfer_q4_requantize_test`, now `ninfer_vocabulary_transcode_test`): per-group error never worse than plain
`absmax/7` rounding, and the routed Q4 head matches an fp64 oracle at T=1..32. **OK.**

**Greedy output divergence** (one real prompt, 300 tokens, greedy, int8 KV, through `ninfer`):
`base` and `fp16state` are **byte-identical**. `q4head` first diverges from `base` at character
~210 — an early wording choice ("the cache evicts the least recently used item" vs "the cache
maintains a fixed maximum size and evicts..."), consistent with the perplexity cost showing up
immediately rather than only at long range. `q4head` and `both` differ from each other by two
characters over 300 tokens (a near-tied argmax flip on top of `q4head`'s already-altered
distribution) — `fp16state`'s effect is not perfectly inert once stacked on a coarser head, even
though it's inert alone.

**`ninfer_bench`, synthetic corpus (`bench/fixtures/bench_corpus.ids`), `-r 3`:** plain (T=1, no
speculation) decode tok/s moved base 44.65→45.16 (2 interleaved reps) vs `q4head` 45.27→46.47 —
consistently faster by +1.4%/+2.9%, close to the naive bandwidth-share prediction. **MTP3 decode
told a dramatically different story that turned out to be a measurement artifact**: base 71.05 tok/s
at 31.1% acceptance vs `q4head` **100.19 tok/s at 53.8% acceptance** (+41%). This is the exact
pitfall TODO.md already documents for this fixture (98.4% repeated bigrams, DFlash reports 100%
acceptance on it at every draft count) — quantizing the verification head apparently makes it agree
with the draft head *more* often on this repetitive text, which is a statement about the fixture,
not the model. **Do not quote the 41% figure for anything.** `fp16state` on the same fixture: 72.40
tok/s / 30.2% acceptance, i.e. flat.

**`tools/bench/run_chat_decode.py`, real prompts (syv-ai's eight `bench/prompts_real.jsonl`,
thinking-off, greedy, 256 max tokens, 2 interleaved reps) — this is the authoritative speed number,
since it doesn't share the synthetic corpus's repetition:**

| C | arm | decode tok/s (rep0, rep1) | mean | vs base | accept% |
|---|---|---|---:|---:|---:|
| 1 | base | 110.33, 105.70 | 108.02 | — | 62.9% |
| 1 | `q4head` | 110.13, 106.19 | 108.16 | +0.1% (noise) | 60.9% |
| 1 | `fp16state` | 107.16, 103.89 | 105.53 | −2.3% (noise) | 60.3% |
| 1 | `both` | 110.18, 108.19 | 109.19 | +1.1% (noise) | 60.6% |
| 8 | base | 420.09, 425.35 | 422.72 | — | 61.4% |
| 8 | `q4head` | 435.99, 436.36 | 436.18 | **+3.2%** | 61.1% |
| 8 | `fp16state` | 437.50, 425.06 | 431.28 | **+2.0%** | 61.3% |
| 8 | `both` | 451.57, 447.69 | 449.63 | **+6.4%** | 61.8% |

C1's rep0→rep1 drop (all four arms slower by 3-5% in rep1) is the same between-process spread
TODO.md already names; none of the C1 deltas clear that noise floor. C8's gains do — acceptance
stays in a tight 60.6-61.8% band across all four arms there, confirming the MTP3-bench-corpus jump
above was fixture-specific, not a real acceptance effect from either trade.

**Why perplexity alone doesn't clear `fp16state`.** It scores through prefill in 1024-token chunks,
so the state is only rounded at three chunk boundaries per 4096-token window, where decode rounds it
every round. The drift test and the divergence check exist because of this gap, and both came back
clean.

## Vocabulary transcoding -- `--lm-head-q6` and `--embedding-q4`

Qwen3.8-27B stores both vocabulary matrices as `W8G32_F16S`, 248320 x 5120: **1,288 MiB each**,
together 2.5 GiB of a 17.9 GB resident model. Neither needs eight bits.

> Naming: the v3 container and this code now spell that coding `q8_g32_fp16` / `QType::Q8_G32_FP16`,
> and Q4G64/Q6G64 are `Q4_G64_FP16`/`Q6_G64_FP16`. The measurements below predate the rename and use
> the old spellings throughout; the stored bytes and the codecs are the same.

`src/artifact/transcode.{h,cpp}` requantizes a row-split W8G32 payload into a narrower group-64
row-split encoding on the host, while the weights load. The binder plans the target encoding's size,
so the arena genuinely shrinks; the materializer writes the target bytes after the direct-I/O
upload, before anything else (bindings, the overlay eviction mirror) can observe the tensor. Scale
selection is the `--lm-head-q4` search above with `qmax` of 7 or 31. It is multi-threaded and adds
a few seconds to load.

| flag | tensor | stored as | device memory | kernel |
|---|---|---|---:|---|
| `--lm-head-q4` | output head | Q4G64 | -644 MiB | Q4 linear (existing) |
| `--lm-head-q6` | output head | Q6G64 | -341 MiB | Q6 linear (existing, same routes as the 35B head) |
| `--embedding-q4` | token embedding | Q4G64 | -644 MiB (27B), -258 MiB (35B-A3B) | Q4 embedding gather (new) |
| `--embedding-q6` | token embedding | Q6G64 | -341 MiB (27B), -143 MiB (35B-A3B) | Q6 embedding gather (existing; `[248320,2048]` now qualified) |

`--lm-head-q4` and `--lm-head-q6` are mutually exclusive, as are `--embedding-q4` and `--embedding-q6`. Where
the artifact already stores the requested Q6 format (Qwen3.6-27B groupwise ships a Q6 embedding and
head), the Q6 flag is a no-op.

**Measured 2026-09-14**, same box and corpus protocol as above (`--quick`, 4096/2048), but **rk8v4 KV**,
Qwen3.8-27B:

| flags | overall PPL | vs base | free after weights | context gained |
|---|---:|---:|---:|---:|
| base | 4.346413 | -- | 6,563,576,832 B | -- |
| `--embedding-q4` | 4.343738 | -0.062% | 7,239,007,232 B | +24.3K tokens |
| `--lm-head-q6` | 4.346931 | +0.012% | 6,921,157,632 B | +12.9K tokens |
| `--embedding-q4 --lm-head-q6` | 4.344221 | **-0.050%** | 7,596,588,032 B | **+37.2K tokens** |
| `--lm-head-q4` | 4.376552 | +0.693% | 7,239,007,232 B | +24.3K tokens |

Context gained is the freed bytes over this configuration's 27,744 B per token of rk8v4 KV with
MTP (MTP3, draft head, one lane); on this box that moves the largest single-lane context from about
182K to 220K tokens. Free-after-weights is the engine's own startup arithmetic, and the three
rows that should agree do so to the byte (both transcodes together free exactly the sum of each).

Both perplexity differences on the embedding are inside run-to-run noise for this protocol; the
embedding quantization simply costs nothing measurable, and six bits on the head costs +0.01%.
Before building the transcoder these were measured by **fake quantization** -- writing Q4/Q6 codes
and scales into the W8 layout, which represents them exactly, and scoring through the unchanged W8
kernels -- and the real transcoded runs reproduce those numbers to every printed digit, which is also
the end-to-end check that the codec and the new Q4 gather are exact.

**Decode speed** (`tools/bench/run_chat_decode.py`, eight chat prompts, thinking off, greedy, 512 output
tokens, MTP3 with the draft head, rk8v4, arms interleaved in one sitting, three repetitions, median):

| C | arm | decode tok/s | vs base |
|---|---|---:|---:|
| 1 | base | 102.01 | -- |
| 1 | `--embedding-q4` | 105.55 | +3.5% (noise; never slower in a pair) |
| 1 | `--lm-head-q6` | 99.78 | **-2.2%** (-5.4%, -5.1%, -2.0% paired) |
| 1 | `--lm-head-q4` | 109.36 | +7.2% |
| 1 | `--embedding-q4 --lm-head-q6` (separate sitting) | 96.78 vs 101.95 | -5.1% |
| 4 | `--embedding-q4 --lm-head-q6` (separate sitting) | 300.44 vs 307.03 | -1.0% (mixed signs) |

The embedding costs nothing: its gather touches a handful of rows per step. The Q6 head measured
above was slower at one lane because this shape had no Q6 decode kernel -- T <= 7 routed to the
generic 8-row SIMT tiles. Since 2026-10-02 T = 1..2 take a per-row Q6 GEMV
(`src/ops/linear/q6/q6_rowsplit_gemv.cu`) and T = 3..16 a small-T MMA
(`src/ops/linear/q6/q6_small_t_mma.cu`). On the head shape
(`bench/ops/linear_schedule_bench.cu q6:248320x5120`, sm_86) the GEMV runs 1159 us at T=1 against
1381 us for the SIMT tile (857 GB/s), and the small-T MMA holds 1154-1175 us from T=3 to T=8
(T=4: 1651 for the GEMV; T=8: 2450 for the 64-row MMA it replaced, which had made a DFlash2 K=7
verify about 1 ms slower than with the 8-bit head). End to end, an artifact with a 6-bit head (plus 6-bit embedding and 4-bit mixer outputs
in layers 36-63) decodes 2.4% faster than today's 8-bit-head artifact at C1 and 2.1% faster under
MTP3 (`ninfer_bench`, three interleaved rounds, int8 KV). The decode rows in the table above
predate the GEMV and have not been re-run with `--lm-head-q6`.

**The 35B-A3B is different.** Its 2048-wide embedding does not quantize for free: `--embedding-q4`
measured 4.373904 -> 4.389698 (**+0.36%**, same protocol), about the cost of NVFP4 KV, for 258 MiB.
It is supported there as a trade rather than recommended. `--embedding-q6` is the option for that
model: Q6G64 through the existing Q6 gather, **4.373904 -> 4.375419 (+0.035%)** for 143 MiB (about
18K tokens of its rk8v4 KV). The two are mutually exclusive. Its head is already Q6G64 in the
artifact, so `--lm-head-q4`/`--lm-head-q6` are rejected at startup on that model.

**Overlay vision has a floor on how far these tensors may shrink.** Its exclusive fallback borrows
the encode window from the evict-ranked weights (head, embedding, draft head, MTP), so they must
cover one window. On the 27B they are about 3 GiB and nothing here comes close. On the 35B-A3B they
are about a GiB against a 960 MiB window at the default `--vision-max-merged 16384`, so a smaller
embedding there (Q4, or Q6 alongside other savings) no longer covers it. Startup then refuses with
the sizes and the three ways out: a lower `--vision-max-merged`, `--vision-residency resident`, or
dropping a shrinking flag.

**Overlay vision** works with transcoded tensors, because the eviction mirror is captured after
the materializer writes them: with `--vision --vision-residency overlay --embedding-q4 --lm-head-q6`
and KV too small to fund the encode window, the image request opened an exclusive window that evicted
144 MiB of the Q6 head and restored it from the mirror, and a greedy text completion before and
after the image was byte-identical.

Qualification: `ninfer_vocabulary_transcode_test` checks the codec against an independent decoder
(every group no worse than round-to-nearest, padded and all-zero groups included) and routes
transcoded 248320x5120 heads through `ops::linear` at T = 1..32 under the A16 linear criterion;
`ninfer_embedding_test` qualifies the Q4 gather at [248320,5120] and [248320,2048] against an FP64
oracle, including CUDA Graph replay.

## `--mtp-experts-q4` -- the 35B-A3B's MTP experts in the text layers' formats

The Qwen3.6-35B-A3B artifact stores its MTP draft layer's routed experts as W8G32 -- `routed_gate_up`
262144x2048 and `routed_down` 524288x512, about 816 MiB of the 856 MiB MTP head -- while every text
layer stores its routed experts as Q4G64 gate_up with a Q5 or Q6 down. `--mtp-experts-q4` transcodes
the draft layer's pair at load (the same `artifact/transcode` path as the vocabulary flags) to Q4G64
gate_up and Q6G64 down, a combination every `sparse_moe` route already serves for text layers 34, 38
and 39. The MTP workspace is sized for both pairs. The 27B's MTP layer is dense, so the flag is
rejected there.

The draft layer only proposes tokens, and the target model verifies every one, so this cannot change
what the model scores or which distribution it samples from; perplexity does not run MTP at all. What
it can change is acceptance, and -- as with any change in acceptance pattern -- which verify widths
run, so greedy text can differ at the reduction-order level this file's speculative-decoding entry
already describes.

**Measured 2026-09-14**, `run.bat qwen36-35b-a3b`'s exact server profile (C1, rk8v4, MTP3 +
draft head, overlay vision, 32 host state slots), arms alternated at each rung, desktop holding a
steady ~505 MiB:

| context | launcher | `--mtp-experts-q4` | `--mtp-experts-q4 --gdn-state-fp16` |
|---:|---|---|---|
| 114,688 | 1.23 GiB runtime / 535 MiB free | 877 MiB free | 1.17 GiB / 947 MiB free |
| 131,072 | 393 MiB free | 745 MiB free | 790 MiB free |
| 147,456 | 254 MiB free | 606 MiB free | 666 MiB free |
| 163,840 | 107 MiB free | 466 MiB free | 526 MiB free |
| 180,224 | refused | 326 MiB free | 386 MiB free |
| 196,608 | refused | 186 MiB free | 246 MiB free |

About 350 MiB is freed, two full rungs of rk8v4 context at 7,969 B per token. Decode (`run_chat_decode.py`,
eight chat prompts, 512 tokens, three interleaved repetitions): **296.29 -> 294.93 tok/s median
(-0.5%)**, acceptance 61.17% -> 60.55%.

`--gdn-state-fp16` on this model saves a further ~60 MiB of device state and halves the pinned host
state image (1.92 GiB -> 1005 MiB at 32 slots).

## `--mlp-a8-decode` -- integer-activation MLP gate_up at decode widths

Added after the other two, and measured differently because perplexity cannot see it. The kernel,
its per-width sweep and the five variants that were measured and rejected live in the header of
`src/ops/linear/q4/q4_small_t_mma_i8.cuh`; the README carries the user-facing table. The short
version:

- The Op is 4-6% faster than the BF16 small-T kernel from sixteen columns up, and slower at eight,
  so the route is admitted for 16..32 columns only.
- End to end at C8 with MTP3 it is **+1.28%** over four interleaved repetitions (431.5 -> 437.0
  tok/s), which is what gate_up's share of a round predicts.
- Quality evidence is the FP64 oracle bound, 0.0080-0.0371 relative L2 across 2..32 columns against
  the 0.04 allowance, plus the fact that output demonstrably changes at C8.

**Why perplexity is silent on it, and what to use instead.** The route is admitted only in the
verify phase, and `CausalScoring` runs the prefill phase, so scoring never reaches it -- the flag is
accepted by `ninfer-perplexity` and changes nothing there.

That phase guard is load-bearing, and the first cut of this did not have it. Width alone is not
enough: `causal_score` sends its remainder through prefill unchanged, so a 1,041-token input scores
as 1,024 + 16 and the 16-column tail would have taken the lossy route in the middle of a
measurement that advertises itself as unaffected. Reviewed and fixed before merge. That is a genuine gap in the evidence
rather than a clean bill of health: the trade is only exercised by cohort decode, so the honest
checks are the oracle bound above and a greedy-divergence comparison at C8, both of which this
branch has. A stronger number would need a scorer that can run at cohort widths.

**The finding that outlived the kernel.** `docs/performance.md` called C8 tensor-rate bound, which
is why int8 looked like a 4.7x lever. `ncu` says otherwise: the kernel sits at 74% L1/TEX against
42% DRAM and 47% SM, so operand movement through L1 and shared memory is the limit and the int8
MMA's arithmetic advantage is mostly unspendable at this shape. The largest single win in the whole
exercise was not the instruction swap but deleting a redundant copy of the staged activation slab.
That reading applies to the other small-T kernels too -- they share the shape and the staging
pattern -- and is the reason a cp.async ring lost at every width here.

## `--no-prefill-a8` -- the escape hatch for the integer prefill routes

The odd one out: this is the only flag here that turns a trade *off*. The integer-activation
prefill routes are default-on because they are cheap enough to be, and the flag exists so they can
be priced -- both arms of every number in
[Integer activations for every registered prefill projection](../performance.md#integer-activations-for-every-registered-prefill-projection)
were measured by flipping it on the same build and card.

Scoring runs the prefill phase, so unlike `--mlp-a8-decode` perplexity sees these routes directly.
Quick corpus, RTX 3090, Qwen3.8-27B groupwise-int, `--kv-dtype int8`:

| arm | overall perplexity | against A16 |
|---|---:|---:|
| `--no-prefill-a8` (every projection A16) | 4.342982 | � |
| every registered integer route (default) | 4.343155 | **+0.004%** |

That is inside run-to-run noise, and inside the +0.05% this fork requires before a lossy route is
on by default. The supporting evidence is the FP64 oracle bound in
`ninfer_linear_swiglu_q4a8_int_test`: 0.010-0.020 relative L2 across every destination range of
the five registered profiles, against the 0.04 allowance A8 activation compute is held to.

Per-group activation scaling is what keeps the cost this low, and it is a deliberate choice rather
than an accident of the schedule. `tools/w4a8_real_weight_probe.cu` measures a per-token absmax --
what Marlin and the vLLM stacks use -- at 1.2e-2 relative L2 rising to **1.29e-1** as outlier
channels grow, against 9e-3 to 2.0e-2 for a scale per group of 64. It is about 20% faster and it is
not worth it here; a single large channel otherwise starves every other channel in the token.

## Weight encoding and layout -- `grouped_search` and the 2026-10 27B layout

Measured 2026-10-01 and -02 on this box while building the default `qwen3_8_27b` artifact
(`docs/weight-conversion.md` describes the method). Every NInfer row is Qwen3.8-27B with the
encoding exported bit-exactly into Q8_0 GGUF blocks (`tools/eval/gguf_eval.py export`) and scored by
llama.cpp b11316 against Unsloth's Q8_0 on the
[held-out corpus](../perplexity.md#held-out-corpus): 8 chunks of 4,096 tokens in each of six
streams, averaged. *Mean KLD* is the KL divergence from the Q8_0 distribution (lower is better);
*Text weights* are the NInfer bytes of the text layers, embedding and head; *Read per token* is what
a decode step streams (the embedding costs one row, so it is left out). The shipped artifact's file
is 18.98 GB because it also carries vision, MTP, DFlash2 and the proposal head.

**Encoder, on the layout before 2026-10 (Q4/Q5 projections, Q8 embedding and head)**

| Variant | What it is | Text weights (GB) | Read per token (GB) | Mean KLD | Top-1 same | PPL vs Q8_0 |
|---|---|---:|---:|---:|---:|---:|
| `rtn` | round-to-nearest, `grouped_absmax`: the upstream artifact's encoding | 17.04 | 15.69 | 0.0376 | 93.50% | +1.26% |
| `search` | scale search, no importance weighting | 17.04 | 15.69 | 0.0371 | 93.75% | +0.71% |
| `searchneg` | scale search with signed scales | 17.04 | 15.69 | 0.0358 | 93.97% | +0.59% |
| `search-imatrix` | scale search weighted by Unsloth's imatrix | 17.04 | 15.69 | 0.0280 | 94.78% | +0.16% |
| **`searchneg-imatrix`** | `grouped_search`: imatrix and signed scales (the shipped encoder) | **17.04** | **15.69** | **0.0262** | **94.91%** | **+0.06%** |
| `autoround-g64` | the public AutoRound W4A16 g64 checkpoint's codes imported; tensors it lacks use RTN | 17.04 | 15.69 | 0.0351 | 93.97% | +0.39% |
| `hybrid-autoround-public` | the same import; tensors it lacks use `grouped_search` | 17.04 | 15.69 | 0.0297 | 94.58% | -0.44% |

**Layout, all with `grouped_search`**

| Variant | What it is | Text weights (GB) | Read per token (GB) | Mean KLD | Top-1 same | PPL vs Q8_0 |
|---|---|---:|---:|---:|---:|---:|
| `searchneg-imatrix-allq4` | every projection Q4 | 15.63 | 14.27 | 0.0457 | 93.19% | +3.31% |
| `searchneg-imatrix-emb4-head6` | Q4 embedding, Q6 head | 16.01 | 15.33 | 0.0282 | 94.77% | +0.20% |
| `searchneg-imatrix-realloc` | Q4 embedding, Q6 head, attention/GDN outputs Q8 | 16.83 | 16.15 | 0.0265 | 94.88% | +0.13% |
| `alloc-today-v1` | Q4 embedding; attention/GDN outputs Q8 in layers 0-35, Q4 in 36-63 | 16.72 | 16.04 | 0.0282 | 94.81% | +0.35% |
| `ladder1-existing-routes` | Q6 embedding and head, outputs Q8 | 17.14 | 16.15 | 0.0256 | 94.86% | -0.01% |
| `ladder2-mlp5-late` | ladder 1, MLP gate/up/down Q5 in layers 20-63 | 18.12 | 17.13 | 0.0176 | 95.99% | +0.05% |
| `ladder3-down4-early` | ladder 2, MLP down Q4 in layers 0-19 | 17.90 | 16.91 | 0.0190 | 95.91% | +0.21% |
| `ladder4-attn-kv` | ladder 3, attention key Q6 and value Q8 | 17.96 | 16.96 | 0.0168 | 96.11% | +0.37% |
| `c1-existing` | C1: Q6 embedding and head, outputs Q4 in layers 36-63 | 16.22 | 15.22 | 0.0281 | 94.60% | +0.30% |
| `c1-down4late` | C1b: C1 and MLP down Q4 in layers 36-63 | 15.90 | 14.91 | 0.0304 | 94.29% | +0.49% |
| **`release-emb4`** | **shipped**: C1b with a Q4 embedding | **15.59** | **14.91** | **0.0318** | **94.25%** | **+0.59%** |
| `c2-lean` | C2: C1b, attention key Q6 and value Q8, GDN value and z Q4 in layers 36-63 (needs kernel routes that are not built) | 15.79 | 14.80 | 0.0295 | 94.29% | +0.74% |
| `unsloth-allocation` | Unsloth UD-Q4_K_XL's per-tensor allocation in our formats, Q4 embedding | 16.53 | 15.86 | 0.0215 | 95.63% | +0.99% |
| `unsloth-allocation-emb6` | the same with a Q6 embedding | 16.85 | 15.86 | 0.0200 | 95.68% | +0.77% |

**llama.cpp reference**

| Variant | What it is | Text weights (GB) | Read per token (GB) | Mean KLD | Top-1 same | PPL vs Q8_0 |
|---|---|---:|---:|---:|---:|---:|
| `unsloth-ud-q4_k_xl` | Unsloth UD-Q4_K_XL, its own formats (GGUF file size) | 17.56 | n/a | 0.0156 | 96.61% | +0.34% |
| `unsloth-ud-q5_k_xl` | Unsloth UD-Q5_K_XL, its own formats (GGUF file size) | 20.88 | n/a | 0.0095 | 97.76% | +0.13% |

What the rows say:

- **The imatrix is most of the encoder gain**: 0.0376 to 0.0280 for the same bytes, against 0.0371
  from scale search alone. Signed scales take it to 0.0262 with the imatrix (0.0358 without).
- **The public AutoRound codes lose to this encoder on the same tensors** (0.0297 against 0.0262),
  and beat plain RTN only by a little (0.0351 against 0.0376).
- **The shipped layout is smaller and closer to Q8_0 than the old one**: 15.59 GB and 0.0318
  against 17.04 GB and 0.0376, and 14.91 GB read per token against 15.69. Its Q4 embedding costs
  Chinese Wikipedia (KLD 0.0794 to 0.0859) and chat (0.0355 to 0.0373) against a Q6 one, and leaves
  the English, arXiv and code streams within 0.0001; that is the 0.0304 to 0.0318 between `c1-down4late`
  and `release-emb4`.
- **C2 would be better and smaller than C1b** (0.0295 at 14.80 GB read against 0.0304 at 14.91) but
  needs a Q4 GDN value/z pair and an attention key/value upgrade that have no kernel routes.
- **Unsloth's allocation is worth about 45% of its lead and its formats the rest**: their layout in
  our formats scores 0.0215 (16.53 GB), between our 0.0262 and their 0.0156. At similar bytes
  UD-Q4_K_XL (17.56 GB, 0.0156) is still clearly better than anything NInfer's formats reach; the
  closest, `ladder4-attn-kv`, needs 17.96 GB for 0.0168.
- **All-Q4 is not an option**: 0.0457 and +3.31% perplexity for 1.4 GB.
