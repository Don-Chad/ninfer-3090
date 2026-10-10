# Launcher profiles

`scripts/run.sh` and `scripts/run.bat` serve a model on one RTX 3090:

```
run.sh <model> [profile]
```

| model | profiles | default profile on Linux | on Windows |
|---|---|---|---|
| `qwen38-27b` | `tuned`, `int8`, `c8` | DFlash2, one lane, 262,144 tokens | DFlash2, one lane, 188,416 tokens |
| `qwen36-35b-a3b` | `tuned` | MTP3, three lanes, 262,144-token pool, cuBLAS prefill | MTP3, two lanes, 212,992-token pool, cuBLAS prefill |

Lanes share one KV pool: `--kv-capacity` is the pool and `--max-context` the per-request cap, and
the `tuned` profiles set both to the same value. Any one request can use the whole context, but the
lanes' requests together hold at most that many tokens at a time.

`tuned` is the recommended profile. `int8` (one user, 64K of INT8 KV, the quality default) and `c8`
(eight lanes at 8K) are the older reference profiles, with every serving flag fixed. `c8` sizes its
context cache per lane -- one extra device StateImage per lane (`--device-state-slots 8`, 147 MiB
each in BF16) and the same 8 GiB pinned Host context budget as `tuned` -- so rotating agents keep
their conversations cached. Every `tuned` profile and `c8` pass `--host-context-mib 8192`; `int8`
passes no cache flag and gets the engine default (8 GiB plus eight StateImages). The former
vision-only launchers are gone: `tuned` serves vision in overlay residency, which costs about
10 MiB, and `NINFER_VISION=off` turns it off.

The Qwen3.8-27B `tuned` profile has three speculative variants, selected with `NINFER_SPEC`. Each
carries the context, lane count and prefill chunk that fit it:

| `NINFER_SPEC` | flags | context (Linux / Windows) | lanes (Linux / Windows) |
|---|---|---|---|
| `dflash2` (default) | `--spec dflash2 --draft-tokens 7 --lm-head-draft --prefill-cublas --prefill-chunk 4096 --kv-dtype rk4v4 --gdn-state-fp16 --vision --vision-residency overlay` | 262,144 / 188,416 | 1 / 1 |
| `mtp` | `--spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --prefill-chunk 2048 --kv-dtype rk4v4 --gdn-state-fp16 --vision --vision-residency overlay` | 262,144 / 262,144 | 2 / 2 |
| `none` | the `mtp` set without speculation | 262,144 / 262,144 | 2 / 2 |

The default artifact (`download-model qwen38-27b`) stores the token embedding as Q4 and the head as
Q6, so none of the variants pass the `--embedding-q4` or `--lm-head-q6` load-time transcodes. The
memory measurements below were taken with the upstream artifact: for `mtp` and `none` its
transcodes left the same formats on the device, while under `dflash2` its head stayed Q8, so the
DFlash2 figures overstate today's weights by about 0.7 GiB.

The first is the fastest at one stream (prefill about 1.7x and decode about 1.39x the previous
defaults) and the second is the full context with a second lane, still fast. The Qwen3.6-35B-A3B
`tuned` profile runs MTP3 + draft head at 262,144 tokens with three lanes on Linux and two on
Windows.

MTP DRAFT COUNT, 2026-09-28. MTP now accepts `NINFER_DRAFT_TOKENS` up to 15 (it was capped at
five). The `mtp` profiles keep three: it is best or within 2% on prose, while 7, 9 and 15 lose 12%,
27% and 38% on a short story. On output that reproduces the prompt (editing and returning a file)
11 to 15 decode 1.7-1.85x faster than three, and on newly written code seven is about 10% faster;
the table is in [performance](../performance.md#choosing-the-draft-count-rtx-3090-qwen38-27b). Eight and above add a second CUDA
Graph topology class on the 27B, about 64 MiB more reserved per lane, which the 27B `mtp` profile's
~1.2 GiB of headroom covers. Measured single-stream with the CLI; the two-lane server was not
re-measured at larger counts.

RK4V4 DEFAULTS, 2026-09-24. Every `tuned` profile moved from `rk8v4` to `rk4v4` KV (Lloyd-Max 4-bit
keys, 31% smaller than `rk8v4` at the same decode speed, +0.10% perplexity over it; see the
README's `rk4v4` section). Measured by starting `ninfer-serve` with each profile's exact flags on
the desktop RTX 3090 (1.35 GiB held by the desktop):

```
  profile                      lanes  context    result
  --------------------------------------------------------------------------------------------
  27B dflash2 (rk8v4, before)  1      131,072    starts; 150K refused
  27B dflash2                  1      180,224    starts; 184,320 refused (needs ~1.45 GB more
                                                  for 262,144: 6.45 GB against 5.00 available)
  27B mtp                      2      262,144    starts, 23.4 of 24.5 GiB used
  35B mtp                      3      262,144    starts, 24.0 of 24.5 GiB used
  35B mtp                      4      262,144    refused, 48 MB short
```

The 35B Windows default keeps one lane below its edge (two). The 27B DFlash2 Windows default was
172,032 (one rung below the edge above) until the 2026-10-02 artifact, whose Q6 head and Q4 mixer
outputs and MLP down in layers 36-63 hold 0.7 GiB less on the device under DFlash2. Re-measured
that day with the desktop holding 1.6 GiB (more than above), it starts at 188,416 with 721 MiB free
and is refused at 196,608, so the default is 188,416; the upstream artifact under the same
conditions had stepped down to 150,720. The Linux defaults are the headless extrapolation -- the full context for
DFlash2 and three 35B lanes -- and the step-down ladder catches a card that falls short. DFlash2 and
MTP generation on `rk4v4` were checked end to end through the server at these settings.

The sections below record the measurements behind the earlier `rk8v4` defaults. Both are measured in
[performance](../performance.md#recommended-configurations-rtx-3090-qwen38-27b). The
`qwen3_8_27b.ninfer` that `download-model` fetches is the DFlash2 bundle and carries the MTP weights
too, so one file serves all three.

Overrides, from the environment, so a launcher never needs editing: `NINFER_MODEL` (artifact path),
`NINFER_MODEL_DIR`, `NINFER_SERVER`, `NINFER_HOST`, `NINFER_PORT` for every profile; `tuned` also
reads `NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_CAPACITY` (Linux), `NINFER_KV_DTYPE`,
`NINFER_SPEC`, `NINFER_DRAFT_TOKENS`, `NINFER_PREFILL_CHUNK`, `NINFER_VISION`,
`NINFER_VISION_RESIDENCY`, `NINFER_HOST_CONTEXT_MIB` (default 8192), `NINFER_FALLBACK`, `NINFER_MIN_P` (default 0.03) and
`NINFER_PRESENCE_PENALTY` (default 0.5). The last two are the loop guard: they are process-level sampling
overrides for the `tuned` profile only, replacing the registered presets in thinking and non-thinking mode, and
`default` omits the flag so the registered preset stays in force. Loopback is the default host: `0.0.0.0` publishes an unauthenticated
endpoint to every network the machine is on, so it is opt-in per run.

The `tuned` profile also exposes further serving flags, each appended only when its variable is set so the
defaults stay as measured: `NINFER_MAX_OUTPUT_TOKENS`, `NINFER_MLP_A8_DECODE=on`,
`NINFER_CONTEXT_STORE` (+ `NINFER_CONTEXT_STORE_MAX_GIB`) and `NINFER_AUTO_HOST_CACHE=on` (+
`NINFER_HOST_CACHE_PERCENT`, `_RESERVE_MIB`, `_MAX_MIB`). `--auto-host-cache` sizes the Host context budget
itself and refuses `--host-context-mib`, so with it on the launcher drops that flag and
`NINFER_HOST_CONTEXT_MIB` (and the step-down's halving of it) is ignored.

**When the card is busy.** A desktop or another job holding VRAM can leave too little for the default
context, and a machine short of RAM can fail to pin the Host context budget, so there are two ways
to be refused: the engine's runtime reservation, or pinning host memory. The `tuned` profile
therefore steps down when startup is refused for either reason (the launcher looks for the engine's
own `runtime reservation requires` and `cudaMallocHost failed` messages): an eighth of the context at
a time, at most five times, with the prefill chunk capped at 2048 from the second step and
`--host-context-mib` halved every second step (8192, 8192, 4096, 4096, 2048, 2048; floor 0).
`--kv-capacity auto` was tried first and does not help, because the engine still has to reserve room
for one full `--max-context` sequence. Measured on this fork's 3090 with a Windows desktop holding
2.8 GiB, on the earlier engine: `run.bat qwen38-27b` was refused at 131,072, 114,688 and 98,304 and
started at 81,920 (chunk 2048), and the same launcher on Linux under WSL sees the same card. Values
the caller sets (`NINFER_CONTEXT`, `NINFER_PREFILL_CHUNK`, `NINFER_HOST_CONTEXT_MIB`,
`NINFER_KV_CAPACITY`) are never second-guessed, `NINFER_FALLBACK=off` disables the step-down, and the
`int8` and `c8` profiles never step down.

What follows is the measurement history behind the `tuned` defaults, moved here from the launchers'
own headers so the launchers stay readable. Figures marked "MTP profile" predate the cuBLAS prefill
route and `--lm-head-q6` in the launcher; the route's workspace (543 MiB at chunk 2048, 661 MiB at
4096) comes off each headroom figure and `--lm-head-q6` returns 341 MiB.

## Qwen3.8-27B, `tuned`

Qwen3.8-27B on one RTX 3090, tuned for context and prefix reuse rather than for caution.

`run.sh qwen38-27b int8` serves 65,536 tokens of INT8 and leaves 2.85 GiB of the card unused. This
profile spends it: rk8v4 KV, speculation plus the draft head, the cuBLAS prefill route, and the
tuned context cache.

Why the two variants differ: DFlash2's draft weights and its refusal of `--lm-head-q6` cost about 65K tokens of context between them, so the same flags on DFlash2 load at 130K and fail at 150K, while the `mtp` set was verified at 200,000 tokens by loading it. The `mtp` set uses chunk 2048 rather than 4096 because the larger chunk costs about 300 MiB of workspace for its last 4% of prefill, and at that context the workspace is the binding constraint. `+0.156%` perplexity comes from the cuBLAS route and `+0.083%` from rk8v4.

WHAT rk8v4 BUYS. The same KV that holds 171,648 INT8 tokens holds 226,560 rk8v4 tokens - +33%
context for +0.082% perplexity. It is opt-in precisely because INT8 is the quality default.

CONTEXT CACHE. A checkpoint is a KV prefix plus a StateImage, and on this model the StateImage
is 147 MiB flat regardless of prefix length - 48 GDN layers of 128x128x48 FP32 recurrent state
plus conv - or 74.5 MiB with --gdn-state-fp16, which this profile uses, so the 8 GiB Host context
budget (--host-context-mib 8192) holds about twice as many retained StateImages as it would in FP32.
That budget is host memory, not device, and is shared by retained StateImages, KV pages and pause
snapshots. (The earlier engine's catalog, sized with --max-shared-prefixes 8 and 32 host state
slots, took prefix reuse from 8.4% to 98.3% on a multi-preamble workload; that figure has not been
re-measured on the current engine, which has no catalog to size.)

MEMORY FLAGS, both free on quality (docs/maintainer/quality-trade-experiments.md):
```
  --embedding-q4    token embedding stored as Q4 at load: -644 MiB of weights, perplexity
                    4.346413 -> 4.343738 (noise), decode unchanged.
  --gdn-state-fp16  recurrent state stored as FP16: -72 MiB per device state slot (four at two
                    lanes), perplexity unchanged, greedy output bit-identical.
```
They add about 0.91 GiB to the two-lane headroom estimated below, taking the 212,992 default from
+0.63 GiB to about +1.54 GiB. Measured on the Windows box, they buy one full rung at one lane
(see Windows measurements below). --lm-head-q6 frees another 341 MiB for +0.01% perplexity but costs 2-5% of
single-lane decode until a Q6 small-T kernel exists, so only the mtp profile -- the one that is
buying context -- passes it, and DFlash/DFlash2 refuse it.

MEASURED, with a Windows desktop running (a headless box has roughly 1.5 GiB more to spend), and
BEFORE the two memory flags this profile now passes -- the earlier profile's figures:

```
  lanes  KV      context   vision   runtime    free after startup
  ------------------------------------------------------------------
  1      int8     65,536   off      2.73 GiB   2.85 GiB   <- what `run.sh qwen38-27b int8` does
  1      rk8v4    49,152   overlay  1.82 GiB   3.33 GiB
  1      rk8v4    98,304   overlay  3.09 GiB   2.07 GiB
  1      rk8v4   131,072   off      3.93 GiB   1.68 GiB
  1      rk8v4   131,072   overlay  3.94 GiB   1.59 GiB
  2      rk8v4   131,072   overlay  4.32 GiB   1.26 GiB
  1      rk8v4   163,840   overlay  4.78 GiB   763.2 MiB
```

With --embedding-q4 --gdn-state-fp16, measured 2026-09-14 on the same card and profile shape:

```
  lanes  KV      context   vision   runtime    free after startup
  ------------------------------------------------------------------
  1      rk8v4   131,072   overlay  3.80 GiB   2.47 GiB
  1      rk8v4   163,840   overlay  4.65 GiB   1.63 GiB
  1      rk8v4   196,608   overlay  5.49 GiB   798.1 MiB
```

Vision is on: overlay residency costs about 10 MiB of runtime reservation, so there is no reason
to trade it away. The former 32K vision-only launcher is `tuned` with `NINFER_CONTEXT=32768`; vision is already on.

WHY NOT 262,144 LIKE THE 35B-A3B. The runtime reservation is dead linear in context - seven
points from 49,152 to 163,840 fit

    runtime_bytes = 0.553 GiB + 27,719 x context        (worst residual 3.9 MiB)

at 27.07 KiB/token, and a second lane adds a flat 0.38 GiB. Without the memory flags a headless
3090 has about 7.06 GiB for the reservation, so 262,144 needs 7.32 GiB at one lane and 7.70 GiB at
two - it does not fit either way; the zero-margin ceilings are roughly 252,000 tokens at C1 and
237,000 at C2. With --embedding-q4 --gdn-state-fp16 the budget grows to about 7.69 GiB (644 MiB of
weights freed) and the line becomes 0.41 GiB + 27,719 x context, +0.24 GiB per extra lane (71.7 MiB
per FP16 state slot); the measured 196,608 rung at 5.49 GiB lies on it. On that estimate 262,144
fits with about +0.51 GiB at one lane and +0.27 GiB at two.

That is not a tuning failure, it is the model: the 27B spends 16 full-attention layers x 4
kv_heads x 256 head_dim per token against the 35B-A3B's 10 x 2 x 256, which is 3.2x the KV per
token - 27.07 KiB against roughly 7.8. The 35B-A3B reaches 262,144 because its KV is cheap.

The mtp profile's default is 212,992 at two lanes: 6.15 GiB predicted with the memory flags,
leaving about +1.54 GiB before the cuBLAS route's workspace (0.54 GiB) and --lm-head-q6 (which
returns 0.33 GiB) are counted. 196,608 is the more cautious rung and 262,144 the aggressive one.
The dflash2 profile defaults to 131,072 at ONE lane: that is where its measured ceiling sits (loads
at 130K, fails at 150K on a Windows desktop), its advantage is largest at one stream (+38.6% decode
at C1 against +31.6% at C2), and the draft weights take the headroom a second lane would use.
Both are extrapolated rather than measured on this launcher's exact command line - this machine
cannot start the larger rungs - so treat the first start as the confirmation and drop a rung if it
refuses. Rungs: 229376 / 212992 / 196608 / 163840 / 131072 / 114688 / 98304 / 65536.

### Windows measurements

MEASURED on this machine with the desktop running, which is the pessimistic case. Without the two
memory flags (the earlier profile):

```
  lanes  KV      context   vision   runtime    free after startup
  ------------------------------------------------------------------
  1      int8     65,536   off      2.73 GiB   2.85 GiB   <- what `run.bat qwen38-27b int8` does
  1      rk8v4   131,072   off      3.93 GiB   1.68 GiB
  1      rk8v4   131,072   overlay  3.94 GiB   1.59 GiB
  2      rk8v4   131,072   overlay  4.32 GiB   1.26 GiB
  1      rk8v4   163,840   overlay  4.78 GiB   763.2 MiB
```

With --embedding-q4 --gdn-state-fp16 (2026-09-14, arms alternated at each rung, desktop holding
1.25 GiB until 212,992, then 0.44 GiB):

```
  context    without flags               with flags                  + --lm-head-q6
  ---------------------------------------------------------------------------------------
  131,072    3.94 GiB / 1.70 GiB free    3.80 GiB / 2.47 GiB free    2.79 GiB free
  163,840    4.79 GiB /  873 MiB free    4.65 GiB / 1.63 GiB free    1.96 GiB free  <- mtp default
  196,608    5.63 GiB /    0 free        5.49 GiB /  798 MiB free    1.11 GiB free
  229,376    refused                     6.34 GiB /    0 free        276 MiB free
  245,760    refused                     refused                     starts, 0 free
```

Windows keeps one lane by default: a desktop holds roughly 1.5 GiB of the card, so the
headroom above is what you actually have. Vision is on -- overlay residency costs about 10 MiB
of runtime reservation, so there is no reason to trade it away. `tuned` with `NINFER_KV_DTYPE=int8` remains
for the plain 32K image profile.
Rungs if startup refuses: 196608 / 163840 / 131072 / 114688 / 98304 / 65536.

## Qwen3.6-35B-A3B, `tuned`

Qwen3.6-35B-A3B on one RTX 3090 -- single user. Runs the Ninja build directly out of
build-ninja\apps, not a packaged release.

READ THIS FIRST: the maximum context on this machine is not a fixed number. The engine sizes
its runtime reservation from whatever VRAM is free after the weights land, and Windows' own
GPU usage moved by 650-900 MB during a single afternoon of measuring -- explorer, SearchHost,
ShellHost and CrossDeviceResume come and go. That is worth 40,000-50,000 tokens of context.
Two consecutive probes of the same configuration reported 1,938 MB and 1,078 MB available.

The defaults below are therefore sized with margin, not at the cliff. The ladder shows what
was measured under a busy desktop (~21.9 GiB free) and what the same profile reached under a
quiet one (~22.6 GiB free). If a value fails to start, drop one rung.

```
  profile                    default   free @busy   reached @quiet   decode
  ----------------------------------------------------------------------------
  A  MTP3 + draft head       147,456      666 MiB          196,608   ~240 tok/s
  B  no speculation          196,608      375 MiB          262,144   ~183 tok/s
```

Profile A's row is with --mtp-experts-q4 --gdn-state-fp16 (below), measured 2026-09-14 with the
desktop at ~505 MiB; 196,608 is the largest rung that started, with 246 MiB free. Before those
flags it was 81,920 by default, 193 MiB free at busy, and 131,072 at quiet. Profile B predates
them and does not use MTP, so only --gdn-state-fp16 would apply there.

MEMORY FLAGS on profile A (2026-09-14, docs/maintainer/quality-trade-experiments.md). The MTP draft
layer's routed experts ship as W8 (816 MiB) while the text layers' are Q4/Q6; --mtp-experts-q4
stores them the same way at load, and --gdn-state-fp16 halves the recurrent state. Drafts are
verified by the target, so neither changes what the model scores: decode 296.3 -> 294.9 tok/s,
acceptance 61.2%% -> 60.6%%. Same profile, arms alternated per rung, desktop at ~505 MiB:

```
  context    without flags     with both flags
  ------------------------------------------------
  114,688    535 MiB free      947 MiB free
  147,456    254 MiB free      666 MiB free   <- default now; more margin than 114,688 had
  163,840    107 MiB free      526 MiB free
  196,608    refused           246 MiB free
```

WHAT SPECULATION COSTS IN MEMORY, exact, read from the artifact directory:

```
  MTP head            856 MiB   (--spec mtp)
  draft head          136 MiB   (--lm-head-draft, requires MTP)
  both                992 MiB

  converted to context:        int8 (10,560 B/tok)   rk8v4 (7,969 B/tok)
  draft head alone                   13,504 tokens         17,896 tokens
  MTP head alone                     85,003 tokens        112,646 tokens
  both                               98,507 tokens        130,542 tokens
```

So speculation does not buy context, it spends it: turning MTP off is worth roughly half the
native 262,144 back. The draft head is the cheap half of that pair -- 136 MiB for a measured
+1.9% to +15.6% decode, against the MTP head's 856 MiB for +38%.

MEASURED MAX CONTEXT, all four combinations, taken under the quiet-desktop condition BEFORE the
memory flags (pre-flag history; the active profile's current figures are in MEMORY FLAGS above):

```
  KV       speculation        max context   free after startup
  ------------------------------------------------------------
  rk8v4    none                   262,144         ~256 MiB      native maximum
  rk8v4    MTP3 + draft head      131,072         ~184 MiB
  int8     none                   196,608         ~344 MiB
  int8     MTP3 + draft head       94,208         ~292 MiB
```

With --mtp-experts-q4 --gdn-state-fp16 the rk8v4 MTP3 + draft head row reaches 196,608 (246 MiB
free, measured 2026-09-14 with the desktop at ~505 MiB, a busier condition than this table's).

rk8v4 is worth +33% context unspeculated and +39% with speculation, for +0.082% perplexity.
int8 cannot reach the native 262,144 at all -- 204,800 already over-runs the reservation.

HOW MUCH MTP IS WORTH, ninfer_bench tg512, rk8v4, 8K context:

```
  Qwen3.6-35B-A3B (MoE, ~3B active)   no spec 183.49   MTP3 253.50 (1.38x)   +draft 280.38
  Qwen3.8-27B     (dense)             no spec  39.67   MTP3  63.21 (1.59x)
```

Speculation helps this MoE *less* than the dense 27B -- 1.38x against 1.59x -- despite higher
acceptance (66.1% against 54.6%). That is structural: speculation pays by amortising one weight
read across several accepted tokens, and in a dense model every token reads the same weights.
Here each drafted token routes to its own 8 of 256 experts, so verifying four touches far more
expert weight than verifying one. The 35B also starts far less memory-bound, 183 against 40
tok/s, because only ~3B of 35B parameters are active per token.

--lm-head-draft drafts over 131,072 rows instead of the full 248,320 output head: cheaper to
draft, but it cannot propose anything outside that subset. Whether that pays is entirely
content-dependent and the synthetic benchmark corpus is a bad guide -- on the dense 27B it
reverses the sign. Re-measured on this model with real content, rk8v4, MTP3, greedy:

```
  content                        without    with     delta   acceptance without -> with
  ---------------------------------------------------------------------------------------
  ninfer_bench tg512 (synthetic)  253.50   280.38   +10.6%     66.15% -> 66.02%
  pure code generation            320.32   326.32    +1.9%     89.81% -> 83.82%
  mixed prose + code              188.08   217.42   +15.6%     39.09% -> 41.12%
```

It helps on every content type here, by very different margins, and on mixed content it raises
acceptance rather than costing any. Note the spread in absolute terms: 320 tok/s on pure code
against 188 on mixed prose, because code is far more predictable. Any single decode figure for
this model is really a statement about the text being generated.

CONTEXT CACHE, measured on the earlier engine. The flags below (--max-shared-prefixes,
--host-state-slots, --auto-prefix-grid, --max-private-continuations) no longer exist: the current
engine retains conversation endpoints and stable input boundaries inside one Device KV pool plus one
pinned Host context budget (--host-context-mib), with nothing else to size. The measurements are kept
as history and have not been repeated on the current engine. Then: the catalog defaults were too
small and it did not show up as an error, only as prefill you keep paying. A checkpoint is a KV prefix plus a StateImage, and on this model the
StateImage is 61.4 MiB *flat* regardless of prefix length -- 30 GDN layers of 128x128x32 FP32
recurrent state plus conv. Unlike KV pages, which several checkpoints of one conversation share,
it cannot be shared between two frontiers at all: the recurrent state at token N is a function
of every token before it. That fixed cost is why the shipped catalog is small.

Measured, eight distinct ~1430-token preambles round-robined four times, reuse after round one:

```
  --max-shared-prefixes    reuse    what happens
  ------------------------------------------------------------------------------------
  4  (the old default)      8.4%    only one preamble ever stays cached; constant thrash
  8                        98.3%    all eight stay; prefill 0.317 s -> 0.044 s
  16                       98.3%    no better than 8 once the catalog fits the working set
```

Sizing rule: --max-shared-prefixes should match the number of distinct preambles in play, and
--host-state-slots roughly twice (shared + private). Raising them costs pinned HOST memory and
nothing on the device -- measured on this exact profile, both settings resolve the full 114,688
tokens with byte-identical 492.4 MiB free after startup; the only change is host state pinned
982.6 MiB -> 1.92 GiB and 26 ms more startup.

--auto-prefix-grid offers shared candidates on a content-independent token grid, so two requests
that merely start alike -- the same pasted document in two different chats, the same few-shot
preamble inside one user message -- converge on the same frontier without any client hint. A
grid point is only ever materialised once two independent callers have both asked for it, so it
cannot waste a slot speculatively. Measured on a prompt with no structural boundary at all:
0% -> 82.6% reuse, prefill -71%, TTFT -64%, and the cold requests before it warms are unchanged.

LONG CONVERSATIONS do not need any of the above -- they run on the private turn-closure path,
which is what --max-private-continuations sizes. Measured with a 194-turn coding-agent loop
(tool calls, file pastes, whole history resent each turn) growing to 64,668 tokens: exactly one
cache miss, on turn 1. Computed prefill stayed flat at ~340 tokens per turn no matter how long
the history got, and TTFT went 0.085 s at 3k to 0.177 s at 64k. The limit on turn count is
--max-context, not the cache.

VISION: on, via --vision-residency overlay (ported from Don-Chad/ninfer-3090#21). Resident
residency cost 261 MiB of the ~450 MiB this profile has free after startup -- too tight to risk.
Overlay keeps the Vision tower host-pinned and streams each image through a borrowed device
window instead, so --vision no longer costs any resident capacity at this context. Measured on
this exact profile: KV capacity still resolves the full 114,688 tokens, free-after-startup was
453.86 MiB (this boot; it moves with desktop GPU load like everything else on this page), and a
real image request completed in 0.52 s wall (overlay window 180 ms, 60 MiB evicted in 5.2 ms,
restored in 4.7 ms, 268 MiB staged from host) with decode unaffected at 284.6 tok/s. If VRAM is
tighter on a given boot and the server refuses to start, drop --vision first before dropping the
context rung -- it is the newest addition, not the load-bearing one.

Maximum context on this machine is not a fixed number: the engine sizes its runtime reservation
from whatever VRAM is free after the weights land, and under WSL the Windows desktop is still
holding part of the card. The default below is sized with margin. If it fails to start, drop one
rung: 114688 / 98304 / 90112 / 81920.

## Pinned Host context on Windows

`--host-context-mib 8192` pins the full 8 GiB of host RAM on Windows, as on Linux. Earlier builds of
this fork clamped the pinned host allocation on Windows to (free VRAM - 1 GiB) / 2, after a
measurement where the largest single `cudaMallocHost` tracked free VRAM (WDDM charging pinned memory
to the card). At the `tuned` profiles' residency that left the old `--host-kv-mib 8192` at about
4.6 GiB or less (302 MiB for the 27B MTP profile and none at all for the 35B-A3B on one measured
boot). That coupling is gone on current drivers: on the RTX 3090 with driver 616.64, 22,528 MiB
resident and 804 MiB free, `cudaHostAlloc` pinned 16 GiB in one allocation and 32 GiB in 1 GiB
chunks, reported free VRAM fell by a fixed ~0.5 GiB, and a further 512 MiB `cudaMalloc` still
succeeded. The clamp was removed, so a Windows machine running the launchers now really holds 8 GiB
of RAM pinned for the context cache. Lower `NINFER_HOST_CONTEXT_MIB` on a box short of RAM; `0`
keeps the cache on the card.
