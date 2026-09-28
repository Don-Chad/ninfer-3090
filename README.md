# NInfer-3090

NInfer-3090 is a specialized C++20/CUDA inference engine for **Qwen3.8-27B** and Qwen3.6 on one
24 GB NVIDIA GeForce RTX 3090. Qwen3.8-27B is a first-class, tested target: the native SM86
runtime loads its official groupwise `.ninfer` artifact, serves OpenAI- and Anthropic-compatible
APIs, and supports paged KV, compatible-prefix reuse, CUDA Graphs, MTP speculative decoding,
reasoning-effort control, ReplaySSM state transactions, and concurrent cohorts through **C8**.

Community project, maintained on a best-effort basis. Issues and PRs are very welcome, but support
and feature requests are not guaranteed.



On an RTX 3090, Qwen3.8-27B supports a measured **171K-token INT8 context** with the standard
1 GiB safety headroom, **226K tokens** with the opt-in RotorQuant `rk8v4` profile, and the full
**262,144-token native context** with the opt-in [`rk4v4`](#lloyd-max-4-bit-keys-rk4v4) profile,
even alongside MTP3 speculation and the draft head (where `rk8v4` stops at 182,336).

> **RotorQuant `rk8v4` is available again**, ported onto the `kv_cache_append` Op that now owns KV
> quantization. `--kv-dtype rk8v4` reaches a measured **226,560-token context** in about the same
> KV the INT8 profile spends on 171,648 tokens, for **+0.082% perplexity**. It is opt-in; INT8
> remains the default and the quality-default profile.

This fork targets `sm_86`. Blackwell-only NVFP4/W4A4 and FP8 A8 tensor-core *weight and
activation* execution are unavailable. FP8 and NVFP4 weights are admitted through their A16
dequantizing routes. The paged runtime's KV-cache storage is a separate axis from weight/activation
kernels: all seven KV formats, including row-scaled FP8 E4M3, are measured and available on SM86 —
see [`docs/config-calculator.html`](docs/config-calculator.html).

The goal is the make the utmost rippin Qwen inference stack for the 3000 series. Gladly taking PR's, all help much appreciated. 

**New in v0.11.0: prompt processing is roughly twice as fast, and the recommended profile is the fast one.**
Qwen3.8-27B prefill reaches 2,989 tok/s at 4K (`--prefill-cublas --prefill-chunk 4096`, +0.156%
perplexity, opt-in) and 1,649 tok/s on the default route, single-stream decode reaches 187 tok/s with
DFlash2 (+39% over MTP3), and one `run` launcher picks the flags for you. See the
[v0.11.0 release notes](RELEASE_NOTES_0.11.0.md) and the
[measured configurations](docs/performance.md#recommended-configurations-rtx-3090-qwen38-27b).

Previous: [v0.10.0](RELEASE_NOTES_0.10.0.md) (tensor-core small-T kernels: MTP3 decode 1.5x at C1 and
1.7x at C8), [v0.9.1](RELEASE_NOTES_0.9.1.md), [v0.9.0](RELEASE_NOTES_0.9.0.md).

> **Model files are now v3.** This release follows upstream NInfer onto the v3 `.ninfer` container.
> A v2 file from an earlier release is refused at load; upgrade it in place of a re-download with
> `python tools/upgrade_ninfer_v2_to_v3.py OLD.ninfer NEW.ninfer` (standard library only, works on
> Windows and Linux, and keeps the weight bytes unchanged). See
> [weight conversion](docs/weight-conversion.md#upgrade-an-existing-v2-artifact).

## Quick start

**You do not need to build anything.** Grab the prebuilt archive for your platform from
[the latest release](https://github.com/ashalliants/ninfer-3090/releases/latest) — Windows x64 and
Linux x64 are both published, each with `ninfer-serve`, the CLI, the benchmark tool, every launcher
and, on Windows, the DLLs. Unpack it, run two scripts, and point your harness at
`http://127.0.0.1:8080/v1`. It is an OpenAI-compatible endpoint, so anything that speaks
`/v1/chat/completions` works; leave the API key blank.

The launchers below run **Qwen3.6-35B-A3B** with the settings this project measured as the best
overall trade on one 24 GB RTX 3090: `rk4v4` KV (twice INT8's context per GiB for +0.21%
perplexity), MTP3 speculation plus the draft head, vision through the overlay residency so it costs no resident
capacity, and the tuned context cache (8 shared prefixes, 32 host state slots, automatic prefix
grid) that takes prefix reuse from 8.4% to 98.3% on a multi-preamble workload.

### Windows 11 — full 256K context, two lanes

Download `ninfer-rtx3090-windows-x64-*.zip`, unzip it, and from that folder:

```powershell
.\download-model.bat qwen36-35b-a3b       # downloads qwen3_6_35b_a3b.ninfer (~21 GB, resumable)
.\run.bat qwen36-35b-a3b                   # serves on 127.0.0.1:8080, 2 lanes sharing 262,144 tokens
```

Double-clicking either file asks which model instead. `NINFER_HOST`, `NINFER_PORT`, `NINFER_MODEL` and `NINFER_SERVER` override it without editing the
file; `set NINFER_HOST=0.0.0.0` exposes it to the LAN, unauthenticated.

### Headless Linux — full 256K context, three lanes, everything on

Download `ninfer-rtx3090-linux-x64-*.tar.gz`, unpack it, and from that folder:

```bash
tar -xzf ninfer-rtx3090-linux-x64-*.tar.gz && cd ninfer-rtx3090-linux-x64-*/
./download-model.sh qwen36-35b-a3b       # downloads qwen3_6_35b_a3b.ninfer (~21 GB, resumable)
./run.sh qwen36-35b-a3b                  # 3 lanes sharing 262,144 tokens, MTP3 + draft head, vision
```

Both launchers run the native 262,144-token maximum with `rk4v4`, MTP3 speculation plus the draft
head, and vision in overlay residency, all at once: two lanes on Windows, three on Linux. Three
lanes were measured to start even beside a desktop (2026-09-24); four fall 48 MB short there, so a
headless card may take `NINFER_CONCURRENCY=4`.

Lanes share one KV pool: `--kv-capacity` is the pool and `--max-context` the per-request cap,
and the launchers set both to 262,144. Any one request can use the full native context, but the
lanes' requests together hold at most 262,144 tokens at a time, so two users can't each keep a
200K conversation resident. A lane adds only its fixed state, not a second pool.

`NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_CAPACITY`, `NINFER_SPEC`, `NINFER_VISION`,
`NINFER_MODEL`, `NINFER_HOST` and `NINFER_PORT` override it. The launcher binds `127.0.0.1`;
set `NINFER_HOST=0.0.0.0` to expose it, which is unauthenticated.

### Which profile

With `rk8v4` the full profile needed the ~1.5 GiB a desktop holds, so Windows ran one user at
147,456 tokens. `rk4v4` stores the same context in 31% less memory, which is what lets every
launcher run at the full context:

| Profile (`rk4v4`) | lanes | context | speculation | vision | starts beside a desktop |
|---|---|---|---|---|---|
| **Headless Linux — default** | 3 | 262,144 | MTP3 + draft | overlay | yes (measured) |
| **Windows — default** | 2 | 262,144 | MTP3 + draft | overlay | yes (measured) |
| `NINFER_CONCURRENCY=3` | 3 | 262,144 | MTP3 + draft | overlay | yes, 24.0 of 24.5 GiB used |
| `NINFER_CONCURRENCY=4` | 4 | 262,144 | MTP3 + draft | overlay | no, 48 MB short; headless should fit |

`--kv-capacity` is the shared pool and `--max-context` is the per-request cap, so a second lane
does not cost twice the memory unless you also want twice the per-request context.

**To size a profile that is not in this table, open
[`docs/config-calculator.html`](docs/config-calculator.html) in a browser.** It is a single
self-contained file in this repository — no network access needed — that takes a model, a KV
format and a context length and tells you whether it fits in 24 GiB, what the largest context you
could run instead would be, and what the choice costs in decode speed and perplexity. Every
constant in it is measured on an RTX 3090 against this fork.

If startup refuses, drop a lane or a context rung first — 262144 / 196608 / 131072 / 114688 / 98304
/ 81920. Speculation is the next lever, worth 992 MiB (MTP head 856 MiB, draft head 136 MiB, roughly
185,000 rk4v4 tokens) at the cost of dropping decode to ~183 tok/s. Drop `--vision` last: in overlay
residency it costs no resident capacity, and the `evictable pool window exceeds the evictable tail`
message some boxes show is a symptom of the reservation already being tight, not a context ceiling.

### Or Qwen3.8-27B

The other common choice, and a dense model rather than an MoE, so it is slower per token but more
predictable. Same shape of command:

```powershell
.\download-model.bat qwen38-27b          # downloads qwen3_8_27b.ninfer (~19 GB, resumable)
.\run.bat qwen38-27b                      # one user, 172,032 tokens, DFlash2, cuBLAS prefill, rk4v4, vision
```

```bash
./download-model.sh qwen38-27b
./run.sh qwen38-27b                       # one user, 262,144 tokens (headless), same flags
```

The launcher's default is the fast profile: `--spec dflash2 --draft-tokens 7 --lm-head-draft
--prefill-cublas --prefill-chunk 4096 --kv-dtype rk4v4 --embedding-q4 --gdn-state-fp16 --vision
--vision-residency overlay`, about 1.7x the previous prefill and 1.39x the decode. Beside a desktop
it starts at up to 180,224 tokens (`rk8v4`: 131,072), so Windows defaults to 172,032; DFlash2's
draft weights and its refusal of `--lm-head-q6` are why it stops short of 262,144 there, and a
headless card gets the ~1.45 GB back that the full context needs. For the full context on any
card, and a second lane, run it with `NINFER_SPEC=mtp` (Windows: `set NINFER_SPEC=mtp && run.bat qwen38-27b`; Linux:
`NINFER_SPEC=mtp ./run.sh qwen38-27b`), which swaps in `--spec mtp --draft-tokens 3 --lm-head-draft
--prefill-cublas --prefill-chunk 2048 --kv-dtype rk4v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16
--vision --vision-residency overlay` at 262,144 tokens, with two lanes sharing that pool. Both sets are measured in
[performance](docs/performance.md#recommended-configurations-rtx-3090-qwen38-27b).

The older reference profiles are `run.sh qwen38-27b int8` (one user at 65,536 tokens of INT8, which
leaves 2.85 GiB of the card unused) and `run.sh qwen38-27b c8` (eight lanes at 8K). Prefer the
default `tuned` profile unless you specifically want INT8's quality default or `c8`'s aggregate
throughput. Every profile's host and port default to `127.0.0.1:8080` and accept the same
`NINFER_HOST`/`NINFER_PORT` overrides. The measurement history behind each default is in
[launcher profiles](docs/maintainer/launcher-profiles.md).

| Profile | lanes | context | KV | vision | runtime | free (desktop) |
|---|---|---|---|---|---|---|
| `int8` profile | 1 | 65,536 | int8 | off | 2.73 GiB | 2.85 GiB |
| **`tuned`** (default, DFlash2), Windows | 1 | 172,032 | rk4v4 | overlay | – | starts up to 180,224 beside a desktop |
| `tuned` (default, DFlash2), Linux | 1 | 262,144 | rk4v4 | overlay | – | headless extrapolation; steps down if refused |
| `NINFER_SPEC=mtp` | 2 | 262,144 | rk4v4 | overlay | – | 23.4 of 24.5 GiB used beside a desktop |

The `rk4v4` rows were measured by starting `ninfer-serve` with each launcher's flags on a desktop
RTX 3090 (2026-09-24); earlier `rk8v4` figures are in
[launcher profiles](docs/maintainer/launcher-profiles.md). The
DFlash2 profile takes one lane because its advantage is largest at one stream (+38.6% decode at C1,
+31.6% at C2) and the draft weights use the headroom a second lane would need.

All `-maxctx` rows run with `--embedding-q4 --gdn-state-fp16`, which the launchers pass: the token
embedding is stored as Q4 (-644 MiB of weights) and the GDN state as FP16 (-72 MiB per device state
slot), both measured free on quality. The Windows figures were measured on the launcher's exact
profile; before those flags the same card started 131,072 with 1.59 GiB free and 163,840 with
763 MiB. The Linux runtime estimate is the earlier 6.43 GiB less four FP16 state slots.

Vision is on in both. Overlay residency keeps the tower host-pinned and streams each image through
a borrowed device window, so it costs about **10 MiB** of runtime reservation — measured 3.93 GiB
without it against 3.94 GiB with, at the same context. There is no reason to trade it away.

**What limits the 27B's context.** The runtime reservation is linear in context: seven measured
points from 49,152 to 163,840 fit `runtime = 0.553 GiB + 27,719 × context` with a worst residual of
3.9 MiB, and a second lane adds a flat 0.38 GiB. Without the memory flags a headless 3090 has roughly
7.06 GiB for the reservation, so 262,144 would need 7.32 GiB at one lane and 7.70 GiB at two — it
does not fit either way, and the zero-margin ceilings are about 252,000 tokens at C1 and 237,000 at
C2.

The two flags the launchers now pass move both sides of that sum. `--embedding-q4` frees 644 MiB of
weights, so the reservation budget grows to about 7.69 GiB; `--gdn-state-fp16` takes 71.7 MiB off
each device state slot, so the intercept drops to 0.41 GiB at one lane and each second lane adds
0.24 GiB. The measured 196,608 rung (5.49 GiB) sits exactly on that line. On that estimate the native
262,144 fits a headless card with about +0.51 GiB at one lane and +0.27 GiB at two — extrapolated,
not measured, since no desktop machine can start it.

That is the model, not the tuning. The 27B spends 16 full-attention layers × 4 kv_heads × 256
head_dim per token against the 35B-A3B's 10 × 2 × 256 — **3.2× the KV per token**, 27.07 KiB
against roughly 7.8. The 35B-A3B reaches the native maximum because its KV is cheap.

So the Linux launcher's `mtp` profile defaults to **212,992** — 6.15 GiB predicted at two lanes with
the flags, leaving about +1.54 GiB (it was 6.43 GiB and +0.63 GiB before them). `NINFER_CONTEXT=196608` is the
more cautious rung at about +1.96 GiB, and `NINFER_CONTEXT=262144` the aggressive one at about
+0.27 GiB. All three are extrapolated rather than measured, since a desktop machine cannot start
them, so treat the first headless start as the confirmation and drop a rung if it refuses.

One thing to know: this model's StateImage is **147 MiB** in FP32, 2.4× the 35B-A3B's, because it has
48 GDN layers with 48 value heads, and **74.5 MiB** with the `--gdn-state-fp16` the launchers pass.
`--host-state-slots 32` therefore pins **2.34 GiB of host memory** (4.59 GiB without the flag) — host,
not device, and the price of taking prefix reuse from 8.4% to 98.3%. Lower it if the box is short
on RAM.

Qwen3.8-27B reaches 171,648 INT8 tokens or 226,560 with `rk8v4`; the figures above are what fits
alongside speculation and the tuned cache with a desktop running.

## Choose a platform

Both platforms ship a prebuilt archive; building from source is optional and covered
[further down](#building-from-source).

| Platform | Delivery | Guide |
|---|---|---|
| Linux x64 | Prebuilt release archive, or Docker / native source build | [Linux guide](docs/rtx-3090-linux.md) |
| Windows 11 x64 | Prebuilt release archive | [Windows guide](docs/rtx-3090-windows.md) |

### Linux

1. Download and unpack the latest
   [Linux release](https://github.com/ashalliants/ninfer-3090/releases/latest)
   (`ninfer-rtx3090-linux-x64-*.tar.gz`).
2. Run `./download-model.sh qwen36-35b-a3b` or `./download-model.sh qwen38-27b` to fetch a model
   (`qwen36-27b` is the third). Each pins a HuggingFace revision, stage under a revision-scoped name so a resume can only ever continue the
   same artifact, and verify size and SHA-256 before promoting it. Interrupted downloads resume.
3. Run `./run.sh qwen36-35b-a3b` (recommended) or `./run.sh qwen38-27b` for the dense 27B.

If you would rather build, the Dockerfile is the shortest path on Bazzite and other distributions:

```bash
docker build --tag ninfer-3090:sm86 .
```

The Linux guide covers the GPU check, the native Ubuntu build, model mounts and the server command.

### Windows 11

1. Download and unzip the latest
   [Windows release](https://github.com/ashalliants/ninfer-3090/releases/latest)
   (`ninfer-rtx3090-windows-x64-*.zip`).
2. Double-click `download-model.bat` and pick a model (or run `download-model.bat qwen36-35b-a3b`,
   `qwen38-27b` or `qwen36-27b` from a terminal). Each pins a HuggingFace revision, stages under a
   revision-scoped name so a resume can only ever continue the same artifact, and verifies size and
   SHA-256 before promoting it. Interrupted downloads resume.
3. Double-click `run.bat` and pick a model, or run one of:

| Command | Best for |
|---|---|
| `run.bat qwen36-35b-a3b` | **Recommended.** Qwen3.6-35B-A3B, two lanes sharing a 262K pool (any one request up to 262K), rk4v4, vision, tuned cache |
| `run.bat qwen38-27b` | **Recommended for 27B.** Qwen3.8-27B, one user, 172K context, DFlash2, cuBLAS prefill, rk4v4, tuned cache; `NINFER_SPEC=mtp` for the full 262K and a second lane |
| `run.bat qwen38-27b int8` | Qwen3.8-27B, one interactive user, INT8 quality default, 64K context |
| `run.bat qwen38-27b c8` | Qwen3.8-27B, multiple users or agents, highest aggregate throughput, 8K context |

The default profiles serve images too (vision in overlay residency), so there is no separate vision
launcher.

The API is then available at `http://127.0.0.1:8080/v1`. The Windows archive includes the required
applications and DLLs.

## Building from source

Only needed if you are changing the code — the release archives above are prebuilt for `sm_86`.

`scripts/build.ps1` (Windows) and `scripts/build.sh` (Linux/WSL) pin the toolchain this project
needs and fail with a message naming the real cause when one is missing. Three things are not the
defaults on a typical machine: MSVC 14.4x from **VS 2022 BuildTools** (CUDA 12.8 rejects VS 2026's
14.50), **CUDA 12.8** forced through `CUDACXX`, and the **Ninja** generator.

```powershell
.\scripts\build.ps1                  # configure + build into build-ninja
.\scripts\build.ps1 -Test            # ... and run the test suite
.\scripts\build.ps1 -Package         # ... and build the release archive
```

```bash
./scripts/build.sh --test --package
```

## Qwen3.8-27B support and RTX 3090 results

### Decode after the small-T kernels

MTP3 decode is **1.5x faster at C1 and 1.7x at C8** than v0.9.1, measured before/after on the
same card. The MTP verify round and the concurrent-cohort round both run 4-32 token columns, and
those widths used to fall between the single-token GEMVs and the prefill GEMM tiles. They now
run on tensor-core small-T kernels built for exactly that range; the
[performance page](docs/performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode)
has the kernel-level story.

Both builds ran side by side on one rented RTX 3090 (Linux, CUDA 12.8, 350 W power limit, ~1.56
GHz under load) in one session, with INT8 KV, MTP3, the optimized draft head, CUDA Graphs and
greedy sampling unless noted. Output quality is untouched: perplexity on the quick corpus is
bit-identical (4.342425), and only decode-width routes changed.

**Reasoning cohort**: `tools/bench/run_qwen38_replayssm_cohort_sweep.py`, the harness behind the
v0.9.1 table further down, with 1,024 output tokens per request.

| Cohort | v0.9.1 decode | small-T kernels | change |
|---:|---:|---:|---:|
| C1 | 63.25 tok/s | **92.88 tok/s** | +47% |
| C2 | 88.76 tok/s | **168.03 tok/s** | +89% |
| C4 | 136.95 tok/s | **270.79 tok/s** | +98% |
| C8 | 221.46 tok/s | **376.16 tok/s** | +70% |

**Thinking-off chat**: the eight prompts of
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)'s
`bench/prompts_real.jsonl`, with 1,024 output tokens and `reasoning_effort: none`. Decode is their
metric, C × 1000 / mean TPOT, so the right-hand column can be read against it.

| | v0.9.1 | small-T kernels | their patched vLLM stack (their README) |
|---|---:|---:|---:|
| C1, greedy | 74.6 tok/s | **113.2 tok/s** | 111-124 tok/s (MTP) |
| C1, temperature 0.7 | 72.6 tok/s | **112.8 tok/s** | — |
| C8, greedy | 268.6 tok/s | **460.4 tok/s** | 407.3 tok/s |

Read the last column with two caveats. Their card is capped at 250 W and this one ran at 350 W.
They also report 5-8% run-to-run spread, so C1 is parity, not a win. Tokens per round did not
move (2.88 at C1 on these prompts), so every gain here is round cost.

`ninfer_bench` (256 tokens, three repetitions): plain decode 39.98 -> **47.15 tok/s**, MTP3
53.96 -> **85.29 tok/s**. An MTP3 verify round now costs 1.15x a plain decode step, down from
about 1.5x; vLLM/Marlin on the same card is 1.14x.

Four and five draft tokens no longer cost more than they return. Until the GDN conv projection
was moved onto the same kernels they measured 93.8 and 91.5 tok/s, against three draft tokens'
109.4; now they land within 1% of it (108.7 and 108.1). `--draft-tokens 3` stays the
recommendation.

### v0.9.1 long-output cohort (earlier host)

Measured on a different RTX 3090 host before the small-T kernels; compare within a table, not
across the two sections.

Qwen3.8-27B is validated from one through eight simultaneous users. ReplaySSM cuts the memory cost
of speculative decoding, allowing the faster MTP3 mode to remain enabled at C8. The table below is
the new sustained test: every request generated 1,024 tokens with CUDA Graphs enabled.

The prompts were **29-34 input tokens** and the server's maximum context window was **8,192 tokens
per request**. Each measured sequence therefore reached roughly 1,053-1,058 tokens including its
generated output. This is a long-output/decode benchmark, not an 8K-prompt or long-prefill test.
C1 used an 8,192-token shared KV pool; C2-C8 used 16,384 tokens so every requested output could be
admitted simultaneously.

| Cohort | Total output | End-to-end throughput | Decode throughput | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 tokens | **77.84 tok/s** | **78.71 tok/s** | 71.27% | 133 ms | 19,475 MiB |
| C2 | 2,048 tokens | **94.75 tok/s** | **96.04 tok/s** | 62.17% | 225 ms | 19,919 MiB |
| C4 | 4,096 tokens | **136.43 tok/s** | **139.91 tok/s** | 66.15% | 420 ms | 20,247 MiB |
| C8 | 8,192 tokens | **240.34 tok/s** | **250.26 tok/s** | 69.74% | 866 ms | 20,903 MiB |

C1 is the responsive choice for a single user. C8 delivers **3.2x the total throughput** when
several requests are active. The C8 long-output test uses a 16K shared KV pool so all eight
1,024-token responses can be admitted together.

Concurrent decode extents are what the sm_86 kernel routes are selected for. A decode round covers
`concurrency x (draft window + 1)` token columns, and above the single-token point the cost of a
route is set by its padded tile width rather than by the live column count. Selecting the narrowest
tile that still covers each extent is worth 40% at C4 and C8; C1, whose four columns already sit on
the exact-T routes, is unchanged.

### Prompt-processing speed

Prompt processing was tested separately with **4,362 fresh input tokens per request**, an 8,192-token
per-request context window, 512-token prefill chunks, INT8 KV, ReplaySSM/MTP3, CUDA Graphs, and
prefix reuse disabled. Each request generated only 16 tokens so the run measures prefill rather
than long decode.

| Cohort | Total fresh input | Aggregate prefill | Active-prefill speed | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| C1 | 4,362 tokens | **861.51 tok/s** | 893.98 tok/s | 4,893 ms | 19,114 MiB |
| C2 | 8,724 tokens | **853.86 tok/s** | 883.95 tok/s | 7,478 ms | 19,697 MiB |
| C4 | 17,448 tokens | **847.26 tok/s** | 874.49 tok/s | 12,692 ms | 20,894 MiB |
| C8 | 34,896 tokens | **844.10 tok/s** | 870.94 tok/s | 23,028 ms | 23,207 MiB |

`Aggregate prefill` is total fresh input tokens divided by the complete request-wave time, so it is
the user-facing throughput number. NInfer currently processes one long prefill at a time; cohort
batching accelerates decode, but does not multiply prompt ingestion. Consequently C1-C8 remain near
844-862 input tok/s while queued requests increase mean TTFT. `Active-prefill speed` excludes queue
waiting and measures only the server's recorded prefill phase.

### Integer-activation MLP at decode (`--mlp-a8-decode`)

Opt-in, off by default, Qwen3.8-27B on `sm_86`. The MLP gate_up projection already runs its full
prefill tiles through the s8 tensor cores; this flag extends that to the widths a cohort round
decodes at, quantising activations to s8 with one scale per (token, 64-k group) and feeding
`mma.m16n8k32.s8.s8.s32`. It buys a little speed and costs a little fidelity.

**The Op, paired against the BF16 small-T kernel inside one sitting** (the card drifts several
percent between sittings, so only the pairing is meaningful), cold, median of 15:

| columns | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| run 1 | +10.7% | -5.8% | -3.4% | -4.8% | -4.8% | -5.0% | -4.5% |
| run 2 | +7.5% | +4.3% | -0.7% | -5.9% | -6.1% | -4.0% | -6.4% |

It wins from sixteen columns up and loses at eight, so the route is admitted for 16..32 columns
only and every narrower width stays on the BF16 kernel. A cohort round reaches those widths through
concurrency: eight lanes verifying four MTP columns each is thirty-two.

**End to end it is worth about a percent.** Eight concurrent thinking-off chat requests, MTP3,
INT8 KV, greedy, four interleaved repetitions:

| | decode |
|---|---|
| default | 431.5 tok/s |
| `--mlp-a8-decode` | **437.0 tok/s** (+1.28%) |

That is the expected size rather than a disappointment: gate_up is roughly a quarter of a C8 round,
so four to six percent off it arrives as one percent overall. The flag won three of the four paired
repetitions and tied the fourth, against a spread of about 1.7% within the unflagged arm alone.

**What it costs.** Output changes -- this is a lossy trade, not a free one. Against an FP64 oracle
the Op measures 0.0080 to 0.0371 relative L2 across 2..32 columns, inside the 0.04 allowance the
integer-activation path is held to everywhere else in the tree. Perplexity cannot see this trade at
all: the route is admitted only in the verify phase, and scoring runs the prefill phase, so
`ninfer-perplexity` reports the same score with and without the flag. Judge it on the oracle bound and on your own outputs.

Not recommended for single-stream use, where it does nothing: one request decodes one column per
step, far below the sixteen the route needs.

### RotorQuant KV (`rk8v4`)

`rk8v4` is an experimental, opt-in KV-cache mode for Qwen3.8-27B: keys keep the rotated INT8
group-64 encoding, and values are stored as signed 4-bit codes, two per byte, over a **group-32**
scale. It buys context, not speed.

Values use a finer group than keys because four bits resolve a group to only 15 levels, so a single
outlier would otherwise set the quantization step for 64 neighbours. Halving the value group to 32
costs one extra FP16 scale per 64 dimensions and **halves the perplexity penalty**, from +0.146% to
+0.082%, without costing any context at the automatic-sizing boundary.

Unlike the pre-merge implementation, **values are not rotated**. The old code applied an H64
rotation to both K and V and undid the value rotation with a separate pass over the attention
output. Upstream's `kv_cache_append` contract stores values from the represented BF16 source
directly, and measurement showed that is sufficient, so this port keeps it: there is no
inverse-rotation kernel and no extra pass over the output.

### Lloyd-Max 4-bit keys (`rk4v4`)

`rk4v4` keeps `rk8v4`'s value plane and halves its keys: each rotated key dimension is a 4-bit
index into the 16-level Lloyd-Max quantizer for a Gaussian, with one FP16 scale per 64 dimensions.
That is the TurboQuant idea (rotate, then snap each coordinate to a fixed non-uniform codebook)
without its 1-bit residual stage, which measured no better. It is **31% smaller than `rk8v4`**
(17,920 B/token on the 27B against 26,112) and 3% larger than `nvfp4`, which it beats on every
other axis.

Each codebook level is stored as a fixed INT8 code, so an expanded key is an ordinary rotated INT8
key and attention keeps `rk8v4`'s INT8 tensor-core QK path unchanged. Both attention kernels load
the packed keys into registers ahead of use and expand them with byte permutes straight into the
existing INT8 key tile, which adds no shared memory; that is what keeps it off the slow path the
`nvfp4` and `k8v4` kernels take (see [TODO.md](TODO.md), "KV decode falloff").

Measured together on one RTX 3090 (315 W cap), Qwen3.8-27B groupwise-int, 2026-09-23, this build:

| KV profile | Bytes/token | Perplexity | vs `int8` | Decode 4K / 16K / 32K | Prefill at 32K |
|---|---:|---:|---:|---|---:|
| `int8` | 33,792 | 4.343155 | — | 44.6 / 43.1 / 41.5 tok/s | 1,363 tok/s |
| `rk8v4` | 26,112 | 4.347943 | +0.110% | 46.1 / 44.5 / 42.4 tok/s | 1,377 tok/s |
| **`rk4v4`** | **17,920** | 4.352432 | +0.214% | 45.1 / 43.7 / 41.9 tok/s | 1,361 tok/s |
| `nvfp4` | 18,432 | 4.353589 | +0.240% | 43.8 / 40.7 / 36.3 tok/s | 895 tok/s |

Perplexity is `ninfer-perplexity --quick` on `ninfer-ppl-1m-v1` (4096/2048). Decode is
`ninfer_bench -pg P,128`, no speculation, the mean of two interleaved runs for `rk8v4` and `rk4v4`;
the card slows about 0.4 tok/s per run as it heats, and against that drift `rk4v4` sits within ±1% of
`rk8v4` at every depth. `int8` and `nvfp4` are single runs taken last, so they read slightly low.
Qwen3.6-35B-A3B, same protocol: `rk4v4` decodes at 99.5-99.9% of `rk8v4` at 4K-32K (172.5 against
173.4 tok/s at 32K) and 19% faster than `nvfp4` (144.9). With MTP3 and the draft head on the 27B,
`rk4v4` is within about 0.5% of `rk8v4` at 32K; on four real prompts (greedy, 400 tokens) its draft
acceptance was 65.8% against `rk8v4`'s 67.2% and `int8`'s 63.2%. Across the attention op
benchmark's 18 decode/verify shapes it is at a geometric mean of 1.001x `rk8v4`'s time and 2-3x
faster than `nvfp4` at 32K.

**Context, measured with `--kv-capacity auto` and the standard 1 GiB headroom** on this box with a
desktop running (1.8 GiB in use):

| Qwen3.8-27B, one request | `rk8v4` | `rk4v4` |
|---|---:|---:|
| no speculation | 228,032 tokens | **262,144** (native maximum; 1.75 GiB still free) |
| MTP3 + draft head | 182,336 tokens | **262,144** (native maximum) |
| KV payload at 131,072 tokens | 3.19 GiB | **2.19 GiB** |

For a shared pool serving several lanes, the same memory holds 1.46x as many `rk4v4` tokens as
`rk8v4` tokens.

### Choosing a KV format

All seven SM86 KV formats, measured on Qwen3.8-27B. Size and perplexity are what most people weigh;
the decode column is the one that surprises, because the smallest formats are not the fastest.

| KV profile | Bytes/token | KV at 2,048 tokens | Perplexity | vs `bf16` | Decode at 32K depth |
|---|---:|---:|---:|---:|---:|
| `bf16` | 65,536 | 128.00 MiB | 4.342517 | — | 31.93 tok/s |
| `int8` | 33,792 | 66.00 MiB | 4.342425 | −0.0021% | **33.63 tok/s** |
| `fp8` | 33,024 | 64.50 MiB | 4.344724 | +0.0508% | 30.34 tok/s |
| `rk8v4` | 26,112 | 51.00 MiB | 4.346413 | +0.0897% | 33.17 tok/s |
| `k8v4` | 25,728 | 50.25 MiB | 4.347258 | +0.1092% | 28.90 tok/s |
| `nvfp4` | **18,432** | **36.00 MiB** | 4.352201 | +0.2229% | 29.86 tok/s |
| `rk4v4` | 17,920 | 35.00 MiB | see [above](#lloyd-max-4-bit-keys-rk4v4) | +0.214% vs `int8` | ≈ `rk8v4` |

The `rk4v4` row was measured in a later session on a newer build, so it is stated against `int8`
and `rk8v4` from that session rather than mixed into these columns.

Perplexity is `ninfer-perplexity` on the fixed `ninfer-ppl-1m-v1` corpus, `--quick`, context/stride
4096/2048, 261,167 scored tokens — the same corpus and window for every row. Decode is 128 timed
steps on top of a 32,768-token prefill, no speculation, and is the **mean of two independent runs**
because this card is power-capped at 315 W of its 350 W default and the SM clock drifts 1,665–1,755
MHz with temperature; single runs on the 27B vary by up to 5%. Attention re-reads the whole cache
each step, so a format's cost only shows at depth.

**These numbers were all re-measured in September 2026 and several moved.** Three of the six
formats — `fp8`, `nvfp4` and `k8v4` — had their perplexity scored through a data race in the
quantized attention kernels that corrupted every prefill output column except the last. Greedy
generation was unaffected, which is why it went unnoticed. With that fixed, `nvfp4` improved by
0.154% and `fp8` by 0.057%, against a 0.019% drift on the formats the fix did not touch, and all
three now reproduce bit-identically run to run.

Three of these six are worth using:

- **`int8`** is the default for good reason. Its perplexity is indistinguishable from `bf16` — the
  two are within 0.0001 of each other, which is below this harness's own reproducibility — and it
  has the flattest decode curve on the 27B (−4.8% from 4K to 32K).
- **`rk8v4`** is the best all-round choice: 23% smaller than INT8 for +0.09% perplexity, the
  flattest curve on the 35B (−9.5%), and on that model it is also the **fastest** format at every
  depth measured, ahead of INT8 by about 2%.
- **`nvfp4`** buys the most context by a wide margin — 45% smaller than INT8. On the 35B with MTP3
  and the draft head, on a machine running a desktop, it is the only format that still reaches the
  full 262,144 native context: `rk8v4` gets to about 231,000 and INT8 to about 179,000 there.
  Headless, `rk8v4` clears 262,144 as well. It costs about 15% of decode speed at 32K and +0.22%
  perplexity.
- **`rk4v4`** now takes `nvfp4`'s place for context: 31% smaller than `rk8v4` and within 3% of
  `nvfp4`'s size, with better perplexity than `nvfp4` and `rk8v4`'s decode and prefill speed. It
  costs +0.10% perplexity over `rk8v4`. Use it when context or concurrent lanes are what you are
  short of.

`fp8` and `k8v4` are still hard to recommend, but the reason has changed and it is worth stating
precisely rather than as "no niche".

- **`fp8` is no longer dominated on all three axes.** It now has *better* perplexity than `rk8v4`
  — 4.344724 against 4.346413, a real and reproducible 0.039% — which was not true before the race
  was fixed. What that edge costs is 26% more KV memory per token and 8.5% of decode speed at 32K
  depth. A genuine trade, and a bad one for almost everyone, but a trade rather than a strict loss.
- **`k8v4` is dominated.** It is 1.5% smaller than `rk8v4` and pays for it with 13% less decode
  speed at depth, the worst falloff of any format on the 35B (−25.9%), and slightly worse
  perplexity. There is no configuration in which that 1.5% is worth it.

Per-format numbers for the 35B, plus a fit calculator that solves for context and memory, are in
[`docs/config-calculator.html`](docs/config-calculator.html).

At the 1 GiB automatic-sizing boundary INT8 spends 5.40 GiB of KV on 171,648 tokens; rk8v4 spends
5.51 GiB on 226,560.

Decode cost depends on whether you speculate. The packed value plane halves value traffic but adds
an unpack, and those very nearly cancel: without speculation, decode measured 39.07 tok/s on INT8
against 38.91 on rk8v4, and C1 prefill within about 1%. With MTP3, C1 decode falls about 5%, from a
mean 81.61 tok/s across three runs to 77.39, because lower value precision reduces draft acceptance
from 71.27% to 65.86%. That is an accuracy effect on speculation rather than a slower kernel, and it
is the part the finer value group does **not** fix: group-32 recovers 44% of the perplexity penalty
but only about 15% of the acceptance loss, because acceptance turns on exact token agreement rather
than on mean error.

Use `rk8v4` when context is the binding constraint. Use `--kv-dtype int8`, which reaches 171,648
tokens at the 1 GiB headroom boundary, when decode throughput under speculation matters more.

### Qwen3.8 vision

The same Qwen3.8 artifact supports images. Start the server with `--vision`, MTP3, INT8 KV, and a
32K maximum context. `run.bat qwen38-27b` already serves images (vision in overlay residency);
this plain profile is `NINFER_SPEC=mtp`, `NINFER_KV_DTYPE=int8` and `NINFER_CONTEXT=32768`.

A 1,920×1,080 image expanded to 2,074 prompt tokens and was read correctly. Measured TTFT was
3.29 seconds, decode reached 98.1 tok/s, MTP acceptance was 96.7%, and startup retained 2.16 GiB
free VRAM. The artifact also declares multi-image and video support; this release test directly
validated a single image.

## Qwen3.6-35B-A3B RTX 3090 results

Measured with the compact 20.84 GiB 35B-A3B artifact, a 4K shared INT8 group-64 paged KV pool,
CUDA Graphs, MTP3, greedy decoding, and no competing GPU workload:

| Concurrent requests | 128 output tokens each | Observed VRAM |
|---:|---:|---:|
| 1 | 162.7 aggregate tok/s | 22,427 MiB |
| 2 | 267.9 aggregate tok/s | 22,743 MiB |
| 4 | 366.2 aggregate tok/s | 23,377 MiB |
| 6 | 383.4 aggregate tok/s | 24,038 MiB |
| 8 | rejected at startup | about 503 MiB over the safe reservation limit |

A longer 512-token-per-request check reached **286.8 tok/s at C1** and **399.1 aggregate tok/s at
C2**. These short-prompt measurements include request-level timing and are not directly comparable
to v0.3.1's 1,500-token adaptive prompt-lookup benchmark.

Compatible-prefix reuse was validated end to end: a repeated 26-token prompt reused 24 tokens,
reducing measured prefill from 371 ms to 10 ms.

### Qwen3.6-35B vision

The compact 35B artifact includes its vision encoder and accepts images through the same OpenAI-
compatible API. Start the server with `--vision` and leave speculative decoding disabled. The
`run.bat qwen36-35b-a3b` already serves images; for this profile set `NINFER_SPEC=none`.

The safe RTX 3090 profile is **one request, 32K maximum context, INT8 KV, vision enabled, and MTP
disabled**. A current v0.6 test processed three 1,920×1,080 images correctly. Each image expanded
to a 2,081-token prompt; engine TTFT was 3.76–4.07 seconds, decode was about 159 tok/s, and peak
VRAM was 23,944 MiB. This leaves little room for another GPU workload.

MTP is intentionally off for this profile. At 32K, speculative recurrent state would exceed the
3090 memory budget; KV compression alone does not recover enough memory. Text-only 35B profiles can
still use MTP3 as documented above.

## Several GPUs: pipeline stages (`--devices A,B,...`)

`--devices 0,1` splits the model's layers into one pipeline stage per GPU. Each stage owns its
layers whole: weights, the KV cache of its attention layers, the recurrent state of its GDN layers
and the scratch it runs in. The point is memory. A model that does not fit one card, or a context
that does not, spreads across several, and every card's memory is usable for KV.

```
ninfer-serve model.ninfer --devices 0,1
ninfer model.ninfer --devices 0,1,2 --stage-layers 20,22,22 --prompt "..."
```

- **`--stage-layers A,B,...`** sets the layers per stage. Without it the split follows each
  device's free memory, so the first GPU, which also carries the embedding and head, takes fewer
  layers and the most KV cache fits on every card at once.
- **The first GPU also holds the embedding, the output head and the round state.** The last stage
  sends the residual back to it, one extra hop per forward pass.
- **It is a memory feature, not a speed feature.** The stages run in sequence and each reads only
  its own weights, so a single stream decodes about as fast as one GPU, minus the boundary hops.
- **Linux only for real multi-GPU.** Repeating one id (`--devices 0,0`) puts several stages on one
  card, saves no memory, and exercises the whole stage path; it is how the path is tested without a
  second GPU, and it works on Windows too.
- **Works with a split:** the context cache and prefix reuse, CUDA graphs, and MTP. **Not yet:**
  DFlash/DFlash2 and vision, which are refused at startup with a message saying so.
- **Boundary transfers stage through pinned host memory.** Peer access is not needed, and no
  consumer PCIe pair measured so far offers it. Measured with `tools/tp_probe.cu` on rented 2x A4000
  and 2x 3090 PCIe boxes, a staged transfer took about 0.03 ms at a decode-sized payload and several
  milliseconds at a prefill-chunk-sized one, depending on the slot's link width.

Measured on two rented Linux boxes (Qwen3.6-27B, int8 KV, no peer access on either): on 2x RTX 3090
(PCIe 3.0 x16) greedy output is byte-identical to one card, decode is 48.5 tok/s against 46.9 on one
card (105.3 against 100.2 with MTP3), prefill is unchanged, and `--kv-capacity auto` resolves the
full 262,144-token context that one 24 GB card refuses. On 2x RTX A4000 the 27B runs at 262,144
tokens with 24.1 tok/s decode (52.6 with MTP3) and 825 tok/s prefill. The tables, the cases checked
and the design are in `docs/maintainer/pipeline-parallel-plan.md`.

## Capabilities

- Native SM86 CLI and server applications for Linux and Windows.
- A prebuilt Windows archive with tested launchers.
- OpenAI Chat Completions, Responses, and Anthropic-compatible APIs.
- Structured JSON output (`response_format` / `text.format` / `output_config.format`), enforced
  during sampling under every speculative backend; see [docs/serving.md](docs/serving.md#structured-output).
- ReplaySSM and MTP3 for higher throughput without exceeding 24 GB VRAM.
- `low`, `medium`, and `xhigh` reasoning modes.
- Qwen3.8 image understanding with ReplaySSM and MTP3.
- Prefix reuse for faster repeated or shared prompts.
- Qwen3.6-35B image understanding with a guarded 32K profile.
- Windows one-user and eight-user launchers with safe tested defaults.

## Supported artifacts

| Model | Artifact | Size | Notes |
|---|---|---:|---|
| Qwen3.6-35B-A3B | [pinned v3 artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6) | 21.23 GiB | **Recommended; fetched by `download-model qwen36-35b-a3b`. Carries the DFlash bundle for `--spec dflash`** |
| Qwen3.6-27B | [pinned v3 artifact](https://huggingface.co/neroued/Qwen3.6-27B-NInfer/tree/3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829) | 16.29 GiB | Supported with more runtime headroom |
| **Qwen3.8-27B** | [pinned v3 artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/tree/1cbd84e7221e51186bd7f093a149912d2489625b) | 19.03 GiB | **Validated at C1, C2, C4 and C8/MTP3 with ReplaySSM. Carries the DFlash2 bundle for `--spec dflash2`** |

This release reads only the v3 `.ninfer` container. v1 and v2 files from earlier releases are
refused at load. Upgrade an existing official v2 file in place of re-downloading it:

```text
python tools/upgrade_ninfer_v2_to_v3.py models/qwen3_8_27b.ninfer models/qwen3_8_27b.v3.ninfer
```

The upgrade needs only the Python standard library, runs on Windows and Linux, keeps every weight
byte, and installs the maintained chat template. The published RTX 3090 figures in this repository
were measured against the v2 revisions (`18dfc887` for Qwen3.8-27B, `c8b8c1c0`/`560f227e` for
Qwen3.6-35B-A3B); their weight bytes are unchanged in v3.

**Optional bundles cost nothing in VRAM unless selected**: the DFlash and DFlash2 weights are bound
only with `--spec dflash`/`--spec dflash2`, so the resident-weight figures for other profiles
still hold.

## Models and platform support

Linux users build the applications from source or use the Docker image. Windows users can use the
prebuilt archive, which includes the applications and required DLLs. Both platforms require an
RTX 3090 or RTX 3090 Ti and a recent NVIDIA driver.

Download the
[pinned Qwen3.8 v3 artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/tree/1cbd84e7221e51186bd7f093a149912d2489625b)
as `models/qwen3_8_27b.ninfer`, or run `download-model.sh qwen38-27b` (`.bat` on Windows), which verifies size and
SHA-256 before putting the file in place. The 27B numbers in this repository were measured against
the v2 revision `18dfc887`, whose weights this v3 file carries unchanged.

**Every downloader shipped here is pinned and verified, with one deliberate exception.** The Nix
app `download-qwen36-35b-v2` tracks upstream `main`, which is what it is for — trying a newer
artifact than the measured one — and that is exactly why it cannot verify anything: a moving
reference has no size or hash to state, and no partial file can be safely resumed against it. It
writes to its own filename, refuses to resume, and is not the artifact the tests or the published
figures use. Prefer a pinned downloader unless you specifically want a newer upstream build.

For Qwen3.6-35B-A3B, download the
[pinned v3 artifact](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/tree/ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6)
for DFlash support, or run `download-model.sh qwen36-35b-a3b` (`.bat` on Windows) instead, which verifies size and
SHA-256 before putting the file in place. A load error about the container version means either an
older executable reading a v3 file, or this executable reading a v1/v2 file that needs
`tools/upgrade_ninfer_v2_to_v3.py`.

Developers can build from source on Windows or Linux. Windows uses Visual Studio 2022 and vcpkg.
Linux uses GCC 13 with system packages or the pinned vcpkg manifest. Both builds require CUDA 12.8
or newer and CMake 3.28 or newer.

See the [Windows build guide](docs/rtx-3090-windows.md) or the
[Linux build guide](docs/rtx-3090-linux.md). Ordinary Windows release users do not need these tools.

## Qwen3.8 reasoning effort

Qwen3.8-27B supports distinct reasoning-effort modes. `medium` uses the model's normal thinking
prompt. `xhigh` injects the checkpoint's extended deliberation instruction, asking it to validate
assumptions and consider alternatives. This is a real prompt-template change, not a sampling alias.

| Value | Qwen3.8 behavior |
|---|---|
| `none` | Disable thinking |
| `low` | Keep reasoning brief and focused |
| `medium` | Use normal Qwen3.8 thinking |
| `xhigh` | Use extended deliberation and verification |

OpenAI Chat Completions accepts a top-level `reasoning_effort` field:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Solve this carefully..."}],
  "reasoning_effort": "xhigh",
  "max_tokens": 4096
}
```

OpenAI Responses uses `"reasoning": {"effort": "xhigh"}`. Anthropic Messages uses
`"output_config": {"effort": "xhigh"}`. For the native CLI, pass
`--reasoning-effort low|medium|xhigh`; use `--no-thinking` instead of an effort to disable
reasoning. Chat Completions returns hidden reasoning separately as `message.reasoning_content`.

## Serving APIs

The server supports:

- OpenAI Chat Completions;
- OpenAI Responses Core with streaming and local continuation state;
- Anthropic Messages;
- compatible-prefix reuse;
- prompt-rendered function tools and parsed tool calls;
- bounded pending-request admission and JSONL request logs.

See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

Qwen3.8-27B artifacts carrying the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 4` (see [CLI usage](docs/cli.md) for why four rather than seven),
with draft counts 1..15 and either full or optimized
proposal heads. **DFlash2 runs with `--vision`**, verified on a 3090 against the committed
`image_chart` fixture: target verification carries its own continuation RoPE position, so a
multimodal row keeps its per-sequence `rope_delta` through verification rather than being
re-indexed by DFlash's logical positions. On that fixture it accepts 85.7% of drafts (7.00
tokens/round) and produces byte-identical output to the non-speculative vision run. `--spec
dflash` (v1) is a 35B-A3B backend and is refused by a 27B artifact, with or without vision.

## How cohort batching works

The C number is the maximum number of requests NInfer can run together. C1 favors one interactive
user; C8 can combine up to eight active requests into each GPU step for much higher total output.

Follow-up requests do not need to arrive at the same instant. When a running request finishes, the
next waiting request can join at a safe generation boundary. Empty or finished lanes are skipped,
so a C8 server also works normally with only one, two, or four active users.

This is deliberately more bounded than datacenter-style dynamic batching. The maximum number of
users and GPU memory are chosen when the server starts. In return, memory use stays predictable on
a 24 GB card and the server can reuse fast CUDA Graphs instead of rebuilding work continuously.

## Current limits

- One process owns one model on one RTX 3090.
- Concurrency is fixed at startup and limited to 1-8 by the API; compact 35B fits C1-C6 and
  Qwen3.8-27B fits C8/8K with MTP3 through ReplaySSM.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- Multi-GPU execution is pipeline stages only (no tensor parallelism), Linux only, and does not yet cover speculative decoding or vision. No CPU/GPU weight offload.
- Tool calls are returned to the client but are not executed by NInfer.
- NVFP4 A4, FP8 A8, and TMA kernels require Blackwell and are unavailable on SM86. FP8 and NVFP4
  weights are admitted through their A16 dequantizing routes.
- The paged runtime exposes seven KV formats on SM86 — `bf16`, `int8` group-64, row-scaled FP8 E4M3
  `fp8`, RotorQuant `rk8v4`, Lloyd-Max `rk4v4`, `k8v4` and `nvfp4`. INT8 remains the quality default and the rest are
  opt-in; [`docs/config-calculator.html`](docs/config-calculator.html) has the measured size,
  speed and perplexity of each. Note that this is KV storage only: NVFP4 A4 and FP8 A8 *weight and
  activation* kernels still require Blackwell and are unavailable here.

## Validation

The v0.6.0 Windows gate covered Qwen3.8 generation, materialization, request memory, admission,
paged KV, prefix reuse, speculative rounds, and SM86 W8 Linear paths.

The v0.6.1 Linux source gate completed all 245 Docker compile and link steps with CUDA 13.1 on
Ubuntu 24.04. Both Linux applications returned their `--help` output with GPU access enabled.
A real-artifact Linux generation and Linux performance qualification remain open.

## Upstream

NInfer-3090 is derived from [Neroued/ninfer](https://github.com/Neroued/ninfer). The upstream project
targets RTX 5090/`sm_120a`; this fork carries the Windows and Linux SM86 compatibility layer,
compact 35B artifact support, and RTX 3090-specific schedules and memory planning.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the complete, maintained credit list.

- [airtonix](https://github.com/airtonix) added Linux and Docker build and release support in
  [PR #1](https://github.com/Don-Chad/ninfer-3090/pull/1).
- [ColeWheatley](https://github.com/ColeWheatley) contributed SM86 runtime-count/GDN residency
  fixes, ECC diagnostics, and the GeForce-safe Docker fix in
  [PR #7](https://github.com/Don-Chad/ninfer-3090/pull/7).
- [justinlime](https://github.com/justinlime) added NixOS build support in
  [PR #5](https://github.com/Don-Chad/ninfer-3090/pull/5).
- [sry9681](https://github.com/sry9681) contributed the device-wide GPU-memory startup fix in
  [PR #6](https://github.com/Don-Chad/ninfer-3090/pull/6).
- [iamwavecut](https://github.com/iamwavecut) contributed the swscale destination-alignment
  JPEG safety fix in [PR #11](https://github.com/Don-Chad/ninfer-3090/pull/11).
- [nasedkinpv](https://github.com/nasedkinpv) contributed the tool-call parser crash fix in
  [PR #12](https://github.com/Don-Chad/ninfer-3090/pull/12).
- [wmehanna](https://github.com/wmehanna) contributed in-place system-turn rendering for
  Claude Code prefix reuse in [PR #13](https://github.com/Don-Chad/ninfer-3090/pull/13).

## Contributing

Please read the [Pull Request Policy](PR_POLICY.md) before opening an issue or pull request.
It explains how to keep changes focused and how to document correctness, performance, VRAM, and
compatibility evidence.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
