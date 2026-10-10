# NInfer-3090 v0.12.0

**35 merged pull requests in eight days.** Since v0.11.0 this fork gained real multi-GPU inference
across separate machines, a new KV-cache format that matches NVFP4's size at RK8V4's speed,
grammar-constrained structured JSON output under every speculative-decode backend, an engine that
recovers from a worker fault instead of locking up, Prometheus metrics, on-disk session
persistence, per-request prompt grafts that now run under speculation, and a first look at a
second GPU family (the CMP 170HX). Below is every change with its measurement; nothing here is
asserted without a number next to it, and the things that were tried and rejected are recorded
too, in the same spirit as v0.11.0.

## Who this is for

- **Anyone who owns more than one GPU.** Pipeline parallelism is real now: verified byte-identical
  to single-card output on rented 2x RTX 3090 and 2x RTX A4000 hardware, including a model that
  does not fit on one A4000 at all.
- **Anyone who wants more context or faster decode without a quality cliff.** `rk4v4` KV reaches
  the 27B's native 262,144-token context at `rk8v4`'s speed, in NVFP4's memory footprint.
- **Anyone building on top of the API.** Structured JSON output, session persistence via `/slots`,
  Prometheus metrics, and an engine that survives a worker-level fault instead of returning 503
  until someone restarts it.
- **Agent and long-context users.** `--auto-long-anchors` gives Chat Completions and Anthropic
  requests the same "restore below a mid-history edit" capability that only the Responses API had
  before.
- **Anyone curious about other Ampere-class silicon.** The CMP 170HX (an unlocked GA100 mining
  card) runs Qwen3.8-27B correctly today, at roughly 3090 speed, completely untuned.

## Multi-GPU: real pipeline parallelism, verified on real hardware (#110)

The model's layers now split into pipeline stages across up to eight GPUs on Linux
(`--devices A,B,...`), each stage owning its own weights, KV cache and GDN state. This replaces
the old expert-offload split, which only ever moved a layer's MLP to a second card and had never
been verified on real multi-GPU hardware. CUDA graphs, MTP, and the context cache/prefix reuse all
work across stages; vision and DFlash/DFlash2 under a split are not supported yet and refuse with a
clear startup message.

Verified on rented hardware (design and the full measurement set: `docs/maintainer/pipeline-parallel-plan.md`):

| | 2x RTX 3090 (PCIe x16) | 2x RTX A4000 |
|---|---|---|
| Stage equivalence | byte-identical to one card, every row (graphs/eager, staged transport, uneven split, MTP, prefix reuse, 3 stages) | model doesn't fit one 16 GB card; split-invariant, matches the 3090 output |
| Decode | 48.5 tok/s split vs 46.9 one card (MTP3: 105.3 vs 100.2) | 24.1 tok/s (MTP3: 52.6) |
| Prefill | unchanged | 825 tok/s |
| `--kv-capacity auto` @ 262,144 ctx | one card refuses; split fits | fits |

The local single-3090 A/B against master showed no decode/prefill regression and identical
perplexity (`--quick --kv-dtype int8`, 4.552656 both). **Not in this release:** tensor parallelism
(design and transport-only measurements exist), prefill micro-chunk overlap between stages, and
vision or DFlash under a split.

## New KV format: `rk4v4` — NVFP4's size, `rk8v4`'s speed (#115)

`--kv-dtype rk4v4` rotates keys and stores them as 4-bit indices into a Gaussian Lloyd-Max
codebook (one FP16 scale per 64 dimensions), paired with `rk8v4`'s packed int4 values — the
TurboQuant idea without its 1-bit residual stage. It is now what the `tuned` launcher profiles use.

| Qwen3.8-27B, RTX 3090, same session | `int8` | `rk8v4` | **`rk4v4`** | `nvfp4` |
|---|---:|---:|---:|---:|
| KV bytes/token | 33,792 | 26,112 | **17,920** | 18,432 |
| Perplexity (`--quick`) | 4.343155 | +0.110% | **+0.214%** | +0.240% |
| Decode at 32K, no spec | 41.5 | 42.4 | **41.9** | 36.3 |
| Prefill at 32K | 1,363 | 1,377 | **1,361** | 895 |
| Largest context, no spec | – | 228,032 | **262,144 (native max)** | – |
| Largest context, MTP3 + draft head | – | 182,336 | **262,144 (native max)** | – |

Within ±1% of `rk8v4` decode speed at 4K-32K, 12-19% faster than `nvfp4`, and the same memory now
holds 1.46x the tokens `rk8v4` does (2.19 GiB vs 3.19 GiB at a 131K pool). Every `tuned` launcher
profile now starts at higher context than before purely from the memory this frees:

| profile | before (`rk8v4`) | Windows now | Linux now |
|---|---|---|---|
| 27B DFlash2 (default) | 131,072, 1 lane | 172,032 (up to 180,224) | 262,144* |
| 27B `NINFER_SPEC=mtp` | 163,840 x1 / 212,992 x2 | 262,144, 2 lanes | 262,144, 2 lanes |
| 35B MTP3 | 147,456 x1 / 262,144 x2 | 262,144, 2 lanes (3 start) | 262,144, 3 lanes |

\* Extrapolated for a headless card; the launcher's automatic step-down catches a card that falls
short. Full derivation, the attempts that were slower before landing on the shipped scheduling, and
the routing changes that came along with it: PR #115.

## Structured JSON output, under every speculation mode (#128, hardened by #143)

Grammar-constrained JSON is now available through all three APIs: OpenAI Chat `response_format`
(`json_object`, `json_schema`), Responses `text.format`, and Anthropic `output_config.format`.
[XGrammar](https://github.com/mlc-ai/xgrammar) v0.2.5.post1 is vendored as a local static library.
Every sampled position gets its own mask, including every speculative-verification column — a
draft the grammar forbids is always rejected — under MTP, DFlash, DFlash2, and no speculation
alike.

| Mode | Unconstrained | `json_object` | Change |
|---|---|---|---|
| DFlash2 K=7 | 187.3 tok/s | 183.3 tok/s | −2% |
| MTP3 + lookup | 96.6 tok/s | 90.8 tok/s | −6% |
| No speculation | 46.7 tok/s | 45.9 tok/s | −2% |

First structured request after startup compiles its grammar in about 190 ms (a repeated schema
adds about 6 ms); unconstrained requests cost at most ~0.5% from the extra CUDA-graph segmentation
DFlash2 needs to build masks mid-round. **Follow-up fix (#143):** a token whose decoded text
crossed a client's stop string (e.g. `}STOPjunk` under stop string `STOP`) could be masked out even
though the client would only ever see the safe prefix `}`. This was a soft degradation — the
sampler picked a different, still-valid token — never a crash, and is now closed with the same
prefix logic the preview path already used. Known remaining gap, tracked separately (#141): a
same-token caller stop can still be masked out specifically during speculative mask construction.

## Reliability: the engine survives a worker fault instead of latching (#136, #121, #122)

Previously, any exception reaching a worker's catch-all failed every request, latched the engine
unavailable, and kept `/health` at 503 until a restart. Now the worker synchronizes the device,
runs the same cleanup the latch path already used, and — if that cleanup verifiably leaves the
Program empty — fails only the requests actually running, keeps the queue, and carries on. It still
latches after three recoveries with no successful request in between, so a persistent fault can't
spin forever, and a real CUDA error still aborts the process rather than being "recovered" from.
The new `engine_recoveries` counter is in the request log's context statistics.

Two of the three known triggers for that latch are separately fixed at the root:

- **#121**: a vision KV loan could take pages an in-flight request had already reserved but not yet
  materialized, tripping the pool's own invariant check and failing every request on that engine
  core.
- **#122**: an entitlement-counting bug (ported from a sibling fork, same code, same bug) could
  latch the engine when a shared long anchor was demoted to host storage. Also fixed alongside it:
  a cancelled Anthropic stream was logging every client disconnect as a 500 instead of a 499.

## Observability and operations

- **`GET /metrics` in Prometheus format (#125).** `llamacpp:`-prefixed series match llama.cpp's
  existing metric names, so current dashboards and scrapers work unchanged; `ninfer:`-prefixed
  series add completed/failed requests, prefix-cache hit tokens, and speculative draft/accept
  totals. Requires the API key when one is configured, like `/v1/load`.
- **`GET /props` (#127).** A read-only, llama.cpp-shaped snapshot: context size, default generation
  settings, total slots, model alias/path, modalities.
- **Session persistence through `/slots` (#131).** `GET /slots` reports each private context-cache
  cell's retained session, depth and digest. `--slot-save-path DIR` enables
  `POST /slots/{id}?action=save|restore|erase`; `--auto-save-evicted` writes a bound session to
  disk before an involuntary eviction destroys it. A restored session continues with **token-identical**
  greedy output to a never-evicted control. A 39-token session saves to 295 MiB (two
  ~150 MB recurrent-state images) in about 0.3 s each way. *(Superseded: the `POST /slots` actions and
  the `--slot-save-path` / `--auto-save-evicted` flags were retired in a later release; use
  `--context-store DIR`, see [docs/serving.md](docs/serving.md#context-store).)*
- **`best_reuse_prompt_tokens` in `request_done.materialization` (#124).** Distinguishes "no
  candidate offered any reuse" (a prefix-matching problem) from "a candidate did, and the planner
  priced it out" (a policy question) — previously both looked identical from the log.
- **`--auto-long-anchors N` (#123).** Chat Completions and Anthropic requests can now restore below
  a mid-history edit the same way Responses-with-`prompt_cache_key` already could. Defaults to the
  existing `--max-long-anchors-per-continuation` cap (2); `0` disables it.

## Per-request prompt grafts, now under speculative decoding (#114, #137)

A phantom-KV graft file loaded at startup and bound to a name can be injected as a hidden
conversation prefix on a per-request basis, in either of two modes: token replay (exact KV state,
replayed like a normal prefill) or direct KV injection (pre-computed KV tensors written straight
into a pinned shared-prefix slot, skipping the prefill entirely). The shared prefix is cached once,
not once per conversation. **#137 removes the startup refusal for direct grafts under
`--spec`:** MTP and DFlash2 now both serve graft requests, verified at 4 prompts x 160 tokens with
no measurable acceptance-rate cost from the graft's zero-filled draft context in that sample.

## Performance

- **Int8-family prompt kernel, unmasked interior key blocks (#140), +10-13% attention op / +4%
  end-to-end.** `prompt_i8.cuh` (serving `int8`, `rk8v4`, `rk4v4`) now skips the causal test, score
  masking, softmax zero-selects and V-dequant bound test for key blocks that lie wholly below the
  query tile's first row — only blocks crossing the diagonal keep the masked path. Kernel
  benchmark: −10.2% to −13.5% depending on KV format and context length (negative = faster, this is
  a cost-reduction measurement); end to end, `ninfer_bench pp65536` on Qwen3.8-27B went from
  1166.5/1173.5 to 1215.5/1213.7 tok/s, **+4.0%**.
- **GDN: two-stage kernels replace the chunked path (#117).** Backported from upstream. Fuses the
  old WY/WU + state-passing + output pipeline into a `prepare` stage (Q/K L2-norm and control-matrix
  prep) and a `recurrence` stage (persistent FP32 state, BF16 MMA with FP32 accumulation). Chunk
  size drops from 64 to 16, and every prefill of 16+ tokens now takes the chunked route directly.
- **BF16 causal attention reorganized and its CUDA graphs stabilized (#119).** Partitions are now
  derived on-device from the live row length instead of fixed at capture time, so one captured
  graph stays update-compatible across ragged batches. This is a structural port of two of eight
  planned upstream commits; the other six are blocked because this fork's FP8/NVFP4/K8V4 attention
  kernels are a different design (dequantize-to-BF16, no native tensor-core path on sm_86) rather
  than a reorganized version of upstream's native-FP8 kernels — recorded as a deliberate deferral,
  not an oversight.
- **MTP draft window raised from 5 to 15 (#129).** The default stays at 3; K up to 15 is now
  available and needed a real bug fixed first — MTP's graph grouping copied the 35B's routing rule
  and put two attention-kernel-incompatible ranges in one CUDA graph group at K=15, failing at
  startup. The attention op now reports its own kernel count instead.

## Serving API

- **Context limit under three field names (#118).** `GET /v1/models` now reports `max_model_len`
  (vLLM/llama.cpp), `context_window` (Anthropic), and `context_length` (OpenRouter/Ollama), since
  OpenAI's own spec has no single standard name.
- **Output limit sized per lane, not a fixed 8192 (#127).** A request that omits its token limit
  (and no `--default-max-tokens` override is set) now gets the largest output that still lets every
  configured lane be admitted at once — computed from the same reservation formula the request
  planner already uses, MTP/DFlash included. With one lane, that is the whole remaining context.
- **`--reasoning-effort` server default (#127)**, used only when a thinking request omits its own
  effort.
- **Anthropic system messages at any history position (#127).** `[user, system, user]`, a system
  message after an assistant turn, and a trailing system message are now accepted and rendered in
  place instead of rejected, preserving prefix reuse.
- **`/v1/models` advertises modalities (#127, #118).** `architecture.input_modalities` /
  `output_modalities`, with `image`/`video` listed only when `--vision` is on.

## A second GPU family: the CMP 170HX (#109, handover / not yet tuned)

An unlocked NVIDIA CMP 170HX (GA100 silicon, 64 GB, sold as a mining card) is added as an
**unmeasured compatibility target** (`CMAKE_CUDA_ARCHITECTURES=80`). Qwen3.8-27B loads and
generates correct text on real rented hardware at roughly 3090-like speed with **completely
untuned** routes: 45 tok/s decode, 1.5k tok/s default prefill / 2.5k with the cuBLAS route. The
card is on paper 2.8x the BF16 tensor throughput and 1.9x the memory bandwidth of a 3090, so this
is a floor, not a ceiling — the remaining work is a tuning pass, recorded as a handover for
whoever picks it up next, not shipped as a finished target.

## Build, tooling and portability

- **Tests and microbenchmarks now link into one bundle executable each (#112).** Every test or
  benchmark that linked the Op library used to embed its own full copy of the kernel image
  (~450 MB on sm_86); a full build was spending on the order of 80 GB on roughly 200 near-identical
  copies. `ninfer_tests` (135 programs) is now 467 MB total, `ninfer_benches` (66 programs) 525 MB.
  Each CTest entry still runs its program in its own process, so exit codes, globals and CUDA
  context behavior are unchanged.
- **Parallel, resumable model downloads (#111).** `download-model` uses `aria2c` (16 ranges) when
  it's on `PATH`, falling back to `curl` — single-stream `curl` against these artifacts had been
  seen throttling to ~1 MB/s or stalling outright. A CI regression this introduced (the hosted
  runner has `aria2c` preinstalled, so the download-fixture tests were silently downloading real
  multi-GB files instead of exercising the stub, costing ~11 minutes every run) is fixed in #120.
- **Windows build is warning-clean (#147).** A full clean rebuild under MSVC v143 surfaced 274
  warning lines across 10 sites in first-party code (unreachable-loop dead code from an
  early-return pattern, a few genuinely-unused declarations, one real signed/unsigned mismatch, and
  a link flag being forwarded somewhere it didn't belong); all are fixed with no behavior change.
  A related portability fix (#146): MSVC doesn't treat `std::sqrt` as usable in a `constexpr`
  initializer the way GCC does, which was blocking a from-scratch Windows test build.
- **`run.bat` no longer crashes when a graft file is present (#149).** A batch-quoting bug in the
  startup banner broke on the normal case, not an edge case — anyone with a graft file configured
  hit a syntax error instead of a running server.
- **GCC build break fixed (#142).** An unqualified `friend class Program;` inside a nested
  namespace declared a *different* class than intended; MSVC silently accepted the mistake, GCC
  correctly rejected it, breaking the Linux build. One qualified line fixes it.

## Vision fix (#145)

Supersedes an interim fix (#138) that correctly closed a premature-KV-loan race but broke two other
things in the process: vision encode/decode overlap was silently and permanently disabled (the
opportunistic overlay tier could never engage again), and an internal-error log path started
leaking raw exception text — including internal paths and CUDA error strings — to a field meant
only for the separate measurement writer. This release keeps the race fix and relocates the call
site instead of deleting it, restoring the overlap, and gives internal failures a safe phase-derived
error code instead of leaking `machine_message`.

## What we tried and rejected, on purpose

- **`rk2v4-e8`, a 2-bit E8 key coding**, was simulated before being built: **+3.08%** perplexity on
  Qwen3.8-27B against the `int8` baseline (every domain +2.2% to +3.9%), about 14x `rk4v4`'s own
  +0.214% cost for keys alone. Not built; recorded in `docs/performance.md` so it isn't re-proposed.
- **Four kernel and build ideas from a sibling sm_89 fork**, measured directly on sm_86 rather than
  assumed to transfer: disabling relocatable device code (inside noise here, their card measured
  +2.7-4.9%), SM-count-based schedule sizing (23-33% faster in isolation, −0.2% to −1.2% end to end
  because DFlash2's captured graphs already pick a kernel variant), and two others. Recorded next to
  each affected constant so the experiments aren't repeated.
- **Six of eight planned upstream causal-attention reorg commits**, deferred rather than ported: the
  reorg assumes native-tensor-core FP8/NVFP4/K8V4 kernels, and this fork's sm_86 kernels for those
  precisions dequantize to BF16 instead (no native FP8 path exists on Ampere/early Ada). Porting the
  reorg would mean rewriting those kernels' actual math, not just their file layout.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench` | throughput benchmark against the public Engine route |
| `run.bat` / `run.sh` | serving profiles: `run <model> [profile]` |
| `download-model.bat` / `.sh` | pinned, resumable, checksum-verified model downloads |
| `README.md`, `SHA256SUMS.txt`, `LICENSE`, `VERSION` | the guide, checksums for every file, licence |

The Windows archive also carries the DLLs it needs: FFmpeg, libcurl, zlib, and NVIDIA's cuBLAS
runtime (`cublas64_12.dll` and `cublasLt64_12.dll`, redistributed under the CUDA Toolkit EULA,
included as `NVIDIA-CUDA-EULA.txt`), so no CUDA Toolkit is required. Linux links cuBLAS, FFmpeg and
glibc dynamically; see `docs/release-archive-linux.md` for the exact runtime requirements (built on
Ubuntu 24.04, glibc 2.38+). Model artifacts are not included: `download-model` fetches
Qwen3.6-35B-A3B (21 GB), Qwen3.8-27B (19 GB, the DFlash2 bundle, which also carries the MTP
weights), or Qwen3.6-27B (16 GB).

## Known issues

- **Tensor parallelism is not built.** Only pipeline parallelism; see "Multi-GPU" above.
- **Vision and DFlash/DFlash2 refuse a multi-GPU split.** Pipeline parallelism works with MTP and
  no speculation only, for now.
- **The same-token caller stop under speculative mask construction** (#141) can still mask out the
  one token that would have produced an exact same-token stop; the sampler picks a different
  grammar-valid token instead, so this is a soft degradation, not a crash.
- **The CMP 170HX target is untuned.** It runs correctly at roughly 3090 speed on hardware that is
  on paper substantially faster; treat `sm_80` as a compatibility floor, not a recommendation.
- **NVFP4 and K8V4 KV still require a Blackwell GPU** and are refused on `sm_86` with a clear
  message; this is unchanged from v0.11.0.
- **Speculative decoding is still not bit-identical to width-1 greedy decode** (unchanged from
  v0.11.0): verification evaluates several columns in one pass, so a near-tie argmax can flip.

## Upgrading

Nothing here requires a model re-download or a config rewrite. A few defaults moved, all in a
direction that gives you more of what you were already asking for:

- **The `tuned` launcher profiles now default to `--kv-dtype rk4v4`** instead of `rk8v4`, which is
  why every profile's starting context grew (see the KV-format table above). Pass
  `NINFER_KV_DTYPE=rk8v4` to keep the old behavior.
- **An omitted output token limit is no longer capped at a fixed 8192.** It now sizes to the
  largest completion that still admits every configured lane. If you were relying on the old fixed
  cap as an implicit safety limit, set `--default-max-tokens` explicitly.
- **`--auto-long-anchors` is on by default** (capped at 2, matching the existing
  `--max-long-anchors-per-continuation`). Pass `--auto-long-anchors 0` to disable it.

Everything else — multi-GPU, structured output, `/metrics`, `/slots`, prompt grafts, the sm_80
target — is opt-in and off unless you ask for it.
