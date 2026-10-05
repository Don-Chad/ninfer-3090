# NInfer-3090 v0.13.0

**A new Qwen3.8-27B: 1.46 GB smaller, measurably closer to a Q8_0 reference, and faster under
speculative decoding.** Since v0.12.0 this fork converted the 27B again with an importance-weighted
scale search, wrote the kernels the new layout needed, and made the result the default 27B. Around
it: per-request `thinking_budget` on every endpoint plus a fix for a 400 that long agent sessions hit
every turn, prompt grafts that finally use the context cache, and a pass over the launchers and docs.
11 merged pull requests, 65 commits, 129 files. Every number below says what it was measured against;
the places where the result was a tie or a loss are listed too, in the same spirit as v0.12.0.

## Who this is for

- **Anyone running Qwen3.8-27B.** Re-download the model (`download-model qwen38-27b`, 19 GB) and
  use this release's binaries. It is the same model in the same container with a better weight
  encoding.
- **Anyone driving long agent sessions** (Claude Code, pi, Cline and similar). The
  `thinking_budget_capacity_insufficient` 400 that appeared once a conversation grew into the
  boundary window is gone, and grafted requests no longer re-prefill the whole history every turn.
- **Anyone who scripts the launchers.** A custom chat template is one environment variable now.

## The new default model: imatrix-searched Qwen3.8-27B (#159)

Published at [`WarlaxZ/Qwen3.8-27B-NInfer-3090`](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NInfer-3090),
pinned in the downloaders: 18,982,458,624 bytes, against 20.44 GB for upstream's artifact. **It needs
this release's binaries**; older builds refuse it at startup. `run.bat` and `run.sh` no longer pass
the load-time transcodes (`--embedding-q4`, `--lm-head-q6`) the upstream file needed, and their
launcher checks now refuse both on the 27B.

What changed in the conversion:

- **`grouped_search` encoder.** An imatrix-weighted FP16 scale search, never worse than
  round-to-nearest, with signed scales (the tensor-formats contract now allows them).
  `tools/convert/imatrix.py` imports a llama.cpp imatrix.
- **A held-out evaluation corpus and a llama.cpp KLD harness.** 12 streams (Wikipedia en/zh, arXiv,
  new GitHub code, our own code, synthetic chat). The harness exports NInfer encodings bit-exactly
  into Q8_0 GGUF blocks so they can be scored against Unsloth's Q8_0 with the same tooling.
- **The full table behind the choice** (#163): 21 NInfer variants plus UD-Q4_K_XL and UD-Q5_K_XL
  in `docs/maintainer/quality-trade-experiments.md`: the encoder ablation, the layout ladder, and what the
  shipped layout is. The rows are regenerated from saved results, not retyped.

**Quality** (`ninfer-perplexity --quick --kv-dtype int8`, same binaries, against upstream's artifact):

| Corpus | Upstream | New | Change |
|---|---:|---:|---:|
| `perplexity-1m` | 4.3428 | 4.3023 | -0.93% |
| held-out | 4.3396 | 4.3178 | -0.50% |

Mean KL divergence from Unsloth's Q8_0 on the held-out corpus: upstream's encoding 0.0376, this
layout with a 6-bit embedding 0.0304, **the shipped 4-bit-embedding artifact 0.0318**. The honest
caveats: the 4-bit embedding costs Chinese Wikipedia (0.0794 to 0.0859) and chat (0.0355 to 0.0373),
and **Unsloth's UD-Q4_K_XL is still clearly better per byte at 0.0156.** That is 15% lower KL than
upstream's encoding, and it is still twice that of Unsloth's UD-Q4_K_XL.

**Speed** (`ninfer_bench`, rk4v4, 8K context, two interleaved rounds of three, same binary):

| Workload | Upstream | New | Change |
|---|---:|---:|---:|
| Plain decode, tg128 | 47.1 tok/s | 49.2 | +4.5% |
| MTP3 + draft head | 147.4 | 151.5 | +2.8% |
| DFlash2 K=7 + draft head | 251.1 | 261.2 | +4.0% |
| Prefill pp4096 | 1,713 tok/s | 1,727 | +0.8% |
| Prefill pp4096, `--prefill-cublas` | 3,196 | 3,185 | -0.4% (tie) |
| Prefill pp16384, `--prefill-cublas` | 2,806 | 2,802 | -0.1% (tie) |

The bench corpus drafts unusually well (7.3 tokens per round), so the same comparison was repeated
on **24 varied chat prompts** (code in eight languages, prose, maths, translation, Chinese; greedy,
512 tokens, one stream through `ninfer-serve`). That fixture is now in the repo:
DFlash2 K=7 **125.9 to 133.5 tok/s (+6.1%)**, MTP3 **111.5 to 115.4 (+3.5%)**, with tokens per
round essentially unchanged (3.53 to 3.57 and 2.87 to 2.84).

**Memory.** DFlash2 weights fall from 17.7 to 17.0 GiB. Beside a desktop holding 1.6 GiB the DFlash2
profile now starts at **188,416 tokens** (the Windows default, up from 172,032) with 721 MiB free
and is refused at 196,608; upstream's artifact had to step down to 150,720 the same day.

## New kernels that made it faster

The artifact's layout moved weight bytes between formats, and the existing small-T routes were
tuned for the old shapes. Each of these was measured and several fixed regressions the artifact
itself would have introduced (all on sm_86; none are on the published sm_120a tables):

- **Q6 vocabulary head** (#159): a small-T MMA kernel for the head verify. The 6-bit head's T=3..16
  routes sat at about 2.4 ms from T=5, which made a DFlash2 K=7 verify about 1 ms slower than with
  the 8-bit head. The new kernel holds 1,154-1,175 us from T=3 to T=8 (about 850 GB/s), plus a
  GEMV for T=1..2 and, in a follow-up, T=17..32.
- **Q4 MLP down projection**: a Q4 `linear_add` at 5120x17408 and a format-generic A8 add. Our own
  first version **regressed T=9..16** (the 24-column K-split kernel ran 112.6 us, slower than the Q5
  route it replaced); a dedicated T=9..16 kernel fixed that and is 15-26% faster than the
  regression (89-113 us at K=17408, 39-43 us at K=6144). A 17..32-column variant followed.
- **Long-context attention.** The 27B's small-T attention is capped at 41 splits: 164 partial CTAs,
  exactly one two-CTA wave on the 3090's 82 SMs, where the shared cap of 85 produced 2.07 waves.
  Op bench, four interleaved rounds: 5-27% faster from 16K keys at T=1, 4 and 8; at 128K
  447 to 344 us (T=1), 602 to 468 (T=4), 781 to 572 (T=8). End to end through `run.bat`'s DFlash2
  profile, a 133K-token conversation decodes at **89.8 vs 82.5 tok/s (+8.8%)** with identical
  acceptance. Caps 82, 123 and 164 were each tried on the 27B and were worse than 41 somewhere in 32K-128K.
  The 35B-A3B got the same one-wave treatment at 82 splits (its two KV heads also made 340 CTAs):
  11-37% faster in the op bench at every depth from 16K to 224K (MTP3's T=4 verify 658 to 447 us at
  224K), **but not run end to end**; only a v2 copy of the 35B artifact was on the test host.
  **Cost:** the change reorders a float reduction, so greedy output is not bit-identical to before.
  On a 68.5K-token prompt the greedy path diverged, accepted fewer tokens, and raw throughput
  fell from 114 to 100 tok/s for that run even though a decode round was 4.4% shorter. Short chats
  were unaffected (identical outputs on the 24-prompt set, rounds 26.77 to 27.11 ms, inside
  between-process spread).

## Corrected headline numbers

Re-measuring the 27B for the README (one sitting, upstream's artifact on the same binaries, the
documented `run_chat_decode` harness): DFlash2 C1 **140 tok/s (upstream's artifact 127)**, MTP3 C8
**477 tok/s (upstream 482, a tie that we are not calling a win)**, `ninfer_bench` pp4096 on the
cuBLAS route 3,185 (upstream 3,196). **The earlier README figures of 187 and 523 tok/s came from runs
whose workload was not recorded and that this harness does not reproduce on either artifact; they
are withdrawn.** The README and its performance banner carry the re-measured values, plus a
second chart comparing the two artifacts on size, KL, decode and context.

## 35B-A3B launcher: faster prefill, less context beside a busy desktop

The 35B profile now prefills through cuBLAS at chunk 4096 (the route only engages from chunk 2048 on
this model's dense projections): **5,470 to 8,848 tok/s** on a 4K prompt (`ninfer_bench` pp4096).
The larger runtime reservation costs context: measured with the desktop holding 1.3 GiB, the old
profile started at 262,144 with 154 MiB free and the new one is refused at 229,376 (27 MB short), so
the **Windows default becomes 212,992**. `NINFER_PREFILL_CHUNK=1024` keeps the full 262,144 at
7,140 tok/s. The Linux default stays 262,144 with its automatic step-down; **that combination was
not re-measured**, and **the route's quality cost on the 35B is not measured** (the perplexity tool
does not engage it, so its scores are identical by construction). The README also now states the
27B's equivalent trade plainly: cuBLAS prefill at chunk 4096 reserves 1,536 MiB for 3,246 tok/s,
roughly 44K tokens of context for 83% faster prefill than the default route at chunk 1024;
chunk 2048 keeps 92% of that speed for about 14K tokens back.

## Serving API

- **Per-request `thinking_budget` on Chat Completions and Responses** (#157), following the
  existing `graft` extension pattern: a positive integer, or `null` for the server default; anything
  else is a 400 with `param: thinking_budget`. Previously only Anthropic's `thinking.budget_tokens`
  could set it per request, and a `thinking_budget` key in a Chat body was silently dropped.
- **Long sessions no longer hit `thinking_budget_capacity_insufficient`** (#157, continuing #156 by
  @mgscreativa, whose commits are kept in history). When the remaining output room is just above the
  budget, the engine used to 400 every retry as a conversation grew into that window. It now lowers
  the effective budget to leave room for the early-close control, or, when even that cannot fit,
  lets thinking run to the output limit (the response can then end inside the reasoning with empty
  content, which `docs/serving.md` now says). The 400 and its error kind are removed; logs report
  requested and effective budgets separately. #157 also fixed a data race in that code (the effective
  budget was read without a lock while the worker wrote it) and replaced a test that asserted a
  variable it had just assigned.
- **`reasoning.summary` accepted on `/v1/responses`** (#152). Clients such as pi send
  `reasoning: {effort, summary: "auto"}` and got a 400. `auto`, `concise` and `detailed` are now
  accepted as hints (anything else is a 400); NInfer produces no summaries, so responses still carry
  `summary: null`.
- **`--default-graft NAME`** (#151): requests that name no `graft` get it; `"graft": ""` opts out; a
  named graft overrides. Works on all three protocols. `run.bat` opts in with
  `NINFER_DEFAULT_GRAFT=on`.
- **Direct-KV grafts use the context cache** (#155). Agent clients that send a graft on every request
  re-prefilled the whole 39-42K history each turn (about 2K tok/s, 21-24 s time to first token). The
  graft's pinned slot stays the root and captures layer on top, with placeholder ids derived from
  the graft's hash so only the same graft matches. A first version that counted rebuild cost from the
  end of the graft broke an engine invariant and failed every capture; it was caught in live use, not
  by unit tests, and reverted before merge. **Not measured:** cached-vs-uncached output equality,
  host-tier behaviour for pages shared with the pinned graft, and rk4v4 with a graft (only rk8v4 was
  tried).
- **The shipped graft** is now the trained v3 graft served under the name `godmode` (#153, #154):
  the v1 graft produced garbage on the 27B (two of two requests ran to the length limit) and is
  removed, as is the untrained 35B graft. Clients must send `"graft": "godmode"`; `v1` is now a 400
  `unknown_graft`. About four of six requests still drift into Chinese mid-thought on a q4 artifact:
  reduced, not fixed. `run.bat` now actually loads it and warns when the file is missing. The `.bin`
  is distributed through Git LFS, which GitHub refuses on this public fork, so it is not in the
  source tree; the release does not ship it either.

## Launchers and documentation

- **`NINFER_CHAT_TEMPLATE`** (#161, closes #160): `run.bat` and `run.sh` pass a custom Jinja chat
  template through to `--chat-template`, including across the automatic step-down retries.
- **Qwen3.8-27B is the recommended model** everywhere (#158): README, both archive guides and the
  `.bat` menus. `run.sh` and `download-model.sh` still require an explicit model rather than
  silently starting a 20 GB download.
- **`docs/cli.md` was fixed and reordered** (#158): it claimed FP8 KV in every example when they all
  pass `int8`, recommended two different KV formats in one section, and documented
  `--thinking-budget` as combinable with `--reasoning-effort none`, which is rejected. The options
  missing from its table are documented now.
- The README gained a performance banner, badges and an artifact-comparison chart.
- **`run_chat_decode.py`** (#163) compares artifacts and speculative backends in one sitting:
  `;model=PATH` per arm, `--spec mtp|dflash2|none`, `--model-id`. Two Windows bugs it exposed are
  fixed (non-UTF-8 prompt reads, relative server path). In the first A/B it ran, 116.4 vs 122.3
  tok/s at an identical 36.3% acceptance; its absolute numbers differ from #159's because it uses a
  different chunk size, no `--gdn-state-fp16`, and a different aggregation.

## Credits

@mgscreativa for the thinking-budget boundary fix (#156) behind #157, and earlier for the KV-loan
race diagnosis (#138). Both are in `CONTRIBUTORS.md`.

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
glibc dynamically; see `docs/release-archive-linux.md` for the runtime requirements (built on
Ubuntu 24.04, glibc 2.38+). Model artifacts are not included: `download-model` fetches
Qwen3.8-27B (19 GB, the DFlash2 bundle, which also carries the MTP weights), Qwen3.6-35B-A3B
(21 GB) or Qwen3.6-27B (16 GB).

## Known issues

- **The new 27B artifact is not a free lunch.** The 4-bit embedding costs Chinese text and chat a
  little KL, and Unsloth's UD-Q4_K_XL is still much closer to the Q8_0 reference per byte.
- **Long-context attention change is not bit-identical** (see above): one 68.5K-token greedy run
  diverged and drafted worse. Short chats and a 133K run were unaffected or faster.
- **MTP3 at eight lanes is a tie, not a gain** (477 vs 482 tok/s on the re-measured harness).
- **The `godmode` graft still drifts** into Chinese on roughly four of six requests on q4 artifacts.
- **Tensor parallelism is not built; vision and DFlash/DFlash2 refuse a multi-GPU split** (unchanged
  from v0.12.0). NVFP4 and K8V4 KV still require a Blackwell GPU.
- **Speculative decoding is still not bit-identical to width-1 greedy decode** (unchanged).
- **Not verified for this release:** the `.bat` launcher menus were reviewed by reading, not run
  interactively; the full ctest suite was run per PR on affected tests only, not re-run as a whole
  on the release commit.

## Upgrading

- **Qwen3.8-27B: download the new artifact.** `download-model qwen38-27b` fetches the pinned
  revision. `run.bat` and `run.sh` no longer pass the load-time transcode flags (`--embedding-q4`,
  `--lm-head-q6`) that upstream's artifact needed, and **we did not test the launchers against the
  old artifact**, so re-download rather than mix them.
- **Graft clients:** send `"graft": "godmode"` instead of `"graft": "v1"`.
- **Thinking budgets:** if you handled `thinking_budget_capacity_insufficient` in a client, you can
  delete that branch.
- **35B-A3B on Windows:** the default context is 212,992 (was 262,144) because of the faster prefill
  route; set `NINFER_PREFILL_CHUNK=1024` to trade back.

Everything else is unchanged from v0.12.0.
