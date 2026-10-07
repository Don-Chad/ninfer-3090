# NInfer-3090 v0.15.0

**Streams that are already writing no longer freeze while the server reads someone else's long prompt.**
If you run `ninfer-serve` for several agents at once, this is the one default that moves: while one
request is reading a long prompt, the others keep generating at a usable speed instead of crawling at
about two words a second. The price is that the long prompt is read more slowly while that happens
(about a third to a half slower, measured below). You can turn the old behaviour back on with one flag.
The server also now tells you which version it is. 8 merged pull requests (#182, #183, #184, #185, #186,
#187, #188, #189), of which only #183 and #185 change what you see.

## What this means for you

- **You run several coding agents on one card, and some send huge prompts.** Upgrade and do nothing.
  Agents that are mid-answer used to stall for roughly half a second between every word whenever
  another request was reading a long prompt (2 words/s). They now keep going at about 19 words/s in the
  same situation. The cost is that a 24,700-token prompt takes about 20.5 s to read instead of 13.0 s
  while that other stream is generating; the table has the full trade-off.
- **You mostly run one agent at a time.** Nothing changes. With nothing else generating, the new
  setting has no effect.
- **You care more about reading long prompts quickly than about other streams moving.** Add
  `--decode-rounds-per-prefill 1`. That restores the v0.14.3 behaviour exactly.
- **You run speculative decoding (MTP or DFlash2).** Try `--decode-rounds-per-prefill 8` to `12`. Each
  speculative round is a little longer, so the default costs prefill more (see below).
- **You run a rented or shared server and want to know what is deployed.** `ninfer-serve --version`,
  `GET /health` and an `X-NInfer-Version` header on every reply now say which build is running.
- **A default moved.** `--decode-rounds-per-prefill` defaults to automatic. If you rely on a long prompt
  being read at full speed while other requests generate, set it to `1`.

## Other streams keep moving during a long prompt: `--decode-rounds-per-prefill`

The server used to alternate strictly: read one chunk of the long prompt (hundreds of milliseconds),
then produce one word for everybody else, then another chunk. A generating stream therefore got about
one word per chunk and only a few percent of the card. Now, after each chunk of prompt reading, the
server runs up to N rounds of generation before the next chunk. The default (`0`) picks N from the chunk
size: the chunk size divided by 64, so 16 at the default chunk of 1,024 and 64 at 4,096.

Measured on an RTX 3090, Qwen3.8-27B, `--prefill-cublas --max-concurrency 4 --no-prefix-reuse`, one
stream generating while one 24,700-token prompt is read, two runs per row (pairs agree within about 3%).
"Before" is the old alternation (`--decode-rounds-per-prefill 1`):

| Setting | Words/s for the generating stream | Long prompt reading speed | Long prompt vs before |
|---|---|---|---|
| Chunk 1,024, before | 2.0 | 1,910 tok/s (13.0 s) | - |
| Chunk 1,024, 8 rounds | 11.7 | 1,490 tok/s (16.6 s) | 22% slower |
| **Chunk 1,024, automatic (16)** | **18.9** | **1,208 tok/s (20.5 s)** | **37% slower** |
| Chunk 1,024, 32 rounds | 26.8 | 860 tok/s (28.7 s) | 55% slower |
| Chunk 4,096, before | 0.85 | 2,370 tok/s (10.4 s) | - |
| Chunk 4,096, 16 rounds | 8.1 | 2,010 tok/s (12.3 s) | 15% slower |
| **Chunk 4,096, automatic (64)** | **20.9** | **1,337 tok/s (18.5 s)** | **44% slower** |

Costs and limits, stated plainly:

- **The long prompt is slower while anything else is generating**: 37% at the default with chunk 1,024,
  44% at chunk 4,096. Our own target was about 40%, so chunk 4,096 is slightly over; `--decode-rounds-per-prefill
  32` is the dial there. We estimate a 138,000-token prompt would take about 115 s instead of 72 s under
  that load. That figure is extrapolated from the rates above, not measured.
- **The longest single pause did not shrink.** The chunk already in progress still has to finish
  (about 615 ms at chunk 1,024, about 2 s at 4,096). Streams now move in bursts of N words between those
  pauses rather than one word per pause. With the ratio on, chunk 1,024 with cuBLAS reads within about 10%
  of the speed of chunk 4,096 and has a roughly three times shorter worst pause, so 1,024 is the better
  interactive setting.
- **A new request's first chunk waits at most N generation rounds** (about N x 21 ms) behind the others.
- **Speculative decoding** (same test, chunk 1,024, one stream plus one long prompt, two runs each):

  | Mode | Setting | Stream events/s | Long prompt reading speed |
  |---|---|---|---|
  | MTP, `--draft-tokens 3` (18,500-token prompt) | before | 2.1 | 1,950 tok/s |
  | MTP | automatic (16) | 16.8 | 1,070 tok/s (45% slower) |
  | DFlash2, `--draft-tokens 7 --lm-head-draft` (17,000-token prompt) | before | 2.1 | 1,960 tok/s |
  | DFlash2 | automatic (16) | 16.4 | 1,080 tok/s (45% slower) |

  Speculative rounds take 27-29 ms against 21 ms, which is why the prompt pays more. These count streamed
  events, and one event can carry several words, so the real word rate is at least this. The MTP and
  DFlash2 rows used different prompt lengths and context sizes because DFlash2 ran out of memory at the
  larger one. On this 27B artifact DFlash2 also needs `--lm-head-draft` or startup fails; the previous
  release fails the same way, so this is not new.
- **Not tested:** five streams at once as a gateway would produce, chunk 512, runs without cuBLAS
  prefill, DFlash on the 35B-A3B, and speculation with several streams generating. The divisor of 64 is
  a starting value picked from single-stream runs.

## The server says which version it is

Until now nothing a running server exposed told you which build it was. Now:

- `ninfer-serve --version` and `ninfer --version` print it without loading a model.
- `GET /health` returns `{"status":"ok","version":"..."}`.
- Every response carries an `X-NInfer-Version` header, including the "still loading" 503 and
  authentication failures.
- `/props` reports it in `build_info`, and the `server_start` log record includes it.

The text is the release version, plus `+<commit>` for a build that is not exactly a release, plus
`-dirty` if the source had uncommitted changes. The release archives report the plain version. This was
checked live against `qwen3_8_27b.ninfer` (503 while loading, then 200, with the header on both), and the
`server_start` record only in a unit test, not in a real log file.

## Also fixed

- Linux developers only: `ninfer_artifact_materialization_test` failed on every GNU `ld` build with
  "failed loading leaked pinned or device allocations" because the test did not watch `cudaHostAlloc`
  (#184, from a community contributor). This never affected the shipped binaries.

## Under the hood

Pull requests #186-#189 remove code and files nothing referenced: unused device and host helpers, three
INT8 prompt experiment switches that no build ever set, seven one-off benchmark scripts, two large
unused data files, and two build batch files that would break the build tree. Nothing user-visible
changes. Full ctest after the Ops cleanup (#187) passed 160 of 160 on the author's machine. #186 and #189
were checked by build or Python tests only.

## Verification

Built from the release branch on Windows (MSVC 2022, CUDA 12.8) and Linux (WSL Ubuntu, CUDA 12.8),
both `sm_86`, targeted at an RTX 3090.

- **Smoke test on the real card (Windows archive, RTX 3090, nothing overridden):** `run.bat qwen38-27b`
  started the default profile (Qwen3.8-27B, context 188,416, rk4v4 KV, DFlash2 with draft head, vision
  overlay), answered a chat completion, and `GET /health` and the `X-NInfer-Version` header reported the
  version. The Windows binaries report `0.15.0-rtx3090+708a0a5f` because they were built from the release
  branch before the merge; the Linux build, from a `git archive`, reports `0.15.0-rtx3090`.
- **Archives:** both checksum files verify, every file matches its inner `SHA256SUMS.txt`, the Windows
  executables start with only Windows on `PATH`, and the Linux archive's `ninfer-serve --version`,
  `--help` and `run.sh --help` work from a fresh unpack. Linux dynamic dependencies show no missing
  libraries on the build machine.
- **Not run for this release:** the full test suite (it was run on the individual pull requests, not on
  this tree), the real-model tests, the Linux binaries against a model, and any new speed measurement:
  the tables above are the pull request's own numbers, not re-measured on these binaries.

## Downloads

Two archives: the Windows zip (cuBLAS DLLs included) and the Linux tar.gz (needs glibc 2.38+, CUDA 12.8
runtime with cuBLAS, FFmpeg 6; see `docs/release-archive-linux.md`). Check them with the `SHA256SUMS`
files. Details for each option are in `docs/serving.md`.
