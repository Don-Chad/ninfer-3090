# NInfer-3090 v0.15.1

**A short request no longer waits behind someone else's huge prompt, and a server can now keep its
cached conversations across restarts and reuse them after they were pushed out of memory.**
One default moves: with three or more serving lanes, the server now reads two long prompts at once
instead of one (details and cost below). Everything else is new and off until you turn it on. 8 merged
pull requests (#191, #192, #193, #194, #195, #196, #197, #199).

## What this means for you

- **You run several coding agents on one card (`--max-concurrency 3` or more).** Upgrade and do nothing.
  A small request that arrives while a huge prompt is being read now gets its first word in about 2 s
  instead of waiting for the whole huge prompt (112 s in our test). The cost: the huge prompt itself
  finishes later. Set `--max-prefill-lanes 1` to get the v0.15.0 behaviour back.
- **You run with `--max-concurrency 2` or less.** Nothing changes. Two lanes are full as soon as one
  stream is writing and one huge prompt is being read, so the new default cannot help there (measured,
  see below).
- **Your server restarts, or a rented box is rebooted, and you lose your cached conversations.** Add
  `--context-store DIR`. Sessions are written to disk in the background, saved on a clean shutdown, and
  loaded again at start-up, so the next turn of an agent reads only what is new. In our hard-kill test
  95% of the next turn's prompt was served from the restored cache.
- **A conversation that was cached, then dropped from memory, then continued minutes later used to be read
  from scratch.** With `--context-store DIR` it is now loaded back from disk instead (7,800 tokens in 1.3 s
  in our test).
- **Your agents send a huge `max_tokens` (100,000+) and other sessions' caches keep disappearing.** The
  server used to set aside room for the whole stated output before starting, throwing cached sessions out
  to make it. Two opt-in fixes: `--output-reservation-tokens N` (reserve as the answer grows; nothing is
  cut off unless memory really runs out) or `--max-output-tokens N` (a hard cap on every request).
  Pick N above your longest real answer. Real answers in our gateway log are 100-4,300 tokens.
- **You run a gateway or a dashboard.** `/metrics` has new series for cache hits and evictions, abandoned
  requests, cancelled prompt reading and the context store. Names are listed in `docs/serving.md`.
- **You do none of the above.** Only the lane default changes, and only from three lanes up.

## A small request no longer waits for a huge prompt: `--max-prefill-lanes` now defaults to 2

Before, one request at a time could be reading its prompt. A short request that arrived during a long
prompt was queued until the long one finished; clients gave up (a gateway log showed requests dying at
exactly 60 s with no output). From three serving lanes up the default is now two.

Measured on an RTX 3090, Qwen3.8-27B, rk4v4 KV, 131k context, one run per row: one stream already
generating, one cold ~66,000-token prompt, and a short request 3 s later.

| Lanes | Prefill lanes | Short request finished after |
|---|---|---|
| 3 | 1 (v0.15.0 default) | 112 s |
| **3** | **2 (new default)** | **1.8 s** |
| 2 | 1 or 2 | 113 s (both lanes busy; not helped) |
| 2, with `--max-output-tokens 8192` | 2 | 114 s (not helped) |

Costs and limits:

- **The long prompt finishes later**, by the reading work of every request that passes it. We did not
  re-measure the long prompt's finishing time against the old default.
- **Running streams slow down during any prompt reading.** The generating stream dropped from about 48 to
  about 16 words/s in this test, with one prefill lane as well as with two, so this is not new.
- The second lane costs about 0.02 GiB of capacity.
- One run per row, no production-size mix. If you already pass `--max-prefill-lanes`, nothing changes.

## Keep cached conversations across restarts: `--context-store DIR` (off by default)

Sessions the server retains are written to `DIR` as deduplicated chunks, so a growing conversation only
writes its new tail. Idle sessions (`--context-store-idle-seconds`, default 30) and evicted ones are
written in the background, anything unsaved is flushed on a clean shutdown, and at start-up the most
recently used sessions are loaded back (the start-up log says how many). Size and age are bounded by
`--context-store-max-gib` (default: half the free space of the volume at start-up) and
`--context-store-ttl-hours` (default 168, i.e. 7 days).

Measured on an RTX 3090, Qwen3.8-27B, rk4v4, killed without a flush, restarted:

- 3 sessions (44,000 + 2 x 10,000 tokens): all restored in 4.5 s; **95.2%** of the next turns' prompt
  tokens came from the cache (61,194 of 64,296). Writing them took 6.45 GB, with 3.08 GB avoided by
  deduplication.
- **A 119,000-token session was not fully restored**: the 3 GiB host cache could not hold its 4 GiB image.
  Restore now skips what does not fit and continues, but we did not re-measure that deep case end to end
  afterwards. In the run where the big session was not restored, its next turn was read from scratch
  (125 s).

Reading back while running: when a request continues a conversation that was pushed out of memory and the
store has it at least 4,096 tokens deeper than anything cached, the server loads it back instead of
re-reading. Test: a ~9,000-token conversation evicted by an unrelated one, then continued: 7,814 tokens
loaded in 1.30 s, and the output was identical to a conversation that never left the card.

Limits, stated plainly:

- **Other requests pause while a session loads back** (1.3 s for 7,800 tokens; about 1 s per 2 GiB on an
  SSD, not measured on a slow disk or at 100,000+ tokens).
- Loading back may push out the least recently used sessions to make room (at most 8 per request), by
  plain recency rather than the server's usual value ranking.
- Restore is limited by free cache space, not by the store size; sessions that do not fit stay on disk.
- If a background write fails, the session is still treated as saved. An eviction-time save can stall
  admission for about 1 s for very deep sessions.
- **Not compatible with DFlash2**: the server refuses that combination at start-up.
- Not tested with several simultaneous requests or at production context sizes. Timings were taken while
  the machine was also playing video, so only token counts are firm.
- A background write is skipped while any request is waiting or prefilling, so a session that stays busy
  is only saved at shutdown or when it is evicted. Not built: a bucket/S3 backend, sharing between servers.

## Huge `max_tokens` no longer has to evict other people's caches

A request used to reserve memory for its prompt plus its whole stated output before starting. Test: 4
growing sessions (one 50,000 tokens, three 25,000), 29 requests, 3 GiB host cache, every request stating
`max_tokens` 32,000. "Cold" is prompt tokens that had to be read again because the cache had been thrown
out. One run per row.

| Setting | Cold share | Avoidable full re-reads |
|---|---|---|
| Tiny reservation (`max_tokens` 16, reference) | 17.1% | 0 |
| Stated 32,000, v0.15.0 behaviour | 30.1% | 4 |
| `--output-reservation-tokens 2048` | 17.1% | 0 |
| `--max-output-tokens 4096` | 17.1% | 0 |

- **`--output-reservation-tokens N`** reserves the prompt plus at most N output tokens, and extends by up
  to 1,024 tokens before the answer runs out. Growth never evicts cached sessions; it only takes free
  pages. A 1,200-token answer crossing several growths came out token-for-token identical to a fully
  reserved one. **The price:** if memory is held by cached sessions, an answer longer than N stops at N
  with the normal "length" finish, where before the server would have evicted a cache and carried on.
  With several lanes, requests can collectively use up the free pages and some stop early (shown in a
  test: two requests got 364 and 623 of 900 tokens). Counters: `output_reservation_growths_total`,
  `output_reservation_exhaustions_total`.
- **`--max-output-tokens N`** caps every request's output at N; a request that reaches it ends with the
  usual length finish, not an error. It truncates legitimately long answers, so it is your decision.
  The two flags can be combined.
- Not measured: 200,000-token sessions, concurrent traffic, cancellations; the MTP backend was not
  exercised with `--output-reservation-tokens` (same formula).

## New `/metrics` series

- Cache: selections by starting point (hit rate is `1 - root / sum`), pressure events and searches,
  transfer bytes and seconds, occupancy per pool.
- Queue: requests that left the queue before being admitted and how long they waited; prefills cancelled
  part way, the tokens they had computed, and how many kept a checkpoint a retry can resume from. In
  our test a cancelled 4,127-token prompt kept 1,536 tokens and the retry resumed from there.
- Context store: writes, restores, loaded-back sessions, failures.

`waiting_expired_requests_total` was verified by reading the code only, not by a test. The live `/metrics`
endpoint for the cache series (#191) was checked by unit test, not against a running server.

## Also fixed

- Every in-flight request failing with "materialized sequence does not match its active entitlement" and
  the whole cache being dropped: an agent continuing a conversation after another session sharing its
  system prompt was evicted could trigger it. Reproduced on a recorded four-conversation run (failed on
  request 16 of 16 before, passes after) and on a mix with an 82,000-token session (failed on request 10,
  0 failures in 9 rounds after). A different crash (`active capture replacement effect changed after
  reservation`) is untouched.

## Verification

Built for `sm_86`, targeted at an RTX 3090. Windows: MSVC 2022, CUDA 12.8. Linux: WSL Ubuntu, CUDA 12.8.

- Per-pull-request tests were run by their authors and are quoted above: real-model scenarios on the 3090
  (`context-store`, `store-hydration`, `lazy-output-reservation`, `cancelled-prefill-progress`,
  `endpoint-anchor-adoption`, `concurrent`, `pressure-resume`, `worker-failure-recovery` and others) and
  the serve option, metrics and resource-manager unit tests.
- **Not run for this release:** the full `ctest` suite and the other real-model tests; the release only
  builds `ninfer`, `ninfer-serve` and `ninfer_bench`.
- **Known failure, not from these changes:** the real-model scenario `interleaved-prefill-cancel-mtp`
  fails with "MTP: cancelling the short request did not cancel it" on `master` too (the short request
  finishes in ~0.28 s, before the cancel lands). An older build from 6 October passes it. It was not
  bisected.
- **Smoke test on the real card (Windows archive, RTX 3090, nothing overridden):** `run.bat qwen38-27b`
  started the default profile (Qwen3.8-27B, context 188,416, rk4v4 KV, DFlash2 with draft head, vision
  overlay) in 8.5 s of engine start-up, answered a chat completion, and `GET /health` and the
  `X-NInfer-Version` header reported the version. The Windows binaries report
  `0.15.1-rtx3090+d630fffb2` because they were built from the release branch before the merge; the Linux
  build, from a `git archive`, reports `0.15.1-rtx3090`.
- **Archives:** both checksum files verify and every file matches its inner `SHA256SUMS.txt`. The three
  Windows executables start with only Windows on `PATH`. The Linux archive unpacks and runs
  `./ninfer-serve --help` and `./run.sh --help`, and `scripts/check-linux-scripts.sh` passes.
- **Not run on real hardware:** the Linux binaries were not run against a model, and none of the new flags
  (`--context-store`, `--output-reservation-tokens`, `--max-output-tokens`) were run on the release
  archives; their evidence is the per-pull-request measurements above.
