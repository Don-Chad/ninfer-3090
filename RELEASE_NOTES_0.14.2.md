# NInfer-3090 v0.14.2

**Long prompts no longer make everyone else wait, and the server is easier to run on shared or rented
machines.** If you run `ninfer-serve` for a team or a fleet of coding agents, three things change for you:
one huge prompt can be processed alongside short ones, you can see which saved conversation lives in
which file, and you can cap how much RAM the cache grabs. All three are opt-in. If you change nothing,
the server behaves as it did in v0.14.1. 3 merged pull requests (#175, #176, #177).

## What this means for you

- **Your agents send some very long prompts and some short ones.** Turn on `--max-prefill-lanes 3`.
  A short request that used to sit behind a long one now gets its first word in about a second and a
  half instead of two and a half (or under a second with MTP/DFlash2). The long request takes
  roughly 4-9% longer.
- **You run on a rented GPU or a box other people share.** Add `--host-cache-max-mib` next to
  `--auto-host-cache` so the cache cannot grow into memory your neighbours need. We added it after a
  production server was killed by the host when RAM ran short.
- **You save conversations to disk and bring them back after a restart.** `GET /slots` now tells you
  which file each live conversation is saved in, when it was last used, and how often it was reused,
  so you can choose what to keep without guessing. *(Superseded: later releases retired the slot
  save/restore API and `snapshot_file`; use `--context-store DIR`, see
  [docs/serving.md](docs/serving.md#context-store).)*
- **You do none of the above.** Nothing changes. No default moved.

## Several long prompts at once: `--max-prefill-lanes`

Reading a prompt in (prefill) used to be one-at-a-time. A 200,000-token prompt takes about six minutes
on a 3090, and everyone behind it waited, even people whose prompt was already cached and needed almost
no work.

With `--max-prefill-lanes N` up to N requests are read in together, and the card's time goes first to
whoever has the least left to read. The default stays at 1, which is the old behaviour.

What we measured (RTX 3090, Qwen3.8-27B, same machine before and after):

| Situation | Before | After (3 lanes) |
|---|---|---|
| Short request behind a 3,000-token prompt | 2.5 s to first word | 1.5 s |
| Same, with MTP | 3.0 s | 0.8 s |
| Same, with DFlash2 (2 lanes) | 2.7 s | 0.7 s |
| Short request behind a 200,000-token prompt (3 lanes, room for one long prompt) | waited behind the whole prompt (~6 minutes alone) | 2.9 s |

Things to know before you turn it on:

- **Order is no longer strictly first come, first served among prompts being read in.** A prompt with
  less left finishes first, even when it arrived later. A starvation guard (`--prefill-max-skip`,
  default 8) stops anything waiting forever. We tested that guard in unit tests, not in a real run.
- **The long prompt itself gets a little slower** (about 4-9%) while others share the card.
- **Memory is the real limit.** Each request reserves room for its whole prompt and answer. If the
  card can only hold one 200,000-token prompt, a second one waits for room however many lanes you set.
- **Raise `--pending-timeout-ms`** above the longest prompt a request might queue behind. The 30-second
  default failed our own 200,000-token test.
- Pick a lane count for how many long prompts you expect at the same time. Two lanes with two long
  prompts still left a short request waiting 3-5 seconds.

## `/slots` says which file a conversation is saved in

> **Superseded.** The slot save/restore API, `--slot-save-path`, `--auto-save-evicted` and the
> `snapshot_file` field were retired after this release. Conversations now persist through
> `--context-store DIR` ([docs/serving.md](docs/serving.md#context-store)); `GET /slots` stays, read-only,
> without `snapshot_file`. The text below describes what this release shipped.

Each live conversation in `GET /slots` now also reports `snapshot_file`, `last_used_unix_ms`,
`reuse_count` and `reused_tokens` (all `null` for an empty slot). The reason is practical: the server
moves a conversation between slots between turns, so the slot number is not a stable name for it. The
file name is. Saved and restored, a 100,000-token conversation resumed in 0.2-0.3 s instead of
re-reading for roughly one to three minutes. Existing fields are unchanged; there are just more of them.

## `--host-cache-max-mib`: a ceiling for the RAM cache

`--auto-host-cache` uses nearly all free RAM, on the assumption that the server has the machine to
itself. On a shared box that is a trap: the memory is pinned and cannot be taken back. Measured on a
2x3090 rental with a 126 GB host:

| Setting | RAM pinned by the cache | Free RAM afterwards |
|---|---|---|
| `--auto-host-cache` | 70.5 GiB | 26 GiB |
| with `--host-cache-reserve-mib 12288` | 62.5 GiB | 35 GiB |
| with `--host-cache-max-mib 10240` | 8.8 GiB | 99 GiB |

On a real agent workload (five concurrent 100,000-150,000-token sessions) the RAM cache was not used
at all, so the cap cost nothing. A server with many more simultaneous conversations should set the cap
to match. It keeps no default and requires `--auto-host-cache`.

## Also fixed

- Two prompts prefilling together could corrupt each other's output (the first request's attention
  cache was written to the wrong place). Found by test while building lanes and fixed there; it cannot
  occur at the default of one lane.
- A flaky prefix-cache test (`shared-replacement`) that failed in roughly one run in three now waits for
  the engine to settle before it checks.

## Verification

Built from master at `943d44e8` on Windows (MSVC 2022, CUDA 12.8, all 765 build steps) and Linux
(WSL Ubuntu, CUDA 12.8; `ninfer`, `ninfer-serve`, `ninfer_bench`), both `sm_86`, tested on an RTX 3090.

- **Full test suite (Windows):** 160 tests, 152 passed, 0 failed, 8 real-model tests skipped without a
  model. About 14 minutes.
- **Real-model tests on the 27B (`qwen3_8_27b.ninfer`):** prefix cache (including the interleaved-prefill
  scenarios), scoring, vision workspace, DFlash2 and multi-GPU stages passed. `ninfer_qwen3_5_dflash_real_test`
  fails because this artifact has no DFlash (v1) component; that decode path is not pursued and is not
  covered. The 35B MoE real test ran for 42 s on `qwen3_6_35b_a3b.ninfer` with no failure, but the runner
  reports it as skipped, so treat it as unconfirmed. `ninfer_qwen3_5_loading_real_test` skipped.
- **Not re-run for this release:** the lane timings in the tables above are from the pull requests'
  own measurements (single runs, which vary 15-20% on this machine), not from these binaries. The
  `/slots` fields were checked against a real engine on Linux by their author, not on Windows. We did
  not test `--max-prefill-lanes` above 3, DFlash2 beyond about 3,000-token prompts, or vision together
  with a 200,000-token prompt.
- **Linux archive:** not run against a model here; see the packaging checks in the release process.

## Downloads

Two archives: the Windows zip (cuBLAS DLLs included) and the Linux tar.gz (needs glibc 2.38+, CUDA 12.8
runtime with cuBLAS, FFmpeg 6; see `docs/release-archive-linux.md`). Check them with the `SHA256SUMS`
files. Details for each option are in `docs/serving.md`.
