# NInfer-3090 v0.14.3

**If a client times out halfway through a huge prompt and retries, the server now picks up where it
got to instead of starting again, and the RAM cache is harder to over-size on shared or vision-enabled
servers.** One change is on by default (the retry behaviour); the other is a new opt-in flag. 2 merged
pull requests (#179, #180).

## What this means for you

- **Your agents send very long prompts and sometimes time out and retry.** Nothing to do: this is on by
  default. A retry now reuses the work the cancelled attempt had already done. In our test, a
  195,000-token prompt cancelled after about three minutes was read in 86 seconds on the retry,
  instead of 250 seconds from scratch.
- **You run `--auto-host-cache` on a shared or rented machine.** Add `--host-cache-percent 50` (or
  whatever share you can spare) to say "never pin more than this share of the machine's RAM", even if a
  neighbour starts later or two servers start together.
- **You run with image input on and `--auto-host-cache`.** The cache now leaves room for the image
  caches (about 3 GiB more by default), so it pins less than before. If you saw the host kill the server
  for running out of RAM, this is the likely fix, but we could not confirm that was the cause.
- **None of the above applies to you.** The retry behaviour still helps you; nothing else changes.
  The one default that moved is listed next.

## Defaults that changed

- **Retries resume (`--progress-anchor-tokens`, default 16384).** While a prompt is being read, the
  server now keeps a checkpoint every 16,384 tokens, and if the request is cancelled it keeps the
  deepest one for the next attempt. Set `--progress-anchor-tokens 0` to turn it off and get the old
  behaviour. The checkpoints share the existing per-conversation budget (`--max-long-anchors-per-continuation`,
  default 2) with the checkpoints the server already keeps at message boundaries.
- **Image input with `--auto-host-cache`:** the memory held back for image caches is now included, so the
  RAM cache is smaller than in v0.14.2 on those setups.

## Retries that resume instead of restarting

Before: a client that gave up partway through reading a very long prompt and tried again started from
zero every time. A single enormous message has no natural place to keep a checkpoint, so each retry was
cancelled at the same point and the server never finished it.

Now the server checkpoints as it reads, and keeps the deepest checkpoint if the request is cancelled.

Measured on an RTX 3090 with Qwen3.8-27B, one request at a time, 1,024-token reading chunks:

| Case | Without the change | With it |
|---|---|---|
| 195,000-token prompt, cancelled after about 150,000 tokens (173 s), then retried | retry reads all 195,000 tokens: 250 s | retry reuses 147,456 tokens: 86 s |
| Output of the retried request | | identical to the uncancelled run (greedy) |
| 4,127-token prompt, cancelled at 1,650, retry (stride 512) | reuses 0 | reuses 1,536, same output |

Cost on requests that are not cancelled (about 31,700-token prompts, 4 runs each): a checkpoint every
16,384 or 4,096 tokens showed no measurable change (about 22.2 s either way); a checkpoint every 1,024
tokens (30 checkpoints) added about 3.4%. The default is 16,384.

Things to know:

- **We measured this in the engine, not through `ninfer-serve` over HTTP.** Through HTTP the server also
  needs about 20 seconds to notice that a client has gone, so expect that much extra before the
  checkpoint is kept.
- **A client that times out earlier than a full read and keeps retrying should finish eventually**, since
  each attempt gets as far as its own timeout allows. That is reasoning, not something we measured.
- **Checkpoints compete with message-boundary checkpoints** for the same two slots per conversation. On
  long multi-turn prompts this could cost a few cache hits when you rewrite earlier turns. We did not
  measure how often.
- **Not covered:** several long prompts at once at 200,000 tokens; a request cancelled while it is
  still holding a checkpoint it just reused (needs more than one request at a time) is discarded as
  before.
- **Saved-conversation counters** (`reuse_count` in `GET /slots`) are not updated for a cancelled
  prompt.
- This does not stop a client from sending the same request twice at once. If yours does that, a guard
  in front of the server is still worth having.

## `--host-cache-percent`: a share-of-RAM ceiling for the cache

`--auto-host-cache` sizes itself from the memory that is free when the server starts. On a shared box
that can be too much if a neighbour grows later or two servers start together.

`--host-cache-percent N` (1-100, needs `--auto-host-cache`) caps the pinned cache at N% of the machine's
total RAM, or of the container's memory limit when that is tighter. The smallest of this, `--host-cache-max-mib`
and "free memory minus the reserve" wins. No default; leave it off and nothing changes.

With image input on, the reserve now also covers `--media-cache-mib` and `--media-live-mib`, which grow
after the cache is sized. The default reserve for image runs goes from 3 GiB to about 6 GiB.

We tested the sizing arithmetic in unit tests (percent, cap, reserve, container limits on made-up
directory trees). We did not test it on a real container or start a real model with the new flag.

## Under the hood

- Reserve sums in the host cache saturate rather than wrap.
- Review fixes on cancelled-prefill retention: space for the checkpoint is checked before the lane's
  state is released, a minimum spacing between checkpoints is enforced, and a cancelled prompt no longer
  leaves a stale saved-file binding on a slot.

## Verification

Built from master at `c53cc7b2` on Windows (MSVC 2022, CUDA 12.8; `ninfer`, `ninfer-serve`, `ninfer_bench`)
and Linux (WSL Ubuntu, CUDA 12.8; the same three), both `sm_86`, on an RTX 3090.

- **Packaged builds:** checksums match for both archives. The three Windows executables start with only
  Windows on `PATH`. A fresh unpack of the Linux archive runs `ninfer-serve --help` and `run.sh --help`,
  and its dynamic dependencies are unchanged from v0.14.2. Both new flags appear in `--help`.
- **Real card:** the packaged Windows build started through `run.bat qwen38-27b` with nothing overridden.
  The default 188,416-token context was refused because the desktop held about 2 GiB of the card, the
  launcher stepped down to 164,864 on its own, and the server answered a chat request correctly (about
  152 tokens/s). The startup log shows the new checkpoints ("progress every 16384 tokens").
- **Not run for this release:** the full test suite and the real-model tests were not re-run on these
  binaries. The figures above for retry-resume and its overhead are from #180's own measurements (one
  machine, engine API, not these archives), and #179's tests are the unit tests it describes. We did not
  start the server with `--host-cache-percent`, and nothing in this release was run on the Linux archive
  beyond `--help`.

## Downloads

Two archives: the Windows zip (cuBLAS DLLs included) and the Linux tar.gz (needs glibc 2.38+, CUDA 12.8
runtime with cuBLAS, FFmpeg 6; see `docs/release-archive-linux.md`). Check them with the `SHA256SUMS`
files. Details for each option are in `docs/serving.md`.
