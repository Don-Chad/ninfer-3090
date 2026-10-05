# NInfer-3090 v0.14.0

**One flag to size the RAM cache for you: `--auto-host-cache`.** If you run NInfer as a server that
mostly handles long agent conversations, this is the release to pick up. The server can now work out
how much of the machine's RAM to give to its prompt cache, instead of you guessing four numbers.
The rest of the release is smaller: vision now works across several GPUs, `download-model` can fetch
the `godmode` graft, the tuned launchers add a repetition guard, and rebuilding from source is much
faster. 5 merged pull requests, 18 commits.

## What this means for you

- **Running a server on a box that does nothing else** (a rented GPU, say): replace four cache flags
  with `--auto-host-cache`. Long conversations then stay cached when they leave the GPU, so a returning
  agent does not make the engine re-read 100,000 tokens from scratch.
- **Running on your own desktop:** nothing changes unless you pass the flag. The launchers do not use it.
- **Using vision with more than one GPU:** it is no longer refused (see the caveat below).
- **Using the `godmode` graft:** `download-model` now fetches it for you when your Hugging Face account
  has access.
- **Building from source:** install a compiler cache once and rebuilds take minutes, not tens of minutes.

## `--auto-host-cache`: let the server size its RAM cache

NInfer keeps finished conversations in a cache so that the next turn only has to process what is new. When
the GPU runs short, the cache spills to ordinary RAM instead of throwing a conversation away. How much RAM,
and how many conversations to remember, used to be four settings you chose by hand:
`--host-kv-mib`, `--host-state-slots`, `--max-private-continuations` and `--max-shared-prefixes`. The
defaults are small (8 GiB and a handful of conversations), which is fine for one person and far too little
for a server with many long conversations. A 100,000-token conversation takes about 1.7 GiB of cache with
`rk4v4` KV on the 27B, so 8 GiB holds four or five of them.

With the new flag the server looks at how much RAM is actually free once the model has loaded, keeps a
safety margin, and uses the rest:

```
# before
ninfer-serve model.ninfer ... --host-kv-mib 8192 --host-state-slots 32 \
  --max-private-continuations 8 --max-shared-prefixes 8

# after
ninfer-serve model.ninfer ... --auto-host-cache
```

How it decides:

- **How much RAM it can use.** The smaller of what the system reports as available and what is left under
  the container's memory limit, so a rented container is sized by its own limit rather than the host's RAM.
  On Windows it also stays within what the GPU can pin (see below).
- **The margin.** It leaves `--host-cache-reserve-mib` unpinned, 3072 by default. I measured the server
  taking about 2 GiB more after the cache is sized (CUDA and cuBLAS state, the HTTP server, buffers), and
  a container that goes over its limit is killed, so the margin is deliberately not tiny. You can lower it
  once you have watched your own server under load.
- **How it splits the rest.** One eighth buys saved conversation states (each about 75 MiB on the 27B, at
  most 128); the remainder is cache for the prompt tokens themselves, which is what limits how many long
  conversations survive. The number of conversations it will remember follows from that.
- **What the startup log says.** The `context cache |` line shows the sizes it chose.

What I measured (Qwen3.8-27B on an RTX 3090, in a Linux machine with about 29 GiB of free RAM, using the
packaged release binary and the default margin): it picked 26 saved states and 13.8 GiB of cache,
remembering up to 34 conversations. A prompt of about 1,850 tokens sent twice reused 1,849 tokens of the
second request from the cache. The machine still had about 12 GiB free after startup, so on this machine the
default is on the cautious side; I did not investigate why it measured less free RAM than was there (the
model is read through the Windows file system on this setup). Pinning 15.7 GiB took about 26 seconds,
roughly 0.6 to 0.85 GiB per second, so a machine with 100 GiB to pin should expect a couple of minutes at
startup (an extrapolation, not a measurement).

Things to know:

- It replaces the four flags and refuses them if you pass them as well, and refuses `--no-prefix-reuse`.
  `--device-state-slots` and `--max-long-anchors-per-continuation` still work alongside it.
- It assumes the machine runs only this server. Another large process on the same box can still run it
  out of memory.
- Sizing happens once, at startup; it does not follow RAM that changes later.
- **On Windows the effect is small by design.** Windows charges pinned RAM against the GPU's memory, so a
  card the model nearly fills gets a small host cache, down to none. This is the existing Windows limit,
  not something new. This feature is aimed at Linux servers.
- **Not tested where it matters most:** a real memory-limited container such as a rented cloud instance.
  The limit-reading code is tested against fabricated cgroup trees (nested limits, a tighter child,
  cgroup v1, a namespace root) and runs on a real Linux (WSL) machine without a limit, but a limited
  container has not been tried.
- **Not measured:** how much it improves cache hits on real 40,000 to 200,000-token agent workloads. The
  data that motivated it came from a different provider's cache, so it shows how large prompts are, not
  how well our cache would do. Please check the hit rate on your own traffic.

## Smaller changes

- **Vision with several GPUs.** `--vision` (resident and overlay) now works together with a
  `--devices` split instead of being refused. The vision tower runs on the first GPU as before and the
  later stages see an ordinary result. Checked with `--devices 0,0` (two stages on one GPU), where the
  output was byte-identical to a single GPU. **Not checked on two real GPUs**, with video, several images
  or concurrent requests. DFlash and DFlash2 are still refused under a split.
- **The `godmode` graft downloads itself.** The graft cannot live in the repository (GitHub refuses large
  files on this public fork), so it is now in a private Hugging Face repository. `download-model` for
  `qwen38-27b` fetches it into `artifacts/grafts/` when `HF_TOKEN` (or `hf auth login`) has access; without
  access it says so in one line and the model download still succeeds. `NINFER_GRAFTS=off` skips it and
  `NINFER_GRAFT_DIR` changes where it goes. `run.sh` still does not load grafts; use `--graft` by hand.
- **Repetition guard in the tuned launchers.** The `tuned` profiles for the 27B and the 35B-A3B now pass
  `--min-p 0.03 --presence-penalty 0.5`, to reduce loops on the small quantised models. Override with
  `NINFER_MIN_P` and `NINFER_PRESENCE_PENALTY`, or set either to `default` to leave the model's own preset.
  **This changes behaviour:** it replaces the preset in both modes (on the 27B, thinking presence penalty
  goes from 0 to 0.5 and non-thinking from 1.5 to 0.5). A client that sends its own values still wins.
  The values are untested for loop rate or code quality on our quantisation, and it does not stop
  exact-phrase loops across turns. At release time the default 27B profile of both `run.bat` and `run.sh`
  started with the guard shown in its banner and served a request; the 35B-A3B profiles were not run.
- **Faster rebuilds for contributors.** The build now uses ccache automatically when it is installed,
  shared across all git worktrees of a checkout. On this machine a full clean rebuild of the server and
  the test suite went from over ten minutes to 1.4 minutes once the cache was warm, and a second worktree
  at the same commit built in 4.3 minutes. One-time setup: `scripts\setup-ccache.ps1` (Windows) or
  `scripts/setup-ccache.sh` (Linux). It does not share between machines, and Windows debug builds cannot
  be cached. Nothing here changes the binaries.

## Checked for this release

Both archives were built from the same commit (the merge of the last pull request) on an RTX 3090.

- **Windows zip (1.6 GiB):** every file matches its checksum, the zip matches the published checksum, and
  all three executables start with only Windows on the `PATH` (no CUDA Toolkit needed). The packaged
  server started with `--auto-host-cache`, served requests and reused its cache. `run.bat qwen38-27b`
  with nothing overridden started the DFlash2 profile at 188,416 tokens and served a request.
- **Linux tar.gz (1.1 GiB):** checksums match, a fresh unpack starts `ninfer-serve`, `ninfer`,
  `ninfer_bench` and `run.sh --help`, and no libraries are missing (it needs glibc 2.38 or newer). The
  packaged server with `--auto-host-cache` served requests and reused its cache. `run.sh qwen38-27b`
  with nothing overridden stepped down twice because the card was shared with a Windows desktop (262,144,
  then 229,376, then 196,608 tokens, as designed) and then served a request.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench` | throughput benchmark against the public Engine route |
| `run.bat` / `run.sh` | serving profiles: `run <model> [profile]` |
| `download-model.bat` / `.sh` | pinned, resumable, checksum-verified model downloads |
| `README.md`, `SHA256SUMS.txt`, `LICENSE`, `VERSION` | the guide, checksums for every file, licence |

The Windows archive also carries the DLLs it needs: FFmpeg, libcurl, zlib, and NVIDIA's cuBLAS runtime
(`cublas64_12.dll` and `cublasLt64_12.dll`, redistributed under the CUDA Toolkit EULA, included as
`NVIDIA-CUDA-EULA.txt`), so no CUDA Toolkit is required. Linux links cuBLAS, FFmpeg and glibc
dynamically; see `docs/release-archive-linux.md` for the requirements (built on Ubuntu 24.04, glibc 2.38
or newer). Model files are not included: `download-model` fetches them.

## Known issues

- Everything marked "not tested" or "not measured" above.
- Unchanged from v0.13.0: tensor parallelism is not built; DFlash and DFlash2 refuse a multi-GPU split;
  NVFP4 and K8V4 KV need a Blackwell GPU; speculative decoding is not bit-identical to plain greedy
  decoding; the `godmode` graft still drifts into Chinese on roughly four of six requests on a q4 model.
- The full test suite was not re-run on the release commit; each pull request ran the tests for what it
  changed.

## Upgrading

- Nothing is required. Model files from v0.13.0 work unchanged.
- To use the new cache sizing, replace the four cache flags with `--auto-host-cache` as above and check the
  `context cache |` line in the startup log.
- If you rely on the tuned launcher profiles' sampling, note the new default `--min-p` and presence
  penalty, or set `NINFER_MIN_P=default NINFER_PRESENCE_PENALTY=default` to keep the old behaviour.

Everything else is unchanged from v0.13.0.
