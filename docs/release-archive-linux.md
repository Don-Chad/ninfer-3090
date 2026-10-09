# NInfer-3090 for Linux — release archive

This is the README that ships **inside** the Linux release archive, where every file sits in one
directory. If you are reading it in a checkout instead, the launchers and downloaders it names live
under `scripts/`, and
[docs/rtx-3090-linux.md](https://github.com/ashalliants/ninfer-3090/blob/master/docs/rtx-3090-linux.md)
is the guide to building from source. Links here are absolute on purpose: the archive ships no
`docs/` directory.

Check `VERSION` for the release this archive was cut from, and `RELEASE_NOTES_*.md` for what
changed.

## Requirements

- x86-64 Linux with glibc 2.38 or newer (the binaries are built on Ubuntu 24.04)
- GeForce RTX 3090 (or 3090 Ti) with a recent NVIDIA driver
- The CUDA 12.8 runtime libraries, **including cuBLAS** (`libcudart.so.12`, `libcublas.so.12`,
  `libcublasLt.so.12`). The cuBLAS prefill route links it, so the binaries need it even when you do
  not turn that route on. From NVIDIA's CUDA repository: `sudo apt install cuda-libraries-12-8`
- FFmpeg 6 shared libraries and libcurl (`sudo apt install ffmpeg libcurl4t64` on Ubuntu 24.04)
- `curl` for the downloaders

Model artifacts are **not** included — they are 17–21 GB each. The downloaders below fetch them.

## Quick start

From the directory you unpacked, two commands:

```bash
./download-model.sh qwen38-27b         # ~19 GB, resumable, verifies size and SHA256; the DFlash2 bundle, it also carries the MTP weights
./run.sh qwen38-27b                    # serves on http://127.0.0.1:8080/v1
```

The downloader writes into `models/` beside these files, which is where the launcher looks. Set
`NINFER_MODEL_DIR` to keep artifacts elsewhere, or `NINFER_MODEL` to point the launcher at a single
file.

That is the recommended profile: Qwen3.8-27B, one lane with all 262,144 tokens, DFlash2 speculation
plus the draft head (fastest), `rk4v4` KV, cuBLAS prefill, and vision in overlay residency. Drop a
context rung (229376 / 196608 / 163840 / 131072 / 98304) if startup refuses.

For the longest context shared by two lanes instead, `NINFER_SPEC=mtp ./run.sh qwen38-27b` runs the
MTP profile at 262,144 tokens with a smaller prefill chunk: slower decode, more
context.

`./run.sh qwen38-27b int8` and `./run.sh qwen38-27b c8` are the older INT8 profiles — one user at
65,536 tokens, and eight concurrent users at 8,192 each.

For the Qwen3.6-35B-A3B MoE instead, which serves more lanes:

```bash
./download-model.sh qwen36-35b-a3b     # ~21 GB, resumable, verifies size and SHA256
./run.sh qwen36-35b-a3b                # serves on http://127.0.0.1:8080/v1
```

That profile runs three lanes, `rk4v4` KV, MTP3 speculation plus the draft head, and vision in
overlay residency, with the lanes sharing one 262,144-token pool (any one request can use all of
it). Three lanes start even beside a desktop, so a headless card has room to spare
(`NINFER_CONCURRENCY=4` should fit there).

The endpoint is OpenAI-compatible, so anything that speaks `/v1/chat/completions` works. Leave the
API key blank.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench` | throughput benchmark against the public Engine route |
| `download-model.sh` | pinned, resumable artifact downloads with verification |
| `run.sh` | serving profiles: `run.sh <model> [profile]` |
| `SHA256SUMS.txt` | checksums for every file in this directory |

## Overrides

Nothing here needs editing. `NINFER_SERVER`, `NINFER_MODEL_DIR`, `NINFER_MODEL`, `NINFER_HOST`,
`NINFER_PORT` and `NINFER_CHAT_TEMPLATE` (a local Jinja file, passed to `--chat-template`,
overriding the artifact's built-in template) work for every profile.

| profile | also reads |
|---|---|
| `run.sh <model>` (`tuned`) | `NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_CAPACITY`, `NINFER_KV_DTYPE`, `NINFER_SPEC` (27B: `dflash2`, `mtp`, `none`; 35B: `mtp`, `none`), `NINFER_VISION`, `NINFER_VISION_RESIDENCY`, `NINFER_DRAFT_TOKENS`, `NINFER_PREFILL_CHUNK`, `NINFER_HOST_STATE_SLOTS`, `NINFER_FALLBACK` (`off` turns the automatic step-down off) |
| `run.sh qwen38-27b int8`, `run.sh qwen38-27b c8` | nothing further; every serving flag is fixed |
| `download-model.sh <model>` | `NINFER_MODEL_DIR`, `NINFER_SKIP_SHA256` |

> `NINFER_HOST=0.0.0.0` exposes the server to your network **unauthenticated**. The launchers bind
> `127.0.0.1` for that reason. If you set it, put something in front of it.

## If startup refuses

**The default profile handles this for you.** If `run.sh` is refused at startup for lack of GPU
memory (a desktop, or another job, is holding VRAM), it steps down by itself -- an eighth of the
context at a time, up to five times, and from the second step with a 2048 prefill chunk and fewer host state slots -- says what
it did, and starts. It only does this for the defaults: a `NINFER_CONTEXT`, `NINFER_PREFILL_CHUNK`,
`NINFER_HOST_STATE_SLOTS` or `NINFER_KV_CAPACITY` you set is honoured as given, and
`NINFER_FALLBACK=off` turns it off.

For those cases the message names the numbers. Drop a context rung first —
`NINFER_CONTEXT=196608`, then 163840, 131072, 98304 (the launcher's own step-down starts from 262144 and
takes an eighth off each rung, so begin lower only if it keeps refusing). Speculation is the next lever
(`NINFER_SPEC=none`), worth about 992 MiB at the cost of decode speed. Drop vision last: in overlay
residency it costs almost nothing resident, and an `evictable pool window exceeds the evictable
tail` message means the reservation is already tight rather than that the context is too large.

The pinned Host context budget (`--host-context-mib`, by default 8 GiB plus eight model state
images) behaves differently here than on Windows: on Linux it really does pin that much host RAM. On
Windows/WDDM a pinned host allocation is charged against the card, so the runtime clamps it hard.
Same flag, different platform behaviour, by design.

On a machine that only serves this process, pass `--auto-host-cache` to ninfer-serve instead of
`--host-context-mib`: it sizes the budget from the RAM that is free after the model loads (or from
the container's memory limit), leaving `--host-cache-reserve-mib` (default 3072) unpinned. The launchers do not use it by default. See
the [serving guide](https://github.com/ashalliants/ninfer-3090/blob/master/docs/serving.md#automatic-host-cache).

## Full documentation

<https://github.com/ashalliants/ninfer-3090>
