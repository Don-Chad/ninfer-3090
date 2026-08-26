# NInfer-3090 v0.6.1

This release ships native Windows and Linux x64 binaries for the RTX 3090, with
the current Qwen3.8-27B artifact and the same explicit SM86 runtime profile on
both platforms.

## Changes

- Adds the upstream swscale destination-alignment fix for JPEG decoding, including
  its 300x200 regression fixture.
- Renders Anthropic and Claude Code system reminders in place after a conversation
  begins, preserving prefix reuse across turns.
- Streams tool-call whitespace without repeatedly scanning the buffered prefix.
- Ships Windows ZIP and Linux tarball packages with their platform launch helpers.

## Contributors

- [iamwavecut](https://github.com/iamwavecut) — JPEG safety fix, [PR #11](https://github.com/Don-Chad/ninfer-3090/pull/11).
- [nasedkinpv](https://github.com/nasedkinpv) — tool-call parser crash fix, [PR #12](https://github.com/Don-Chad/ninfer-3090/pull/12).
- [wmehanna](https://github.com/wmehanna) — in-place system turns, [PR #13](https://github.com/Don-Chad/ninfer-3090/pull/13).

## Validation

The release build and smoke gate uses these components:

- Windows: Visual Studio 2022 and CUDA 12.8
- Linux: Ubuntu 22.04, CUDA Toolkit 12.8, GCC 11, CMake 3.28, Ninja, and vcpkg FFmpeg

Each packaged `ninfer` binary completes a real RTX 3090 Qwen3.8-27B generation before publication.

## Limits

- This release is a correctness and distribution update, not a new throughput claim.
- Keep the GPU idle while loading the 27B model; a 24 GB RTX 3090 needs the
  documented explicit KV settings for higher-concurrency profiles.
