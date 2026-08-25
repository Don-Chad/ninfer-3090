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

## Validation

The release build and smoke gate uses these components:

- Ubuntu 24.04
- CUDA Toolkit 13.1
- GCC 13
- CMake and Ninja
- FFmpeg and curl system packages

Each packaged `ninfer` binary completes a real RTX 3090 Qwen3.8-27B generation before publication.

## Limits

- This release is a correctness and distribution update, not a new throughput claim.
- Keep the GPU idle while loading the 27B model; a 24 GB RTX 3090 needs the
  documented explicit KV settings for higher-concurrency profiles.
