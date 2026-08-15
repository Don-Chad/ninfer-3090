# NInfer-3090 v0.6.1

This release adds a Linux source and container build for the RTX 3090 and RTX 3090 Ti.
It does not change model execution, artifact compatibility, or the qualified Windows performance profiles.

## Changes

- Adds an Ubuntu 24.04 and Docker build guide for `sm_86`.
- Supports the pinned vcpkg manifest on Linux.
- Documents Linux and Windows as separate delivery paths.
- Adds Bash download, launcher, and historical packaging scripts.
- Keeps the existing pkg-config dependency path for normal Linux packages.
- Keeps all RTX 3090 kernel schedules and memory profiles unchanged.

## Linux validation

The Docker build completed all 245 compile and link steps with these components:

- Ubuntu 24.04
- CUDA Toolkit 13.1
- GCC 13
- CMake and Ninja
- FFmpeg and curl system packages

The resulting `ninfer` and `ninfer-serve` applications returned their `--help` output with GPU access enabled.

## Limits

- The project does not publish a prebuilt Linux archive.
- A real model artifact was not available for the Linux acceptance run.
- This release does not publish Linux throughput or memory results.
- The Windows v0.6.0 model and performance evidence remains the current qualified runtime evidence.
