# RTX 3090 release bundles

Run the packaging script from the repository root after both verified builds exist:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release.ps1
```

Runtime revision `0.2.0-rtx3090-v2` creates two explicitly versioned directories and archives
under `dist/`:

- `ninfer-rtx3090-windows-x64-0.2.0-rtx3090-v2`: native Windows CLI, server, benchmark, vcpkg
  DLLs, and `VERSION` marker;
- `ninfer-rtx3090-linux-x64-0.2.0-rtx3090-v2`: Ubuntu 24.04 / WSL2 CLI, server, benchmark, and
  `VERSION` marker;
- `SHA256SUMS.txt`: archive hashes for release verification.

Generated binaries and archives are ignored by Git because GitHub source repositories should not
contain build products. Upload the `.zip`, `.tar.gz`, and `SHA256SUMS.txt` as GitHub Release assets.
The packaging guide itself is tracked.

Model artifacts are not included. Download either the 16.29 GiB `qwen3_6_27b.ninfer` or 20.84 GiB
`qwen3_6_35b_a3b.ninfer` artifact from the repositories linked in the project README and verify its
published SHA-256 separately. The 35B-A3B artifact requires `--text-only` on an RTX 3090.

The Linux bundle targets Ubuntu 24.04 and dynamically uses the CUDA 12 runtime, FFmpeg 6, curl, and
the standard C++ runtime from the target system. The Windows bundle statically includes its CUDA
runtime and packages its FFmpeg/curl/zlib DLLs; users need a current NVIDIA driver compatible with
CUDA 13.x and the Microsoft Visual C++ 2022 runtime, but not the full CUDA Toolkit. Building either
platform from current source requires CUDA Toolkit 13.0 or newer.
