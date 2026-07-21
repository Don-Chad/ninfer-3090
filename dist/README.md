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

The Linux bundle targets Ubuntu 24.04 and dynamically uses CUDA 12, FFmpeg 6, curl, and the standard
C++ runtime. The Windows bundle includes its FFmpeg/curl/zlib DLLs and requires the NVIDIA driver
and Microsoft Visual C++ 2022 runtime.
