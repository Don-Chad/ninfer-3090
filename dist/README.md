# RTX 3090 release bundles

For v0.4.0, run the Windows-only packaging script from the repository root after the verified
native build exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release-v040.ps1
```

It creates a versioned directory and archive under `dist/`:

- `ninfer-rtx3090-windows-x64-*`: native Windows CLI, server, benchmark, and vcpkg DLLs;
- `SHA256SUMS-v0.4.0.txt`: archive hash for release verification.

Generated binaries and archives are ignored by Git because GitHub source repositories should not
contain build products. Upload the `.zip` and versioned checksum file as GitHub Release assets.
The packaging guide itself is tracked.

Model artifacts are not included. Download either the 16.29 GiB `qwen3_6_27b.ninfer` or 20.84 GiB
`qwen3_6_35b_a3b.ninfer` artifact from the repositories linked in the project README and verify its
published SHA-256 separately. The compact 35B artifact is text-only; leave `--vision` disabled.

The Windows bundle includes its FFmpeg/curl/zlib DLLs and requires the NVIDIA driver and Microsoft
Visual C++ 2022 runtime.
