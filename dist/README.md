# RTX 3090 release bundles

Run the Windows packaging script from the repository root after the verified native build exists.
It reads the release from `VERSION` and needs the matching `RELEASE_NOTES_<version>.md`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release.ps1
```

It creates a versioned directory and archive under `dist/`:

- `ninfer-rtx3090-windows-x64-*`: native Windows CLI, server, benchmark, and vcpkg DLLs;
- `SHA256SUMS-v<version>-windows.txt`: archive hash for release verification.

Generated binaries and archives are ignored by Git because GitHub source repositories should not
contain build products. Upload the `.zip` and versioned checksum file as GitHub Release assets.
The packaging guide itself is tracked.

Model artifacts are not included. Download the recommended `qwen3_8_27b.ninfer` (Qwen3.8-27B) with
`download-model.bat qwen38-27b`, or one of the other artifacts listed in the project README's Models
table; the downloader pins a revision and verifies size and SHA-256.

The Windows bundle includes its FFmpeg/curl/zlib DLLs and requires the NVIDIA driver and Microsoft
Visual C++ 2022 runtime.
