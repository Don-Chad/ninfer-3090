# RTX 3090 native Windows build

This is the verified native Windows build of the Qwen3.6-27B and text-only Qwen3.6-35B-A3B RTX
3090 port. It uses MSVC and CUDA
directly; WSL is not involved at runtime. The Linux/WSL build remains documented separately in
[rtx-3090-wsl.md](rtx-3090-wsl.md).

For the fastest verified 35B-A3B configuration, use INT8 KV, MTP K=2, the optimized head, and
`--text-only`. The current controlled tg128 result is 265.59 +/- 0.71 tok/s; see
[the full 35B-A3B report](rtx-3090-35b-a3b.md). Text-only mode rejects media and is required for
the full artifact to fit in 24 GiB.

## Verified environment

- Windows 11
- GeForce RTX 3090, compute capability 8.6, 24 GB
- Visual Studio 2022 Community, MSVC 19.44
- CUDA Toolkit 13.2.78
- CMake 4.0.1 using the `Visual Studio 17 2022` generator
- Visual Studio's bundled vcpkg
- vcpkg FFmpeg 8.1.2 with the `zlib` feature and curl 8.21

The source requires CUDA Toolkit 13.0 or newer but deliberately rejects GPU architectures other than
`sm_86`. CUDA 13 requires MSVC's conforming preprocessor; the build sets it automatically.

## Configure and build

The manifest pins its vcpkg registry baseline and explicitly enables `ffmpeg[zlib]`. The first
configure can take several minutes while vcpkg builds FFmpeg. Later builds use vcpkg's binary
cache.

From a normal PowerShell terminal at the repository root:

```powershell
$vcpkgToolchain = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\vcpkg\scripts\buildsystems\vcpkg.cmake'
$cudaToolkit = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'

cmake -S . -B build-windows `
  -G 'Visual Studio 17 2022' -A x64 -T "cuda=$cudaToolkit" `
  -DCMAKE_TOOLCHAIN_FILE="$vcpkgToolchain" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DNINFER_BUILD_APPS=ON `
  -DBUILD_TESTING=ON `
  -DNINFER_BUILD_BENCHMARKS=ON `
  -DNINFER_BUILD_TOOLS=ON

cmake --build build-windows --config Release --parallel
```

The runnable products and their required vcpkg DLLs are copied to:

```text
build-windows\apps\Release\ninfer.exe
build-windows\apps\Release\ninfer-serve.exe
```

## Test

The row-split cross-language golden test requires CPU PyTorch, NumPy, and Safetensors. Keep this
environment separate from the WSL test environment:

```powershell
uv venv .venv-windows-tests --python 3.12
uv pip install --python .venv-windows-tests\Scripts\python.exe torch `
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-windows-tests\Scripts\python.exe numpy safetensors

$testPython = (Resolve-Path .venv-windows-tests\Scripts\python.exe).Path -replace '\\', '/'
cmake -S . -B build-windows "-DPython3_EXECUTABLE=$testPython"
cmake --build build-windows --config Release --parallel

$env:NINFER_QWEN3_6_27B_WEIGHTS = (Resolve-Path models\qwen3_6_27b.ninfer)
ctest --test-dir build-windows -C Release --output-on-failure -j 1
```

The final verified run passed all 60 registered tests in 180.27 seconds. The real 27B integration test
passed; only the optional 35B real-artifact test skipped because that artifact was not present.
Tests are intentionally serialized so only one CUDA test owns the GPU at a time.

## Run

The recommended native 3090 configuration is optimized MTP-3:

```powershell
build-windows\apps\Release\ninfer.exe models\qwen3_6_27b.ninfer `
  --prompt 'Explain prefill and decode in three sentences.' `
  --max-context 8192 --max-new 128 --kv-dtype bf16 `
  --mtp-draft-tokens 3 --lm-head-draft
```

For more KV capacity within 24 GB, use `--kv-dtype int8` and size `--max-context` for the actual
workload.

## Native benchmark

The controlled comparison used an 8,192-token capacity, BF16 KV, five measured repetitions after
one warm-up, pp512, and tg128:

```powershell
build-windows\bench\Release\ninfer_bench.exe `
  --weights models\qwen3_6_27b.ninfer `
  -p 512 -n 128 -r 5 --warmup 1 `
  --max-ctx 8192 --kv-dtype bf16 `
  --mtp-draft-tokens 3 --lm-head-draft `
  --output json --output-file profiles\bench\windows-3090-mtp3.json
```

Measured results:

| Configuration | Load | Upload | pp512 | tg128 | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| MTP disabled | 4.34 s | 3.50 s | 957.51 tok/s | 38.14 tok/s | n/a |
| optimized MTP-3 | 5.23 s | 3.47 s | 936.64 tok/s | **64.23 tok/s** | 60.7% |

The MTP-3 run reserved 16.28 GiB for weights, 1.26 GiB for sequence state, 0.53 GiB for KV
payload, and 1.90 GiB for reusable workspace. Peak measured decode workspace was 1.16 MiB.

Native CUDA 13.2 differs slightly from the CUDA 12.8 WSL build. The exact Q4 34816x5120 T=2
candidate measured 197.63 microseconds with R8C4 versus 203.78 microseconds with R4C4, so Windows
uses R8C4 for that one shape. The end-to-end MTP-1 change was small: 55.53 to 55.57 tok/s. T=3/4
retain R4C4. The GDN selections also remained correct natively: `small-split10` measured 10.24
microseconds at T=4 and cooperative split-8 measured 22.53 microseconds at T=512.

## Windows implementation notes

- Artifact metadata uses `CreateFileMappingW`/`MapViewOfFile`; staged weight uploads use a separate
  `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` handle with 4,096-byte alignment.
- Remote-media host validation initializes Winsock and links `ws2_32`.
- Windows uses vcpkg's configuration-aware FFmpeg and curl CMake integration. This prevents a
  Release executable from accidentally depending on `libcurl-d.dll`.
- The built `avcodec-62.dll` imports `z.dll`, confirming that FFmpeg's zlib feature is active.
- The CUDA runtime is linked statically while MSVC and vcpkg use `/MD`; the final link explicitly
  discards CUDA's conflicting `LIBCMT` default request so only one C runtime is present.
