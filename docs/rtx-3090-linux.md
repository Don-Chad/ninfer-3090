# Build NInfer for RTX 3090 on Linux

This guide builds the `sm_86` runtime for one NVIDIA GeForce RTX 3090 or RTX 3090 Ti.
The project does not publish a qualified Linux binary release yet.

Do not change `CMAKE_CUDA_ARCHITECTURES` to `89`.
The RTX 4090 fork uses Ada-specific schedules that do not apply to the RTX 3090.

## Container build

The repository Dockerfile gives the shortest build path on Bazzite and other Linux distributions.
It uses Ubuntu 24.04, CUDA 13.1, GCC 13, Ninja, FFmpeg, and curl.

Install Docker and the NVIDIA Container Toolkit first.
Then make sure that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.1.2-runtime-ubuntu24.04 nvidia-smi
```

Build the image from the repository root:

```bash
docker build --tag ninfer-3090:sm86 .
```

Run the Qwen3.8 server with a model directory from the host:

```bash
docker run --rm --gpus all \
  --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-3090:sm86 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 65536 --kv-capacity 65536 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The API is available at `http://127.0.0.1:8080/v1`.

## Native Ubuntu 24.04 build

Install the CUDA Toolkit 12.8 or newer from NVIDIA.
Then install the host compiler and media dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential gcc-13 g++-13 cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

Select GCC 13 for host and CUDA compilation:

```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13
```

Configure and build the Linux applications:

```bash
cmake -S . -B build-sm86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build-sm86 --parallel 2
```

The build creates these applications:

```text
build-sm86/apps/ninfer
build-sm86/apps/ninfer-serve
```

## Optional vcpkg dependencies

Linux can use the pinned `vcpkg.json` manifest instead of system FFmpeg and curl packages.
Bootstrap the pinned vcpkg revision:

```bash
git clone https://github.com/microsoft/vcpkg.git "$HOME/.local/share/vcpkg"
git -C "$HOME/.local/share/vcpkg" checkout 4bca8fd8654e5ba76f92661db7bfe954768ad8ef
"$HOME/.local/share/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
```

Add these options to the native CMake command:

```text
-DCMAKE_TOOLCHAIN_FILE=$HOME/.local/share/vcpkg/scripts/buildsystems/vcpkg.cmake
-DVCPKG_TARGET_TRIPLET=x64-linux
```

## Bash scripts

The `scripts/` directory contains Bash versions of each Windows download, launcher, and packaging
script. Download scripts save models under `scripts/models` by default:

```bash
./scripts/download-qwen38.sh
./scripts/download-qwen36-35b-vision.sh
```

Set `NINFER_MODEL_DIR` to use another model directory. Each launcher also accepts a model path:

```bash
./scripts/run-qwen38-c1.sh /path/to/qwen3_8_27b.ninfer
./scripts/run-qwen38-c8.sh /path/to/qwen3_8_27b.ninfer
./scripts/run-qwen38-vision.sh /path/to/qwen3_8_27b.ninfer
./scripts/run-qwen36-35b-vision.sh /path/to/qwen3_6_35b_a3b.ninfer
```

The launchers use `build-sm86/apps/ninfer-serve` when the executable is not beside the script.
Set `NINFER_SERVER` to select another executable.

The `package-release-v*.sh` scripts create Linux `tar.gz` archives from their matching historical
build directories. The PowerShell scripts continue to create Windows ZIP archives.

## Validation

Make sure that the applications start:

```bash
./build-sm86/apps/ninfer --help
./build-sm86/apps/ninfer-serve --help
```

Run one short generation with the real Qwen3.8 artifact:

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in two sentences." \
  --max-context 8192 --max-new 32 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

A successful compile does not qualify Linux performance.
Record the GPU, driver, CUDA Toolkit, compiler, workload, and result before you publish Linux measurements.
