{
  description = "NInfer-3090: SM86 (RTX 3090) inference engine for Qwen3.8-27B and Qwen3.6";

  # Pin the nixpkgs revision currently used on this machine (CUDA 12.9 /
  # cuda-merged-12.9, gcc13Stdenv, ffmpeg 6.1). CUDA must be >= 12.8 and the
  # nvcc host compiler must stay within CUDA 12.9's supported GCC range (<= 14),
  # so the build pins gcc13 instead of following the default (gcc15) stdenv.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/241313f4e8e508cb9b13278c2b0fa25b9ca27163";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # NVIDIA CUDA toolkit components
      };

      cuda = pkgs.cudaPackages.cudatoolkit; # cuda-merged-12.9 (nvcc >= 12.8)

      ninfer = pkgs.gcc13Stdenv.mkDerivation {
        pname = "ninfer-3090";
        version = "0.6.1";
        src = self;

        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          cuda
        ];

        buildInputs = [
          pkgs.ffmpeg_6-full # libavformat60/avcodec60/avutil58/swscale7, as validated in the container
          pkgs.curl
        ];

        # The project hard-requires CMAKE_CUDA_ARCHITECTURES=86 (see
        # CMakeLists.txt) and CUDA >= 12.8. gcc13 matches the upstream-validated
        # Linux toolchain and stays inside nvcc 12.9's supported host-compiler
        # range. CC/CXX come from gcc13Stdenv, so CUDA host code uses the same
        # compiler as the C++ side.
        cmakeFlags = [
          "-DCMAKE_CUDA_ARCHITECTURES=86"
          "-DCMAKE_CUDA_COMPILER=${cuda}/bin/nvcc"
          "-DCUDAToolkit_ROOT=${cuda}"
          "-DCMAKE_BUILD_TYPE=Release"
          "-DNINFER_BUILD_APPS=ON"
          "-DBUILD_TESTING=OFF"
          "-DNINFER_BUILD_BENCHMARKS=OFF"
        ];

        # Ninja job pool for single CUDA device-link slot.
        enableParallelBuilding = true;

        # nixpkgs' cmake helper builds in-source by default, so the executables
        # land directly under apps/.
        installPhase = ''
          runHook preInstall
          install -Dm755 apps/ninfer $out/bin/ninfer
          install -Dm755 apps/ninfer-serve $out/bin/ninfer-serve
          runHook postInstall
        '';

        meta = with pkgs.lib; {
          description = "SM86 CUDA inference engine for Qwen3.8-27B / Qwen3.6 on one RTX 3090";
          homepage = "https://github.com/Don-Chad/ninfer-3090";
          license = licenses.asl20;
          platforms = [ "x86_64-linux" ];
          maintainers = [ ];
        };
      };

      # One resumable downloader per registered artifact, mirroring
      # scripts/download-*.sh. NINFER_MODEL_DIR overrides the target directory.
      mkDownload =
        { name, filename, url, description }:
        pkgs.writeShellScriptBin name ''
          set -euo pipefail
          model_dir=''${NINFER_MODEL_DIR:-$HOME/models}
          model="$model_dir/${filename}"
          mkdir -p "$model_dir"
          echo "Downloading ${description} to $model (resumable)..."
          ${pkgs.curl}/bin/curl -L -C - --fail --output "$model" '${url}'
          echo "Model ready: $model"
        '';

      # Qwen3.8-27B (official artifact; validated at C1/C2/C4/C8 on RTX 3090).
      download-qwen38-27b = mkDownload {
        name = "download-qwen38-27b";
        filename = "qwen3_8_27b.ninfer";
        url = "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer";
        description = "Qwen3.8-27B NInfer model";
      };

      # Qwen3.6-27B (groupwise artifact).
      download-qwen36-27b = mkDownload {
        name = "download-qwen36-27b";
        filename = "qwen3_6_27b.ninfer";
        url = "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/main/qwen3_6_27b.ninfer";
        description = "Qwen3.6-27B NInfer model";
      };

      # Qwen3.6-35B-A3B compact v1 (pinned revision; the measured 24 GB profile).
      download-qwen36-35b = mkDownload {
        name = "download-qwen36-35b";
        filename = "qwen3_6_35b_a3b.ninfer";
        url = "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/c8b8c1c0df4c74df3c190c6aa3a7f24dc614721c/qwen3_6_35b_a3b.ninfer";
        description = "Qwen3.6-35B-A3B compact v1 (pinned) model";
      };

      # Qwen3.6-35B-A3B upstream v2 (includes DFlash payload; not the measured artifact).
      download-qwen36-35b-v2 = mkDownload {
        name = "download-qwen36-35b-v2";
        filename = "qwen3_6_35b_a3b_v2.ninfer";
        url = "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/main/qwen3_6_35b_a3b.ninfer";
        description = "Qwen3.6-35B-A3B upstream v2 (with DFlash) model";
      };
    in
    {
      packages.${system} = {
        default = ninfer;
        inherit ninfer;
        inherit download-qwen38-27b download-qwen36-27b download-qwen36-35b download-qwen36-35b-v2;
      };

      apps.${system} = {
        default = {
          type = "app";
          program = "${ninfer}/bin/ninfer-serve";
          meta.description = "NInfer-3090 OpenAI/Anthropic HTTP server";
        };
        cli = {
          type = "app";
          program = "${ninfer}/bin/ninfer";
          meta.description = "NInfer-3090 one-shot CLI generation";
        };
        serve = {
          type = "app";
          program = "${ninfer}/bin/ninfer-serve";
          meta.description = "NInfer-3090 OpenAI/Anthropic HTTP server";
        };
        download-qwen38-27b = {
          type = "app";
          program = "${download-qwen38-27b}/bin/download-qwen38-27b";
          meta.description = "Download the Qwen3.8-27B .ninfer artifact (17 GB)";
        };
        download-qwen36-27b = {
          type = "app";
          program = "${download-qwen36-27b}/bin/download-qwen36-27b";
          meta.description = "Download the Qwen3.6-27B .ninfer artifact (17 GB)";
        };
        download-qwen36-35b = {
          type = "app";
          program = "${download-qwen36-35b}/bin/download-qwen36-35b";
          meta.description = "Download the Qwen3.6-35B-A3B compact v1 .ninfer artifact (21 GB)";
        };
        download-qwen36-35b-v2 = {
          type = "app";
          program = "${download-qwen36-35b-v2}/bin/download-qwen36-35b-v2";
          meta.description = "Download the Qwen3.6-35B-A3B upstream v2 .ninfer artifact (21 GB)";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        inputsFrom = [ ninfer ];
        # Make the CUDA runtime and the NVIDIA driver's libcuda discoverable for
        # ad-hoc testing of store-built binaries from inside the shell.
        shellHook = ''
          export CUDA_PATH=${cuda}
          export LD_LIBRARY_PATH=${cuda}/lib:$LD_LIBRARY_PATH
          # libcuda.so.1 ships with the NVIDIA driver on NixOS
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
          fi
          echo "NInfer-3090 development shell (CUDA 12.9, sm_86)"
          echo "  nix build                                -> build ninfer + ninfer-serve"
          echo "  nix run .#serve -- <serve args>          -> run the HTTP server"
          echo "  nix run .#download-qwen38-27b            -> Qwen3.8-27B artifact (17 GB)"
          echo "  nix run .#download-qwen36-27b            -> Qwen3.6-27B artifact"
          echo "  nix run .#download-qwen36-35b           -> Qwen3.6-35B-A3B compact v1 (21 GB)"
          echo "  nix run .#download-qwen36-35b-v2       -> Qwen3.6-35B-A3B upstream v2 (21 GB)"
        '';
      };
    };
}