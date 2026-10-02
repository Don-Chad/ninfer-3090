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

      # Target architecture of the packaged build: 86 for the RTX 3090 this fork is tuned on. The
      # CMake configure also accepts 80 (unmeasured GA100 compatibility target) and 89; change this
      # one binding to package for another.
      cudaArch = "86";

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

        # CMakeLists.txt accepts CMAKE_CUDA_ARCHITECTURES 80, 86 or 89 (cudaArch above) and
        # requires CUDA >= 12.8. gcc13 matches the upstream-validated
        # Linux toolchain and stays inside nvcc 12.9's supported host-compiler
        # range. CC/CXX come from gcc13Stdenv, so CUDA host code uses the same
        # compiler as the C++ side.
        cmakeFlags = [
          "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
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
        {
          name,
          filename,
          url,
          description,
          revision ? null,
          expectedSize ? null,
          expectedSha256 ? null,
        }:
        pkgs.writeShellScriptBin name ''
          set -euo pipefail
          expected_size='${if expectedSize == null then "" else toString expectedSize}'
          expected_sha256='${if expectedSha256 == null then "" else expectedSha256}'
          stage='${if revision == null then "latest" else revision}'
          resumable=${if revision == null then "0" else "1"}

          model_dir=''${NINFER_MODEL_DIR:-$HOME/models}
          model="$model_dir/${filename}"

          # curl -C - resumes by appending at the current file length, without checking what wrote
          # those bytes. Downloading straight onto "$model" therefore splices a leftover partial
          # from one revision into another whenever a pin changes -- a file of plausible size that
          # is corrupt throughout. Staging under a name that carries the revision means a resume
          # can only ever continue the same artifact -- but an unpinned URL (revision == null,
          # resolving whatever upstream currently calls "main") has no immutable name to stage
          # under: "main" can move between two invocations of this same command, so a leftover
          # ".latest.part" could belong to an older "main" than the one this run would fetch.
          # Unpinned entries therefore never resume: any existing partial is discarded first and
          # the download restarts from zero, at the cost of resumability.
          part="$model.$stage.part"
          mkdir -p "$model_dir"

          # Verifies a path against expected_size and, unless NINFER_SKIP_SHA256=1,
          # expected_sha256 (either check is skipped if its expected value is empty, i.e. for the
          # unpinned entries, which supply neither). Used both for an existing "$model" -- so a
          # same-sized-but-corrupt file is not accepted forever just because it happened to pass
          # once, or was replaced out from under this script -- and for a freshly downloaded
          # "$part", as one check that cannot drift out of sync with itself.
          verify() {
            if [ -n "$expected_size" ]; then
              [ "$(wc -c < "$1" | tr -d '[:space:]')" = "$expected_size" ] || return 1
            fi
            if [ -n "$expected_sha256" ] && [ "''${NINFER_SKIP_SHA256:-0}" != '1' ]; then
              local actual_sha256
              actual_sha256="$(sha256sum -- "$1" | cut -d' ' -f1)"
              [ "$actual_sha256" = "$expected_sha256" ] || return 1
            fi
            return 0
          }

          # Only short-circuit when there is metadata to verify against: an unpinned entry
          # (expected_size empty) has nothing for verify() to check, so it would return success
          # for any existing regular file -- including an empty or corrupt one left over from an
          # interrupted run -- and this path explicitly does not resume those (see above).
          if [ -n "$expected_size" ] && [ -f "$model" ] && verify "$model"; then
            echo "Model already present: $model"
            exit 0
          fi

          if [ "$resumable" = "1" ]; then
            echo "Downloading ${description} to $model (resumable)..."
            ${pkgs.curl}/bin/curl -L -C - --fail --output "$part" '${url}'
          else
            echo "Downloading ${description} to $model (unpinned URL, not resumable)..."
            rm -f -- "$part"
            ${pkgs.curl}/bin/curl -L --fail --output "$part" '${url}'
          fi

          if ! verify "$part"; then
            echo "Downloaded file at $part failed verification. Delete it and retry." >&2
            exit 1
          fi

          mv -f -- "$part" "$model"
          echo "Model ready: $model"
        '';

      # Qwen3.8-27B (this fork's qwen3_8_27b conversion, with vision, MTP and DFlash2), pinned to
      # match `scripts/download-model.{sh,bat} qwen38-27b`. The size and sha256 are what the
      # revision's artifact-manifest.json states.
      download-qwen38-27b = mkDownload {
        name = "download-qwen38-27b";
        filename = "qwen3_8_27b.ninfer";
        revision = "d47f2732d369acaec76dc44228f67c72b081df2f";
        url = "https://huggingface.co/WarlaxZ/Qwen3.8-27B-NInfer-3090/resolve/d47f2732d369acaec76dc44228f67c72b081df2f/qwen3_8_27b.ninfer";
        expectedSize = 18982458624;
        expectedSha256 = "7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706";
        description = "Qwen3.8-27B NInfer model";
      };

      # Qwen3.6-27B (groupwise artifact), pinned to match `scripts/download-model.sh qwen36-27b`.
      download-qwen36-27b = mkDownload {
        name = "download-qwen36-27b";
        filename = "qwen3_6_27b.ninfer";
        revision = "3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829";
        url = "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829/qwen3_6_27b.ninfer";
        expectedSize = 17495538688;
        expectedSha256 = "9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b";
        description = "Qwen3.6-27B NInfer model";
      };

      # Qwen3.6-35B-A3B (official v3 artifact, with the DFlash bundle), pinned to match
      # `scripts/download-model.sh qwen36-35b-a3b`. The DFlash bundle costs nothing in VRAM unless
      # --spec dflash is selected, so the published RTX 3090 concurrency measurements -- taken
      # against the v2 pins c8b8c1c0/560f227e, whose weight bytes v3 preserves -- still apply.
      # See README.md and docs/rtx-3090-windows.md.
      download-qwen36-35b = mkDownload {
        name = "download-qwen36-35b";
        filename = "qwen3_6_35b_a3b.ninfer";
        revision = "ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6";
        url = "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6/qwen3_6_35b_a3b.ninfer";
        expectedSize = 22790484480;
        expectedSha256 = "3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84";
        description = "Qwen3.6-35B-A3B (pinned, with DFlash) model";
      };

      # Whatever the repository currently calls main. Kept as an escape hatch for trying a newer
      # upstream artifact, so it is deliberately unpinned and deliberately writes to its own
      # filename: it is not the measured profile and it is not what the tests expect. Now that
      # download-qwen36-35b carries DFlash, this no longer exists to supply it.
      #
      # It is now the only downloader in the tree that cannot verify what it fetches, and that is
      # inherent rather than an oversight: `main` can move between runs, so there is no size or
      # hash to state and no safe way to resume a partial file. mkDownload therefore refuses to
      # resume this one at all (see `resumable` above) -- an interrupted fetch restarts from zero
      # rather than being spliced. A truncated download that still reaches the end of the stream
      # cannot be detected here. Prefer download-qwen36-35b unless you are specifically testing a
      # newer upstream artifact.
      download-qwen36-35b-v2 = mkDownload {
        name = "download-qwen36-35b-v2";
        filename = "qwen3_6_35b_a3b_v2.ninfer";
        url = "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/main/qwen3_6_35b_a3b.ninfer";
        description = "Qwen3.6-35B-A3B upstream main (unpinned) model";
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
          meta.description = "Download the pinned Qwen3.6-35B-A3B .ninfer artifact, with DFlash (21 GB)";
        };
        download-qwen36-35b-v2 = {
          type = "app";
          program = "${download-qwen36-35b-v2}/bin/download-qwen36-35b-v2";
          meta.description = "Download the Qwen3.6-35B-A3B upstream main (unpinned) .ninfer artifact (21 GB)";
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
          echo "  nix run .#download-qwen38-27b            -> Qwen3.8-27B artifact (19 GB)"
          echo "  nix run .#download-qwen36-27b            -> Qwen3.6-27B artifact"
          echo "  nix run .#download-qwen36-35b           -> Qwen3.6-35B-A3B pinned, with DFlash (21 GB)"
          echo "  nix run .#download-qwen36-35b-v2       -> Qwen3.6-35B-A3B upstream main, unpinned (21 GB)"
        '';
      };
    };
}