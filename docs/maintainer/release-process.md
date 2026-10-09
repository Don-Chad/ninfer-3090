# Cutting a release

How a `vX.Y.Z-rtx3090` release is built, packaged, verified and published from this repository. The
steps are short; the gotchas below are the reason to read the page, because each one cost real time
or very nearly shipped a broken archive.

## The flow

1. **Prepare.** Branch `release/vX.Y.Z` from `master`. Set `VERSION` to the full tag (for example
   `0.11.0-rtx3090`), write `RELEASE_NOTES_X.Y.Z.md`, refresh the README highlights, open a PR and
   merge it. Everything version-specific is derived from `VERSION` by the packagers; there is nothing
   to copy and edit.
2. **Build both platforms from the same commit, after `VERSION` is set.** The binaries embed
   `VERSION` (`--version`, `GET /health`, the `X-NInfer-Version` header), so build from the release
   branch tip, not from a commit before the `VERSION` bump. A binary built from the pre-merge
   release-branch commit reports `<VERSION>+<commit>`; the tag lands on the merge commit, so that
   suffix is expected there, but a build from `master` before the bump would report the previous
   release. Check `ninfer-serve --version` against the archive name in the smoke test.
   - Windows: the VS 2022 BuildTools environment and an existing Ninja tree (see `AGENTS.md`,
     "Windows build environment"). Turn `NINFER_BUILD_BENCHMARKS` on: the archive ships
     `ninfer_bench.exe`. Build only `ninfer ninfer-serve ninfer_bench`.
   - Linux: under WSL, from a **fresh** `git archive` of the commit, targets `ninfer ninfer-serve
     ninfer_bench`, `-DCMAKE_CUDA_ARCHITECTURES=86`, `-DBUILD_TESTING=OFF` (the release only needs the
     products). See the WSL notes below.
3. **Package.** `scripts/package-release.ps1` (Windows zip) and `scripts/package-release.sh` (Linux
   tar.gz), or `build.ps1 -Package` / `build.sh --package`. Each writes the archive and
   `SHA256SUMS-vX.Y.Z-{windows,linux}.txt` under `dist/`.
4. **Smoke-test on the real machine** (below), not just the unit tests.
5. **Publish.** Merge the release PR, tag the merge commit (annotated), push the tag, then
   `gh release create vX.Y.Z-rtx3090 <zip> <tar.gz> <both SHA256SUMS files> -R ashalliants/ninfer-3090
   --notes-file RELEASE_NOTES_X.Y.Z.md --title "..." --latest`.

## Gotchas

**Always pass `-R ashalliants/ninfer-3090` to `gh`.** With no `-R` it resolves to the `upstream` remote
(`Don-Chad/ninfer-3090`) and silently reads or writes the wrong fork. The same applies to the babysit
helper (also set `GH_REPO`).

**Windows executables import cuBLAS, and the archive must carry it.** The cuBLAS prefill route makes
every executable import `cublas64_12.dll` (which imports `cublasLt64_12.dll`). A machine without the
CUDA Toolkit then fails to start any of them (`0xC0000135`), even when `--prefill-cublas` is never used.
The developer machine hides this because the Toolkit is on its `PATH`. `package-release.ps1` copies both
DLLs from `%CUDA_PATH%\bin` and **refuses to produce an archive whose executables do not start with only
Windows on `PATH`**, so this cannot regress silently. Do not weaken that check.

**Redistributing NVIDIA's DLLs has conditions.** The CUDA Toolkit EULA (Attachment A) lists the cuBLAS
DLLs as redistributable, including versioned names such as `cublas64_12.dll`. The grant requires that our
application has material functionality of its own, that the DLLs are used only by it, and that our
distribution terms are consistent with the EULA and protect NVIDIA's rights. The packager therefore ships
the toolkit's `EULA.txt` as `NVIDIA-CUDA-EULA.txt`, and the archive README names the two DLLs as NVIDIA's,
outside the project's Apache-2.0 licence. Re-read the EULA of the toolkit that builds a release.

**Linux links cuBLAS, FFmpeg and glibc dynamically.** The Linux archive does not ship `libcublas` (the
Lt library alone is about 500 MB), so `docs/release-archive-linux.md` states the requirements: glibc 2.38
or newer (built on Ubuntu 24.04), the CUDA 12.8 runtime libraries including cuBLAS, FFmpeg 6 and libcurl.
Check them with `ldd` on the new binaries and update the guide if they change.

**Asset sizes.** Each executable is about 460-490 MB (every kernel is linked in). The Windows zip is about
1.5 GB with the cuBLAS DLLs and the Linux tar.gz about 1 GB. GitHub rejects release assets over 2 GiB, so
watch the Windows zip if more DLLs are added.

**Smoke-test the launcher default on the real card.** A busy desktop holding 2-3 GiB of the 3090 refuses
the default DFlash2 profile on the engine's runtime reservation. `run.{sh,bat}` therefore steps down by
itself when startup is refused for lack of GPU memory or fails to pin the Host context budget; see `docs/maintainer/launcher-profiles.md`. `--kv-capacity
auto` does not avoid this, because the engine still reserves room for one full `--max-context` sequence.
Start the launcher with nothing overridden and confirm it serves a request.

### Building on WSL

- **Use a fresh export**, not an old snapshot: `git archive` the commit to a new directory, so stale files
  cannot leak in. To add later documentation or script changes, extract only those paths. Extracting a
  whole `git archive` over a build tree stamps every source file with the commit time and forces a full
  rebuild.
- **`ninja: fatal: posix_spawn: Resource temporarily unavailable`** means the session's cgroup hit its
  process cap, not that `-j` is too high. Look at `/sys/fs/cgroup/init.scope/pids.{max,current}` with shell
  built-ins (a failing `fork` breaks `ps`). When leftover processes from other tooling fill it (they did:
  about thirty `pnpm`/`node` processes with 67 threads each), do not kill them; run the build in its own
  scope instead: `wsl.exe -u root -e systemd-run --scope --quiet --uid=<you> --gid=<you> -p
  TasksMax=infinity -- bash -lc '...'`.
- **Background long jobs with the harness, not the shell.** `wsl.exe -e bash -lc "cmd & disown"` does not
  survive; run the whole `wsl.exe` invocation as a background task and read its log.
- **A model on `/mnt/c` loads at about 200 MiB/s**, so a 19 GB artifact takes minutes per attempt. Copy it
  into the WSL filesystem for repeated runs.
- `BUILD_TESTING` is `OFF` by default; `ctest` then reports "No tests were found". Turn it on to run the
  suite, which the release itself does not need on Linux.

### Windows notes

- Reconfiguring the existing Ninja tree is the one thing that breaks it; a second tree configured against
  a worktree (with the vcpkg installed directory reused read-only and `-DVCPKG_MANIFEST_INSTALL=OFF`) is
  safe. A full ops build takes about ten minutes; incremental rebuilds under a minute.
- Extract the finished zip to a temporary folder and run each executable with only Windows on `PATH`
  before uploading, even though the packager now does the same check.
- `cmd` batch files: a trailing `^` on a `rem` line continues the comment onto the next line, an unquoted
  `>>` in `set /a` is a redirect, and `Tee-Object` writes UTF-16 (`findstr` cannot search it).

## Checks before publishing

- `scripts/check-linux-scripts.sh` passes.
- Both archives: every file matches its inner `SHA256SUMS.txt`, and the outer checksum file verifies.
- Windows: all three executables start with only Windows on `PATH`.
- Linux: a fresh unpack runs `./ninfer-serve --help` and `./run.sh --help`.
- The default launcher profile starts and serves a chat completion on the real 3090.
