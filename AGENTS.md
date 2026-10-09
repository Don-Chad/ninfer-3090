# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For implementation tasks, record the user's requirements, promised work, affected implementations
and routes, and the quality dimensions to be evaluated before judging results. For comparative
claims, also identify the baseline, workloads, metrics, aggregation method, and acceptance criteria.
Keep this accounting current as the task develops. Explain justified changes and preserve the
earlier findings; never narrow the scope or change the criteria after seeing unfavorable results
to make the task appear successful. This accounting may live in the working conversation or
existing task artifacts; it does not require a new permanent planning document.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

Requests for analysis or proposals for review do not authorize applying them;
implementation authorization must cover that same scope.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance, with an
optional layer pipeline across several GPUs on Linux. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path.
This fork targets **`sm_86`** and is tuned on **NVIDIA GeForce RTX 3090** (24 GB), built with
CUDA 12.8. Upstream (`Neroued/ninfer`) targets `sm_120a` on RTX 5090; that is where its schedules,
route tables and published measurements come from, and none of it is authoritative here -- every
route table this fork inherited and re-measured on sm_86 turned out to be wrong by 12-41%. Treat an
upstream tuning constant as a hypothesis until measured on this card. The build environment and the
compatibility constraints are in "Windows build environment (RTX 3090 fork host)" below.

Generation uses one resident model on one GPU, one to eight resident execution lanes fixed at
startup, bounded FIFO ingress with finite bypass, resource-pressure preemption with
Snapshot/Replay recovery, and one compact decode batch per round. The layer pipeline over up to
eight GPUs (`--devices`, Linux only; each stage owns whole layers with their KV and state, and the
head, round state and sampling stay on the first device; design in
`docs/maintainer/pipeline-parallel-plan.md`) is parked while it is ported to the replaced context
engine: its sources stay in the tree, and startup rejects more than one device. Tensor
parallelism is not built.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- v3 `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

## Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed
statement is individually true. For every implementation task:

1. Cover the entire agreed deliverable, its completion status, and all affected or
   evaluated dimensions: behavior, numerical semantics, interfaces, architecture,
   performance, resources, and maintenance. Distinguish completed, incomplete, and
   unverified work; never describe an unmeasured aspect as unchanged.

2. Put favorable and unfavorable findings in the final reply itself, including
   regressions, costs, rejected approaches, failures subsequently fixed, unresolved
   issues, and verification gaps. Explain their disposition. Group repetition
   without hiding distinct problems or exceptions. Small or unexplained adverse
   results must remain visible; attachments cannot substitute for disclosure.

3. Make comparisons representative and comparable. State the baseline, workload,
   conditions, metrics, coverage, outcome distribution, worst changes, and exceptions.
   Distinguish new capability, fallback replacement, and improvement to an optimized
   implementation. Keep claims within the measured scope; neither a best case nor
   an average may stand in for the full results.

4. Apply the same evidence standard to gains and regressions. Label uncertainty;
   do not dismiss slowdowns as noise without evidence. Explain changes to scope,
   baselines, methods, or acceptance criteria and preserve earlier adverse findings.
   Never change these choices to manufacture a favorable conclusion.

5. Reuse sufficient evidence. Additional or repeated checks must satisfy required
   verification, replace invalidated evidence, or resolve a concrete question that
   could change implementation or acceptance. Once the deliverable and acceptance
   conditions are satisfied, stop and report. Report review checks existing work
   and findings; it must not become a new audit, sweep, or reporting-tool project.
   Disclose remaining uncertainty without silently making it a new requirement.
   Disclosure does not excuse unmet completion conditions.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

### Windows build environment (RTX 3090 fork host)

Verified 2026-09 by building all 444 targets and running the full 100-test suite. An earlier
revision of this section was wrong on four points and sent at least one agent down a dead end;
the corrections are called out at the bottom so anyone reading git history is not re-confused.

**Use the existing `build-ninja/` tree. Do not reconfigure and do not delete it.** It is already
configured correctly: Release, `CMAKE_CUDA_ARCHITECTURES=86`, `BUILD_TESTING=ON`,
`NINFER_BUILD_APPS=ON`, FFMPEG resolved through the vcpkg manifest into
`build-ninja/vcpkg_installed/x64-windows`. Configure on this host is slow; there is nothing to
gain by redoing it.

The one real trap is that **two MSVC toolchains are installed and the wrong one is first on
PATH**:

| | version | path |
|---|---|---|
| VS 2026 Community | MSVC v145 `14.50.35717` | `C:\Program Files\Microsoft Visual Studio\18\Community` |
| VS 2022 BuildTools | MSVC v143 `14.44.35207` | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |

CUDA 12.8 hard-errors on v145 (`C1189: unsupported Microsoft Visual Studio version!`), and
`nvcc` picks up whatever `cl.exe` PATH offers, which by default is VS 2026. `build-ninja`'s cache
pins the VS 2022 compiler, so the fix is simply to put that environment in front before building.
Note the `(x86)` in the path — that is the detail the earlier revision missed when it concluded
VS 2022 was absent.

From PowerShell, import the VS 2022 BuildTools environment and build:

```powershell
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
Set-Location C:\ninfer-fork\ninfer-3090\build-ninja
cmake --build .                                  # everything
cmake --build . --target <name>                  # one target
ctest -j2                                        # full suite, ~13 min (159 tests, see below)
ctest -R <regex> --output-on-failure             # one test
```

Full-suite timing, measured 2026-10-06 on the RTX 3090 with the GPU otherwise idle: 159 tests, 768 s
wall at `-j2`, 8 of them skipped without `NINFER_TEST_ARTIFACT` (the `*_real_test` family). The
earlier "~70 s / 100 tests" figure is stale. Three op tests account for most of it and are not
related to engine or scheduler changes: `ninfer_softmax_attention_dflash2_test` (~575 s),
`ninfer_context_kv_materialize_test` (~258 s) and `ninfer_softmax_attention_test` (~184 s); exclude
them with `ctest -E` when iterating on something else. Building only `ninfer_tests` leaves five
standalone test programs unbuilt, so `ctest` reports `public_api`, `jinja`, `artifact_materialization`,
`effective_thinking_budget` and `context_cost_measure` as Not Run, and `chat_templates` and
`artifact_writer_interop` as Failed (they launch those programs). That is a missing build, not a
failure: build `all`, or those targets, first.

That import is reliable; the environment it produces builds cleanly. Adding only the compiler's
`bin` directory to PATH is *not* enough — the standard library headers go missing
(`fatal error C1083: Cannot open include file: 'cstdint'`), because `INCLUDE` and `LIB` come from
vcvars too.

For a standalone `.cu` probe outside the build tree, `nvcc` can be driven directly. Git Bash
mangles MSVC-style flags (`/wd4819` becomes a path), so use PowerShell:

```powershell
nvcc -O3 -arch=sm_86 -allow-unsupported-compiler probe.cu -o probe.exe
```

`-allow-unsupported-compiler` is needed there and only there, because a bare `nvcc` invocation
finds VS 2026 on PATH. It is not needed for the CMake build and is not in the cache.

Other host facts:

- CMake 3.31.3 at `C:\Program Files\CMake`; Ninja 1.11.1 from the Python 3.13 pip install.
- Profilers are installed: Nsight Systems 2024.6.2 (`nsys.exe` under
  `target-windows-x64/`) and Nsight Compute 2025.1.0 (`ncu.bat`). `ncu` needs GPU performance
  counters, which are admin-only by default — run it from an elevated shell rather than changing
  `RmProfilingAdminOnly`, which needs two reboots and loosens a system-wide setting.
- Compilation needs no GPU, so building is always possible. **Executing** CUDA tests, benchmarks
  or perplexity runs needs free VRAM, and the user often has the server loaded — ask before
  assuming the device is free.
- `nvcc` writes `.exp`/`.lib` next to any `-o` target; keep probe builds out of the repo root.
- An untracked `config.bat` in the repo root is a leftover from the earlier, incorrect recipe. It
  sets up the VS 2026 v145 environment with `-allow-unsupported-compiler` and reconfigures into
  `build/`. It is not the supported path; prefer `build-ninja` as above.

**Compiler cache (ccache) — use it, every tree shares one.** A full rebuild here is 10+ minutes
(a few multi-minute CUDA translation units, then single-threaded device and final links), and
editing a public header such as `include/ninfer/types.h` recompiles nearly everything. Several
git worktrees are active at once and each changes only a few files, so builds go through one
shared ccache. `cmake/CompilerCache.cmake` enables it automatically when ccache is found and
prints `Compiler cache: <path> (base <main checkout>)` at configure; if it prints
`ccache not found`, run the setup once and reconfigure:

```powershell
.\scripts\setup-ccache.ps1        # Windows: installs a pinned ccache, writes the shared settings
scripts/setup-ccache.sh           # Linux/WSL: configures an installed ccache (apt install ccache)
```

- **Sharing across worktrees works through `base_dir`**: ccache rewrites absolute paths under
  the main checkout (`C:\ninfer-fork\ninfer-3090`, found from git's common directory) to be relative,
  and every tree has the same `<tree>\build-ninja` beside `<tree>\src`, so the same source hits the
  same entry in any worktree. A worktree outside that directory (e.g. `C:\nb106`) does not share.
  Files that include the shared vcpkg headers (FFmpeg) resolve those paths differently per tree and
  may still miss.
- **WSL has its own cache** (`~/.cache/ccache`; Linux objects cannot be shared with Windows ones).
  WSL builds are `git archive` snapshots under `~` (`~/ninfer-rel-*`) with no `.git`, so CMake passes no
  base there and ccache's own `base_dir` applies, which `scripts/setup-ccache.sh` sets to `$HOME`.
  Install and configure once: `wsl.exe -u root -e apt-get install -y ccache`, then
  `bash scripts/setup-ccache.sh` as the normal WSL user (it installs too when run with sudo or root).
- **Check it:** `ccache -s` after a build. A fresh worktree configured from the same commit should
  hit nearly everything; if it misses, `base_dir` is not set (`ccache --show-config | findstr base_dir`).
- **Do not** clear the cache (`ccache -C`), disable it, or point `CCACHE_DIR` elsewhere to "fix" a
  build problem; configure with `-DNINFER_COMPILER_CACHE=OFF` for a one-off uncached build instead.
- Release is the default and is cached. MSVC `/Zi` (Debug, RelWithDebInfo) cannot be cached.
- **A worktree session needs its own build tree** (`build-ninja` at the main checkout compiles the
  main checkout's source, not yours). With the cache the first build of a new worktree is mostly hits:

```powershell
# from the worktree root, after importing the VS 2022 environment above
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=OFF `
  -DCMAKE_TOOLCHAIN_FILE="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_INSTALLED_DIR=C:/ninfer-fork/ninfer-3090/build-ninja/vcpkg_installed `
  -DVCPKG_MANIFEST_INSTALL=OFF "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -Xcompiler=/Zc:__cplusplus"
```

Tests register as phony targets that all link into `tests\ninfer_tests.exe`; build the
`ninfer_tests` target (not a single test's name) before `ctest`. To turn the cache on in an
existing tree such as the main `build-ninja`, re-run `cmake -S . -B build-ninja` once from the VS 2022
environment; the next build recompiles once, filling the cache.

Corrections to the previous revision of this section, all verified false:

1. "VS 2022 BuildTools is **not** installed, so caches pinning `14.44.35207` are stale —
   reconfigure fresh." It is installed, under `Program Files (x86)`, and those caches are live and
   correct. Reconfiguring is the one thing that actually breaks the build here.
2. "Do not invoke `vcvars64.bat` from a non-interactive shell." The VS 2022 BuildTools vcvars
   imports cleanly and non-interactively, as above. The variable flood described belongs to the
   VS 2026 vcvars, not this one.
3. "FFMPEG is required and was missing." It is provided by the vcpkg manifest and already present
   in `build-ninja/vcpkg_installed/x64-windows`. No `-DFFMPEG_DIR` is needed.
4. "Configure must pass `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler`." Not for this build;
   the cache carries `-Xcompiler=/Zc:__cplusplus` and nothing else, because it uses v143.

## Commits

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.
