# NInfer-3090 v0.14.1

**A server stability fix: one oversized or unplannable request no longer kills the other requests
that are running.** If you run `ninfer-serve` for agents that send large prompts, with
`--auto-prefix-grid` or several clients sharing prompts, pick this up. There are no new features
and no flag changes. 3 merged pull requests (#171, #172, #173), 7 non-merge commits.

## What this means for you

- **Before:** admitting a large request could make the server fail every running request with
  `internal error generation`, clear the context cache, and recover. Repeated, it could latch the
  engine. After the fix, running requests keep streaming and the cache is kept.
- **After:** the request that cannot be planned fails alone, the server logs it, and everything else
  carries on.
- If you never saw `internal error generation`, nothing changes for you.

## What was fixed

- **Shared-prefix planning no longer throws on many candidates (#172).** The planner searched subsets
  of shared prefix candidates and threw if more than 7 survived its filters. That count depends on
  the traffic: `--auto-prefix-grid` alone proposes 8, plus structural and client boundaries. The throw
  happened inside admission, so the engine failed all running requests. It now keeps the 7 strongest
  candidates (pressure-capable first, then most distinct callers asking for it, then rebuild cost).
- **Candidate ranking counts callers, not repeats (#172, follow-up).** The first version ranked by how
  many times a prefix was asked for, so one session repeating itself could outrank a prefix two
  independent callers wanted, and narrowing would drop the latter. It now ranks by distinct reuse
  domains.
- **A planning failure fails only that request (#173).** Any exception while planning a waiting
  request was handled as a worker failure. It now fails the one request, does not count towards the
  latch, and logs `engine admission failure ... contained` with the exception text.
- **Worker failures are logged, and a latched engine exits (#171).** Worker failures now appear in the
  operational log, including the queued request ids a latch fails and a sanitised latch reason. A
  latched server exits instead of staying up refusing everything, so a supervisor can restart it, and a
  single repeating client can no longer latch the engine on its own. Review fixes in the same PR make
  the recovery path non-throwing and close a race between failing queued requests and new enqueues.
- **Test tool:** `tools/latch_repro/loadgen.py` is a load generator that reproduces the large-request
  failure against a running server (port from `NI_PORT`, default 8011).

## Verification

- **Tests added with the fixes:** a regression test for more shared boundaries than the subset search
  handles, ranking tests, worker-recovery and operational-log renderer tests, and a real-model test
  `admission-planning-failure`. The PR record says the real-model test fails without the containment
  (running request failed, one recovery) and passes with it. I did not re-run the real-model test
  for this release.
- **Release build and packaging:** built from master at `0c00dfc1` on Windows (MSVC 2022, CUDA 12.8)
  and Linux (WSL Ubuntu, CUDA 12.8), both `sm_86`. Both archives' file checksums verify; the three
  Windows executables start with only Windows on `PATH`; on Linux a fresh unpack runs
  `./ninfer-serve --help` and `./run.sh --help`.
- **Not verified:** I did not start the launcher or serve a request on the RTX 3090 for this release,
  and did not re-run the test suite or the load generator against these binaries. The fixes rely on the
  tests recorded in their PRs. Behaviour on other GPUs is untested.

## Downloads

Same two archives as before: the Windows zip (cuBLAS DLLs included) and the Linux tar.gz (needs glibc
2.38+, CUDA 12.8 runtime with cuBLAS, FFmpeg 6; see `docs/release-archive-linux.md`). Verify with the
`SHA256SUMS` files.
