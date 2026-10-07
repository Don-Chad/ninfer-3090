---
name: release
description: Cut a NInfer-3090 GitHub release end to end. Asks for the version (suggests one), builds the Windows and Linux (WSL) binaries from the same commit, packages them, writes customer-facing release notes, and publishes. Use when the user says "cut a release", "build the release", "ship vX.Y.Z" or similar.
---

# Cut a NInfer-3090 release

The mechanics and every known trap are in `docs/maintainer/release-process.md`; read it first, and
`AGENTS.md` "Windows build environment" for the VS 2022 import. This skill is the order of operations
and the judgement calls. If the doc and this skill disagree, the doc wins; fix the skill.

Outward-facing steps (pushing, merging, tagging, creating the release) are listed below with a
confirmation point. Everything before that is local.

## 1. Pick the version

1. `git fetch origin --tags`, then find the latest tag (`git tag --sort=-creatordate | head`) and read
   `git log <latest>..origin/master --oneline` plus `gh pr list -R ashalliants/ninfer-3090 --state merged
   --search "merged:>=<last release date>"` to see what is shipping.
2. Suggest a version and say why in one line. Tags are `vX.Y.Z-rtx3090`; `VERSION` holds the full tag
   without the `v` (`0.14.3-rtx3090`).
   - Patch bump: only fixes, docs, internal changes.
   - Minor bump: a new flag, endpoint, model, profile or a visible speed/memory change.
   - Never suggest a major bump unless the user raises it.
3. Ask with `AskUserQuestion`: the suggested version first (marked Recommended), the other bump as the
   second option; "Other" covers a custom number. Do not proceed without an answer.
4. Stop if `master` has uncommitted tracked changes or the working tree is not at `origin/master`; ask.
   Untracked scratch files (e.g. `graft-pr.md`) are not a blocker and must not be committed.

## 2. Write the release notes first

`package-release.{ps1,sh}` refuses to run without `RELEASE_NOTES_X.Y.Z.md`, so write it before building.
Work from the merged PRs and the diff, not commit subjects. Read the PR bodies for measured numbers.
Use `RELEASE_NOTES_0.14.2.md` (or the newest one) as the shape template.

### Who the notes are for

The reader runs NInfer, or buys access to it, to drive coding agents. They are not an inference
engineer. Their profiles are in `C:\ninfer-fork\ninfer-gateway\launch_plan.md` section 4; in short:

- **The capped power user**: hit a frontier-plan limit, wants a second lane that never stops. Cares
  about: does it stay up, does it queue my subagents, how fast is the first word.
- **The always-on agent operator**: overnight loops, agent fleets, 24/7 streams. Cares about: crashes,
  recovery, concurrency, memory ceilings on shared or rented GPUs, no surprises on upgrade.
- **The GPU-less Qwen fan**: cares about quality and whether setup is two minutes.

Not the audience: people who want kernel details. Those go in a short "Under the hood" tail at most.

### How to write them

- **Open with one bolded sentence on what changes for the reader**, then "What this means for you":
  a bullet per situation ("You run on a rented GPU..."), ending with the "you do nothing and nothing
  changes" bullet when true. Say explicitly whether any default moved.
- Lead with outcomes in the reader's units (seconds to first word, crashes avoided, GB freed), then the
  flag or setting that gets it. Plain words: "reading the prompt (prefill)", not "prefill".
- Every number carries its baseline: GPU, model, before and after on the same machine. A table is
  good for before/after. Do not write a speedup you did not find in a PR body or measure.
- Include costs and caveats honestly, in the section they belong to ("The long prompt itself gets 4-9%
  slower"). `AGENTS.md` requires adverse results and unverified items to appear in the reply and notes;
  never drop one because it reads badly.
- Group by what the reader does, not by PR. List PR numbers once at the top or at the end.
- Breaking changes and upgrade steps go in a section the reader cannot miss, near the top.
- Finish with "Also fixed" (one line each, symptom first, not cause) and install/upgrade pointers.
- Title for the GitHub release: `NInfer-3090 vX.Y.Z - <the headline outcome in plain words>`.
- No marketing superlatives, no emoji, no internal jargon (ReplaySSM, cohort, lane) without a gloss.

Show the user the draft and incorporate edits before building, because it is cheap to change now and
the packager embeds nothing from it. Numbers can be refined after smoke tests.

Order matters for speed: sources are identical whether or not the notes are committed, so start the
builds (step 4) as soon as the version is chosen and the branch exists, and let the user review the draft
while they run. The notes only gate packaging.

Tooling notes: the harness blocks foreground `sleep`; wait with Monitor until-loops or background-task
notifications. A PowerShell command containing a regex next to `Remove-Item` can be refused as a
"protected path"; put the logic in a `.ps1` in the scratchpad instead. Old monitors keep firing after a
retry, so match each event to the task id you are waiting on.

## 3. Prepare the release branch

Branch `release/vX.Y.Z` from `origin/master` (use a worktree if the main checkout is busy, with its own
build tree per `AGENTS.md`). Set `VERSION`, add the notes, refresh the README highlights if a headline
number or capability changed. Commit with a conventional subject (`docs: release vX.Y.Z`) and the
attribution lines from the session. Do not push yet.

## 4. Build both platforms from the same commit

The binaries do not embed `VERSION`; build from the release branch commit. The GPU is usually
loaded by the user's server, but compiling needs none.

**Windows** (PowerShell): import the VS 2022 BuildTools environment (AGENTS.md), use the existing
`build-ninja` tree without reconfiguring, ensure `NINFER_BUILD_BENCHMARKS` is ON, then
`cmake --build . --target ninfer ninfer-serve ninfer_bench`. Run as a background task; ccache makes
most of it hits. Check that the tree compiles the release commit's source (worktree trap in memory
`worktree-build-tree-gotcha`).

**Linux** (WSL), in parallel with the Windows build: export a fresh `git archive` of the release
commit to `~/ninfer-rel-X.Y.Z`, configure with `-DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=OFF
-DNINFER_BUILD_BENCHMARKS=ON`, build the same three targets. Run it with the Bash tool's
`run_in_background` on the `wsl.exe -e bash -lc "..."` call, never `& disown`. If ninja dies with
`posix_spawn: Resource temporarily unavailable`, use the `systemd-run --scope` recipe in the doc. Verify
`ccache -s` hits rather than clearing the cache.

Write the WSL steps to a script in the scratchpad and run it as `wsl.exe -u root -e systemd-run --scope
--quiet --uid=<you> --gid=<you> -p TasksMax=infinity -- bash -l /mnt/c/.../script.sh`. From the Bash tool
set `MSYS_NO_PATHCONV=1`, or Git Bash rewrites `/mnt/c/...` into `C:/Program Files/Git/mnt/c/...` and the
launch fails instantly with exit 127 (it still looks "completed"; read the output). The build takes about
11 minutes and Windows about the same on ccache hits.

Do not wait idle on one while the other builds; check both logs when notified.

## 5. Package

- Windows: `scripts\package-release.ps1` (or `build.ps1 -Package`). It copies the cuBLAS DLLs and
  refuses an archive whose executables fail with only Windows on `PATH`; never weaken that.
- Linux: the `git archive` export has no `VERSION` bump or notes yet if it was taken before the release
  commit, and the packager refuses without them. Copy `VERSION` and `RELEASE_NOTES_X.Y.Z.md` from
  `/mnt/c/...` into the export first. Then run `scripts/package-release.sh` inside WSL against the Linux build tree, then copy
  `dist/*linux*` back to `C:\ninfer-fork\ninfer-3090\dist` (or the release worktree's `dist`).
- Result: two archives and `SHA256SUMS-vX.Y.Z-{windows,linux}.txt`. Asset must be under 2 GiB each
  (GitHub's limit); report the sizes.

## 6. Verify before publishing

Run the "Checks before publishing" list in the doc: `scripts/check-linux-scripts.sh`, checksums
(inner and outer) for both archives, the Windows exes starting with a stripped `PATH` from an
extracted copy, the Linux archive's `./ninfer-serve --help` and `./run.sh --help` from a fresh unpack,
and `ldd` for new dynamic dependencies (update `docs/release-archive-linux.md` if they changed).

The real-card smoke test (default launcher profile serves a chat completion) needs free VRAM: ask the
user whether the 3090 is free before running it. If they say no, do not skip silently: record it as
unverified in the final report.

Smoke-test recipe: extract the zip into the scratchpad, make `models\` a directory junction to
`C:\ninfer-fork\ninfer-3090\models`, set `PATH` to `C:\Windows\System32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0`
(`run.bat` pipes through `powershell`; a bare System32/Windows `PATH` makes it fail), and call
`cmd /c "<full path>\run.bat qwen38-27b"` with the full path. Wait for `/v1/models` to list the model,
send one chat completion, then stop the server by matching its path, never by name alone. A step-down
from 188,416 context is expected while the desktop holds 2 GiB.

The archives embed `RELEASE_NOTES_X.Y.Z.md`, so finish the notes (including the Verification section)
before the final package, or repackage both afterwards.

If any check fails, fix and re-package; do not publish and note the failure and its fix in the report.

## 7. Publish (confirm first)

Show the user: version, commit, asset names and sizes, the final notes, and any unverified checks.
Wait for explicit approval, then:

1. Push the branch, open a PR to `master` (`gh pr create -R ashalliants/ninfer-3090`), merge it once
   checks pass. **Always pass `-R ashalliants/ninfer-3090`**; without it `gh` targets the upstream fork.
2. Update local `master`, create an annotated tag `vX.Y.Z-rtx3090` on the merge commit, push it.
3. `gh release create vX.Y.Z-rtx3090 <zip> <tar.gz> <both SHA256SUMS> -R ashalliants/ninfer-3090
   --notes-file RELEASE_NOTES_X.Y.Z.md --title "NInfer-3090 vX.Y.Z - ..." --latest`
4. Confirm with `gh release view ... --json name,isDraft,assets,tagName -R ashalliants/ninfer-3090` (there
   is no `isLatest` field) and `gh release list -L 2 -R ashalliants/ninfer-3090`, which shows "Latest":
   four assets, state `uploaded`, sizes equal to the local files.

## 8. Report

Give the release URL, asset sizes, what was verified and what was not, and any adverse findings
from the notes (regressions, costs). Update `docs/maintainer/release-process.md` with any new trap
hit, and clean `dist/` and the `~/ninfer-rel-*` export only after the release is confirmed.
