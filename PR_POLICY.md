# Pull Request Policy

Thanks for contributing to `ninfer-3090`! I much appreciate anyone willing to put time and effort into empowering people on local hardware.

This project is performance-sensitive C++/CUDA code running close to hardware. Small changes can have unexpected effects on correctness, determinism, VRAM usage, latency, throughput, or compatibility. Because of that, we prefer PRs that are easy to understand, test, benchmark, and revert.

## Keep PRs focused

One PR should ideally do one thing.

Please avoid combining new features, unrelated bug fixes, refactors, formatting changes, dependency changes, and architecture changes into the same PR unless they are genuinely inseparable.

If a feature needs several distinct pieces, prefer multiple PRs in dependency order. A PR should be small enough that we can reasonably answer: what changed, why did it change, and what caused a regression if one appears?

## Large changes should be split

Large architectural changes are welcome, but they should usually arrive as a series of independently testable PRs:

1. Core primitive or infrastructure
2. Feature using that primitive
3. Optional optimization
4. Cleanup after the feature has proven stable

Being able to merge, benchmark, revert, and bisect each part independently is important.

## Draft PRs are encouraged

Please open work-in-progress changes as **Draft PRs** for early technical discussion, benchmarking, design review, community testing, and conflict discovery. Keep the PR in Draft while the implementation is still moving substantially. Mark it **Ready for review** when you believe it is actually mergeable.

A Draft PR may still receive code review and testing from the community.

## Include evidence

For performance-related changes, describe the GPU, model/configuration, runtime flags, concurrency, context length, before/after numbers, and whether the results measure throughput, latency, TTFT, decode speed, VRAM usage, or another metric.

Avoid isolated performance claims without enough information to reproduce them. For correctness changes, include a regression test or focused reproducer whenever possible.

## Performance changes must preserve correctness

A speedup is only useful if the output remains correct. Where relevant, test output parity, deterministic behavior under the same execution conditions, concurrency, long-context behavior, memory pressure, failure/recovery paths, repeated requests, and supported GPU configurations.

If exact bitwise output is not expected across different batch compositions or kernel schedules, document the expected determinism contract clearly.

## Risky features should be opt-in first

Experimental or invasive runtime changes should generally be introduced behind a flag and default to **off**. This is especially important for memory residency, KV cache behavior, speculative decoding, CUDA graphs, custom kernels, quantization, host/device memory movement, scheduling, concurrency, and model-architecture-specific behavior.

Once a feature has proven stable, we can reconsider the default.

## No unrelated cleanup

Please resist the temptation to clean up nearby code while implementing a feature. Unrelated cleanup makes review and regression analysis much harder. If you find something else worth fixing, opening another small PR is usually better.

## Bug fixes

A good bug-fix PR contains a clear description of the failure, a reproducer when practical, the root cause, the smallest reasonable fix, and a regression test if possible.

If the fix already exists upstream, link the upstream issue, PR, or commit and say whether the change is a clean cherry-pick or adapted for this fork.

## Upstream work

This repository intentionally carries hardware-specific work that may diverge from upstream. If a change is simultaneously being developed or reviewed upstream, it can make sense to let the upstream implementation settle first and then port the reviewed version here.

Please link relevant upstream PRs and avoid maintaining multiple rapidly diverging copies of the same unfinished implementation unless there is a strong reason.

## Review comments

Review is collaborative. Reviewers are encouraged to question correctness assumptions, locking and concurrency, blocking behavior, unexplained constants, ownership/lifetime, synchronization, error handling, benchmark methodology, performance regressions, and hidden coupling between features.

Authors should not feel that review comments imply rejection. Finding problems before merge is the point of review.

## Maintainer bandwidth

This is a community open-source project. Opening an issue or PR does not guarantee that the maintainer will personally implement, debug, benchmark, or merge it.

Well-scoped contributions with good tests and evidence are much easier to review. Community review, testing, fixes, and follow-up contributions are strongly encouraged.

## Before requesting review

Please make sure that:

- the PR has one clear purpose;
- unrelated changes have been removed;
- the branch is reasonably up to date;
- relevant tests pass;
- new behavior is tested;
- performance claims include measurements;
- experimental behavior is opt-in where appropriate;
- limitations and known issues are documented; and
- the PR description explains both the design and the user-visible effect.
