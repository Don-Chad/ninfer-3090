# M3 Readiness

> Status: pending regeneration of the Qwen3.6 chat-template M2.8 baseline.
> Scope: placeholder for the official M3 entry evidence package for qwen3.6-ultraspeed.

The previous pre-chat-template baseline evidence is no longer valid for M3 readiness. The controller must
rerun the official baseline with `.messages.json` prompt fixtures rendered through the Qwen3.6 chat
template, fixed prompt ids, and stop tokens `[248046, 248044]`.

Final readiness will be filled after the real q5090 runs produce valid local raw reports and committed
chat-template output/prefill gate summaries under `docs/bench/baselines/`.

## Latest Local Smoke Observation

Commit `9e878885ee547e7b98ffe680e5f395a38f8e2b9b` replaced the fixture prompts with common user
questions and added the decoded-text nonempty gate. A local `cn_short` smoke run on that clean commit
still generated only whitespace tokens, so `tools/bench/make_baseline_summary.py --baseline-class smoke`
correctly rejected the decoded manifest with:

```text
decoded manifest must contain nonempty clean output for cn_short
```

No new smoke summary is committed yet. The remaining blocker is therefore not only prompt wording; the
current real-weight inference path must first produce nonempty assistant text for the short common
questions.

## Pending Evidence

- Local raw e2e reports under `profiles/e2e/` for `m3_output_gate` and `m3_prefill_gate`.
- Committed baseline summaries with `prompt_format`, `stop_token_ids`, tokenizer chat-template hash, and
  generation-config hash.
- Required output-gate cases `cn_short`, `en_short`, `code_short`, and `math_short`; required prefill-gate
  case `long_2k`.
- Memory accounting, hidden allocation status, workspace lifetime policy, q5090 identity, and first-wave
  M3 target ordering from the regenerated run.
