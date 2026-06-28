# Quality Rules and Test Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the persistent root `AGENTS.md` rules approved in the quality-rules design spec.

**Architecture:** This implementation creates only the durable repository rule file. Test cleanup,
compatibility-code cleanup, subagent fan-out, and historical document archival remain separate work
after the rule file exists.

**Tech Stack:** Markdown documentation in the repository root and `docs/superpowers`.

---

### Task 1: Add Persistent Repository Rules

**Files:**
- Create: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-06-28-quality-rules-and-test-governance-design.md`

- [ ] **Step 1: Create `AGENTS.md`**

Create `AGENTS.md` at the repository root with exactly the persistent rules from the approved spec:

```markdown
# AGENTS.md

## Scope

These rules apply to the entire repository.

This project is a from-scratch C++/CUDA inference engine specialized for Qwen3.6-27B on one RTX
5090. It is not a general-purpose runtime, compatibility layer, or model zoo.

If any instruction outside this file conflicts with this file, follow this file.

## No Backward Compatibility

This project does not preserve backward compatibility.

When changing project-owned code, tests, scripts, CLIs, fixtures, schemas, or active documentation:

- Replace the old behavior directly with the new behavior.
- Delete deprecated aliases, compatibility shims, legacy flags, legacy fields, fallback branches,
  and transition code.
- Do not keep old and new behavior side by side.
- Do not add deprecation warnings for project-owned behavior; remove the behavior.
- Do not preserve tests whose only purpose is to prove old behavior still works.

Do not design for migration windows, external legacy users, old artifact formats, or old command
surfaces unless the user explicitly redefines the project scope.

## Implementation Style

Prefer direct, explicit implementation over framework-like abstraction.

Use the project layering deliberately:

- L0 owns reusable infrastructure.
- L1 owns operator APIs and CUDA implementations.
- L2 owns the Qwen3.6 model card and static schedule.
- Tools and benchmarks own their practical CLI/report behavior.

Do not add abstractions for hypothetical future models, generic runtime flexibility, legacy
surfaces, or speculative reuse.

## Testing Policy

Tests are not added by default.

A test is allowed only when it protects a real, observable project risk and can fail for a
meaningful regression. Tests that only increase coverage numbers, lock implementation shape, or
preserve compatibility are not allowed.

## Restricted TDD

TDD is allowed only for the categories below. For other work, use build checks, existing tests,
sanitizer runs, benchmarks, review checklists, or manual verification as appropriate.

### Hard Whitelist

1. Numerical correctness for CUDA kernels, operators, or model block parity.
   - Must have a clear mathematical oracle.
   - Must use real project shapes where applicable.
   - Must include stress or edge cases that can expose wrong math.

2. Binary and file-format contracts.
   - Examples: q5090 parser, pack/unpack, manifest, CRC, shape, dtype, layout, and weight-loading
     boundaries.
   - The test must reject malformed or dangerous input that could otherwise load wrong weights or
     corrupt runtime behavior.

3. Real CLI and report/schema contracts.
   - Examples: benchmark reports, summaries, tokenizer tools, fixture manifests, JSON fields
     consumed by scripts or docs.
   - The test must validate user-visible or downstream-consumed behavior.

### Conditional Whitelist

4. End-to-end observable behavior.
   - Allowed only with a small number of canonical fixtures.
   - Must validate final user-visible output or downstream-consumed artifacts.
   - Must not test internal call paths, private functions, or source layout.

5. GPU memory and lifetime risks.
   - Allowed only when the test actually runs the risky behavior or is paired with
     `compute-sanitizer`.
   - Examples: out-of-bounds access, use-after-rewind, arena lifetime, KV/state/workspace shape
     mismatch, repeated prefill/decode lifetime bugs.
   - Source-code string scanning does not qualify.

6. Reproduced bug regression.
   - The bug must have a known failing behavior before the fix.
   - The test must document the trigger condition, expected behavior, and why existing tests missed
     it.
   - Prefer the smallest layer that exposes the bug.
   - Delete the regression test later if the underlying risk disappears.

## Forbidden Tests

Do not add tests that:

- scan source files for strings, function names, call order, or private layout;
- preserve deprecated aliases, compatibility shims, legacy flags, or legacy fields;
- only test getters, setters, default constructors, enum spellings, or trivial mappings;
- only check that a document or file exists;
- duplicate another test without increasing risk coverage;
- lock implementation details that should remain free to change;
- exist only to satisfy a TDD process or coverage metric.

## Rewriting Bad Tests

If an existing bad test protects a real risk, replace it with a behavior, numerical,
parser/schema, integration, or sanitizer-backed test before removing the bad test.

If a test does not protect a real risk, remove it without replacement.

Source-structure tests are not acceptable. If the only enforceable claim is a preferred
implementation shape, enforce it through review, not through tests.

## Verification

Before claiming a change is complete, run the smallest verification set that proves the changed
behavior:

- build affected targets;
- run affected tests;
- run `compute-sanitizer` for GPU memory/lifetime changes when available;
- run benchmark/report tooling checks for schema or CLI changes.

Do not add low-value tests to compensate for weak verification.
```

- [ ] **Step 2: Mark the design spec implemented**

In `docs/superpowers/specs/2026-06-28-quality-rules-and-test-governance-design.md`, change the
status line to:

```markdown
> Status: implemented. Date: 2026-06-28.
```

- [ ] **Step 3: Verify the durable rule file exists**

Run:

```bash
test -f AGENTS.md
```

Expected: exits 0.

- [ ] **Step 4: Verify no cleanup-stage language entered `AGENTS.md`**

Run:

```bash
rg -n "subagent|archive|historical|docs/plans|cleanup campaign|test-audit|phase|temporary" AGENTS.md
```

Expected: no matches and exit code 1.

- [ ] **Step 5: Verify the required rule anchors are present**

Run:

```bash
rg -n "No Backward Compatibility|Restricted TDD|Hard Whitelist|Conditional Whitelist|Forbidden Tests|Rewriting Bad Tests|Verification" AGENTS.md
```

Expected: one match for each listed heading.

- [ ] **Step 6: Review the diff**

Run:

```bash
git diff -- AGENTS.md docs/superpowers/specs/2026-06-28-quality-rules-and-test-governance-design.md
```

Expected: `AGENTS.md` contains only persistent rules; the spec status changes from
`pending user review` to `implemented`.

- [ ] **Step 7: Commit**

Run:

```bash
git add AGENTS.md \
        docs/superpowers/specs/2026-06-28-quality-rules-and-test-governance-design.md \
        docs/superpowers/plans/2026-06-28-quality-rules-and-test-governance.md
git commit -m "docs: add repository quality rules"
```

Expected: commit succeeds.
