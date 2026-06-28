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
