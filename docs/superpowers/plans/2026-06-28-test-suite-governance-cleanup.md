# Test Suite Governance Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the existing test suite and active compatibility surfaces into alignment with
`AGENTS.md`.

**Architecture:** Cleanup proceeds in safe layers. Delete tests that protect no real risk, remove
project-owned compatibility entry points, and keep invalid-but-risk-covering structure tests until
valid behavior or sanitizer-backed replacements exist.

**Tech Stack:** C++20/CUDA CTest targets, Python pytest tools, CMake test registration.

---

## Audit Classification

### KEEP

- `tests/test_arena.cpp`
- `tests/test_device.cpp`
- `tests/test_engine_memory_stats.cpp`
- `tests/test_kv_cache.cpp`
- `tests/test_model_bind.cpp`
- `tests/test_model_blocks.cpp`
- `tests/test_model_config.cpp`
- `tests/test_q5090_pack_golden.cpp`
- `tests/test_q5090_parser.cpp`
- `tests/test_state_store.cpp`
- `tests/test_tensor.cpp`
- `tests/test_weight_store.cpp`
- `tests/fixtures/make_q5090_fixture.py`
- `tests/kernels/op_check.h`
- `tests/kernels/op_tester.h`
- `tests/kernels/gdn_ref.h`
- `tests/kernels/q5090_pack.h`
- `tests/kernels/test_argmax.cpp`
- `tests/kernels/test_causal_conv1d.cpp`
- `tests/kernels/test_gated_delta_rule.cpp`
- `tests/kernels/test_gdn_gating.cpp`
- `tests/kernels/test_gqa_attention.cpp`
- `tests/kernels/test_l2norm.cpp`
- `tests/kernels/test_linear.cpp`
- `tests/kernels/test_residual_add.cpp`
- `tests/kernels/test_rmsnorm.cpp`
- `tests/kernels/test_rope.cpp`
- `tests/kernels/test_sigmoid_gate_mul.cpp`
- `tests/kernels/test_silu_and_mul.cpp`
- `tests/test_bench_report_tools.py`
- `tests/test_bench_tokenizer_tools.py`
- `tools/q5090_convert/tests/test_packing.py`
- `tools/q5090_convert/tests/test_tensor_plan.py`

### DELETE Or MERGE Without Replacement

- `tests/test_dtype.cpp`
- `tests/test_engine_load.cpp`
- `tests/test_hardening_cleanup_structure.cpp`
- `tests/kernels/test_gdn_common.cpp`
- `tests/kernels/test_linear_dense_structure.cpp`

### REWRITE Before Deletion

- `tests/test_graph_readiness_structure.cpp`
- `tests/test_runtime_tap_structure.cpp`
- `tests/test_weight_store_real.cpp`
- `tests/kernels/test_embed_gather.cpp`
- `tests/test_e2e_bench_support.cpp`
- `tests/test_decode_e2e_report_redaction.py`

### KEEP With Local Pruning

- `tests/test_weight.cpp`
- `tests/test_bench_report_tools.py`
- `tests/test_bench_tokenizer_tools.py`

## Task 1: First Safe Cleanup Batch

**Files:**
- Delete: `tests/test_dtype.cpp`
- Delete: `tests/test_engine_load.cpp`
- Delete: `tests/test_hardening_cleanup_structure.cpp`
- Delete: `tests/kernels/test_gdn_common.cpp`
- Delete: `tests/kernels/test_linear_dense_structure.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `bench/e2e_bench_support.cpp`
- Modify: `tests/test_e2e_bench_support.cpp`
- Modify: `src/model/qwen3_6_27b.cpp`
- Modify: `tools/parity/greedy_match.py`
- Modify: `tests/test_runtime_tap_structure.cpp`

- [x] Remove CMake registrations for deleted tests.
- [x] Delete the five invalid standalone tests listed above.
- [x] Remove the `--eos-token-id` compatibility alias from e2e bench argument parsing and usage.
- [x] Keep only `--stop-token-id` behavior in `tests/test_e2e_bench_support.cpp`.
- [x] Remove FileTap's old `layer_%02d.f32` output.
- [x] Update `tools/parity/greedy_match.py` to consume `layer_%02d_mlp.f32`.
- [x] Remove compatibility-only assertions from `tests/test_runtime_tap_structure.cpp`.

Verification:

```bash
cmake --build build -j --target qus_e2e_bench_support_test qus_runtime_tap_structure_test qus_engine_memory_stats_test qus_linear_test
ctest --test-dir build --output-on-failure -R '^(qus_e2e_bench_support_test|qus_runtime_tap_structure_test|qus_engine_memory_stats_test|qus_linear_test)$'
pytest -q tools/q5090_convert/tests tests/test_bench_report_tools.py tests/test_bench_tokenizer_tools.py tests/test_decode_e2e_report_redaction.py
rg -n 'eos-token-id|deprecated|legacy|compat' include src bench tools tests --glob '!third_party/**'
```

Expected:

- Build succeeds.
- Listed CTest targets pass or GPU-dependent targets skip cleanly where no CUDA device is available.
- Pytest passes.
- Compatibility scan has no project-owned compatibility entry points. A test that verifies a removed
  flag is rejected may still mention the old flag text.

Actual verification:

```bash
cmake -S . -B build
cmake --build build -j --target qus_e2e_bench_support_test qus_runtime_tap_structure_test qus_engine_memory_stats_test qus_linear_test
ctest --test-dir build --output-on-failure -R '^(qus_e2e_bench_support_test|qus_runtime_tap_structure_test|qus_engine_memory_stats_test|qus_linear_test)$'
ctest --test-dir build --output-on-failure
pytest -q tests tools/q5090_convert/tests
```

Result: all commands exited 0; full CTest reported 30/30 passing; pytest reported 69 passed and
39 subtests passed.

## Remaining Tasks

### Task 2: Replace Graph-Readiness Source Scan

Replace `tests/test_graph_readiness_structure.cpp` with behavior or sanitizer-backed coverage for
device position behavior and capture-hostile allocation risk. Reuse existing `test_model_bind.cpp`
coverage for canonical conv1d binding instead of duplicating it.

### Task 3: Replace Runtime Tap Source Scan

Replace `tests/test_runtime_tap_structure.cpp` with behavior coverage for EOS stopping, invalid EOS
rejection, FileTap output files, MemoryTap compile/use path, layer dump invocation, and workspace
lifetime around tap data. Delete source-string checks after replacements exist.

### Task 4: Rewrite E2E Bench Support Tests

Keep `.ids` parsing, stop-token list behavior, fixture manifest loading, max-context rejection, and
raw/error JSON report behavior. Parse emitted JSON structurally instead of relying on brittle
substring checks.

### Task 5: Rewrite Real q5090 Artifact Test

Keep the large-artifact integration intent in `tests/test_weight_store_real.cpp`, but replace
manifest string search with structured JSON validation. Keep the test explicit about optional
large-artifact availability.

### Task 6: Extend Embed Gather Shape Coverage

Extend `tests/kernels/test_embed_gather.cpp` with real hidden-size coverage for `[5120,T]` output
while preserving the dense/Q6 oracle behavior.

### Task 7: Prune Python Tool Test Duplicates

Move the unique CLI default-redaction assertion from
`tests/test_decode_e2e_report_redaction.py` into the tokenizer/decode tool tests, then delete the
duplicate file. Trim private-helper and prose-locking assertions from bench report/tokenizer tests
without removing schema or artifact coverage.

### Task 8: L0 Test Pruning

Fold meaningful dtype-size and dense-weight risks into behavior tests. Delete or trim standalone
enum/default/descriptor-field smoke checks in `tests/test_weight.cpp` and any remaining trivial L0
assertions.
