# M2 — Model Card + Engine Integration & Validation — Plan (Codex-self-contained)

> Self-contained for Codex. Reuses the execution machinery in
> [`docs/plans/l1-tier1-simple-ops.md`](l1-tier1-simple-ops.md) (subagent workflow + prompt templates
> + ncu procedure) and the frozen test framework. The design is
> [`docs/l2-model-card-design.md`](../l2-model-card-design.md) — **read it fully** (the §4 schedule is
> aligned to the real L1 op signatures). All 13 L1 ops are implemented and tested.

**Goal:** Build the L2 model card (`config.h` + `model.h` + `qwen3_6_27b.cpp`) and the `Engine`, wiring
every L1 op into the prefill/decode schedule, then **validate the whole forward graph**: per-layer
numerical parity + end-to-end greedy token-match vs the reference. This is the M2 correctness baseline
(`design.md` §12) — the first time the full pipeline runs and is proven correct. **Performance is M3+,
not gated here.**

**Architecture:** The card is a thin `Qwen3_6_27B` object (bindings + schedule); the `Engine` owns the
resources + outer loop (design §2). Straight-line C++ over `constexpr` dims; the §4 sketch is the
spec. token-ids in → token-ids out, greedy.

**Tech Stack:** C++20, CUDA 13.1 (sm_120), CMake ≥ 3.28, gcc 13.3, build dir `build/`. Real weights:
the q5090 file under `out/` (see `out/manifest.json`); reference model in `~/vllm` (or `~/llama.cpp`).

---

## Hard rules
- Build the card per `l2-model-card-design.md` §1–§11 (don't re-derive decisions). No virtual dispatch,
  no per-step `cudaMalloc`, device-side `pos`/`token`, work_ bump-reset each step (design §6).
- L1 ops + the frozen test framework are read-only.
- **FORMAT before every commit:** `clang-format -i <new/changed .h/.cuh/.cpp/.cu>` then
  `clang-format --dry-run --Werror <…>` (exit 0; repo `.clang-format`).
- Work on `master`, one commit per task.

## Components to build
- `include/qus/model/config.h` — `ModelConfig` (constexpr, design §3.1) + `is_full`/`full_idx`/`gdn_idx`
  + `kCfg` + derived consts (`kAttnScale=1/√256`, `kGdnScale=1/√128`, `rms_eps`, rope dims).
- `include/qus/model/model.h` — `Qwen3_6_27B`: `FullLayerW`/`GdnLayerW`/`MlpW` (over `const Weight*` /
  `const Tensor*`, design §3.3), `StepState` (design §3.5), ctor (binds) + `prefill`/`decode_step`,
  holds references to Engine resources, optional `Tap` template (design §8).
- `src/model/qwen3_6_27b.cpp` — `bind()` (`WeightStore` → per-layer structs via
  `(TextCore, SourceKind, layer)`), `attn_mix`/`gdn_mix`/`mlp_tail`/`run_layers`, `prefill`/`decode_step`
  (exactly design §4).
- `include/qus/runtime/engine.h` + `src/runtime/engine.cpp` — `Engine`: owns `DeviceContext`, 3
  `DeviceArena`s, `WeightStore`, `KVCache`, `GdnState`, `StepState`, the card; `load(path)` (load
  weights, size caches, construct card) + `generate(prompt, max_new) -> vector<int>` (prefill + decode
  loop + next-token D2H readback + EOS).
- `StepState`: `token[1]` I32, `pos[1]` I32, `logits[vocab]` (dtype follows `lm_head` out; bf16 v1) —
  persistent device buffers (design §3.5).

**Cache construction (confirm dims):** `KVCache(cache_arena, /*full_layers=*/16, max_context,
/*num_kv_heads=*/4, /*head_dim=*/256, bf16)`; `GdnState(cache_arena, /*gdn_layers=*/48,
/*conv_dim=*/10240, /*conv_width=*/4, /*value_heads=*/48, /*value_head_dim=*/128, /*key_head_dim=*/128)`
with **`ssm[gidx]` laid out `[dk,dv,Hv]` AR-transposed** (design §4 notes). Verify `GdnState`'s ctor
produces that layout; if not, fix the ctor (small L0 change) — the grouped GDN kernel depends on it.

---

## Validation (the M2 gate)
1. **Wiring smoke (no real model):** unit-test `bind()` + each block helper on a fixture. Build a small
   q5090 fixture (`tests/fixtures/make_q5090_fixture.py`, `--limit-text-layers` for a few layers) with
   the real per-tensor dims; assert `bind()` resolves every pointer, and that `attn_mix`/`gdn_mix`/
   `mlp_tail` run on one layer with correct output shapes and `compute-sanitizer` clean.
2. **Per-layer parity (real weights):** with the `FileTap` build, dump our per-layer hidden states for
   a **fixed prompt** and compare to reference dumps from vLLM/HF (`~/vllm`) for the same prompt:
   **cosine ≥ 0.999 and max-abs within bf16 tolerance per layer**. First divergent layer localizes any
   bug (e.g. head-map, schedule order, a transform).
3. **Greedy token-match (real weights):** run `Engine::generate` greedy on the fixed prompt; it MUST
   match vLLM (and/or llama.cpp) **token-for-token** for ≥ 128 generated tokens (greedy determinism
   makes this near-free; `design.md` §12). This is the headline correctness gate.

---

## Tasks (each: build + test/sanitizer + clang-format + commit)

### Task m2-1 — `ModelConfig` (config.h)
- **Reading:** design §3.1, `qwen3.6-27b-architecture.md` §2.
- **Files:** `include/qus/model/config.h` (struct + helpers + `kCfg` + derived scales/eps);
  `tests/test_model_config.cpp` (`static_assert`s: `n_full()==16`, `n_gdn()==48`, `full_idx(63)==15`,
  `gdn_idx(62)==47`, `is_full` pattern). Register in `tests/CMakeLists.txt`.
- **DoD:** test PASS, format clean. Commit `feat(model): ModelConfig (frozen dims + schedule helpers)`.

### Task m2-2 — bindings + StepState + `bind()`
- **Reading:** design §2/§3.2–§3.5, `weight-handle-design.md`, `include/qus/core/{weight_store,tensor}.h`.
- **Files:** `include/qus/model/model.h` (FullLayerW/GdnLayerW/MlpW/StepState + class skeleton);
  `src/model/qwen3_6_27b.cpp` (`bind()` mapping `(TextCore, SourceKind, layer)` → fields, per design §3.3
  + the §3.4 packer table). Test `tests/test_model_bind.cpp` (NEEDS_SOURCE_DIR): load a small fixture
  q5090, construct the card, assert every `Weight*`/`Tensor*` is non-null with the expected `(n,k,qtype)`
  / shape for a sampled full + gdn layer.
- **DoD:** bind test PASS (or SKIP w/o GPU/fixture), format clean. Commit `feat(model): bindings + bind()`.

### Task m2-3 — schedule helpers (`attn_mix`/`gdn_mix`/`mlp_tail`/`run_layers`)
- **Reading:** design §4 (the exact schedule), the L1 op headers, `op_tester.h`.
- **Files:** the helpers in `qwen3_6_27b.cpp` exactly per §4 (out-param ops, phase-split by `ph`, q/k/v
  views, `GdnState`/`KVCache` slots). Test `tests/test_model_blocks.cpp`: run `attn_mix` (one full layer)
  and `gdn_mix` (one gdn layer) + `mlp_tail` on fixture weights + random activations, decode (`T=1`) and a
  small prefill (`T=4`); assert output shapes, finite values, and `compute-sanitizer` clean. (Numerical
  correctness of the full stack is gated at parity, not here.)
- **DoD:** block smoke PASS, sanitizer clean, format clean. Commit `feat(model): block schedule helpers`.

### Task m2-4 — drivers + `Engine` (end-to-end smoke)
- **Reading:** design §2/§4/§6, `engine.h`, `include/qus/core/{arena,device,kv_cache,state_store}.h`.
- **Files:** `prefill`/`decode_step` in `qwen3_6_27b.cpp`; `engine.h`/`engine.cpp` (resource ownership +
  `load` + `generate` loop + StepState + D2H next-token readback + EOS + a basic decode-tok/s readout).
  Test `tests/test_engine_smoke.cpp`: on the fixture, `generate(prompt, 8)` runs end to end, returns 8
  ids, no `cudaMalloc` per step (assert workspace arena reset), sanitizer clean.
- **DoD:** smoke PASS, sanitizer clean, format clean. Commit `feat(runtime): Engine load + generate loop`.

### Task m2-5 — reference generation (vLLM)
- **Reading:** `~/vllm` Qwen3.5/Next runner; `out/manifest.json` (the q5090 file path + dims).
- **Files:** `tools/parity/gen_reference.py` — load Qwen/Qwen3.6-27B in vLLM (or HF) on a **fixed prompt**
  (commit the prompt + tokenizer ids), greedy-decode ≥128 tokens, dump: the output token ids and the
  per-(layer,position) residual-stream hidden states (after each decoder layer) for the prompt's last
  position, to `tools/parity/ref/*.npy` (+ a manifest). Document the exact prompt + dump points.
- **DoD:** script runs, produces ref dumps + token ids; documented. Commit `feat(parity): vLLM reference dumps`.

### Task m2-6 — per-layer parity gate
- **Reading:** design §8 (Tap), m2-5 dumps.
- **Files:** a `FileTap` (design §8) + `tools/parity/compare.py` (or a C++ `tests/test_parity.cpp`
  gated on the real weight file + ref dumps via env var): load the **real** q5090, run prefill on the
  fixed prompt with `FileTap`, compare each layer's hidden state to the ref (cosine + max-abs). Report
  the first divergent layer.
- **DoD:** **cosine ≥ 0.999 per layer** vs the reference (fix the kernel/card until it passes; the first
  divergence localizes the bug — head-map, order, a transform). Commit `test(parity): per-layer parity passes`.

### Task m2-7 — greedy token-match gate
- **Reading:** m2-5 ids; `Engine::generate`.
- **Files:** `tests/test_greedy_match.cpp` (gated on the real weights + ref ids): `generate` greedy on
  the fixed prompt; assert **token-for-token equality** with the reference for ≥128 tokens.
- **DoD:** exact token-match (the M2 headline gate); record the basic decode tok/s. Commit
  `test(parity): end-to-end greedy token-match (M2 baseline)`.

---

## Dependencies & notes
- **Real weights:** the q5090 file (e.g. `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus`, per `out/manifest.json`).
  Parity tasks (m2-6/7) require it + a GPU with ~24–26 GB free (`design.md` §8). If absent, those tests
  must SKIP loudly (not silently pass).
- **Reference:** vLLM in `~/vllm` (or llama.cpp in `~/llama.cpp`) running the same model + prompt, greedy.
- **Head map:** the grouped GDN map (`h_v//3`) and the schedule order get their real proof here — a
  per-layer parity failure at a GDN layer is the signal to recheck it.
- Performance (decode tok/s → roofline, fusion, CUDA-graph) is **M3–M5**, not part of M2.

## Done criteria
The card + Engine build; `bind()` + block + engine smokes pass (sanitizer clean); **per-layer parity
≥0.999** and **greedy token-match** vs the reference both pass on the real weights; all code
clang-format clean. At that point the v1 text forward compute graph is complete and correct.

## Self-review notes (author)
- Tasks build bottom-up (config → bind → blocks → engine) with fixture smokes, then the real-model parity
  gates (per-layer then token-match) — the correctness gate, not perf.
- The schedule is implemented exactly per the (now signature-aligned) design §4; GdnState ssm layout +
  grouped head-map are called out as the likely parity-failure suspects.
- clang-format in every task DoD; reuses the Tier-1 subagent workflow/templates.
