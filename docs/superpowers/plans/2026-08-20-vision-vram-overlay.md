# Vision VRAM Overlay via Read-Only Weight Eviction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Rev 2 (2026-08-20).** Changes from Rev 1: resident vision mode is **retained as the default**; overlay is **opt-in** (`--vision-residency overlay`). Weight **streaming returns**, integrated as a block-wise weight provider with chunk-granular eviction (smaller peak footprint, smaller restore traffic). Delivery pipeline added: GitHub fork → feature branch → tested → PR into the upstream 3090 branch → merge into the local farm fork → deploy without waiting for the upstream merge. Every task records numbers into the measurement ledger.

**Goal:** With `--vision-residency overlay`, vision costs zero resident VRAM: vision weights live in pinned host RAM and stream block-by-block through a ~20 MiB ping-pong buffer inside a bounded exclusive window whose remaining VRAM is borrowed from temporarily evicted read-only text-weight chunks — never from the KV cache — then fully restored. Default behavior (`resident`) is bit-for-bit today's.

**Architecture:** Read-only weight groups (lm_head → token embedding → MTP head → tail decoder blocks) are laid out, in overlay mode, on VMM-backed fixed-size physical chunks with stable home VAs and pinned host mirrors. An overlay window computes the exact byte need (ping-pong + workspace + output), unmaps the minimal chunk prefix in ladder order, and remaps those *same physical pages* into a pre-reserved overlay VA range — zero device allocations inside the transaction. The vision encoder runs with a `BlockWeightProvider` that uploads block *n+1* weights on a copy stream while block *n* computes; merged embeddings + grid control land in pinned host memory; chunks remap home and only the evicted byte ranges re-upload from mirrors. A residency gate in `generation_service` serializes windows against concurrent text traffic at any `--max-concurrency`; waiting requests keep KV and session state untouched. Standard chunked prefill then consumes host embeddings via the existing `VisionPrefillPlan` path.

**Tech stack:** C++/CUDA (driver API VMM: `cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`/`cuMemUnmap`), copy/compute stream overlap with events, existing `WorkspaceArena`, CMake + ctest (`BUILD_TESTING=ON`), `gh` for fork/PR.

**Sources of record:** supersedes the 2026-08-19 handoff ("selective KV swap + Vision overlay"). Carried over: the exact memory planner (§7 formula), quiesce discipline, logical-KV-capacity admission (§12), verification spirit (§18), host-embedding→prefill reintegration. Rejected: KV-cache eviction/backup/restore (KV is the only *irreplaceable* device data; read-only weights restore by re-upload and need no backup), `cuMemRelease`+`cuMemCreate` cycles (physical handles are retained, so restore cannot fail by OOM), and the concurrency=1 boundary.

## Global Constraints

- **Resident mode is the default and stays bit-for-bit unchanged.** Overlay is opt-in via `--vision-residency overlay`; with the flag absent, no new allocations, no pinned mirrors, no behavior change. This is also the upstream-mergeable shape.
- The KV cache is never migrated, remapped, rewritten, or inspected. The mechanism must be provably independent of `--kv-dtype` (`bf16` | `int8` | `rk8v4`) and pool kind (`PagedKVPool`, `CyclicKVCache`).
- Works at any `--max-concurrency`. Vision windows are exclusive; concurrent text requests wait on the gate and resume unharmed.
- In overlay mode, every evictable byte and the vision tower have **pinned host mirrors created at artifact load**. No disk I/O on any hot path after load. Pinned RAM is spent only when overlay is enabled, and only as deep as `--overlay-max-vision-tokens` requires.
- The overlay transaction performs **zero device allocations**: unmap/map of pre-created physical chunks plus H2D/D2H copies. Restore is unconditional, idempotent, and re-uploads only evicted chunks.
- CUDA Graphs stay captured across windows (VAs stable); the gate guarantees no launches while chunks are unmapped.
- Overlay embeddings must be bitwise-identical to resident-mode execution of the same kernels and weights (streaming changes *where* weights come from, never their bytes).
- **Measurement ledger:** every task appends its numbers (VRAM watermark MiB, pinned MiB, ms) to `docs/superpowers/plans/2026-08-20-vision-vram-overlay-measurements.md`. Memory and latency figures are first-class deliverables and become the PR evidence.
- **Public-branch hygiene:** the feature branch contains no farm-specific content — no `deploy/`, GPU UUIDs, LAN/Tailscale IPs, or local operational notes. Farm wiring lives only in the local fork's own commits. No AI self-attribution in branch names, commits, or PR text.
- English for code, comments, docs, commits. No TODOs, no compatibility shims beyond the residency flag itself.

## Reference numbers (verified against the checkout on 2026-08-20)

| Fact | Value | Source |
|---|---|---|
| Weights artifact total | 16.96 GiB | `model-cards/Qwen3.8-27B-NInfer/artifact-manifest.json` |
| Vocabulary | 248320 × 5120 | `src/targets/qwen3_6_27b/impl/load/bindings.cpp:348` |
| lm_head + token_embedding (evictable, est.) | ≈ 1.6 GiB | vocab × hidden × ~5.25 bit ×2, exact in Task 1 |
| Vision tower quantized payload | ≈ 282 MiB | handoff §5, re-derived in Task 6 test |
| Overlay need, 1080p / 2074 merged tokens, streamed | ≈ 150 MiB (ping-pong ≈ 20 + workspace ≈ 110 + output ≈ 20) | handoff §5 minus weights |
| Overlay need, max 32768 merged tokens, streamed | ≈ 2.1 GiB | workspace scales linearly |
| KV pool ≈ 5.5–6 GiB @ 120k ⇒ | ≈ 50 KiB/token | budget arithmetic, measured in Task 1 |
| `TensorPlacement` today | `Device`, `ValidateOnly` | `src/artifact/binder.h:13` |
| VMM usage today | none | repo-wide grep |

Streaming vs. bulk upload: identical H2D volume (≈ 282 MiB either way), but streamed transfer hides behind block compute and shrinks the peak overlay footprint by ≈ 260 MiB — so a typical 1080p window evicts ≈ 150 MiB of chunks instead of ≈ 412 MiB, and restore drops from ≈ 17 ms to ≈ 6 ms. The ladder (lm_head + embedding ≈ 1.6 GiB, + MTP head, + tail decoder blocks ≈ 290 MiB each) covers even the maximum streamed burst.

---

### Task 1: Baseline measurements and authority read-through

**Files:** Create: `docs/superpowers/plans/2026-08-20-vision-vram-overlay-measurements.md` (the ledger; local fork only, not the feature branch — the PR gets a summary table).

**Read first:** `docs/maintainer/concurrent-inference-architecture.md`, `docs/maintainer/artifact-container.md`, `docs/maintainer/qwen3.8-27b-artifact.md`, `tests/README.md`.

- [ ] **Step 1: Capture the live VRAM map and host RAM on the farm.**

Run: `ssh aifarm 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv && free -g'` plus one `--log-stats-interval-ms` block from the ninfer-3090 container logs.
Expected: used near 24 GiB; record the weights/KV/workspace split as reported at startup, and the resident cost of `--vision` today (weights + workspace share) — this is the reclaim target.

- [ ] **Step 2: Record exact evictable-group sizes from the artifact.**

Run: the artifact inspection tool referenced in `docs/maintainer/artifact-container.md` (or a 20-line manifest reader) listing bytes for the lm_head group, `text/token_embedding`, `mtp/*`, per-decoder-block payload, `vision/*` total, and the largest single vision block (sizes the ping-pong buffer).
Expected: a table of exact bytes in the ledger; lm_head+embedding ≥ 1.5 GiB, vision ≈ 282 MiB.

- [ ] **Step 3: Confirm pinned budget headroom for overlay mode.**

Expected: host free RAM ≥ ladder depth for `--overlay-max-vision-tokens` (≈ 2.2 GiB at model max; less if capped) + vision mirror ≈ 0.3 GiB + cache budget (default 0.25 GiB) + slack. If not, cap `--overlay-max-vision-tokens` in deployment — do not thin the mirrors.

**Estimate:** 1 day.

---

### Task 2: Fork and feature branch

**Files:** git remotes and branch only; no source changes.

- [ ] **Step 1: Fork the upstream repo under the user's account.**

Run: `gh repo fork Don-Chad/ninfer-3090 --clone=false`
Expected: `iamwavecut/ninfer-3090` exists.

- [ ] **Step 2: Wire remotes in the local checkout and cut the branch from upstream, not from farm history.**

Run:
```
git remote add upstream https://github.com/Don-Chad/ninfer-3090.git 2>/dev/null; git remote add ghfork git@github.com:iamwavecut/ninfer-3090.git 2>/dev/null
git fetch upstream release/v0.6.0-rtx3090
git checkout -b feat/vision-vram-overlay upstream/release/v0.6.0-rtx3090
```
Expected: branch contains zero farm-only commits (`git log --oneline upstream/release/v0.6.0-rtx3090..HEAD` is empty). All Tasks 3–15 commit here; the farm checkout switches back to its own branch for deploy work only in Task 17.

- [ ] **Step 3: Push the branch skeleton.** `git push -u ghfork feat/vision-vram-overlay`

**Estimate:** 0.5 day.

---

### Task 3: Spike — VMM retain+remap under a captured CUDA Graph

**Files:** Create: `tests/test_vmm_graph_remap.cu`, register in `tests/CMakeLists.txt`.

**Produces:** proof that unmap → remap-elsewhere → remap-home → re-upload keeps a pre-captured graph launchable with correct results; measured per-chunk map/unmap latency (validates the 64 MiB chunk-size choice).

- [ ] **Step 1: Write the test.**

```cpp
// Sketch — adapt to the harness in tests/README.md.
CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
size_t gran; cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
const size_t chunk = align_up(64ull << 20, gran);
CUdeviceptr home, overlay;
cuMemAddressReserve(&home, 4 * chunk, gran, 0, 0);
cuMemAddressReserve(&overlay, 4 * chunk, gran, 0, 0);
CUmemGenericAllocationHandle h[4];
for (int i = 0; i < 4; ++i) { cuMemCreate(&h[i], chunk, &prop, 0); map_rw(home + i * chunk, chunk, h[i]); }
fill_pattern(home); capture_graph_reading(home, &graph);      // stream-capture a checksum kernel over home
for (int i = 2; i < 4; ++i) {                                  // evict a suffix — partial-group case
    cuMemUnmap(home + i * chunk, chunk); map_rw(overlay + (i - 2) * chunk, chunk, h[i]);
}
scribble(overlay, 2 * chunk);                                  // vision phase stand-in
for (int i = 2; i < 4; ++i) { cuMemUnmap(overlay + (i - 2) * chunk, chunk); map_rw(home + i * chunk, chunk, h[i]); }
upload_pattern_range(home + 2 * chunk, 2 * chunk);             // restore only evicted bytes from host mirror
launch(graph); assert_checksum_matches();
time_map_unmap_cycles();                                       // report per-chunk µs into the ledger
```

- [ ] **Step 2: Build and run on the 3090.**

Run: `cmake -B build -DBUILD_TESTING=ON && cmake --build build --target test_vmm_graph_remap && ctest --test-dir build -R vmm_graph_remap -V` (adjust to `tests/README.md`).
Expected: PASS; per-chunk map+unmap ≪ 1 ms (ledger). This is the only assumption that could invalidate the design — front-load it.

- [ ] **Step 3: Commit.** `git commit -m "test: prove VMM retain+remap keeps captured graphs valid"`

**Estimate:** 1 day.

---

### Task 4: `EvictableWeightPool` core primitive (chunk-granular)

**Files:** Create: `src/core/evictable_weight_pool.h`, `src/core/evictable_weight_pool.cu`; Test: `tests/test_evictable_weight_pool.cu`. Follow the style of `src/core/arena.{h,cu}` and `tests/test_arena.cpp`.

**Interfaces — Produces (used by Tasks 5, 6, 9):**

```cpp
namespace ninfer::core {
using GroupId = std::uint32_t;

struct OverlayLease {                  // move-only; alive == chunks are away from home
    std::byte*  base() const;          // overlay VA, granularity-aligned
    std::size_t capacity() const;
    std::byte*  bump(std::size_t bytes, std::size_t align);   // linear suballocation, asserts on overflow
};

class EvictableWeightPool {
public:
    // VA reservations only; chunk_bytes default 64 MiB, validated by the Task 3 spike.
    static EvictableWeightPool create(std::size_t home_reserve_bytes,
                                      std::size_t overlay_reserve_bytes,
                                      std::size_t chunk_bytes);
    GroupId add_group(std::string_view name, std::size_t bytes);   // load-time; chunks created + mapped home.
                                                                   // Registration order IS ladder priority.
    void*   home_ptr(GroupId) const;                               // stable for process lifetime
    void    attach_mirror(GroupId, const void* pinned_host, std::size_t bytes);
    OverlayLease evict(std::size_t bytes_needed, CUstream);        // minimal chunk prefix in ladder order;
                                                                   // partial groups allowed (chunk suffix first)
    void restore(OverlayLease&&, CUstream);                        // unmap overlay, map home, H2D only evicted
                                                                   // chunk ranges from mirrors, fence
};
}
```

- [ ] **Step 1: Write failing tests:** home pointers stable across two evict/restore cycles; restore reproduces mirror bytes exactly for partially-evicted groups (checksum on the full group); `bump` alignment; `evict(0)` returns an empty lease without driver calls; evict+restore of 150 MiB ≤ 15 ms and of 1.6 GiB ≤ 120 ms on the 3090 (ledger).
- [ ] **Step 2: Run to verify failure, implement, run to green:** `ctest --test-dir build -R evictable_weight_pool -V`.
- [ ] **Step 3: Commit.** `git commit -m "feat(core): chunk-granular VMM evictable weight pool with pinned mirrors"`

**Estimate:** 4 days.

---

### Task 5: Loader — residency modes, ladder groups, pinned mirrors

**Files:** Modify: `src/artifact/binder.h` (add `TensorPlacement::HostPinned`), `src/artifact/typed_binding.cpp`, `src/artifact/materializer.cpp`, `src/targets/qwen3_6_27b/impl/load/bindings.cpp` (and shared `qwen3_6` binding helpers); Test: extend `tests/test_artifact_materialization.cpp`.

**Interfaces — Consumes:** `EvictableWeightPool` (Task 4), residency mode from options (Task 12 defines the flag; loader takes an enum now). **Produces:**

- `resident` (default): byte-for-byte today's load path. No pool, no mirrors, vision `Device`.
- `overlay`: lm_head, `text/token_embedding`, `mtp/*` head, and the last N decoder blocks (N sized so ladder ≥ max streamed burst for `--overlay-max-vision-tokens`) materialize into pool groups with mirrors attached; `vision_backbone`/`vision/merger/*` materialize `HostPinned` only (permanent pinned mirror, no device copy); binding pointers still come out as ordinary device pointers (`home_ptr`), so kernels and graphs are untouched.

- [ ] **Step 1: Write the failing materialization test:** overlay mode → device residency excludes vision groups and the evictable-group table matches Task 1's byte table; resident mode → allocation set identical to the pre-change build (assert via the loader's existing accounting, not VRAM sampling).
- [ ] **Step 2: Implement.** `HostPinned` keeps the staging buffer as the permanent pinned mirror instead of freeing it.
- [ ] **Step 3: Run** `ctest --test-dir build -R artifact_materialization -V`; then load the real artifact once per mode and record both VRAM watermarks in the ledger (expected overlay saving at load: ≈ 282 MiB weights; workspace share lands in Task 10).
- [ ] **Step 4: Commit.** `git commit -m "feat(load): overlay residency — ladder groups, pinned mirrors, host-resident vision tower"`

**Estimate:** 4 days.

---

### Task 6: Exact overlay planner

**Files:** Create: `src/targets/qwen3_6/impl/runtime/overlay_planner.h` (+ `.cpp` if needed); Test: `tests/targets/` unit test.

**Interfaces — Consumes:** `VisionItemControl`, `workspace_bytes(item)` / `workspace_capacity_bytes` (public in `vision_context.h`), group table (Task 5). **Produces:**

```cpp
struct OverlayPlan {
    std::size_t pingpong_bytes;        // 2 × largest vision block payload (from the artifact, Task 1 ledger)
    std::size_t workspace_bytes;       // exact, for this batch of items
    std::size_t output_bytes;          // merged tokens × 5120 × sizeof(bf16)
    std::size_t total_bytes;           // sum, aligned to chunk size, + fixed allowance
    std::size_t evict_bytes;           // max(0, total − free_vram − idle_workspace)
    std::size_t vision_upload_bytes;   // full tower payload, for telemetry
};
OverlayPlan plan_overlay(std::span<const VisionItemControl>, const GroupTable&,
                         std::size_t free_vram_bytes, std::size_t idle_workspace_bytes);
```

- [ ] **Step 1: Write failing tests** pinning reference points: 2074 merged tokens ⇒ total ≈ 150 MiB (±chunk alignment); 32768 ⇒ ≈ 2.1 GiB; `evict_bytes == 0` when free VRAM + idle workspace already cover the burst; totals never include the full tower payload (streaming invariant).
- [ ] **Step 2: Implement, run** `ctest -R overlay_planner -V`, green.
- [ ] **Step 3: Commit.** `git commit -m "feat(runtime): exact streamed-vision overlay planner"`

**Estimate:** 1.5 days.

---

### Task 7: `BlockWeightProvider` — streamed vision weights

**Files:** Create: `src/targets/qwen3_6/impl/runtime/vision_weight_provider.{h,cpp}`; Modify: the vision encode driver (`src/targets/qwen3_6/impl/vision/control.cpp`, `vision_context_impl.h`) to consume the provider; Test: GPU test with the artifact fixture (`tests/artifact_fixture.h`).

**Interfaces — Produces:**

```cpp
struct VisionBlockWeights { /* device pointer set for one encoder block, matching current bindings */ };

class BlockWeightProvider {
public:
    virtual const VisionBlockWeights& acquire(std::uint32_t block_index, CUstream compute) = 0;
    // ResidentProvider: returns the resident device pointers; zero synchronization, zero cost.
    // StreamingProvider(pinned mirror, pingpong from OverlayLease): uploads block n+1 on a copy
    //   stream while block n computes; acquire() makes the compute stream wait on the block's copy event.
};
```

The encoder loop changes from "read bound pointers" to "read `acquire(i)`" — the only structural change to the vision path, shared by both modes.

- [ ] **Step 1: Write the failing GPU test:** encode one 1080p fixture image twice — `ResidentProvider` vs `StreamingProvider` over the same pinned bytes — and assert bitwise-equal embeddings; measure copy-stall time (target ≈ 0: per-block upload ≈ 9 MiB ≈ 0.4 ms ≪ block compute; ledger).
- [ ] **Step 2: Implement, run, green.**
- [ ] **Step 3: Commit.** `git commit -m "feat(vision): block weight provider with streamed ping-pong upload"`

**Estimate:** 3 days.

---

### Task 8: Residency gate in the serving layer

**Files:** Modify: `src/serve/generation_service.{h,cpp}`; Test: extend `tests/test_admission_policy.cpp` + new concurrency integration test.

**Read first:** `docs/maintainer/concurrent-inference-architecture.md` — the gate must compose with the existing admission/step machinery, not bypass it.

**Interfaces — Produces:**

```cpp
class ResidencyGate {                      // owned by GenerationService; inert in resident mode
public:
    StepPass    enter_step();              // shared side: every decode/prefill step holds one
    WindowGuard acquire_window();          // exclusive: blocks new StepPass, waits in-flight passes drain
};
```

Semantics: vision-bearing requests queue as overlay jobs; one window serves **all currently queued images** up to `--overlay-max-window-items` (batching amortizes evict/restore). Text requests admitted during a window wait — their sessions, KV pages, and MTP state untouched by construction. Cancellation of a queued/running vision item never skips restore. `--pending-timeout-ms` continues to bound waiting.

- [ ] **Step 1: Write failing tests:** (a) `--max-concurrency 4`: three streaming text requests + one vision request → text outputs token-identical to a run without the vision request, and no step overlaps the window (gate counters); (b) two queued vision requests encode in one window; (c) cancelling a vision request mid-queue releases the gate and text proceeds.
- [ ] **Step 2: Implement, run, green.**
- [ ] **Step 3: Commit.** `git commit -m "feat(serve): exclusive residency gate for vision overlay windows"`

**Estimate:** 4 days. (Independent of Task 7 — parallelizable.)

---

### Task 9: Overlay window executor

**Files:** Create: `src/targets/qwen3_6/impl/runtime/vision_overlay.{h,cpp}`; Test: `tests/targets/` GPU test with the real artifact fixture.

**Interfaces — Consumes:** `OverlayPlan` (Task 6), `EvictableWeightPool` (Task 4), `BlockWeightProvider` (Task 7), `VisionContextImpl::encode` (existing). **Produces:**

```cpp
struct HostVisionResult {
    PinnedSpan embeddings;                 // merged tokens × 5120, BF16
    qwen3_6::VisionItemControl control;    // grid THW etc. — position source for prefill
    std::array<std::uint8_t, 32> image_digest;
};
std::vector<HostVisionResult> run_overlay_window(std::span<const VisionItem>,
                                                 EvictableWeightPool&, const OverlayPlan&, CUstream);
```

Transaction order (gate-exclusive): `evict(plan.evict_bytes)` → `bump` {ping-pong, workspace, output} → per item: `encode(item, output, workspace, streaming_provider)` → D2H embeddings to pinned result → `restore` (unconditional, also on every error path — scope guard) → debug-build checksum of restored chunk ranges against mirrors.

- [ ] **Step 1: Write the failing GPU test:** one 1080p fixture image through `run_overlay_window`; embeddings bitwise-equal to `ResidentProvider` reference (Task 7); VRAM watermark returns to pre-window value; an injected failure after `evict` still restores (checksum passes); window wall time recorded (ledger).
- [ ] **Step 2: Implement, run, green.**
- [ ] **Step 3: Commit.** `git commit -m "feat(runtime): transactional streamed vision overlay window"`

**Estimate:** 3.5 days.

---

### Task 10: Workspace decoupling and prefill from host embeddings

**Files:** Modify: `src/targets/qwen3_6/impl/runtime/vision_context_impl.h`, `layouts_impl.h`/`instantiate.h` (workspace sizing), `vision_prefill.h` consumers in `api_impl.h`/`request_plan_impl.h`; Test: extend the Task 9 GPU test into a full generate route.

**Consumes:** `HostVisionResult`. **Produces:** (a) overlay mode sizes the startup `WorkspaceArena` for **text only** — vision workspace always comes from the overlay lease; resident mode keeps today's sizing (this split is where the reclaimed-VRAM prize is realized); (b) prefill consumes pinned embeddings chunk-by-chunk (`--prefill-chunk`) through the existing `VisionPrefillPlan`/`VisionUseSpan` spans, positions derived from the stored `control` exactly as the resident path derives them from a fresh encode.

- [ ] **Step 1: Write the failing route test:** image + prompt → full generation in overlay mode; greedy first-64-token output identical to resident mode on the same build; logical-KV admission: an image whose merged tokens exceed remaining `--kv-capacity` slots is rejected with a clear error before any window opens (handoff §12).
- [ ] **Step 2: Implement, run, green.** Record both modes' post-load VRAM watermarks in the ledger — overlay saving now includes the vision workspace share (expected total ≈ 400–600 MiB; exact number is the Task 17 capacity input).
- [ ] **Step 3: Commit.** `git commit -m "feat(runtime): text-only resident workspace in overlay mode; prefill from host embeddings"`

**Estimate:** 3 days.

---

### Task 11: Session-scoped embedding reuse cache

**Files:** Create: `src/serve/vision_result_cache.{h,cpp}`; wire in `generation_service.cpp`; Test: unit + route test.

**Consumes/Produces:** key = (image digest, preprocessing params, artifact sha256); value = `HostVisionResult`; LRU with byte budget (`--overlay-cache-mib`, default 256). A hit skips the window entirely — repeated turns over the same image (OpenPlotva regenerate/retry flows) cost zero overlay time. Mode-agnostic: in resident mode a hit still skips encode compute.

- [ ] **Step 1: Failing test:** second request with the same image opens no window (gate window counter unchanged) and produces identical output.
- [ ] **Step 2: Implement, run, green. Commit.** `git commit -m "feat(serve): session-scoped vision embedding cache"`

**Estimate:** 2 days.

---

### Task 12: Options, validation, and documentation

**Files:** Modify: `src/serve/serve_options.{h,cpp}`, `docs/cli.md`, `docs/serving.md`, `docs/rtx-3090-linux.md`, `README.md` (VRAM budget sections).

- [ ] **Step 1:** New flags: `--vision-residency resident|overlay` (default `resident` — `--vision` alone behaves exactly as today), `--overlay-max-vision-tokens` (default: model max 32768; sizes ladder depth and pinned budget), `--overlay-max-window-items` (default 4), `--overlay-cache-mib` (default 256). Startup validation in overlay mode: pinned mirrors + cache budget must fit host RAM or fail fast with the measured shortfall.
- [ ] **Step 2:** Update the usage string and docs tables; add an overlay section with the measured budget numbers from the ledger. Run `git diff --check` and the schema tests (`ctest -R 'anthropic_schema|responses_schema|openai'` — API shape unchanged).
- [ ] **Step 3: Commit.** `git commit -m "feat(serve): opt-in overlay vision residency options and docs"`

**Estimate:** 1 day.

---

### Task 13: Telemetry

**Files:** Modify: `src/serve/generation_service.cpp` (log-stats block), `src/serve/request_log.cpp`.

- [ ] **Step 1:** Emit per stats interval: `vision_residency`, `overlay_windows`, `overlay_window_ms` (last/max), `overlay_evicted_mib`, `overlay_restore_ms`, `overlay_stream_stall_ms`, `overlay_cache_hits`, `overlay_queue_wait_ms`. Annotate vision requests in the request log with window duration and cache-hit flag.
- [ ] **Step 2:** Assert fields appear in a route test; commit. `git commit -m "feat(serve): overlay window telemetry"`

**Estimate:** 1 day.

---

### Task 14: Universality matrix and fault injection

**Files:** Create: `tests/test_overlay_matrix.py` (driving the serve binary like `test_bench_matrix.py`); extend GPU tests for fault cases.

The mechanism never touches KV, so universality is by construction — this task proves it and pins it. Pairwise reduction keeps it fast without losing axis coverage (~9 serve runs instead of 24):

- [ ] **Step 1: Axis sweeps.** Dtype sweep: `--kv-dtype {bf16, int8, rk8v4}` at overlay + mtp(`--draft-tokens 3 --lm-head-draft`) + concurrency 1 — text-only outputs unchanged by an interleaved vision request; image route correct; near-full-KV admission rejection. Concurrency sweep: `{1, 4}` at rk8v4 + mtp + overlay. Mode A/B: `{resident, overlay}` at rk8v4 + mtp — token-identical image routes. Spec sweep: `{none, mtp}` at rk8v4 + overlay. Multi-image request in one cell.
- [ ] **Step 2: Fault injection (GPU tests):** cancel mid-encode (restore still runs, next text step correct); simulated H2D failure during restore → retry succeeds (idempotent); debug-build stray-access guard: launching a text kernel while chunks are unmapped traps (negative test on the gate).
- [ ] **Step 3: Commit.** `git commit -m "test: overlay universality matrix and fault injection"`

**Estimate:** 4 days.

---

### Task 15: Performance evidence (the A/B table)

**Files:** Modify: `docs/performance.md`; bench additions beside `tests/test_bench_matrix.py`.

- [ ] **Step 1:** Measure on the 3090 and write into the ledger + `docs/performance.md` as one resident-vs-overlay table: (a) text decode throughput, overlay idle vs resident vs pre-change baseline — required: no regression beyond noise; (b) window overhead = `overlay_window_ms − encode_ms` for 1080p — target ≤ 60 ms streamed (≈ 6 ms restore + ≈ 0 stalled upload + drain + sync); (c) image TTFT resident vs overlay — expected within ~5% (encode ≈ 3.3 s dominates); (d) VRAM watermark per mode; (e) pinned RAM per mode.
- [ ] **Step 2: Commit.** `git commit -m "docs(perf): resident vs overlay measurements on RTX 3090"`

**Estimate:** 2 days.

---

### Task 16: Upstream pull request

**Files:** none (GitHub).

- [ ] **Step 1:** Rebase `feat/vision-vram-overlay` on current `upstream/release/v0.6.0-rtx3090`, re-run the matrix if the rebase touched anything, push to `ghfork`.
- [ ] **Step 2:** Open the PR: `gh pr create --repo Don-Chad/ninfer-3090 --base release/v0.6.0-rtx3090 --head iamwavecut:feat/vision-vram-overlay` — body: motivation (KV-capacity reclaim on 24 GiB cards), design summary (chunked VMM eviction ladder, streamed provider, gate), the Task 15 A/B table, and the explicit invariants (default unchanged; KV untouched; zero-allocation transaction). Follow upstream `CONTRIBUTING.md`.
- [ ] **Step 3:** Track review feedback as follow-up work; it does **not** block Task 17.

**Estimate:** 0.5 day.

---

### Task 17: Merge into the local farm fork and deploy (not waiting for upstream)

**Files:** Modify: `deploy/compose.yaml`, `deploy/README.md`; run `deploy/smoke-protocols.sh`.

- [ ] **Step 1:** On the farm branch: `git merge feat/vision-vram-overlay`; build and pin the new image (`NINFER_SOURCE_REVISION` = merge commit), deploy to GPU2, confirm health.
- [ ] **Step 2:** Enable overlay in compose (`--vision-residency overlay`, plus `--overlay-max-vision-tokens` per the Task 1 pinned budget). Read the measured freed VRAM from startup logs and raise `--kv-capacity`/`--max-context` by the measured reclaim (expected ≈ +8–10k tokens at ≈ 50 KiB/token — use the ledger figure, not the estimate). Align Discovery capacity (precedent: commit `f7c100cb`).
- [ ] **Step 3:** Live verification per the handoff's §18 discipline: one real 1080p multimodal request with a long context → correct answer; follow-up text turn in the same session without re-encode (cache hit in telemetry); a concurrent text request during the window completes; telemetry fields sane; ledger updated with production numbers. A green health endpoint alone is not acceptance.
- [ ] **Step 4: Commit (farm branch only).** `git commit -m "deploy: enable overlay vision residency on GPU2 and reclaim KV capacity"`

**Estimate:** 1 day.

---

## Acceptance criteria

1. `--vision-residency resident` (and flag absent) is bit-for-bit pre-change behavior; overlay is opt-in.
2. In overlay mode, vision costs zero resident VRAM outside windows (weights and workspace), proven by watermarks in the ledger; farm `--kv-capacity` raised by the measured reclaim.
3. Overlay embeddings bitwise-equal to resident mode; full-route outputs token-identical across modes; universality matrix green across `bf16`/`int8`/`rk8v4`, spec modes, and concurrency 1/4; no window ever skips restore.
4. Window overhead ≤ 60 ms over encode for 1080p (streamed); text throughput regression within noise; all numbers in the ledger and `docs/performance.md`.
5. PR open against `Don-Chad/ninfer-3090@release/v0.6.0-rtx3090` with the A/B table; farm deployed from the local merge without waiting for the upstream merge.

## Risks

1. **Graph×VMM interaction differs from expectation** — retired first by the Task 3 spike before any structural work.
2. **Hidden device pointers into evicted chunks** (persistent kernel params, captured copy ops) — mitigated by the debug stray-access trap (Task 14) and by evicting only groups with enumerable consumers (lm_head, embedding, MTP, whole blocks).
3. **Host RAM shortfall for pinned mirrors** — measured in Task 1 before code; bounded by `--overlay-max-vision-tokens`, never by thinning mirrors.
4. **Long window stalls concurrent text** (encode is seconds for huge images) — inherent to one GPU; bounded by `--overlay-max-vision-tokens`/`--overlay-max-window-items`, visible via `overlay_queue_wait_ms`.
5. **Upstream review divergence after farm deploy** — deploy proceeds from the local merge by design; review-driven changes land as follow-ups and re-deploy through the same pipeline.

## Out of scope

- OpenPlotva changes: none — the HTTP contract is unchanged; added latency is invisible inside the dialog's 120 s budget.
- Cross-session/persistent embedding cache (privacy review needed first, per handoff §13).
- Multi-GPU disaggregated encode — revisit only if a zero-stall requirement appears.
