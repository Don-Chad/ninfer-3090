# Qwen3.6 Chat-Template E2E Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current raw-text M2.8 prompt fixture pipeline with Qwen3.6 chat-template-rendered `.ids`, make stop-token handling match the model generation config, update all e2e report/schema/summary/readiness surfaces, and regenerate current evidence so decoded e2e output is meaningful assistant content with thinking disabled.

**Architecture:** Python owns Hugging Face tokenizer loading, Jinja chat-template rendering through `tokenizer.apply_chat_template`, thinking-mode control, tokenizer/generation provenance, manifest generation, fixture regeneration, and decoded sidecars. C++ remains tokenizer-free and consumes `.ids`; it reads fixture identity metadata, applies a normalized stop-token id list during the manual prefill/decode loop, and emits chat-template-aware report fields.

**Tech Stack:** Python 3 standard library plus existing `transformers` tokenizer dependency; C++17 benchmark support code; existing CMake/CTest; local tokenizer path `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16`; existing q5090 runtime artifact for real model runs.

---

## Non-Negotiable Decisions

- [ ] Keep `qus_e2e_bench` tokenizer-free. Do not add C++ tokenizer, Jinja, Hugging Face, or Python runtime dependency.
- [ ] Replace old raw `.txt` fixture sources with committed `.messages.json` sources.
- [ ] Keep canonical C++ input as `.ids`.
- [ ] Render fixture ids in Python with Qwen3.6 chat template, `add_generation_prompt=true`, `add_special_tokens=false`, and `enable_thinking=false`.
- [ ] Replace old single `eos_token_id` e2e identity with canonical `stop_token_ids`.
- [ ] Default Qwen3.6 stop token ids are `[248046, 248044]`, matching local `generation_config.json`.
- [ ] Treat existing raw-text baseline summaries, decoded sidecars, and readiness evidence as invalid. Delete or regenerate them before finishing.
- [ ] Do not regenerate q5090 weights for this change. This plan changes prompt fixtures, reports, summaries, and docs, then reruns benchmarks against the current accepted q5090 artifact.

## Expected Final State

- [ ] `bench/fixtures/prompts/` contains one `.messages.json` and one `.ids` per required case.
- [ ] `bench/fixtures/prompts/` no longer contains required-case `.txt` sources.
- [ ] `bench/fixtures/prompts/m2.8-v1.manifest.json` records chat-template, tokenizer, generation-config, messages, rendered-prompt, ids, and stop-token provenance.
- [ ] Python fixture tooling can regenerate and check fixtures entirely offline from the local tokenizer path.
- [ ] C++ bench CLI supports repeated `--stop-token-id <id>` and keeps deprecated `--eos-token-id <id>` as a one-token alias.
- [ ] Raw e2e reports include `prompt_format`, `messages_path`, `messages_sha256`, `rendered_prompt_sha256`, `add_generation_prompt`, `add_special_tokens`, `chat_template_kwargs`, and `stop_token_ids`.
- [ ] Report comparison and summary tooling treat those fields as hard identity.
- [ ] Decoded sidecars include both raw special-token-preserving text and clean text with special tokens skipped.
- [ ] `docs/bench/baselines/` and `docs/m3-readiness.md` no longer claim readiness from old raw-text reports.
- [ ] Docs describe `.messages.json -> apply_chat_template -> .ids`, not `.txt -> encode -> .ids`.

## Current Files To Touch

- [ ] `tools/bench/tokenizer_common.py`
- [ ] `tools/bench/tokenize_prompts.py`
- [ ] `tools/bench/decode_e2e_report.py`
- [ ] `tools/bench/e2e_report_common.py`
- [ ] `tools/bench/compare_e2e_reports.py`
- [ ] `tools/bench/make_baseline_summary.py`
- [ ] `tools/bench/make_long_prompt.py`
- [ ] `bench/e2e_bench_support.h`
- [ ] `bench/e2e_bench_support.cpp`
- [ ] `bench/e2e_bench.cpp`
- [ ] `tests/test_bench_tokenizer_tools.py`
- [ ] `tests/test_bench_report_tools.py`
- [ ] `tests/test_e2e_bench_support.cpp`
- [ ] `bench/fixtures/prompts/*`
- [ ] `bench/README.md`
- [ ] `tools/bench/README.md`
- [ ] `docs/bench/e2e-report-schema.md`
- [ ] `docs/m2.8-pre-m3-standard.md`
- [ ] `docs/m2.8-pre-m3-standard.zh.md`
- [ ] `docs/m3-readiness.md`
- [ ] `docs/bench/baselines/*`

## Implementation Tasks

### 1. Add Python Chat Fixture Constants And Metadata

- [ ] Update `tools/bench/tokenizer_common.py` with explicit Qwen3.6 fixture policy constants:

```python
PROMPT_FORMAT = "qwen3.6-chat-template"
ADD_GENERATION_PROMPT = True
ADD_SPECIAL_TOKENS = False
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
STOP_TOKEN_IDS = [248046, 248044]
STOP_TOKEN_NAMES = {
    "248046": "<|im_end|>",
    "248044": "<|endoftext|>",
}
MESSAGE_FILE_SUFFIX = ".messages.json"
```

- [ ] Extend tokenizer metadata hashing to include `chat_template.jinja` and `generation_config.json`:

```python
TOKENIZER_HASH_FILES = (
    ("tokenizer_json_sha256", "tokenizer.json"),
    ("tokenizer_config_sha256", "tokenizer_config.json"),
    ("special_tokens_map_sha256", "special_tokens_map.json"),
    ("chat_template_jinja_sha256", "chat_template.jinja"),
    ("generation_config_sha256", "generation_config.json"),
)
```

- [ ] Keep committed metadata paths redacted:

```python
metadata = {
    "tokenizer_source": TOKENIZER_SOURCE,
    "tokenizer_model_id": TOKENIZER_MODEL_ID,
    "tokenizer_path": "",
}
```

- [ ] Add `read_messages(path: Path) -> list[dict[str, str]]`:
  - require JSON array;
  - require at least one message;
  - allow only `role` values `system`, `user`, and `assistant` for this first text-only scope;
  - require `content` to be a non-empty string;
  - reject tool, multimodal, or nested content.

- [ ] Add tests in `tests/test_bench_tokenizer_tools.py` before implementation:

```python
def test_messages_json_validation_rejects_empty_and_bad_roles(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.messages.json"
        path.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty array"):
            common.read_messages(path)

        path.write_text('[{"role":"tool","content":"x"}]\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported role"):
            common.read_messages(path)
```

- [ ] Add a metadata test that asserts `chat_template_jinja_sha256` and `generation_config_sha256` are present when the files exist and are empty strings only when the files are absent.

### 2. Replace Raw Text Encoding With `apply_chat_template`

- [ ] Change `tools/bench/tokenize_prompts.py` so `_encode` is replaced by a chat renderer:

```python
def _render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=common.ADD_GENERATION_PROMPT,
        **common.CHAT_TEMPLATE_KWARGS,
    )
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=common.ADD_GENERATION_PROMPT,
        return_dict=False,
        **common.CHAT_TEMPLATE_KWARGS,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("chat template rendered an empty prompt")
    if not isinstance(ids, list) or not all(isinstance(v, int) for v in ids) or not ids:
        raise RuntimeError("chat template rendered no token ids")
    return rendered, ids
```

- [ ] If the local tokenizer returns a tensor-like or nested structure for ids, normalize only documented simple outputs. Do not silently accept ambiguous structures.

- [ ] Make `_expected_manifest` read `<case>.messages.json`, compute rendered prompt hash from the rendered string, compute ids hash from the written `.ids`, and record:

```json
{
  "name": "cn_short",
  "messages": "cn_short.messages.json",
  "ids": "cn_short.ids",
  "prompt_tokens": 0,
  "messages_sha256": "",
  "rendered_prompt_sha256": "",
  "ids_sha256": "",
  "prompt_format": "qwen3.6-chat-template",
  "add_generation_prompt": true,
  "add_special_tokens": false,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

- [ ] Add manifest-level generation block:

```json
{
  "generation": {
    "stop_token_ids": [248046, 248044],
    "stop_token_names": {
      "248046": "<|im_end|>",
      "248044": "<|endoftext|>"
    },
    "sampling_policy": "greedy; Qwen sampling defaults are provenance only"
  }
}
```

- [ ] Keep `write_fixtures(..., check=True)` strict:
  - stale `.ids` raises `RuntimeError("stale fixture ids: ...")`;
  - stale manifest raises `RuntimeError("stale fixture manifest: ...")`;
  - missing messages file raises `FileNotFoundError` or a clear `RuntimeError`.

- [ ] Update `FakeTokenizer` in `tests/test_bench_tokenizer_tools.py`:

```python
class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False, **kwargs):
        if add_generation_prompt is not True:
            raise AssertionError("fixtures must include assistant generation prompt")
        if kwargs != {"enable_thinking": False}:
            raise AssertionError(f"unexpected chat template kwargs: {kwargs}")
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if not tokenize:
            return rendered
        return [ord(ch) % 251 for ch in rendered]
```

- [ ] Update tokenizer tests to create `<case>.messages.json` instead of `<case>.txt`.

- [ ] Assert the generated manifest no longer contains `txt`, `txt_sha256`, or raw-text fields.

### 3. Replace Committed Prompt Fixtures

- [ ] For each required case, create a committed `<case>.messages.json`:
  - `cn_short.messages.json`
  - `en_short.messages.json`
  - `code_short.messages.json`
  - `math_short.messages.json`
  - `long_2k.messages.json`

- [ ] Preserve the intent of the existing prompts but make each source a messages array:

```json
[
  {
    "role": "user",
    "content": "..."
  }
]
```

- [ ] For `long_2k`, update `tools/bench/make_long_prompt.py` so it writes `long_2k.messages.json` and produces one user message whose content remains long enough to exercise prefill. It must not write `long_2k.txt`.

- [ ] Delete required-case `.txt` fixtures after `.messages.json` files exist and tests expect the new source format.

- [ ] Regenerate fixtures from the local tokenizer path:

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --fixture-dir bench/fixtures/prompts
```

- [ ] Immediately verify committed fixtures:

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --fixture-dir bench/fixtures/prompts \
  --check
```

- [ ] Inspect `cn_short.ids` manually enough to confirm it starts with the Qwen chat-template markers:

```bash
head -n 1 bench/fixtures/prompts/cn_short.ids
```

The exact ids need not be hard-coded in docs, but they should include the rendered prompt's `<|im_start|>user`, `<|im_end|>`, `<|im_start|>assistant`, `<think>`, and `</think>` structure when decoded.

### 4. Add C++ Stop Token List Support

- [ ] Update `bench/e2e_bench_support.h`:

```cpp
struct CaseRunInput {
    std::string name;
    std::string prompt_ids_path;
    std::vector<int> prompt_ids;
    int requested_max_new_tokens = 0;
    std::vector<int> stop_token_ids;
    int max_context = 0;
    std::string prompt_format = "qwen3.6-chat-template";
    std::string messages_path;
    std::string messages_sha256;
    std::string rendered_prompt_sha256;
    bool add_generation_prompt = true;
    bool add_special_tokens = false;
    bool enable_thinking = false;
};

struct RunOptions {
    ...
    std::vector<int> stop_token_ids = {248046, 248044};
    bool explicit_stop_token_ids = false;
};
```

- [ ] Keep `--eos-token-id <id>` as deprecated compatibility input:
  - parsing it sets `stop_token_ids` to exactly `{id}`;
  - parsing it sets `explicit_stop_token_ids=true`;
  - usage text labels it deprecated.

- [ ] Add repeated `--stop-token-id <id>`:
  - first explicit `--stop-token-id` clears defaults;
  - later occurrences append;
  - ids must be non-negative integers;
  - normalize by removing duplicates while preserving first occurrence order;
  - error if the final list is empty.

- [ ] Add helper:

```cpp
std::vector<int> normalize_stop_token_ids(std::vector<int> ids);
bool is_stop_token(int token, const std::vector<int>& stop_token_ids);
```

- [ ] Update usage text:

```text
  --stop-token-id <id>      Stop on token id; repeatable. Default: 248046,248044 for Qwen3.6.
  --eos-token-id <id>       Deprecated alias for exactly one --stop-token-id.
```

- [ ] Add C++ tests in `tests/test_e2e_bench_support.cpp` before implementation:
  - default options contain `[248046, 248044]`;
  - repeated `--stop-token-id 248046 --stop-token-id 248044` parses in order;
  - duplicate ids are deduped;
  - `--eos-token-id 123` yields `[123]`;
  - malformed negative stop ids fail;
  - usage text contains `--stop-token-id` and labels `--eos-token-id` deprecated.

### 5. Add C++ Fixture Manifest Metadata Reading

- [ ] Vendor a small header-only third-party JSON library instead of writing a custom JSON parser. Use `nlohmann/json` unless a blocking integration issue appears.

- [ ] Add the vendored header under:

```text
third_party/nlohmann/json.hpp
```

- [ ] Add the corresponding license notice under:

```text
third_party/nlohmann/LICENSE.MIT
```

- [ ] Update CMake so only `qus_e2e_bench` and `qus_e2e_bench_support_test` receive the `third_party` include path. Do not make the dependency part of `qus_core`.

- [ ] Add benchmark-local manifest support in `bench/e2e_bench_support.{h,cpp}` or separate `bench/e2e_fixture_manifest.{h,cpp}` if that is cleaner.

- [ ] Required data model:

```cpp
struct FixtureCaseMetadata {
    std::string name;
    std::string messages_path;
    std::string messages_sha256;
    std::string ids_path;
    std::string ids_sha256;
    std::string rendered_prompt_sha256;
    std::string prompt_format;
    bool add_generation_prompt = true;
    bool add_special_tokens = false;
    bool enable_thinking = false;
};

struct FixtureManifestMetadata {
    std::string fixture_set;
    std::string manifest_path;
    std::string manifest_sha256;
    std::vector<int> stop_token_ids;
    std::map<std::string, FixtureCaseMetadata> cases_by_name;
};
```

- [ ] Parser requirements:
  - read `bench/fixtures/prompts/m2.8-v1.manifest.json` from `RunOptions.fixture_manifest_path`;
  - fail if `fixture_set` is missing or different from `m2.8-v1`;
  - fail if `generation.stop_token_ids` is absent or empty;
  - fail if a requested case is missing;
  - fail if the case `ids` path does not match the requested `prompt_ids_path`;
  - fail if `prompt_format != "qwen3.6-chat-template"`;
  - fail if `add_generation_prompt != true`;
  - fail if `add_special_tokens != false`;
  - fail if `chat_template_kwargs.enable_thinking != false`;
  - fail if SHA fields are missing or empty.

- [ ] Do not parse JSON with ad hoc substring matching. Parse the fixture manifest through the vendored JSON library and validate field types explicitly.

- [ ] Add C++ tests:
  - valid manifest loads expected fields;
  - missing case fails;
  - old raw manifest containing `txt_sha256` and no `prompt_format` fails;
  - wrong `enable_thinking=true` fails;
  - wrong stop ids fail when report generation compares CLI override to manifest.

### 6. Emit Chat-Template Case Identity In Raw Reports

- [ ] Update `CaseReport` serialization in `bench/e2e_bench_support.cpp` to replace:

```json
"eos_token_id": -1
```

with:

```json
"prompt_format": "qwen3.6-chat-template",
"messages_path": "bench/fixtures/prompts/cn_short.messages.json",
"messages_sha256": "",
"rendered_prompt_sha256": "",
"prompt_ids_path": "bench/fixtures/prompts/cn_short.ids",
"prompt_ids_sha256": "",
"prompt_tokens": 0,
"requested_max_new_tokens": 0,
"add_generation_prompt": true,
"add_special_tokens": false,
"chat_template_kwargs": {
  "enable_thinking": false
},
"stop_token_ids": [248046, 248044]
```

- [ ] Preserve `fixture_set`, `fixture_manifest_path`, and `fixture_manifest_sha256`.

- [ ] Make `stop_token_ids` per-case identity. It must match the normalized run stop list used in the decode loop.

- [ ] Update `bench/e2e_bench.cpp`:
  - load manifest metadata before running cases;
  - attach each case metadata to `CaseRunInput`;
  - if CLI stop token list differs from manifest `generation.stop_token_ids`, fail unless this is an explicitly documented non-gate run mode. For this plan, prefer fail-fast to avoid ambiguous evidence;
  - do not rely on `EngineOptions.eos_token_id` for manual bench loop stopping;
  - check stop tokens after prefill and after every `decode_step()`;
  - use stop reason `stop_token` when a stop token is observed.

- [ ] Expected loop shape:

```cpp
bool stopped = false;
if (input.requested_max_new_tokens > 0) {
    int token = engine.prefill(input.prompt_ids);
    generated.push_back(token);
    stopped = is_stop_token(token, input.stop_token_ids);
}

while (!stopped && decode_steps < requested_decode_steps) {
    int token = engine.decode_step();
    generated.push_back(token);
    stopped = is_stop_token(token, input.stop_token_ids);
}
```

- [ ] Add C++ report serialization tests that assert:
  - `prompt_format` is emitted;
  - `messages_path` and `messages_sha256` are emitted;
  - `chat_template_kwargs.enable_thinking` is emitted as `false`;
  - `stop_token_ids` is emitted as a JSON array;
  - old `eos_token_id` is absent from the case object.

### 7. Update Python Report Validators And Comparison

- [ ] In `tools/bench/e2e_report_common.py`, replace case field requirement `eos_token_id` with:

```python
CHAT_IDENTITY_FIELDS = (
    "prompt_format",
    "messages_path",
    "messages_sha256",
    "rendered_prompt_sha256",
    "add_generation_prompt",
    "add_special_tokens",
    "chat_template_kwargs",
    "stop_token_ids",
)
```

- [ ] Validation rules:
  - `prompt_format == common.PROMPT_FORMAT`;
  - `messages_path` is a non-empty string ending in `.messages.json`;
  - `messages_sha256`, `rendered_prompt_sha256`, and `prompt_ids_sha256` are 64-char lowercase hex strings;
  - `add_generation_prompt is True`;
  - `add_special_tokens is False`;
  - `chat_template_kwargs == {"enable_thinking": False}`;
  - `stop_token_ids` is a non-empty list of non-negative ints.

- [ ] Add a transitional error message for old raw reports:

```python
if "eos_token_id" in case and "stop_token_ids" not in case:
    raise ValueError(f"{name}: old raw-text report schema uses eos_token_id; regenerate chat-template report")
```

- [ ] Update `tools/bench/compare_e2e_reports.py` identity fields so prompt format, message hash, rendered prompt hash, chat-template kwargs, and stop token ids are hard failures.

- [ ] Update tests in `tests/test_bench_report_tools.py`:
  - rename `test_eos_policy_change_is_hard_failure` to `test_stop_token_policy_change_is_hard_failure`;
  - add hard-failure tests for `prompt_format`, `messages_sha256`, `rendered_prompt_sha256`, and `chat_template_kwargs.enable_thinking`;
  - add old-schema rejection test.

### 8. Update Baseline Summary Generation

- [ ] In `tools/bench/make_baseline_summary.py`, include the new case identity fields in each summary case:

```python
{
    "fixture_set": case["fixture_set"],
    "fixture_manifest_path": case["fixture_manifest_path"],
    "fixture_manifest_sha256": case["fixture_manifest_sha256"],
    "prompt_format": case["prompt_format"],
    "messages_path": case["messages_path"],
    "messages_sha256": case["messages_sha256"],
    "rendered_prompt_sha256": case["rendered_prompt_sha256"],
    "prompt_ids_path": case["prompt_ids_path"],
    "prompt_ids_sha256": case["prompt_ids_sha256"],
    "prompt_tokens": case["prompt_tokens"],
    "requested_max_new_tokens": case["requested_max_new_tokens"],
    "add_generation_prompt": case["add_generation_prompt"],
    "add_special_tokens": case["add_special_tokens"],
    "chat_template_kwargs": case["chat_template_kwargs"],
    "stop_token_ids": case["stop_token_ids"],
}
```

- [ ] Add summary validator behavior:
  - reject raw-text reports lacking `prompt_format`;
  - reject `enable_thinking=true`;
  - reject missing or empty `stop_token_ids`.

- [ ] Update summary tests to assert old summaries with `eos_token_id` are rejected.

### 9. Update Decode Sidecars

- [ ] Update `tools/bench/decode_e2e_report.py` to decode generated token ids twice per repeat:
  - raw file with `skip_special_tokens=False`;
  - clean file with `skip_special_tokens=True`.

- [ ] Suggested output names:

```text
<report-stem>/<case>/repeat_<n>.raw.txt
<report-stem>/<case>/repeat_<n>.clean.txt
```

- [ ] Update decoded sidecar manifest to include:

```json
{
  "artifact_type": "qus_decoded_text_artifacts",
  "prompt_format": "qwen3.6-chat-template",
  "chat_template_kwargs": {"enable_thinking": false},
  "add_generation_prompt": true,
  "add_special_tokens": false,
  "stop_token_ids": [248046, 248044],
  "tokenizer": {
    "tokenizer_source": "local_hf",
    "tokenizer_model_id": "Qwen/Qwen3.6-27B",
    "tokenizer_path": "",
    "tokenizer_json_sha256": "",
    "tokenizer_config_sha256": "",
    "special_tokens_map_sha256": "",
    "chat_template_jinja_sha256": "",
    "generation_config_sha256": ""
  }
}
```

- [ ] Update tests to ensure both raw and clean outputs are written and that tokenizer metadata includes chat-template and generation-config hashes.

- [ ] Human smoke check after real run:
  - clean decoded output should be readable assistant answer content;
  - raw decoded output may contain special markers;
  - generated output should not primarily be a long thinking block because fixtures use `enable_thinking=false`.

### 10. Update Documentation

- [ ] Update `bench/README.md` prompt fixture section:
  - replace `.txt + .ids` with `.messages.json + .ids`;
  - document local tokenizer path flag;
  - document `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`;
  - document repeated `--stop-token-id`;
  - show official smoke and gate commands with both Qwen stop ids.

- [ ] Update `tools/bench/README.md` commands:

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --fixture-dir bench/fixtures/prompts
```

- [ ] Update `docs/bench/e2e-report-schema.md`:
  - replace `eos_token_id` with `stop_token_ids`;
  - add chat identity fields;
  - add old raw schema rejection note;
  - include exact JSON examples for case and decoded sidecar manifest.

- [ ] Update `docs/m2.8-pre-m3-standard.md`:
  - G3 must say canonical fixtures are `.messages.json + .ids + manifest`;
  - explain tokenizer/Jinja remains Python-side;
  - update manifest JSON examples;
  - update report schema examples;
  - update M3 gate commands to include `--stop-token-id 248046 --stop-token-id 248044`;
  - state old raw-text evidence is invalid.

- [ ] Update `docs/m2.8-pre-m3-standard.zh.md` with a real Chinese translation of the same changes, not English-heavy fragments.

- [ ] Update `docs/m3-readiness.md` after real benchmark regeneration. It must name the new report path, summary path, manifest SHA, and command.

- [ ] Update `docs/bench/baselines/README.md` to state baseline summaries are chat-template fixture evidence and must include prompt format plus stop token ids.

### 11. Delete Or Regenerate Invalid Baseline Artifacts

- [ ] Before final verification, remove old raw-text summaries from `docs/bench/baselines/` or replace them with regenerated chat-template summaries. Existing files with `eos_token_id` must not remain as current evidence.

- [ ] At minimum, delete or supersede:

```text
docs/bench/baselines/m2.8-p6-step-reset-smoke-summary.json
docs/bench/baselines/m2.8-p7-block-scoped-smoke-summary.json
docs/bench/baselines/m2.8-m3-gate-block-scoped-summary.json
```

- [ ] Preferred new committed summaries:

```text
docs/bench/baselines/m2.8-chat-template-smoke-summary.json
docs/bench/baselines/m2.8-chat-template-m3-gate-summary.json
```

- [ ] Summary JSON must not include local absolute tokenizer paths or profile-local paths that are unavailable from a checkout, except as redacted or documented provenance fields.

### 12. Regenerate Real Evidence

- [ ] Build the benchmark before real runs:

```bash
cmake --build build --target qus_e2e_bench test_e2e_bench_support
```

- [ ] Run a fast real-weight smoke using the new chat-template `cn_short`:

```bash
./build/bench/qus_e2e_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus \
  --output-json profiles/e2e/m2.8-chat-template-smoke.json \
  --case cn_short:bench/fixtures/prompts/cn_short.ids:8 \
  --warmup-repeats 0 \
  --repeats 1 \
  --max-ctx 8192 \
  --stop-token-id 248046 \
  --stop-token-id 248044
```

- [ ] Generate smoke summary:

```bash
python3 tools/bench/make_baseline_summary.py \
  profiles/e2e/m2.8-chat-template-smoke.json \
  --output docs/bench/baselines/m2.8-chat-template-smoke-summary.json
```

- [ ] Decode smoke output:

```bash
python3 tools/bench/decode_e2e_report.py \
  profiles/e2e/m2.8-chat-template-smoke.json \
  --tokenizer-path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --output-dir profiles/e2e/decoded/m2.8-chat-template-smoke
```

- [ ] Run M3 gate evidence using the new chat-template fixtures:

```bash
./build/bench/qus_e2e_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus \
  --output-json profiles/e2e/m2.8-chat-template-m3-gate.json \
  --case cn_short:bench/fixtures/prompts/cn_short.ids:128 \
  --case long_2k:bench/fixtures/prompts/long_2k.ids:1 \
  --warmup-repeats 1 \
  --repeats 3 \
  --max-ctx 8192 \
  --stop-token-id 248046 \
  --stop-token-id 248044
```

- [ ] Generate gate summary:

```bash
python3 tools/bench/make_baseline_summary.py \
  profiles/e2e/m2.8-chat-template-m3-gate.json \
  --output docs/bench/baselines/m2.8-chat-template-m3-gate-summary.json
```

- [ ] Decode gate output:

```bash
python3 tools/bench/decode_e2e_report.py \
  profiles/e2e/m2.8-chat-template-m3-gate.json \
  --tokenizer-path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --output-dir profiles/e2e/decoded/m2.8-chat-template-m3-gate
```

- [ ] If the real q5090 artifact path differs, discover the actual accepted artifact with:

```bash
ls -lh out/*.qus
```

Then update commands, report summaries, and readiness docs with the actual path used.

### 13. Verification Commands

- [ ] Python fixture and report tests:

```bash
python3 -m unittest tests.test_bench_tokenizer_tools tests.test_bench_report_tools
```

- [ ] Fixture check:

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --fixture-dir bench/fixtures/prompts \
  --check
```

- [ ] C++ support test:

```bash
cmake --build build --target test_e2e_bench_support
ctest --test-dir build --output-on-failure -R test_e2e_bench_support
```

- [ ] Full benchmark tool build:

```bash
cmake --build build --target qus_e2e_bench
```

- [ ] Report schema validation through summary generation:

```bash
python3 tools/bench/make_baseline_summary.py \
  profiles/e2e/m2.8-chat-template-smoke.json \
  --output /tmp/m2.8-chat-template-smoke-summary.json
```

- [ ] Old-schema absence checks:

```bash
rg -n '"eos_token_id"|txt_sha256|\\.txt"' bench/fixtures/prompts docs/bench/baselines docs/m3-readiness.md docs/bench/e2e-report-schema.md bench/README.md tools/bench/README.md
```

Expected result: no hits for committed fixture/report/readiness evidence. If docs intentionally mention old schema in an explicit rejection note, the hit must be reviewed and justified.

- [ ] Worktree audit:

```bash
git status --short
git diff --stat
```

### 14. Subagent Review Phase

- [ ] Before finalizing, run multiple review agents in parallel if tool support is available. Assign independent scopes:
  - Python fixture/rendering/provenance review;
  - C++ stop-token/report schema review;
  - report comparison/summary/readiness review;
  - docs and baseline artifact consistency review;
  - real-run decoded output and command reproducibility review.

- [ ] Review prompts must ask for blockers first, then major issues, then minor issues. Each review must cite exact files and lines.

- [ ] Fix all blockers and major issues before final response.

- [ ] Re-run affected verification after fixes.

## Executor Prompt

Use this prompt for the implementation agent:

```text
You are working in /home/neroued/qwen3.6-ultraspeed.

Implement docs/superpowers/plans/2026-06-28-qwen36-chat-template-e2e-inputs.md exactly. Use superpowers:subagent-driven-development or superpowers:executing-plans before editing. Keep C++ tokenizer-free. Python must render Qwen3.6 fixtures with tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False); C++ must consume .ids and use stop_token_ids [248046, 248044].

Important constraints:
- Do not preserve old raw-text baselines as current evidence.
- Do not regenerate q5090 weights for this plan.
- Use local tokenizer path /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16.
- Use current accepted q5090 artifact, normally out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus.
- Update code, tests, fixtures, docs, baseline summaries, and docs/m3-readiness.md together.
- After implementation, run the unit/C++/fixture checks and real smoke plus M3 gate runs from the plan.
- Use multiple subagents in the final review phase, covering Python fixtures, C++ bench/report, report tools, docs/baselines, and real-run output. Fix blockers and major issues before finalizing.

Commit the completed implementation with a concise message after verification.
```

## Self-Review Checklist

- [ ] Every approved design decision has an implementation task.
- [ ] The plan never asks C++ to tokenize, render Jinja, or call Python.
- [ ] The plan replaces `.txt` required fixtures with `.messages.json`.
- [ ] The plan makes `enable_thinking=false` observable in manifest, reports, summaries, decoded sidecars, and docs.
- [ ] The plan replaces `eos_token_id` with `stop_token_ids` across schema, comparison, summaries, tests, and docs.
- [ ] The plan invalidates old raw-text baselines and requires regenerated evidence.
- [ ] The plan includes both smoke and M3 gate real model runs.
- [ ] The plan includes a final multi-scope review phase.
- [ ] No open questions remain.
