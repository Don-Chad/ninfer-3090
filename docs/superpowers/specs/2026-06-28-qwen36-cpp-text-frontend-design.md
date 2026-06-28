# Qwen3.6 C++ Text Frontend Design

Date: 2026-06-28

## Summary

Add a project-owned C++ text frontend for Qwen3.6-27B so the primary `qus` binary accepts
normal text input and prints normal text output.

The runtime `Engine` remains a token-id execution engine. A new `qus::text` layer sits above it:

```text
prompt/messages -> Qwen chat template -> Qwen byte-level BPE ids
    -> Engine prefill/decode -> generated ids -> detokenized text
```

The main `qus` CLI is replaced with a text-first interface. Token-id input/output remains available
only in benchmark and parity/debug tools where ids are the useful contract.

## Context

The current primary binary is explicitly token-id oriented:

- `src/main.cpp` accepts positional nonnegative integer token ids.
- `Engine::prefill`, `Engine::decode_step`, and `Engine::generate` accept and return ids.
- `src/CMakeLists.txt` describes `qus` as "token-ids in -> token-ids out".

The M2.8 benchmark pipeline intentionally keeps `qus_e2e_bench` tokenizer-free. It reads committed
`.ids` fixtures and report tooling decodes those ids through Python/HF sidecars. That remains the
right contract for performance gates and token parity. It is not a good primary user workflow.

This design changes the user-facing `qus` binary, not the core model runtime contract.

## Research Evidence

### Local Qwen3.6 Tokenizer Files

The local tokenizer directory used for research was:

```text
/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
```

Observed files:

- `tokenizer.json`
- `vocab.json`
- `merges.txt`
- `added_tokens.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `generation_config.json`
- `chat_template.jinja`

Observed tokenizer properties from `tokenizer.json`:

- tokenizer class from HF: `Qwen2Tokenizer`
- model type: `BPE`
- base vocab entries: `248044`
- merge entries: `247587`
- normalizer: `NFC`
- pre-tokenizer: `Split` with the Qwen regex, then `ByteLevel`
- byte-level decoder and post-processor
- added token ids from `248044` through `248076`

Important added tokens:

| Token | Id | Special |
|---|---:|---|
| `<|endoftext|>` | 248044 | yes |
| `<|im_start|>` | 248045 | yes |
| `<|im_end|>` | 248046 | yes |
| `<tool_call>` | 248058 | no |
| `<tool_response>` | 248066 | no |
| `<think>` | 248068 | no |
| `</think>` | 248069 | no |

`generation_config.json` defines:

```json
{
  "bos_token_id": 248044,
  "eos_token_id": [248046, 248044],
  "pad_token_id": 248044,
  "do_sample": true,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95
}
```

The C++ text frontend uses the EOS list as stop token ids but keeps the current runtime's greedy
generation policy. Sampling remains out of scope.

### Chat Template Behavior

The Qwen3.6 chat template is Jinja and supports system/developer merging, text content, multimodal
placeholders, tool calls, assistant reasoning content, and `enable_thinking`.

The first C++ frontend implements only the text-only subset:

- `system`
- `user`
- `assistant` with plain string `content`

For a single user prompt with `add_generation_prompt=true` and `enable_thinking=false`, HF renders:

```text
<|im_start|>user
...user text...<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The non-thinking marker sequence is part of the prompt and must be tokenized as normal prompt text.

### vLLM Pattern

vLLM keeps chat rendering and tokenization at the serving/renderer boundary. It calls
`tokenizer.apply_chat_template(...)`, forces `return_dict=false` for flat token ids, then passes ids
to the engine-facing prompt path.

This validates the architectural boundary: text belongs in the frontend, ids belong in the model
engine.

### llama.cpp Pattern

llama.cpp exposes C/C++ tokenize, detokenize, token-to-piece, and chat-template APIs. Its tokenizer
code has Qwen-specific pre-tokenization paths instead of relying on `std::regex` for Unicode property
matching.

The relevant lesson is implementation-level: Qwen byte-level BPE needs explicit Unicode handling,
added-token partitioning, BPE merge ranking, and byte-level decode. Treating this as a simple string
splitter would be incorrect.

## User-Approved Decisions

### D1. Main CLI Becomes Text-First

`qus` is replaced with a text CLI:

```text
qus <weights.qus> --tokenizer <hf-tokenizer-dir> --prompt "..." [options]
qus <weights.qus> --tokenizer <hf-tokenizer-dir> --messages <messages.json> [options]
```

The old positional token-id input is removed from the main CLI. No compatibility mode, alias, or
deprecation path is kept.

### D2. Keep Engine Token-Id API

`Engine` remains token-id based:

- `prefill(span<const int>)`
- `decode_step()`
- `generate(span<const int>, int)`

The new text frontend wraps `Engine`; it does not push tokenizer responsibilities into L0, L1, or L2.

### D3. Keep Benchmark and Parity Tools Token-Id Based

`bench/` and `tools/parity/` continue to consume ids. They are engineering tools, not the primary
human-facing workflow.

`qus_e2e_bench` must remain tokenizer-free because M2.8/M3 performance evidence depends on fixed
prompt ids and stable report identity.

### D4. Support `--prompt` and `--messages`

The first text CLI supports two mutually exclusive input forms:

- `--prompt <text>`: wraps the string as one `{ "role": "user", "content": text }` message.
- `--messages <path>`: reads a JSON array of text-only chat messages.

`--messages` is the canonical chat input. `--prompt` is the common convenience path.

### D5. Text-Only Chat Scope

The first C++ renderer supports:

- optional leading `system` message
- `user` messages
- `assistant` messages with plain string content

It rejects:

- `tool`
- `developer`
- `tool_calls`
- `reasoning_content`
- multimodal content arrays or objects
- message content that is not a string

This is intentionally narrower than the full Qwen3.6 Jinja template. The project is a text decoder
runtime, not a general chat-completions server.

### D6. Fixed Chat Generation Prompt Policy

The text frontend renders messages with:

```text
add_generation_prompt = true
enable_thinking = false
```

There is no first-version CLI flag for thinking mode. The goal is direct assistant answers, matching
the existing M2.8 fixture policy.

### D7. Stop Token Policy

Default stop token ids come from Qwen3.6 generation config:

```text
248046  # <|im_end|>
248044  # <|endoftext|>
```

The CLI may accept repeated `--stop-token-id <id>` to override the default for debugging, but the
documented normal path uses the Qwen defaults loaded from the tokenizer directory. If the
generation config is missing or malformed, the frontend may fall back to these fixed Qwen3.6 ids with
a clear diagnostic.

### D8. Lightweight Dependency Policy

External dependencies are allowed, but must stay small.

The recommended dependency is vendored `utf8proc`:

- use it for NFC normalization;
- use it for Unicode codepoint category checks in the Qwen pre-tokenizer;
- pin the vendored source and license in `third_party/`;
- avoid system dependency drift.

Do not make ICU the default dependency. Do not depend on Boost. PCRE2 is not needed if the Qwen
pre-tokenizer is implemented as a state machine.

## Architecture

Add a new host-only layer:

```text
include/qus/text/
src/text/
```

This layer is above runtime and independent of CUDA kernels.

### Components

#### `QwenTokenizer`

Responsibilities:

- load tokenizer files from an HF tokenizer directory;
- validate this is the supported Qwen3.6 tokenizer shape;
- expose `encode(std::string_view text, EncodeOptions)` returning `std::vector<int>`;
- expose `decode(span<const int> ids, DecodeOptions)` returning UTF-8 text;
- expose token metadata for stop ids and special-token filtering.

Supported tokenizer features:

- NFC normalization;
- added-token partitioning before ordinary BPE;
- Qwen3.6/Qwen2 byte-level pre-tokenizer;
- ByteLevel byte-to-unicode encoding;
- BPE merge ranking from `merges.txt` or `tokenizer.json`;
- fallback to byte tokens when merged token text is absent;
- byte-level decode back to UTF-8;
- optional skip of tokens marked `special=true`.

Rejected tokenizer features:

- non-BPE model types;
- non-Qwen pre-tokenizer shapes;
- tokenizer files with vocab size inconsistent with `model::kCfg.vocab`;
- network downloads;
- arbitrary tokenizer classes.

#### `QwenChatTemplate`

Responsibilities:

- validate text-only messages;
- render the supported Qwen chat subset;
- match HF output for the supported cases.

Supported rendering rules:

```text
system:
  <|im_start|>system
  {trimmed content}<|im_end|>

user:
  <|im_start|>user
  {trimmed content}<|im_end|>

assistant:
  <|im_start|>assistant
  {trimmed content}<|im_end|>

generation prompt with enable_thinking=false:
  <|im_start|>assistant
  <think>

  </think>

```

The exact newline behavior must be checked against HF golden outputs. The renderer should be direct
C++ string construction, not a general Jinja interpreter.

#### `TextGenerationRunner`

Responsibilities:

- own or reference `QwenTokenizer`, `QwenChatTemplate`, and `Engine`;
- resolve prompt/messages input into prompt ids;
- enforce max-context requirements before model execution;
- call `Engine::generate`;
- strip a final stop token before clean text output;
- return both generated ids and generated text for optional diagnostics.

#### CLI Driver

The main `src/main.cpp` becomes text-first.

Initial CLI:

```text
usage:
  qus <weights.qus> --tokenizer <dir> --prompt <text> [options]
  qus <weights.qus> --tokenizer <dir> --messages <messages.json> [options]

options:
  --max-context <tokens>      default: 2048 or documented runtime default
  --max-new <tokens>          default: 128
  --device <id>               default: 0
  --raw-output                keep special tokens in decoded output
  --print-token-ids           print generated token ids to stderr for debugging
  --stop-token-id <id>        repeatable override for Qwen defaults
```

`--prompt` and `--messages` are mutually exclusive. One is required.

Text output goes to stdout. Metrics and diagnostics go to stderr.

## Data Flow

### `--prompt`

1. Read prompt string from CLI.
2. Create messages array with one `user` message.
3. Render Qwen chat prompt with non-thinking assistant generation prefix.
4. Tokenize rendered prompt with added-token parsing enabled.
5. Load weights and run `Engine::generate`.
6. Stop when a stop token is generated or `max_new` is reached.
7. Drop the terminal stop token for clean output.
8. Detokenize generated ids and print text.

### `--messages`

1. Read JSON message array.
2. Validate roles and string content.
3. Render using the same Qwen text-only subset.
4. Continue through the same tokenize/generate/detokenize path.

## Tokenizer Details

### Added-Token Partitioning

The tokenizer must detect added tokens before normal BPE tokenization. It must support Qwen3.6
metadata:

- `single_word`
- `lstrip`
- `rstrip`
- `normalized`
- `special`

For the initial scope, correctness matters most for exact chat markers and thinking markers:

- `<|im_start|>`
- `<|im_end|>`
- `<think>`
- `</think>`
- `<|endoftext|>`

The parser should still load all added tokens so detokenization and future fixtures do not silently
lose known symbols.

### Unicode Pre-Tokenization

Do not use `std::regex` for the Qwen regex. C++ standard regex does not provide the needed Unicode
property semantics.

Implement a Qwen3.6 pre-tokenizer state machine equivalent to:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)
| [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
| \p{N}
| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
| \s*[\r\n]+
| \s+(?!\S)
| \s+
```

`utf8proc` supplies codepoint iteration, NFC normalization, and Unicode category checks. The
project supplies the split logic.

### Byte-Level Encoding

After pre-tokenization, each split word is transformed through the GPT/Qwen byte-level mapping.
The resulting byte-level string is split into initial symbols, then BPE merge ranks are applied until
no ranked pair remains.

The implementation should prioritize clarity over micro-optimization. Tokenization runs once per
request; model decode speed remains the project headline metric.

### Detokenization

Detokenization concatenates token pieces, applies byte-level inverse mapping, and returns UTF-8.

Modes:

- raw: preserve all tokens, including special tokens;
- clean: skip tokens marked `special=true` and remove a terminal stop token before decoding.

Main CLI default is clean.

## Error Handling

The text frontend fails before model execution when:

- tokenizer directory is missing;
- required tokenizer files are missing;
- tokenizer model type is not BPE;
- tokenizer vocab plus added tokens cannot cover `model::kCfg.vocab`;
- generation config EOS field is malformed and no fixed Qwen fallback is allowed;
- messages JSON is not an array;
- messages contain unsupported roles or non-string content;
- both `--prompt` and `--messages` are passed;
- neither input form is passed;
- prompt token count exceeds `max_context`;
- `max_context` cannot fit prompt plus requested decode positions.

Error messages should name the bad file, option, or message index.

## Testing Strategy

Tests are allowed because tokenizer, chat-template, CLI, and file-format behavior are real
user-visible contracts.

### Golden Tokenizer Tests

Generate committed small golden fixtures from local HF tokenizer tooling. Cases must cover:

- ASCII English;
- Chinese;
- punctuation;
- leading/trailing spaces;
- multiple spaces;
- newlines;
- emoji and non-BMP codepoints;
- combining marks that exercise NFC;
- exact chat marker strings;
- `<think>` and `</think>`;
- representative code text;
- `--prompt` chat rendering;
- text-only `--messages` rendering with system/user/assistant.

The C++ tokenizer test checks:

- `encode(text) == expected_ids`;
- `decode(expected_ids, raw) == expected_raw_text`;
- `decode(expected_ids, clean)` skips special tokens where expected;
- invalid tokenizer files are rejected with clear errors.

### CLI Contract Tests

Add focused tests for argument parsing and messages validation. These should validate behavior, not
source layout.

Useful checks:

- `--prompt` and `--messages` are mutually exclusive;
- missing `--tokenizer` fails;
- unsupported message role fails;
- unsupported multimodal content fails;
- default stop ids are loaded or resolved;
- `--print-token-ids` sends ids to stderr, not stdout.

### Integration Verification

Implementation closeout should run the smallest meaningful set:

- build `qus`;
- build and run C++ tokenizer/chat tests;
- run existing benchmark/report tests that should remain tokenizer-free;
- run a short real-weight `qus --prompt ...` smoke if local q5090 weights are available;
- compare C++ rendered prompt ids against Python HF for the committed fixture messages.

Do not add source-scanning tests. Do not add compatibility tests for the removed token-id main CLI.

## Documentation Updates Required During Implementation

Implementation must update active documentation that currently says the main runtime is
token-id-only:

- `README.md`
- `docs/design.md`
- `src/CMakeLists.txt` driver comment
- any user-facing CLI docs that show positional token-id input

M2.8 benchmark docs should remain explicit that `qus_e2e_bench` is tokenizer-free.

## Non-Goals

- General-purpose tokenizer framework.
- Generic Jinja interpreter.
- Full OpenAI-compatible chat-completions server.
- Tool calling.
- Multimodal inputs.
- Developer-role semantics.
- Preserving assistant reasoning content.
- Thinking-mode CLI switch.
- Sampling, temperature, top-k, top-p, RNG.
- Streaming output.
- Changing `Engine` to consume strings.
- Changing `qus_e2e_bench` to consume strings.
- Backward-compatible main CLI token-id mode.

## Risks

### Unicode Split Drift

The largest correctness risk is pre-tokenization drift from HF tokenizers. The mitigation is a
behavioral golden suite from HF outputs and direct implementation of the Qwen state machine.

### Added Token Semantics

Special and non-special added tokens share the same id range. The implementation must use the
tokenizer metadata rather than assuming all added tokens are skipped in clean output.

### Chat Template Subset Drift

The C++ renderer intentionally supports a subset of the full Jinja template. The mitigation is to
reject unsupported message shapes loudly and compare supported shapes against HF-rendered golden ids.

### Dependency Creep

The dependency policy allows `utf8proc` only for Unicode primitives. Do not introduce ICU, Boost, or
a general regex engine unless a later design explicitly changes this decision.

## Approval State

Approved by user:

- main `qus` becomes text-first;
- `--prompt` and `--messages` are both in first scope;
- token-id path remains in benchmark and parity/debug tools;
- lightweight external dependency is allowed;
- prefer vendored `utf8proc`, avoid heavy dependencies.

Open questions: none.
