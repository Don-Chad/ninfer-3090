# Bench Tools

These tools support the M2.8 pre-M3 benchmark standard.

## Tokenizer Policy

Tokenizer-dependent commands use a local Hugging Face tokenizer directory. Resolution order:

1. `--tokenizer-path`
2. `QUS_TOKENIZER_PATH`
3. fail

The tools use `local_files_only=True` and must not download from the network.

## Regenerate Fixtures

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --fixture-dir bench/fixtures/prompts
```

Check committed fixtures:

```bash
python3 tools/bench/tokenize_prompts.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --fixture-dir bench/fixtures/prompts \
  --check
```

## Decode E2E Reports

```bash
python3 tools/bench/decode_e2e_report.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --report profiles/e2e/example.json
```

Decoded text is human-smoke-only. Correctness gates use token ids and report comparison.
