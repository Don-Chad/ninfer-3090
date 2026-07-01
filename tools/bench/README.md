# tools/bench

Offline helper for the `qus_bench` throughput tool. Correctness/parity tooling lives separately
under [`tools/parity`](../parity).

## Corpus baker

`qus_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and long
enough for the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus offline
with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, ids sha256
```

The corpus is curated mixed-domain prose (Chinese / English / code / math) encoded WITHOUT the
chat template and WITHOUT special tokens, then repeated to reach `--min-tokens`. Encoding is
deterministic for a fixed text + tokenizer, so `--check` can verify the committed artifact.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `QUS_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus (default target ~9216 tokens, covers prefill up to max_ctx).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --min-tokens 9216

# Verify the committed .ids + manifest match a fresh bake (nonzero exit on mismatch).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --min-tokens 9216 --check
```

To benchmark prefill lengths beyond the current corpus, re-bake with a larger `--min-tokens`.
