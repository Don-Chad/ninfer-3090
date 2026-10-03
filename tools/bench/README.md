# tools/bench

Maintainer orchestration for the public `ninfer_bench` throughput tool, serving corpus/concurrency
runners, and the external Serve TTFT client. Correctness is owned by the affected suites under
[`tests/`](../../tests/README.md).

## Interleaved A/B between two builds

`run_interleaved_ab.py` compares two executables by alternating them **inside** each repetition.
Use it for any before/after claim on this hardware. The card drifts 3-5% between processes as it
heats and the power cap clamps, which is larger than most effects worth measuring here, so
"all of arm A, then all of arm B" cannot separate a real change from the ramp -- two comparisons
in TODO's history came out with opposite-signed drift on code paths that had not changed.

```bash
python tools/bench/run_interleaved_ab.py \
    --arm-a build-ninja/apps/ninfer.exe \
    --arm-b build-ninja/apps/ninfer-newtile.exe \
    --reps 6 \
    --config 'k6:{exe} models/qwen3_8_27b_dflash2.ninfer --prompt "..." --max-new 256 --greedy --spec dflash2 --draft-tokens 6' \
    --control 'k4:{exe} models/qwen3_8_27b_dflash2.ninfer --prompt "..." --max-new 256 --greedy --spec dflash2 --draft-tokens 4'
```

It reports the **paired median** -- the median of per-repetition ratios -- and how many pairs came
out positive. That is deliberately not the ratio of the two medians: on this box the two statistics
disagree by more than the effects being measured.

`--control` names a configuration whose code path is **identical in both arms**, so any ratio it
shows is pure drift; the tool divides it out per repetition and prints a `normalised` column. In the
GDN-tile comparison that control is what made the result readable, because draft counts 4 and 5
stay on an unchanged route.

Two things it refuses rather than lets you get wrong:

- **both arms must sit in the same directory**, in practice `build-ninja/apps/` under different
  names. An executable copied elsewhere fails DLL resolution on Windows, exits 127 and writes an
  empty log, which reads exactly like a model-loading failure.
- a `--config` with no `{exe}` placeholder, which would run the same binary twice.

`--metric-regex` selects what to compare; it defaults to the CLI's `decode speed ... tok/s` summary
line and needs changing for `ninfer_bench`, which prints CSV. A run where any sample produced no
number exits 1, because a comparison with a hole in it is not a comparison.

## External Serve TTFT

[`ttft/README.md`](ttft/README.md) defines the black-box latency benchmark. The measurement runner
uses frozen text/media requests and public streaming protocols without calling Engine. A separate
controller manages the fixed Qwen3.8-27B NVFP4/FP8 Serve profiles and fresh-process isolation.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

The controller chooses the profile, starts and stops Serve for every sample, runs the external
client, stages the NVFP4 artifact once in `/dev/shm`, stores raw/progress/Serve artifacts below
`profiles/bench/ttft/`, records structured per-request Serve diagnostics, and writes Markdown,
JSON, and CSV summaries. The case catalog, exact profiles, TTFT boundary, and fixture qualification
are documented in the dedicated README.

## Agent rotation (context-cache retention)

`run_agent_rotation.py` drives an already-running `ninfer-serve` with N independent agent
conversations visited round-robin, each growing by one long user turn per visit until it reaches
`--target-tokens`. With more agents than `--max-concurrency`, most visits must restore their
conversation from the Host tier or re-prefill it, and every `--edit-every`th visit rewrites the user
turn two back, which only a long anchor below the edit can serve. The client reports TTFT from the
stream and the weighted cache hit rate from the usage `cached_tokens`, split by first, append and
edit visits; run the server with `--request-log-jsonl` for the reuse path and transfers.

```bash
python3 tools/bench/run_agent_rotation.py --agents 4 --target-tokens 200000 \
    --out profiles/bench/agent-rotation/run.jsonl
```

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v15. The flattened summary and schema-v4 matrix
manifest carry native facts from the report: architecture, public name, actual formats, prefill signature, artifact,
load/read/upload/staging values, Engine memory arenas including the non-additive Vision layout
inside the unified workspace and CUDA Graph allowance, per-test planned logical and
allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

## Serving corpus benchmark

[Published coverage and model results](../../docs/performance.md) identify the recorded runs.
The [serving methodology](../../docs/performance/methodology.md) owns workload definitions,
metric boundaries, aggregation, comparison rules, and publication format. This section describes
runner usage and output files.

`run_serve_corpus.py` accepts explicit `--artifact LABEL=PATH` entries. Labels identify report groups;
the selected artifact supplies the architecture, public name and weight bindings.
Omitting `--mode` selects MTP0 and MTP3; repeat `--mode` to select a subset. Use `dflash7` for
Qwen3.6-35B-A3B DFlash K=7 and `dflash2_7` for Qwen3.8-27B DFlash2 K=7, with companion weights
in the selected artifact. `--sampling greedy` selects exact argmax; the default is stochastic.
Run commands with a selected Python 3.11 interpreter, as in the model-page reproduction entries.

The serial runner writes `run.jsonl`, `summary.csv`, `summary.md`, and per-server logs under
`server/`. JSONL contains the completed requests and responses; CSV/Markdown contain fixture and
category summaries. The output directory is supplied explicitly with `--output`.

Its schema-v7 result and flattened summaries retain the actual `prefill_signature`, request Host
exposure, and decode Host/Device-wait time per round received from the schema-v22 serving records.
Request exposure is a latency distribution value and is never summed across concurrent requests;
worker aggregation uses the serving `throughput.host_work` interval deltas. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Chat decode A/B (`run_chat_decode.py`)

Decode rate on a chat prompt set through `ninfer-serve`'s OpenAI Chat Completions endpoint,
reported as `concurrency x 1000 / mean TPOT` from the server's own request log -- the metric
`vllm bench serve` prints, so results can be read against stacks that publish it. `output
throughput` (generated tokens / wall clock) is printed alongside and includes prefill and the tail.

Arms are interleaved, in order on even passes and reversed on odd, because between-process spread
on one RTX 3090 is 3-5%; always A/B two binaries inside one sitting rather than across sessions.
Each arm may drop or add server flags, so one binary can be compared against itself under different
options, and may name its own artifact (`;model=PATH`), so two artifacts can be compared on one
binary with the same prompts. `--spec` selects `mtp` (the default), `dflash2` or `none` and
`--draft-tokens` the draft count; the `RESULT` line carries the decode rate, the draft acceptance
and the tokens emitted per round, so one run answers both "faster?" and "does it still draft as
well?". `--model-id` is the id the server publishes, which is the artifact's public name
(`qwen3.8-27b` by default, `qwen3.6-35b-a3b` for the 35B-A3B).

```bash
# two binaries, one artifact
python3 tools/bench/run_chat_decode.py \
  --model models/qwen3_8_27b.ninfer --prompts prompts_real.jsonl \
  --concurrency 1 --reps 2 --out profiles/bench/chat-decode \
  --arm base=/path/to/baseline/ninfer-serve --arm new=./build/apps/ninfer-serve

# one binary, two artifacts, DFlash2, drafting acceptance included
python3 tools/bench/run_chat_decode.py --prompts bench/fixtures/chat_prompts.jsonl \
  --spec dflash2 --draft-tokens 7 --kv-dtype rk4v4 --max-tokens 512 --out profiles/bench/chat-decode \
  --arm old=./build/apps/ninfer-serve\;model=old.ninfer --arm new=./build/apps/ninfer-serve\;model=new.ninfer
```

The prompt file is one JSON object per line with a `prompt` string. The comparisons in
[docs/performance.md](../../docs/performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode)
use the eight thinking-off prompts of syv-ai/qwen38-27b-rtx3090's `bench/prompts_real.jsonl`, which
is that project's file and is not vendored here. `bench/fixtures/chat_prompts.jsonl` is this
repository's own set of 24 mixed prompts (code in eight languages, prose, maths, translation and
Chinese); the artifact comparisons of 2026-10 use it with greedy decoding and thinking off.

## Concurrent serving benchmark

`run_serve_concurrency.py` selects `--suite decode-saturation` or `--suite corpus-makespan`.
Their distinct time boundaries and workload dispatch are defined in the
[serving methodology](../../docs/performance/methodology.md#workloads-and-measurement-boundaries).
Repeat `--concurrency` to select C points; each point starts a fresh server. The point report
records the actual Engine configuration, automatic KV capacity, shuffle seed where applicable,
dispatch method, and per-request positions.

Schema-v3 outputs include `points/*.json`, `server/*.jsonl`, and combined `summary.json`, `summary.csv`, and
`summary.md`. Corpus runs also write complete responses in `corpus/<point>/results.jsonl` and
per-request phase summaries in that directory; older campaigns may have only point reports and
server logs. Historical model pages identify the report directory associated with each table.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.
