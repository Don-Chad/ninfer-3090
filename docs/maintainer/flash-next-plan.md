# Qwen3.8-Flash-Next on the RTX 3090: working plan

Temporary. This file tracks the staged work to run Qwen3.8-Flash-Next (GGUF architecture
`qwen4exp`) from the ISTA DASLab GSQ-RCO IQ2_XS GGUF. Delete it when the last milestone lands and
its stable facts have moved to their references (a `qwen4-exp-model.md`, the format and layout
references, `engine-architecture.md`).

## Status

| PR | Scope | Status |
|---|---|---|
| 0 | Strata baselines: decode, MTP, prefill, teacher-forced log-probs on frozen token ids | not started |
| 1 | GGUF reader, nine exact GGML block formats, `ggml_blocks_v1` / `ggml_rows_page4k_v1`, `qwen4_exp` name map, tokenizer synthesis, `--subset dev` | done (Python only) |
| 2 | C++ registration and materialization of the GGML formats and layouts; host exact decoder | next |
| 3-9 | Dense GGML linears, hyper-connections and FP32 head, QSA, PLE frontend and stream residency, model skeleton, MoE on GPU, MoE on CPU | not started |
| 10 | Whole model at 32K, quality gate against Strata | not started |
| 11-14 | Residency policy and miss split, 128K, MTP, vision | not started |

Until PR 2 the C++ reader rejects every artifact that uses a `ggml_*` format.

## Artifact decisions taken in PR 1

- Logical parameters mirror the GGUF tensors one to one, named under `text/`; shapes are the
  reversed GGUF dimensions, so every object holds the GGUF bytes unchanged. The exceptions: routed
  experts are split into per-expert `moe/experts/E/{gate,up,down}` parameters, and
  `ffn_gate_inp_shexp` becomes `moe/shared_score` of shape `[1, 2560]`. The map is the table in
  [`tools/convert/qwen4_exp.py`](../../tools/convert/qwen4_exp.py); unknown, missing or misplaced
  GGUF tensors are errors.
- Each layer's routed experts are two objects: `gate|up` of shape `[512 * 1280, 2560]` with expert
  `e`'s 640 gate rows then its 640 up rows, and `down` of shape `[512 * 2560, 640]`. These are the
  `[512, 1280, 2560]` and `[512, 2560, 640]` banks with identical bytes; the 2-D form is what the
  existing packing groups and per-expert bindings produce, as for Qwen3.5 MoE.
- The n-gram embedding table (`text/ple/table`) uses `ggml_rows_page4k_v1`: 7,111,146 pages,
  29,127,254,016 bytes. It is written in 16 MiB chunks of whole pages.
- GGML-format projections record `AllowA8` (ggml's Q8_1 activation path); BF16 projections keep
  `A16Only`.
- The development subset is a builder option (`--subset dev`), not a second recipe, because a
  recipe cannot drop parameters or shorten the n-gram table. It is 2,930,421,760 bytes, not the
  ~1.2 GB first estimated: the token embedding and output head alone are 675 MB and the three
  expert banks 2.04 GB.

## Export conventions of this GGUF

Settled by evidence in PR 1 (`tests/convert/test_qwen4_exp_real.py` reproduces the data checks):

| Convention | Finding | Evidence |
|---|---|---|
| Unit-offset norms | Every F32 norm except `ssm_norm` stores `1 + gamma` of a BF16 `gamma`: `v - 1` is an exact BF16 word for 100% of values in all ten families, while only 8-67% of the stored `v` are (the BF16 checkpoint plus 1.0 in FP32). All 4,608 `ssm_norm` values are exact BF16 words, so it is the checkpoint's own multiplier. The model multiplies by the stored vector; nothing is subtracted. | data check; llama.cpp `conversion/qwen.py` (+1 on `norm.weight` except `linear_attn.norm`) and `conversion/qwen4exp.py` (+1 on PLE and indexer norms); Strata `gr.hpp`, `ple.hpp` and `gdn.hpp` apply `rms(x) * w` |
| `ssm_a` | `-exp(A_log)`; all 1,728 values are negative | data check; llama.cpp `qwen.py` |
| PLE hash constants | The literal multipliers equal splitmix64 with seed 1234, PLE index 0 and the 248,320-row vocabulary; the head sizes are the 16 consecutive primes from 20,000,000; the table rows are their sum rounded up to 128 | data check against the Infernix derivation |
| `attention/query_gate` | per head, 256 query rows then 256 gate rows | llama.cpp graph, Strata `layer.cpp`, Infernix mapping |

Still inferred, to be confirmed by the PR 7 FP64 block reference:

- **GDN value-head order is tiled** (value head `h` pairs with key head `h % 16`) in `gdn/qkv`'s
  value rows, `gdn/z`, `gdn/a_projection`, `gdn/b_projection`, `gdn/a`, `gdn/dt_bias`,
  `gdn/convolution` and the K axis of `gdn/output`. The basis is provenance: the file's names and
  conventions match llama.cpp's `Qwen4ExpTextModel`, which inherits the grouped-to-tiled reorder,
  and Strata, which runs this file, pairs heads by modulo with no runtime permutation. No HF
  checkpoint is local to compare against.
- **Decision:** the converter keeps the tiled order. `gdn/output`'s K axis cannot be permuted
  exactly in a block format, so the qwen4exp GDN path adopts the tiled order throughout rather than
  the converter undoing it for the other tensors only.

## Tokenizer

The synthesized `tokenizer.json` equals the Qwen3.8-27B artifact's in vocabulary (248,044 entries),
merges (247,587), pre-tokenizer, normalizer, decoder and the ids and contents of all 33 added
tokens. The `special` flag differs on six tokens (`<|fim_prefix|>`, `<|fim_middle|>`,
`<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`): Hugging Face has them non-special,
and llama.cpp's converter stores every `<|...|>` added token as a control token, so the GGUF cannot
say. `--resource tokenizer.json=PATH` keeps the Hugging Face flags after checking the file against
the GGUF. Whether Flash-Next's own Hugging Face tokenizer equals Qwen3.8-27B's is assumed, not
checked. The GGUF's chat template is used as is; it differs from the NInfer Qwen3.8-27B template.

## PR 1 measurements

Host: Ryzen 9 5950X, Samsung 980 PRO (C:), Python 3.12.7, numpy 2.5.3, torch 2.11 (CPU).

| Workload | Result |
|---|---|
| `--subset dev` conversion, GGUF on C: to C: | 2.93 GB in 20 s on the first run, 11 s with the GGUF ranges cached; peak working set 0.69 GiB (0.48 GiB of it is the interpreter with torch loaded) |
| Full n-gram table paging, output discarded | 28.80 GB read in 43 s (0.67 GB/s, CPU-bound in Python); peak working set 0.57 GiB |
| GGUF header parse (73 keys, 248,320-token vocabulary, 1,224 tensors) | 0.7 s |

The full-model conversion has not been run; it needs ~68.4 GB free on C: (see the spec's disk plan).
