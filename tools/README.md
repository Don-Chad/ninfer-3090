# NInfer tools

This directory contains offline artifact tooling, the independent Python reference, numerical
diagnostics, benchmark helpers, and the serving smoke client. Run commands from the repository
root. The normal C++ products live under `apps/` and `bench/`; most Python tools here are invoked
manually with `python -m`.

The registered targets are `qwen3_6_27b` and `qwen3_6_35b_a3b`. Target-specific
tools use those exact keys in their directory names. Shared artifact mechanisms and
checkpoint-reading helpers stay outside a target directory.

## Start by task

| Task | Start here |
|---|---|
| Build the 27B `.ninfer` artifact | [`convert/qwen3_6_27b/`](convert/qwen3_6_27b/) |
| Build the 35B-A3B `.ninfer` artifact | [`convert/qwen3_6_35b_a3b/`](convert/qwen3_6_35b_a3b/) |
| Inspect an artifact directory | [`artifact/inspect.py`](artifact/inspect.py) |
| Verify an artifact against the BF16 checkpoint | [`convert/qwen3_6_27b/verify.py`](convert/qwen3_6_27b/verify.py) |
| Run independent Python inference | [`reference/qwen3_6_27b/README.md`](reference/qwen3_6_27b/README.md) or [`reference/qwen3_6_35b_a3b/README.md`](reference/qwen3_6_35b_a3b/README.md) |
| Compare Python, C++, and source-model activations | [`parity/qwen3_6_27b/README.md`](parity/qwen3_6_27b/README.md) |
| Run the end-to-end performance matrix | [`bench/README.md`](bench/README.md) |
| Exercise a resident OpenAI/Anthropic server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |

## Artifact workflow

Create, inspect, and verify the artifact:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.ninfer

/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects

/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.convert.qwen3_6_27b.verify \
  out/qwen3_6_27b.ninfer \
  --model /path/to/Qwen3.6-27B/base-hf-bf16
```

`artifact/` owns the generic Python `.ninfer` codec and registered numeric/layout formats.
`convert/common/` owns architecture-independent safetensors and quantization helpers.
`convert/qwen3_6/common/` owns narrow family-invariant recipe, shortlist, Vision, resource-name,
and writer mechanics. Each target converter still owns its exact config, complete inventory,
source mapping, draft ranking policy, and verification. Targets never import sibling targets.

The 35B-A3B converter deliberately uses the measured 27B ranking because the two checkpoints have
the same semantic token-id vocabulary:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.convert.qwen3_6_35b_a3b.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16 \
  --out out/qwen3_6_35b_a3b.ninfer
```

The ranking path is fixed by the exact target converter. The sidecar records that its source
measurement target was `qwen3_6_27b`; the selected rows themselves are always gathered from
the 35B BF16 output head before Q4 quantization.

## Python reference and parity

Run artifact-native Text, Vision, and MTP inference:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode briefly." \
  --decode 128 --mtp-draft-tokens 3
```

The reference is an independent diagnostic implementation, not the C++ runtime wrapped in Python
and not an exact generated-token golden for it.
Install its dependencies from
[`reference/qwen3_6_27b/requirements.txt`](reference/qwen3_6_27b/requirements.txt).

The registered 35B-A3B artifact has its own complete Text/MoE/Vision/MTP diagnostic reference:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain sparse MoE routing briefly." \
  --decode 128 --mtp-draft-tokens 3
```

Both references depend inward on `reference/qwen3_6/common/` leaves. Their exact artifact
bindings, weight residency, state, layer schedules, RefModel lifecycles, and CLIs remain separate;
neither target imports its sibling.

For activation comparisons, build `ninfer-qwen3_6_27b-dump` and follow the target parity guide:

```bash
cmake --build build -j --target ninfer-qwen3_6_27b-dump
```

The C++ diagnostic is implemented in [`qwen3_6_27b_dump/main.cpp`](qwen3_6_27b_dump/main.cpp). It is
target-private and is not part of the public `ninfer::Engine` API.

## Performance tools

[`bench/run_ninfer_bench_matrix.py`](bench/run_ninfer_bench_matrix.py) drives the public-Engine
`ninfer_bench` executable and writes local reports under `profiles/bench/`. Inspect its command
matrix without running the model:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
```

[`bench/make_bench_corpus.py`](bench/make_bench_corpus.py) regenerates or checks the committed token
corpus. [`bench/flash_attn_gqa_bench.py`](bench/flash_attn_gqa_bench.py) is an optional
FlashAttention baseline for the target GQA kernel; it is not a runtime dependency.

CUDA operator microbenchmarks themselves live under the repository-level `bench/` directory.

## Serving smoke

After starting `build/apps/ninfer-serve` in another terminal, exercise the advertised OpenAI,
Anthropic, streaming, and multimodal routes:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This is a manual real-server check, not a CTest.

## Directory map

```text
tools/
├── artifact/                         generic .ninfer codec, layouts, formats, and inspector
├── convert/common/                   shared checkpoint-reading and quantization helpers
├── convert/qwen3_6/common/            narrow Qwen3.6-family conversion leaves
├── convert/qwen3_6_27b/      exact-target converter, inventory, recipe, and verifier
├── convert/qwen3_6_35b_a3b/  registered-target converter, inventory, and source recipe
├── reference/qwen3_6/common/          narrow family-invariant reference leaves
├── reference/qwen3_6_27b/    registered-target artifact-native Python reference
├── reference/qwen3_6_35b_a3b/ registered-target artifact-native Python reference
├── parity/qwen3_6_27b/       target numerical comparison tools
├── qwen3_6_27b_dump/                 C++ target activation diagnostic
├── bench/                            corpus generation and performance orchestration
├── smoke/                            resident-server product smoke client
└── freq_corpus/                      tracked draft-head frequency input and experiment records
```

Generated artifacts, profile reports, Python bytecode caches, and model weights are local outputs
and are ignored by Git. The tracked training-frequency file under `freq_corpus/` is different: it
is a registered input to the 27B target's draft-head conversion recipe.
