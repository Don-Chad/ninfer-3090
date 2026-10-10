"""Real Qwen3.8-Flash-Next GSQ-RCO IQ2_XS GGUF checks, opt-in through environment variables.

NINFER_TEST_GGUF                 first file of the GGUF (…-00001-of-00002.gguf); required
NINFER_GGML_BASE_DLL             ggml-base of llama.cpp b11316, for the sampled-block oracle
NINFER_TEST_QWEN4EXP_SUBSET      an existing ``--subset dev`` artifact to verify instead of
                                 converting one into the test's temporary directory (~2.9 GB)
NINFER_TEST_QWEN38_ARTIFACT      a Qwen3.8 .ninfer whose tokenizer the GGUF vocabulary must equal
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.artifact.codecs.ggml_blocks import decode_blocks
from tools.artifact.formats import GGML_BLOCK_FORMATS
from tools.convert import qwen4_exp
from tools.convert.sources.gguf import GgufModel

from .qwen4_exp_verify import verify

GGUF = os.environ.get("NINFER_TEST_GGUF")
pytestmark = pytest.mark.skipif(not GGUF, reason="set NINFER_TEST_GGUF to the qwen4exp GGUF")


@pytest.fixture(scope="module")
def gguf():
    with GgufModel(GGUF) as model:
        yield model


def test_tensor_inventory_matches_the_scan(gguf):
    assert [len(f.tensors) for f in gguf.files] == [1223, 1]
    totals = Counter()
    for tensor in gguf.tensors.values():
        totals[tensor.type.name] += tensor.bytes
    assert dict(totals) == {
        "IQ4_XS": 1_916_446_720,
        "BF16": 1_527_889_920,
        "F32": 10_063_360,
        "IQ3_S": 163_187_200,
        "Q2_0": 11_326_924_800,
        "IQ4_NL": 28_833_315_840,
        "IQ2_S": 18_271_436_800,
        "IQ2_XXS": 4_757_913_600,
        "Q6_K": 94_617_600,
        "F16": 81_920,
        "Q8_0": 12_185_600,
        "IQ1_M": 1_101_004_800,
    }
    expert_types = {
        layer: gguf.tensor(f"blk.{layer}.ffn_gate_exps.weight").type.name for layer in range(48)
    }
    assert {k for k, v in expert_types.items() if v == "IQ2_XXS"} == {
        1, 4, 9, 10, 11, 14, 18, 19, 27, 29, 30
    }
    assert {k for k, v in expert_types.items() if v == "IQ1_M"} == {8, 13, 37}
    for layer, kind in expert_types.items():
        assert gguf.tensor(f"blk.{layer}.ffn_up_exps.weight").type.name == kind
        assert gguf.tensor(f"blk.{layer}.ffn_down_exps.weight").type.name == "Q2_0"
    assert gguf.tensor(qwen4_exp.PLE_TABLE).shape == (320_001_536, 160)


def test_whole_model_maps_every_tensor(gguf):
    model = qwen4_exp.build_model(gguf)
    assert {m.gguf for m in qwen4_exp.name_map(gguf, model.config)} == set(gguf.tensors)
    assert len(model.parameters) == 1224 - 3 * 48 + 3 * 48 * 512
    assert len(model.packing_groups) == 2 * 48
    assert model.token_count == 248_077


@pytest.mark.skipif(
    not os.environ.get("NINFER_GGML_BASE_DLL"), reason="set NINFER_GGML_BASE_DLL for the oracle"
)
@pytest.mark.parametrize("name", sorted(GGML_BLOCK_FORMATS))
def test_sampled_real_blocks_decode_exactly(gguf, name):
    from tools.artifact.gen_ggml_fixtures import Oracle

    spec = GGML_BLOCK_FORMATS[name]
    tensors = [t for t in gguf.tensors.values() if t.type.format == name]
    rng = np.random.default_rng(spec.ggml_type)
    blocks = []
    for _ in range(1000):
        tensor = tensors[rng.integers(len(tensors))]
        index = int(rng.integers(tensor.bytes // spec.block_bytes))
        start = index * spec.block_bytes
        blocks.append(gguf.read_range(tensor, start, start + spec.block_bytes))
    raw = np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(1000, spec.block_bytes)
    expected = Oracle(os.environ["NINFER_GGML_BASE_DLL"]).decode(spec, raw)
    assert np.array_equal(decode_blocks(name, raw).view(np.uint32), expected.reshape(-1))


def test_dev_subset_stores_the_gguf_bytes(gguf, tmp_path):
    subset = qwen4_exp.SUBSETS["dev"]
    path = os.environ.get("NINFER_TEST_QWEN4EXP_SUBSET")
    if path is None:
        from tools.convert.official_recipes import qwen4_exp_gguf
        from tools.convert.pipeline import convert
        from tools.convert.recipe import Recipe

        path = tmp_path / "qwen4exp-subset.ninfer"
        model = qwen4_exp.build_model(gguf, subset=subset)
        recipe = Recipe(model)
        qwen4_exp_gguf(model, recipe, {"base": gguf})
        convert(model, recipe, path, device="cpu")
    summary = verify(path, gguf, subset)
    assert summary["interleaved"] == 3 and summary["paged"] == 1 and summary["widened"] == 1
    assert summary["objects"] == 72


def _bf16_exact(values: np.ndarray) -> np.ndarray:
    as_f32 = values.astype(np.float32)
    return (as_f32.astype(np.float64) == values) & ((as_f32.view(np.uint32) & 0xFFFF) == 0)


def test_norm_storage_convention(gguf):
    """BF16 gammas plus 1.0 rounded to FP32 leave v - 1 exact in BF16 and v usually not."""
    families = Counter()
    for name, tensor in gguf.tensors.items():
        if tensor.type.name != "F32" or "norm" not in name:
            continue
        family = name.split(".", 2)[-1] if name.startswith("blk.") else name
        values = np.frombuffer(gguf.read_range(tensor, 0, tensor.bytes), "<f4").astype(np.float64)
        stored, offset = _bf16_exact(values), _bf16_exact(values - 1.0)
        if family == "ssm_norm.weight":
            assert stored.all(), name  # a plain BF16 multiplier
        else:
            assert offset.all(), name  # 1 + gamma
            families[family] += int((~stored).sum())
    # Every unit-offset family has stored values that are not BF16 words themselves.
    assert set(families) == {
        "hc_attn_norm.weight", "hc_ffn_norm.weight", "output_hc_norm.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight",
        "indexer.k_norm.weight", "ple_norm_key.weight", "ple_norm_query.weight",
        "ple_norm_conv.weight",
    }
    assert all(count > 0 for count in families.values())


def test_gdn_decay_and_ple_hash_constants(gguf):
    for layer in range(48):
        name = f"blk.{layer}.ssm_a"
        if name in gguf.tensors:
            tensor = gguf.tensor(name)
            values = np.frombuffer(gguf.read_range(tensor, 0, tensor.bytes), "<f4")
            assert (values < 0).all(), name  # -exp(A_log)
    config = qwen4_exp.text_config(gguf.metadata)
    mask, gamma = (1 << 64) - 1, 0x9E3779B97F4A7C15

    def splitmix(x):
        x = (x + gamma) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        return x ^ (x >> 31)

    # Seed 1234 and the first PLE layer's index 0, over the 248,320-row embedding vocabulary.
    half = ((1 << 63) - 1) // 248_320 // 2
    derived = [2 * (splitmix((1234 + gamma * (i + 1)) & mask) % half) + 1 for i in range(3)]
    assert config["ple_layer_multipliers"] == derived
    sizes = config["ple_head_vocab_sizes"]

    def is_prime(v):
        return v > 1 and all(v % d for d in range(2, int(v**0.5) + 1))

    candidate = 20_000_000
    for size in sizes:  # consecutive primes from 20,000,000
        while not is_prime(candidate):
            candidate += 1
        assert size == candidate
        candidate += 1
    rows = gguf.tensor(qwen4_exp.PLE_TABLE).shape[0]
    assert rows == -(-sum(sizes) // 128) * 128


@pytest.mark.skipif(
    not os.environ.get("NINFER_TEST_QWEN38_ARTIFACT"),
    reason="set NINFER_TEST_QWEN38_ARTIFACT to a Qwen3.8 .ninfer",
)
def test_tokenizer_equals_qwen38(gguf):
    from tools.artifact.reader import Artifact

    with Artifact(Path(os.environ["NINFER_TEST_QWEN38_ARTIFACT"])) as artifact:
        resources = artifact.directory.components["text"]["resources"]
        reference = json.loads(artifact.read_object(resources["tokenizer.json"]))
        settings = json.loads(artifact.read_object(resources["tokenizer_config.json"]))
    config = qwen4_exp.text_config(gguf.metadata)
    ours = json.loads(qwen4_exp.tokenizer_resources(gguf.metadata, config)["tokenizer.json"])
    assert ours["model"] == reference["model"]
    for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder"):
        assert ours[key] == reference[key], key
    changed = qwen4_exp.check_tokenizer(reference, ours)
    # llama.cpp stores these <|...|> tokens as CONTROL although the HF tokenizer has special=false.
    assert [ours["added_tokens"][i - 248_044]["content"] for i in changed] == [
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
        "<|repo_name|>", "<|file_sep|>",
    ]
    assert settings["pretokenize_regex"] == qwen4_exp.QWEN35_SPLIT_PATTERN
