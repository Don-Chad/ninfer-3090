from __future__ import annotations

import json

import pytest

from tools.convert import qwen4_exp
from tools.convert.__main__ import main
from tools.convert.official_recipes import qwen4_exp_gguf
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GgufError, GgufModel

from . import qwen4_exp_fixture as fixture
from .qwen4_exp_verify import verify


@pytest.fixture
def gguf_path(tmp_path):
    return fixture.write(tmp_path / "tiny.gguf")


def test_config_comes_from_metadata(gguf_path):
    with GgufModel(gguf_path) as gguf:
        config = qwen4_exp.text_config(gguf.metadata)
    assert config["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    assert config["rope_parameters"] == {
        "rope_theta": 1e7,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    }
    assert (config["linear_num_value_heads"], config["linear_value_head_dim"]) == (4, 16)
    assert (config["num_experts"], config["moe_intermediate_size"]) == (4, fixture.F)
    assert config["indexer_compress_ratio"] == 4 and config["ple_layers"] == [1]
    assert config["ple_layer_multipliers"] == [11, 13, 17]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"general.architecture": (8, "qwen35")}, "general.architecture"),
        ({"qwen4exp.attention.compress_ratios": (9, (5, [0, 0, 4, 0]))}, "compress_ratios"),
        ({"qwen4exp.ple.head_offsets": (9, (11, [0, 5, 13, 23]))}, "running sum"),
        ({"qwen4exp.ple.head_offsets": (9, (11, [4, 9, 16, 27]))}, "from zero"),
        ({"qwen4exp.rope.dimension_count": (4, 10)}, "rope sections"),
        ({"qwen4exp.expert_used_count": (4, 5)}, "expert_used_count"),
        (
            {
                "qwen4exp.ple.ngram_size": (4, 1),
                "qwen4exp.ple.head_offsets": (9, (11, [])),
                "qwen4exp.ple.head_vocab_sizes": (9, (11, [])),
                "qwen4exp.ple.layer_multipliers": (9, (11, [11])),
            },
            "ple.ngram_size: expected at least 2",
        ),
        ({"qwen4exp.attention.layer_norm_rms_epsilon": (12, float("inf"))}, "finite"),
        ({"qwen4exp.attention.layer_norm_rms_epsilon": (12, float("nan"))}, "finite"),
        ({"qwen4exp.rope.freq_base": (12, float("inf"))}, "finite"),
        ({"qwen4exp.rope.freq_base": (12, -1.0)}, "positive"),
    ],
)
def test_inconsistent_metadata_is_rejected(tmp_path, changes, message):
    path = fixture.write(tmp_path / "bad.gguf", metadata_changes=changes)
    with GgufModel(path) as gguf, pytest.raises(GgufError, match=message):
        qwen4_exp.text_config(gguf.metadata)


def test_name_map_covers_every_tensor_once(gguf_path):
    with GgufModel(gguf_path) as gguf:
        model = qwen4_exp.build_model(gguf)
        mapped = {item.gguf for item in qwen4_exp.name_map(gguf, model.config)}
        assert mapped == set(gguf.tensors)
    experts = [name for name in model.parameters if "/moe/experts/" in name]
    assert len(experts) == 4 * fixture.E * 3
    assert len(model.parameters) == len(gguf.tensors) - 3 * 4 + len(experts)
    assert len(model.packing_groups) == 8  # gate|up and down banks of each layer
    (gate_up,) = [g for g in model.packing_groups if "text/layers/0/moe/experts/0/gate" in g]
    assert gate_up == tuple(
        f"text/layers/0/moe/experts/{e}/{role}" for e in range(fixture.E) for role in ("gate", "up")
    )
    assert model.parameters["text/layers/0/moe/shared_score"].shape == (1, fixture.H)
    assert model.parameters["text/ple/table"].shape == (fixture.PLE_ROWS, fixture.PLE_DIM)
    assert model.parameters["text/layers/0/gdn/output"].inputs == ("text/layers/0/gdn/gated_output",)
    assert model.token_count == 14 and model.special_token_ids == (10, 11, 12)


@pytest.mark.parametrize(
    ("drop", "extra", "message"),
    [
        (("blk.1.ple_key.weight",), (), "layer 1 is missing"),
        (("output_hc_up.weight",), (), "output_hc_up.weight' is missing"),
        ((), (("blk.0.attn_q.weight", 8, (64, 128)),), "unexpected GGUF tensor"),
        ((), (("blk.2.ple_key.weight", 30, (64, 128)),), "unexpected GGUF tensor"),
        ((), (("rope_freqs.weight", 0, (4,)),), "unknown GGUF tensor"),
        (("blk.0.ssm_out.weight",), (("blk.0.ssm_out.weight", 8, (32, 64)),), "ssm_out.weight: shape"),
        (("blk.1.hc_ffn_norm.weight",), (("blk.1.hc_ffn_norm.weight", 0, (fixture.H,)),), "hc_ffn_norm.weight: shape"),
        (("blk.2.ffn_up_exps.weight",), (("blk.2.ffn_up_exps.weight", 8, (fixture.H, fixture.F + 8, fixture.E)),), "ffn_up_exps.weight: shape"),
        (("blk.3.attn_k.weight",), (("blk.3.attn_k.weight", 8, (fixture.H, 64)),), "attn_k.weight: shape"),
        (("blk.3.indexer.q_proj.weight",), (("blk.3.indexer.q_proj.weight", 30, (fixture.H, 48)),), "indexer.q_proj.weight: shape"),
        (("blk.3.indexer.k_proj.weight",), (("blk.3.indexer.k_proj.weight", 30, (fixture.H, 32)),), "indexer.k_proj.weight: shape"),
        (("blk.3.indexer.q_norm.weight",), (("blk.3.indexer.q_norm.weight", 0, (32,)),), "indexer.q_norm.weight: shape"),
        (("blk.3.indexer.k_norm.weight",), (("blk.3.indexer.k_norm.weight", 0, (8,)),), "indexer.k_norm.weight: shape"),
    ],
)
def test_name_map_rejects_missing_misplaced_and_unknown_tensors(tmp_path, drop, extra, message):
    path = fixture.write(tmp_path / "bad.gguf", drop=drop, extra=extra)
    with GgufModel(path) as gguf, pytest.raises(GgufError, match=message):
        qwen4_exp.build_model(gguf)


def _convert(gguf, out, subset=None):
    model = qwen4_exp.build_model(gguf, subset=subset)
    recipe = Recipe(model)
    qwen4_exp_gguf(model, recipe, {"base": gguf})
    return convert(model, recipe, out, device="cpu")


def test_conversion_stores_gguf_bytes_exactly(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        report = _convert(gguf, tmp_path / "tiny.ninfer")
        summary = verify(tmp_path / "tiny.ninfer", gguf)
    assert summary["interleaved"] == 4 and summary["paged"] == 1 and summary["widened"] == 1
    assert summary["objects"] == len(report["methods"])
    formats = {item["format"] for item in report["methods"]}
    assert {"ggml_q8_0", "ggml_iq4_nl", "ggml_q2_0", "bf16", "fp32"} == formats
    (table,) = [item for item in report["methods"] if item["parameters"] == ("text/ple/table",)]
    assert table["layout"] == "ggml_rows_page4k_v1"
    assert tuple(table["shape"]) == (fixture.PLE_ROWS, fixture.PLE_DIM)


@pytest.mark.parametrize("target", ("gate_up", "ple"))
def test_verification_detects_a_changed_byte(gguf_path, tmp_path, target):
    from tools.artifact.reader import Artifact

    out = tmp_path / "tiny.ninfer"
    with GgufModel(gguf_path) as gguf:
        report = _convert(gguf, out)
    want = "text/ple/table" if target == "ple" else "text/layers/2/moe/experts/0/gate"
    (object_id,) = [m["object"] for m in report["methods"] if want in m["parameters"]]
    with Artifact(out) as artifact:
        obj = artifact.by_id[object_id]
        # The last byte of the PLE object is page-tail padding; the expert byte is real data.
        at = artifact.payload_offset + obj.offset + (obj.bytes - 1 if target == "ple" else 7)
    data = bytearray(out.read_bytes())
    data[at] ^= 0x40
    out.write_bytes(data)
    with GgufModel(gguf_path) as gguf, pytest.raises((AssertionError, ValueError)):
        verify(out, gguf)


def test_ggml_projections_permit_a8_and_bf16_projections_stay_a16(gguf_path, tmp_path):
    from tools.artifact.reader import Artifact

    with GgufModel(gguf_path) as gguf:
        _convert(gguf, tmp_path / "tiny.ninfer")
    with Artifact(tmp_path / "tiny.ninfer") as artifact:
        policies = {(u["parameter"], u["input"]): u["activation_policy"] for u in artifact.directory.uses}
    assert policies[("text/layers/0/gdn/qkv", "text/layers/0/mixer_input")] == "AllowA8"
    assert policies[("text/layers/0/moe/experts/3/down", "text/layers/0/moe/experts/3/product")] == "AllowA8"
    assert policies[("text/layers/0/moe/router", "text/layers/0/ffn_input")] == "A16Only"
    assert policies[("text/output_head", "text/final_hidden")] == "AllowA8"


def test_subset_keeps_chosen_tensors_and_first_ple_rows(gguf_path, tmp_path):
    subset = qwen4_exp.Subset(
        tensors=frozenset({"token_embd.weight", "blk.3.attn_q.weight"}),
        direct_layers=frozenset({1}),
        expert_layers=frozenset({2}),
        ple_rows=227,  # exactly one page
    )
    with GgufModel(gguf_path) as gguf:
        model = qwen4_exp.build_model(gguf, subset=subset)
        assert model.parameters["text/ple/table"].shape == (227, fixture.PLE_DIM)
        assert "text/layers/3/attention/query_gate" in model.parameters
        assert "text/layers/1/ple/key" in model.parameters
        assert "text/layers/0/gdn/qkv" not in model.parameters
        assert "text/layers/1/gdn/qkv" not in model.parameters  # block format, not selected
        assert {n.split("/")[2] for n in model.parameters if "/experts/" in n} == {"2"}
        _convert(gguf, tmp_path / "subset.ninfer", subset)
        summary = verify(tmp_path / "subset.ninfer", gguf, subset)
    assert summary["paged"] == 1 and summary["interleaved"] == 1


def test_cli_converts_a_gguf_subset(gguf_path, tmp_path, monkeypatch):
    monkeypatch.setitem(
        qwen4_exp.SUBSETS,
        "dev",
        qwen4_exp.Subset(frozenset({"blk.0.attn_qkv.weight"}), frozenset(), frozenset({0}), 10),
    )
    out = tmp_path / "cli.ninfer"
    main(["--model", str(gguf_path), "--recipe", "qwen4_exp_gguf", "--subset", "dev",
          "--device", "cpu", "--out", str(out)])
    report = json.loads((tmp_path / "cli.ninfer.conversion.json").read_text())
    assert report["provenance"]["subset"] == "dev"
    assert report["provenance"]["sources"]["base"]["paths"] == [str(gguf_path)]
    with GgufModel(gguf_path) as gguf:
        verify(out, gguf, qwen4_exp.SUBSETS["dev"])
    with pytest.raises(ValueError, match="only the text component"):
        main(["--model", str(gguf_path), "--recipe", "qwen4_exp_gguf", "--components",
              "text,vision", "--device", "cpu", "--out", str(tmp_path / "x.ninfer")])


def test_synthesized_tokenizer_and_override_check(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        config = qwen4_exp.text_config(gguf.metadata)
        resources = qwen4_exp.tokenizer_resources(gguf.metadata, config)
        tokenizer = json.loads(resources["tokenizer.json"])
        assert tokenizer["model"]["vocab"] == {chr(ord("a") + i): i for i in range(10)}
        assert tokenizer["model"]["merges"] == ["a b", "ab c"]
        assert [(t["id"], t["special"]) for t in tokenizer["added_tokens"]] == [
            (10, True), (11, True), (12, True), (13, False)
        ]
        assert tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == (
            qwen4_exp.QWEN35_SPLIT_PATTERN
        )
        settings = json.loads(resources["tokenizer_config.json"])
        assert (settings["eos_token"], settings["pad_token"]) == ("<|im_end|>", "<|endoftext|>")
        assert json.loads(resources["generation_config.json"])["eos_token_id"] == [12, 10]

        # A Hugging Face tokenizer may clear "special" on a <|...|> token llama.cpp made CONTROL.
        candidate = json.loads(resources["tokenizer.json"])
        candidate["added_tokens"][1]["special"] = False
        assert qwen4_exp.check_tokenizer(candidate, tokenizer) == [11]
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(candidate), encoding="utf-8")
        model = qwen4_exp.build_model(gguf, resource_overrides={"tokenizer.json": path})
        assert model.special_token_ids == (10, 12)
        for change, message in (
            (lambda t: t["added_tokens"][3].update(special=True), "special flag conflicts"),
            (lambda t: t["model"]["merges"].pop(), "model.merges"),
            (lambda t: t["added_tokens"][0].update(content="<|eot|>"), "added tokens differ"),
        ):
            bad = json.loads(resources["tokenizer.json"])
            change(bad)
            with pytest.raises(ValueError, match=message):
                qwen4_exp.check_tokenizer(bad, tokenizer)


def test_unsupported_tokenizer_is_rejected(tmp_path):
    path = fixture.write(tmp_path / "t.gguf", metadata_changes={"tokenizer.ggml.pre": (8, "llama3")})
    with GgufModel(path) as gguf, pytest.raises(GgufError, match="qwen35"):
        qwen4_exp.build_model(gguf)


@pytest.mark.parametrize(
    "changes",
    [
        {"qwen4exp.ple.eos_token_id": (4, 14)},  # an unused [PAD] token
        {"qwen4exp.ple.eos_token_id": (4, 99)},  # outside the vocabulary
        {"tokenizer.ggml.bos_token_id": (4, 15)},
    ],
)
def test_generated_stop_ids_must_name_used_tokens(tmp_path, changes):
    path = fixture.write(tmp_path / "t.gguf", metadata_changes=changes)
    with GgufModel(path) as gguf, pytest.raises(GgufError, match="does not name a used token"):
        qwen4_exp.build_model(gguf)


@pytest.mark.parametrize(
    "changes",
    [
        {"general.sampling.temp": (6, float("nan"))},
        {"general.sampling.top_p": (12, float("inf"))},
        {"general.sampling.temp": (12, 1e39)},  # finite, but not representable in FP32
    ],
)
def test_non_finite_sampling_values_are_rejected(tmp_path, changes):
    path = fixture.write(tmp_path / "s.gguf", metadata_changes=changes)
    with GgufModel(path) as gguf, pytest.raises(GgufError, match="finite|overflows FP32"):
        qwen4_exp.build_model(gguf)
