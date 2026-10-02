from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert import imatrix
from tools.convert.methods import grouped_search
from tools.convert.model import Model, Parameter
from tools.convert.quantization.groupwise import search_quantize_matrix
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

from .test_recipe import _write


def _model(values):
    model = Model({"text": {"config": {}}})
    for name, value in values.items():
        model.add(Parameter(name, tuple(value.shape), array_source(value, name), inputs=("input",)))
    model.packing_groups = [tuple(values)]
    return model


def test_grouped_search_packs_inputs_and_matches_the_weighted_oracle(tmp_path):
    generator = torch.Generator().manual_seed(3)
    values = {name: (torch.randn((6, 192), generator=generator) * 0.05).to(torch.bfloat16)
              for name in ("query", "key")}
    importance = torch.rand(192, generator=generator) * 4
    path = tmp_path / "imatrix.safetensors"
    save_file({"query": importance, "key": importance.clone()}, str(path))
    imatrix.load.cache_clear()

    model = _model(values)
    recipe = Recipe(model)
    recipe.assign(("query", "key"), format="q4_g64_fp16", method=grouped_search,
                  parameters={"imatrix": str(path), "negative_scales": True})
    prepared = recipe.prepare(device="cpu", rows_per_chunk=4)
    artifact_path = tmp_path / "search.ninfer"
    _write(artifact_path, model, prepared)

    expected = search_quantize_matrix(torch.cat(list(values.values())), "q4_g64_fp16",
                                      importance=importance, negative_scales=True,
                                      device="cpu")
    with Artifact(artifact_path) as artifact:
        parts = {name: binding_parts(binding, artifact.by_id)[0]
                 for name, binding in artifact.directory.bindings.items()}
        assert parts["query"][0] == parts["key"][0]
        obj = artifact.object(parts["query"][0])
        scales, codes = decode_row_split_codes(artifact.read_object(obj.id), obj.format, obj.shape)
    assert torch.equal(scales, expected.scales)
    assert torch.equal(codes, expected.codes)
    assert bool((scales < 0).any())


def test_grouped_search_rejects_unknown_parameters():
    model = _model({"query": torch.zeros((2, 64), dtype=torch.bfloat16)})
    recipe = Recipe(model)
    recipe.assign("query", format="q4_g64_fp16", method=grouped_search,
                  parameters={"clip": 1.0})
    with pytest.raises(ValueError, match="unknown grouped_search parameters"):
        recipe.prepare(device="cpu", rows_per_chunk=4)


def test_imatrix_import_maps_roles_counts_and_value_head_order(tmp_path):
    gguf = pytest.importorskip("gguf")
    import numpy as np

    config = {"layer_types": ["linear_attention", "full_attention"],
              "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 2}
    path = tmp_path / "imatrix.gguf"
    writer = gguf.GGUFWriter(str(path), "imatrix")
    out_sums = np.arange(8, dtype=np.float32) * 2
    tensors = {
        "blk.0.attn_qkv.weight": np.full(4, 6.0, dtype=np.float32),
        "blk.0.ssm_out.weight": out_sums,
        "blk.1.attn_q.weight": np.full(4, 9.0, dtype=np.float32),
    }
    for name, sums in tensors.items():
        writer.add_tensor(name + ".in_sum2", sums)
        writer.add_tensor(name + ".counts", np.array([2.0], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    vectors = imatrix.from_gguf(path, config)
    assert set(vectors) == {"text/layers/0/gdn/query", "text/layers/0/gdn/key",
                            "text/layers/0/gdn/value", "text/layers/0/gdn/output",
                            "text/layers/1/attention/query", "text/layers/1/attention/gate"}
    assert torch.equal(vectors["text/layers/0/gdn/key"], torch.full((4,), 3.0))
    # GGUF column i of ssm_out is HF column order[i]; the import restores HF order.
    order = imatrix.v_head_order(2, 4, 2)
    hf = vectors["text/layers/0/gdn/output"]
    assert torch.equal(hf[order], torch.from_numpy(out_sums / 2))
    assert not torch.equal(order, torch.arange(8))
