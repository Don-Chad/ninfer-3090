"""Check a converted qwen4exp artifact byte for byte against the GGUF it came from."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np

from tools.artifact.codecs.ggml_blocks import unpage_rows
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert import qwen4_exp
from tools.convert.sources.gguf import GgufModel


def _hash(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def verify(path, gguf: GgufModel, subset: qwen4_exp.Subset | None = None) -> dict:
    """Every tensor object equals its GGUF ranges; returns counts of what was compared.

    Plain objects are compared by SHA-256 over the GGUF byte ranges of their bound parameters in
    binding order. Layout transforms are inverted exactly: expert banks must bind expert e's gate
    then up rows, PLE pages are unpaged (tails must be zero), and F16 widened to FP32 must narrow
    back to the original words.
    """
    config = qwen4_exp.text_config(gguf.metadata)
    mappings = {item.parameter: item for item in qwen4_exp.name_map(gguf, config)}
    summary = defaultdict(int)
    with Artifact(path) as artifact:
        directory = artifact.directory
        parts = defaultdict(list)
        for name, binding in directory.bindings.items():
            for object_id, begin, end in binding_parts(binding, artifact.by_id, name):
                parts[object_id].append((begin, end, name))
        for obj in directory.objects:
            if not isinstance(obj, TensorObject):
                continue
            bound = sorted(parts[obj.id])
            assert bound, f"{obj.id} has no binding"
            ranges = []
            for begin, end, name in bound:
                item = mappings[name]
                tensor = gguf.tensor(item.gguf)
                rows = item.rows or (0, tensor.rows)
                if item.gguf == qwen4_exp.PLE_TABLE and subset is not None:
                    rows = (0, subset.ple_rows)
                ranges.append((tensor, rows, name))
            if obj.layout == "ggml_rows_page4k_v1":
                ((tensor, rows, _),) = ranges
                stored = unpage_rows(artifact.read_object(obj.id), obj.format, obj.shape)
                assert np.array_equal(stored, gguf.read_rows(tensor, *rows)), obj.id
                summary["paged"] += 1
            elif obj.format == "fp32" and ranges[0][0].type.name == "F16":
                ((tensor, _, _),) = ranges
                source = gguf.read_range(tensor, 0, tensor.bytes)
                widened = np.frombuffer(artifact.read_object(obj.id), dtype="<f4")
                assert widened.tobytes() == np.frombuffer(source, "<f2").astype("<f4").tobytes()
                assert widened.astype("<f2").tobytes() == source, obj.id
                summary["widened"] += 1
            else:
                if len(ranges) > 1 and ranges[0][2].endswith("/gate"):
                    names = [name for _, _, name in ranges]
                    experts = len(names) // 2
                    expected = [
                        f"{prefix}{e}/{role}"
                        for prefix in [names[0].rsplit("/", 2)[0] + "/"]
                        for e in range(experts)
                        for role in ("gate", "up")
                    ]
                    assert names == expected, f"{obj.id}: experts are not gate|up interleaved"
                    summary["interleaved"] += 1
                source = _hash(
                    chunk
                    for tensor, (first, last), _ in ranges
                    for chunk in gguf.iter_range(
                        tensor, first * tensor.row_bytes, last * tensor.row_bytes
                    )
                )
                assert _hash(artifact.iter_object(obj.id)) == source, obj.id
                summary["identical"] += 1
            summary["objects"] += 1
            summary["bytes"] += obj.bytes
        resources = directory.components["text"]["resources"]
        synthesized = qwen4_exp.tokenizer_resources(gguf.metadata, config)
        for role, object_id in resources.items():
            data = artifact.read_object(object_id)
            if role.endswith(".json"):
                assert json.loads(data) == json.loads(synthesized[role]), role
            else:
                assert data == synthesized[role], role
    return dict(summary)
