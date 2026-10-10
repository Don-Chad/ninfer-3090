from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.ggml_blocks import decode_blocks
from tools.convert.sources.gguf import (
    GGML_TYPES,
    GgufError,
    GgufFile,
    GgufModel,
    split_paths,
    tensor_source,
)

from .gguf_writer import gguf_bytes, tensor_bytes, write_gguf

EVERY_KIND = [
    ("u8", 0, 250),
    ("i8", 1, -7),
    ("u16", 2, 65000),
    ("i16", 3, -300),
    ("u32", 4, 4_000_000_000),
    ("i32", 5, -2_000_000_000),
    ("f32", 6, 1.5),
    ("bool", 7, True),
    ("text", 8, "Ġhello ☃"),
    ("u64", 10, 23703573157769),
    ("i64", 11, -(2**40)),
    ("f64", 12, 1e-6),
    ("ints", 9, (5, [1, -2, 3])),
    ("words", 9, (8, ["a", "b c", ""])),
    ("flags", 9, (7, [True, False])),
    ("nested", 9, (9, [(4, [1, 2]), (8, ["x"])])),
]
EVERY_KIND_VALUES = {
    "ints": [1, -2, 3],
    "words": ["a", "b c", ""],
    "flags": [True, False],
    "nested": [[1, 2], ["x"]],
}


def _every_type_tensors():
    tensors = []
    for index, (type_id, kind) in enumerate(sorted(GGML_TYPES.items())):
        ne = (kind.block_elems * 2, 3)
        tensors.append((f"t{type_id}", type_id, ne, tensor_bytes(type_id, ne, seed=index)))
    return tensors


@pytest.mark.parametrize("alignment", (32, 64))
def test_reader_round_trips_metadata_and_every_tensor_type(tmp_path, alignment):
    metadata = [("general.alignment", 4, alignment), *EVERY_KIND] if alignment != 32 else EVERY_KIND
    tensors = _every_type_tensors()
    path = write_gguf(tmp_path / "model.gguf", metadata, tensors, alignment=alignment)
    gguf = GgufFile(path)
    assert gguf.alignment == alignment and gguf.data_start % alignment == 0
    for key, _, value in EVERY_KIND:
        assert gguf.metadata[key] == EVERY_KIND_VALUES.get(key, value)
    with GgufModel(path) as model:
        for name, type_id, ne, data in tensors:
            tensor = model.tensor(name)
            assert tensor.type.id == type_id and tensor.ne == ne and tensor.shape == ne[::-1]
            assert tensor.offset % alignment == 0 and tensor.bytes == len(data)
            assert model.read_range(tensor, 0, tensor.bytes) == data
            values = tensor_source(model, name).values()
            kind = GGML_TYPES[type_id]
            if kind.dtype is None:
                expected = torch.from_numpy(decode_blocks(kind.format, data))
                rows = tensor_source(model, name).read_encoded(1, 3)
                assert rows.format == kind.format and rows.scales.numel() == 0
                assert rows.codes.numpy().tobytes() == data[tensor.row_bytes :]
                # A partial element range decodes only the blocks that hold it.
                middle = model.read_values(tensor, 5, kind.block_elems + 7)
                assert torch.equal(middle.view(torch.int32), expected[5 : kind.block_elems + 7].view(torch.int32))
            else:
                expected = torch.frombuffer(bytearray(data), dtype=kind.dtype)
                assert tensor_source(model, name).read_encoded is None
            assert values.dtype == expected.dtype
            assert torch.equal(values.view(torch.uint8), expected.reshape(-1).view(torch.uint8))


def test_tensor_source_selects_rows_without_moving_bytes(tmp_path):
    ne = (64, 4, 3)  # three experts of four rows, Q2_0
    data = tensor_bytes(42, ne)
    path = write_gguf(tmp_path / "m.gguf", [], [("experts", 42, ne, data)])
    with GgufModel(path) as model:
        source = tensor_source(model, "experts", rows=(4, 8), shape=(4, 64))
        assert source.shape == (4, 64)
        assert source.read_encoded(0, 4).codes.numpy().tobytes() == data[4 * 18 : 8 * 18]
        with pytest.raises(GgufError, match="outside the view"):
            source.read_encoded(0, 5)
        with pytest.raises(GgufError, match="does not cover"):
            tensor_source(model, "experts", rows=(4, 8), shape=(8, 32))


def _shards(tmp_path, *, count_field=2, tensors_count=2, second_no=1, second_tensors_count=None):
    stem = tmp_path / "m"
    first = [("split.no", 2, 0), ("split.count", 2, count_field), ("split.tensors.count", 5, tensors_count),
             ("general.architecture", 8, "test")]
    second = [("split.no", 2, second_no), ("split.count", 2, count_field),
              ("split.tensors.count", 5, second_tensors_count or tensors_count)]
    a = ("a", 8, (32, 2), tensor_bytes(8, (32, 2)))
    b = ("b", 20, (32, 3), tensor_bytes(20, (32, 3), seed=1))
    write_gguf(stem.with_name("m-00001-of-00002.gguf"), first, [a])
    write_gguf(stem.with_name("m-00002-of-00002.gguf"), second, [b])
    return stem.with_name("m-00001-of-00002.gguf"), a, b


def test_split_files_join_by_name_and_metadata(tmp_path):
    first, a, b = _shards(tmp_path)
    assert [p.name for p in split_paths(first)] == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
    with GgufModel(first) as model:
        assert set(model.tensors) == {"a", "b"}
        assert model.metadata["general.architecture"] == "test"
        assert model.read_range(model.tensor("b"), 0, len(b[3])) == b[3]
    with pytest.raises(GgufError, match="first split file"):
        split_paths(tmp_path / "m-00002-of-00002.gguf")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"count_field": 3}, "split.count"),
        ({"tensors_count": 3}, "split.tensors.count"),
        ({"second_tensors_count": 3}, "split.tensors.count"),
        ({"second_no": 0}, "split.no"),
    ],
)
def test_bad_split_metadata_is_rejected(tmp_path, options, message):
    first, _, _ = _shards(tmp_path, **options)
    with pytest.raises(GgufError, match=message):
        GgufModel(first)


def test_missing_split_file_is_reported(tmp_path):
    first, _, _ = _shards(tmp_path)
    (tmp_path / "m-00002-of-00002.gguf").unlink()
    with pytest.raises(FileNotFoundError):
        GgufModel(first)


def _two(ne=(32, 2), type_id=8):
    return [("x", type_id, ne, tensor_bytes(type_id, ne)), ("y", 8, (32, 2), tensor_bytes(8, (32, 2), 1))]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: gguf_bytes([], _two(), offsets=[0, 32]), "overlaps"),
        (lambda: gguf_bytes([], _two(), offsets=[0, 128]), "next tensor starts"),
        (lambda: gguf_bytes([], _two(), offsets=[0, 70]), "not aligned"),
        (lambda: gguf_bytes([], [("x", 17, (256, 1), bytes(74))]), "GGML type 17"),
        (lambda: gguf_bytes([], [("x", 8, (48, 1), bytes(51))]), "not a multiple"),
        (lambda: gguf_bytes([], _two())[:-10], "exceeds the file"),
        (lambda: gguf_bytes([], _two())[:40], "truncated GGUF header"),
        (lambda: gguf_bytes([], _two(), tail=bytes(64)), "follow the last tensor"),
        (lambda: gguf_bytes([], _two(), version=2), "version 2"),
        (lambda: b"GGML" + gguf_bytes([], _two())[4:], "not a GGUF file"),
        (lambda: gguf_bytes([("k", 4, 1), ("k", 4, 2)], _two()), "duplicate metadata"),
        (lambda: gguf_bytes([("general.alignment", 4, 48)], _two()), "power of two"),
        (lambda: gguf_bytes([("bad", 13, b"")], []), "value type 13"),
        (lambda: gguf_bytes([], [_two()[0], _two()[0]]), "duplicate tensor"),
    ],
)
def test_malformed_files_fail_with_clear_errors(tmp_path, build, message):
    path = tmp_path / "bad.gguf"
    path.write_bytes(build())
    with pytest.raises(GgufError, match=message):
        GgufFile(path)
