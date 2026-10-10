"""A minimal GGUF v3 writer for tests: metadata of every value type, aligned tensor data."""

from __future__ import annotations

from math import prod
from pathlib import Path
import struct

import numpy as np

from tools.convert.sources.gguf import GGML_TYPES

SCALAR_CODES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_value(kind: int, value) -> bytes:
    """Arrays are ``(item_kind, items)``; nested arrays repeat that form. Bytes are written raw."""
    if isinstance(value, bytes):
        return value
    if kind in SCALAR_CODES:
        return struct.pack("<" + SCALAR_CODES[kind], value)
    if kind == STRING:
        return _string(value)
    if kind == ARRAY:
        item_kind, items = value
        return struct.pack("<IQ", item_kind, len(items)) + b"".join(
            encode_value(item_kind, item) for item in items
        )
    raise ValueError(kind)


def tensor_bytes(type_id: int, ne: tuple[int, ...], seed: int = 0) -> bytes:
    """Random blocks for a block type; finite random words for F32, F16 and BF16."""
    kind = GGML_TYPES[type_id]
    rng = np.random.default_rng(seed)
    if kind.dtype is None:
        size = prod(ne) // kind.block_elems * kind.block_bytes
        return rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
    values = rng.standard_normal(prod(ne)).astype(np.float32)
    if kind.name == "F16":
        return values.astype(np.float16).tobytes()
    if kind.name == "BF16":
        return (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    return values.tobytes()


def gguf_bytes(
    metadata: list[tuple[str, int, object]],
    tensors: list[tuple[str, int, tuple[int, ...], bytes]],
    *,
    alignment: int = 32,
    version: int = 3,
    offsets: list[int] | None = None,
    tail: bytes = b"",
) -> bytes:
    """Serialize a GGUF file. *offsets* overrides the relative data offsets (for bad files)."""
    header = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, kind, value in metadata:
        header += _string(key) + struct.pack("<I", kind) + encode_value(kind, value)
    cursor, placed = 0, []
    for index, (_, _, _, data) in enumerate(tensors):
        offset = offsets[index] if offsets is not None else cursor
        placed.append(offset)
        cursor = -(-(offset + len(data)) // alignment) * alignment
    for (name, type_id, ne, _), offset in zip(tensors, placed):
        header += _string(name) + struct.pack(f"<I{len(ne)}Q", len(ne), *ne)
        header += struct.pack("<IQ", type_id, offset)
    start = -(-len(header) // alignment) * alignment
    end = max((offset + len(data) for (_, _, _, data), offset in zip(tensors, placed)), default=0)
    payload = bytearray(end)
    for (_, _, _, data), offset in zip(tensors, placed):
        payload[offset : offset + len(data)] = data
    return header + bytes(start - len(header)) + bytes(payload) + tail


def write_gguf(path: Path, metadata, tensors, **options) -> Path:
    path.write_bytes(gguf_bytes(metadata, tensors, **options))
    return path
