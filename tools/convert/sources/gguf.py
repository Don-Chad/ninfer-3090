"""Strict GGUF v3 reader: metadata, tensor regions, split files and bounded range reads.

Only the tensor types the converter can store exactly are admitted: F32, F16 and BF16 as direct
values, and the GGML block formats registered in ``tools.artifact.formats``. Any other type is
rejected rather than guessed. Files are read with positional reads of at most ``READ_CHUNK_BYTES``;
nothing is memory-mapped, so a 30 GB shard costs no address space or page-cache pinning.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import os
from pathlib import Path
import re
import struct
from typing import Iterator

import numpy as np
import torch

from tools.artifact.codecs.ggml_blocks import decode_blocks
from tools.artifact.file_io import open_read, pread
from tools.artifact.formats import GGML_BLOCK_FORMATS
from .logical import EncodedRows, LogicalSource

MAGIC = b"GGUF"
VERSION = 3
DEFAULT_ALIGNMENT = 32
MAX_DIMS = 4
READ_CHUNK_BYTES = 64 * 1024 * 1024
_SPLIT_NAME = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


class GgufError(ValueError):
    """A GGUF file does not satisfy the reader's contract."""


@dataclass(frozen=True, slots=True)
class GgmlType:
    """A tensor type the converter can store: direct words or an exact block format."""

    id: int
    name: str
    block_elems: int
    block_bytes: int
    format: str  # artifact format name: a GGML block format or the direct word format
    dtype: torch.dtype | None = None  # direct types only


_DIRECT_TYPES = (
    GgmlType(0, "F32", 1, 4, "fp32", torch.float32),
    GgmlType(1, "F16", 1, 2, "fp32", torch.float16),  # widened exactly to FP32
    GgmlType(30, "BF16", 1, 2, "bf16", torch.bfloat16),
)
GGML_TYPES = {
    **{item.id: item for item in _DIRECT_TYPES},
    **{
        spec.ggml_type: GgmlType(
            spec.ggml_type,
            spec.name.removeprefix("ggml_").upper(),
            spec.block_elems,
            spec.block_bytes,
            spec.name,
        )
        for spec in GGML_BLOCK_FORMATS.values()
    },
}

# GGUF metadata value types: (struct code, byte width) for scalars; 8 string, 9 array.
_SCALARS = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
_STRING, _ARRAY = 8, 9


@dataclass(frozen=True, slots=True)
class GgufTensor:
    name: str
    type: GgmlType
    ne: tuple[int, ...]  # GGUF order: ne[0] is the fastest (K) axis
    path: Path
    offset: int  # absolute file offset of the first byte
    bytes: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Row-major logical shape: the reversed GGUF dimensions, with identical bytes."""
        return tuple(reversed(self.ne))

    @property
    def elements(self) -> int:
        return prod(self.ne)

    @property
    def row_elems(self) -> int:
        return self.ne[0]

    @property
    def rows(self) -> int:
        return self.elements // self.ne[0]

    @property
    def row_bytes(self) -> int:
        return self.ne[0] // self.type.block_elems * self.type.block_bytes


class _Short(Exception):
    """The header continues beyond the bytes read so far."""


class _Cursor:
    def __init__(self, data: bytes, path: Path, complete: bool) -> None:
        self.data = memoryview(data)
        self.pos = 0
        self.path = path
        self.complete = complete

    def take(self, count: int) -> memoryview:
        end = self.pos + count
        if end > len(self.data):
            if not self.complete:
                raise _Short
            raise GgufError(f"{self.path}: truncated GGUF header at byte {self.pos}")
        view = self.data[self.pos : end]
        self.pos = end
        return view

    def scalar(self, code: str, width: int):
        return struct.unpack("<" + code, self.take(width))[0]

    def string(self) -> str:
        length = self.scalar("Q", 8)
        try:
            return str(self.take(length), "utf-8")
        except UnicodeDecodeError as error:
            raise GgufError(f"{self.path}: invalid UTF-8 string") from error

    def value(self, kind: int, depth: int = 0):
        if kind in _SCALARS:
            value = self.scalar(*_SCALARS[kind])
            return bool(value) if kind == 7 else value
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            if depth > 8:
                raise GgufError(f"{self.path}: nested arrays are too deep")
            item_kind, count = self.scalar("I", 4), self.scalar("Q", 8)
            if item_kind in _SCALARS and item_kind != 7:
                code, width = _SCALARS[item_kind]
                return list(struct.unpack(f"<{count}{code}", self.take(count * width)))
            return [self.value(item_kind, depth + 1) for _ in range(count)]
        raise GgufError(f"{self.path}: unknown metadata value type {kind}")


class GgufFile:
    """One validated GGUF v3 file; the header is parsed once, tensor data is read on demand."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.size = self.path.stat().st_size
        window = min(self.size, 16 * 1024 * 1024)
        with self.path.open("rb") as stream:
            while True:
                stream.seek(0)
                data = stream.read(window)
                try:
                    infos = self._parse(data, complete=window >= self.size)
                    break
                except _Short:
                    window = min(self.size, window * 4)
        self.tensors: dict[str, GgufTensor] = {}
        for name, ne, type_id, offset in infos:
            if name in self.tensors:
                raise GgufError(f"{self.path}: duplicate tensor {name!r}")
            self.tensors[name] = self._tensor(name, ne, type_id, offset)
        self._validate_regions()

    def _parse(self, data: bytes, *, complete: bool) -> list:
        cursor = _Cursor(data, self.path, complete)
        if bytes(cursor.take(4)) != MAGIC:
            raise GgufError(f"{self.path}: not a GGUF file")
        version = cursor.scalar("I", 4)
        if version != VERSION:
            raise GgufError(f"{self.path}: GGUF version {version}, expected {VERSION}")
        tensor_count, kv_count = cursor.scalar("Q", 8), cursor.scalar("Q", 8)
        self.metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = cursor.string()
            if key in self.metadata:
                raise GgufError(f"{self.path}: duplicate metadata key {key!r}")
            self.metadata[key] = cursor.value(cursor.scalar("I", 4))
        alignment = self.metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise GgufError(f"{self.path}: general.alignment must be a power of two")
        self.alignment = alignment
        infos = []
        for _ in range(tensor_count):
            name = cursor.string()
            dims = cursor.scalar("I", 4)
            if not 1 <= dims <= MAX_DIMS:
                raise GgufError(f"{self.path}: {name}: {dims} dimensions")
            ne = tuple(cursor.scalar("Q", 8) for _ in range(dims))
            infos.append((name, ne, cursor.scalar("I", 4), cursor.scalar("Q", 8)))
        self.data_start = -(-cursor.pos // alignment) * alignment
        if self.data_start > self.size:
            raise GgufError(f"{self.path}: tensor data starts beyond the end of the file")
        return infos

    def _tensor(self, name: str, ne: tuple[int, ...], type_id: int, offset: int):
        try:
            kind = GGML_TYPES[type_id]
        except KeyError:
            raise GgufError(
                f"{self.path}: {name}: GGML type {type_id} is not supported "
                f"(supported: {sorted(GGML_TYPES)})"
            ) from None
        if any(dim <= 0 for dim in ne):
            raise GgufError(f"{self.path}: {name}: dimensions must be positive, got {ne}")
        if ne[0] % kind.block_elems:
            raise GgufError(
                f"{self.path}: {name}: ne[0]={ne[0]} is not a multiple of the "
                f"{kind.name} block size {kind.block_elems}"
            )
        if offset % self.alignment:
            raise GgufError(f"{self.path}: {name}: offset {offset} is not aligned")
        size = prod(ne) // kind.block_elems * kind.block_bytes
        return GgufTensor(name, kind, ne, self.path, self.data_start + offset, size)

    def _validate_regions(self) -> None:
        size = self.size
        ordered = sorted(self.tensors.values(), key=lambda item: item.offset)
        for index, tensor in enumerate(ordered):
            end = tensor.offset + tensor.bytes
            if end > size:
                raise GgufError(
                    f"{self.path}: {tensor.name}: region [{tensor.offset},{end}) "
                    f"exceeds the file ({size} bytes)"
                )
            if index + 1 == len(ordered) and size - end >= self.alignment:
                raise GgufError(
                    f"{self.path}: {size - end} bytes follow the last tensor {tensor.name}"
                )
            if index + 1 < len(ordered):
                following = ordered[index + 1].offset
                if following < end:
                    raise GgufError(
                        f"{self.path}: {tensor.name} overlaps {ordered[index + 1].name}"
                    )
                spacing = following - tensor.offset
                expected = -(-tensor.bytes // self.alignment) * self.alignment
                if spacing != expected:
                    raise GgufError(
                        f"{self.path}: {tensor.name}: {tensor.bytes} bytes but the next "
                        f"tensor starts {spacing} bytes later"
                    )


def split_paths(path: str | Path) -> tuple[Path, ...]:
    """All files of a split GGUF named ``<stem>-0000N-of-0000M.gguf``, or just *path*."""
    path = Path(path)
    match = _SPLIT_NAME.match(path.name)
    if match is None:
        return (path,)
    count = int(match["count"])
    if int(match["no"]) != 1 or count < 1:
        raise GgufError(f"{path}: name the first split file (-00001-of-{count:05d})")
    return tuple(
        path.with_name(f"{match['stem']}-{index:05d}-of-{count:05d}.gguf")
        for index in range(1, count + 1)
    )


class GgufModel:
    """The tensors and metadata of one GGUF model, possibly split across several files."""

    def __init__(self, paths: str | Path | tuple[str | Path, ...]) -> None:
        if isinstance(paths, (str, Path)):
            paths = split_paths(paths)
        if not paths:
            raise GgufError("a GGUF model needs at least one file")
        self.files = tuple(GgufFile(path) for path in paths)
        self.path = self.files[0].path
        self._check_splits()
        self.metadata = dict(self.files[0].metadata)
        for file in self.files[1:]:
            for key, value in file.metadata.items():
                if key.startswith("split."):
                    continue
                if key in self.metadata and self.metadata[key] != value:
                    raise GgufError(f"{file.path}: metadata {key!r} disagrees with split 0")
                self.metadata.setdefault(key, value)
        self.tensors: dict[str, GgufTensor] = {}
        for file in self.files:
            for name, tensor in file.tensors.items():
                if name in self.tensors:
                    raise GgufError(f"{file.path}: tensor {name!r} is in two split files")
                self.tensors[name] = tensor
        advertised = {file.metadata.get("split.tensors.count") for file in self.files}
        if advertised != {None} and advertised != {len(self.tensors)}:
            raise GgufError(
                f"split.tensors.count {sorted(map(str, advertised))} but the files hold "
                f"{len(self.tensors)}"
            )
        self._fds: dict[Path, int] = {}
        self.bytes_read = 0

    def _check_splits(self) -> None:
        records = [
            (file.metadata.get("split.no"), file.metadata.get("split.count"))
            for file in self.files
        ]
        if len(self.files) == 1 and records[0] == (None, None):
            return
        counts = {count for _, count in records}
        if counts != {len(self.files)}:
            raise GgufError(
                f"split.count {sorted(map(str, counts))} does not match {len(self.files)} files"
            )
        if [number for number, _ in records] != list(range(len(self.files))):
            raise GgufError(f"split.no values {[n for n, _ in records]} are not 0..N-1")

    def __enter__(self) -> GgufModel:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def tensor(self, name: str) -> GgufTensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise GgufError(f"GGUF tensor {name!r} is missing") from None

    def iter_range(
        self,
        tensor: GgufTensor,
        begin: int,
        end: int,
        *,
        chunk_bytes: int = READ_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        """Yield tensor bytes ``[begin, end)`` in bounded positional reads."""
        if not 0 <= begin <= end <= tensor.bytes:
            raise GgufError(f"{tensor.name}: byte range [{begin},{end}) is outside the tensor")
        fd = self._fds.get(tensor.path)
        if fd is None:
            fd = self._fds[tensor.path] = open_read(tensor.path)
        while begin < end:
            count = min(end - begin, chunk_bytes)
            data = pread(fd, count, tensor.offset + begin)
            if len(data) != count:
                raise GgufError(f"{tensor.name}: short read at tensor byte {begin}")
            self.bytes_read += count
            yield data
            begin += count

    def read_range(self, tensor: GgufTensor, begin: int, end: int) -> bytes:
        return b"".join(self.iter_range(tensor, begin, end))

    def read_rows(self, tensor: GgufTensor, begin: int, end: int) -> np.ndarray:
        """Rows ``[begin, end)`` of the ``[rows, ne[0]]`` view as uint8 ``[n, row_bytes]``."""
        if not 0 <= begin <= end <= tensor.rows:
            raise GgufError(f"{tensor.name}: rows [{begin},{end}) outside {tensor.rows}")
        raw = self.read_range(tensor, begin * tensor.row_bytes, end * tensor.row_bytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(end - begin, tensor.row_bytes)

    def read_values(self, tensor: GgufTensor, begin: int, end: int) -> torch.Tensor:
        """Flat C-order elements ``[begin, end)``: direct words, or exact FP32 block decodes."""
        if not 0 <= begin <= end <= tensor.elements:
            raise GgufError(f"{tensor.name}: elements [{begin},{end}) outside the tensor")
        kind = tensor.type
        if kind.dtype is not None:
            raw = self.read_range(tensor, begin * kind.block_bytes, end * kind.block_bytes)
            return torch.frombuffer(bytearray(raw), dtype=kind.dtype)
        first, last = begin // kind.block_elems, -(-end // kind.block_elems)
        raw = self.read_range(tensor, first * kind.block_bytes, last * kind.block_bytes)
        values = decode_blocks(kind.format, raw)
        low = begin - first * kind.block_elems
        return torch.from_numpy(values[low : low + end - begin].copy())


def tensor_source(
    model: GgufModel,
    name: str,
    *,
    rows: tuple[int, int] | None = None,
    shape: tuple[int, ...] | None = None,
) -> LogicalSource:
    """A logical view of one GGUF tensor in its own row-major order, with identical bytes.

    *rows* selects ``[begin, end)`` of the ``[rows, ne[0]]`` matrix view; *shape* reshapes the
    selection without moving bytes. Block-format tensors also expose their exact rows through
    ``read_encoded``; their scales live inside the blocks, so ``EncodedRows.scales`` is empty.
    """
    tensor = model.tensor(name)
    first, last = (0, tensor.rows) if rows is None else rows
    if not 0 <= first < last <= tensor.rows:
        raise GgufError(f"{name}: invalid row selection {rows}")
    k = tensor.row_elems
    if shape is None:
        shape = tensor.shape if rows is None else (last - first, k)
    shape = tuple(shape)
    if prod(shape) != (last - first) * k or shape[-1] != k:
        raise GgufError(f"{name}: shape {shape} does not cover rows [{first},{last}) of K={k}")
    label = name if rows is None else f"{name}[rows {first}:{last}]"
    base = first * k

    def read_values(begin: int, end: int) -> torch.Tensor:
        return model.read_values(tensor, base + begin, base + end)

    read_encoded = None
    if tensor.type.dtype is None:

        def read_encoded(begin: int, end: int) -> EncodedRows:
            if not 0 <= begin <= end <= last - first:
                raise GgufError(f"{label}: encoded rows [{begin},{end}) are outside the view")
            codes = model.read_rows(tensor, first + begin, first + end)
            return EncodedRows(
                tensor.type.format,
                torch.from_numpy(codes.copy()),
                torch.empty((end - begin, 0), dtype=torch.uint8),
            )

    return LogicalSource(shape, label, read_values, read_encoded)
