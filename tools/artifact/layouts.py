"""Persistent tensor layouts shared by NInfer converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import operator
from types import MappingProxyType
from typing import Sequence


from .formats import (
    GGML_BLOCK_FORMATS,
    DirectFormat,
    Fp8RowFormat,
    GgmlBlockFormat,
    Nvfp4Format,
    NumericFormat,
    QuantFormat,
    get_format,
)

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128
ROW_PAGE_BYTES = 4096


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    weight_divisor_offset: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class GgmlBlocksGeometry:
    rows: int
    k: int
    blocks_per_row: int
    row_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class GgmlRowPageGeometry:
    rows: int
    k: int
    row_bytes: int
    rows_per_page: int
    pages: int
    payload_bytes: int

    def row_offset(self, row: int) -> int:
        """Payload offset of logical row *row*; no row crosses a page."""
        if not 0 <= row < self.rows:
            raise ValueError(f"row {row} is outside [0,{self.rows})")
        page, slot = divmod(row, self.rows_per_page)
        return page * ROW_PAGE_BYTES + slot * self.row_bytes


CONTIGUOUS_LE_V1 = Layout("contiguous_le_v1", 256, frozenset(("bf16", "fp32", "int32")))
ROW_SPLIT_K128_V1 = Layout(
    "row_split_k128_v1",
    256,
    frozenset(("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")),
)
BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "block_scale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4",)),
)
ROW_SCALE_V1 = Layout(
    "row_scale_v1",
    256,
    frozenset(("fp8_e4m3fn_row_bf16",)),
)
GGML_BLOCKS_V1 = Layout("ggml_blocks_v1", 256, frozenset(GGML_BLOCK_FORMATS))
GGML_ROWS_PAGE4K_V1 = Layout(
    "ggml_rows_page4k_v1", ROW_PAGE_BYTES, frozenset(GGML_BLOCK_FORMATS)
)

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCK_SCALE_K16_M128X4_V1,
            ROW_SCALE_V1,
            GGML_BLOCKS_V1,
            GGML_ROWS_PAGE4K_V1,
        )
    }
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row_split_k128_v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    base_bytes_per_group = spec.group_size if spec.bits == 8 else spec.group_size // 2
    high_bytes_per_group = (
        0 if spec.bits in (4, 8) else spec.group_size * (spec.bits - 4) // 8
    )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int]
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
    n, k = _shape(shape, rank=2)
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            "block_scale_k16_m128x4_v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * k // spec.group_size
    weight_divisor_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=weight_divisor_offset,
        payload_bytes=weight_divisor_offset + 4,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * 2
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def _ggml_rows(
    format: str | GgmlBlockFormat, shape: Sequence[int], *, rank: int | None = None
) -> tuple[GgmlBlockFormat, int, int]:
    spec = _format(format)
    if not isinstance(spec, GgmlBlockFormat):
        raise ValueError("GGML block layouts require a GGML block format")
    dims = _shape(shape, rank=rank)
    if not dims or len(dims) > 16:
        raise ValueError("GGML block layouts support rank 1 through 16")
    k = dims[-1]
    if k % spec.block_elems:
        raise ValueError(
            f"{spec.name} requires K divisible by {spec.block_elems}, got K={k}"
        )
    return spec, prod(dims[:-1]), k


def ggml_blocks_geometry(
    format: str | GgmlBlockFormat, shape: Sequence[int]
) -> GgmlBlocksGeometry:
    """Rows of K/block_elems consecutive blocks; rows and leading axes are row-major."""
    spec, rows, k = _ggml_rows(format, shape)
    blocks_per_row = k // spec.block_elems
    row_bytes = blocks_per_row * spec.block_bytes
    return GgmlBlocksGeometry(rows, k, blocks_per_row, row_bytes, rows * row_bytes)


def ggml_row_page_geometry(
    format: str | GgmlBlockFormat, shape: Sequence[int]
) -> GgmlRowPageGeometry:
    """Whole rows packed into 4096-byte pages; the unused page tail is zero."""
    spec, rows, k = _ggml_rows(format, shape, rank=2)
    row_bytes = k // spec.block_elems * spec.block_bytes
    if row_bytes > ROW_PAGE_BYTES:
        raise ValueError(
            f"ggml_rows_page4k_v1 requires rows of at most {ROW_PAGE_BYTES} bytes, "
            f"got {row_bytes}"
        )
    rows_per_page = ROW_PAGE_BYTES // row_bytes
    pages = -(-rows // rows_per_page)
    return GgmlRowPageGeometry(
        rows, k, row_bytes, rows_per_page, pages, pages * ROW_PAGE_BYTES
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous_le_v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous_le_v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row_split_k128_v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_K16_M128X4_V1:
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
        return block_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_SCALE_V1:
        if not isinstance(numeric_spec, Fp8RowFormat):
            raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is GGML_BLOCKS_V1:
        return ggml_blocks_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is GGML_ROWS_PAGE4K_V1:
        return ggml_row_page_geometry(numeric_spec, shape).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")
