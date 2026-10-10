"""Exact decoders for the stored GGML block formats, and the two byte layouts that hold them.

Each decoder reproduces ggml's reference ``dequantize_row_*`` (ggml-quants.c, llama.cpp release
b11316) bit for bit: every FP32 product and sum is formed in the same order as the C source,
one IEEE operation at a time, so the results equal ggml's FP32 bit patterns. The tests compare
them against ggml-base.dll from that release. NInfer has no encoder for these formats; the
converter only imports existing blocks.

``ggml_blocks_v1`` stores the blocks unchanged: a row is K / block_elems consecutive blocks and
rows are row-major. ``ggml_rows_page4k_v1`` packs whole rows into 4096-byte pages with a zero
tail, so a row never crosses a page; ``page_rows`` and ``unpage_rows`` are exact inverses.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..formats import GgmlBlockFormat
from ..layouts import (
    ROW_PAGE_BYTES,
    _format,
    ggml_blocks_geometry,
    ggml_row_page_geometry,
)
from . import ggml_tables as tables

_F32 = np.float32
_KMASK = np.asarray(tables.KMASK_IQ2XS, dtype=np.uint8)
_KSIGNS = np.asarray(tables.KSIGNS_IQ2XS, dtype=np.uint8)
_KVALUES_IQ4NL = np.asarray(tables.KVALUES_IQ4NL, dtype=np.int8).astype(_F32)
_IQ2XXS = np.asarray(tables.IQ2XXS_GRID, dtype="<u8").view(np.uint8).reshape(-1, 8)
_IQ2S = np.asarray(tables.IQ2S_GRID, dtype="<u8").view(np.uint8).reshape(-1, 8)
_IQ3S = np.asarray(tables.IQ3S_GRID, dtype="<u4").view(np.uint8).reshape(-1, 4)
_IQ1S = np.asarray(tables.IQ1S_GRID, dtype="<u8").view(np.int8).reshape(-1, 8)
_IQ1S_DELTA = _F32(0.125)


def _fp16(raw: np.ndarray) -> np.ndarray:
    """Little-endian binary16 words from two byte columns, widened exactly to FP32."""
    words = raw[..., 0].astype(np.uint16) | (raw[..., 1].astype(np.uint16) << 8)
    return words.view(np.float16).astype(_F32)


def _apply_signs(values: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """ggml's ``value * (bits & kmask_iq2xs[j] ? -1.f : 1.f)`` over a trailing axis j = 0..7.

    The factor is applied as a negation (sign-bit flip), which equals the product for every
    non-NaN value. IEEE leaves a NaN product's sign unspecified; the reference build negates.
    """
    flip = ((bits[..., None] & _KMASK) != 0).astype(np.uint32) << np.uint32(31)
    return (np.ascontiguousarray(values, dtype=_F32).view(np.uint32) ^ flip).view(_F32)


def _q8_0(b: np.ndarray) -> np.ndarray:
    d = _fp16(b[:, 0:2])
    qs = b[:, 2:34].view(np.int8).astype(_F32)
    return qs * d[:, None]


def _q2_0(b: np.ndarray) -> np.ndarray:
    d = _fp16(b[:, 0:2])
    shifts = np.arange(0, 8, 2, dtype=np.uint8)
    q = (b[:, 2:18, None] >> shifts) & 3
    return (q.reshape(-1, 64).astype(np.int32) - 1).astype(_F32) * d[:, None]


def _iq4_nl(b: np.ndarray) -> np.ndarray:
    d = _fp16(b[:, 0:2])[:, None]
    qs = b[:, 2:18]
    return np.concatenate(
        (d * _KVALUES_IQ4NL[qs & 0xF], d * _KVALUES_IQ4NL[qs >> 4]), axis=1
    )


def _iq4_xs(b: np.ndarray) -> np.ndarray:
    d = _fp16(b[:, 0:2])
    scales_h = b[:, 2].astype(np.int32) | (b[:, 3].astype(np.int32) << 8)
    ib = np.arange(8)
    low = (b[:, 4 + ib // 2] >> (4 * (ib % 2)).astype(np.uint8)) & 0xF
    ls = low.astype(np.int32) | (((scales_h[:, None] >> (2 * ib)) & 3) << 4)
    dl = (d[:, None] * (ls - 32).astype(_F32))[:, :, None]
    qs = b[:, 8:136].reshape(-1, 8, 16)
    y = np.concatenate(
        (dl * _KVALUES_IQ4NL[qs & 0xF], dl * _KVALUES_IQ4NL[qs >> 4]), axis=2
    )
    return y.reshape(-1, 256)


def _q6_k(b: np.ndarray) -> np.ndarray:
    ql = b[:, 0:128].reshape(-1, 2, 64).astype(np.int32)
    qh = b[:, 128:192].reshape(-1, 2, 32).astype(np.int32)
    sc = b[:, 192:208].view(np.int8).reshape(-1, 2, 8).astype(_F32)
    d = _fp16(b[:, 208:210])[:, None, None]
    low, high = ql[:, :, :32], ql[:, :, 32:]
    q = np.stack(
        (
            (low & 0xF) | (((qh >> 0) & 3) << 4),
            (high & 0xF) | (((qh >> 2) & 3) << 4),
            (low >> 4) | (((qh >> 4) & 3) << 4),
            (high >> 4) | (((qh >> 6) & 3) << 4),
        ),
        axis=2,
    )  # [block, half, quarter, l]
    q = (q - 32).astype(_F32)
    l = np.arange(32)
    # quarter j of half n uses scale index is + 2*j with is = l / 16.
    scale = sc[:, :, (l // 16)[None, :] + 2 * np.arange(4)[:, None]]
    return ((d[..., None] * scale) * q).reshape(-1, 256)


def _iq2_xxs(b: np.ndarray) -> np.ndarray:
    d = _fp16(b[:, 0:2])
    q = b[:, 2:66].reshape(-1, 8, 8)
    aux1 = q[:, :, 4:8].astype(np.uint32)
    aux1 = aux1[..., 0] | (aux1[..., 1] << 8) | (aux1[..., 2] << 16) | (aux1[..., 3] << 24)
    db = (d[:, None] * (_F32(0.5) + (aux1 >> 28).astype(_F32))) * _F32(0.25)
    grid = _IQ2XXS[q[:, :, 0:4]].astype(_F32)  # [block, ib32, l, j]
    selectors = (aux1[:, :, None] >> (7 * np.arange(4, dtype=np.uint32))) & 127
    signs = _KSIGNS[selectors]
    return _apply_signs(db[:, :, None, None] * grid, signs).reshape(-1, 256)


def _iq2_s(b: np.ndarray) -> np.ndarray:
    d = _fp16(b[:, 0:2])
    qs = b[:, 2:34].reshape(-1, 8, 4).astype(np.int32)
    signs = b[:, 34:66].reshape(-1, 8, 4)
    qh = b[:, 66:74].astype(np.int32)
    scales = b[:, 74:82]
    l = np.arange(4)
    index = qs | ((qh[:, :, None] << (8 - 2 * l)) & 0x300)
    db = np.stack(
        (
            d[:, None] * (_F32(0.5) + (scales & 0xF).astype(_F32)) * _F32(0.25),
            d[:, None] * (_F32(0.5) + (scales >> 4).astype(_F32)) * _F32(0.25),
        ),
        axis=2,
    )  # [block, ib32, half]
    dl = db[:, :, l // 2]
    grid = _IQ2S[index].astype(_F32)
    return _apply_signs(dl[..., None] * grid, signs).reshape(-1, 256)


def _iq3_s(b: np.ndarray) -> np.ndarray:
    d = _fp16(b[:, 0:2])
    qs = b[:, 2:66].reshape(-1, 8, 4, 2).astype(np.int32)  # [block, group, l, pair]
    qh = b[:, 66:74].astype(np.int32)
    signs = b[:, 74:106].reshape(-1, 8, 4)
    scales = b[:, 106:110]
    group = np.arange(8)
    nibble = (scales[:, group // 2] >> (4 * (group % 2)).astype(np.uint8)) & 0xF
    db = d[:, None] * (1 + 2 * nibble.astype(np.int32)).astype(_F32)
    l = np.arange(4)
    high1 = (qh[:, :, None] << (8 - 2 * l)) & 256
    high2 = (qh[:, :, None] << (7 - 2 * l)) & 256
    grid = np.concatenate(
        (_IQ3S[qs[..., 0] | high1], _IQ3S[qs[..., 1] | high2]), axis=3
    ).astype(_F32)  # [block, group, l, 8]
    return _apply_signs(db[:, :, None, None] * grid, signs).reshape(-1, 256)


def _iq1_m(b: np.ndarray) -> np.ndarray:
    qs = b[:, 0:32].reshape(-1, 8, 4).astype(np.int32)
    qh = b[:, 32:48].reshape(-1, 8, 2).astype(np.int32)
    sc = b[:, 48:56].reshape(-1, 4, 2).astype(np.uint16)
    sc = sc[..., 0] | (sc[..., 1] << 8)
    scale = (
        (sc[:, 0] >> 12)
        | ((sc[:, 1] >> 8) & 0x00F0)
        | ((sc[:, 2] >> 4) & 0x0F00)
        | (sc[:, 3] & 0xF000)
    )
    d = scale.astype(np.uint16).view(np.float16).astype(_F32)
    ib = np.arange(8)
    word = sc[:, ib // 2].astype(np.int32)
    shift = 6 * (ib % 2)
    dl1 = d[:, None] * (2 * ((word >> shift) & 7) + 1).astype(_F32)
    dl2 = d[:, None] * (2 * ((word >> (shift + 3)) & 7) + 1).astype(_F32)
    index = np.stack(
        (
            qs[..., 0] | ((qh[..., 0] << 8) & 0x700),
            qs[..., 1] | ((qh[..., 0] << 4) & 0x700),
            qs[..., 2] | ((qh[..., 1] << 8) & 0x700),
            qs[..., 3] | ((qh[..., 1] << 4) & 0x700),
        ),
        axis=2,
    )
    delta_bits = np.stack(
        (qh[..., 0] & 0x08, qh[..., 0] & 0x80, qh[..., 1] & 0x08, qh[..., 1] & 0x80),
        axis=2,
    )
    delta = np.where(delta_bits != 0, -_IQ1S_DELTA, _IQ1S_DELTA)
    shifted = _IQ1S[index].astype(_F32) + delta[..., None]  # grid[j] + delta[l]
    dl = np.stack((dl1, dl1, dl2, dl2), axis=2)
    return (dl[..., None] * shifted).reshape(-1, 256)


_DECODERS = {
    "ggml_q8_0": _q8_0,
    "ggml_q2_0": _q2_0,
    "ggml_iq4_nl": _iq4_nl,
    "ggml_iq4_xs": _iq4_xs,
    "ggml_q6_k": _q6_k,
    "ggml_iq2_xxs": _iq2_xxs,
    "ggml_iq2_s": _iq2_s,
    "ggml_iq3_s": _iq3_s,
    "ggml_iq1_m": _iq1_m,
}


def _spec(format: str | GgmlBlockFormat) -> GgmlBlockFormat:
    spec = _format(format)
    if not isinstance(spec, GgmlBlockFormat):
        raise ValueError(f"{spec.name} is not a GGML block format")
    return spec


def _bytes(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if data.dtype != np.uint8:
            raise TypeError("GGML block data must be uint8")
        return data.reshape(-1)
    return np.frombuffer(data, dtype=np.uint8)


def decode_blocks(
    format: str | GgmlBlockFormat,
    data: bytes | bytearray | memoryview | np.ndarray,
) -> np.ndarray:
    """Decode whole blocks to one flat FP32 array, exactly as ggml's ``dequantize_row_*``."""
    spec = _spec(format)
    raw = _bytes(data)
    if raw.size % spec.block_bytes:
        raise ValueError(
            f"{spec.name}: {raw.size} bytes is not a whole number of "
            f"{spec.block_bytes}-byte blocks"
        )
    blocks = raw.reshape(-1, spec.block_bytes)
    with np.errstate(all="ignore"):
        values = _DECODERS[spec.name](blocks)
    return np.ascontiguousarray(values, dtype=_F32).reshape(-1)


def decode_ggml_blocks(
    payload: bytes | bytearray | memoryview | np.ndarray,
    format: str | GgmlBlockFormat,
    shape: Sequence[int],
) -> np.ndarray:
    """Decode a complete ``ggml_blocks_v1`` payload to FP32 values of *shape*."""
    geometry = ggml_blocks_geometry(format, shape)
    raw = _bytes(payload)
    if raw.size != geometry.payload_bytes:
        raise ValueError(
            f"ggml_blocks_v1 payload has {raw.size} bytes, expected {geometry.payload_bytes}"
        )
    return decode_blocks(format, raw).reshape(tuple(shape))


def page_rows(
    rows: np.ndarray, format: str | GgmlBlockFormat, shape: Sequence[int], row_begin: int
) -> tuple[int, np.ndarray]:
    """Place encoded rows ``[row_begin, row_begin+len(rows))`` into whole pages.

    *rows* is uint8 ``[count, row_bytes]``. The rows must start a page and either fill whole
    pages or end the tensor, so every returned page is complete. Returns the payload offset and
    the page bytes ``[pages, 4096]``.
    """
    geometry = ggml_row_page_geometry(format, shape)
    count = rows.shape[0]
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != geometry.row_bytes:
        raise TypeError(f"rows must be uint8 [count, {geometry.row_bytes}]")
    end = row_begin + count
    if count <= 0 or row_begin % geometry.rows_per_page or end > geometry.rows:
        raise ValueError("paged rows must start a page and stay inside the tensor")
    if count % geometry.rows_per_page and end != geometry.rows:
        raise ValueError("paged rows must fill whole pages unless they end the tensor")
    pages = -(-count // geometry.rows_per_page)
    out = np.zeros((pages, ROW_PAGE_BYTES), dtype=np.uint8)
    used = geometry.rows_per_page * geometry.row_bytes
    padded = np.zeros((pages * geometry.rows_per_page, geometry.row_bytes), np.uint8)
    padded[:count] = rows
    out[:, :used] = padded.reshape(pages, used)
    return row_begin // geometry.rows_per_page * ROW_PAGE_BYTES, out


def unpage_rows(
    payload: bytes | bytearray | memoryview | np.ndarray,
    format: str | GgmlBlockFormat,
    shape: Sequence[int],
) -> np.ndarray:
    """Recover the ``[rows, row_bytes]`` block rows of a ``ggml_rows_page4k_v1`` payload.

    Rejects a payload whose page tails are not zero, so paging round-trips exactly.
    """
    geometry = ggml_row_page_geometry(format, shape)
    raw = _bytes(payload)
    if raw.size != geometry.payload_bytes:
        raise ValueError(
            f"ggml_rows_page4k_v1 payload has {raw.size} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    pages = raw.reshape(geometry.pages, ROW_PAGE_BYTES)
    used = geometry.rows_per_page * geometry.row_bytes
    if pages[:, used:].any():
        raise ValueError("ggml_rows_page4k_v1 page tails must be zero")
    rows = pages[:, :used].reshape(-1, geometry.row_bytes)
    if rows[geometry.rows :].any():
        raise ValueError("ggml_rows_page4k_v1 unused rows of the last page must be zero")
    return rows[: geometry.rows]
