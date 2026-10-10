"""Regenerate the golden GGML block fixtures from ggml's own reference decoders.

Usage: python -m tools.artifact.gen_ggml_fixtures [--dll PATH] [--out tests/fixtures/ggml]

The oracle is ``dequantize_row_<type>`` exported by ggml-base.dll (or libggml-base.so) of llama.cpp
release b11316; the path comes from ``--dll`` or ``NINFER_GGML_BASE_DLL``. Each fixture
``<format>.npz`` holds ``blocks`` (uint8 [n, block_bytes]) and ``expected`` (the oracle's FP32 bit
patterns as uint32 [n, block_elems]). Blocks are random bytes with structured fields overwritten so
that every grid index, sign pattern and scale field of the format occurs, and the binary16 scales
include zeros of both signs, subnormals, the largest finite values, infinities and NaNs.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np

from .formats import GGML_BLOCK_FORMATS, GgmlBlockFormat

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "ggml"
# ggml's C names for the dequantizers.
GGML_NAMES = {
    "ggml_q8_0": "q8_0",
    "ggml_q6_k": "q6_K",
    "ggml_iq2_xxs": "iq2_xxs",
    "ggml_iq4_nl": "iq4_nl",
    "ggml_iq3_s": "iq3_s",
    "ggml_iq2_s": "iq2_s",
    "ggml_iq4_xs": "iq4_xs",
    "ggml_iq1_m": "iq1_m",
    "ggml_q2_0": "q2_0",
}
# Binary16 scale words: signed zeros, unit, subnormal extremes, normal extremes, inf, NaNs.
SPECIAL_SCALES = (
    0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001, 0x03FF, 0x0400,
    0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01, 0x7D00, 0x1234,
)


class Oracle:
    """ggml's reference dequantizers through ctypes."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.library = ctypes.CDLL(str(path))
        self.library.ggml_type_size.restype = ctypes.c_size_t
        self.library.ggml_type_size.argtypes = (ctypes.c_int,)
        self.library.ggml_blck_size.restype = ctypes.c_int64
        self.library.ggml_blck_size.argtypes = (ctypes.c_int,)

    def geometry(self, spec: GgmlBlockFormat) -> tuple[int, int]:
        return (
            int(self.library.ggml_blck_size(spec.ggml_type)),
            int(self.library.ggml_type_size(spec.ggml_type)),
        )

    def decode(self, spec: GgmlBlockFormat, blocks: np.ndarray) -> np.ndarray:
        blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
        if blocks.ndim != 2 or blocks.shape[1] != spec.block_bytes:
            raise ValueError(f"blocks must be uint8 [n, {spec.block_bytes}]")
        out = np.empty((blocks.shape[0], spec.block_elems), dtype=np.float32)
        function = getattr(self.library, "dequantize_row_" + GGML_NAMES[spec.name])
        function.restype = None
        function.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        function(blocks.ctypes.data, out.ctypes.data, out.size)
        return out.view(np.uint32)


def _put_u16(blocks: np.ndarray, offset: int, values: np.ndarray) -> None:
    blocks[:, offset] = values & 0xFF
    blocks[:, offset + 1] = values >> 8


def _scales(count: int, rng: np.random.Generator) -> np.ndarray:
    special = np.asarray(SPECIAL_SCALES, dtype=np.uint16)
    normal = rng.integers(0x0400, 0x7C00, size=count, dtype=np.uint16)
    normal |= rng.integers(0, 2, size=count, dtype=np.uint16) << 15
    normal[: len(special)] = special
    return normal[:count]


def structured_blocks(spec: GgmlBlockFormat, count: int, seed: int = 0) -> np.ndarray:
    """Random blocks whose structured fields cover the format's whole code domain."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(count, spec.block_bytes), dtype=np.uint8)
    n = np.arange(count)
    d = _scales(count, rng)
    name = spec.name
    if name == "ggml_q6_k":
        _put_u16(b, 208, d)
        b[0, 192:208] = 0x80  # int8 -128 scales
        b[1, 192:208] = 0x7F  # int8 127 scales
    elif name != "ggml_iq1_m":
        _put_u16(b, 0, d)
    if name == "ggml_iq4_xs":
        ls = (8 * n[:, None] + np.arange(8)) % 64  # every 6-bit sub-block scale
        low, high = ls & 0xF, ls >> 4
        b[:, 4:8] = low[:, 0::2] | (low[:, 1::2] << 4)
        scales_h = np.zeros(count, dtype=np.uint16)
        for ib in range(8):
            scales_h |= (high[:, ib] << (2 * ib)).astype(np.uint16)
        _put_u16(b, 2, scales_h)
    elif name == "ggml_iq2_xxs":
        for ib in range(8):
            base = 2 + 8 * ib
            k = 32 * n + 4 * ib  # grid index and sign selector of each of the 4 groups
            for l in range(4):
                b[:, base + l] = (k + l) % 256
            aux = np.zeros(count, dtype=np.uint64)
            for l in range(4):
                aux |= (((k + l) % 128).astype(np.uint64)) << np.uint64(7 * l)
            aux |= (((n + ib) % 16).astype(np.uint64)) << np.uint64(28)
            for byte in range(4):
                b[:, base + 4 + byte] = (aux >> np.uint64(8 * byte)) & 0xFF
    elif name == "ggml_iq2_s":
        qh = np.zeros((count, 8), dtype=np.uint8)
        for ib in range(8):
            for l in range(4):
                v = (32 * n + 4 * ib + l) % 1024
                b[:, 2 + 4 * ib + l] = v & 0xFF
                qh[:, ib] |= ((v >> 8) << (2 * l)).astype(np.uint8)
        b[:, 66:74] = qh
        b[:, 34:66] = (32 * n[:, None] + np.arange(32)) % 256  # every sign byte
        nibble = np.arange(16, dtype=np.uint8)  # every 4-bit scale in each block
        b[:, 74:82] = nibble[0::2] | (nibble[1::2] << 4)
    elif name == "ggml_iq3_s":
        qh = np.zeros((count, 8), dtype=np.uint8)
        for g in range(8):
            for j in range(8):  # j = 2*l + pair
                v = (64 * n + 8 * g + j) % 512
                b[:, 2 + 8 * g + j] = v & 0xFF
                qh[:, g] |= ((v >> 8) << j).astype(np.uint8)
        b[:, 66:74] = qh
        b[:, 74:106] = (32 * n[:, None] + np.arange(32)) % 256
        nibble = (8 * n[:, None] + np.arange(8)) % 16
        b[:, 106:110] = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    elif name == "ggml_iq1_m":
        for ib in range(8):
            v = [(32 * n + 4 * ib + l) % 2048 for l in range(4)]
            for l in range(4):
                b[:, 4 * ib + l] = v[l] & 0xFF
            delta = b[:, 32 + 2 * ib : 34 + 2 * ib] & 0x88  # keep random delta-sign bits
            b[:, 32 + 2 * ib] = delta[:, 0] | (v[0] >> 8) | ((v[1] >> 8) << 4)
            b[:, 33 + 2 * ib] = delta[:, 1] | (v[2] >> 8) | ((v[3] >> 8) << 4)
        words = np.zeros((count, 4), dtype=np.uint16)
        for w in range(4):
            fields = [(n + 4 * w + f) % 8 for f in range(4)]  # four 3-bit sub-block scales
            word = fields[0] | (fields[1] << 3) | (fields[2] << 6) | (fields[3] << 9)
            words[:, w] = word.astype(np.uint16) | (((d >> (4 * w)) & 0xF) << 12)
        for w in range(4):
            _put_u16(b, 48 + 2 * w, words[:, w])
    return b


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dll", default=os.environ.get("NINFER_GGML_BASE_DLL"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.dll:
        parser.error("provide --dll or NINFER_GGML_BASE_DLL")
    oracle = Oracle(args.dll)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in GGML_BLOCK_FORMATS.values():
        if oracle.geometry(spec) != (spec.block_elems, spec.block_bytes):
            raise SystemExit(f"{spec.name}: oracle geometry {oracle.geometry(spec)} differs")
        count = 256 if spec.block_elems <= 64 else 64
        blocks = structured_blocks(spec, count)
        path = args.out / f"{spec.name}.npz"
        np.savez_compressed(path, blocks=blocks, expected=oracle.decode(spec, blocks))
        print(f"{path}: {count} blocks, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
