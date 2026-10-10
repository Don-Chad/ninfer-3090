"""GGML block formats: exact decoders against ggml b11316, and the two block layouts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tools.artifact.codecs.ggml_blocks import (
    decode_blocks,
    decode_ggml_blocks,
    page_rows,
    unpage_rows,
)
from tools.artifact.formats import GGML_BLOCK_FORMATS, get_format
from tools.artifact.layouts import (
    encoded_size,
    ggml_blocks_geometry,
    ggml_row_page_geometry,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ggml"
FORMATS = sorted(GGML_BLOCK_FORMATS)
ORACLE = os.environ.get("NINFER_GGML_BASE_DLL")


def _fixture(name: str):
    with np.load(FIXTURES / f"{name}.npz", allow_pickle=False) as data:
        return data["blocks"], data["expected"]


@pytest.mark.parametrize("name", FORMATS)
def test_decoder_matches_ggml_reference_bit_for_bit(name):
    blocks, expected = _fixture(name)
    spec = get_format(name)
    assert blocks.shape[1] == spec.block_bytes and expected.shape[1] == spec.block_elems
    decoded = decode_blocks(name, blocks).view(np.uint32).reshape(expected.shape)
    mismatch = np.argwhere(decoded != expected)
    assert mismatch.size == 0, f"first mismatch at block/element {mismatch[0].tolist()}"


@pytest.mark.parametrize("name", FORMATS)
def test_fixture_stays_small_and_covers_special_scales(name):
    assert (FIXTURES / f"{name}.npz").stat().st_size < 64_000
    _, expected = _fixture(name)
    values = expected.view(np.float32)
    # Signed zeros, infinities and NaNs from the special binary16 scales reach the output.
    assert np.isnan(values).any() and np.isinf(values).any()
    assert ((expected == 0x80000000).any() and (expected == 0).any())


def _iq2_xxs_fields(blocks):
    q = blocks[:, 2:66].reshape(-1, 8, 8).astype(np.uint32)
    aux = q[..., 4] | q[..., 5] << 8 | q[..., 6] << 16 | q[..., 7] << 24
    signs = np.stack([(aux >> (7 * l)) & 127 for l in range(4)], axis=-1)
    return q[..., :4], signs


def test_fixtures_cover_every_grid_index_and_sign_pattern():
    blocks, _ = _fixture("ggml_iq2_xxs")
    grid, signs = _iq2_xxs_fields(blocks)
    assert set(np.unique(grid)) == set(range(256))
    assert set(np.unique(signs)) == set(range(128))

    blocks, _ = _fixture("ggml_iq2_s")
    qs = blocks[:, 2:34].reshape(-1, 8, 4).astype(np.int32)
    qh = blocks[:, 66:74].astype(np.int32)
    index = qs | ((qh[:, :, None] << (8 - 2 * np.arange(4))) & 0x300)
    assert set(np.unique(index)) == set(range(1024))
    assert set(np.unique(blocks[:, 34:66])) == set(range(256))

    blocks, _ = _fixture("ggml_iq3_s")
    qs = blocks[:, 2:66].reshape(-1, 8, 8).astype(np.int32)
    qh = blocks[:, 66:74].astype(np.int32)
    index = qs | (((qh[:, :, None] >> np.arange(8)) & 1) << 8)
    assert set(np.unique(index)) == set(range(512))

    blocks, _ = _fixture("ggml_iq1_m")
    qs = blocks[:, 0:32].reshape(-1, 8, 4).astype(np.int32)
    qh = blocks[:, 32:48].reshape(-1, 8, 2).astype(np.int32)
    high = np.stack((qh[..., 0] & 7, qh[..., 0] >> 4 & 7, qh[..., 1] & 7, qh[..., 1] >> 4 & 7), -1)
    assert set(np.unique(qs | high << 8)) == set(range(2048))
    assert {0, 0x08, 0x80, 0x88} == set(np.unique(qh & 0x88))
    words = blocks[:, 48:56].reshape(-1, 4, 2).astype(np.int32)
    words = words[..., 0] | words[..., 1] << 8
    fields = np.stack([(words >> (3 * f)) & 7 for f in range(4)], -1)
    assert set(np.unique(fields)) == set(range(8))

    blocks, _ = _fixture("ggml_iq4_xs")
    scales_h = blocks[:, 2].astype(np.int32) | blocks[:, 3].astype(np.int32) << 8
    ib = np.arange(8)
    low = (blocks[:, 4 + ib // 2] >> (4 * (ib % 2))) & 0xF
    ls = low | (((scales_h[:, None] >> (2 * ib)) & 3) << 4)
    assert set(np.unique(ls)) == set(range(64))

    blocks, _ = _fixture("ggml_q6_k")
    scales = blocks[:, 192:208].view(np.int8)
    assert scales.min() == -128 and scales.max() == 127


@pytest.mark.skipif(not ORACLE, reason="set NINFER_GGML_BASE_DLL to ggml-base of llama.cpp b11316")
@pytest.mark.parametrize("name", FORMATS)
def test_decoder_matches_live_oracle_on_random_blocks(name):
    from tools.artifact.gen_ggml_fixtures import Oracle

    oracle = Oracle(ORACLE)
    spec = get_format(name)
    assert oracle.geometry(spec) == (spec.block_elems, spec.block_bytes)
    rng = np.random.default_rng(int.from_bytes(name.encode(), "little") % 2**32)
    blocks = rng.integers(0, 256, size=(100_000, spec.block_bytes), dtype=np.uint8)
    expected = oracle.decode(spec, blocks)
    decoded = decode_blocks(name, blocks).view(np.uint32).reshape(expected.shape)
    assert np.array_equal(decoded, expected)


def test_block_geometry_and_rejections():
    assert ggml_blocks_geometry("ggml_iq2_s", (512, 640, 2560)).payload_bytes == 268_697_600
    assert ggml_blocks_geometry("ggml_q2_0", (512, 2560, 640)).row_bytes == 180
    assert encoded_size("ggml_blocks_v1", "ggml_iq4_xs", (248_320, 2560)) == 337_715_200
    assert encoded_size("ggml_blocks_v1", "ggml_q8_0", (64,)) == 68
    with pytest.raises(ValueError, match="divisible by 256"):
        encoded_size("ggml_blocks_v1", "ggml_iq1_m", (4, 640))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_split_k128_v1", "ggml_q8_0", (4, 64))
    with pytest.raises(ValueError, match="does not accept format 'bf16'"):
        encoded_size("ggml_blocks_v1", "bf16", (4, 64))
    with pytest.raises(ValueError, match="GGML block format"):
        ggml_blocks_geometry("bf16", (4, 64))
    with pytest.raises(ValueError, match="whole number"):
        decode_blocks("ggml_q2_0", bytes(17))
    with pytest.raises(ValueError, match="expected 36"):
        decode_ggml_blocks(bytes(34), "ggml_q2_0", (2, 64))


def test_row_pages_hold_whole_rows_and_invert_exactly():
    # The PLE table: IQ4_NL rows of 160 values are 90 bytes, 45 per 4096-byte page.
    geometry = ggml_row_page_geometry("ggml_iq4_nl", (320_001_536, 160))
    assert (geometry.row_bytes, geometry.rows_per_page) == (90, 45)
    assert geometry.pages == 7_111_146 and geometry.payload_bytes == 29_127_254_016
    assert geometry.row_offset(0) == 0 and geometry.row_offset(44) == 44 * 90
    assert geometry.row_offset(45) == 4096 and geometry.row_offset(1000) == 22 * 4096 + 10 * 90

    shape = (100, 160)
    rows = np.random.default_rng(3).integers(0, 256, size=(100, 90), dtype=np.uint8)
    payload = bytearray(ggml_row_page_geometry("ggml_iq4_nl", shape).payload_bytes)
    for begin in (0, 45, 90):  # whole pages, then the final partial page
        end = min(begin + 45, 100)
        offset, pages = page_rows(rows[begin:end], "ggml_iq4_nl", shape, begin)
        payload[offset : offset + pages.size] = pages.tobytes()
    assert len(payload) == 3 * 4096
    for row in (0, 44, 45, 99):
        at = ggml_row_page_geometry("ggml_iq4_nl", shape).row_offset(row)
        assert bytes(payload[at : at + 90]) == rows[row].tobytes()
    assert np.array_equal(unpage_rows(bytes(payload), "ggml_iq4_nl", shape), rows)

    with pytest.raises(ValueError, match="start a page"):
        page_rows(rows[1:46], "ggml_iq4_nl", shape, 1)
    with pytest.raises(ValueError, match="whole pages"):
        page_rows(rows[:10], "ggml_iq4_nl", shape, 0)
    dirty = bytearray(payload)
    dirty[4095] = 1
    with pytest.raises(ValueError, match="tails must be zero"):
        unpage_rows(bytes(dirty), "ggml_iq4_nl", shape)
    with pytest.raises(ValueError, match="at most 4096"):
        ggml_row_page_geometry("ggml_q8_0", (4, 32 * 121))
