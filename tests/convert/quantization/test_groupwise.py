from __future__ import annotations

import torch

from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)
from tools.artifact.formats import get_format
from tools.convert.quantization.groupwise import quantize_matrix, search_quantize_matrix


def test_quantization_uses_stored_fp16_scale_and_zero_padding() -> None:
    weight = torch.tensor(
        [[-7.0, -1.0, 0.0, 1.0, 7.0, 3.5, -3.5, 0.25] + [0.0] * 57],
        dtype=torch.bfloat16,
    )
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert quantized.codes.shape == (1, 2, 64)
    assert quantized.scales.shape == (1, 2)
    expected_codes = torch.zeros((1, 2, 64), dtype=torch.int8)
    expected_codes[0, 0, :8] = torch.tensor([-7, -1, 0, 1, 7, 4, -4, 0])
    assert torch.equal(quantized.codes, expected_codes)
    assert quantized.scales.tolist() == [[1.0, 0.0]]
    assert torch.count_nonzero(quantized.codes[0, 1]) == 0
    assert quantized.scales[0, 1] == 0

    payload = encode_row_split(
        quantized.codes, quantized.scales, "q4_g64_fp16", weight.shape
    )
    scales, codes = decode_row_split_codes(payload, "q4_g64_fp16", tuple(weight.shape))
    assert torch.equal(scales, quantized.scales)
    assert torch.equal(codes, quantized.codes)
    decoded = dequantize_row_split(
        payload, "q4_g64_fp16", tuple(weight.shape), dtype=torch.float32
    )
    expected = expected_codes.float().reshape(1, 128)[:, :65]
    assert torch.equal(decoded, expected)


def test_quantization_uses_canonical_scale_rounding_on_cuda() -> None:
    # BF16 word 0x3636 divided by Q4 qmax is a known CUDA division boundary:
    # approximate device division rounds the FP16 scale to word 7, while the
    # ordered RNE oracle requires word 6.
    value = torch.tensor([0x3636], dtype=torch.int16).view(torch.bfloat16)
    weight = value.repeat(64).reshape(1, 64)
    cpu = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(cpu.scales.view(torch.int16)[0, 0]) == 6
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "q4_g64_fp16", device="cuda")
        assert torch.equal(
            cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16)
        )
        assert torch.equal(cuda.codes.cpu(), cpu.codes)


def test_quantization_uses_reciprocal_multiply_and_ties_to_even() -> None:
    words = torch.tensor([0x41A0B334, 0x417C7BFF], dtype=torch.int32).view(
        torch.float32
    )
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, :2] = words
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 0x41BD
    assert quantized.codes[0, 0, :2].tolist() == [7, 6]
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "q4_g64_fp16", device="cuda")
        assert torch.equal(cuda.scales.cpu(), quantized.scales)
        assert torch.equal(cuda.codes.cpu(), quantized.codes)

    tie_weight = torch.zeros((1, 64), dtype=torch.float32)
    tie_weight[0, :7] = torch.tensor([7.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    ties = quantize_matrix(tie_weight, "q4_g64_fp16", device="cpu")
    assert ties.codes[0, 0, :7].tolist() == [7, 0, 2, 2, 0, -2, -2]


def test_nonzero_scale_underflow_uses_smallest_fp16_subnormal() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = torch.finfo(torch.float32).tiny
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 1


def _weighted_error(weight, quantized, importance, group_size):
    rows, groups, _ = quantized.codes.shape
    decoded = quantized.scales.float().unsqueeze(-1) * quantized.codes.float()
    padded = torch.zeros((rows, groups * group_size))
    padded[:, : weight.shape[1]] = weight.float()
    residual = padded.reshape(rows, groups, group_size) - decoded
    weights = torch.zeros(groups * group_size)
    weights[: weight.shape[1]] = importance
    return (weights.reshape(1, groups, group_size) * residual**2).sum(dim=2)


def test_search_is_never_worse_than_round_to_nearest() -> None:
    generator = torch.Generator().manual_seed(7)
    for format, group_size in (("q4_g64_fp16", 64), ("q5_g64_fp16", 64), ("q8_g32_fp16", 32)):
        weight = torch.randn((48, 200), generator=generator) * 0.02
        weight[3, 5] = 0.4  # one outlier group
        importance = torch.rand(200, generator=generator) * 3
        rtn = quantize_matrix(weight, format, device="cpu")
        for negative in (False, True):
            searched = search_quantize_matrix(
                weight, format, importance=importance, negative_scales=negative,
                device="cpu",
            )
            spec = get_format(format)
            assert int(searched.codes.min()) >= spec.qmin
            assert int(searched.codes.max()) <= spec.qmax
            assert bool((searched.scales.float() >= 0).all()) or negative
            before = _weighted_error(weight, rtn, importance, group_size)
            after = _weighted_error(weight, searched, importance, group_size)
            assert bool((after <= before).all())
            assert float(after.sum()) < float(before.sum())


def test_search_uses_full_range_code_with_negative_scale() -> None:
    # Positive extreme 8.0: the full-range scale -1.0 maps it to code -8 exactly, and every
    # other value is an exact multiple, so the negative scale is error free.
    weight = torch.zeros((1, 64))
    weight[0, :4] = torch.tensor([8.0, -3.0, 5.0, 1.0])
    positive = search_quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    full = search_quantize_matrix(
        weight, "q4_g64_fp16", negative_scales=True, device="cpu"
    )
    assert float(positive.scales[0, 0]) > 0
    assert float(full.scales[0, 0]) == -1.0
    assert full.codes[0, 0, :4].tolist() == [-8, 3, -5, -1]


def test_search_keeps_zero_groups_canonical() -> None:
    weight = torch.zeros((2, 64))
    weight[1, 0] = 1.0
    searched = search_quantize_matrix(
        weight, "q4_g64_fp16", negative_scales=True, device="cpu"
    )
    assert float(searched.scales[0, 0]) == 0
    assert torch.count_nonzero(searched.codes[0]) == 0
