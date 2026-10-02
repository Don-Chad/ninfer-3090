"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats; this
module only performs the registered numeric transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.artifact.layouts import (
    row_split_geometry,
)
from tools.artifact.formats import QuantFormat, get_format

_FP16_MIN_SUBNORMAL = 2.0**-24


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


def _canonical_scale_words(
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical binary16 scales and binary32 reciprocals on the host.

    CUDA division is not correctly rounded at every binary16 scale boundary.
    The host oracle performs the specified division in binary64, explicitly
    rounds through binary32 and binary16, then computes the reciprocal in the
    same way.  A binary32 input divided by these small integer denominators has
    enough binary64 precision for the final binary32 rounding to be exact.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    Scales are rounded to binary16 before codes are selected because those are
    the exact scales consumed after loading.  Padding values are zero and do
    not affect a partially populated final group.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    scales = host_scales.to(target)
    reciprocal = host_reciprocal.to(target)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)


# Scale multipliers tried around each base scale. The list contains 1.0, so the plain
# round-to-nearest result of ``quantize_matrix`` is always a candidate.
SEARCH_RATIOS = tuple(round(0.70 + 0.02 * step, 2) for step in range(25))


def _candidate_scale_words(
    base: torch.Tensor, ratio: float, nonzero: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round signed candidate scales ``base * ratio`` to binary16 on the host.

    Rounding follows ``_canonical_scale_words``: binary64 arithmetic, then binary32, then
    binary16. A nonzero group whose candidate underflows takes the smallest subnormal of the
    same sign; overflowing candidates become infinite and are rejected by the caller.
    """

    host = base.detach().cpu().numpy().astype(np.float32, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        scale = (host.astype(np.float64) * ratio).astype(np.float32).astype(np.float16)
    keep = nonzero.cpu().numpy()
    underflow = (scale == 0) & keep
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.where(
            host[underflow] < 0, -_FP16_MIN_SUBNORMAL, _FP16_MIN_SUBNORMAL
        ).astype(np.float16)
    reciprocal = np.zeros(host.shape, dtype=np.float32)
    usable = keep & np.isfinite(scale) & (scale != 0)
    reciprocal[usable] = (1.0 / scale[usable].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def search_quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    importance: torch.Tensor | None = None,
    negative_scales: bool = False,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Choose each group's binary16 scale by minimizing weighted squared error.

    Candidates are ``SEARCH_RATIOS`` times two bases: ``absmax / qmax`` (the
    ``quantize_matrix`` scale) and the full-range scale ``extreme / qmin``, which maps the
    group's largest-magnitude value to the most negative code. A full-range scale is negative
    when that value is positive; it is a candidate only when ``negative_scales`` is set.
    ``importance`` holds one nonnegative weight per input channel (length K), for example
    the mean squared activation; without it every channel weighs the same. The
    ``quantize_matrix`` result is the first candidate and only a strictly smaller error
    replaces it, so the weighted error is never worse than round-to-nearest.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    baseline = quantize_matrix(weight, spec, device=device)
    target = baseline.codes.device
    geometry = row_split_geometry(spec, weight.shape)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical
    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)

    if importance is None:
        weights = torch.ones(
            (1, geometry.groups_per_row, spec.group_size), device=target
        )
    else:
        if importance.dim() != 1 or importance.numel() != geometry.k:
            raise ValueError(
                f"importance must have one value per input channel ({geometry.k})"
            )
        values = importance.detach().to(device=target, dtype=torch.float32)
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("importance must be finite and nonnegative")
        padded = torch.zeros(geometry.k_pad, dtype=torch.float32, device=target)
        padded[: geometry.k] = values
        weights = padded.reshape(1, geometry.groups_per_row, spec.group_size)

    def error(scales: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        residual = grouped - scales.to(torch.float32).unsqueeze(-1) * codes.to(
            torch.float32
        )
        return (weights * residual * residual).sum(dim=2)

    best_codes = baseline.codes.clone()
    best_scales = baseline.scales.clone()
    best_error = error(best_scales, best_codes)

    magnitude = grouped.abs()
    max_abs, position = magnitude.max(dim=2)
    extreme = grouped.gather(2, position.unsqueeze(-1)).squeeze(-1)
    nonzero = max_abs > 0
    bases = [max_abs / float(spec.qmax), extreme / float(spec.qmin)]
    for base in bases:
        for ratio in SEARCH_RATIOS:
            host_scales, host_reciprocal = _candidate_scale_words(base, ratio, nonzero)
            scales = host_scales.to(target)
            reciprocal = host_reciprocal.to(target)
            valid = nonzero & torch.isfinite(scales.to(torch.float32)) & (scales != 0)
            if not negative_scales:
                valid &= scales > 0
            codes = torch.clamp(
                torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
            ).to(torch.int8)
            candidate_error = error(scales, codes)
            better = valid & (candidate_error < best_error)
            if bool(better.any()):
                best_error = torch.where(better, candidate_error, best_error)
                best_scales = torch.where(better, scales, best_scales)
                best_codes = torch.where(better.unsqueeze(-1), codes, best_codes)
    return QuantizedMatrix(codes=best_codes, scales=best_scales)
