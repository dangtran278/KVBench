import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class AffineQuantParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int


def _dtype_nbits(bits: int) -> torch.dtype:
    # storage dtype (not packed yet)
    if bits <= 8:
        return torch.uint8
    if bits <= 16:
        return torch.int16
    return torch.int32


def affine_quantize_per_group_last_dim(
    x: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, AffineQuantParams]:
    """Quantize along last dim in contiguous groups.

    x: (..., d)
    returns:
      q: (..., d) integer tensor (unpacked)
      params: scale/zero_point with shape (..., n_groups, 1)
    """
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"expected float x, got {x.dtype}")
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by group_size {group_size}")

    qmin = 0
    qmax = (1 << bits) - 1

    d = x.shape[-1]
    ng = d // group_size
    xg = x.reshape(*x.shape[:-1], ng, group_size)
    xmin = xg.amin(dim=-1, keepdim=True)
    xmax = xg.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
    zero_point = (qmin - xmin / scale).round().clamp(qmin, qmax)

    q = (xg / scale + zero_point).round().clamp(qmin, qmax)
    q = q.to(_dtype_nbits(bits)).reshape_as(x)
    params = AffineQuantParams(scale=scale, zero_point=zero_point, qmin=qmin, qmax=qmax)
    return q, params


def affine_dequantize_per_group_last_dim(
    q: torch.Tensor,
    params: AffineQuantParams,
    group_size: int,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    d = q.shape[-1]
    ng = d // group_size
    qg = q.reshape(*q.shape[:-1], ng, group_size).to(torch.float32)
    x = (qg - params.zero_point.to(torch.float32)) * params.scale.to(torch.float32)
    return x.reshape_as(q).to(out_dtype)


def build_nf_lut(bits: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """NormalFloat-like signposts in [-1, 1], size 2**bits.

    Approximate evenly spaced percentiles of N(0,1),
    then normalized separately on negative and positive halves and merged.
    """
    n = 2**bits
    if n < 2:
        raise ValueError("bits too small")

    # Use symmetric percentiles excluding extremes.
    # We create n points via inverse CDF at evenly spaced probabilities.
    # Then normalize to [-1,1].
    ps = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, device=device, dtype=torch.float32)
    # erfinv-based inverse CDF
    z = math.sqrt(2.0) * torch.erfinv(2 * ps - 1)
    z = z / z.abs().max().clamp_min(1e-6)
    return z.to(dtype)


def quantize_to_lut(
    x: torch.Tensor,
    lut: torch.Tensor,
    *,
    xmin: Optional[torch.Tensor] = None,
    xmax: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize x to indices into lut after affine normalizing to [-1,1].

    Returns (codes, scale, offset) such that:
      x_hat ~= lut[codes] * scale + offset
    """
    x_f = x.to(torch.float32)
    if xmin is None:
        xmin = x_f.amin(dim=-1, keepdim=True)
    if xmax is None:
        xmax = x_f.amax(dim=-1, keepdim=True)
    offset = (xmax + xmin) / 2.0
    scale = ((xmax - xmin) / 2.0).clamp_min(1e-6)
    xn = (x_f - offset) / scale

    # compute nearest lut index
    # xn: (..., d), lut: (L,)
    # broadcasting: (..., d, L)
    diffs = (xn.unsqueeze(-1) - lut.view(*([1] * xn.ndim), -1)).abs()
    codes = diffs.argmin(dim=-1).to(torch.uint8)
    return codes, scale.to(x.dtype), offset.to(x.dtype)


def dequantize_from_lut(
    codes: torch.Tensor,
    lut: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    x = lut[codes.to(torch.long)].to(torch.float32)
    x = x * scale.to(torch.float32) + offset.to(torch.float32)
    return x.to(out_dtype)

