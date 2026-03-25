from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .quant_utils import build_nf_lut, dequantize_from_lut, quantize_to_lut


@dataclass
class KvQuantLut:
    lut: torch.Tensor  # (L,)
    # Per-channel thresholds for outliers (flattened over heads*dim for K; tokenwise for V)
    thr_low: torch.Tensor
    thr_high: torch.Tensor


@dataclass
class KvQuantCacheState:
    # Quantized codes for long-term cache (unpacked codes, not bitpacked)
    k_codes: Optional[torch.Tensor] = None  # (b, h, t, d) uint8 indices
    k_scale: Optional[torch.Tensor] = None  # (b, h, t, 1)
    k_offset: Optional[torch.Tensor] = None  # (b, h, t, 1)
    v_codes: Optional[torch.Tensor] = None  # (b, h, t, d)
    v_scale: Optional[torch.Tensor] = None
    v_offset: Optional[torch.Tensor] = None

    # Sparse outlier storage (dense mask + values for simplicity)
    k_outlier_mask: Optional[torch.Tensor] = None  # (b, h, t, d) bool
    k_outliers: Optional[torch.Tensor] = None  # (b, h, t, d) fp16 (zeros where not outlier)
    v_outlier_mask: Optional[torch.Tensor] = None
    v_outliers: Optional[torch.Tensor] = None

    # Optional fp16 prefix cache (attention sink style)
    k_fp16_prefix: Optional[torch.Tensor] = None  # (b, h, t0, d)
    v_fp16_prefix: Optional[torch.Tensor] = None

    total_len: int = 0


class KvQuantCache:
    """KVQuant-style KV-cache quantization (NUQ/NF + sparse outliers).

    - Store codes (not packed bits) and dequantize on demand
    - Store outliers explicitly via a mask/value tensor
    """

    def __init__(
        self,
        *,
        bits: int,
        outlier_percent: float,
        first_few_fp16: int = 0,
        use_nf: bool = False,
    ):
        self.bits = int(bits)
        self.outlier_percent = float(outlier_percent)
        self.first_few_fp16 = int(first_few_fp16)
        self.use_nf = bool(use_nf)

        self._lut: Optional[torch.Tensor] = None
        self._k_thr_low: Optional[torch.Tensor] = None
        self._k_thr_high: Optional[torch.Tensor] = None
        self._v_thr_low: Optional[torch.Tensor] = None
        self._v_thr_high: Optional[torch.Tensor] = None

    def init_state(self) -> KvQuantCacheState:
        return KvQuantCacheState()

    def set_luts(self, *, lut: KvQuantLut, lut_v: Optional[KvQuantLut] = None) -> None:
        self._lut = lut.lut
        self._k_thr_low = lut.thr_low
        self._k_thr_high = lut.thr_high
        if lut_v is None:
            self._v_thr_low = lut.thr_low
            self._v_thr_high = lut.thr_high
        else:
            self._v_thr_low = lut_v.thr_low
            self._v_thr_high = lut_v.thr_high

    def _ensure_lut(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._lut is None:
            self._lut = build_nf_lut(self.bits, device=device, dtype=dtype if self.use_nf else torch.float16)
        return self._lut.to(device=device, dtype=dtype)

    def append(self, state: KvQuantCacheState, k: torch.Tensor, v: torch.Tensor) -> KvQuantCacheState:
        """Append new KV (already RoPE-applied for K if model uses RoPE).

        k, v: (b, h, t_new, d)
        """
        b, h, t_new, d = k.shape
        device = k.device
        dtype = k.dtype
        lut = self._ensure_lut(device, dtype=torch.float16)

        # fp16 prefix storage
        if state.total_len < self.first_few_fp16:
            keep = min(self.first_few_fp16 - state.total_len, t_new)
            if keep > 0:
                k_pref = k[:, :, :keep, :].contiguous()
                v_pref = v[:, :, :keep, :].contiguous()
                state.k_fp16_prefix = k_pref if state.k_fp16_prefix is None else torch.cat([state.k_fp16_prefix, k_pref], dim=-2)
                state.v_fp16_prefix = v_pref if state.v_fp16_prefix is None else torch.cat([state.v_fp16_prefix, v_pref], dim=-2)
                k = k[:, :, keep:, :]
                v = v[:, :, keep:, :]
                t_new = k.shape[-2]
                state.total_len += keep
                if t_new == 0:
                    return state

        # outlier thresholds (fallback: dynamic per append if thresholds absent)
        if self._k_thr_low is None or self._k_thr_high is None:
            # per-channel thresholds (flatten heads*dim)
            flat = k.reshape(b, h, t_new, d).reshape(b, h * d, t_new).transpose(1, 2)  # (b, t, h*d)
            # torch.quantile with a tensor of quantiles has shape-order differences across versions.
            # Use scalar quantiles for deterministic (b, t_new) outputs.
            thr_low = torch.quantile(
                flat.float(),
                self.outlier_percent / 2.0,
                dim=-1,
            )  # (b, t_new)
            thr_high = torch.quantile(
                flat.float(),
                1.0 - self.outlier_percent / 2.0,
                dim=-1,
            )  # (b, t_new)
            # Broadcast across heads and channel dim.
            thr_low = thr_low[:, None, :, None]  # (b, 1, t_new, 1)
            thr_high = thr_high[:, None, :, None]  # (b, 1, t_new, 1)
        else:
            # broadcast per-(h*d)
            thr_low = self._k_thr_low.to(device=device, dtype=dtype).view(1, h, 1, d)
            thr_high = self._k_thr_high.to(device=device, dtype=dtype).view(1, h, 1, d)

        k_out_mask = (k < thr_low) | (k > thr_high)
        k_outliers = torch.where(k_out_mask, k, torch.zeros_like(k))
        k_in = torch.where(k_out_mask, torch.zeros_like(k), k)

        # tokenwise thresholds for V if absent: use tokenwise quantiles
        if self._v_thr_low is None or self._v_thr_high is None:
            vf = v.float()
            thr_low_v = torch.quantile(vf, self.outlier_percent / 2, dim=-1, keepdim=True)
            thr_high_v = torch.quantile(vf, 1 - self.outlier_percent / 2, dim=-1, keepdim=True)
        else:
            thr_low_v = self._v_thr_low.to(device=device, dtype=dtype).view(1, h, 1, d)
            thr_high_v = self._v_thr_high.to(device=device, dtype=dtype).view(1, h, 1, d)

        v_out_mask = (v < thr_low_v) | (v > thr_high_v)
        v_outliers = torch.where(v_out_mask, v, torch.zeros_like(v))
        v_in = torch.where(v_out_mask, torch.zeros_like(v), v)

        # Quantize to LUT with affine normalization to [-1,1]
        k_codes, k_scale, k_offset = quantize_to_lut(k_in, lut)
        v_codes, v_scale, v_offset = quantize_to_lut(v_in, lut)

        def cat_or_set(a, bcat):
            return bcat if a is None else torch.cat([a, bcat], dim=-2)

        state.k_codes = cat_or_set(state.k_codes, k_codes)
        state.k_scale = cat_or_set(state.k_scale, k_scale)
        state.k_offset = cat_or_set(state.k_offset, k_offset)
        state.k_outlier_mask = cat_or_set(state.k_outlier_mask, k_out_mask)
        state.k_outliers = cat_or_set(state.k_outliers, k_outliers)

        state.v_codes = cat_or_set(state.v_codes, v_codes)
        state.v_scale = cat_or_set(state.v_scale, v_scale)
        state.v_offset = cat_or_set(state.v_offset, v_offset)
        state.v_outlier_mask = cat_or_set(state.v_outlier_mask, v_out_mask)
        state.v_outliers = cat_or_set(state.v_outliers, v_outliers)

        state.total_len += t_new
        return state

    def materialize(self, state: KvQuantCacheState, *, out_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.k_codes is None or state.v_codes is None:
            if state.k_fp16_prefix is None:
                raise RuntimeError("cache is empty")
            return state.k_fp16_prefix.to(out_dtype), state.v_fp16_prefix.to(out_dtype)

        device = state.k_codes.device
        lut = self._ensure_lut(device, dtype=torch.float16)
        k = dequantize_from_lut(state.k_codes, lut, state.k_scale, state.k_offset, out_dtype=out_dtype)
        v = dequantize_from_lut(state.v_codes, lut, state.v_scale, state.v_offset, out_dtype=out_dtype)

        if state.k_outliers is not None:
            k = k + state.k_outliers.to(out_dtype)
        if state.v_outliers is not None:
            v = v + state.v_outliers.to(out_dtype)

        if state.k_fp16_prefix is not None:
            k = torch.cat([state.k_fp16_prefix.to(out_dtype), k], dim=-2)
            v = torch.cat([state.v_fp16_prefix.to(out_dtype), v], dim=-2)
        return k, v
