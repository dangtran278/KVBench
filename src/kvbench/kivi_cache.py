from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .quant_utils import (
    AffineQuantParams,
    affine_dequantize_per_group_last_dim,
    affine_quantize_per_group_last_dim,
)


@dataclass
class KiviCacheState:
    # Quantized long-term storage
    k_q: Optional[torch.Tensor] = None  # (b, kvh, t, d) uint8/int
    k_params: Optional[AffineQuantParams] = None
    v_q: Optional[torch.Tensor] = None  # (b, kvh, t, d)
    v_params: Optional[AffineQuantParams] = None

    # Full-precision residual window
    k_fp: Optional[torch.Tensor] = None  # (b, kvh, t_fp, d)
    v_fp: Optional[torch.Tensor] = None  # (b, kvh, t_fp, d)

    # Total tokens seen (for bookkeeping)
    total_len: int = 0


class KiviCache:
    """KIVI-style KV-cache quantization.

    - Store most tokens quantized with per-group affine params
    - Keep a fp16 residual window of the most recent `residual_length` tokens
    """

    def __init__(self, *, k_bits: int, v_bits: int, group_size: int, residual_length: int):
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.group_size = int(group_size)
        self.residual_length = int(residual_length)

    def init_state(self) -> KiviCacheState:
        return KiviCacheState()

    def _flush_if_full(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> None:
        if state.k_fp is None or state.v_fp is None:
            return
        t_fp = state.k_fp.shape[-2]
        if t_fp < self.residual_length:
            return

        # Flush entire residual window into quant storage, then clear fp window.
        k_flush = state.k_fp
        v_flush = state.v_fp

        # Quantize along head_dim (last dim), per-group.
        k_q_new, k_p_new = affine_quantize_per_group_last_dim(
            k_flush, bits=self.k_bits, group_size=self.group_size
        )
        v_q_new, v_p_new = affine_quantize_per_group_last_dim(
            v_flush, bits=self.v_bits, group_size=self.group_size
        )

        if state.k_q is None:
            state.k_q, state.k_params = k_q_new, k_p_new
            state.v_q, state.v_params = v_q_new, v_p_new
        else:
            state.k_q = torch.cat([state.k_q, k_q_new], dim=-2)
            state.v_q = torch.cat([state.v_q, v_q_new], dim=-2)
            # params concat along token axis (group params are per token group)
            assert state.k_params is not None and state.v_params is not None
            state.k_params.scale = torch.cat([state.k_params.scale, k_p_new.scale], dim=-2)
            state.k_params.zero_point = torch.cat([state.k_params.zero_point, k_p_new.zero_point], dim=-2)
            state.v_params.scale = torch.cat([state.v_params.scale, v_p_new.scale], dim=-2)
            state.v_params.zero_point = torch.cat([state.v_params.zero_point, v_p_new.zero_point], dim=-2)

        state.k_fp = None
        state.v_fp = None

    def append(self, state: KiviCacheState, k: torch.Tensor, v: torch.Tensor) -> KiviCacheState:
        """Append new kv for current step.

        k, v: (b, kvh, t_new, d)
        """
        if state.k_fp is None:
            state.k_fp = k
            state.v_fp = v
        else:
            state.k_fp = torch.cat([state.k_fp, k], dim=-2)
            state.v_fp = torch.cat([state.v_fp, v], dim=-2)

        state.total_len += k.shape[-2]
        self._flush_if_full(state, out_dtype=k.dtype)
        return state

    def materialize(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full (K,V) in out_dtype for attention.

        Shapes:
          K: (b, kvh, t_total, d)
          V: (b, kvh, t_total, d)
        """
        parts_k = []
        parts_v = []
        if state.k_q is not None:
            assert state.k_params is not None and state.v_q is not None and state.v_params is not None
            k_deq = affine_dequantize_per_group_last_dim(
                state.k_q, state.k_params, self.group_size, out_dtype=out_dtype
            )
            v_deq = affine_dequantize_per_group_last_dim(
                state.v_q, state.v_params, self.group_size, out_dtype=out_dtype
            )
            parts_k.append(k_deq)
            parts_v.append(v_deq)
        if state.k_fp is not None:
            parts_k.append(state.k_fp.to(out_dtype))
            parts_v.append(state.v_fp.to(out_dtype))
        if not parts_k:
            raise RuntimeError("cache is empty")
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

