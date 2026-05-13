from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch

from .quant_utils import (
    AffineQuantParams,
    affine_dequantize_per_group_last_dim,
    affine_dequantize_per_group_token_dim,
    affine_quantize_per_group_last_dim,
    affine_quantize_per_group_token_dim,
)

ALLOWED_KIVI_BITS = (2, 4, 8)


def validate_kivi_bits(k_bits: int, v_bits: int) -> None:
    k_bits = int(k_bits)
    v_bits = int(v_bits)
    if k_bits not in ALLOWED_KIVI_BITS or v_bits not in ALLOWED_KIVI_BITS:
        raise ValueError(
            "KIVI only supports k_bits/v_bits in {2, 4, 8}; "
            f"got k_bits={k_bits}, v_bits={v_bits}"
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
    # Cached dequantized quantized-prefix for official_like mode.
    cache_k_deq_prefix: Optional[torch.Tensor] = None
    cache_v_deq_prefix: Optional[torch.Tensor] = None
    cache_q_len: int = 0
    cache_dtype: Optional[torch.dtype] = None
    # Counters and diagnostics metadata.
    quant_append_count: int = 0
    total_flushed_tokens: int = 0
    last_flush_len: int = 0
    # Lightweight telemetry buffer (optional consumers can read/reset externally).
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class KiviCache:
    """KIVI-style KV-cache quantization.

    - Store most tokens quantized with per-group affine params
    - Keep a fp16 residual window of the most recent `residual_length` tokens

    Quantization axes (paper): K groups along *token* dim (stats across tokens per
    channel); V groups along *head_dim* (stats across channels per token). See
    _flush_if_full and materialize — K must use affine_*_token_dim, V uses
    affine_*_per_group_last_dim on (..., t, d).
    """

    def __init__(
        self,
        *,
        k_bits: int,
        v_bits: int,
        group_size: int,
        residual_length: int,
        k_residual_length: Optional[int] = None,
        v_residual_length: Optional[int] = None,
        kivi_mode: str = "legacy",
        diagnostics: bool = False,
        telemetry_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        validate_kivi_bits(self.k_bits, self.v_bits)
        self.group_size = int(group_size)
        self.residual_length = int(residual_length)
        self.k_residual_length = int(k_residual_length) if k_residual_length is not None else self.residual_length
        self.v_residual_length = int(v_residual_length) if v_residual_length is not None else self.residual_length
        self.kivi_mode = str(kivi_mode)
        self.diagnostics = bool(diagnostics)
        self.telemetry_callback = telemetry_callback
        if self.kivi_mode not in ("legacy", "official_like"):
            raise ValueError(f"unknown kivi_mode={self.kivi_mode}")
        if self.residual_length % self.group_size != 0:
            raise ValueError(
                f"residual_length {self.residual_length} must be divisible by group_size {self.group_size} "
                "for strict queue flushing on token axis"
            )
        if self.k_residual_length % self.group_size != 0:
            raise ValueError(
                f"k_residual_length {self.k_residual_length} must be divisible by group_size {self.group_size}"
            )
        if self.v_residual_length <= 0:
            raise ValueError("v_residual_length must be positive")

    def init_state(self) -> KiviCacheState:
        return KiviCacheState()

    def _invalidate_prefix_cache(self, state: KiviCacheState) -> None:
        state.cache_k_deq_prefix = None
        state.cache_v_deq_prefix = None
        state.cache_q_len = 0
        state.cache_dtype = None

    def _emit_telemetry(self, state: KiviCacheState, payload: dict[str, Any]) -> None:
        enriched = {
            "total_len": int(state.total_len),
            "k_q_len": int(state.k_q.shape[-2]) if state.k_q is not None else 0,
            "v_q_len": int(state.v_q.shape[-2]) if state.v_q is not None else 0,
            "k_fp_len": int(state.k_fp.shape[-2]) if state.k_fp is not None else 0,
            "v_fp_len": int(state.v_fp.shape[-2]) if state.v_fp is not None else 0,
            **payload,
        }
        state.telemetry.append(enriched)
        if self.telemetry_callback is not None:
            self.telemetry_callback(enriched)

    def _record_parity_warning(self, state: KiviCacheState, kind: str, details: dict[str, Any]) -> None:
        self._emit_telemetry(state, {"event": "parity_warning", "kind": kind, **details})

    def _quant_error_stats(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        params: AffineQuantParams,
        *,
        token_axis: bool,
    ) -> dict[str, float]:
        if token_axis:
            deq = affine_dequantize_per_group_token_dim(q, params, self.group_size, out_dtype=x.dtype)
        else:
            deq = affine_dequantize_per_group_last_dim(q, params, self.group_size, out_dtype=x.dtype)
        err = (deq.to(torch.float32) - x.to(torch.float32)).abs()
        return {"mean_abs_err": float(err.mean().item()), "max_abs_err": float(err.max().item())}

    def _append_quantized(self, state: KiviCacheState, k_flush: torch.Tensor, v_flush: torch.Tensor) -> None:
        k_q_new, k_p_new = affine_quantize_per_group_token_dim(
            k_flush, bits=self.k_bits, group_size=self.group_size
        )
        v_q_new, v_p_new = affine_quantize_per_group_last_dim(
            v_flush, bits=self.v_bits, group_size=self.group_size
        )
        if self.diagnostics:
            k_stats = self._quant_error_stats(k_flush, k_q_new, k_p_new, token_axis=True)
            v_stats = self._quant_error_stats(v_flush, v_q_new, v_p_new, token_axis=False)
        else:
            k_stats = {}
            v_stats = {}
        if state.k_q is None:
            state.k_q, state.k_params = k_q_new, k_p_new
            state.v_q, state.v_params = v_q_new, v_p_new
        else:
            state.k_q = torch.cat([state.k_q, k_q_new], dim=-2)
            state.v_q = torch.cat([state.v_q, v_q_new], dim=-2)
            assert state.k_params is not None and state.v_params is not None
            state.k_params.scale = torch.cat([state.k_params.scale, k_p_new.scale], dim=-2)
            state.k_params.zero_point = torch.cat([state.k_params.zero_point, k_p_new.zero_point], dim=-2)
            state.v_params.scale = torch.cat([state.v_params.scale, v_p_new.scale], dim=-3)
            state.v_params.zero_point = torch.cat([state.v_params.zero_point, v_p_new.zero_point], dim=-3)
        state.quant_append_count += 1
        state.total_flushed_tokens += int(k_flush.shape[-2])
        state.last_flush_len = int(k_flush.shape[-2])
        self._invalidate_prefix_cache(state)
        self._emit_telemetry(
            state,
            {
                "event": "quant_append",
                "flush_len": int(k_flush.shape[-2]),
                "k_mean_abs_err": k_stats.get("mean_abs_err"),
                "k_max_abs_err": k_stats.get("max_abs_err"),
                "v_mean_abs_err": v_stats.get("mean_abs_err"),
                "v_max_abs_err": v_stats.get("max_abs_err"),
            },
        )

    def _prefill_partition_official_like(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        t = int(k.shape[-2])
        # Official-like target:
        #   K: quantized prefix in residual blocks + fp remainder
        #   V: quantized prefix [:-R_v] + fp tail [-R_v:]
        k_quant_len = (t // self.k_residual_length) * self.k_residual_length
        v_quant_len = max(0, t - self.v_residual_length)
        # To keep K/V quantized prefix aligned in this cache format, use common prefix.
        common_quant_len = min(k_quant_len, v_quant_len)
        if common_quant_len > 0:
            common_quant_len = (common_quant_len // self.group_size) * self.group_size
        k_q = k[..., :common_quant_len, :] if common_quant_len > 0 else None
        k_fp = k[..., common_quant_len:, :]
        v_q = v[..., :common_quant_len, :] if common_quant_len > 0 else None
        v_fp = v[..., common_quant_len:, :]
        return k_q, k_fp, v_q, v_fp

    def _flush_if_full_legacy(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> None:
        if state.k_fp is None or state.v_fp is None:
            return
        # Flush in R-token blocks until exactly R tokens remain in the FP16 residual.
        # Uses strict > (not >=) so that a full R-token window is always preserved.
        while state.k_fp is not None and state.k_fp.shape[-2] > self.k_residual_length:
            k_flush = state.k_fp[..., : self.k_residual_length, :]
            v_flush = state.v_fp[..., : self.k_residual_length, :]
            state.k_fp = state.k_fp[..., self.k_residual_length :, :]
            state.v_fp = state.v_fp[..., self.k_residual_length :, :]
            if state.k_fp.shape[-2] == 0:
                state.k_fp = None
                state.v_fp = None

            self._append_quantized(state, k_flush, v_flush)

    def _flush_if_full_official_like(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> None:
        if state.k_fp is None or state.v_fp is None:
            return
        # Keep sliding fp tails while respecting group constraints.
        # For asymmetric residual settings, enforce both bounds by flushing enough
        # grouped tokens so neither K nor V tail drifts unbounded.
        while state.k_fp is not None and state.v_fp is not None:
            k_len = int(state.k_fp.shape[-2])
            v_len = int(state.v_fp.shape[-2])
            if k_len != v_len:
                raise RuntimeError(f"K/V fp tail length mismatch: k_len={k_len} v_len={v_len}")
            if k_len <= self.k_residual_length and v_len <= self.v_residual_length:
                break
            overflow_k = max(0, k_len - self.k_residual_length)
            overflow_v = max(0, v_len - self.v_residual_length)
            # Flush enough to satisfy the tighter side; group alignment is applied below.
            overflow = max(overflow_k, overflow_v)
            flush_len = (overflow // self.group_size) * self.group_size
            if flush_len <= 0:
                # If either side violates bound but cannot flush due to group granularity,
                # keep warning telemetry and stop; next append may make it flushable.
                if overflow_k > 0:
                    self._record_parity_warning(
                        state,
                        "k_tail_pending_group_flush",
                        {"k_len": k_len, "k_residual_length": self.k_residual_length, "overflow_k": overflow_k},
                    )
                if overflow_v > 0:
                    self._record_parity_warning(
                        state,
                        "v_tail_pending_group_flush",
                        {"v_len": v_len, "v_residual_length": self.v_residual_length, "overflow_v": overflow_v},
                    )
                break

            k_flush = state.k_fp[..., :flush_len, :]
            v_flush = state.v_fp[..., :flush_len, :]
            state.k_fp = state.k_fp[..., flush_len:, :]
            state.v_fp = state.v_fp[..., flush_len:, :]
            if state.k_fp.shape[-2] == 0:
                state.k_fp = None
            if state.v_fp.shape[-2] == 0:
                state.v_fp = None
            self._append_quantized(state, k_flush, v_flush)

        if state.k_fp is not None and state.k_fp.shape[-2] > (self.k_residual_length + self.group_size - 1):
            self._record_parity_warning(
                state,
                "k_tail_bound",
                {
                    "k_fp_len": int(state.k_fp.shape[-2]),
                    "k_tail_bound": int(self.k_residual_length + self.group_size - 1),
                },
            )
        if state.v_fp is not None and state.v_fp.shape[-2] > (self.v_residual_length + self.group_size - 1):
            self._record_parity_warning(
                state,
                "v_tail_bound",
                {
                    "v_fp_len": int(state.v_fp.shape[-2]),
                    "v_tail_bound": int(self.v_residual_length + self.group_size - 1),
                },
            )

    def _flush_if_full(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> None:
        if self.kivi_mode == "legacy":
            self._flush_if_full_legacy(state, out_dtype=out_dtype)
        else:
            self._flush_if_full_official_like(state, out_dtype=out_dtype)

    def append(self, state: KiviCacheState, k: torch.Tensor, v: torch.Tensor) -> KiviCacheState:
        """Append new kv for current step.

        k, v: (b, kvh, t_new, d)
        """
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"expected 4D KV tensors, got k={tuple(k.shape)} v={tuple(v.shape)}")
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}")
        if k.shape[-2] <= 0:
            raise ValueError("append expects at least one token on KV token axis")
        if state.k_fp is None:
            state.k_fp = k
            state.v_fp = v
        else:
            state.k_fp = torch.cat([state.k_fp, k], dim=-2)
            state.v_fp = torch.cat([state.v_fp, v], dim=-2)

        state.total_len += k.shape[-2]
        self._flush_if_full(state, out_dtype=k.dtype)
        if state.k_q is not None and state.v_q is not None and int(state.k_q.shape[-2]) != int(state.v_q.shape[-2]):
            self._record_parity_warning(
                state,
                "kv_q_len_mismatch",
                {"k_q_len": int(state.k_q.shape[-2]), "v_q_len": int(state.v_q.shape[-2])},
            )
        if state.k_fp is not None and state.v_fp is not None and int(state.k_fp.shape[-2]) != int(state.v_fp.shape[-2]):
            self._record_parity_warning(
                state,
                "kv_fp_len_mismatch",
                {"k_fp_len": int(state.k_fp.shape[-2]), "v_fp_len": int(state.v_fp.shape[-2])},
            )
        return state

    def append_prefill_storage(self, state: KiviCacheState, k: torch.Tensor, v: torch.Tensor) -> KiviCacheState:
        """Storage-only append used after fp16 prefill compute."""
        if self.kivi_mode == "official_like":
            k_q, k_fp, v_q, v_fp = self._prefill_partition_official_like(k, v)
            prefill_quant_len = int(k_q.shape[-2]) if k_q is not None else 0
            if k_q is not None and v_q is not None:
                self._append_quantized(state, k_q, v_q)
            state.k_fp = k_fp if state.k_fp is None else torch.cat([state.k_fp, k_fp], dim=-2)
            state.v_fp = v_fp if state.v_fp is None else torch.cat([state.v_fp, v_fp], dim=-2)
            state.total_len += int(k.shape[-2])
            self._emit_telemetry(
                state,
                {
                    "event": "prefill_partition",
                    "prefill_len": int(k.shape[-2]),
                    "prefill_quant_len": prefill_quant_len,
                    "prefill_fp_len": int(k_fp.shape[-2]),
                },
            )
            if state.k_q is not None and state.v_q is not None and int(state.k_q.shape[-2]) != int(state.v_q.shape[-2]):
                self._record_parity_warning(
                    state,
                    "kv_q_len_mismatch_after_prefill",
                    {"k_q_len": int(state.k_q.shape[-2]), "v_q_len": int(state.v_q.shape[-2])},
                )
            if state.k_fp is not None and state.v_fp is not None and int(state.k_fp.shape[-2]) != int(state.v_fp.shape[-2]):
                self._record_parity_warning(
                    state,
                    "kv_fp_len_mismatch_after_prefill",
                    {"k_fp_len": int(state.k_fp.shape[-2]), "v_fp_len": int(state.v_fp.shape[-2])},
                )
            return state
        return self.append(state, k, v)

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
            q_len = int(state.k_q.shape[-2])
            can_reuse = (
                state.cache_k_deq_prefix is not None
                and state.cache_v_deq_prefix is not None
                and state.cache_q_len == q_len
                and state.cache_dtype == out_dtype
            )
            if can_reuse:
                k_deq = state.cache_k_deq_prefix
                v_deq = state.cache_v_deq_prefix
            else:
                # Must mirror _flush_if_full: K token-axis dequant, V last-dim dequant.
                k_deq = affine_dequantize_per_group_token_dim(
                    state.k_q, state.k_params, self.group_size, out_dtype=out_dtype
                )
                v_deq = affine_dequantize_per_group_last_dim(
                    state.v_q, state.v_params, self.group_size, out_dtype=out_dtype
                )
                if self.kivi_mode == "official_like":
                    state.cache_k_deq_prefix = k_deq
                    state.cache_v_deq_prefix = v_deq
                    state.cache_q_len = q_len
                    state.cache_dtype = out_dtype
            parts_k.append(k_deq)
            parts_v.append(v_deq)
        if state.k_fp is not None:
            parts_k.append(state.k_fp.to(out_dtype))
            parts_v.append(state.v_fp.to(out_dtype))
        if not parts_k:
            raise RuntimeError("cache is empty")
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

