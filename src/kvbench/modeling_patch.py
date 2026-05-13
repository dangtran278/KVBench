from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .fused_kivi import kivi_fused_qk_scores
from .fused_kvquant import kvquant_fused_qk_scores
from .kivi_cache import KiviCache, KiviCacheState, validate_kivi_bits
from .quant_utils import affine_dequantize_per_group_last_dim
from .kvquant_cache import KvQuantCache, KvQuantCacheState


@dataclass
class PatchedCacheState:
    per_layer: list[Any]


def reset_kvbench_state(model: nn.Module) -> None:
    """Reset internal KV cache states before a new prompt."""
    for m in model.modules():
        if hasattr(m, "reset_kvbench_state"):
            m.reset_kvbench_state()


def collect_kivi_telemetry(model: nn.Module, *, clear: bool = True) -> list[dict[str, Any]]:
    """Collect per-layer KIVI telemetry records from patched attention modules."""
    out: list[dict[str, Any]] = []
    for layer_idx, layer in enumerate(getattr(getattr(model, "model", None), "layers", []) or []):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not isinstance(getattr(attn, "cache_impl", None), KiviCache):
            continue
        state = getattr(attn, "_kvbench_state", None)
        if state is None or not hasattr(state, "telemetry"):
            continue
        records = list(getattr(state, "telemetry", []))
        for rec in records:
            out.append({"layer_idx": layer_idx, **rec})
        if clear:
            state.telemetry = []
    return out


def collect_kvquant_telemetry(model: nn.Module, *, clear: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for layer_idx, layer in enumerate(getattr(getattr(model, "model", None), "layers", []) or []):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not isinstance(getattr(attn, "cache_impl", None), KvQuantCache):
            continue
        state = getattr(attn, "_kvbench_state", None)
        if state is None or not hasattr(state, "telemetry"):
            continue
        records = list(getattr(state, "telemetry", []))
        for rec in records:
            out.append({"layer_idx": layer_idx, **rec})
        if clear:
            state.telemetry = []
    return out


class AttentionCacheAdapter(nn.Module):
    """Attention wrapper that uses KV-cache quantizer.

    This adapter intentionally avoids HF `past_key_values` bookkeeping for quantization.
    It keeps its own internal cache state and computes attention by materializing
    (dequantizing) KV as needed.
    """

    def __init__(
        self,
        attn: nn.Module,
        *,
        cache_impl: Any,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        enable_parity_checks: bool = False,
        drift_probe_interval: int = 0,
        enable_fused_kivi_qk: bool = False,
        fused_kivi_qk_backend: str = "auto",
        enable_fused_kvquant_qk: bool = False,
        fused_kvquant_qk_backend: str = "auto",
    ):
        super().__init__()
        self.attn = attn
        self.cache_impl = cache_impl
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(head_dim)
        self.enable_parity_checks = bool(enable_parity_checks)
        self.drift_probe_interval = int(drift_probe_interval)
        self.enable_fused_kivi_qk = bool(enable_fused_kivi_qk)
        self.fused_kivi_qk_backend = str(fused_kivi_qk_backend)
        self.enable_fused_kvquant_qk = bool(enable_fused_kvquant_qk)
        self.fused_kvquant_qk_backend = str(fused_kvquant_qk_backend)

        # Persist across decode steps.
        self._kvbench_state: Optional[Any] = None

    def reset_kvbench_state(self) -> None:
        self._kvbench_state = None

    def _get_rope(self, position_ids: Optional[torch.LongTensor], k_for_rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for RoPE based on the underlying attention module."""
        rotary_emb = getattr(self.attn, "rotary_emb", None)
        if rotary_emb is None:
            raise ValueError("Underlying attention missing rotary_emb.")

        # Llama-style rotary_emb typically accepts (x, position_ids)
        try:
            cos, sin = rotary_emb(k_for_rope, position_ids)
        except TypeError:
            # Some variants accept seq_len instead.
            try:
                cos, sin = rotary_emb(k_for_rope, seq_len=int(k_for_rope.shape[-2]))
            except TypeError:
                cos, sin = rotary_emb(k_for_rope)
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,  # ignored for quantization
        output_attentions: bool = False,
        use_cache: bool = False,  # ignored; cache is internal
        cache_position: Optional[torch.LongTensor] = None,  # ignored
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        is_kivi_cache = isinstance(self.cache_impl, KiviCache)
        is_kvquant_cache = isinstance(self.cache_impl, KvQuantCache)

        # Projections.
        query_states = self.attn.q_proj(hidden_states)
        key_states = self.attn.k_proj(hidden_states)
        value_states = self.attn.v_proj(hidden_states)

        # Shape to (b, h, t, d) and (b, hk, t, d).
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE on K (and Q) as in HF Llama/Mistral attention.
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self._get_rope(position_ids, value_states)

        # Import HF helpers allowed for correctness.
        mod = self.attn.__class__.__module__
        if "llama" in mod:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        elif "mistral" in mod:
            from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
        else:
            raise ValueError(f"Unsupported attention module for patching: {mod}")

        # apply_rotary_pos_emb for llama expects (q, k, cos, sin, position_ids) in some versions.
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        except TypeError:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Internal state.
        if self._kvbench_state is None:
            self._kvbench_state = self.cache_impl.init_state()
        state = self._kvbench_state
        # KIVI paper semantics:
        # - Prefill math (q_len > 1) uses exact fp tensors for attention.
        # - KV compression is storage-side and happens after prefill compute.
        used_fused_kivi_qk = False
        fused_kv_len = 0
        fused_logits_len = 0
        if is_kivi_cache and q_len > 1:
            has_past = int(getattr(state, "total_len", 0)) > 0
            if has_past:
                k_past, v_past = self.cache_impl.materialize(state, out_dtype=query_states.dtype)
                k_all = torch.cat([k_past, key_states.to(query_states.dtype)], dim=-2)
                v_all = torch.cat([v_past, value_states.to(query_states.dtype)], dim=-2)
            else:
                k_all = key_states.to(query_states.dtype)
                v_all = value_states.to(query_states.dtype)
            defer_append_until_after_attn = True
        else:
            has_past = int(getattr(state, "total_len", 0)) > 0
            # Decode-only partial fusion for KIVI:
            # - Fused path computes K-side logits (QK) on quantized K-prefix.
            # - V-side remains on existing dequant/materialize-style path.
            # This is intentionally not full upstream-equivalent fusion yet.
            if (
                is_kivi_cache
                and self.enable_fused_kivi_qk
                and q_len == 1
                and has_past
                and getattr(state, "k_q", None) is not None
                and getattr(state, "k_params", None) is not None
                and query_states.is_cuda
            ):
                try:
                    # Quantized K-prefix logits (past prefix only).
                    qk_quant, fused_backend = kivi_fused_qk_scores(
                        query_states,
                        state.k_q,
                        state.k_params.scale,
                        state.k_params.zero_point,
                        group_size=self.cache_impl.group_size,
                        num_kv_groups=self.num_kv_groups,
                        backend=self.fused_kivi_qk_backend,
                        return_backend=True,
                    )
                    qk_quant = qk_quant.to(query_states.dtype)

                    # FP K-tail logits (past tail only).
                    if state.k_fp is not None:
                        k_fp_rep = repeat_kv(state.k_fp.to(query_states.dtype), self.num_kv_groups)
                        qk_fp = torch.matmul(query_states, k_fp_rep.transpose(-2, -1))
                    else:
                        qk_fp = None

                    # Current-step K logits (must be included so logits length matches V length).
                    k_cur_rep = repeat_kv(key_states.to(query_states.dtype), self.num_kv_groups)
                    qk_cur = torch.matmul(query_states, k_cur_rep.transpose(-2, -1))

                    logits_parts = [qk_quant]
                    if qk_fp is not None:
                        logits_parts.append(qk_fp)
                    logits_parts.append(qk_cur)
                    attn_weights = torch.cat(logits_parts, dim=-1) / (self.head_dim**0.5)

                    # V path for partial fusion: dequantized prefix + fp tail + current V.
                    v_parts = []
                    if state.v_q is not None and state.v_params is not None:
                        v_q_deq = affine_dequantize_per_group_last_dim(
                            state.v_q,
                            state.v_params,
                            self.cache_impl.group_size,
                            out_dtype=query_states.dtype,
                        )
                        v_parts.append(v_q_deq)
                    if state.v_fp is not None:
                        v_parts.append(state.v_fp.to(query_states.dtype))
                    v_parts.append(value_states.to(query_states.dtype))
                    v_all = repeat_kv(torch.cat(v_parts, dim=-2), self.num_kv_groups)

                    # Guard against K/V length drift in fused path.
                    if int(attn_weights.shape[-1]) != int(v_all.shape[-2]):
                        raise RuntimeError(
                            "fused_kivi_qk length mismatch: "
                            f"logits_len={int(attn_weights.shape[-1])} v_len={int(v_all.shape[-2])}"
                        )

                    state = self.cache_impl.append(state, key_states, value_states)
                    self._kvbench_state = state
                    defer_append_until_after_attn = False
                    used_fused_kivi_qk = True
                    fused_kv_len = int(v_all.shape[-2])
                    fused_logits_len = int(attn_weights.shape[-1])
                    if hasattr(state, "telemetry"):
                        state.telemetry.append(
                            {
                                "event": "fused_kivi_qk",
                                "backend": fused_backend,
                                "qk_quant_len": int(qk_quant.shape[-1]),
                                "qk_total_len": fused_logits_len,
                            }
                        )
                except Exception:
                    used_fused_kivi_qk = False

            if (
                (not used_fused_kivi_qk)
                and is_kvquant_cache
                and self.enable_fused_kvquant_qk
                and q_len == 1
                and has_past
                and query_states.is_cuda
            ):
                try:
                    lut = self.cache_impl.get_triton_lut(query_states.device)
                    qk_past, fused_backend = kvquant_fused_qk_scores(
                        query_states,
                        getattr(state, "k_codes", None),
                        getattr(state, "k_scale", None),
                        getattr(state, "k_offset", None),
                        lut,
                        getattr(state, "k_outliers", None),
                        getattr(state, "k_fp16_prefix", None),
                        num_kv_groups=self.num_kv_groups,
                        backend=self.fused_kvquant_qk_backend,
                        return_backend=True,
                    )
                    k_cur_rep = repeat_kv(key_states.to(query_states.dtype), self.num_kv_groups)
                    qk_cur = torch.matmul(query_states, k_cur_rep.transpose(-2, -1))
                    attn_weights = torch.cat([qk_past.to(query_states.dtype), qk_cur], dim=-1) / (self.head_dim**0.5)

                    _, v_past = self.cache_impl.materialize(state, out_dtype=query_states.dtype)
                    v_all = repeat_kv(torch.cat([v_past, value_states.to(query_states.dtype)], dim=-2), self.num_kv_groups)
                    if int(attn_weights.shape[-1]) != int(v_all.shape[-2]):
                        raise RuntimeError(
                            "fused_kvquant_qk length mismatch: "
                            f"logits_len={int(attn_weights.shape[-1])} v_len={int(v_all.shape[-2])}"
                        )
                    state = self.cache_impl.append(state, key_states, value_states)
                    self._kvbench_state = state
                    defer_append_until_after_attn = False
                    used_fused_kivi_qk = True
                    fused_kv_len = int(v_all.shape[-2])
                    fused_logits_len = int(attn_weights.shape[-1])
                    if hasattr(state, "telemetry"):
                        state.telemetry.append(
                            {
                                "event": "fused_kvquant_qk",
                                "backend": fused_backend,
                                "qk_past_len": int(qk_past.shape[-1]),
                                "qk_total_len": fused_logits_len,
                            }
                        )
                except Exception:
                    used_fused_kivi_qk = False

            if (not used_fused_kivi_qk) and has_past:
                k_past, v_past = self.cache_impl.materialize(state, out_dtype=query_states.dtype)
                k_all = torch.cat([k_past, key_states.to(query_states.dtype)], dim=-2)
                v_all = torch.cat([v_past, value_states.to(query_states.dtype)], dim=-2)
            elif not used_fused_kivi_qk:
                k_all = key_states.to(query_states.dtype)
                v_all = value_states.to(query_states.dtype)
            if not used_fused_kivi_qk:
                # Decode path keeps current behavior: write new KV before returning.
                state = self.cache_impl.append(state, key_states, value_states)
                self._kvbench_state = state
                defer_append_until_after_attn = False
        if self.enable_parity_checks:
            # Expected materialized length depends on whether append already occurred:
            # - Prefill path (deferred append): k_all/v_all = past + current.
            # - Decode path (q_len==1): append happens before materialize, so
            #   materialized length already equals current state.total_len.
            if is_kivi_cache and q_len > 1:
                expected_kv_len = int(getattr(state, "total_len", 0)) + int(key_states.shape[-2])
            else:
                expected_kv_len = int(getattr(state, "total_len", 0))
            if used_fused_kivi_qk:
                got_k_len = fused_logits_len
                got_v_len = fused_kv_len
            else:
                got_k_len = int(k_all.shape[-2])
                got_v_len = int(v_all.shape[-2])
            if got_k_len != expected_kv_len or got_v_len != expected_kv_len:
                if hasattr(state, "telemetry"):
                    state.telemetry.append(
                        {
                            "event": "parity_warning",
                            "kind": "kv_materialization_mismatch",
                            "expected_kv_len": expected_kv_len,
                            "got_k_len": got_k_len,
                            "got_v_len": got_v_len,
                            "used_fused_kivi_qk": bool(used_fused_kivi_qk),
                        }
                    )

        if not used_fused_kivi_qk:
            # Repeat kv to match query heads.
            k_all = repeat_kv(k_all, self.num_kv_groups)
            v_all = repeat_kv(v_all, self.num_kv_groups)
            # Attention logits and output.
            attn_weights = torch.matmul(query_states, k_all.transpose(-2, -1)) / (self.head_dim**0.5)

        # Use HF-provided causal mask only during prefill (q_len>1).
        # For incremental decoding (q_len==1), the adapter cache already contains only past tokens,
        # so we can skip mask to avoid shape mismatches against HF's computed target_length.
        if attention_mask is not None and q_len > 1:
            if attention_mask.dim() == 4:
                am = attention_mask
                # Align to attn_weights shape (b, 1, q_len, kv_len).
                am = am[..., : attn_weights.shape[-2], : attn_weights.shape[-1]]
                attn_weights = attn_weights + am
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if (
            is_kivi_cache
            and self.drift_probe_interval > 0
            and q_len == 1
            and int(getattr(state, "total_len", 0)) % self.drift_probe_interval == 0
            and hasattr(state, "telemetry")
        ):
            state.telemetry.append(
                {
                    "event": "drift_probe",
                    "seq_len": int(getattr(state, "total_len", 0)),
                    "attn_max": float(attn_weights.max().item()),
                    "attn_mean": float(attn_weights.mean().item()),
                }
            )
        attn_output = torch.matmul(attn_weights, v_all)  # (b, hq, t, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.attn.o_proj(attn_output)
        if defer_append_until_after_attn:
            state = self.cache_impl.append_prefill_storage(state, key_states, value_states)
            self._kvbench_state = state
        if self.enable_parity_checks:
            actual_total = int(getattr(self._kvbench_state, "total_len", 0))
            expected_total = int(getattr(state, "total_len", 0))
            if actual_total != expected_total and hasattr(state, "telemetry"):
                state.telemetry.append(
                    {
                        "event": "parity_warning",
                        "kind": "kv_state_total_mismatch",
                        "expected_total": expected_total,
                        "actual_total": actual_total,
                    }
                )

        present = None
        return attn_output, (attn_weights if output_attentions else None), present


def patch_hf_model_kv_cache(
    model: nn.Module,
    *,
    method: str,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 128,
    nuq_bits: int = 4,
    outlier_percent: float = 0.01,
    first_few_fp16: int = 0,
    use_nf: bool = False,
    kivi_mode: str = "official_like",
    k_residual_length: Optional[int] = None,
    v_residual_length: Optional[int] = None,
    kivi_diagnostics: bool = False,
    kivi_parity_checks: bool = False,
    kivi_drift_probe_interval: int = 0,
    kivi_enable_fused_qk: bool = True,
    kivi_fused_qk_backend: str = "auto",
    kvquant_enable_fused_qk: bool = True,
    kvquant_fused_qk_backend: str = "auto",
) -> Tuple[nn.Module, PatchedCacheState]:
    """Patch a HF decoder-only Llama-family model to use our KV cache quantizers."""
    if method == "fp16":
        return model, PatchedCacheState(per_layer=[])

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Unsupported model shape: expected model.model.layers")

    cache_states: list[Any] = []
    for layer in model.model.layers:
        attn = layer.self_attn
        num_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
        num_kv_heads = getattr(attn, "num_key_value_heads", None)
        head_dim = getattr(attn, "head_dim", None)

        if num_heads is None or num_kv_heads is None or head_dim is None:
            cfg = getattr(model, "config", None)
            if cfg is None:
                raise ValueError("Could not infer head config for attention")
            num_heads = cfg.num_attention_heads
            num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            head_dim = cfg.hidden_size // cfg.num_attention_heads

        if method.startswith("kivi"):
            validate_kivi_bits(k_bits, v_bits)
            cache_impl = KiviCache(
                k_bits=k_bits,
                v_bits=v_bits,
                group_size=group_size,
                residual_length=residual_length,
                k_residual_length=k_residual_length,
                v_residual_length=v_residual_length,
                kivi_mode=kivi_mode,
                diagnostics=kivi_diagnostics,
            )
            cache_states.append(cache_impl.init_state())
        elif method.startswith("kvquant"):
            cache_impl = KvQuantCache(bits=nuq_bits, outlier_percent=outlier_percent, first_few_fp16=first_few_fp16, use_nf=use_nf)
            cache_states.append(cache_impl.init_state())
        else:
            raise ValueError(f"Unknown method {method}")

        layer.self_attn = AttentionCacheAdapter(
            attn,
            cache_impl=cache_impl,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            enable_parity_checks=kivi_parity_checks and method.startswith("kivi"),
            drift_probe_interval=kivi_drift_probe_interval if method.startswith("kivi") else 0,
            enable_fused_kivi_qk=kivi_enable_fused_qk and method.startswith("kivi"),
            fused_kivi_qk_backend=kivi_fused_qk_backend if method.startswith("kivi") else "auto",
            enable_fused_kvquant_qk=kvquant_enable_fused_qk and method.startswith("kvquant"),
            fused_kvquant_qk_backend=kvquant_fused_qk_backend if method.startswith("kvquant") else "auto",
        )

    return model, PatchedCacheState(per_layer=cache_states)

