from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .kivi_cache import KiviCache, KiviCacheState
from .kvquant_cache import KvQuantCache, KvQuantCacheState


@dataclass
class PatchedCacheState:
    per_layer: list[Any]


def reset_kvbench_state(model: nn.Module) -> None:
    """Reset internal KV cache states before a new prompt."""
    for m in model.modules():
        if hasattr(m, "reset_kvbench_state"):
            m.reset_kvbench_state()


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
    ):
        super().__init__()
        self.attn = attn
        self.cache_impl = cache_impl
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(head_dim)

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
        state = self.cache_impl.append(state, key_states, value_states)
        self._kvbench_state = state

        k_all, v_all = self.cache_impl.materialize(state, out_dtype=query_states.dtype)

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
        attn_output = torch.matmul(attn_weights, v_all)  # (b, hq, t, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.attn.o_proj(attn_output)

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
            cache_impl = KiviCache(k_bits=k_bits, v_bits=v_bits, group_size=group_size, residual_length=residual_length)
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
        )

    return model, PatchedCacheState(per_layer=cache_states)

