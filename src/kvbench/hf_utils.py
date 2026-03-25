from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    use_flash_attn_2: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, any]:
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    config.use_cache = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cache_dir,
        trust_remote_code=False,
        attn_implementation=("flash_attention_2" if use_flash_attn_2 else "eager"),
    )
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def perplexity_on_tokens(model, input_ids: torch.Tensor) -> float:
    # input_ids: (1, T)
    out = model(input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="mean",
    )
    return float(torch.exp(loss).item())

