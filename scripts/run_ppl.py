import argparse

import torch
from datasets import load_dataset

from kvbench.hf_utils import load_model_and_tokenizer, perplexity_on_tokens
from kvbench.modeling_patch import patch_hf_model_kv_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--method", type=str, default="fp16", choices=["fp16", "kivi2", "kivi4", "kvquant_nuq3_1p", "kvquant_nuq4_1p"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--cache_dir", type=str, default=None)

    # KIVI-style params
    ap.add_argument("--k_bits", type=int, default=2)
    ap.add_argument("--v_bits", type=int, default=2)
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--residual_length", type=int, default=128)

    # KVQuant-style params
    ap.add_argument("--nuq_bits", type=int, default=4)
    ap.add_argument("--outlier_percent", type=float, default=0.01)
    ap.add_argument("--first_few_fp16", type=int, default=0)
    ap.add_argument("--use_nf", action="store_true")

    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.model, device=args.device, cache_dir=args.cache_dir, use_flash_attn_2=False)
    model, _ = patch_hf_model_kv_cache(
        model,
        method=args.method,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
        nuq_bits=args.nuq_bits,
        outlier_percent=args.outlier_percent,
        first_few_fp16=args.first_few_fp16,
        use_nf=args.use_nf,
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens).input_ids.to(args.device)

    ppl = perplexity_on_tokens(model, input_ids)
    print(f"ppl={ppl:.4f}")


if __name__ == "__main__":
    main()

