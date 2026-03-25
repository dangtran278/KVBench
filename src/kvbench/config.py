from dataclasses import dataclass
from typing import Literal, Optional


KvMethod = Literal[
    "fp16",
    "kivi2",
    "kivi4",
    "kvquant_nuq3_1p",
    "kvquant_nuq4_1p",
]


@dataclass(frozen=True)
class KvQuantConfig:
    method: KvMethod

    # Common
    device: str = "cuda"
    dtype: str = "float16"

    # KIVI-style
    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 32
    residual_length: int = 128

    # KVQuant-style
    nuq_bits: int = 4
    outlier_percent: float = 0.01  # 1%
    calib_nsamples: int = 16
    # If set, keep first N tokens in fp16 in the cache
    first_few_fp16: int = 0
    # Whether to use NF (NormalFloat-like) LUT instead of kmeans NUQ
    use_nf: bool = False

    # Model selection (proposal)
    model_accuracy: str = "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
    model_efficiency: str = "/datasets/ai/llama2/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/"
    cache_dir: Optional[str] = None

