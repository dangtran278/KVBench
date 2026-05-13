'''
python scripts/run_matrix_new.py \
  --device cuda \
  --output_dir logs \
  --run_ppl \
  --run_passkey \
  --run_memory \
  --run_latency \
  --run_throughput \
  --run_scaling
'''
import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Prefer this repo's `src/kvbench` when running `python scripts/run_matrix_new.py`
# (avoids stale site-packages installs missing newer modules like `bench_utils`).
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    _src = str(_repo_src)
    if _src not in sys.path:
        sys.path.insert(0, _src)

import torch
from datasets import load_dataset

from kvbench.bench_utils import cuda_peak_memory_gb, reset_cuda_peak_stats, time_cuda_callable_ms
from kvbench.config import KvQuantConfig
from kvbench.hf_utils import load_model_and_tokenizer, perplexity_on_tokens
from kvbench.modeling_patch import (
    collect_kivi_telemetry,
    collect_kvquant_telemetry,
    patch_hf_model_kv_cache,
    reset_kvbench_state,
)
from run_passkey import build_passkey_prompt, extract_passkey_prediction, greedy_decode_next_tokens


RUNS_COLUMNS = [
    "run_id",
    "timestamp",
    "model",
    "method",
    "bits",
    "outlier_percent",
    "group_size",
    "residual_length",
    "first_few_fp16",
    "task",
    "context_len",
    "batch_size",
    "prefill_tokens",
    "decode_tokens",
    "seed",
    "success",
    "oom",
    "error_msg",
]

METRICS_COLUMNS = [
    "run_id",
    "metric_name",
    "metric_value",
    "unit",
    "split",
    "notes",
]


def ensure_csv(path: str, columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_csv(path: str, columns: list[str], row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow({k: row.get(k, "") for k in columns})


def append_jsonl(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def flush_kivi_telemetry(
    model,
    *,
    telemetry_path: str,
    metrics_path: str,
    run_id: str,
) -> None:
    rows = collect_kivi_telemetry(model, clear=True) + collect_kvquant_telemetry(model, clear=True)
    append_jsonl(telemetry_path, rows)
    if not rows:
        return
    backend_counts: dict[str, int] = {}
    total = 0
    for row in rows:
        if row.get("event") not in ("fused_kivi_qk", "fused_kvquant_qk"):
            continue
        backend = str(row.get("backend", "unknown"))
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        total += 1
    if total <= 0:
        return
    append_csv(
        metrics_path,
        METRICS_COLUMNS,
        {
            "run_id": run_id,
            "metric_name": "fused_qk_events_total",
            "metric_value": total,
            "unit": "count",
            "split": "debug",
            "notes": "decode fused QK backend events",
        },
    )
    for backend, count in sorted(backend_counts.items()):
        append_csv(
            metrics_path,
            METRICS_COLUMNS,
            {
                "run_id": run_id,
                "metric_name": f"fused_qk_events_{backend}",
                "metric_value": count,
                "unit": "count",
                "split": "debug",
                "notes": f"backend={backend}",
            },
        )
        append_csv(
            metrics_path,
            METRICS_COLUMNS,
            {
                "run_id": run_id,
                "metric_name": f"fused_qk_ratio_{backend}",
                "metric_value": float(count) / float(total),
                "unit": "ratio",
                "split": "debug",
                "notes": f"backend={backend}",
            },
        )


def method_params(method: str) -> dict:
    if method == "kivi2":
        return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.01)
    if method == "kivi4":
        return dict(k_bits=4, v_bits=4, nuq_bits=4, outlier_percent=0.01)
    if method == "kvquant_nuq3_1p":
        return dict(k_bits=2, v_bits=2, nuq_bits=3, outlier_percent=0.01)
    if method == "kvquant_nuq4_1p":
        return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.01)
    return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.0)


def is_kvbench_patched(model) -> bool:
    try:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if not layers:
            return False
        return hasattr(layers[0].self_attn, "cache_impl")
    except Exception:
        return False


@torch.no_grad()
def run_ppl_task(model, tok, max_tokens: int, device: str, ppl_prefill_tokens: int | None = None) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
    return float(perplexity_on_tokens(model, input_ids, prefill_tokens=ppl_prefill_tokens))


@torch.no_grad()
def run_passkey_task(model, tok, context_tokens: int, decode_tokens: int, seed: int, device: str) -> tuple[float, int, int, str]:
    random.seed(seed)
    torch.manual_seed(seed)
    prompt, key = build_passkey_prompt(context_tokens)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=context_tokens).to(device)
    reset_kvbench_state(model)
    gen_ids = greedy_decode_next_tokens(model, enc["input_ids"], max_new_tokens=decode_tokens)
    gen = tok.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    pred = extract_passkey_prediction(gen)
    tail = gen[-120:].replace("\n", " ")
    return (1.0 if pred == key else 0.0), int(key), int(pred) if pred is not None else -1, tail


def run_memory_task(
    model,
    tok,
    context_tokens: int,
    decode_tokens: int,
    seed: int,
    device: str,
    chunked_prefill_tokens: int = 0,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    prompt, _ = build_passkey_prompt(context_tokens)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=context_tokens).to(device)
    reset_kvbench_state(model)
    reset_cuda_peak_stats(device)
    use_internal_cache = is_kvbench_patched(model)
    out, past_key_values = prefill_with_optional_chunks(
        model,
        enc["input_ids"],
        chunk_size=chunked_prefill_tokens,
        use_internal_cache=use_internal_cache,
    )
    if decode_tokens > 0:
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_pos = int(enc["input_ids"].shape[-1])
        for _ in range(decode_tokens):
            if use_internal_cache:
                pos_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=enc["input_ids"].device)
                out = model(next_token, position_ids=pos_ids, use_cache=False)
            else:
                out = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            cur_pos += 1
    return cuda_peak_memory_gb(device)


@torch.no_grad()
def prefill_with_optional_chunks(model, input_ids: torch.Tensor, *, chunk_size: int = 0, use_internal_cache: bool = True):
    if chunk_size <= 0:
        if use_internal_cache:
            out = model(input_ids, use_cache=False)
            return out, None
        out = model(input_ids, use_cache=True)
        return out, out.past_key_values
    total = int(input_ids.shape[-1])
    start = 0
    last_out = None
    past_key_values = None
    bsz = input_ids.shape[0]
    while start < total:
        end = min(total, start + chunk_size)
        chunk = input_ids[:, start:end]
        if use_internal_cache:
            pos_ids = torch.arange(start, end, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(bsz, 1)
            last_out = model(chunk, position_ids=pos_ids, use_cache=False)
        else:
            last_out = model(chunk, past_key_values=past_key_values, use_cache=True)
            past_key_values = last_out.past_key_values
        start = end
    return last_out, past_key_values


@torch.no_grad()
def run_decode_latency_task(
    model,
    tok,
    context_tokens: int,
    decode_tokens: int,
    warmup_tokens: int,
    seed: int,
    device: str,
    chunked_prefill_tokens: int = 0,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    prompt, _ = build_passkey_prompt(context_tokens)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=context_tokens).to(device)
    reset_kvbench_state(model)
    use_internal_cache = is_kvbench_patched(model)
    out, past_key_values = prefill_with_optional_chunks(
        model,
        enc["input_ids"],
        chunk_size=chunked_prefill_tokens,
        use_internal_cache=use_internal_cache,
    )
    start_pos = int(enc["input_ids"].shape[-1])
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    cur_pos = start_pos

    def one_decode_step() -> None:
        nonlocal next_token, cur_pos, past_key_values
        if use_internal_cache:
            pos_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=enc["input_ids"].device)
            out = model(next_token, position_ids=pos_ids, use_cache=False)
        else:
            out = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_pos += 1

    ms_per_token = time_cuda_callable_ms(one_decode_step, device=device, steps=decode_tokens, warmup=warmup_tokens)
    return {"decode_latency_ms_per_token": ms_per_token}


@torch.no_grad()
def run_prefill_throughput_task(
    model,
    tok,
    context_tokens: int,
    batch_size: int,
    seed: int,
    device: str,
    chunked_prefill_tokens: int = 0,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    prompt, _ = build_passkey_prompt(context_tokens)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=context_tokens).to(device)
    input_ids = enc["input_ids"].repeat(batch_size, 1)
    reset_kvbench_state(model)
    reset_cuda_peak_stats(device)
    t0 = time.perf_counter()
    use_internal_cache = is_kvbench_patched(model)
    prefill_with_optional_chunks(
        model,
        input_ids,
        chunk_size=chunked_prefill_tokens,
        use_internal_cache=use_internal_cache,
    )
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = max(1e-6, t1 - t0)
    total_tokens = int(batch_size * input_ids.shape[-1])
    tps = float(total_tokens / elapsed)
    mem = cuda_peak_memory_gb(device)
    return {"throughput_tokens_per_sec": tps, **mem}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="logs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode_tokens", type=int, default=10)
    ap.add_argument("--ppl_tokens", type=int, default=4096)
    ap.add_argument(
        "--ppl_prefill_tokens",
        type=int,
        default=512,
        help="Prefill tokens before decode streaming (use 512/1024 for KIVI stress; default: 512)",
    )
    ap.add_argument("--passkey_contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["fp16", "kivi4", "kivi2", "kvquant_nuq4_1p", "kvquant_nuq3_1p"],
    )
    ap.add_argument("--run_ppl", action="store_true", help="Run Wikitext-2 perplexity")
    ap.add_argument("--run_passkey", action="store_true", help="Run passkey retrieval")
    ap.add_argument("--run_memory", action="store_true", help="Run peak VRAM task")
    ap.add_argument("--run_latency", action="store_true", help="Run decode latency task")
    ap.add_argument("--run_throughput", action="store_true", help="Run throughput and max batch task")
    ap.add_argument("--run_scaling", action="store_true", help="Run memory scaling across contexts")
    ap.add_argument("--latency_context_tokens", type=int, nargs="+", default=[4096])
    ap.add_argument("--latency_decode_tokens", type=int, default=64)
    ap.add_argument("--latency_warmup_tokens", type=int, default=8)
    ap.add_argument("--throughput_context_tokens", type=int, default=4096)
    ap.add_argument("--throughput_batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--chunked_prefill_tokens", type=int, default=0, help="If >0, prefill in chunks of this size")
    ap.add_argument("--kivi_mode", type=str, default="official_like", choices=["legacy", "official_like"])
    ap.add_argument("--k_residual_length", type=int, default=None)
    ap.add_argument("--v_residual_length", type=int, default=None)
    ap.add_argument("--kivi_diagnostics", action="store_true")
    ap.add_argument("--kivi_parity_checks", action="store_true")
    ap.add_argument("--kivi_drift_probe_interval", type=int, default=0)
    ap.add_argument("--kivi_telemetry", action="store_true", help="Write KIVI cache telemetry jsonl records")
    ap.add_argument("--kivi_enable_fused_qk", action="store_true", help="Enable decode fused K-side QK path for KIVI")
    ap.add_argument(
        "--kivi_fused_qk_backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "torch"],
        help="Backend for fused KIVI QK path (auto/triton/torch).",
    )
    ap.add_argument("--kvquant_enable_fused_qk", action="store_true", help="Enable decode fused K-side QK path for KVQuant")
    ap.add_argument(
        "--kvquant_fused_qk_backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "torch"],
        help="Backend for fused KVQuant QK path (auto/triton/torch).",
    )
    args = ap.parse_args()

    if not args.run_ppl and not args.run_passkey and not args.run_memory and not args.run_latency and not args.run_throughput and not args.run_scaling:
        args.run_ppl = True
        args.run_passkey = True

    cfg = KvQuantConfig(method="fp16", device=args.device, cache_dir=args.cache_dir)
    runs_path = os.path.join(args.output_dir, "runs.csv")
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    telemetry_path = os.path.join(args.output_dir, "kivi_telemetry.jsonl")
    ensure_csv(runs_path, RUNS_COLUMNS)
    ensure_csv(metrics_path, METRICS_COLUMNS)
    effective_ppl_prefill = args.ppl_prefill_tokens if args.ppl_prefill_tokens is not None else args.ppl_tokens - 1

    for method in args.methods:
        params = method_params(method)

        # Accuracy model according to proposal.
        model_path = cfg.model_accuracy
        model, tok = load_model_and_tokenizer(
            model_path,
            device=args.device,
            cache_dir=args.cache_dir,
            use_flash_attn_2=False,
        )
        model, _ = patch_hf_model_kv_cache(
            model,
            method=method,
            k_bits=params["k_bits"],
            v_bits=params["v_bits"],
            group_size=cfg.group_size,
            residual_length=cfg.residual_length,
            k_residual_length=args.k_residual_length,
            v_residual_length=args.v_residual_length,
            nuq_bits=params["nuq_bits"],
            outlier_percent=params["outlier_percent"],
            first_few_fp16=cfg.first_few_fp16,
            use_nf=cfg.use_nf,
            kivi_mode=args.kivi_mode,
            kivi_diagnostics=args.kivi_diagnostics,
            kivi_parity_checks=args.kivi_parity_checks,
            kivi_drift_probe_interval=args.kivi_drift_probe_interval,
            kivi_enable_fused_qk=args.kivi_enable_fused_qk,
            kivi_fused_qk_backend=args.kivi_fused_qk_backend,
            kvquant_enable_fused_qk=args.kvquant_enable_fused_qk,
            kvquant_fused_qk_backend=args.kvquant_fused_qk_backend,
        )

        if args.run_ppl:
            run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
            err = ""
            success = 0
            oom = 0
            ppl = None
            try:
                ppl = run_ppl_task(
                    model,
                    tok,
                    max_tokens=args.ppl_tokens,
                    device=args.device,
                    ppl_prefill_tokens=args.ppl_prefill_tokens,
                )
                success = 1
            except RuntimeError as e:
                err = str(e)
                if "out of memory" in err.lower():
                    oom = 1
            except Exception:
                err = traceback.format_exc(limit=1).strip().replace("\n", " ")

            append_csv(
                runs_path,
                RUNS_COLUMNS,
                {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": "llama3_8b",
                    "method": method,
                    "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                    "outlier_percent": params["outlier_percent"],
                    "group_size": cfg.group_size,
                    "residual_length": cfg.residual_length,
                    "first_few_fp16": cfg.first_few_fp16,
                    "task": "ppl_wikitext2",
                    "context_len": args.ppl_tokens,
                    "batch_size": 1,
                    "prefill_tokens": effective_ppl_prefill,
                    "decode_tokens": max(0, args.ppl_tokens - effective_ppl_prefill),
                    "seed": args.seed,
                    "success": success,
                    "oom": oom,
                    "error_msg": err,
                },
            )
            if ppl is not None:
                append_csv(
                    metrics_path,
                    METRICS_COLUMNS,
                    {
                        "run_id": run_id,
                        "metric_name": "wikitext2_ppl_4k",
                        "metric_value": ppl,
                        "unit": "ppl",
                        "split": "test",
                        "notes": (
                            f"max_tokens={args.ppl_tokens},prefill_tokens={effective_ppl_prefill},kivi_mode={args.kivi_mode},"
                            f"k_residual_length={args.k_residual_length},v_residual_length={args.v_residual_length},"
                            f"kivi_diagnostics={int(args.kivi_diagnostics)},kivi_parity_checks={int(args.kivi_parity_checks)},"
                            f"kivi_drift_probe_interval={args.kivi_drift_probe_interval},kivi_telemetry={int(args.kivi_telemetry)},"
                            f"kivi_enable_fused_qk={int(args.kivi_enable_fused_qk)},kivi_fused_qk_backend={args.kivi_fused_qk_backend},"
                            f"kvquant_enable_fused_qk={int(args.kvquant_enable_fused_qk)},kvquant_fused_qk_backend={args.kvquant_fused_qk_backend}"
                        ),
                    },
                )
            if args.kivi_telemetry and method.startswith(("kivi", "kvquant")):
                flush_kivi_telemetry(model, telemetry_path=telemetry_path, metrics_path=metrics_path, run_id=run_id)

        if args.run_passkey:
            for context_len in args.passkey_contexts:
                run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                err = ""
                success = 0
                oom = 0
                acc = None
                key = -1
                pred = -1
                tail = ""
                try:
                    acc, key, pred, tail = run_passkey_task(
                        model,
                        tok,
                        context_tokens=context_len,
                        decode_tokens=args.decode_tokens,
                        seed=args.seed,
                        device=args.device,
                    )
                    success = 1
                except RuntimeError as e:
                    err = str(e)
                    if "out of memory" in err.lower():
                        oom = 1
                except Exception:
                    err = traceback.format_exc(limit=1).strip().replace("\n", " ")

                append_csv(
                    runs_path,
                    RUNS_COLUMNS,
                    {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": "llama3_8b",
                        "method": method,
                        "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                        "outlier_percent": params["outlier_percent"],
                        "group_size": cfg.group_size,
                        "residual_length": cfg.residual_length,
                        "first_few_fp16": cfg.first_few_fp16,
                        "task": "passkey",
                        "context_len": context_len,
                        "batch_size": 1,
                        "prefill_tokens": context_len,
                        "decode_tokens": args.decode_tokens,
                        "seed": args.seed,
                        "success": success,
                        "oom": oom,
                        "error_msg": err,
                    },
                )
                if acc is not None:
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "passkey_accuracy",
                            "metric_value": acc,
                            "unit": "ratio",
                            "split": "eval",
                            "notes": f"context_len={context_len},target={key},pred={pred},tail={tail}",
                        },
                    )
                if args.kivi_telemetry and method.startswith(("kivi", "kvquant")):
                    flush_kivi_telemetry(model, telemetry_path=telemetry_path, metrics_path=metrics_path, run_id=run_id)

        if args.run_memory:
            for context_len in args.passkey_contexts:
                run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                err = ""
                success = 0
                oom = 0
                peak_alloc = None
                peak_reserved = None
                try:
                    mem = run_memory_task(
                        model,
                        tok,
                        context_tokens=context_len,
                        decode_tokens=args.decode_tokens,
                        seed=args.seed,
                        device=args.device,
                        chunked_prefill_tokens=args.chunked_prefill_tokens,
                    )
                    peak_alloc = mem["peak_allocated_gb"]
                    peak_reserved = mem["peak_reserved_gb"]
                    success = 1
                except RuntimeError as e:
                    err = str(e)
                    if "out of memory" in err.lower():
                        oom = 1
                except Exception:
                    err = traceback.format_exc(limit=1).strip().replace("\n", " ")

                append_csv(
                    runs_path,
                    RUNS_COLUMNS,
                    {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": "llama3_8b",
                        "method": method,
                        "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                        "outlier_percent": params["outlier_percent"],
                        "group_size": cfg.group_size,
                        "residual_length": cfg.residual_length,
                        "first_few_fp16": cfg.first_few_fp16,
                        "task": "memory_peak",
                        "context_len": context_len,
                        "batch_size": 1,
                        "prefill_tokens": context_len,
                        "decode_tokens": args.decode_tokens,
                        "seed": args.seed,
                        "success": success,
                        "oom": oom,
                        "error_msg": err,
                    },
                )
                if peak_alloc is not None:
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "kv_cache_peak_vram_gb",
                            "metric_value": peak_alloc,
                            "unit": "GB",
                            "split": "eval",
                            "notes": f"context_len={context_len},decode_tokens={args.decode_tokens},kind=allocated",
                        },
                    )
                if args.kivi_telemetry and method.startswith(("kivi", "kvquant")):
                    flush_kivi_telemetry(model, telemetry_path=telemetry_path, metrics_path=metrics_path, run_id=run_id)
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "kv_cache_peak_vram_reserved_gb",
                            "metric_value": peak_reserved,
                            "unit": "GB",
                            "split": "eval",
                            "notes": f"context_len={context_len},decode_tokens={args.decode_tokens},kind=reserved",
                        },
                    )

        if args.run_latency:
            for latency_ctx in args.latency_context_tokens:
                run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                err = ""
                success = 0
                oom = 0
                latency = None
                try:
                    out = run_decode_latency_task(
                        model,
                        tok,
                        context_tokens=latency_ctx,
                        decode_tokens=args.latency_decode_tokens,
                        warmup_tokens=args.latency_warmup_tokens,
                        seed=args.seed,
                        device=args.device,
                        chunked_prefill_tokens=args.chunked_prefill_tokens,
                    )
                    latency = out["decode_latency_ms_per_token"]
                    success = 1
                except RuntimeError as e:
                    err = str(e)
                    if "out of memory" in err.lower():
                        oom = 1
                except Exception:
                    err = traceback.format_exc(limit=1).strip().replace("\n", " ")

                append_csv(
                    runs_path,
                    RUNS_COLUMNS,
                    {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": "llama3_8b",
                        "method": method,
                        "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                        "outlier_percent": params["outlier_percent"],
                        "group_size": cfg.group_size,
                        "residual_length": cfg.residual_length,
                        "first_few_fp16": cfg.first_few_fp16,
                        "task": "decode_latency",
                        "context_len": latency_ctx,
                        "batch_size": 1,
                        "prefill_tokens": latency_ctx,
                        "decode_tokens": args.latency_decode_tokens,
                        "seed": args.seed,
                        "success": success,
                        "oom": oom,
                        "error_msg": err,
                    },
                )
                if latency is not None:
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "decode_latency_ms_per_token",
                            "metric_value": latency,
                            "unit": "ms/token",
                            "split": "eval",
                            "notes": f"context_len={latency_ctx},decode_tokens={args.latency_decode_tokens},warmup={args.latency_warmup_tokens}",
                        },
                    )
                if args.kivi_telemetry and method.startswith(("kivi", "kvquant")):
                    flush_kivi_telemetry(model, telemetry_path=telemetry_path, metrics_path=metrics_path, run_id=run_id)

        if args.run_throughput:
            max_ok_batch = 0
            first_oom_batch = -1
            for batch_size in sorted(args.throughput_batch_sizes):
                run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                err = ""
                success = 0
                oom = 0
                throughput = None
                peak_alloc = None
                try:
                    out = run_prefill_throughput_task(
                        model,
                        tok,
                        context_tokens=args.throughput_context_tokens,
                        batch_size=batch_size,
                        seed=args.seed,
                        device=args.device,
                        chunked_prefill_tokens=args.chunked_prefill_tokens,
                    )
                    throughput = out["throughput_tokens_per_sec"]
                    peak_alloc = out["peak_allocated_gb"]
                    success = 1
                    max_ok_batch = max(max_ok_batch, batch_size)
                except RuntimeError as e:
                    err = str(e)
                    if "out of memory" in err.lower():
                        oom = 1
                        if first_oom_batch < 0:
                            first_oom_batch = batch_size
                except Exception:
                    err = traceback.format_exc(limit=1).strip().replace("\n", " ")

                append_csv(
                    runs_path,
                    RUNS_COLUMNS,
                    {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": "llama3_8b",
                        "method": method,
                        "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                        "outlier_percent": params["outlier_percent"],
                        "group_size": cfg.group_size,
                        "residual_length": cfg.residual_length,
                        "first_few_fp16": cfg.first_few_fp16,
                        "task": "throughput_prefill",
                        "context_len": args.throughput_context_tokens,
                        "batch_size": batch_size,
                        "prefill_tokens": args.throughput_context_tokens,
                        "decode_tokens": 0,
                        "seed": args.seed,
                        "success": success,
                        "oom": oom,
                        "error_msg": err,
                    },
                )
                if throughput is not None:
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "throughput_tokens_per_sec",
                            "metric_value": throughput,
                            "unit": "tokens/sec",
                            "split": "eval",
                            "notes": f"context_len={args.throughput_context_tokens},batch_size={batch_size}",
                        },
                    )
                if args.kivi_telemetry and method.startswith(("kivi", "kvquant")):
                    flush_kivi_telemetry(model, telemetry_path=telemetry_path, metrics_path=metrics_path, run_id=run_id)
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "kv_cache_peak_vram_gb",
                            "metric_value": peak_alloc,
                            "unit": "GB",
                            "split": "eval",
                            "notes": f"context_len={args.throughput_context_tokens},batch_size={batch_size},kind=allocated",
                        },
                    )
                if oom == 1:
                    break

            run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
            append_csv(
                runs_path,
                RUNS_COLUMNS,
                {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": "llama3_8b",
                    "method": method,
                    "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                    "outlier_percent": params["outlier_percent"],
                    "group_size": cfg.group_size,
                    "residual_length": cfg.residual_length,
                    "first_few_fp16": cfg.first_few_fp16,
                    "task": "max_batch_search",
                    "context_len": args.throughput_context_tokens,
                    "batch_size": max_ok_batch,
                    "prefill_tokens": args.throughput_context_tokens,
                    "decode_tokens": 0,
                    "seed": args.seed,
                    "success": 1,
                    "oom": 0,
                    "error_msg": "",
                },
            )
            append_csv(
                metrics_path,
                METRICS_COLUMNS,
                {
                    "run_id": run_id,
                    "metric_name": "max_batch_size_under_vram",
                    "metric_value": max_ok_batch,
                    "unit": "batch",
                    "split": "eval",
                    "notes": f"context_len={args.throughput_context_tokens},first_oom_batch={first_oom_batch}",
                },
            )

        if args.run_scaling:
            last_success = -1
            first_oom = -1
            for context_len in sorted(args.passkey_contexts):
                run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                err = ""
                success = 0
                oom = 0
                peak_alloc = None
                try:
                    mem = run_memory_task(
                        model,
                        tok,
                        context_tokens=context_len,
                        decode_tokens=1,
                        seed=args.seed,
                        device=args.device,
                        chunked_prefill_tokens=args.chunked_prefill_tokens,
                    )
                    peak_alloc = mem["peak_allocated_gb"]
                    success = 1
                    last_success = context_len
                except RuntimeError as e:
                    err = str(e)
                    if "out of memory" in err.lower():
                        oom = 1
                        if first_oom < 0:
                            first_oom = context_len
                except Exception:
                    err = traceback.format_exc(limit=1).strip().replace("\n", " ")

                append_csv(
                    runs_path,
                    RUNS_COLUMNS,
                    {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": "llama3_8b",
                        "method": method,
                        "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                        "outlier_percent": params["outlier_percent"],
                        "group_size": cfg.group_size,
                        "residual_length": cfg.residual_length,
                        "first_few_fp16": cfg.first_few_fp16,
                        "task": "memory_scaling",
                        "context_len": context_len,
                        "batch_size": 1,
                        "prefill_tokens": context_len,
                        "decode_tokens": 1,
                        "seed": args.seed,
                        "success": success,
                        "oom": oom,
                        "error_msg": err,
                    },
                )
                if peak_alloc is not None:
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "memory_scaling_vram_gb",
                            "metric_value": peak_alloc,
                            "unit": "GB",
                            "split": "eval",
                            "notes": f"context_len={context_len}",
                        },
                    )
                if args.kivi_telemetry and method.startswith(("kivi", "kvquant")):
                    flush_kivi_telemetry(model, telemetry_path=telemetry_path, metrics_path=metrics_path, run_id=run_id)
                if oom == 1:
                    break

            run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
            append_csv(
                metrics_path,
                METRICS_COLUMNS,
                {
                    "run_id": run_id,
                    "metric_name": "long_context_boundary",
                    "metric_value": last_success,
                    "unit": "tokens",
                    "split": "eval",
                    "notes": f"last_success_context={last_success},first_oom_context={first_oom}",
                },
            )

        # Free model between methods.
        del model
        torch.cuda.empty_cache()

    print(f"Wrote {runs_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

