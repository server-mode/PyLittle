from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def gpu_mem_mb() -> float | None:
    if not _NVML_OK:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return float(mem.used) / (1024**2)
    except Exception:
        return None


def run_case(model: str, prompt: str, tokens: int, paging_window: int | None, device: str, strategy: str | None) -> Dict[str, Any]:
    from pylittle import Engine
    from pylittle.adapters.hf_runtime import hf_generate
    import torch

    # Preload model+tokenizer so we can measure generation-only KV growth.
    if str(model).lower() in ("synthetic-gpt2", "synthetic"):
        from pylittle.adapters import hf_runtime as _hr

        model_obj, tok_obj = _hr._make_synthetic_gpt2(device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_obj = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True)
        try:
            if device == "cuda" and torch.cuda.is_available():
                model_obj.to("cuda")
        except Exception:
            pass
        model_obj.eval()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc_before = int(torch.cuda.memory_allocated())
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    else:
        alloc_before = None

    before = gpu_mem_mb()
    t0 = time.perf_counter()
    it = hf_generate(Engine.load("stub.bin", device=device), model, prompt, max_new_tokens=tokens, temperature=0.8,
                    stream=True, device=device, strategy=None, paging_window=paging_window,
                    model_obj=model_obj, tokenizer_obj=tok_obj)
    out = "".join(list(it))
    dt = time.perf_counter() - t0
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc_after = int(torch.cuda.memory_allocated())
        peak_alloc = int(torch.cuda.max_memory_allocated())
        alloc_delta = alloc_after - alloc_before if alloc_before is not None else None
        peak_delta = peak_alloc - alloc_before if alloc_before is not None else None
    else:
        alloc_after = None
        peak_alloc = None
        alloc_delta = None
        peak_delta = None
    after = gpu_mem_mb()
    return {
        "paging_window": paging_window,
        "len": len(out),
        "seconds": round(dt, 4),
        "throughput_chars_s": round(len(out) / dt, 2) if dt > 0 else None,
        "cuda_alloc_before_mb": (alloc_before / (1024**2)) if alloc_before is not None else None,
        "cuda_alloc_after_mb": (alloc_after / (1024**2)) if alloc_after is not None else None,
        "cuda_alloc_delta_mb": (alloc_delta / (1024**2)) if alloc_delta is not None else None,
        "cuda_peak_alloc_mb": (peak_alloc / (1024**2)) if peak_alloc is not None else None,
        "cuda_peak_delta_mb": (peak_delta / (1024**2)) if peak_delta is not None else None,
        "gpu_mem_before_mb": before,
        "gpu_mem_after_mb": after,
        "gpu_mem_delta_mb": (after - before) if (after is not None and before is not None) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="synthetic-gpt2", help="Use 'synthetic-gpt2' for offline smoke tests")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--prompt", default="Tell me a story about memory paging. ")
    ap.add_argument("--single-case", default=None, choices=["no_window", "window"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single_case == "no_window":
        print(json.dumps(run_case(args.model, args.prompt, args.tokens, None, args.device, None)))
        return 0
    if args.single_case == "window":
        print(json.dumps(run_case(args.model, args.prompt, args.tokens, args.window, args.device, None)))
        return 0

    # Isolate each run in a fresh process to keep CUDA allocator baselines comparable.
    env = os.environ.copy()
    cmd_base = [sys.executable, os.path.abspath(__file__), "--model", args.model, "--tokens", str(args.tokens), "--device", args.device, "--window", str(args.window), "--prompt", args.prompt]
    no_win = subprocess.check_output(cmd_base + ["--single-case", "no_window"], env=env, text=True)
    win = subprocess.check_output(cmd_base + ["--single-case", "window"], env=env, text=True)
    res = {
        "case_no_window": json.loads(no_win.strip().splitlines()[-1]),
        "case_window": json.loads(win.strip().splitlines()[-1]),
    }
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
