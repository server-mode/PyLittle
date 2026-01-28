"""
Benchmark HF (vanilla) vs HF+PyLittle policy wrapper.
Measures latency, optional streaming throughput (chars/s), optional GPU stats via NVML.

Usage (PowerShell):
    python d:\PyLittle\tools\bench_hf_vs_pylittle.py --preset tiny --device auto --strategy low_vram_auto --stream
    python d:\PyLittle\tools\bench_hf_vs_pylittle.py --model facebook/opt-350m --tokens 128 --device cuda --strategy low_vram_auto

Requires: transformers, torch; optional: pynvml for GPU stats.
"""
from __future__ import annotations
import argparse
from time import perf_counter
import json
import statistics as stats

from pylittle import Engine, config
from pylittle.adapters.hf_runtime import hf_generate
from pylittle.utils.metrics import stopwatch, gpu_stats


def run_vanilla(model: str, prompt: str, tokens: int, temperature: float, device: str, stream: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from transformers import TextIteratorStreamer
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True)
    # Robust device selection: if user forces cuda but CUDA isn't available, fallback to cpu
    if device == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    elif device == "auto" and torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cuda" if device == "cuda" else "cpu"
    mdl.to(dev).eval()
    inp = tok(prompt, return_tensors="pt").to(dev)
    # warm-up
    with torch.no_grad():
        _ = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=8)
    if stream:
        # streaming throughput (chars/s)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            t0 = perf_counter()
            _ = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=tokens, streamer=streamer)
            total_chars = 0
            total_tokens = 0
            for chunk in streamer:
                total_chars += len(chunk)
                # approximate tokens for chunk (may not be exact due to boundaries)
                try:
                    total_tokens += len(tok(chunk, add_special_tokens=False).input_ids)
                except Exception:
                    pass
            dt = perf_counter() - t0
        th = total_chars / dt if dt > 0 else 0.0
        ts = (total_tokens / dt) if dt > 0 else None
        return {"throughput_chars_s": th, "len": total_chars, "device": dev, "tokens_generated": total_tokens, "tokens_s": ts}
    # non-stream timed (median of 3)
    times = []
    out = None
    with torch.no_grad():
        for _ in range(3):
            t0 = perf_counter()
            out = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=tokens)
            times.append(perf_counter() - t0)
    text = tok.decode(out[0], skip_special_tokens=True)
    # token count (new tokens = total - input length)
    try:
        new_tokens = out.shape[-1] - int(inp["input_ids"].shape[-1])
    except Exception:
        new_tokens = None
    return {"latency_s": stats.median(times), "len": len(text), "text": text, "device": dev, "tokens_generated": new_tokens}


def run_pylittle(model: str, prompt: str, tokens: int, temperature: float, device: str,
                 strategy_name: str | None = None, stream: bool = False,
                 paging_window: int | None = None,
                 no_quant: bool = False, no_offload: bool = False):
    eng = Engine.load("stub.bin", device=device, config=config.load_profile("balanced"))
    # Compute report device similar to vanilla (actual HF compute device)
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device_used = "cpu"
    elif device == "auto" and torch.cuda.is_available():
        device_used = "cuda"
    else:
        device_used = "cuda" if device == "cuda" else "cpu"
    plan = None
    if strategy_name == "low_vram_auto":
        # assume 7B ~ 14e9 bytes fp16 baseline (dummy for now); plan only affects HF load for now
        plan = config.budgeter(desired_model_size_b=int(14e9), target_device=device)
        if plan and no_quant:
            plan["quant"] = None
        if plan and no_offload:
            plan["offload"] = False
    elif strategy_name == "gpu_only":
        # Force GPU-only: no offload, optional 4bit quant; keep device cuda
        plan = {"device": "cuda", "quant": None if no_quant else "4bit", "offload": False}
    # streaming throughput or non-stream latency
    if stream:
        # warm-up stream
        _ = list(hf_generate(
            eng, model, prompt, max_new_tokens=8, temperature=temperature,
            stream=True, device=device, strategy=plan, paging_window=paging_window
        ))
        total_chars = 0
        t0 = perf_counter()
        # approximate tokens by re-tokenizing chunks here
        try:
            from transformers import AutoTokenizer as _AT
            _tok = _AT.from_pretrained(model)
        except Exception:
            _tok = None
        total_tokens = 0
    for chunk in hf_generate(
            eng, model, prompt, max_new_tokens=tokens, temperature=temperature,
            stream=True, device=device, strategy=plan, paging_window=paging_window
        ):
            total_chars += len(chunk)
            if _tok is not None:
                try:
                    total_tokens += len(_tok(chunk, add_special_tokens=False).input_ids)
                except Exception:
                    pass
    dt = perf_counter() - t0
    th = total_chars / dt if dt > 0 else 0.0
    ts = (total_tokens / dt) if dt > 0 else None
    return {"throughput_chars_s": th, "len": total_chars, "device": device_used, "strategy": plan, "tokens_generated": total_tokens, "tokens_s": ts}
    # non-stream: warm-up then median of 3
    _ = hf_generate(eng, model, prompt, max_new_tokens=8, temperature=temperature, stream=False, device=device, strategy=plan)
    times = []
    text = ""
    for _ in range(3):
        with stopwatch() as get_dt:
            text = hf_generate(eng, model, prompt, max_new_tokens=tokens, temperature=temperature, stream=False, device=device, strategy=plan)
            dt = get_dt()
            if callable(dt):
                dt = dt()
        times.append(dt)
    return {"latency_s": stats.median(times), "len": len(text), "text": text, "device": device_used, "strategy": plan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model name. Overrides --preset if set.")
    ap.add_argument("--preset", default="tiny", choices=["tiny", "350m", "410m", "1b"], help="Model preset selector.")
    ap.add_argument("--prompt", default="Hello from PyLittle")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--strategy", default=None, choices=["low_vram_auto", "gpu_only"])
    ap.add_argument("--stream", action="store_true", help="Measure streaming throughput (chars/s) instead of latency.")
    ap.add_argument("--paging-window", type=int, default=None, help="Enable native KV paging (demo): recent-token window size")
    ap.add_argument("--no-quant", action="store_true", help="Disable quant in strategy plan")
    ap.add_argument("--no-offload", action="store_true", help="Disable offload in strategy plan")
    args = ap.parse_args()

    # resolve preset -> model
    preset_map = {
        "tiny": "sshleifer/tiny-gpt2",
        "350m": "facebook/opt-350m",
        "410m": "EleutherAI/pythia-410m-deduped",
        "1b": "microsoft/phi-1_5",  # lightweight 1.3B-class alternative could be used
    }
    model = args.model or preset_map[args.preset]

    base_gpu = gpu_stats()
    v = run_vanilla(model, args.prompt, args.tokens, args.temperature, args.device, stream=args.stream)
    mid_gpu = gpu_stats()
    p = run_pylittle(model, args.prompt, args.tokens, args.temperature, args.device, args.strategy,
                     stream=args.stream, paging_window=args.paging_window,
                     no_quant=args.no_quant, no_offload=args.no_offload)
    end_gpu = gpu_stats()

    speedup = None
    if not args.stream and (v.get("latency_s") and p.get("latency_s")):
        if v["latency_s"] > 0 and p["latency_s"] > 0:
            speedup = v["latency_s"] / p["latency_s"]

    out = {}
    if args.stream:
        out.update({
            "vanilla": {"throughput_chars_s": round(v.get("throughput_chars_s", 0.0), 2), "len": v.get("len"), "tokens_generated": v.get("tokens_generated"), "tokens_s": (round(v.get("tokens_s", 0.0),2) if v.get("tokens_s") else None)},
            "pylittle": {"throughput_chars_s": round(p.get("throughput_chars_s", 0.0), 2), "len": p.get("len"), "tokens_requested": args.tokens, "tokens_generated": p.get("tokens_generated"), "tokens_s": (round(p.get("tokens_s", 0.0),2) if p.get("tokens_s") else None)},
        })
    else:
        out.update({
            "vanilla": {"latency_s": round(v["latency_s"], 4), "len": v["len"], "tokens_generated": v.get("tokens_generated")},
            "pylittle": {"latency_s": round(p["latency_s"], 4), "len": p["len"], "tokens_requested": args.tokens},
            "speedup_x": round(speedup, 3) if speedup else None,
        })
    out.update({
        "devices": {"vanilla": v.get("device"), "pylittle": p.get("device")},
        "strategy": p.get("strategy"),
        "gpu_before": base_gpu,
        "gpu_mid": mid_gpu,
        "gpu_after": end_gpu,
    })
    # VRAM delta if stats available
    try:
        if base_gpu and end_gpu:
            out["vram_delta_mb"] = round(end_gpu["mem_used_mb"] - base_gpu["mem_used_mb"], 2)
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
