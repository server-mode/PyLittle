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
import sys

from pylittle import Engine, config
from pylittle.adapters.hf_runtime import hf_generate, hf_load_model
from pylittle.utils.metrics import stopwatch, gpu_stats, nvml_peak_memory, torch_cuda_peak_memory
from pylittle.utils.simulate import simulated_cuda_vram_limit, pcie_gen_lanes_to_gbps, simulate_pcie_sleep
from pylittle.utils.pcie_microbench import pcie_microbench_cached


def run_vanilla(model: str, prompt: str, tokens: int, temperature: float, device: str, stream: bool = False, *, local_files_only: bool = False):
    if str(model).lower() in ("synthetic-gpt2", "synthetic"):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        try:
            from transformers.cache_utils import DynamicCache  # type: ignore
        except Exception:
            DynamicCache = None

        if device == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        elif device == "auto" and torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cuda" if device == "cuda" else "cpu"

        cfg = GPT2Config(
            vocab_size=256,
            n_positions=2048,
            n_ctx=2048,
            n_embd=256,
            n_layer=4,
            n_head=4,
            bos_token_id=32,
            eos_token_id=10,
        )
        mdl = GPT2LMHeadModel(cfg).to(dev).eval()
        # For simulated PCIe throttling: rough KV bytes/token.
        try:
            dt = getattr(mdl, "dtype", torch.float16)
            dtype_bytes = int(getattr(dt, "itemsize", 2) or 2)
        except Exception:
            dtype_bytes = 2
        kv_bytes_per_tok = int(2 * int(getattr(cfg, "n_layer", 0) or 0) * int(getattr(cfg, "n_embd", 0) or 0) * dtype_bytes)
        ids = torch.tensor([[b % 256 for b in prompt.encode("utf-8", errors="ignore")] or [32]], dtype=torch.long, device=dev)

        # warm-up
        with torch.inference_mode():
            _ = mdl(input_ids=ids, use_cache=True)

        total_chars = 0
        with nvml_peak_memory() as nvml_peak, torch_cuda_peak_memory() as cuda_peak:
            t0 = perf_counter()
            with torch.inference_mode():
                past = None
                input_ids = ids
                for i in range(tokens):
                    cur = input_ids if past is None else input_ids[:, -1:]
                    pkv = past
                    if DynamicCache is not None and pkv is not None and not isinstance(pkv, DynamicCache):
                        try:
                            pkv = DynamicCache.from_legacy_cache(pkv)
                        except Exception:
                            pkv = past
                    out = mdl(input_ids=cur, past_key_values=pkv, use_cache=True)
                    next_id = torch.argmax(out.logits[:, -1, :], dim=-1)
                    input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)
                    past = out.past_key_values
                    v = int(next_id.item())
                    total_chars += 1 if 32 <= v < 127 else 1
                    # Optional simulated PCIe (uses globals set in main)
                    try:
                        gbps = globals().get("_SIM_PCIE_GBPS")
                        overhead = globals().get("_SIM_PCIE_OVERHEAD_US", 50.0)
                        if gbps is not None:
                            simulate_pcie_sleep(kv_bytes_per_tok, float(gbps), float(overhead))
                    except Exception:
                        pass
            dt = perf_counter() - t0
        th = total_chars / dt if dt > 0 else 0.0
        if stream:
            # Synthetic doesn't have true TTFT here; approximate as 0.
            return {
                "throughput_chars_s": th,
                "len": total_chars,
                "device": dev,
                "tokens_generated": tokens,
                "tokens_s": (tokens / dt if dt > 0 else None),
                "total_time_s": dt,
                "time_to_first_token_s": 0.0,
                "nvml_peak": (nvml_peak.as_dict() if nvml_peak is not None else None),
                "cuda_peak": cuda_peak,
            }
        return {
            "latency_s": dt,
            "len": total_chars,
            "device": dev,
            "tokens_generated": tokens,
            "nvml_peak": (nvml_peak.as_dict() if nvml_peak is not None else None),
            "cuda_peak": cuda_peak,
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from transformers import TextIteratorStreamer
    # Decide compute device early so we can pick a reasonable dtype.
    if device == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    elif device == "auto" and torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cuda" if device == "cuda" else "cpu"

    print(f"[INFO] vanilla: loading tokenizer for {model} (local_files_only={local_files_only})", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(model, local_files_only=bool(local_files_only))

    def _is_cuda_oom(e: Exception) -> bool:
        s = str(e).lower()
        return ("cuda out of memory" in s) or ("out of memory" in s and "cuda" in s)

    # For CUDA inference, fp16 is the typical baseline and avoids slow fp32 GPU transfers.
    load_torch_dtype = torch.float16 if dev == "cuda" else None
    print(f"[INFO] vanilla: loading model for {model} (torch_dtype={load_torch_dtype})", file=sys.stderr, flush=True)
    t_load0 = perf_counter()
    mdl = AutoModelForCausalLM.from_pretrained(
        model,
        use_safetensors=True,
        local_files_only=bool(local_files_only),
        low_cpu_mem_usage=True,
        torch_dtype=load_torch_dtype,
    )
    print(f"[INFO] vanilla: model loaded in {perf_counter()-t_load0:.2f}s", file=sys.stderr, flush=True)

    try:
        print(f"[INFO] vanilla: moving model to {dev}", file=sys.stderr, flush=True)
        t_mv0 = perf_counter()
        mdl.to(dev).eval()
        print(f"[INFO] vanilla: model on {dev} in {perf_counter()-t_mv0:.2f}s", file=sys.stderr, flush=True)
    except Exception as e:
        if dev == "cuda" and _is_cuda_oom(e):
            dev = "cpu"
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"[INFO] vanilla: OOM -> fallback to {dev}", file=sys.stderr, flush=True)
            mdl.to(dev).eval()
        else:
            raise
    inp = tok(prompt, return_tensors="pt").to(dev)
    # warm-up (keep minimal to avoid long stalls on large models)
    with torch.no_grad():
        try:
            print(f"[INFO] vanilla: warm-up generate", file=sys.stderr, flush=True)
            _ = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=1)
        except Exception as e:
            if dev == "cuda" and _is_cuda_oom(e):
                dev = "cpu"
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                mdl.to(dev).eval()
                inp = tok(prompt, return_tensors="pt")
                print(f"[INFO] vanilla: warm-up retry on cpu", file=sys.stderr, flush=True)
                _ = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=1)
            else:
                raise
    if stream:
        # streaming throughput (chars/s) + token-level count + peak memory
        # For simulated PCIe throttling: rough KV bytes/token.
        try:
            dt = getattr(mdl, "dtype", torch.float16)
            dtype_bytes = int(getattr(dt, "itemsize", 2) or 2)
        except Exception:
            dtype_bytes = 2
        try:
            cfg = getattr(mdl, "config", None)
            n_layer = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)) or 0)
            hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)) or 0)
        except Exception:
            n_layer, hidden = 0, 0
        kv_bytes_per_tok = int(2 * n_layer * hidden * dtype_bytes) if (n_layer > 0 and hidden > 0) else 0

        class _CountingStreamer(TextIteratorStreamer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokens_generated = 0
                self._t0 = perf_counter()
                self.time_to_first_token_s = None

            def put(self, value):  # token ids
                try:
                    n = int(value.shape[-1])
                except Exception:
                    try:
                        n = len(value)
                    except Exception:
                        n = 0
                if self.tokens_generated == 0 and self.time_to_first_token_s is None:
                    self.time_to_first_token_s = perf_counter() - self._t0
                self.tokens_generated += max(0, n)
                # Optional simulated PCIe (uses globals set in main)
                try:
                    gbps = globals().get("_SIM_PCIE_GBPS")
                    overhead = globals().get("_SIM_PCIE_OVERHEAD_US", 50.0)
                    if gbps is not None and kv_bytes_per_tok > 0:
                        simulate_pcie_sleep(int(max(1, n)) * kv_bytes_per_tok, float(gbps), float(overhead))
                except Exception:
                    pass
                return super().put(value)

        streamer = _CountingStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        total_chars = 0
        with nvml_peak_memory() as nvml_peak, torch_cuda_peak_memory() as cuda_peak:
            with torch.no_grad():
                t0 = perf_counter()
                try:
                    _ = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=tokens, streamer=streamer)
                except Exception as e:
                    if dev == "cuda" and _is_cuda_oom(e):
                        dev = "cpu"
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        mdl.to(dev).eval()
                        inp2 = tok(prompt, return_tensors="pt")
                        _ = mdl.generate(**inp2, do_sample=True, temperature=temperature, max_new_tokens=tokens, streamer=streamer)
                    else:
                        raise
                for chunk in streamer:
                    total_chars += len(chunk)
                dt = perf_counter() - t0
        th = total_chars / dt if dt > 0 else 0.0
        total_tokens = getattr(streamer, "tokens_generated", None)
        ts = (total_tokens / dt) if (total_tokens is not None and dt > 0) else None
        return {
            "throughput_chars_s": th,
            "len": total_chars,
            "device": dev,
            "tokens_generated": total_tokens,
            "tokens_s": ts,
            "total_time_s": dt,
            "time_to_first_token_s": getattr(streamer, "time_to_first_token_s", None),
            "nvml_peak": (nvml_peak.as_dict() if nvml_peak is not None else None),
            "cuda_peak": cuda_peak,
        }
    # non-stream timed (median of 3)
    times = []
    out = None
    with torch.no_grad():
        for _ in range(3):
            t0 = perf_counter()
            try:
                out = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=tokens)
            except Exception as e:
                if dev == "cuda" and _is_cuda_oom(e):
                    dev = "cpu"
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    mdl.to(dev).eval()
                    inp = tok(prompt, return_tensors="pt")
                    out = mdl.generate(**inp, do_sample=True, temperature=temperature, max_new_tokens=tokens)
                else:
                    raise
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
        # Use a better model size estimate when provided (bytes of fp16 weights).
        desired_b = globals().get("_MODEL_SIZE_B")
        if desired_b is None:
            desired_b = int(14e9)
        plan = config.budgeter(desired_model_size_b=int(desired_b), target_device=device)
        if isinstance(plan, dict):
            plan["desired_model_size_b"] = int(desired_b)
        if plan and no_quant:
            plan["quant"] = None
        if plan and no_offload:
            plan["offload"] = False
        if plan and isinstance(plan, dict):
            plan["try_fast_cuda_first"] = True

    elif strategy_name == "gpu_only":
        # Force GPU-only: no offload, optional 4bit quant; keep device cuda
        plan = {"device": "cuda", "quant": None if no_quant else "4bit", "offload": False}
        if isinstance(plan, dict):
            desired_b = globals().get("_MODEL_SIZE_B")
            if desired_b is not None:
                plan["desired_model_size_b"] = int(desired_b)

    # Optional: enable torch.compile for PyLittle (single-GPU only; set by main)
    try:
        if isinstance(plan, dict) and bool(globals().get("_PYLITTLE_COMPILE", False)):
            plan["compile_model"] = True
    except Exception:
        pass

    # Speed knobs for PyLittle path (doesn't affect vanilla).
    try:
        import torch
        if device_used == "cuda" and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass

    # Optional: benchmark may inject simulated PCIe settings into the plan dict via globals.
    # If present, hf_runtime will throttle streamer/loop accordingly.
    try:
        if isinstance(plan, dict):
            # These module globals are set in main() when sim args are provided.
            gbps = globals().get("_SIM_PCIE_GBPS")
            overhead_us = globals().get("_SIM_PCIE_OVERHEAD_US")
            sim_vram_mb = globals().get("_SIM_VRAM_MB")
            if gbps is not None:
                plan["sim_pcie_gbps"] = float(gbps)
                plan["sim_pcie_overhead_us"] = float(overhead_us) if overhead_us is not None else 50.0
            if sim_vram_mb is not None:
                plan["sim_vram_mb"] = int(sim_vram_mb)

            if bool(globals().get("_NATIVE_KV_PAGER", False)):
                plan["use_native_kv_pager"] = True
                plan["kv_pager_page_bytes"] = int(globals().get("_KV_PAGER_PAGE_BYTES", 2 * 1024 * 1024))

            if bool(globals().get("_LOCAL_FILES_ONLY", False)):
                plan["local_files_only"] = True

            # Optional: real PCIe microbench + heuristics (deep strategy for weak PCIe).
            pcie_bw = globals().get("_PCIE_BW_GBPS")
            pcie_lat = globals().get("_PCIE_LAT_US")
            if pcie_bw is not None:
                plan["pcie_bw_gbps"] = float(pcie_bw)
            if pcie_lat is not None:
                plan["pcie_latency_us"] = float(pcie_lat)

            static_force = bool(globals().get("_STATIC_LAYER_MAP_FORCE", False))
            weak = bool(globals().get("_PCIE_WEAK", False))
            if plan.get("offload") and (static_force or weak):
                plan["static_layer_map"] = True
                plan["static_cuda_layers"] = int(globals().get("_STATIC_CUDA_LAYERS", 4))
    except Exception:
        pass

    # Load once + sanity report
    print(f"[INFO] pylittle: loading model/tokenizer for {model}", file=sys.stderr, flush=True)
    model_obj, tok_obj, load_report = hf_load_model(model, device=device, strategy=plan, engine_device_hint=getattr(eng, "_device", None))
    # streaming throughput or non-stream latency
    if stream:
        # warm-up stream (minimal)
        _ = list(hf_generate(
            eng, model, prompt, max_new_tokens=1, temperature=temperature,
            stream=True, device=device, strategy=plan, paging_window=paging_window,
            model_obj=model_obj, tokenizer_obj=tok_obj,
        ))
        total_chars = 0
        with nvml_peak_memory() as nvml_peak, torch_cuda_peak_memory() as cuda_peak:
            t0 = perf_counter()
            stream_it = hf_generate(
                eng, model, prompt, max_new_tokens=tokens, temperature=temperature,
                stream=True, device=device, strategy=plan, paging_window=paging_window,
                model_obj=model_obj, tokenizer_obj=tok_obj,
            )
            chunk_steps = 0
            for chunk in stream_it:
                chunk_steps += 1
                total_chars += len(chunk)
            dt = perf_counter() - t0

        kv_pager_stats = None
        try:
            st = getattr(stream_it, "kv_pager_state", None)
            if isinstance(st, dict):
                kv_pager_stats = st.get("stats")
        except Exception:
            kv_pager_stats = None

        kv_state = None
        try:
            kv_state = getattr(stream_it, "kv_state", None)
        except Exception:
            kv_state = None

        # Token-level count if available (CountingStreamer); fallback to step count for manual decode.
        total_tokens = getattr(stream_it, "tokens_generated", None)
        if total_tokens is None:
            total_tokens = chunk_steps
        th = total_chars / dt if dt > 0 else 0.0
        ts = (total_tokens / dt) if dt > 0 else None
        return {
            "throughput_chars_s": th,
            "len": total_chars,
            "device": device_used,
            "strategy": plan,
            "paging_window": paging_window,
            "tokens_generated": total_tokens,
            "tokens_s": ts,
            "total_time_s": dt,
            "time_to_first_token_s": getattr(stream_it, "time_to_first_token_s", None),
            "kv_pager_stats": kv_pager_stats,
            "kv_state": kv_state,
            "nvml_peak": (nvml_peak.as_dict() if nvml_peak is not None else None),
            "cuda_peak": cuda_peak,
            "load": load_report,
        }
    # non-stream: warm-up then median of 3
    _ = hf_generate(eng, model, prompt, max_new_tokens=1, temperature=temperature, stream=False, device=device, strategy=plan, model_obj=model_obj, tokenizer_obj=tok_obj)
    times = []
    text = ""
    for _ in range(3):
        with stopwatch() as get_dt:
            text = hf_generate(eng, model, prompt, max_new_tokens=tokens, temperature=temperature, stream=False, device=device, strategy=plan, model_obj=model_obj, tokenizer_obj=tok_obj)
            dt = get_dt()
            if callable(dt):
                dt = dt()
        times.append(dt)
    return {"latency_s": stats.median(times), "len": len(text), "text": text, "device": device_used, "strategy": plan, "paging_window": paging_window, "load": load_report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model name. Overrides --preset if set.")
    ap.add_argument("--preset", default="tiny", choices=["synthetic", "tiny", "350m", "410m", "1b"], help="Model preset selector.")
    ap.add_argument("--prompt", default="Hello from PyLittle")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--strategy", default=None, choices=["low_vram_auto", "gpu_only"])
    ap.add_argument("--stream", action="store_true", help="Measure streaming throughput (chars/s) instead of latency.")
    ap.add_argument("--paging-window", type=int, default=None, help="Enable native KV paging (demo): recent-token window size")
    ap.add_argument("--no-quant", action="store_true", help="Disable quant in strategy plan")
    ap.add_argument("--no-offload", action="store_true", help="Disable offload in strategy plan")
    ap.add_argument("--model-size-b", type=float, default=None, help="Approx model size in bytes (fp16). Helps low_vram_auto decisions.")
    ap.add_argument("--pylittle-compile", action="store_true", help="Enable torch.compile for PyLittle when possible (single-GPU only).")
    ap.add_argument("--native-kv-pager", action="store_true", help="Enable native KVPager bookkeeping in manual decode path (prototype).")
    ap.add_argument("--kv-pager-page-bytes", type=int, default=2*1024*1024, help="Page size for native KVPager prototype.")
    ap.add_argument("--local-files-only", action="store_true", help="Only use local HF cache (no downloads).")
    ap.add_argument("--sim-vram-mb", type=int, default=None, help="Simulate a smaller GPU by limiting per-process CUDA memory (MB).")
    ap.add_argument("--sim-pcie-gen", default=None, help="Simulate PCIe generation (e.g. 1.1, 2.0, 3.0, 4.0).")
    ap.add_argument("--sim-pcie-lanes", type=int, default=None, help="Simulate PCIe lanes (e.g. 4 for x4).")
    ap.add_argument("--sim-pcie-overhead-us", type=float, default=50.0, help="Per-token PCIe latency overhead (microseconds).")
    ap.add_argument("--pcie-microbench", action="store_true", help="Measure real H2D/D2H bandwidth/latency with torch copies and include in JSON.")
    ap.add_argument("--pcie-microbench-iters", type=int, default=50, help="Iterations per size for --pcie-microbench.")
    ap.add_argument("--pcie-microbench-sizes", default="4096,1048576,67108864", help="Comma-separated sizes in bytes for --pcie-microbench.")
    ap.add_argument("--pcie-microbench-cache", action="store_true", default=True, help="(Alias; default-on) Cache PCIe microbench results in outputs/.")
    ap.add_argument("--pcie-microbench-no-cache", action="store_true", help="Disable PCIe microbench caching.")
    ap.add_argument("--pcie-microbench-cache-ttl-s", type=float, default=7*24*3600, help="TTL seconds for --pcie-microbench-cache.")
    ap.add_argument("--pcie-weak-threshold-gbps", type=float, default=2.0, help="If measured PCIe GB/s is below this and offload is on, enable static layer map.")
    ap.add_argument("--static-layer-map", action="store_true", help="Force static layer pinning map when offload is on (requires accelerate).")
    ap.add_argument("--static-cuda-layers", type=int, default=4, help="How many early transformer layers to pin on CUDA for static layer map.")
    args = ap.parse_args()

    # Globals used to pass auxiliary info into helpers without changing many signatures.
    global _PCIE_BW_GBPS, _PCIE_LAT_US, _PCIE_WEAK
    global _STATIC_LAYER_MAP_FORCE, _STATIC_CUDA_LAYERS

    pcie_report = None
    if bool(args.pcie_microbench):
        try:
            sizes = [int(x.strip()) for x in str(args.pcie_microbench_sizes).split(",") if x.strip()]
        except Exception:
            sizes = [4096, 1 << 20, 64 << 20]
        if not bool(args.pcie_microbench_no_cache):
            pcie_report = pcie_microbench_cached(
                cache_path=None,
                ttl_s=float(args.pcie_microbench_cache_ttl_s),
                device="cuda",
                sizes_bytes=sizes,
                iters=int(args.pcie_microbench_iters),
                warmup=10,
                pinned=True,
            )
        else:
            # No-cache mode: write to a per-run cache file so nothing is reused.
            import time as _time
            pcie_report = pcie_microbench_cached(
                cache_path=f"outputs/pcie_microbench_cache_run_{int(_time.time())}.json",
                ttl_s=0.0,
                device="cuda",
                sizes_bytes=sizes,
                iters=int(args.pcie_microbench_iters),
                warmup=10,
                pinned=True,
            )

        # Derive a single "effective" bandwidth and latency heuristic.
        try:
            h2d = pcie_report.get("h2d_gbps") or {}
            d2h = pcie_report.get("d2h_gbps") or {}
            # Use the largest tested size as the steady-state bandwidth proxy.
            szs = [int(x) for x in (pcie_report.get("sizes_bytes") or [])]
            max_sz = max(szs) if szs else (64 << 20)
            bw_h2d = float(h2d.get(str(max_sz))) if h2d.get(str(max_sz)) is not None else None
            bw_d2h = float(d2h.get(str(max_sz))) if d2h.get(str(max_sz)) is not None else None
            eff_bw = None
            if bw_h2d is not None and bw_d2h is not None:
                eff_bw = min(bw_h2d, bw_d2h)
            elif bw_h2d is not None:
                eff_bw = bw_h2d
            elif bw_d2h is not None:
                eff_bw = bw_d2h

            lat_h2d = (pcie_report.get("h2d_latency_us") or {}).get("4096")
            lat_d2h = (pcie_report.get("d2h_latency_us") or {}).get("4096")
            eff_lat = None
            if lat_h2d is not None and lat_d2h is not None:
                eff_lat = max(float(lat_h2d), float(lat_d2h))
            elif lat_h2d is not None:
                eff_lat = float(lat_h2d)
            elif lat_d2h is not None:
                eff_lat = float(lat_d2h)
        except Exception:
            eff_bw = None
            eff_lat = None

        _PCIE_BW_GBPS = eff_bw
        _PCIE_LAT_US = eff_lat
        try:
            _PCIE_WEAK = (eff_bw is not None) and (float(eff_bw) < float(args.pcie_weak_threshold_gbps))
        except Exception:
            _PCIE_WEAK = False
    else:
        _PCIE_BW_GBPS = None
        _PCIE_LAT_US = None
        _PCIE_WEAK = False

    _STATIC_LAYER_MAP_FORCE = bool(args.static_layer_map)
    _STATIC_CUDA_LAYERS = int(args.static_cuda_layers)

    # resolve preset -> model
    preset_map = {
        "synthetic": "synthetic-gpt2",
        "tiny": "sshleifer/tiny-gpt2",
        "350m": "facebook/opt-350m",
        "410m": "EleutherAI/pythia-410m-deduped",
        "1b": "microsoft/phi-1_5",  # lightweight 1.3B-class alternative could be used
    }
    model = args.model or preset_map[args.preset]

    # Provide a better size estimate for low_vram_auto (fp16 bytes)
    preset_size_b = {
        "synthetic": int(50e6),
        "tiny": int(120e6),
        "350m": int(700e6),
        "410m": int(820e6),
        "1b": int(3_000_000_000),
    }
    global _MODEL_SIZE_B, _PYLITTLE_COMPILE
    _MODEL_SIZE_B = int(args.model_size_b) if args.model_size_b is not None else preset_size_b.get(args.preset)
    _PYLITTLE_COMPILE = bool(args.pylittle_compile)

    global _NATIVE_KV_PAGER, _KV_PAGER_PAGE_BYTES
    _NATIVE_KV_PAGER = bool(args.native_kv_pager)
    _KV_PAGER_PAGE_BYTES = int(args.kv_pager_page_bytes)

    global _LOCAL_FILES_ONLY
    _LOCAL_FILES_ONLY = bool(args.local_files_only)

    sim_pcie_gbps = None
    if args.sim_pcie_gen is not None and args.sim_pcie_lanes is not None:
        try:
            sim_pcie_gbps = pcie_gen_lanes_to_gbps(args.sim_pcie_gen, args.sim_pcie_lanes)
        except Exception:
            sim_pcie_gbps = None

    # If user provided a simulated PCIe setting (and we didn't run a real microbench),
    # use it as the heuristic signal for "weak PCIe".
    try:
        if (_PCIE_BW_GBPS is None) and (sim_pcie_gbps is not None):
            _PCIE_BW_GBPS = float(sim_pcie_gbps)
            _PCIE_LAT_US = None
            _PCIE_WEAK = float(sim_pcie_gbps) < float(args.pcie_weak_threshold_gbps)
    except Exception:
        pass

    # Expose to run_pylittle() without changing its signature.
    global _SIM_PCIE_GBPS, _SIM_PCIE_OVERHEAD_US, _SIM_VRAM_MB
    _SIM_PCIE_GBPS = sim_pcie_gbps
    _SIM_PCIE_OVERHEAD_US = args.sim_pcie_overhead_us
    _SIM_VRAM_MB = args.sim_vram_mb

    with simulated_cuda_vram_limit(args.sim_vram_mb) as sim_vram_report:
        base_gpu = gpu_stats()
        v = run_vanilla(model, args.prompt, args.tokens, args.temperature, args.device, stream=args.stream, local_files_only=bool(args.local_files_only))
        mid_gpu = gpu_stats()

        # Pass PCIe throttling into the HF strategy dict (only affects HF adapter paths).
        if sim_pcie_gbps is not None:
            if isinstance(v, dict):
                v.setdefault("simulation", {})
                v["simulation"]["pcie_gbps"] = sim_pcie_gbps

        p = run_pylittle(model, args.prompt, args.tokens, args.temperature, args.device, args.strategy,
                         stream=args.stream, paging_window=args.paging_window,
                         no_quant=args.no_quant, no_offload=args.no_offload)
        end_gpu = gpu_stats()

    # Add sim config into the pylittle strategy for throttling
    try:
        if sim_pcie_gbps is not None:
            st = p.get("strategy") if isinstance(p, dict) else None
            if st is None:
                st = {}
                if isinstance(p, dict):
                    p["strategy"] = st
            if isinstance(st, dict):
                st["sim_pcie_gbps"] = sim_pcie_gbps
                st["sim_pcie_overhead_us"] = float(args.sim_pcie_overhead_us)
                # also in return payload for visibility
                p["strategy"] = st
    except Exception:
        pass

    speedup = None
    if not args.stream and (v.get("latency_s") and p.get("latency_s")):
        if v["latency_s"] > 0 and p["latency_s"] > 0:
            speedup = v["latency_s"] / p["latency_s"]

    out = {}
    if args.stream:
        def _split(prefill_s, total_s, tokens_gen):
            try:
                if prefill_s is None or total_s is None:
                    return {"prefill_s": prefill_s, "decode_s": None, "decode_tokens_s": None}
                pre = float(prefill_s)
                tot = float(total_s)
                dec = max(0.0, tot - pre)
                if tokens_gen is None or dec <= 0:
                    return {"prefill_s": pre, "decode_s": dec, "decode_tokens_s": None}
                return {"prefill_s": pre, "decode_s": dec, "decode_tokens_s": (float(tokens_gen) / dec)}
            except Exception:
                return {"prefill_s": prefill_s, "decode_s": None, "decode_tokens_s": None}

        v_split = _split(v.get("time_to_first_token_s"), v.get("total_time_s"), v.get("tokens_generated"))
        p_split = _split(p.get("time_to_first_token_s"), p.get("total_time_s"), p.get("tokens_generated"))
        out.update({
            "vanilla": {
                "throughput_chars_s": round(v.get("throughput_chars_s", 0.0), 2),
                "len": v.get("len"),
                "tokens_generated": v.get("tokens_generated"),
                "tokens_s": (round(v.get("tokens_s", 0.0), 2) if v.get("tokens_s") else None),
                "prefill_s": v_split.get("prefill_s"),
                "decode_s": v_split.get("decode_s"),
                "decode_tokens_s": v_split.get("decode_tokens_s"),
                "time_to_first_token_s": v.get("time_to_first_token_s"),
                "nvml_peak": v.get("nvml_peak"),
                "cuda_peak": v.get("cuda_peak"),
            },
            "pylittle": {
                "throughput_chars_s": round(p.get("throughput_chars_s", 0.0), 2),
                "len": p.get("len"),
                "tokens_requested": args.tokens,
                "tokens_generated": p.get("tokens_generated"),
                "tokens_s": (round(p.get("tokens_s", 0.0), 2) if p.get("tokens_s") else None),
                "prefill_s": p_split.get("prefill_s"),
                "decode_s": p_split.get("decode_s"),
                "decode_tokens_s": p_split.get("decode_tokens_s"),
                "time_to_first_token_s": p.get("time_to_first_token_s"),
                "kv_pager_stats": p.get("kv_pager_stats"),
                "kv_state": p.get("kv_state"),
                "nvml_peak": p.get("nvml_peak"),
                "cuda_peak": p.get("cuda_peak"),
            },
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
        "pylittle_load": p.get("load"),
        "gpu_before": base_gpu,
        "gpu_mid": mid_gpu,
        "gpu_after": end_gpu,
        "simulation": {
            "sim_vram": sim_vram_report,
            "sim_pcie": {
                "gen": args.sim_pcie_gen,
                "lanes": args.sim_pcie_lanes,
                "gbps": sim_pcie_gbps,
                "overhead_us": args.sim_pcie_overhead_us,
            },
        },
    })

    if pcie_report is not None:
        out["pcie_microbench"] = pcie_report
    # VRAM delta if stats available
    try:
        if base_gpu and end_gpu:
            out["vram_delta_mb"] = round(end_gpu["mem_used_mb"] - base_gpu["mem_used_mb"], 2)
    except Exception:
        pass

    # Would-OOM heuristic under simulated VRAM: compare observed NVML peak vs limit.
    try:
        lim = args.sim_vram_mb
        if lim is not None:
            v_peak = None
            p_peak = None
            try:
                v_peak = (out.get("vanilla") or {}).get("nvml_peak", {}).get("peak_mb")
            except Exception:
                v_peak = None
            try:
                p_peak = (out.get("pylittle") or {}).get("nvml_peak", {}).get("peak_mb")
            except Exception:
                p_peak = None
            out["would_oom"] = {
                "vanilla": (v_peak is not None and float(v_peak) > float(lim)),
                "pylittle": (p_peak is not None and float(p_peak) > float(lim)),
                "limit_mb": lim,
                "vanilla_peak_mb": v_peak,
                "pylittle_peak_mb": p_peak,
            }
    except Exception:
        pass

    # Verdict gate: under sim VRAM, PyLittle should (a) fit and (b) be faster.
    try:
        verdict = {"pass": None, "reasons": []}
        wou = out.get("would_oom") or {}
        if isinstance(wou, dict) and wou.get("limit_mb") is not None:
            if wou.get("pylittle") is True:
                verdict["pass"] = False
                verdict["reasons"].append("pylittle_would_oom_under_sim_vram")
            if wou.get("vanilla") is True:
                verdict["reasons"].append("vanilla_would_oom_under_sim_vram")

        v_ts = None
        p_ts = None
        try:
            v_ts = (out.get("vanilla") or {}).get("tokens_s")
        except Exception:
            v_ts = None
        try:
            p_ts = (out.get("pylittle") or {}).get("tokens_s")
        except Exception:
            p_ts = None
        # Speed gate is only meaningful in the default (non-paging-window) mode.
        if (args.paging_window is None) and p_ts is not None and v_ts is not None:
            if float(p_ts) <= float(v_ts):
                verdict["pass"] = False if verdict["pass"] is not True else verdict["pass"]
                verdict["reasons"].append("pylittle_tokens_s_not_greater_than_vanilla")
            else:
                if verdict["pass"] is None:
                    verdict["pass"] = True

        # Dependency sanity: if user requested quant/offload, ensure it actually activated.
        try:
            st = out.get("strategy") if isinstance(out, dict) else None
            lr = out.get("pylittle_load") if isinstance(out, dict) else None
            if isinstance(st, dict) and isinstance(lr, dict):
                qreq = st.get("quant")
                if qreq in ("4bit", "8bit"):
                    qinfo = lr.get("quant") or {}
                    if isinstance(qinfo, dict) and not bool(qinfo.get("quant_active")):
                        verdict["pass"] = False if verdict["pass"] is not True else verdict["pass"]
                        verdict["reasons"].append("quant_requested_but_inactive")
                if bool(st.get("offload")):
                    if not bool(lr.get("used_device_map")):
                        verdict["pass"] = False if verdict["pass"] is not True else verdict["pass"]
                        verdict["reasons"].append("offload_requested_but_device_map_unused")
        except Exception:
            pass

        # If we have a sim VRAM gate, require pylittle fit.
        if isinstance(wou, dict) and wou.get("limit_mb") is not None and wou.get("pylittle") is False:
            if verdict["pass"] is None:
                verdict["pass"] = True
        out["verdict"] = verdict
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
