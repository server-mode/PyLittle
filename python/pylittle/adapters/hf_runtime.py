"""Hugging Face runtime adapter for PyLittle.

This is a bridge layer for running Transformers models while we build native backends.

Key paths:
- Default: `model.generate(...)` (+ streamer)
- KV-window: token-by-token loop with `past_key_values` truncation (single-device only)
- Synthetic: fully-offline tiny GPT-2-like model for smoke testing without HF downloads
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional
import time


try:  # optional native async copy
    from pylittle._pylittle import MemoryManager as _NativeMM, KVPager as _NativePager  # type: ignore
except Exception:  # pragma: no cover
    _NativeMM = None
    _NativePager = None


def _pick_device(engine_device: str | None = None) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _sample_next_token(logits, temperature: float):
    import torch

    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / float(temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _estimate_kv_bytes_per_token(model) -> int:
    """Rough estimate of KV bytes per generated token for batch=1.

    Used only for simulated PCIe throttling.
    """
    try:
        import torch

        cfg = getattr(model, "config", None)
        n_layer = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)) or 0)
        hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)) or 0)
        if n_layer <= 0 or hidden <= 0:
            return 0
        dt = getattr(model, "dtype", torch.float16)
        dtype_bytes = int(getattr(dt, "itemsize", 2) or 2)
        # K and V per layer per token ~ hidden_size each.
        return int(2 * n_layer * hidden * dtype_bytes)
    except Exception:
        return 0


def _maybe_simulate_pcie(strategy: Optional[Dict[str, Any]], n_tokens: int, kv_bytes_per_token: int) -> None:
    if not strategy or not isinstance(strategy, dict):
        return
    gbps = strategy.get("sim_pcie_gbps")
    if gbps is None:
        return
    try:
        gbps_f = float(gbps)
    except Exception:
        return
    if gbps_f <= 0:
        return
    overhead_us = strategy.get("sim_pcie_overhead_us", 50.0)
    try:
        overhead_us_f = float(overhead_us)
    except Exception:
        overhead_us_f = 50.0
    # Approximate: per token, transfer its KV footprint once.
    bytes_total = int(max(0, n_tokens)) * int(max(0, kv_bytes_per_token))
    if bytes_total <= 0:
        # Still apply fixed per-token overhead to represent latency.
        bytes_total = 0
    # seconds = bytes / (GB/s) + overhead
    seconds = (float(bytes_total) / (gbps_f * 1e9)) + (overhead_us_f * 1e-6 * float(max(1, n_tokens)))
    if seconds > 0:
        time.sleep(seconds)


class _StreamWithMetrics(Iterator[str]):
    def __init__(self, it: Iterator[str]):
        self._it = it
        self.tokens_generated: int = 0
        self.time_to_first_token_s: Optional[float] = None
        self._t0 = time.perf_counter()
        self.kv_pager_state: Optional[dict] = None

    def __iter__(self):
        return self

    def __next__(self) -> str:
        v = next(self._it)
        if self.tokens_generated == 0 and self.time_to_first_token_s is None:
            self.time_to_first_token_s = time.perf_counter() - self._t0
        # For manual decode we treat each yielded chunk as >=1 token.
        self.tokens_generated += 1
        return v


def _get_strategy_kw(strategy: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if not strategy or not isinstance(strategy, dict):
        return default
    return strategy.get(key, default)


def _maybe_to_cache(past):
    if past is None:
        return None
    try:
        from transformers.cache_utils import DynamicCache  # type: ignore

        if isinstance(past, DynamicCache):
            return past
        # legacy tuple/list -> DynamicCache
        return DynamicCache.from_legacy_cache(past)
    except Exception:
        return past


def _cache_seq_len(past, head_candidates: tuple[int, ...] | None) -> int:
    if past is None:
        return 0
    # Cache API
    try:
        if hasattr(past, "get_seq_length"):
            v = past.get_seq_length()
            if isinstance(v, int):
                return v
    except Exception:
        pass
    # Legacy tuple
    try:
        if isinstance(past, (tuple, list)) and len(past) > 0:
            pk = past[0][0]
            if getattr(pk, "dim", lambda: 0)() == 4:
                seq_dim = _infer_seq_dim_from_heads(pk, head_candidates)
                return int(pk.shape[seq_dim])
    except Exception:
        pass
    return 0


def _maybe_crop_cache(past, window: int):
    if past is None or window is None or window <= 0:
        return past
    # Cache API
    try:
        if hasattr(past, "crop"):
            out = past.crop(window)
            return out if out is not None else past
    except Exception:
        pass
    return past


def _input_device_for_model(model, fallback: str) -> str:
    # Best-effort: figure out where inputs should live when using device_map.
    try:
        emb = getattr(model, "get_input_embeddings", None)
        if callable(emb):
            w = emb().weight
            d = getattr(w, "device", None)
            if d is not None and str(d) != "meta":
                return str(d)
    except Exception:
        pass
    try:
        for p in model.parameters():
            d = getattr(p, "device", None)
            if d is not None and str(d) != "meta":
                return str(d)
            break
    except Exception:
        pass
    return fallback


def _summarize_hf_device_map(model) -> Dict[str, int] | None:
    try:
        dm = getattr(model, "hf_device_map", None)
        if not isinstance(dm, dict):
            return None
        out: Dict[str, int] = {}
        for _, v in dm.items():
            k = str(v)
            out[k] = out.get(k, 0) + 1
        return out
    except Exception:
        return None


def _get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in str(path).split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _build_static_device_map(model: Any, cuda_layers: int) -> tuple[Dict[str, str] | None, list[str]]:
    """Build a deterministic (static) layer pinning map.

    This is a best-effort helper to reduce offload thrash on weak PCIe by pinning a prefix
    of transformer layers on CUDA and keeping the rest on CPU.

    Returns: (device_map, warnings)
    """
    warnings: list[str] = []
    try:
        n = int(cuda_layers)
    except Exception:
        n = 0
    if n <= 0:
        warnings.append("static_layer_map requested but static_cuda_layers<=0")
        return None, warnings

    blocks = None
    prefix = None
    try:
        if _get_attr_path(model, "model.layers") is not None:
            blocks = _get_attr_path(model, "model.layers")
            prefix = "model.layers"
        elif _get_attr_path(model, "transformer.h") is not None:
            blocks = _get_attr_path(model, "transformer.h")
            prefix = "transformer.h"
        elif _get_attr_path(model, "gpt_neox.layers") is not None:
            blocks = _get_attr_path(model, "gpt_neox.layers")
            prefix = "gpt_neox.layers"
        elif _get_attr_path(model, "model.decoder.layers") is not None:
            blocks = _get_attr_path(model, "model.decoder.layers")
            prefix = "model.decoder.layers"
    except Exception:
        blocks = None
        prefix = None

    if blocks is None or prefix is None:
        warnings.append("static_layer_map unsupported for this model architecture (no known block prefix)")
        return None, warnings

    try:
        total_layers = len(blocks)
    except Exception:
        total_layers = 0
    if total_layers <= 0:
        warnings.append("static_layer_map could not determine number of layers")
        return None, warnings

    n = min(n, total_layers)
    dm: Dict[str, str] = {"": "cpu"}

    embed_candidates = (
        "model.embed_tokens",
        "model.decoder.embed_tokens",
        "transformer.wte",
        "transformer.wpe",
        "gpt_neox.embed_in",
        "embed_tokens",
    )
    for name in embed_candidates:
        try:
            if _get_attr_path(model, name) is not None:
                dm[name] = "cuda"
        except Exception:
            pass

    for i in range(n):
        dm[f"{prefix}.{i}"] = "cuda"

    tail_dev = "cuda" if n >= total_layers else "cpu"
    tail_candidates = (
        "lm_head",
        "model.norm",
        "model.decoder.final_layer_norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
    )
    for name in tail_candidates:
        try:
            if _get_attr_path(model, name) is not None:
                dm[name] = tail_dev
        except Exception:
            pass

    warnings.append(f"static_layer_map applied: cuda_layers={n}/{total_layers}")
    return dm, warnings


def _auto_static_cuda_layers(model: Any, strategy: Optional[Dict[str, Any]]) -> tuple[int, list[str]]:
    """Choose a reasonable default for static layer pinning.

    Uses:
    - model layer count (best-effort)
    - GPU budget (sim_vram_mb if provided, else torch device total)
    - desired_model_size_b (fp16 bytes) + quant factor if provided
    """

    warnings: list[str] = []
    total_layers = 0
    # Find transformer blocks to get layer count.
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        try:
            blocks = _get_attr_path(model, path)
            if blocks is not None:
                total_layers = int(len(blocks))
                break
        except Exception:
            continue

    if total_layers <= 0:
        return 4, ["static_layer_map auto: could not determine layer count; defaulting to 4"]

    # VRAM budget
    budget_b = None
    try:
        if strategy and isinstance(strategy, dict) and strategy.get("sim_vram_mb") is not None:
            mb = int(strategy.get("sim_vram_mb"))
            if mb > 0:
                budget_b = float(mb) * 1024.0 * 1024.0
                warnings.append(f"static_layer_map auto: using sim_vram_mb={mb}")
    except Exception:
        budget_b = None

    if budget_b is None:
        try:
            import torch

            if torch.cuda.is_available():
                total = float(torch.cuda.get_device_properties(0).total_memory)
                budget_b = total
        except Exception:
            budget_b = None

    if budget_b is None or budget_b <= 0:
        return min(4, total_layers), ["static_layer_map auto: cuda VRAM unknown; defaulting to 4"]

    # Use a conservative fraction for weights residency.
    weight_budget_b = float(budget_b) * 0.65

    # Model size estimate
    desired_b = None
    try:
        if strategy and isinstance(strategy, dict) and strategy.get("desired_model_size_b") is not None:
            desired_b = float(strategy.get("desired_model_size_b"))
    except Exception:
        desired_b = None

    if desired_b is None or desired_b <= 0:
        # No size estimate; pick a small-but-not-tiny fraction.
        n = max(1, min(total_layers, 4))
        warnings.append("static_layer_map auto: desired_model_size_b missing; defaulting to 4")
        return n, warnings

    # Adjust for quant if requested (roughly).
    try:
        q = strategy.get("quant") if (strategy and isinstance(strategy, dict)) else None
        if q == "4bit":
            desired_b = desired_b * 0.25
        elif q == "8bit":
            desired_b = desired_b * 0.5
    except Exception:
        pass

    per_layer_b = float(desired_b) / float(max(1, total_layers))
    # Reserve some headroom.
    per_layer_b = max(per_layer_b, 1.0)
    n_layers = int(weight_budget_b // per_layer_b)
    n_layers = max(1, min(total_layers, n_layers))

    warnings.append(f"static_layer_map auto: chose cuda_layers={n_layers}/{total_layers} (budget~{weight_budget_b/1e9:.2f}GB)")
    return n_layers, warnings


def _quant_sanity(model, quant_requested: str | None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "quant_requested": quant_requested,
        "bnb_available": None,
        "quant_active": False,
        "details": {},
    }
    if not quant_requested:
        return info

    bnb_ok = False
    try:
        import bitsandbytes as bnb  # noqa: F401

        bnb_ok = True
    except Exception:
        bnb_ok = False
    info["bnb_available"] = bnb_ok

    # Fast-path flags used by Transformers
    try:
        info["details"]["is_loaded_in_4bit"] = bool(getattr(model, "is_loaded_in_4bit", False))
        info["details"]["is_loaded_in_8bit"] = bool(getattr(model, "is_loaded_in_8bit", False))
    except Exception:
        pass

    active = bool(info["details"].get("is_loaded_in_4bit")) or bool(info["details"].get("is_loaded_in_8bit"))

    # Module scan as fallback
    if not active and bnb_ok:
        try:
            from bitsandbytes.nn import Linear4bit, Linear8bitLt  # type: ignore

            has_4 = False
            has_8 = False
            for m in model.modules():
                if isinstance(m, Linear4bit):
                    has_4 = True
                if isinstance(m, Linear8bitLt):
                    has_8 = True
                if has_4 and has_8:
                    break
            info["details"]["has_bnb_linear4bit"] = has_4
            info["details"]["has_bnb_linear8bit"] = has_8
            active = has_4 or has_8
        except Exception:
            pass

    info["quant_active"] = active
    return info


def hf_load_model(
    model_name_or_path: str,
    *,
    device: str = "auto",
    strategy: Optional[Dict[str, Any]] = None,
    engine_device_hint: str | None = None,
):
    """Load HF model+tokenizer with a low-VRAM plan and return a sanity report.

    Returns: (model, tokenizer, report)
    """
    import torch
    import sys

    def _jsonable(x: Any):
        # Best-effort conversion for reporting only.
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        try:
            if isinstance(x, (list, tuple)):
                return [_jsonable(v) for v in x]
            if isinstance(x, dict):
                return {str(k): _jsonable(v) for k, v in x.items()}
        except Exception:
            pass
        # torch.dtype and many HF objects are not JSON serializable.
        try:
            import torch as _torch

            if isinstance(x, getattr(_torch, "dtype", ())):
                return str(x)
        except Exception:
            pass
        return str(x)

    # Device normalization (match hf_generate semantics)
    if strategy and isinstance(strategy, dict) and strategy.get("device") in ("cpu", "cuda"):
        dev = str(strategy.get("device"))
    elif device is None or str(device).lower() == "auto":
        dev = _pick_device(engine_device_hint)
    else:
        dev = str(device).lower()

    try:
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
    except Exception:
        dev = "cpu"

    is_synthetic = str(model_name_or_path).lower() in ("synthetic-gpt2", "synthetic")
    used_device_map = False
    defer_static_dispatch = False
    static_cuda_layers = None
    report: Dict[str, Any] = {
        "model": model_name_or_path,
        "device": dev,
        "is_synthetic": is_synthetic,
        "used_device_map": False,
        "device_map_summary": None,
        "quant": None,
        "warnings": [],
        "load_kwargs": None,
        "requested": None,
        "static_layer_map": None,
        "static_cuda_layers": None,
    }

    if is_synthetic:
        model, tokenizer = _make_synthetic_gpt2(dev)
        report["model_class"] = "synthetic-gpt2"
        report["is_encoder_decoder"] = False
        report["quant"] = _quant_sanity(model, None)
        return model, tokenizer, report

    # Many text-only models still transitively touch vision utilities inside transformers.
    # On small Windows setups this can pull in torchvision/sympy and make imports very slow.
    # If users need vision models they can unset this env var.
    try:
        import os

        os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    except Exception:
        pass

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    def _is_cuda_oom(e: Exception) -> bool:
        s = str(e).lower()
        return ("cuda out of memory" in s) or ("out of memory" in s and "cuda" in s)

    def _system_total_ram_mb() -> int | None:
        # Best-effort physical RAM detection (Windows-friendly).
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) == 0:
                return None
            return int(st.ullTotalPhys // (1024 * 1024))
        except Exception:
            return None

    def _parse_mem_to_mb(v: Any) -> int | None:
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                # Assume MB.
                return int(v)
            s = str(v).strip().lower()
            if s.endswith("gib"):
                return int(float(s[:-3].strip()) * 1024)
            if s.endswith("gb"):
                return int(float(s[:-2].strip()) * 1000)
            if s.endswith("mib"):
                return int(float(s[:-3].strip()))
            if s.endswith("mb"):
                return int(float(s[:-2].strip()))
            return int(float(s))
        except Exception:
            return None

    def _format_mb(mb: int) -> str:
        return f"{int(mb)}MiB"

    def _headroom_mb_for_budget(budget_mb: int) -> int:
        if budget_mb >= 8192:
            return 1024
        if budget_mb >= 6144:
            return 768
        return 512

    def _override_max_memory_for_sim(load_kwargs: Dict[str, Any], sim_vram_mb: int):
        try:
            mm = load_kwargs.get("max_memory")
            if not isinstance(mm, dict):
                mm = {}
            safe_gpu_mb = max(512, int(sim_vram_mb) - _headroom_mb_for_budget(int(sim_vram_mb)))
            mm[0] = _format_mb(safe_gpu_mb)
            total_ram_mb = _system_total_ram_mb()
            if total_ram_mb is not None and total_ram_mb > 0:
                cpu_cap_mb = max(8192, int(total_ram_mb * 0.85))
                cpu_cap_mb = min(cpu_cap_mb, 128 * 1024)
                mm["cpu"] = _format_mb(cpu_cap_mb)
            else:
                mm["cpu"] = "64GiB"
            load_kwargs["max_memory"] = mm
            report["warnings"].append(
                f"sim_vram_mb={int(sim_vram_mb)} overrides max_memory -> gpu={mm.get(0)} cpu={mm.get('cpu')}"
            )
        except Exception:
            pass

    try:
        local_only = bool(_get_strategy_kw(strategy, "local_files_only", False))
        offload_req = bool(_get_strategy_kw(strategy, "offload", False))
        quant_req = _get_strategy_kw(strategy, "quant", None)
        report["requested"] = {
            "local_files_only": local_only,
            "offload": offload_req,
            "quant": quant_req,
        }
        print(
            f"[INFO] pylittle.hf: hf_load_model start model={model_name_or_path} dev={dev} local_only={local_only} offload={offload_req} quant={quant_req}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass

    # If offload was requested, Transformers typically relies on accelerate's device_map plumbing.
    # If accelerate is missing, prefer a safe CPU fallback (rather than silently ignoring offload
    # and then OOM-ing on CUDA).
    offload_requested = bool(_get_strategy_kw(strategy, "offload", False))
    if offload_requested:
        try:
            import accelerate  # noqa: F401
        except Exception:
            report["warnings"].append("accelerate not available; offload disabled and forcing CPU compute")
            # Force compute device to CPU for this load to avoid accidental .to('cuda') later.
            dev = "cpu"
            report["device"] = dev

    tokenizer_kwargs: Dict[str, Any] = {}
    for k in ("trust_remote_code", "revision", "token", "local_files_only"):
        v = _get_strategy_kw(strategy, k, None)
        if v is not None:
            tokenizer_kwargs[k] = v
    try:
        print(f"[INFO] pylittle.hf: loading tokenizer ({tokenizer_kwargs})", file=sys.stderr, flush=True)
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    try:
        print("[INFO] pylittle.hf: tokenizer loaded", file=sys.stderr, flush=True)
    except Exception:
        pass

    load_kwargs: Dict[str, Any] = {"use_safetensors": True, "low_cpu_mem_usage": True}
    for k in ("trust_remote_code", "revision", "token", "local_files_only", "attn_implementation"):
        v = _get_strategy_kw(strategy, k, None)
        if v is not None:
            load_kwargs[k] = v
    td = _get_strategy_kw(strategy, "torch_dtype", None)
    if td is not None:
        try:
            if isinstance(td, str):
                # Transformers accepts torch_dtype="auto".
                if td.strip().lower() != "auto":
                    td = getattr(torch, td)
        except Exception:
            pass
        load_kwargs["torch_dtype"] = td

    # PyLittle default perf/memory knobs (only when CUDA):
    # - Prefer fp16 (unless using bnb quant)
    # - Prefer FlashAttention2 if available, else SDPA
    def _flash_attn_available() -> bool:
        try:
            import flash_attn  # noqa: F401

            return True
        except Exception:
            return False

    if dev == "cuda":
        try:
            if "attn_implementation" not in load_kwargs:
                load_kwargs["attn_implementation"] = "flash_attention_2" if _flash_attn_available() else "sdpa"
        except Exception:
            pass

    offload_req_for_dtype = bool(_get_strategy_kw(strategy, "offload", False))
    quant_requested = _get_strategy_kw(strategy, "quant", None)
    # If no explicit dtype and not using bnb quant, default to fp16 on CUDA.
    if dev == "cuda" and "torch_dtype" not in load_kwargs and quant_requested not in ("4bit", "8bit"):
        try:
            # With device_map/offload, dtype casting during shard load can be expensive and
            # can cause large transient allocations. Prefer torch_dtype="auto".
            load_kwargs["torch_dtype"] = "auto" if offload_req_for_dtype else torch.float16
        except Exception:
            pass

    try_fast_first = bool(_get_strategy_kw(strategy, "try_fast_cuda_first", False))
    if quant_requested in ("4bit", "8bit"):
        # Only enable if bnb import succeeds; sanity check later will confirm.
        try:
            import bitsandbytes as bnb  # noqa: F401

            load_kwargs.update({
                "load_in_4bit": quant_requested == "4bit",
                "load_in_8bit": quant_requested == "8bit",
            })
        except Exception:
            report["warnings"].append("bitsandbytes not available; quant disabled")
            quant_requested = None

    if _get_strategy_kw(strategy, "offload", False) and dev != "cpu":
        # Optional deterministic mapping: CPU load first, then dispatch with a static device_map.
        # This is intended for weak PCIe setups where auto/offload can cause severe thrash.
        try:
            defer_static_dispatch = bool(_get_strategy_kw(strategy, "static_layer_map", False))
        except Exception:
            defer_static_dispatch = False
        static_cuda_layers = _get_strategy_kw(strategy, "static_cuda_layers", None)

        if defer_static_dispatch:
            try:
                import accelerate  # noqa: F401
            except Exception:
                defer_static_dispatch = False
                report["warnings"].append("static_layer_map requested but accelerate not available; using device_map='auto'")

        if defer_static_dispatch:
            # Incompatible with try_fast_first: we explicitly want a device_map path.
            try_fast_first = False
            used_device_map = True
        else:
            load_kwargs["device_map"] = "auto"
            if _get_strategy_kw(strategy, "max_memory", None):
                raw_mm = strategy["max_memory"]
                mm: Dict[Any, Any] = {}
                for k, v in raw_mm.items():
                    if isinstance(k, str) and k.lower() == "cuda":
                        mm[0] = v
                    elif isinstance(k, str) and k.isdigit():
                        mm[int(k)] = v
                    else:
                        mm[k] = v
                load_kwargs["max_memory"] = mm
            else:
                # If user is simulating a smaller GPU, enforce max_memory accordingly.
                sim_vram_mb = _get_strategy_kw(strategy, "sim_vram_mb", None)
                if sim_vram_mb is not None:
                    try:
                        sim_mb = int(sim_vram_mb)
                        if sim_mb > 0:
                            # Keep a safety margin for allocator fragmentation.
                            gpu_gib = max(1, int((sim_mb / 1024.0) * 0.90))
                            load_kwargs["max_memory"] = {0: f"{gpu_gib}GiB", "cpu": "64GiB"}
                            report["warnings"].append(f"max_memory auto-set from sim_vram_mb={sim_mb}")
                    except Exception:
                        pass
            used_device_map = True

        # If we're simulating VRAM, force the GPU cap even if a profile provided max_memory.
        try:
            sim_vram_mb2 = _get_strategy_kw(strategy, "sim_vram_mb", None)
            if sim_vram_mb2 is not None:
                sim_mb2 = int(sim_vram_mb2)
                if sim_mb2 > 0:
                    _override_max_memory_for_sim(load_kwargs, sim_mb2)
                    # Under a simulated low VRAM cap, a fast full-GPU attempt is very likely
                    # to OOM and can fragment the allocator. Prefer direct device_map load.
                    try_fast_first = False
                    report["warnings"].append("try_fast_cuda_first disabled due to sim_vram_mb")
        except Exception:
            pass

    def _load_with_retry(kwargs: Dict[str, Any]):
        # Only fall back to Seq2Seq when the config/type indicates that's required.
        model_class_local = "AutoModelForCausalLM"
        try:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs), model_class_local
        except ValueError as e:
            s = str(e)
            if ("Unrecognized configuration class" in s) or ("for this kind of AutoModel" in s):
                model_class_local = "AutoModelForSeq2SeqLM"
                return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs), model_class_local
            raise

    # If offload requested, optionally try a fast single-GPU load first (fp16, no device_map)
    # to maximize tokens/s when the model actually fits.
    model = None
    model_class = "AutoModelForCausalLM"
    if dev == "cuda" and used_device_map and try_fast_first:
        fast_kwargs = dict(load_kwargs)
        fast_kwargs.pop("device_map", None)
        fast_kwargs.pop("max_memory", None)
        try:
            report["load_kwargs"] = _jsonable(dict(fast_kwargs))
            model, model_class = _load_with_retry(fast_kwargs)
            try:
                if torch.cuda.is_available():
                    model.to("cuda")
                    used_device_map = False
                    report["warnings"].append("try_fast_cuda_first: loaded on single GPU (offload plan not used)")
            except Exception as e:
                # If move-to-cuda fails, drop to offload path.
                if _is_cuda_oom(e):
                    try:
                        del model
                    except Exception:
                        pass
                    model = None
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    raise
        except Exception as e:
            if _is_cuda_oom(e):
                model = None
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                # If failure is due to unsupported attn_implementation, retry without it.
                s = str(e).lower()
                if "attn_implementation" in s or "flash_attention_2" in s or "sdpa" in s:
                    try:
                        fast_kwargs.pop("attn_implementation", None)
                        model, model_class = _load_with_retry(fast_kwargs)
                        model.to("cuda")
                        used_device_map = False
                        report["warnings"].append("attn_implementation removed after load failure")
                    except Exception:
                        model = None
                else:
                    raise

    if model is None:
        # Normal path: load with (potentially) device_map/offload/quant.
        report["load_kwargs"] = _jsonable(dict(load_kwargs))
        try:
            brief = {k: load_kwargs.get(k) for k in ("local_files_only", "device_map", "max_memory", "torch_dtype", "attn_implementation") if k in load_kwargs}
            print(f"[INFO] pylittle.hf: loading model ({brief})", file=sys.stderr, flush=True)
        except Exception:
            pass
        try:
            # If device_map/offload is in use, retry CUDA OOM by lowering GPU max_memory.
            attempt_kwargs = dict(load_kwargs)
            retries = [1.0, 0.9, 0.8]
            last_e = None
            for r in retries:
                try:
                    if r != 1.0 and isinstance(attempt_kwargs.get("max_memory"), dict) and 0 in attempt_kwargs["max_memory"]:
                        mb0 = _parse_mem_to_mb((attempt_kwargs.get("max_memory") or {}).get(0))
                        if mb0 is not None:
                            mm0 = dict(attempt_kwargs.get("max_memory") or {})
                            mm0[0] = _format_mb(max(512, int(mb0 * r)))
                            attempt_kwargs = dict(attempt_kwargs)
                            attempt_kwargs["max_memory"] = mm0
                            report["warnings"].append(f"CUDA OOM retry: lowering max_memory[0] -> {mm0[0]}")
                    model, model_class = _load_with_retry(attempt_kwargs)
                    break
                except Exception as e:
                    last_e = e
                    if _is_cuda_oom(e) and used_device_map and dev == "cuda":
                        try:
                            import gc

                            gc.collect()
                        except Exception:
                            pass
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue
                    raise
            else:
                if last_e is not None:
                    raise last_e
        except Exception as e:
            # If failure is due to unsupported attention backend, retry without it.
            s = str(e).lower()
            if "attn_implementation" in s or "flash_attention_2" in s or "sdpa" in s:
                report["warnings"].append("attn_implementation caused load failure; retrying without it")
                load_kwargs2 = dict(load_kwargs)
                load_kwargs2.pop("attn_implementation", None)
                report["load_kwargs"] = _jsonable(dict(load_kwargs2))
                try:
                    brief2 = {k: load_kwargs2.get(k) for k in ("local_files_only", "device_map", "max_memory", "torch_dtype") if k in load_kwargs2}
                    print(f"[INFO] pylittle.hf: retry loading model (no attn_implementation) ({brief2})", file=sys.stderr, flush=True)
                except Exception:
                    pass
                model, model_class = _load_with_retry(load_kwargs2)
            else:
                raise

    if defer_static_dispatch:
        try:
            from accelerate import dispatch_model  # type: ignore

            report["static_layer_map"] = True
            if static_cuda_layers is None:
                n_layers, warns0 = _auto_static_cuda_layers(model, strategy)
                for w in warns0:
                    report["warnings"].append(w)
            else:
                try:
                    n_layers = int(static_cuda_layers)
                except Exception:
                    n_layers = 0
            report["static_cuda_layers"] = int(n_layers)
            dm, warns = _build_static_device_map(model, n_layers)
            for w in warns:
                report["warnings"].append(w)
            if dm is None:
                report["warnings"].append("static_layer_map could not build a device_map; keeping CPU model")
            else:
                model = dispatch_model(model, device_map=dm)
                used_device_map = True
        except Exception as e:
            report["warnings"].append(f"static_layer_map dispatch failed: {e}")

    try:
        print("[INFO] pylittle.hf: model loaded", file=sys.stderr, flush=True)
    except Exception:
        pass

    if dev == "cuda" and not used_device_map:
        try:
            if torch.cuda.is_available():
                model.to(dev)
        except Exception:
            pass
    model.eval()

    # Optional torch.compile (PyTorch 2.x). Only safe when single-device.
    try:
        compile_req = bool(_get_strategy_kw(strategy, "compile_model", False))
    except Exception:
        compile_req = False
    if compile_req and dev == "cuda" and not used_device_map:
        try:
            if hasattr(torch, "compile"):
                model = torch.compile(model, mode="reduce-overhead")
                report["warnings"].append("torch.compile enabled")
        except Exception as e:
            report["warnings"].append(f"torch.compile failed: {e}")

    try:
        report["model_class"] = model_class
        report["is_encoder_decoder"] = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    except Exception:
        report["model_class"] = model_class
        report["is_encoder_decoder"] = None

    # Tokenizer pad token fallback
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass

    dm_summary = _summarize_hf_device_map(model)
    report["used_device_map"] = bool(used_device_map)
    report["device_map_summary"] = dm_summary
    if used_device_map and not dm_summary:
        report["warnings"].append("device_map requested but hf_device_map missing (accelerate not active?)")

    qinfo = _quant_sanity(model, quant_requested if isinstance(quant_requested, str) else None)
    report["quant"] = qinfo
    if qinfo.get("quant_requested") in ("4bit", "8bit") and not qinfo.get("quant_active"):
        report["warnings"].append("quant requested but does not appear active")

    return model, tokenizer, report


class _SyntheticByteTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors: str = "pt"):
        import torch

        data = text.encode("utf-8", errors="ignore")
        ids = [b % self.vocab_size for b in data] or [32]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids}

    def decode(self, ids, skip_special_tokens: bool = True):
        out_chars = []
        for x in ids:
            try:
                v = int(x)
            except Exception:
                continue
            if 32 <= v < 127:
                out_chars.append(chr(v))
            elif v in (9, 10, 13):
                out_chars.append(chr(v))
            else:
                out_chars.append(" ")
        return "".join(out_chars)


def _make_synthetic_gpt2(device: str):
    from transformers import GPT2Config, GPT2LMHeadModel

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
    model = GPT2LMHeadModel(cfg)
    model.to(device)
    model.eval()
    tok = _SyntheticByteTokenizer(vocab_size=256)
    return model, tok


def _head_candidates(model) -> tuple[int, ...] | None:
    try:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return None
        cand: list[int] = []
        for name in ("num_key_value_heads", "num_attention_heads", "n_head"):
            v = getattr(cfg, name, None)
            if isinstance(v, int) and v > 0:
                cand.append(v)
        # unique, stable order
        return tuple(dict.fromkeys(cand)) or None
    except Exception:
        return None


def _infer_seq_dim_from_heads(k, head_candidates: tuple[int, ...] | None) -> int:
    # Common layouts:
    # - (B, H, S, D) => seq dim = 2
    # - (B, S, H, D) => seq dim = 1
    if getattr(k, "dim", lambda: 0)() != 4:
        return 2
    if head_candidates:
        try:
            if int(k.shape[1]) in head_candidates:
                return 2
            if int(k.shape[2]) in head_candidates:
                return 1
        except Exception:
            pass
    return 2


def _truncate_past_kv(past_key_values, window: int, head_candidates: tuple[int, ...] | None = None):
    if past_key_values is None or window is None or window <= 0:
        return past_key_values
    new_past = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            new_past.append(layer)
            continue
        k, v = layer[0], layer[1]
        try:
            if getattr(k, "dim", lambda: 0)() == 4 and getattr(v, "dim", lambda: 0)() == 4:
                seq_dim = _infer_seq_dim_from_heads(k, head_candidates)
                slicer = [slice(None)] * 4
                slicer[seq_dim] = slice(-window, None)
                k2 = k[tuple(slicer)]
                v2 = v[tuple(slicer)]
                new_past.append((k2, v2) + tuple(layer[2:]))
            else:
                new_past.append(layer)
        except Exception:
            new_past.append(layer)
    return tuple(new_past)


def hf_generate(engine, model_name_or_path: str, prompt: str,
                max_new_tokens: int = 32,
                temperature: float = 0.8,
                stream: bool = True,
                device: Optional[str] = None,
                strategy: Optional[Dict[str, Any]] = None,
                paging_window: Optional[int] = None,
                model_obj: Any | None = None,
                tokenizer_obj: Any | None = None) -> str | Iterator[str]:
    """
    Generate text using transformers model, with PyLittle Engine providing policy/stream control.
    If stream=True, returns an iterator of text chunks; else returns the full string.
    """
    import torch

    # Normalize device: map 'auto' to concrete device using torch availability,
    # but allow strategy to override (e.g., force CPU offload)
    if strategy and isinstance(strategy, dict) and strategy.get("device") in ("cpu", "cuda"):
        dev = str(strategy.get("device"))
    elif device is None or str(device).lower() == "auto":
        dev = _pick_device(getattr(engine, "_device", "cpu"))
    else:
        dev = str(device).lower()

    # If user forces CUDA but it's not available, fallback early (also affects synthetic).
    try:
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
    except Exception:
        dev = "cpu"

    used_device_map = False

    if model_obj is not None or tokenizer_obj is not None:
        if model_obj is None or tokenizer_obj is None:
            raise ValueError("Provide both model_obj and tokenizer_obj, or neither")
        model = model_obj
        tokenizer = tokenizer_obj
        is_synthetic = str(model_name_or_path).lower() in ("synthetic-gpt2", "synthetic")
        # Infer device_map usage from the loaded model instance.
        try:
            used_device_map = isinstance(getattr(model, "hf_device_map", None), dict)
        except Exception:
            used_device_map = False
    else:
        is_synthetic = str(model_name_or_path).lower() in ("synthetic-gpt2", "synthetic")

    if model_obj is None:
        try:
            model, tokenizer, load_report = hf_load_model(
                model_name_or_path,
                device=dev,
                strategy=strategy,
                engine_device_hint=getattr(engine, "_device", None),
            )
            used_device_map = bool(load_report.get("used_device_map"))
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name_or_path}': {e}")

    # Prepare inputs and overlap H2D copy when using CUDA (hub models only).
    if dev == "cuda" and not used_device_map and not is_synthetic and model_obj is None:
        inp_cpu = tokenizer(prompt, return_tensors="pt")
        inputs: Dict[str, Any] = {}

        if _NativeMM is not None:
            try:
                mm = _NativeMM()
                has_cuda = bool(getattr(mm, "has_cuda", lambda: False)())
            except Exception:
                mm = None
                has_cuda = False
        else:
            mm = None
            has_cuda = False

        if mm is not None and has_cuda:
            mm_stream = mm.create_stream()
            try:
                for k, t in inp_cpu.items():
                    if not hasattr(t, "data_ptr"):
                        inputs[k] = t
                        continue
                    tc = t.contiguous()
                    dst = tc.to("cuda")
                    nbytes = int(tc.element_size() * tc.numel())
                    ok = mm.copy_to_device_async(int(dst.data_ptr()), int(tc.data_ptr()), nbytes, mm_stream)
                    if not ok:
                        dst = tc.to("cuda", non_blocking=True)
                    inputs[k] = dst
                mm.synchronize_stream(mm_stream)
            finally:
                if mm_stream is not None:
                    mm.destroy_stream(mm_stream)
        else:
            s_copy = torch.cuda.Stream()
            with torch.cuda.stream(s_copy):
                inputs = {
                    k: (v.to("cuda", non_blocking=True) if hasattr(v, "to") else v)
                    for k, v in tokenizer(prompt, return_tensors="pt").items()
                }
            torch.cuda.current_stream().wait_stream(s_copy)
    else:
        raw = tokenizer(prompt, return_tensors="pt")
        target_dev = dev
        if used_device_map and not is_synthetic:
            # With device_map/offload, inputs should generally be placed on the device where
            # the input embeddings live (often CUDA). Avoid forcing CPU which can trigger
            # warnings and may be slower.
            target_dev = _input_device_for_model(model, fallback="cpu")
        if hasattr(raw, "to"):
            inputs = raw.to(target_dev)
        else:
            inputs = {k: (v.to(target_dev) if hasattr(v, "to") else v) for k, v in raw.items()}

    # KV-window (or synthetic) decode path. Only safe when model is on a single device and decoder-only.
    is_enc_dec = False
    try:
        is_enc_dec = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    except Exception:
        is_enc_dec = False

    if (is_synthetic or (paging_window and int(paging_window) > 0)) and not used_device_map and not is_enc_dec:
        window = int(paging_window) if (paging_window and int(paging_window) > 0) else None
        head_cand = _head_candidates(model)
        kv_bytes_per_tok = _estimate_kv_bytes_per_token(model)

        kv_state: dict | None = None
        if window is not None:
            kv_state = {"window": int(window), "cache_seq_len": 0}

        # Native pager prototype (bookkeeping only): track KV growth and sliding-window residency.
        pager = None
        seq_id = 0
        pager_state: Optional[dict] = None
        if _NativeMM is not None and _NativePager is not None and window is not None:
            try:
                if bool(_get_strategy_kw(strategy, "use_native_kv_pager", False)):
                    mm = _NativeMM()
                    page_bytes = int(_get_strategy_kw(strategy, "kv_pager_page_bytes", 2 * 1024 * 1024) or (2 * 1024 * 1024))
                    pager = _NativePager(mm, page_bytes)
                    pager.add_sequence(seq_id)
                    pager_state = {"stats": None}
            except Exception:
                pager = None

        def _iter_stream() -> Iterator[str]:
            input_ids = inputs["input_ids"]
            past = None
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    cur_ids = input_ids if past is None else input_ids[:, -1:]

                    past = _maybe_to_cache(past)
                    past_len = _cache_seq_len(past, head_cand)

                    attn = torch.ones(
                        (cur_ids.shape[0], past_len + cur_ids.shape[1]),
                        device=cur_ids.device,
                        dtype=torch.long,
                    )
                    try:
                        if hasattr(model, "prepare_inputs_for_generation"):
                            model_inputs = model.prepare_inputs_for_generation(
                                cur_ids,
                                past_key_values=past,
                                attention_mask=attn,
                                use_cache=True,
                            )
                        else:
                            model_inputs = {
                                "input_ids": cur_ids,
                                "attention_mask": attn,
                                "past_key_values": past,
                                "use_cache": True,
                            }
                    except Exception:
                        model_inputs = {
                            "input_ids": cur_ids,
                            "attention_mask": attn,
                            "past_key_values": past,
                            "use_cache": True,
                        }

                    out = model(**model_inputs)
                    logits = out.logits[:, -1, :]
                    next_id = _sample_next_token(logits, temperature)
                    input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)
                    past = out.past_key_values
                    past = _maybe_to_cache(past)
                    if window is not None:
                        past = _maybe_crop_cache(past, window)
                        if past is not None and not hasattr(past, "crop"):
                            past = _truncate_past_kv(past, window, head_candidates=head_cand)
                        try:
                            if kv_state is not None:
                                kv_state["cache_seq_len"] = int(_cache_seq_len(past, head_cand))
                        except Exception:
                            pass

                    # Pager bookkeeping: count KV bytes/token and request a window.
                    if pager is not None and kv_bytes_per_tok > 0:
                        try:
                            if hasattr(pager, "append_kv_bytes"):
                                pager.append_kv_bytes(seq_id, int(kv_bytes_per_tok), 1)
                            # request_window uses "recent_tokens" concept
                            pager.request_window(seq_id, int(window))
                            if pager_state is not None:
                                pager_state["stats"] = str(pager.stats())
                        except Exception:
                            pass

                    txt = tokenizer.decode([int(next_id.item())], skip_special_tokens=True)
                    if txt:
                        _maybe_simulate_pcie(strategy, 1, kv_bytes_per_tok)
                        yield txt

        if stream:
            wrapped = _StreamWithMetrics(_iter_stream())
            wrapped.kv_pager_state = pager_state
            try:
                setattr(wrapped, "kv_state", kv_state)
            except Exception:
                pass
            return wrapped
        return "".join(list(_iter_stream()))

    # Default transformers generation
    do_sample = (temperature is not None) and (float(temperature) > 0.0)
    if stream:
        from transformers import TextIteratorStreamer

        class _CountingStreamer(TextIteratorStreamer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokens_generated = 0
                self._t0 = time.perf_counter()
                self.time_to_first_token_s = None

            def put(self, value):
                try:
                    n = int(value.shape[-1])
                except Exception:
                    try:
                        n = len(value)
                    except Exception:
                        n = 0
                if self.tokens_generated == 0 and self.time_to_first_token_s is None:
                    self.time_to_first_token_s = time.perf_counter() - self._t0
                self.tokens_generated += max(0, n)
                kv_bytes_per_tok = _estimate_kv_bytes_per_token(model)
                _maybe_simulate_pcie(strategy, max(1, n), kv_bytes_per_tok)
                return super().put(value)

        streamer = _CountingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            gen_kwargs = {
                "do_sample": bool(do_sample),
                "max_new_tokens": int(max_new_tokens),
                "streamer": streamer,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
            _ = model.generate(**inputs, **gen_kwargs)
        return streamer
    else:
        with torch.no_grad():
            gen_kwargs = {
                "do_sample": bool(do_sample),
                "max_new_tokens": int(max_new_tokens),
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
            out = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return text
