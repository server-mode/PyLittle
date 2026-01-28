"""
Hugging Face runtime adapter for PyLittle.
- Uses transformers to run a small LLM while letting PyLittle handle streaming + policy hooks.
- This is a temporary bridge until native backends are fully implemented.
"""
from __future__ import annotations
from typing import Iterator, Optional, Dict, Any

try:  # optional native pager
    from pylittle._pylittle import MemoryManager as _NativeMM, KVPager as _NativePager  # type: ignore
except Exception:  # pragma: no cover
    _NativeMM = None
    _NativePager = None


def _pick_device(engine_device: str | None = None) -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # TODO: rocm check if needed
    except Exception:
        pass
    return "cpu"


def hf_generate(engine, model_name_or_path: str, prompt: str,
                max_new_tokens: int = 32,
                temperature: float = 0.8,
                stream: bool = True,
                device: Optional[str] = None,
                strategy: Optional[Dict[str, Any]] = None,
                paging_window: Optional[int] = None) -> str | Iterator[str]:
    """
    Generate text using transformers model, with PyLittle Engine providing policy/stream control.
    If stream=True, returns an iterator of text chunks; else returns the full string.
    """
    # Lazy import to avoid imposing dependency on base install
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch

    # Normalize device: map 'auto' to concrete device using torch availability,
    # but allow strategy to override (e.g., force CPU offload)
    if strategy and isinstance(strategy, dict) and strategy.get("device") in ("cpu", "cuda"):
        dev = str(strategy.get("device"))
    elif device is None or str(device).lower() == "auto":
        dev = _pick_device(getattr(engine, "_device", "cpu"))
    else:
        dev = str(device).lower()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    load_kwargs: Dict[str, Any] = {"use_safetensors": True}
    # Apply low-VRAM strategy (quant/offload) if provided
    used_device_map = False
    if strategy:
        if strategy.get("quant") in ("4bit", "8bit"):
            try:
                import bitsandbytes as bnb  # noqa: F401
                load_kwargs.update({
                    "load_in_4bit": strategy["quant"] == "4bit",
                    "load_in_8bit": strategy["quant"] == "8bit",
                })
            except Exception:
                pass  # if bnb not available, skip quant load
    if strategy.get("offload"):
            # accelerate-style offload
            load_kwargs.update({
                "device_map": "auto",
            })
            if strategy.get("max_memory"):
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
            used_device_map = True

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model with strategy {load_kwargs}: {e}. Ensure torch>=2.6 and safetensors present."
        )
    if dev == "cuda" and not used_device_map:
        # When using device_map=auto, accelerate will place model modules; .to(cuda) may be unnecessary.
        # Also, on CPU-only torch, this would throw. Try and fallback silently.
        try:
            import torch
            if torch.cuda.is_available():
                model.to(dev)
        except Exception:
            pass
    model.eval()

    # Prepare inputs and overlap H2D copy when using CUDA
    try:
        import torch
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
    except Exception:
        dev = "cpu"

    if dev == "cuda" and not used_device_map:
        import torch
        inp_cpu = tokenizer(prompt, return_tensors="pt")  # keep on CPU
        inputs = {}
        # Prefer native MemoryManager async copy if available
        if _NativeMM is not None:
            try:
                mm = _NativeMM()
                has_cuda = getattr(mm, "has_cuda", lambda: False)()
            except Exception:
                mm = None
                has_cuda = False
        else:
            mm = None
            has_cuda = False

        if mm is not None and has_cuda:
            stream = mm.create_stream()
            try:
                for k, t in inp_cpu.items():
                    if not hasattr(t, "data_ptr"):
                        # non-tensor entries (if any)
                        inputs[k] = t
                        continue
                    tc = t.contiguous()
                    dst = tc.to("cuda")  # allocate on device with same dtype/shape
                    nbytes = int(tc.element_size() * tc.numel())
                    ok = mm.copy_to_device_async(int(dst.data_ptr()), int(tc.data_ptr()), nbytes, stream)
                    if not ok:
                        # fallback to non_blocking torch copy
                        dst = tc.to("cuda", non_blocking=True)
                    inputs[k] = dst
                mm.synchronize_stream(stream)
            finally:
                if stream is not None:
                    mm.destroy_stream(stream)
        else:
            # Torch stream-based overlap
            s_copy = torch.cuda.Stream()
            with torch.cuda.stream(s_copy):
                inputs = {k: (v.to("cuda", non_blocking=True) if hasattr(v, "to") else v)
                          for k, v in tokenizer(prompt, return_tensors="pt").items()}
            torch.cuda.current_stream().wait_stream(s_copy)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)

    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
            )
        # Optionally wrap with pager updates
        if paging_window and _NativePager is not None:
            mm = _NativeMM() if _NativeMM is not None else None
            pager = _NativePager(mm, 256 * 1024) if mm is not None else None
            seq_id = 1
            if pager is not None:
                pager.add_sequence(seq_id)
                def _gen():
                    for chunk in streamer:
                        # append simulated KV bytes (len bytes) and request window
                        try:
                            pager.append_kv(seq_id, chunk.encode('utf-8'))
                            pager.request_window(seq_id, int(paging_window))
                        except Exception:
                            pass
                        yield chunk
                return _gen()
        return streamer
    else:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return text
