from __future__ import annotations
from typing import Dict, Optional
from .utils.hw_probe import detect_hardware

PROFILES = {
    "low_vram": {
        "safety": {"max_gpu_temp": 80, "max_gpu_util": 90},
        "kv_cache": {"precision": "q4"},
    "runtime": {"offload": True, "fused_attention": False, "fused_dequant_gemm": False},
    },
    "balanced": {
        "safety": {"max_gpu_temp": 85, "max_gpu_util": 95},
        "kv_cache": {"precision": "q8"},
    "runtime": {"offload": True, "fused_attention": False, "fused_dequant_gemm": False},
    },
}

def load_profile(name: str) -> Dict:
    return PROFILES.get(name, PROFILES["balanced"]).copy()

def auto_detect() -> Dict:
    hw = detect_hardware()
    prefer = "cuda" if hw.get("gpu") and hw["gpu"].get("vendor") == "nvidia" else ("rocm" if hw.get("gpu") and hw["gpu"].get("vendor") == "amd" else "cpu")
    return {"prefer": prefer, "hw": hw}


def budgeter(desired_model_size_b: int, target_device: Optional[str] = None) -> Dict:
    """Return a strategy dict for low-VRAM machines.
    Heuristics (simple):
    - If no CUDA: CPU offload + 4bit.
    - If model comfortably fits into a fraction of VRAM (<= 50%), avoid offload/quant for speed.
    - If tight fit (50–90%), use 8bit + offload with VRAM cap.
    - If VRAM < ~8GB or model large vs VRAM, use 4bit + offload.
    """
    hw = detect_hardware()
    gpu = (hw.get("gpu") or {})
    vram_gb = float(gpu.get("vram_gb", 0) or 0)
    device = target_device or ("cuda" if gpu.get("vendor") == "nvidia" else "cpu")
    # If CUDA requested but not actually available in this Python env, force CPU plan
    try:
        import torch  # type: ignore
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        if device == "cuda":
            device = "cpu"
    if device == "cuda":
        if not vram_gb:
            return {"device": "cuda", "quant": None, "offload": False}
        vram_bytes = vram_gb * (1024**3)
        fit_ratio = desired_model_size_b / vram_bytes if vram_bytes > 0 else 1.0
        # Small models relative to VRAM: keep it simple for speed
        if fit_ratio <= 0.5:
            return {"device": "cuda", "quant": None, "offload": False}
        # Very constrained VRAM or large model: 4bit + offload
        if vram_gb < 8 or fit_ratio > 0.9:
            return {
                "device": "cuda",
                "quant": "4bit",
                "offload": True,
                "max_memory": {0: f"{int(vram_gb*0.9)}GiB", "cpu": "20GiB"},
            }
        # Medium pressure: 8bit + offload
        return {
            "device": "cuda",
            "quant": "8bit",
            "offload": True,
            "max_memory": {0: f"{int(vram_gb*0.9)}GiB", "cpu": "20GiB"},
        }
    # CPU path
    return {"device": "cpu", "quant": "4bit", "offload": True, "max_memory": {"cpu": "20GiB"}}
