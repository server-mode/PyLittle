import platform
from typing import Any, Dict, Optional


def _probe_gpu_torch() -> Optional[Dict[str, Any]]:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_b = getattr(props, "total_memory", None)
            name = getattr(props, "name", "cuda")
            vram_gb = (total_b / (1024**3)) if total_b else None
            return {"vendor": "nvidia", "name": str(name), "vram_gb": float(vram_gb) if vram_gb else None}
    except Exception:
        pass
    return None


def _probe_gpu_nvml() -> Optional[Dict[str, Any]]:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        vram_gb = mem.total / (1024**3)
        return {"vendor": "nvidia", "name": name.decode() if isinstance(name, bytes) else str(name), "vram_gb": float(vram_gb)}
    except Exception:
        return None


def _probe_ram_psutil() -> Optional[float]:
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().total / (1024**3))
    except Exception:
        return None


def detect_hardware() -> Dict[str, Any]:
    gpu = _probe_gpu_torch() or _probe_gpu_nvml()
    ram_gb = _probe_ram_psutil()
    return {
        "os": platform.system(),
        "cpu": platform.processor(),
        "gpu": gpu,
        "ram_gb": ram_gb,
    }
