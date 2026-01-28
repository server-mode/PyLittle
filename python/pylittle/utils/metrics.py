from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict

try:
    import torch
except Exception:  # optional
    torch = None

# NVML optional probing
try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

@contextmanager
def stopwatch():
    t0 = time.time()
    yield (lambda: time.time() - t0)


def gpu_stats() -> Dict[str, float] | None:
    if not _HAS_NVML:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "gpu_util": float(util.gpu),
            "mem_util": float(util.memory),
            "mem_used_mb": mem.used / (1024 * 1024),
            "mem_total_mb": mem.total / (1024 * 1024),
            "temp_c": float(temp),
        }
    except Exception:
        return None


def torch_tokens_per_sec(n_tokens: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return n_tokens / seconds
