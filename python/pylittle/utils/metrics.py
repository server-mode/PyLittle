from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

import threading

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


class NvmlPeakMemory:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ok = bool(_HAS_NVML)
        self.start_mb: Optional[float] = None
        self.end_mb: Optional[float] = None
        self.peak_mb: Optional[float] = None

    def _read_mb(self) -> Optional[float]:
        if not _HAS_NVML:
            return None
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            return float(mem.used) / (1024 * 1024)
        except Exception:
            return None

    def start(self):
        if not self.ok or self._thread is not None:
            return self
        self.start_mb = self._read_mb()
        self.peak_mb = self.start_mb
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                v = self._read_mb()
                if v is not None:
                    if self.peak_mb is None or v > self.peak_mb:
                        self.peak_mb = v
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return self
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        self._thread = None
        self.end_mb = self._read_mb()
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nvml": bool(self.ok),
            "start_mb": self.start_mb,
            "end_mb": self.end_mb,
            "peak_mb": self.peak_mb,
        }


@contextmanager
def nvml_peak_memory(interval_s: float = 0.05):
    tracker = NvmlPeakMemory(interval_s=interval_s).start()
    try:
        yield tracker
    finally:
        tracker.stop()


class GpuMemoryTrace:
    """Periodic GPU memory sampler with a time series.

    Prefers NVML (total device memory used). Falls back to torch CUDA allocator
    (memory_allocated) when NVML isn't available.
    """

    def __init__(self, interval_s: float = 0.05, max_samples: int = 20000):
        self.interval_s = float(interval_s)
        self.max_samples = int(max_samples)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[dict[str, float]] = []
        self.ok = bool(_HAS_NVML) or (torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
        self.backend = "nvml" if _HAS_NVML else ("torch_cuda" if self.ok else "none")
        self.start_mb: Optional[float] = None
        self.end_mb: Optional[float] = None
        self.peak_mb: Optional[float] = None
        self._t0: Optional[float] = None

    def _read_mb(self) -> Optional[float]:
        if _HAS_NVML:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                return float(mem.used) / (1024 * 1024)
            except Exception:
                return None
        if torch is not None:
            try:
                if not torch.cuda.is_available():
                    return None
                return float(torch.cuda.memory_allocated()) / (1024 * 1024)
            except Exception:
                return None
        return None

    def start(self):
        if not self.ok or self._thread is not None:
            return self
        self._t0 = time.perf_counter()
        v0 = self._read_mb()
        self.start_mb = v0
        self.peak_mb = v0
        if v0 is not None:
            self.samples.append({"t_s": 0.0, "mb": float(v0)})
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                v = self._read_mb()
                if v is not None and self._t0 is not None:
                    t_s = time.perf_counter() - self._t0
                    if len(self.samples) < self.max_samples:
                        self.samples.append({"t_s": float(t_s), "mb": float(v)})
                    if self.peak_mb is None or v > self.peak_mb:
                        self.peak_mb = float(v)
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return self
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        self._thread = None
        self.end_mb = self._read_mb()
        return self

    def as_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "backend": str(self.backend),
            "interval_s": float(self.interval_s),
            "start_mb": self.start_mb,
            "end_mb": self.end_mb,
            "peak_mb": self.peak_mb,
            "samples": (list(self.samples) if include_samples else None),
        }


@contextmanager
def gpu_memory_trace(interval_s: float = 0.05, max_samples: int = 20000):
    tracker = GpuMemoryTrace(interval_s=interval_s, max_samples=max_samples).start()
    try:
        yield tracker
    finally:
        tracker.stop()


@contextmanager
def torch_cuda_peak_memory():
    if torch is None:
        yield None
        return
    try:
        if not torch.cuda.is_available():
            yield None
            return
        torch.cuda.synchronize()
        before = int(torch.cuda.memory_allocated())
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        ctx = {"alloc_before": before}
        yield ctx
        torch.cuda.synchronize()
        after = int(torch.cuda.memory_allocated())
        peak = int(torch.cuda.max_memory_allocated())
        ctx["alloc_after"] = after
        ctx["peak_alloc"] = peak
        ctx["alloc_delta"] = after - before
        ctx["peak_delta"] = peak - before
    except Exception:
        yield None


def torch_tokens_per_sec(n_tokens: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return n_tokens / seconds
