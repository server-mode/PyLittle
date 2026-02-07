from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


def pcie_gen_lanes_to_gbps(gen: str | float | int, lanes: int) -> float:
    """Approximate one-direction bandwidth in GB/s.

    Uses effective payload rates (very rough):
    - PCIe 1.x: 0.25 GB/s per lane
    - PCIe 2.x: 0.5 GB/s per lane
    - PCIe 3.x: 0.985 GB/s per lane
    - PCIe 4.x: 1.969 GB/s per lane

    Example: gen=1.1, lanes=4 -> ~1.0 GB/s.
    """
    try:
        g = float(gen)
    except Exception:
        g = 1.0

    per_lane = 0.25
    if g >= 4.0:
        per_lane = 1.969
    elif g >= 3.0:
        per_lane = 0.985
    elif g >= 2.0:
        per_lane = 0.5
    else:
        per_lane = 0.25

    try:
        l = int(lanes)
    except Exception:
        l = 1
    l = max(1, l)
    return float(per_lane * l)


def simulate_pcie_sleep(num_bytes: int, gbps: float, overhead_us: float = 50.0) -> None:
    if gbps is None or gbps <= 0:
        return
    try:
        n = int(num_bytes)
    except Exception:
        return
    if n <= 0:
        return
    seconds = (float(n) / (float(gbps) * 1e9)) + (float(overhead_us) * 1e-6)
    if seconds > 0:
        time.sleep(seconds)


def _cuda_total_mem_mb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / (1024 * 1024)
    except Exception:
        return None


@contextmanager
def simulated_cuda_vram_limit(limit_mb: int | None, device_index: int = 0):
    """Best-effort VRAM limit using PyTorch per-process fraction.

    Notes:
    - This cannot truly change physical VRAM.
    - It mainly influences the CUDA caching allocator and can help mimic OOM behavior.
    - If unsupported, this becomes a no-op.

    Yields a report dict.
    """
    report: Dict[str, Any] = {
        "limit_mb": limit_mb,
        "total_mb": None,
        "fraction": None,
        "applied": False,
        "warning": None,
    }

    if limit_mb is None:
        yield report
        return

    total = _cuda_total_mem_mb()
    report["total_mb"] = total
    if total is None or total <= 0:
        report["warning"] = "cuda_not_available_or_unknown_total"
        yield report
        return

    frac = float(limit_mb) / float(total)
    frac = max(0.01, min(1.0, frac))
    report["fraction"] = frac

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(frac, device=device_index)
            report["applied"] = True
    except Exception as e:
        report["warning"] = f"set_per_process_memory_fraction_failed: {e}"

    try:
        yield report
    finally:
        # There isn't a reliable "restore" API. Keep it simple.
        pass
