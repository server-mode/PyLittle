from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import json
import os
import time


@dataclass
class PcieBenchResult:
    ok: bool
    device: str
    pinned: bool
    sizes_bytes: List[int]
    iters: int
    warmup: int
    h2d_gbps: Dict[int, float]
    d2h_gbps: Dict[int, float]
    h2d_latency_us: Dict[int, float]
    d2h_latency_us: Dict[int, float]
    notes: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "device": self.device,
            "pinned": self.pinned,
            "sizes_bytes": list(self.sizes_bytes),
            "iters": int(self.iters),
            "warmup": int(self.warmup),
            "h2d_gbps": {str(k): float(v) for k, v in self.h2d_gbps.items()},
            "d2h_gbps": {str(k): float(v) for k, v in self.d2h_gbps.items()},
            "h2d_latency_us": {str(k): float(v) for k, v in self.h2d_latency_us.items()},
            "d2h_latency_us": {str(k): float(v) for k, v in self.d2h_latency_us.items()},
            "notes": list(self.notes),
        }


def _default_cache_path() -> str:
    # Keep cache inside repo outputs/ by default.
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
        out_dir = os.path.join(repo, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, "pcie_microbench_cache.json")
    except Exception:
        return "pcie_microbench_cache.json"


def _cache_key(*, device: str, pinned: bool, sizes_bytes: Iterable[int | str]) -> str:
    # Best-effort key; include both hardware identity and the microbench settings.
    parts: list[str] = []
    try:
        import torch

        parts.append(f"torch={getattr(torch, '__version__', 'unknown')}")
        parts.append(f"cuda={getattr(getattr(torch, 'version', None), 'cuda', None)}")
        if torch.cuda.is_available():
            try:
                parts.append(f"gpu={torch.cuda.get_device_name(0)}")
            except Exception:
                parts.append("gpu=unknown")
    except Exception:
        parts.append("torch=missing")

    parts.append(f"device={device}")
    parts.append(f"pinned={bool(pinned)}")
    try:
        szs = _parse_sizes(sizes_bytes)
        parts.append("sizes=" + ",".join(str(s) for s in szs))
    except Exception:
        parts.append("sizes=unknown")
    return "|".join(parts)


def _load_cache(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def pcie_microbench_cached(
    *,
    cache_path: str | None = None,
    ttl_s: float = 7.0 * 24.0 * 3600.0,
    device: str = "cuda",
    sizes_bytes: Iterable[int | str] = (4096, 1 << 20, 64 << 20),
    iters: int = 50,
    warmup: int = 10,
    pinned: bool = True,
) -> Dict[str, Any]:
    """Run or reuse a cached PCIe microbench result.

    Returns a JSON-friendly dict with added keys:
    - from_cache
    - cache_key
    - timestamp
    """

    path = cache_path or _default_cache_path()
    key = _cache_key(device=device, pinned=pinned, sizes_bytes=sizes_bytes)

    cache = _load_cache(path)
    entry = cache.get(key) if isinstance(cache, dict) else None
    now = time.time()
    try:
        if isinstance(entry, dict):
            ts = float(entry.get("timestamp", 0.0) or 0.0)
            if ts > 0 and (now - ts) <= float(ttl_s):
                out = dict(entry.get("result") or {})
                out["from_cache"] = True
                out["cache_key"] = key
                out["timestamp"] = ts
                return out
    except Exception:
        pass

    res = pcie_microbench(device=device, sizes_bytes=sizes_bytes, iters=iters, warmup=warmup, pinned=pinned).as_dict()
    out = dict(res)
    out["from_cache"] = False
    out["cache_key"] = key
    out["timestamp"] = now
    try:
        cache = cache if isinstance(cache, dict) else {}
        cache[key] = {"timestamp": now, "result": res}
        _save_cache(path, cache)
    except Exception:
        pass
    return out


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs2 = sorted(float(x) for x in xs)
    mid = len(xs2) // 2
    if len(xs2) % 2 == 1:
        return xs2[mid]
    return 0.5 * (xs2[mid - 1] + xs2[mid])


def _parse_sizes(sizes_bytes: Iterable[int | str]) -> List[int]:
    out: List[int] = []
    for v in sizes_bytes:
        try:
            n = int(v)
        except Exception:
            continue
        if n > 0:
            out.append(n)
    return out


def pcie_microbench(
    *,
    device: str = "cuda",
    sizes_bytes: Iterable[int | str] = (4096, 1 << 20, 64 << 20),
    iters: int = 50,
    warmup: int = 10,
    pinned: bool = True,
) -> PcieBenchResult:
    """Best-effort PCIe microbench using torch tensor copies.

    Measures H2D and D2H for a few sizes and reports median bandwidth and latency.

    Notes:
    - Requires CUDA.
    - Uses pinned host memory when requested.
    - Uses CUDA events for timing to reduce CPU scheduling noise.
    """

    notes: List[str] = []
    try:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            return PcieBenchResult(
                ok=False,
                device=device,
                pinned=pinned,
                sizes_bytes=_parse_sizes(sizes_bytes),
                iters=iters,
                warmup=warmup,
                h2d_gbps={},
                d2h_gbps={},
                h2d_latency_us={},
                d2h_latency_us={},
                notes=["cuda_not_available"],
            )

        dev = torch.device(device)
        sizes = _parse_sizes(sizes_bytes)
        sizes = sizes if sizes else [4096, 1 << 20, 64 << 20]
        it = max(1, int(iters))
        wu = max(0, int(warmup))

        h2d_gbps: Dict[int, float] = {}
        d2h_gbps: Dict[int, float] = {}
        h2d_lat_us: Dict[int, float] = {}
        d2h_lat_us: Dict[int, float] = {}

        # Ensure device is awake
        if dev.type == "cuda":
            torch.cuda.synchronize()

        for nbytes in sizes:
            # Use uint8 to match bytes sizing
            cpu = torch.empty((nbytes,), dtype=torch.uint8, device="cpu", pin_memory=bool(pinned))
            gpu = torch.empty((nbytes,), dtype=torch.uint8, device=dev)

            # Warm-up copies (and cache alloc paths)
            for _ in range(wu):
                gpu.copy_(cpu, non_blocking=True)
                cpu.copy_(gpu, non_blocking=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()

            # Timed copies using CUDA events
            h2d_ms: List[float] = []
            d2h_ms: List[float] = []
            if dev.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                for _ in range(it):
                    start.record()
                    gpu.copy_(cpu, non_blocking=True)
                    end.record()
                    end.synchronize()
                    h2d_ms.append(float(start.elapsed_time(end)))

                for _ in range(it):
                    start.record()
                    cpu.copy_(gpu, non_blocking=True)
                    end.record()
                    end.synchronize()
                    d2h_ms.append(float(start.elapsed_time(end)))
            else:
                # CPU-only fallback (shouldn't be used for PCIe)
                t0 = perf_counter()
                for _ in range(it):
                    gpu.copy_(cpu)
                h2d_ms = [1000.0 * (perf_counter() - t0) / it]
                t1 = perf_counter()
                for _ in range(it):
                    cpu.copy_(gpu)
                d2h_ms = [1000.0 * (perf_counter() - t1) / it]

            mh = _median(h2d_ms)
            md = _median(d2h_ms)
            if mh is not None and mh > 0:
                h2d_gbps[nbytes] = float(nbytes) / (mh / 1000.0) / 1e9
                h2d_lat_us[nbytes] = float(mh) * 1000.0
            if md is not None and md > 0:
                d2h_gbps[nbytes] = float(nbytes) / (md / 1000.0) / 1e9
                d2h_lat_us[nbytes] = float(md) * 1000.0

        return PcieBenchResult(
            ok=True,
            device=str(device),
            pinned=bool(pinned),
            sizes_bytes=_parse_sizes(sizes_bytes),
            iters=it,
            warmup=wu,
            h2d_gbps=h2d_gbps,
            d2h_gbps=d2h_gbps,
            h2d_latency_us=h2d_lat_us,
            d2h_latency_us=d2h_lat_us,
            notes=notes,
        )

    except Exception as e:
        return PcieBenchResult(
            ok=False,
            device=device,
            pinned=pinned,
            sizes_bytes=_parse_sizes(sizes_bytes),
            iters=int(iters) if iters is not None else 0,
            warmup=int(warmup) if warmup is not None else 0,
            h2d_gbps={},
            d2h_gbps={},
            h2d_latency_us={},
            d2h_latency_us={},
            notes=[f"exception: {e}"],
        )
