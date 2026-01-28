from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time

from . import config as _config
from .utils.logging import get_logger
from .memory import MemoryManager as PyMemoryManager
try:
    from ._pylittle import MemoryManager as NativeMemoryManager  # type: ignore
except Exception:  # pragma: no cover - optional
    NativeMemoryManager = None  # type: ignore

try:
    from _pylittle import Engine as NativeEngine, GenerateResult as NativeGenerateResult  # type: ignore
except Exception:  # pragma: no cover - optional
    NativeEngine = None
    NativeGenerateResult = None

log = get_logger(__name__)

@dataclass
class GenerateResult:
    text: str

class Engine:
    def __init__(self):
        self._model_path: Optional[str] = None
        self._device: str = "cpu"
        self._policy = {"max_gpu_temp": 85, "max_gpu_util": 95}
        # naive budgets: 0 GPU for now, 2GB host
        # Prefer native memory manager if present for demos; keep Python one as fallback for tests
        self._mem = PyMemoryManager(gpu_budget_mb=0, host_budget_mb=2048)
        self._throttle = 0.0

    @classmethod
    def load(cls, model_path: str, device: str = "auto", config: Optional[dict] = None) -> "Engine":
        # If native engine exists, prefer it for load path later; for now we keep Python stub signature
        eng = cls()
        hw = _config.auto_detect() if device == "auto" else {"prefer": device}
        eng._device = hw.get("prefer", "cpu")
        eng._model_path = model_path
        if config:
            eng._policy.update(config.get("safety", {}))
        log.info(f"[STUB] Loaded model {model_path} on {eng._device}")
        return eng

    def set_safety_policy(self, max_gpu_temp: int = 85, max_gpu_util: int = 95):
        self._policy.update({"max_gpu_temp": max_gpu_temp, "max_gpu_util": max_gpu_util})
        # simulate throttle from policy
        self._throttle = 0.003 if max_gpu_temp <= 80 else 0.0

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.8, stream: bool = False):
        del max_tokens, temperature
        text = f"[STUB/{self._device}] {prompt}"

        def _streamer():
            for ch in text:
                yield ch
                time.sleep(0.002 + self._throttle)

        if stream:
            return _streamer()
        return GenerateResult(text=text)

    def get_stats(self) -> dict:
        return {"device": self._device, "model": self._model_path, "policy": self._policy, "memory": self._mem.stats()}

# convenience re-export
config = _config
