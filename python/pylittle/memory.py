from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import os
import time
import tempfile
import threading

@dataclass
class Block:
    key: str
    size_bytes: int
    where: str  # 'gpu' | 'host' | 'disk'
    path: Optional[str] = None
    last_use: float = field(default_factory=time.time)

class MemoryManager:
    def __init__(self, gpu_budget_mb: int = 0, host_budget_mb: int = 2048, cache_dir: Optional[str] = None):
        self.gpu_budget = gpu_budget_mb * 1024 * 1024
        self.host_budget = host_budget_mb * 1024 * 1024
        self.gpu_used = 0
        self.host_used = 0
        self.blocks: Dict[str, Block] = {}
        self._lock = threading.Lock()
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "pylittle_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _evict_if_needed(self, tier: str):
        with self._lock:
            budget = self.gpu_budget if tier == 'gpu' else self.host_budget
            used = self.gpu_used if tier == 'gpu' else self.host_used
            if used <= budget:
                return
            over = used - budget
            # LRU eviction of blocks in this tier
            cand = [b for b in self.blocks.values() if b.where == tier]
            cand.sort(key=lambda b: b.last_use)
            for b in cand:
                if over <= 0:
                    break
                self._move_to_disk(b)
                over -= b.size_bytes

    def _move_to_disk(self, b: Block):
        # Simulate evict by writing placeholder file
        if b.where == 'gpu':
            self.gpu_used -= b.size_bytes
        elif b.where == 'host':
            self.host_used -= b.size_bytes
        path = os.path.join(self.cache_dir, f"{b.key}.bin")
        with open(path, 'wb') as f:
            f.truncate(b.size_bytes)
        b.where = 'disk'
        b.path = path
        b.last_use = time.time()

    def alloc(self, key: str, size_bytes: int, prefer: str = 'gpu') -> Block:
        with self._lock:
            if key in self.blocks:
                b = self.blocks[key]
                b.last_use = time.time()
                return b
            # try prefer tier
            if prefer == 'gpu' and self.gpu_used + size_bytes <= self.gpu_budget:
                self.gpu_used += size_bytes
                b = Block(key, size_bytes, 'gpu')
            elif self.host_used + size_bytes <= self.host_budget:
                self.host_used += size_bytes
                b = Block(key, size_bytes, 'host')
            else:
                # spill straight to disk
                b = Block(key, size_bytes, 'disk')
                path = os.path.join(self.cache_dir, f"{key}.bin")
                with open(path, 'wb') as f:
                    f.truncate(size_bytes)
                b.path = path
            self.blocks[key] = b
        # post-evict to respect budgets
        self._evict_if_needed('gpu')
        self._evict_if_needed('host')
        return b

    def touch(self, key: str):
        with self._lock:
            if key in self.blocks:
                self.blocks[key].last_use = time.time()

    def bring_to(self, key: str, tier: str):
        with self._lock:
            b = self.blocks[key]
            if b.where == tier:
                b.last_use = time.time()
                return
            # move cost model omitted; only update counters
            if tier == 'gpu':
                if self.gpu_used + b.size_bytes > self.gpu_budget:
                    # need eviction; mark and evict after
                    pass
                self.gpu_used += b.size_bytes
            elif tier == 'host':
                if self.host_used + b.size_bytes > self.host_budget:
                    pass
                self.host_used += b.size_bytes
            # decrement old tier
            if b.where == 'gpu':
                self.gpu_used -= b.size_bytes
            elif b.where == 'host':
                self.host_used -= b.size_bytes
            b.where = tier
            b.last_use = time.time()
        self._evict_if_needed('gpu')
        self._evict_if_needed('host')

    def free(self, key: str):
        with self._lock:
            b = self.blocks.pop(key, None)
            if not b:
                return
            if b.where == 'gpu':
                self.gpu_used -= b.size_bytes
            elif b.where == 'host':
                self.host_used -= b.size_bytes
            if b.path and os.path.exists(b.path):
                try:
                    os.remove(b.path)
                except OSError:
                    pass

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'gpu_budget': self.gpu_budget,
                'gpu_used': self.gpu_used,
                'host_budget': self.host_budget,
                'host_used': self.host_used,
                'blocks': len(self.blocks),
            }
