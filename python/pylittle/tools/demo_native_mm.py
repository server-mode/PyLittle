from __future__ import annotations
import argparse
import os
from pathlib import Path

try:
    from ._pylittle import MemoryManager  # type: ignore
except Exception:
    try:
        from .._pylittle import MemoryManager  # type: ignore
    except Exception:
        MemoryManager = None  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-Path", type=str, required=True, help="File to map")
    ap.add_argument("-Offset", type=int, default=0, help="Offset to prefetch")
    ap.add_argument("-Bytes", type=int, default=1<<20, help="Bytes to prefetch")
    args = ap.parse_args()

    if MemoryManager is None:
        print("Native module not available. Build C++ bindings to run this demo.")
        return 1

    file_path = str(Path(args.Path))
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return 1

    mm = MemoryManager()
    h = mm.map_file(file_path)
    if h < 0:
        print("map_file failed")
        return 1
    ok = mm.prefetch(h, int(args.Offset), int(args.Bytes))
    print(f"prefetch ok={ok}")
    print("stats:", mm.stats())
    mm.unmap(h)
    print("unmapped. stats:", mm.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
