from __future__ import annotations
import os
import argparse
import random

try:
    from pylittle._pylittle import MemoryManager, KVPager  # type: ignore
except Exception:
    MemoryManager = None
    KVPager = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1)
    ap.add_argument("--page-bytes", type=int, default=256*1024, help="Page size in bytes")
    ap.add_argument("--tokens", type=int, default=2000, help="Simulated tokens to append")
    ap.add_argument("--window", type=int, default=512, help="Recent tokens to keep on GPU")
    args = ap.parse_args()

    if MemoryManager is None or KVPager is None:
        print("Native module not available. Build C++ bindings to run this demo.")
        return 1

    mm = MemoryManager()
    pager = KVPager(mm, args.page_bytes)
    pager.add_sequence(args.seq)

    # Append pseudo-KV bytes (1 token ~ 1 byte for demo)
    for _ in range(args.tokens // 256):
        blob = os.urandom(256)
    pager.append_kv(args.seq, blob)
    # Request window and show stats
    got = pager.request_window(args.seq, args.window)
    print("window_bytes:", got)
    print("stats:", pager.stats())
    # Evict some
    pager.evict_lru(args.page_bytes)
    print("after_evict:", pager.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
