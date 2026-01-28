"""
Integration smoke test for PyLittle.
Run:
    python tests/integration_smoke_test.py
Assumes 'pylittle' is installed (editable or wheel) and available in PYTHONPATH.
"""
from __future__ import annotations
import sys
import time

try:
    from pylittle import Engine, config
except Exception as e:
    print("[ERROR] pylittle import failed:", e)
    sys.exit(1)


def main():
    print("[PyLittle] Integration smoke test starting…")

    # Load with auto hardware profile
    profile = config.load_profile("low_vram")
    eng = Engine.load("stub.bin", device="auto", config=profile)
    eng.set_safety_policy(max_gpu_temp=80, max_gpu_util=90)

    # Non-stream generate
    res = eng.generate("Explain quicksort in one sentence.", max_tokens=32, temperature=0.7)
    assert hasattr(res, "text") and isinstance(res.text, str)
    print("[non-stream]", res.text)

    # Stream generate
    print("[stream] ", end="", flush=True)
    t0 = time.time()
    chunks = []
    for ch in eng.generate("Stream hello world.", stream=True):
        chunks.append(ch)
        print(ch, end="", flush=True)
    print()
    dt = time.time() - t0
    print(f"[stream] took {dt:.3f}s, {len(chunks)} chars")

    # Engine stats
    stats = eng.get_stats()
    assert "device" in stats and "memory" in stats
    print("[stats]", stats)

    print("[PyLittle] Integration smoke test: PASS")


if __name__ == "__main__":
    main()
