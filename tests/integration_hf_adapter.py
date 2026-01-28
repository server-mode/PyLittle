"""
Integration test with Hugging Face transformers, optional (skips if HF or torch not installed).
Run manually: python tests/integration_hf_adapter.py
"""
from __future__ import annotations
import os
import sys

try:
    from transformers import AutoTokenizer  # type: ignore
    import torch  # type: ignore
except Exception:
    print("[SKIP] transformers/torch not installed.")
    sys.exit(0)

from pylittle import Engine, config
from pylittle.adapters.hf_runtime import hf_generate


def main():
    if not os.environ.get("PYLITTLE_HF_TEST"):
        print("[SKIP] Set PYLITTLE_HF_TEST=1 to run this test.")
        sys.exit(0)

    model = os.environ.get("PYLITTLE_HF_MODEL", "sshleifer/tiny-gpt2")
    prompt = "Hello from PyLittle with HF:"

    eng = Engine.load("stub.bin", device="auto", config=config.load_profile("balanced"))
    # Non-stream
    text = hf_generate(eng, model, prompt, max_new_tokens=16, temperature=0.9, stream=False)
    assert isinstance(text, str)
    print("[HF non-stream]", text[:120])

    # Stream
    print("[HF stream] ", end="")
    for chunk in hf_generate(eng, model, prompt, max_new_tokens=16, temperature=0.9, stream=True):
        print(chunk, end="", flush=True)
    print()
    print("[HF] PASS")


if __name__ == "__main__":
    main()
