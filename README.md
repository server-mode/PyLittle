# PyLittle

PyLittle is a Python-first, hardware-aware LLM inference library built to run big models on modest machines (e.g., 6–8GB VRAM) efficiently. It targets near-GPU-class performance on low-VRAM hardware via quantization, offload, and aggressive memory scheduling, while exposing a simple Python API.

Status: Milestone 0/1 in-progress. Python API and HF integration are usable; native core/backends are under active development.

## Breakthrough features (vision and ongoing implementation)

- Low-VRAM first design
	- Quantized weights (Q4/Q8) and KV-cache quantization (Q4) to shrink memory footprint dramatically.
	- Hierarchical offload (VRAM ↔ pinned host RAM ↔ disk via mmap) with async prefetch and LRU eviction.
	- Budgeter that reads device capabilities and plans allocation (weights/KV/scratch) for best fit on weak GPUs.

- Multi-backend core (plug-in architecture)
	- CPU, CUDA (NVIDIA), ROCm (AMD), Vulkan (cross-vendor). Backends are pluggable and can add custom kernels.
	- Fused kernels roadmap (fused attention, dequant-on-the-fly, persistent kernels) with stream overlap.

- Safety and durability
	- Thermal/usage monitoring (NVML/ROCm SMI) and policy-based throttling to protect old/weak machines.
	- Graceful degrade: reduce batch/context/precision or offload to CPU automatically.

- Python-first developer experience
	- Clean API compatible with NumPy/Torch and HF adapters for easy adoption.
	- Optional native engine via pybind11 for maximum performance when available.

## Quick start (Python API)

```python
from pylittle import Engine, config

eng = Engine.load("models/7b/pylittle_q4.bin", device="auto", config=config.load_profile("low_vram"))
eng.set_safety_policy(max_gpu_temp=80, max_gpu_util=90)
print(eng.generate("Hello", max_tokens=8).text)
```

## Low-VRAM strategy with HF integration (today)

Until native kernels land, PyLittle can drive Hugging Face models with a low-VRAM strategy (bitsandbytes 4/8-bit + accelerate offload + safetensors):

- Install optional deps: `transformers`, `safetensors`, `accelerate`, `bitsandbytes`, `pynvml` (for GPU stats).
- Use the benchmark harness with an automatic low-VRAM plan:

```powershell
python d:\PyLittle\tools\bench_hf_vs_pylittle.py --model sshleifer/tiny-gpt2 --prompt "Hello" --tokens 128 --device auto --strategy low_vram_auto
```

What you get now:
- Safetensors-first loading to avoid unsafe torch.load code paths.
- If available, 4-bit/8-bit load and device_map=auto with max_memory to keep VRAM within a budget.
- A baseline for VRAM delta and latency to compare against vanilla HF.

In the near term (Milestone 2/3/5), the same interface will leverage native quantized kernels, offload, and KV-cache quantization for real speedups on weak hardware.

## CLI

```powershell
pylittle "Hello from PyLittle" --device auto
```

## Benchmark HF vs PyLittle policy

```powershell
python d:\PyLittle\tools\bench_hf_vs_pylittle.py --model sshleifer/tiny-gpt2 --prompt "Hello" --tokens 64 --device cuda --strategy low_vram_auto
```

Outputs include:
- `vanilla`/`pylittle` latencies and token lengths
- `speedup_x` (when applicable)
- GPU stats (when `pynvml` present): util/temp/mem
- `vram_delta_mb`: estimated VRAM change during run

## Roadmap (milestones)

See `docs/architecture.md`. Highlights:

- Milestone 1 — Core prototype (CPU + Py API)
- Milestone 2 — Memory manager + quantized runtime
	- Global pool + LRU cache, offload VRAM↔host↔disk, budgeter, dequant-on-the-fly
- Milestone 3 — CUDA backend
	- cuBLAS for matmul, async transfers, pinned host memory, Q4 dequant integration
- Milestone 4 — ROCm / Vulkan backends
- Milestone 5 — KV-cache quantization + long-context streaming (paging)
- Milestone 6 — API polish, Torch/HF integration, packaging
- Milestone 7 — Testing, benchmarks, safety features
- Milestone 8 — Optimization & community

## Build (native)

See `docs/how_to_build.md` for CMake and bindings. Pybind11 bindings build is optional; the Python API falls back to a pure-Python stub if native module is unavailable.

## Optional dependencies

- `transformers`, `safetensors`, `accelerate`: HF runtime integration and offload
- `bitsandbytes`: 4-bit/8-bit quantized loading
- `pynvml`: GPU telemetry for benchmarks and safety policies

## License

Apache-2.0
