# PyLittle

PyLittle is a Python-first, hardware-aware LLM inference library focused on *making real Hugging Face models usable and measurable on modest GPUs* (e.g. 6–8GB VRAM, “mining” GPUs on risers, weak PCIe links). Today, PyLittle ships a HF runtime adapter + benchmarking/reporting tools; native kernels/backends are still under active development.

Status: Milestone 0/1 in-progress. Python API and HF integration are usable; native core/backends are under active development.

## What’s working today

- HF adapter with low-VRAM-friendly defaults
	- Safetensors-first loading; prefers fp16 on CUDA when not quantized.
	- Attention backend selection: prefers `flash_attention_2` when installed, else SDPA.
	- Optional offload plan (`device_map=auto` + `max_memory`) when enabled by strategy.
- Benchmarking with real-time metrics
	- `tokens_s`, TTFT (time-to-first-token), peak VRAM (NVML + torch CUDA peak), and prefill/decode split.
	- Automatic `PASS/FAIL` verdict in JSON (fit + speed gate).
- Hardware simulation (best-effort)
	- Simulated VRAM cap via `torch.cuda.set_per_process_memory_fraction`.
	- Simulated weak PCIe (e.g. PCIe 1.1 x4) via per-token sleep proportional to estimated KV traffic.
- GUI report window
	- Tkinter table view that runs benchmarks and displays key metrics + verdict.
- KV paging prototype (bookkeeping)
	- Optional native KV pager *stats* during manual decode (prototype; not yet integrated into attention kernels).

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

- Install optional deps: `transformers`, `safetensors`, `accelerate`, `bitsandbytes`, and one of `nvidia-ml-py`/`pynvml` (for GPU stats).
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

### Presets (fast way to test real models)

```powershell
python d:\PyLittle\tools\bench_hf_vs_pylittle.py --preset 1b --device cuda --strategy low_vram_auto --stream --tokens 64
```

Available presets: `synthetic`, `tiny`, `350m`, `410m`, `1b`.

### GUI report window

```powershell
python d:\PyLittle\tools\gui_bench_report.py
```

### Simulate 6–8GB VRAM + “mining GPU” PCIe

This is useful for GPUs running on risers (x1/x4) or old platforms with slow PCIe.

```powershell
# Simulate an 8GB-cap GPU and PCIe 1.1 x4 (≈ 1.0 GB/s)
python d:\PyLittle\tools\bench_hf_vs_pylittle.py --preset 1b --device cuda --strategy low_vram_auto --stream --tokens 64 --sim-vram-mb 8192 --sim-pcie-gen 1.1 --sim-pcie-lanes 4

# Simulate a tighter 6GB budget
python d:\PyLittle\tools\bench_hf_vs_pylittle.py --preset 410m --device cuda --strategy low_vram_auto --stream --tokens 64 --sim-vram-mb 6144
```

Notes:
- The VRAM cap is best-effort and not a true physical limit.
- PCIe simulation is an approximation based on estimated KV bytes/token.

### Real-model example result (≥1B)

On a real run with `microsoft/phi-1_5` (preset `1b`), simulated `8GB` VRAM and PCIe `1.1 x4`, PyLittle outperformed vanilla HF on both fit and decode speed (PASS verdict). Example numbers from one run:

- `verdict.pass = true`
- Decode throughput: vanilla `decode_tokens_s ≈ 38`, PyLittle `decode_tokens_s ≈ 71`
- Peak VRAM (NVML): ~`4.7GB` on both paths under the simulated cap

These numbers vary by GPU, driver, and PyTorch/Transformers versions; treat them as a sanity-check baseline.

### Offline / cache-only runs

If you already have the model in your HF cache, you can force “no downloads”:

```powershell
python d:\PyLittle\tools\bench_hf_vs_pylittle.py --preset 1b --device cuda --strategy low_vram_auto --stream --tokens 64 --local-files-only
```

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
- `nvidia-ml-py` (or `pynvml`): GPU telemetry for benchmarks and safety policies
- `flash_attn`: optional FlashAttention2 backend (when compatible)

## License

Apache-2.0
