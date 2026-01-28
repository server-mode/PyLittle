# Build instructions (native)

Requirements:
- CMake >= 3.18
- C++17 compiler
- Optional: CUDA toolkit / Vulkan SDK
- Python 3.9+

Steps:
1. Create and activate a virtualenv, then install pybind11
	- Windows PowerShell:
	  - python -m venv .venv
	  - .\.venv\Scripts\Activate.ps1
	  - python -m pip install -U pip pybind11
2. Configure and build
	- mkdir build && cd build
	- cmake -DPYLITTLE_BUILD_PYBIND=ON ..
	- cmake --build . --config Release
3. Make the built extension importable (dev flow)
	- Copy the built .pyd into the Python package folder:
	  - copy build\bindings\Release\_pylittle.cp<pyver>-win_amd64.pyd python\pylittle\
	- Or add build/bindings/Release to PYTHONPATH

Packaging (later): build wheels with cibuildwheel.

## Demos

- KV paging demo (native):
	- python -m pylittle.tools.demo_kv_paging --tokens 4096 --window 1024

- Overlap demo (CUDA streams; falls back to CPU sim):
	- python d:\PyLittle\tools\demo_overlap.py --size-mb 32 --iters 5

- HF vs PyLittle bench with paging demo + disable quant/offload:
	- python d:\PyLittle\tools\bench_hf_vs_pylittle.py --preset tiny --device cuda --strategy low_vram_auto --paging-window 512 --no-quant --no-offload --stream
