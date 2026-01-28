# Architecture

High-level components:
- Python API (`python/pylittle`): Engine wrapper, config, adapters, utils
- Core C++ (`core`): engine, memory manager, quantization
- Backends (`backends`): CPU, CUDA, ROCm, Vulkan
- Bindings (`bindings`): pybind11 module exposing C++ Engine
- Tools: conversion, quantization, benchmarks

Key abstractions:
- IBackend: alloc/free, memcpy_async, gemm, attention
- MemoryManager: pools (GPU/pinned/host), LRU swap, budgeter, prefetch
- Scheduler: token loop, overlap compute and transfer
- Safety: NVML/ROCm-SMI monitoring, throttling policies

Data flow:
1. Load model metadata and quantized weights (mmap where possible)
2. Budget memory for weights, KV cache, scratch
3. During generation, prefetch next tensors, evict LRU when needed
4. Quantized kernels dequantize on-the-fly

See roadmap in README.
