# Python API (initial)

from pylittle import Engine, config

Engine.load(model_path, device='auto', config=None)
Engine.generate(prompt, max_tokens=128, temperature=0.8, stream=False)
Engine.set_safety_policy(max_gpu_temp=85, max_gpu_util=95)

Notes:
- Current implementation is a Python stub for Milestone 1 prototype.
- Future versions will call into C++ core via pybind11.
