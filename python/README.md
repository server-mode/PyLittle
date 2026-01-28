# PyLittle (Python package)

This is the Python package for PyLittle. See repository root README for project overview.

Quick start:

```python
from pylittle import Engine, config
eng = Engine.load("models/7b/pylittle_q4.bin", device="auto", config=config.load_profile("low_vram"))
print(eng.generate("Hello").text)
```
