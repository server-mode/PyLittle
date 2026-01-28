from __future__ import annotations
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# Ensure Windows event loop supports subprocess (Proactor)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# Lightweight GPU stats util (NVML optional)
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

app = FastAPI(title="PyLittle Bench Dashboard")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/favicon.ico")
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


def gpu_info() -> Dict[str, Any]:
    if not _NVML_OK:
        return {"nvml": False}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        name = pynvml.nvmlDeviceGetName(h).decode("utf-8") if isinstance(pynvml.nvmlDeviceGetName(h), bytes) else str(pynvml.nvmlDeviceGetName(h))
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "nvml": True,
            "name": name,
            "mem_total_mb": round(mem.total / (1024**2), 2),
            "mem_used_mb": round(mem.used / (1024**2), 2),
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "temp_c": temp,
        }
    except Exception:
        return {"nvml": False}


async def run_bench(payload: Dict[str, Any], send):
    # Run the existing bench script with given params and stream progress
    import subprocess, sys
    args = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "bench_hf_vs_pylittle.py")]
    # map UI payload to CLI
    preset = payload.get("preset") or "tiny"
    device = payload.get("device") or "auto"
    strategy = payload.get("strategy") or "low_vram_auto"
    paging_window = payload.get("paging_window")
    tokens = str(payload.get("tokens") or 64)
    stream = payload.get("stream", True)
    cli = ["--preset", preset, "--device", device, "--strategy", strategy, "--tokens", tokens]
    if stream:
        cli.append("--stream")
    if paging_window:
        cli += ["--paging-window", str(paging_window)]
    # Ensure child process can import local packages
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bind = os.path.join(root, "build", "bindings", "Release")
    py_pkg = os.path.join(root, "python")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([x for x in [env.get("PYTHONPATH", ""), py_pkg, bind, root] if x])

    try:
        proc = await asyncio.create_subprocess_exec(
            *args, *cli,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=root,
        )
        # Stream lines
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                await send({"type": "log", "ts": datetime.utcnow().isoformat(), "line": line.decode("utf-8", errors="ignore").rstrip()})
            except WebSocketDisconnect:
                proc.terminate()
                return
        code = await proc.wait()
        await send({"type": "done", "code": code})
    except NotImplementedError:
        # Fallback for environments without asyncio subprocess support
        import subprocess
        with subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "bench_hf_vs_pylittle.py"), *cli],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=root,
            text=True,
            bufsize=1,
        ) as p:
            loop = asyncio.get_running_loop()
            try:
                while True:
                    line = await loop.run_in_executor(None, p.stdout.readline)  # type: ignore[arg-type]
                    if not line:
                        break
                    await send({"type": "log", "ts": datetime.utcnow().isoformat(), "line": line.rstrip()})
            finally:
                code = p.wait()
                await send({"type": "done", "code": code})


@app.websocket("/ws")
async def ws(ws: WebSocket):
    try:
        await ws.accept()
    except Exception:
        return
    try:
        while True:
            try:
                data_text = await ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                break
            try:
                msg = json.loads(data_text)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid json"}))
                continue
            if msg.get("type") == "gpu":
                await ws.send_text(json.dumps({"type": "gpu", "data": gpu_info()}))
            elif msg.get("type") == "bench":
                async def send(obj):
                    try:
                        await ws.send_text(json.dumps(obj))
                    except (WebSocketDisconnect, RuntimeError):
                        # client disconnected mid-run
                        pass
                await send({"type": "gpu", "data": gpu_info()})
                await run_bench(msg.get("data", {}), send)
            else:
                await ws.send_text(json.dumps({"type": "error", "message": "unknown message"}))
    except Exception:
        # swallow unexpected errors to avoid noisy stacktraces after client closes
        pass
