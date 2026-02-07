from __future__ import annotations

import json
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk


BENCH = r"d:\PyLittle\tools\bench_hf_vs_pylittle.py"


def _run_bench(args: list[str]) -> dict:
    cmd = [sys.executable, BENCH] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"bench failed: {p.returncode}")
    out = p.stdout.strip().splitlines()[-1].strip()
    return json.loads(out)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyLittle Benchmark Report")
        self.geometry("980x560")

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=10)

        self.preset = tk.StringVar(value="synthetic")
        self.device = tk.StringVar(value="cuda")
        self.strategy = tk.StringVar(value="gpu_only")
        self.tokens = tk.IntVar(value=64)
        self.paging_window = tk.IntVar(value=0)
        self.sim_vram = tk.IntVar(value=8192)
        self.sim_pcie_gen = tk.StringVar(value="1.1")
        self.sim_pcie_lanes = tk.IntVar(value=4)

        ttk.Label(top, text="Preset").grid(row=0, column=0, sticky="w")
        ttk.Combobox(top, textvariable=self.preset, values=["synthetic", "tiny", "350m", "410m", "1b"], width=12, state="readonly").grid(row=0, column=1, padx=6)

        ttk.Label(top, text="Device").grid(row=0, column=2, sticky="w")
        ttk.Combobox(top, textvariable=self.device, values=["auto", "cuda", "cpu"], width=8, state="readonly").grid(row=0, column=3, padx=6)

        ttk.Label(top, text="Strategy").grid(row=0, column=4, sticky="w")
        ttk.Combobox(top, textvariable=self.strategy, values=["gpu_only", "low_vram_auto"], width=12, state="readonly").grid(row=0, column=5, padx=6)

        ttk.Label(top, text="Tokens").grid(row=0, column=6, sticky="w")
        ttk.Entry(top, textvariable=self.tokens, width=6).grid(row=0, column=7, padx=6)

        ttk.Label(top, text="KV window").grid(row=0, column=8, sticky="w")
        ttk.Entry(top, textvariable=self.paging_window, width=6).grid(row=0, column=9, padx=6)

        sim = ttk.LabelFrame(self, text="Mô phỏng GPU yếu")
        sim.pack(fill=tk.X, padx=10)

        ttk.Label(sim, text="VRAM limit (MB)").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(sim, textvariable=self.sim_vram, width=8).grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(sim, text="PCIe gen").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Entry(sim, textvariable=self.sim_pcie_gen, width=6).grid(row=0, column=3, padx=6)

        ttk.Label(sim, text="Lanes").grid(row=0, column=4, sticky="w", padx=6)
        ttk.Entry(sim, textvariable=self.sim_pcie_lanes, width=6).grid(row=0, column=5, padx=6)

        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=10, pady=8)

        self.run_btn = ttk.Button(btns, text="Chạy benchmark", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT)

        self.status = tk.StringVar(value="Sẵn sàng")
        ttk.Label(btns, textvariable=self.status).pack(side=tk.LEFT, padx=10)

        # table
        cols = ("metric", "vanilla", "pylittle")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=14)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=300 if c == "metric" else 320, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # log
        self.log = tk.Text(self, height=8)
        self.log.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

    def _set_rows(self, data: dict):
        for i in self.tree.get_children():
            self.tree.delete(i)

        v = data.get("vanilla", {})
        p = data.get("pylittle", {})

        def row(name: str, a, b):
            self.tree.insert("", "end", values=(name, a, b))

        model_name = (data.get("pylittle_load") or {}).get("model") or self.preset.get()
        row("model", model_name, model_name)
        row("device", (data.get("devices") or {}).get("vanilla"), (data.get("devices") or {}).get("pylittle"))
        st = data.get("strategy")
        if isinstance(st, dict):
            row("strategy(device/quant/offload)", f"{st.get('device')}/{st.get('quant')}/{st.get('offload')}", f"{st.get('device')}/{st.get('quant')}/{st.get('offload')}")
        row("tokens_generated", v.get("tokens_generated"), p.get("tokens_generated"))
        row("tokens_s", v.get("tokens_s"), p.get("tokens_s"))
        row("decode_tokens_s", v.get("decode_tokens_s"), p.get("decode_tokens_s"))
        row("prefill_s", v.get("prefill_s"), p.get("prefill_s"))
        row("decode_s", v.get("decode_s"), p.get("decode_s"))
        row("throughput_chars_s", v.get("throughput_chars_s"), p.get("throughput_chars_s"))
        row("TTFT (s)", v.get("time_to_first_token_s"), p.get("time_to_first_token_s"))

        vpk = (v.get("nvml_peak") or {}).get("peak_mb")
        ppk = (p.get("nvml_peak") or {}).get("peak_mb")
        row("NVML peak (MB)", vpk, ppk)

        wou = data.get("would_oom") or {}
        if wou:
            row("Would OOM @ sim VRAM", wou.get("vanilla"), wou.get("pylittle"))

        ver = data.get("verdict") or {}
        if ver:
            row("VERDICT PASS", ver.get("pass"), ver.get("pass"))
            rs = ver.get("reasons")
            if rs:
                row("verdict_reasons", "; ".join(map(str, rs)), "; ".join(map(str, rs)))

        sim = data.get("simulation") or {}
        if sim:
            row("Sim VRAM limit (MB)", (sim.get("sim_vram") or {}).get("limit_mb"), (sim.get("sim_vram") or {}).get("limit_mb"))
            sp = (sim.get("sim_pcie") or {})
            row("Sim PCIe", f"gen={sp.get('gen')} x{sp.get('lanes')} (~{sp.get('gbps')} GB/s)", f"gen={sp.get('gen')} x{sp.get('lanes')} (~{sp.get('gbps')} GB/s)")

    def on_run(self):
        self.run_btn.configure(state="disabled")
        self.status.set("Đang chạy...")
        self.log.delete("1.0", tk.END)

        args = [
            "--preset", self.preset.get(),
            "--device", self.device.get(),
            "--strategy", self.strategy.get(),
            "--tokens", str(int(self.tokens.get())),
            "--stream",
        ]

        pw = int(self.paging_window.get())
        if pw > 0:
            args += ["--paging-window", str(pw)]

        sv = int(self.sim_vram.get())
        if sv > 0:
            args += ["--sim-vram-mb", str(sv)]

        gen = self.sim_pcie_gen.get().strip()
        lanes = int(self.sim_pcie_lanes.get())
        if gen and lanes > 0:
            args += ["--sim-pcie-gen", gen, "--sim-pcie-lanes", str(lanes)]

        def worker():
            try:
                data = _run_bench(args)
                self.after(0, lambda: self._set_rows(data))
                self.after(0, lambda: self.status.set("Xong"))
                self.after(0, lambda: self.log.insert(tk.END, json.dumps(data, indent=2)))
            except Exception as e:
                self.after(0, lambda: self.status.set("Lỗi"))
                self.after(0, lambda: self.log.insert(tk.END, str(e)))
            finally:
                self.after(0, lambda: self.run_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
