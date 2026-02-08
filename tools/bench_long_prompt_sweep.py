"""Long prompt sweep benchmark.

Goal: demonstrate bounded KV-cache (sliding window) behavior and observe VRAM over time.

Runs multiple prompt lengths (in tokens): 1k/4k/8k/16k by default.
Outputs JSONL with:
- prompt_tokens_target / prompt_tokens_actual
- kv_state cache_seq_len over time (from PyLittle manual decode path)
- GPU memory trace samples + peak_mb

PowerShell examples:
  D:/PyLittle/.venv/Scripts/python.exe d:/PyLittle/tools/bench_long_prompt_sweep.py --preset 1b --device cuda --paging-window 256 --local-files-only
  D:/PyLittle/.venv/Scripts/python.exe d:/PyLittle/tools/bench_long_prompt_sweep.py --model microsoft/phi-1_5 --prompt-tokens 1024,4096 --gen-tokens 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter

from pylittle import Engine, config
from pylittle.adapters.hf_runtime import hf_load_model, hf_generate
from pylittle.utils.metrics import gpu_memory_trace
from pylittle.utils.simulate import simulated_cuda_vram_limit, pcie_gen_lanes_to_gbps


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(float(part)))
    return out


def _make_prompt_with_tokens(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Build a prompt whose tokenization is (approximately) target_tokens.

    We create a long string, tokenize, then decode the first N tokens back to text
    so that a subsequent tokenize() yields N tokens for most tokenizers.
    """
    target = int(target_tokens)
    if target <= 0:
        return "", 0

    # Start with something stable.
    base = "Hello"
    chunk = " hello"

    # Exponential growth to reduce tokenize calls.
    prompt = base
    cur = 0
    tries = 0
    while cur < target and tries < 30:
        tries += 1
        # Increase by doubling.
        prompt = prompt + (chunk * (2 ** min(tries, 14)))
        try:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            cur = int(ids.shape[-1])
        except Exception:
            # Fallback: just return the text.
            return prompt, cur

    try:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        ids = ids[:target]
        txt = tokenizer.decode(ids, skip_special_tokens=True)
        actual = int(tokenizer(txt, return_tensors="pt")["input_ids"][0].shape[-1])
        return txt, actual
    except Exception:
        return prompt, cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model name. Overrides --preset if set.")
    ap.add_argument("--preset", default="1b", choices=["synthetic", "tiny", "350m", "410m", "1b"], help="Convenience preset.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paging-window", type=int, default=256, help="Enable sliding KV window (manual decode path).")
    ap.add_argument("--prompt-tokens", default="1024,4096,8192,16384", help="Comma-separated prompt lengths (tokens).")
    ap.add_argument("--gen-tokens", type=int, default=2, help="New tokens to generate per run.")
    ap.add_argument("--interval-s", type=float, default=0.05, help="GPU memory sampling interval.")
    ap.add_argument("--out", default="d:/PyLittle/outputs/long_sweep.jsonl", help="Output JSONL path.")
    ap.add_argument("--plot", action="store_true", help="Write a PNG plot (requires matplotlib).")
    ap.add_argument("--plot-path", default="d:/PyLittle/outputs/long_sweep.png")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--sim-vram-mb", type=int, default=None)
    ap.add_argument("--sim-pcie-gen", default=None)
    ap.add_argument("--sim-pcie-lanes", type=int, default=None)
    ap.add_argument("--sim-pcie-overhead-us", type=float, default=50.0)
    ap.add_argument("--allow-over-max", action="store_true", help="Attempt prompts longer than model max context (may error).")
    args = ap.parse_args()

    preset_map = {
        "synthetic": "synthetic-gpt2",
        "tiny": "sshleifer/tiny-gpt2",
        "350m": "facebook/opt-350m",
        "410m": "EleutherAI/pythia-410m-deduped",
        "1b": "microsoft/phi-1_5",
    }
    model = args.model or preset_map[args.preset]

    prompt_sizes = _parse_int_list(args.prompt_tokens)
    if not prompt_sizes:
        print("No prompt sizes specified", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sim_pcie_gbps = None
    if args.sim_pcie_gen is not None and args.sim_pcie_lanes is not None:
        try:
            sim_pcie_gbps = pcie_gen_lanes_to_gbps(args.sim_pcie_gen, args.sim_pcie_lanes)
        except Exception:
            sim_pcie_gbps = None

    strategy = {
        "device": args.device,
        "local_files_only": bool(args.local_files_only),
    }
    if sim_pcie_gbps is not None:
        strategy["sim_pcie_gbps"] = float(sim_pcie_gbps)
        strategy["sim_pcie_overhead_us"] = float(args.sim_pcie_overhead_us)

    eng = Engine.load("stub.bin", device=args.device, config=config.load_profile("balanced"))

    # Load once.
    mdl, tok, load_report = hf_load_model(
        model,
        device=args.device,
        strategy=strategy,
        engine_device_hint=getattr(eng, "_device", None),
    )

    # Best-effort max context detection.
    max_ctx = None
    try:
        cfg = getattr(mdl, "config", None)
        for k in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0:
                max_ctx = int(v)
                break
    except Exception:
        max_ctx = None

    records: list[dict] = []

    with simulated_cuda_vram_limit(args.sim_vram_mb) as sim_vram_report:
        for target_tokens in prompt_sizes:
            # If we know max context, we can skip prompt construction for over-max targets.
            if max_ctx is not None and int(target_tokens) > int(max_ctx) and not bool(args.allow_over_max):
                prompt_text, actual_tokens = "", int(target_tokens)
                print(
                    f"[INFO] sweep: prompt target={target_tokens} skipped (model_max_context={max_ctx})",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                prompt_text, actual_tokens = _make_prompt_with_tokens(tok, int(target_tokens))
                print(f"[INFO] sweep: prompt target={target_tokens} actual={actual_tokens}", file=sys.stderr, flush=True)

            # Skip (recorded) when beyond model context unless explicitly allowed.
            if max_ctx is not None and int(actual_tokens) > int(max_ctx) and not bool(args.allow_over_max):
                rec = {
                    "model": model,
                    "device": args.device,
                    "strategy": strategy,
                    "paging_window": int(args.paging_window),
                    "prompt_tokens_target": int(target_tokens),
                    "prompt_tokens_actual": int(actual_tokens),
                    "model_max_context": int(max_ctx),
                    "skipped": True,
                    "skip_reason": "prompt_exceeds_model_max_context",
                    "gen_tokens": int(args.gen_tokens),
                    "kv_state_final": None,
                    "kv_samples": [],
                    "gpu_mem": None,
                    "sim_vram": sim_vram_report,
                    "load": load_report,
                }
            else:
                kv_samples: list[dict] = []
                out_text = ""
                final_kv_state = None
                err = None
                t0 = perf_counter()
                with gpu_memory_trace(interval_s=float(args.interval_s)) as trace:
                    try:
                        it = hf_generate(
                            eng,
                            model,
                            prompt_text,
                            max_new_tokens=int(args.gen_tokens),
                            temperature=0.8,
                            stream=True,
                            device=args.device,
                            strategy=strategy,
                            paging_window=int(args.paging_window) if int(args.paging_window) > 0 else None,
                            model_obj=mdl,
                            tokenizer_obj=tok,
                        )
                        step = 0
                        for ch in it:
                            step += 1
                            out_text += ch
                            kv_state = getattr(it, "kv_state", None)
                            try:
                                kv_samples.append({
                                    "t_s": float(perf_counter() - t0),
                                    "step": int(step),
                                    "cache_seq_len": (
                                        int(kv_state.get("cache_seq_len"))
                                        if isinstance(kv_state, dict) and kv_state.get("cache_seq_len") is not None
                                        else None
                                    ),
                                })
                            except Exception:
                                pass
                        try:
                            final_kv_state = getattr(it, "kv_state", None)
                        except Exception:
                            final_kv_state = None
                    except Exception as e:
                        err = repr(e)

                total_s = perf_counter() - t0
                rec = {
                    "model": model,
                    "device": args.device,
                    "strategy": strategy,
                    "paging_window": int(args.paging_window),
                    "prompt_tokens_target": int(target_tokens),
                    "prompt_tokens_actual": int(actual_tokens),
                    "model_max_context": int(max_ctx) if max_ctx is not None else None,
                    "skipped": False,
                    "error": err,
                    "gen_tokens": int(args.gen_tokens),
                    "total_time_s": float(total_s),
                    "kv_state_final": final_kv_state,
                    "kv_samples": kv_samples,
                    "gpu_mem": trace.as_dict(include_samples=True),
                    "sim_vram": sim_vram_report,
                    "load": load_report,
                    "text_sample": out_text[:200],
                }

            records.append(rec)

            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.plot:
        try:
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore

            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

            # VRAM traces
            for rec in records:
                samples = (rec.get("gpu_mem") or {}).get("samples") or []
                if not samples:
                    continue
                xs = [s["t_s"] for s in samples]
                ys = [s["mb"] for s in samples]
                ax0.plot(xs, ys, label=f"{rec['prompt_tokens_actual']} tok")
            ax0.set_title("GPU memory used over time")
            ax0.set_xlabel("t (s)")
            ax0.set_ylabel("MB")
            ax0.legend(loc="best")

            # KV cache seq_len samples
            for rec in records:
                kvs = rec.get("kv_samples") or []
                xs = [s["t_s"] for s in kvs if s.get("cache_seq_len") is not None]
                ys = [s["cache_seq_len"] for s in kvs if s.get("cache_seq_len") is not None]
                if xs:
                    ax1.plot(xs, ys, label=f"{rec['prompt_tokens_actual']} tok")
            ax1.set_title("KV cache seq_len over time (post-crop)")
            ax1.set_xlabel("t (s)")
            ax1.set_ylabel("cache_seq_len")
            ax1.legend(loc="best")

            os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
            fig.tight_layout()
            fig.savefig(args.plot_path, dpi=140)
            print(f"[INFO] wrote plot: {args.plot_path}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] plot skipped: {e}", file=sys.stderr)

    print(json.dumps({"ok": True, "out": args.out, "plot": (args.plot_path if args.plot else None), "count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
