from time import perf_counter
from pylittle import Engine, config

def main():
    eng = Engine.load("stub.bin", device="auto", config=config.load_profile("balanced"))
    t0 = perf_counter()
    out = eng.generate("Benchmark this output.")
    dt = perf_counter() - t0
    print({
        "latency_s": round(dt, 4),
        "device": eng.get_stats()["device"],
        "len": len(out.text),
    })

if __name__ == "__main__":
    main()
