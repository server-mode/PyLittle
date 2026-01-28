import argparse
from .api import Engine

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--model", default="stub.bin")
    p.add_argument("--device", default="auto")
    p.add_argument("--stream", action="store_true")
    args = p.parse_args()

    eng = Engine.load(args.model, device=args.device)
    if args.stream:
        for ch in eng.generate(args.prompt, stream=True):
            print(ch, end="", flush=True)
        print()
    else:
        res = eng.generate(args.prompt)
        print(res.text)

if __name__ == "__main__":
    main()
