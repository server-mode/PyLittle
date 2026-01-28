# Stub: Convert HF models to PyLittle format

def convert_from_hf(model_name_or_path: str, out_path: str, quantize: bool = True, qbits: int = 4):
    # TODO: implement real converter (Milestone 6)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"STUB-HF-CONVERT {model_name_or_path} q{qbits}={quantize}\n")
    return out_path
