from pylittle import Engine

if __name__ == "__main__":
    eng = Engine.load("stub.bin")
    print(eng.generate("Explain quicksort.").text)
