import unittest
from pylittle import Engine, config

class TestBasic(unittest.TestCase):
    def test_engine_load_and_generate(self):
        eng = Engine.load("stub.bin", device="auto", config=config.load_profile("low_vram"))
        res = eng.generate("Hello", max_tokens=4)
        self.assertIn("Hello", res.text)

    def test_stream(self):
        eng = Engine.load("stub.bin")
        out = "".join(list(eng.generate("ABC", stream=True)))
        self.assertIn("ABC", out)

if __name__ == "__main__":
    unittest.main()
