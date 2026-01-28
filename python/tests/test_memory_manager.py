import unittest
from pylittle.memory import MemoryManager

class TestMemoryManager(unittest.TestCase):
    def test_alloc_and_evict(self):
        mm = MemoryManager(gpu_budget_mb=0, host_budget_mb=1)  # 1 MB
        b1 = mm.alloc('a', 800_000, prefer='host')
        b2 = mm.alloc('b', 800_000, prefer='host')
        st = mm.stats()
        # Over budget => at least one block spilled to disk
        self.assertGreaterEqual(st['blocks'], 2)
        self.assertLessEqual(st['host_used'], st['host_budget'])

    def test_bring_to(self):
        mm = MemoryManager(gpu_budget_mb=0, host_budget_mb=2)
        mm.alloc('a', 500_000)
        mm.bring_to('a', 'host')
        self.assertLessEqual(mm.stats()['host_used'], mm.stats()['host_budget'])

if __name__ == '__main__':
    unittest.main()
