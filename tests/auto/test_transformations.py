import unittest
from triangle_cubature import integrate


class TestModuleA(unittest.TestCase):
    def test_adder(self):
        a = 5
        b = 10
        self.assertEqual(15, module_A.add(a, b))


if __name__ == '__main__':
    unittest.main()
