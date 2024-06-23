import unittest
from dev_tools.polynomials import Monomial
import numpy as np


class TestPolynomial(unittest.TestCase):
    def test_monomials(self):
        coordinates = np.array([
            [10., 0.],
            [0., 10.],
            [5., 5.]
        ])

        # Test-01
        # -------
        x_exponent = 2
        y_exponent = 0
        coefficient = 1./3.
        monomial = Monomial(
            x_exponent=x_exponent,
            y_exponent=y_exponent,
            coefficient=coefficient)
        expected_result = np.array([
            1/3 * 10.**2,
            0.,
            1/3 * 5.**2
        ])
        self.assertTrue(np.allclose(
            expected_result, monomial.eval_at(coordinates)))

        # Test-02
        # -------
        x_exponent = 0
        y_exponent = 3
        coefficient = 1./3.
        monomial = Monomial(
            x_exponent=x_exponent,
            y_exponent=y_exponent,
            coefficient=coefficient)
        expected_result = np.array([
            0.,
            1/3 * 10.**3,
            1/3 * 5.**3
        ])
        self.assertTrue(np.allclose(
            expected_result, monomial.eval_at(coordinates)))

        # Test-03
        # -------
        x_exponent = 2
        y_exponent = 3
        coefficient = 1./3.
        monomial = Monomial(
            x_exponent=x_exponent,
            y_exponent=y_exponent,
            coefficient=coefficient)
        expected_result = np.array([
            0.,
            0.,
            1/3 * 5.**2 * 5.**3
        ])
        self.assertTrue(np.allclose(
            expected_result, monomial.eval_at(coordinates)))

    def test_polynomials(self):
        pass


if __name__ == '__main__':
    unittest.main()
