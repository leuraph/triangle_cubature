import unittest
from dev_tools.polynomials import Monomial, Polynomial
from dev_tools.polynomials import integrate_on_triangle
from dev_tools.utils import generate_random_triangle
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

        # Test-04: randomized tests
        # -------------------------
        np.random.seed(42)
        n_random_tests = 100
        for _ in range(n_random_tests):
            n_coordinates = 100
            coordinates = np.random.rand(n_coordinates, 2)
            coefficient = np.random.rand()
            x_exponent = np.random.randint(10)
            y_exponent = np.random.randint(10)
            monomial = Monomial(
                x_exponent=x_exponent,
                y_exponent=y_exponent,
                coefficient=coefficient)

            values = monomial.eval_at(coordinates)
            for value, coordinate in zip(values, coordinates):
                expected_value = (
                    coefficient *
                    coordinate[0]**x_exponent *
                    coordinate[1]**y_exponent)
                self.assertAlmostEqual(expected_value, value)

    def test_polynomials(self):
        pass

    def test_integrate_on_triangle(self) -> None:
        # sanity-check: integrating identity must yield area of triangle
        # --------------------------------------------------------------

        identity = Monomial(x_exponent=0, y_exponent=0, coefficient=1.)
        p_id = Polynomial(monomials=[identity])

        np.random.seed(42)
        random_triangle = generate_random_triangle()
        area = 0.5*np.linalg.det(np.column_stack([random_triangle, np.ones(3)]))

        calculated_area = integrate_on_triangle(
            polynomial=p_id, vertices=random_triangle)

        self.assertAlmostEqual(area, calculated_area)

        # TODO sanity-check: integrating linear
        # --------------------------------

if __name__ == '__main__':
    unittest.main()
