import unittest
import numpy as np
import triangle_cubature
from dev_tools import utils
from dev_tools import polynomials


class TestCubatureRules(unittest.TestCase):
    def test_midpoint(self):
        # -----
        # setup
        # -----
        midpoint_rule = triangle_cubature.rule_factory.get_rule(
            rule=triangle_cubature.rule_factory.CubatureRule.MIDPOINT)
        np.random.seed(42)

        # -----------------------------------------------------
        # midpoint must have exactness order of degree 1, i.e.
        # polynomials must be integrated exactly up to degree 1
        # -----------------------------------------------------
        n_tests = 5
        for _ in range(n_tests):
            random_triangle = utils.generate_random_triangle()
            random_linear_polynomial = polynomials.get_random_polynomial(
                degree=1)
            calculated_result = \
                triangle_cubature.integrate.integrate_on_triangle(
                    f=random_linear_polynomial.eval_at,
                    triangle=random_triangle,
                    rule=midpoint_rule)
            expected_result = polynomials.integrate_on_triangle(
                polynomial=random_linear_polynomial,
                vertices=random_triangle)
            self.assertAlmostEqual(calculated_result, expected_result)

        # TODO test on mesh

    def test_lauffer(self):
        # -----
        # setup
        # -----
        lauffer_rule = triangle_cubature.rule_factory.get_rule(
            rule=triangle_cubature.rule_factory.CubatureRule.LAUFFER)
        np.random.seed(42)

        # -----------------------------------------------------
        # LAUFFER must have exactness order of degree 1, i.e.
        # polynomials must be integrated exactly up to degree 1
        # -----------------------------------------------------
        n_tests = 5
        for _ in range(n_tests):
            random_triangle = utils.generate_random_triangle()
            random_linear_polynomial = polynomials.get_random_polynomial(
                degree=1)
            calculated_result = \
                triangle_cubature.integrate.integrate_on_triangle(
                    f=random_linear_polynomial.eval_at,
                    triangle=random_triangle,
                    rule=lauffer_rule)
            expected_result = polynomials.integrate_on_triangle(
                polynomial=random_linear_polynomial,
                vertices=random_triangle)
            self.assertAlmostEqual(calculated_result, expected_result)

        # TODO test on mesh


if __name__ == '__main__':
    unittest.main()
