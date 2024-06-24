import unittest
import numpy as np
import triangle_cubature
from dev_tools import utils
from dev_tools import polynomials
from p1afempy.io_helpers import read_mesh, read_boundary_condition
from p1afempy.refinement import refineNVB
from pathlib import Path
from tqdm import tqdm
from triangle_cubature.rule_factory import CubatureRuleEnum
from triangle_cubature.rule_factory import get_rule
import warnings


class TestCubatureRules(unittest.TestCase):
    def test_cubature_rules(self):
        # -----
        # setup
        # -----
        self.rules_to_test: list[CubatureRuleEnum] = [
            triangle_cubature.rule_factory.CubatureRuleEnum.MIDPOINT,
            triangle_cubature.rule_factory.CubatureRuleEnum.LAUFFER,
            triangle_cubature.rule_factory.CubatureRuleEnum.SMPLX1
        ]

        # reading and refining a mesh (same for all tests)
        base_path = Path('tests/data/simple_square_mesh/')
        path_to_coordinates = base_path / Path('coordinates.dat')
        path_to_elements = base_path / Path('elements.dat')
        path_to_boundary = base_path / Path('boundary.dat')

        coordinates, elements = read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        # there's only one boundary
        boundaries = [read_boundary_condition(
            path_to_boundary=path_to_boundary)]

        n_refinements = 2
        for _ in range(n_refinements):
            marked_elements = np.arange(elements.shape[0])
            coordinates, elements, boundaries, _ = refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries
            )

        # -----
        # TESTS
        # -----
        np.random.seed(42)

        for cubature_rule in [get_rule(rule) for rule in self.rules_to_test]:
            waip = cubature_rule.weights_and_integration_points
            name = cubature_rule.name
            degree_of_exactness = cubature_rule.degree_of_exactness
            # test on single triangle
            print(
                f'unit testing {name}-cubature'
                ' on single triangles...')
            n_tests = 5
            for _ in tqdm(range(n_tests)):
                random_triangle = utils.generate_random_triangle()
                random_polynomial = polynomials.get_random_polynomial(
                    degree=degree_of_exactness)
                calculated_result = \
                    triangle_cubature.integrate.integrate_on_triangle(
                        f=random_polynomial.eval_at,
                        triangle=random_triangle,
                        weights_and_integration_points=waip)
                expected_result = polynomials.integrate_on_triangle(
                    polynomial=random_polynomial,
                    vertices=random_triangle)
                self.assertAlmostEqual(calculated_result, expected_result)

            # test on mesh
            print(
                f'unit testing {name}-cubature'
                ' on refined mesh...')

            random_triangle = utils.generate_random_triangle()
            random_polynomial = polynomials.get_random_polynomial(
                degree=degree_of_exactness)
            expected_result = polynomials.integrate_on_mesh(
                polynomial=random_polynomial,
                elements=elements,
                vertices=coordinates,
                display_progress=True)
            calculated_result = triangle_cubature.integrate.integrate_on_mesh(
                f=random_polynomial.eval_at,
                coordinates=coordinates,
                elements=elements,
                weights_and_integration_points=waip)
            self.assertAlmostEqual(expected_result, calculated_result)

    def tearDown(self):
        # At the end of the test, check for uncovered cubature rules
        all_cubature_rules = set(item for item in CubatureRuleEnum)
        covered_cubature_rules = set(self.rules_to_test)
        uncovered_rules = all_cubature_rules - covered_cubature_rules
        if uncovered_rules:
            warnings.warn(
                "There are cubature rules not covered by this unit test, i.e."
                f"Uncovered cubature rules: {uncovered_rules}."
                "You should add them to `test_cubature_rules.py`")


if __name__ == '__main__':
    unittest.main()
