import unittest
import numpy as np
import triangle_cubature
from dev_tools import utils
from dev_tools import polynomials
from p1afempy.io_helpers import read_mesh, read_boundary_condition
from p1afempy.refinement import refineNVB
from pathlib import Path
from tqdm import tqdm


class TestCubatureRules(unittest.TestCase):
    def test_midpoint(self):
        # -----
        # setup
        # -----
        midpoint_rule = triangle_cubature.rule_factory.get_rule(
            rule=triangle_cubature.rule_factory.CubatureRuleEnum.MIDPOINT)
        np.random.seed(42)

        # -----------------------------------------------------
        # midpoint must have exactness order of degree 1, i.e.
        # polynomials must be integrated exactly up to degree 1
        # -----------------------------------------------------
        print('unit testing midpoint-cubature on single triangles...')
        n_tests = 5
        for _ in tqdm(range(n_tests)):
            random_triangle = utils.generate_random_triangle()
            random_linear_polynomial = polynomials.get_random_polynomial(
                degree=1)
            calculated_result = \
                triangle_cubature.integrate.integrate_on_triangle(
                    f=random_linear_polynomial.eval_at,
                    triangle=random_triangle,
                    weights_and_integration_points=midpoint_rule.weights_and_integration_points)
            expected_result = polynomials.integrate_on_triangle(
                polynomial=random_linear_polynomial,
                vertices=random_triangle)
            self.assertAlmostEqual(calculated_result, expected_result)

        # test on mesh
        print('unit testing midpoint-cubature on refined mesh...')
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

        random_triangle = utils.generate_random_triangle()
        random_linear_polynomial = polynomials.get_random_polynomial(
            degree=1)
        expected_result = polynomials.integrate_on_mesh(
            polynomial=random_linear_polynomial,
            elements=elements,
            vertices=coordinates,
            display_progress=True)
        calculated_result = triangle_cubature.integrate.integrate_on_mesh(
            f=random_linear_polynomial.eval_at,
            coordinates=coordinates,
            elements=elements,
            weights_and_integration_points=midpoint_rule.weights_and_integration_points)
        self.assertAlmostEqual(expected_result, calculated_result)

    def test_lauffer(self):
        # -----
        # setup
        # -----
        lauffer_rule = triangle_cubature.rule_factory.get_rule(
            rule=triangle_cubature.rule_factory.CubatureRuleEnum.LAUFFER)
        np.random.seed(42)

        # -----------------------------------------------------
        # LAUFFER must have exactness order of degree 1, i.e.
        # polynomials must be integrated exactly up to degree 1
        # -----------------------------------------------------
        # test on single triangles
        print('unit testing lauffer-cubature on single triangles...')
        n_tests = 5
        for _ in tqdm(range(n_tests)):
            random_triangle = utils.generate_random_triangle()
            random_linear_polynomial = polynomials.get_random_polynomial(
                degree=1)
            calculated_result = \
                triangle_cubature.integrate.integrate_on_triangle(
                    f=random_linear_polynomial.eval_at,
                    triangle=random_triangle,
                    weights_and_integration_points=lauffer_rule.weights_and_integration_points)
            expected_result = polynomials.integrate_on_triangle(
                polynomial=random_linear_polynomial,
                vertices=random_triangle)
            self.assertAlmostEqual(calculated_result, expected_result)

        # test on mesh
        print('unit testing lauffer-cubature on refined mesh...')
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

        random_triangle = utils.generate_random_triangle()
        random_linear_polynomial = polynomials.get_random_polynomial(
            degree=1)
        expected_result = polynomials.integrate_on_mesh(
            polynomial=random_linear_polynomial,
            elements=elements,
            vertices=coordinates,
            display_progress=True)
        calculated_result = triangle_cubature.integrate.integrate_on_mesh(
            f=random_linear_polynomial.eval_at,
            coordinates=coordinates,
            elements=elements,
            weights_and_integration_points=lauffer_rule.weights_and_integration_points)
        self.assertAlmostEqual(expected_result, calculated_result)


if __name__ == '__main__':
    unittest.main()
