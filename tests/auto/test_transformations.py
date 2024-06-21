import unittest
from triangle_cubature.transformations \
    import transform_weights, transform_integration_points, get_jacobian, \
    transform_weights_and_integration_points
import numpy as np
from triangle_cubature.weights_and_integration_points \
    import WeightsAndIntegrationPoints


class TestTransformations(unittest.TestCase):
    def test_transform_weights(self):
        reference_weights = np.array([1.])
        physical_coordinates = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        jacobian = get_jacobian(physical_triangle=physical_coordinates)
        transformed_weights = transform_weights(
            reference_weights=reference_weights,
            jacobian=jacobian)
        self.assertTrue(
            np.allclose(reference_weights, transformed_weights))

    def test_transform_coordinates(self):
        reference_coordinates = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1./3., 1./3.]
        ])
        physical_coordinates = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        jacobian = get_jacobian(physical_triangle=physical_coordinates)
        transformed_coordinates = transform_integration_points(
            reference_integration_points=reference_coordinates,
            p1=physical_coordinates[0, :],
            jacobian=jacobian)
        self.assertTrue(
            np.allclose(reference_coordinates, transformed_coordinates))

    def transform_weights_and_integration_points(self):
        # -------------------------------------------------------------
        # Testing the trivial case, i.e. ref. triangle = phys. triangle
        # -------------------------------------------------------------
        reference_integration_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1./3., 1./3.]
        ])
        reference_weights = np.array([0.25, 0.25, 0.25, 0.25])

        reference_rule = WeightsAndIntegrationPoints(
            weights=reference_weights,
            integration_points=reference_integration_points)

        physical_triangle = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])

        transformed = \
            transform_weights_and_integration_points(
                weights_and_integration_points=reference_rule,
                physical_triangle=physical_triangle)

        self.assertTrue(np.allclose(reference_weights, transformed.weights))
        self.assertTrue(np.allclose(reference_integration_points,
                                    transformed.integration_points))

        # TODO add non-trivial tests

    def test_jacobian(self):
        np.random.seed(42)
        for _ in range(100):
            physical_triangle = TestTransformations.generate_random_triangle()
            x_1 = physical_triangle[0, 0]
            y_1 = physical_triangle[0, 1]
            x_2 = physical_triangle[1, 0]
            y_2 = physical_triangle[1, 1]
            x_3 = physical_triangle[2, 0]
            y_3 = physical_triangle[2, 1]
            expected_jacobian = np.array([
                [x_2 - x_1, x_3 - x_1],
                [y_2 - y_1, y_3 - y_1]
            ])
            calculated_jacobian = get_jacobian(
                physical_triangle=physical_triangle)
            self.assertTrue(np.allclose(
                expected_jacobian, calculated_jacobian
            ))

    @staticmethod
    def generate_random_triangle():
        # Step 1: Get three random angles strictly between 0 and 2*pi
        angles = np.sort(np.random.uniform(0, 2*np.pi, 3))

        # Step 2: Generate three random lengths strictly between 0 and 1
        lengths = np.random.uniform(0, 1, 3)

        # Step 3: Generate three coordinates with these angles and lengths
        x_coords = lengths * np.cos(angles)
        y_coords = lengths * np.sin(angles)
        coords = np.vstack((x_coords, y_coords)).T

        # Step 4: Generate two random stretches in x and y direction
        stretch_x = np.random.uniform(0.5, 2.0)  # Random stretch factor for x
        stretch_y = np.random.uniform(0.5, 2.0)  # Random stretch factor for y

        # Stretch the coordinates
        coords[:, 0] *= stretch_x
        coords[:, 1] *= stretch_y

        # Step 5: Generate two random displacements in x and y direction
        displacement_x = np.random.uniform(-1, 1)  # Random displacement for x
        displacement_y = np.random.uniform(-1, 1)  # Random displacement for y

        # Displace the coordinates
        coords[:, 0] += displacement_x
        coords[:, 1] += displacement_y

        return coords


if __name__ == '__main__':
    unittest.main()
