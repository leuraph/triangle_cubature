import unittest
from triangle_cubature.transformations \
    import transform_weights, transform_integration_points
import numpy as np


class TestTransformations(unittest.TestCase):
    def test_transform_weights(self):
        reference_weights = np.array([1.])
        physical_coordinates = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        transformed_weights = transform_weights(
            physical_triangle=physical_coordinates,
            reference_weights=reference_weights)
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
        transformed_coordinates = transform_integration_points(
            physical_triangle=physical_coordinates,
            reference_integration_points=reference_coordinates)
        self.assertTrue(
            np.allclose(reference_coordinates, transformed_coordinates))


if __name__ == '__main__':
    unittest.main()
