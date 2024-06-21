import numpy as np
from p1afempy.data_structures import CoordinatesType


def transform_weights(
        physical_triangle: CoordinatesType,
        reference_weights: np.ndarray) -> np.ndarray:
    # TODO implement
    return np.zeros_like(reference_weights)


def transform_integration_points(
        physical_triangle: CoordinatesType,
        reference_integration_points: CoordinatesType) -> CoordinatesType:
    # TODO implement
    return np.zeros_like(reference_integration_points)


def get_jacobian(physical_triangle: CoordinatesType) -> np.ndarray:
    p1 = physical_triangle[0, :]
    p2 = physical_triangle[1, :]
    p3 = physical_triangle[2, :]
    return np.vstack((p2 - p1, p3 - p1)).T
