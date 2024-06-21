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
    # TODO implement
    return np.zeros(shape=(2, 2))
