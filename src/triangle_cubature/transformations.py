import numpy as np
from p1afempy.data_structures import CoordinatesType


def transform_weights(physical_triangle: CoordinatesType,
                      reference_weights: np.ndarray) -> np.ndarray:
    jacobian_determinant = np.linalg.det(
        get_jacobian(physical_triangle=physical_triangle)
    )
    return jacobian_determinant * reference_weights


def transform_integration_points(
        physical_triangle: CoordinatesType,
        reference_integration_points: CoordinatesType) -> CoordinatesType:
    p1 = physical_triangle[0, :]
    jacobian = get_jacobian(physical_triangle=physical_triangle)

    transformed_integration_points = []
    for reference_integration_point in reference_integration_points:
        transformed_integration_point = p1 + jacobian.dot(
            reference_integration_point)
        transformed_integration_points.append(transformed_integration_point)

    return np.array(transformed_integration_points)


def get_jacobian(physical_triangle: CoordinatesType) -> np.ndarray:
    p1 = physical_triangle[0, :]
    p2 = physical_triangle[1, :]
    p3 = physical_triangle[2, :]
    return np.vstack((p2 - p1, p3 - p1)).T
