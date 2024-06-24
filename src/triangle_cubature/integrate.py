from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryConditionType
from triangle_cubature.cubature_rule \
    import WeightsAndIntegrationPoints
from triangle_cubature.transformations import \
    transform_weights_and_integration_points
import numpy as np


def integrate_on_triangle(
        f: BoundaryConditionType, triangle: CoordinatesType,
        weights_and_integration_points: WeightsAndIntegrationPoints) -> float:
    transformed = transform_weights_and_integration_points(
        weights_and_integration_points=weights_and_integration_points,
        physical_triangle=triangle)
    return np.dot(transformed.weights, f(transformed.integration_points))


def integrate_on_mesh(
        f: BoundaryConditionType,
        coordinates: CoordinatesType,
        elements: ElementsType,
        weights_and_integration_points: WeightsAndIntegrationPoints) -> float:
    sum = 0.
    for triangle in elements:
        sum += integrate_on_triangle(
            f=f,
            triangle=coordinates[triangle, :],
            weights_and_integration_points=weights_and_integration_points)
    return sum
