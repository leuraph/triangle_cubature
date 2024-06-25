from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryConditionType
from triangle_cubature.cubature_rule \
    import CubatureRuleEnum
from triangle_cubature.transformations import \
    transform_weights_and_integration_points
from triangle_cubature.rule_factory import get_rule
import numpy as np


def integrate_on_triangle(
        f: BoundaryConditionType, triangle: CoordinatesType,
        cubature_rule: CubatureRuleEnum) -> float:
    waip = get_rule(rule=cubature_rule).weights_and_integration_points
    transformed = transform_weights_and_integration_points(
        weights_and_integration_points=waip,
        physical_triangle=triangle)
    return np.dot(transformed.weights, f(transformed.integration_points))


def integrate_on_mesh(
        f: BoundaryConditionType,
        coordinates: CoordinatesType,
        elements: ElementsType,
        cubature_rule: CubatureRuleEnum) -> float:
    sum = 0.
    for triangle in elements:
        sum += integrate_on_triangle(
            f=f,
            triangle=coordinates[triangle, :],
            cubature_rule=cubature_rule)
    return sum
