from p1afempy.data_structures import CoordinatesType, ElementsType
from triangle_cubature.weights_and_integration_points \
    import WeightsAndIntegrationPoints


def integrate_on_triangle(
        f: callable,
        triangle: CoordinatesType,
        rule: WeightsAndIntegrationPoints) -> float:
    # TODO IMPLEMENT
    return 0.


def integrate_on_mesh(
        f: callable,
        coordinates: CoordinatesType,
        elements: ElementsType,
        rule: WeightsAndIntegrationPoints) -> float:
    sum = 0.
    for triangle in coordinates:
        sum += integrate_on_triangle(
            coordinates=triangle,
            rule=rule)
    return sum
