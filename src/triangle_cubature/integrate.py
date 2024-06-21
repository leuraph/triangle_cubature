from p1afempy.data_structures import CoordinatesType
from weights_and_coordinates import WeightsAndCoordinates


def integrate_on_triangle(
        coordinates: CoordinatesType,
        rule: WeightsAndCoordinates) -> float:
    # TODO IMPLEMENT
    return 0.


def integrate_on_mesh(
        coordinates: CoordinatesType,
        rule: WeightsAndCoordinates) -> float:
    sum = 0.
    for triangle in coordinates:
        sum += integrate_on_triangle(
            coordinates=triangle,
            rule=rule)
    return sum
