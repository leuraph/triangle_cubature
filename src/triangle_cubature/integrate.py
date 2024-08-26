from p1afempy.data_structures import \
    CoordinatesType, ElementsType
from triangle_cubature.cubature_rule \
    import CubatureRuleEnum
from triangle_cubature.transformations import \
    transform_weights_and_integration_points
from triangle_cubature.rule_factory import get_rule
from typing import Callable
import numpy as np


def integrate_on_triangle(
        f: Callable[[CoordinatesType], np.ndarray],
        triangle: CoordinatesType,
        cubature_rule: CubatureRuleEnum) -> float:
    """
    approximates the integral of the function provided
    on the triangle at hand using the specified cubature rule

    parameters
    ----------
    f: Callable[[CoordinatesType], np.ndarray]
        the function to be integrated
    triangle: CoordinatesType
        the coordinates of the triangle's vertices
        in counter-clockwise order
    cubature_rule: CubatureRuleEnum
        the cubature rule to be used

    returns
    -------
    float: the approximated value of the integral

    notes
    -----
    - the function f must be able to
      handle inputs of shape (N, 2), i.e.
      coordinates as array
    """
    waip = get_rule(rule=cubature_rule).weights_and_integration_points
    transformed = transform_weights_and_integration_points(
        weights_and_integration_points=waip,
        physical_triangle=triangle)
    return np.dot(transformed.weights, f(transformed.integration_points))


def integrate_on_mesh(
        f: Callable[[CoordinatesType], np.ndarray],
        coordinates: CoordinatesType,
        elements: ElementsType,
        cubature_rule: CubatureRuleEnum) -> float:
    """
    approximates the integral of the function provided
    over the mesh at hand using the specified cubature rule

    parameters
    ----------
    f: Callable[[CoordinatesType], np.ndarray]
        the function to be integrated
    coordinates: CoordinatesType
        vrtices of the mesh
    elements: ElementsType
        the elements of the mesh
    cubature_rule: CubatureRuleEnum
        the cubature rule to be used

    returns
    -------
    float: the approximated value of the integral

    notes
    -----
    - the function f must be able to
      handle inputs of shape (N, 2), i.e.
      coordinates as array
    """

    # fully vectorized midpoint integration, based on the following paper:
    # --------------------------------------------------------------------
    # Funken, Stefan, Dirk Praetorius, and Philipp Wissgott.
    # Efficient Implementation of Adaptive P1-FEM in Matlab.
    # Computational Methods in Applied Mathematics 11,
    # no. 4 (1 January 2011): 460–90. https://doi.org/10.2478/cmam-2011-0026.
    if cubature_rule == CubatureRuleEnum.MIDPOINT:
        c1 = coordinates[elements[:, 0]]
        d21 = coordinates[elements[:, 1]] - c1
        d31 = coordinates[elements[:, 2]] - c1

        # vector of element areas 4*|T|
        areas = 0.5 * (d21[:, 0]*d31[:, 1] - d21[:, 1] * d31[:, 0])

        midpoints = c1 + (d21 + d31)/3.

        f_on_midpoints = f(midpoints)

        return np.dot(areas, f_on_midpoints)

    sum = 0.
    for triangle in elements:
        sum += integrate_on_triangle(
            f=f,
            triangle=coordinates[triangle, :],
            cubature_rule=cubature_rule)
    return sum
