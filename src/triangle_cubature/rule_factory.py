from enum import Enum
import numpy as np
from triangle_cubature.weights_and_integration_points \
    import WeightsAndIntegrationPoints
from dataclasses import dataclass


class CubatureRuleEnum(Enum):
    MIDPOINT = 1
    LAUFFER = 2


@dataclass
class CubatureRule:
    weights_and_integration_points: WeightsAndIntegrationPoints
    degree_of_exactness: int
    name: str


def get_rule(rule: CubatureRuleEnum) -> WeightsAndIntegrationPoints:
    """
    given a cubature rule, returns the corresponding
    weight(s) and integration point(s)

    Notes
    -----
    - the rules correspond to the rules as specified in [1]

    References
    ----------
    - [1] Stenger, Frank.
      'Approximate Calculation of Multiple Integrals (A. H. Stroud)'.
      SIAM Review 15, no. 1 (January 1973): 234-35.
      https://doi.org/10.1137/1015023. p. 306-315
    """
    if rule == CubatureRuleEnum.MIDPOINT:
        weights = np.array([1./2.])
        integration_points = np.array([1./3., 1./3.]).reshape(1, 2)
        name = 'midpoint'
        degree_of_exactness = 1
        return WeightsAndIntegrationPoints(
            weights=weights,
            integration_points=integration_points)
    if rule == CubatureRuleEnum.LAUFFER:
        integration_points = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        weights = np.array([
            1/3 * 0.5,
            1/3 * 0.5,
            1/3 * 0.5
        ])
        name = 'lauffer'
        degree_of_exactness = 1
        return WeightsAndIntegrationPoints(
            integration_points=integration_points,
            weights=weights)
    raise ValueError('specified rule does not exist.')
