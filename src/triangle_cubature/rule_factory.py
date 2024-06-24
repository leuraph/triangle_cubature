from enum import Enum
import numpy as np
from triangle_cubature.weights_and_integration_points \
    import WeightsAndIntegrationPoints


class CubatureRule(Enum):
    MIDPOINT = 1
    LAUFFER = 2


def get_rule(rule: CubatureRule) -> WeightsAndIntegrationPoints:
    if rule == CubatureRule.MIDPOINT:
        return WeightsAndIntegrationPoints(
            weights=np.array([1./2.]),
            integration_points=np.array([1./3., 1./3.]).reshape(1, 2)
        )
    if rule == CubatureRule.LAUFFER:
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
        return WeightsAndIntegrationPoints(
            integration_points=integration_points,
            weights=weights)
    raise ValueError('specified rule does not exist.')
