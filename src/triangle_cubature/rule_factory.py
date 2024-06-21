from enum import Enum
import numpy as np
from triangle_cubature.weights_and_integration_points \
    import WeightsAndIntegrationPoints


class CubatureRule(Enum):
    MIDPOINT = 1


def get_rule(rule: CubatureRule) -> WeightsAndIntegrationPoints:
    if rule == CubatureRule.MIDPOINT:
        return WeightsAndIntegrationPoints(
            weights=np.array([1./2.]),
            integration_points=np.array[[1./3., 1./3.]]
        )
