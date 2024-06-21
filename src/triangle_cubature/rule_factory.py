from enum import Enum
import numpy as np
from weights_and_coordinates import WeightsAndCoordinates


class CubatureRule(Enum):
    MIDPOINT = 1


def get_rule(rule: CubatureRule) -> WeightsAndCoordinates:
    if rule == CubatureRule.MIDPOINT:
        return WeightsAndCoordinates(
            weights=np.array([1./2.]),
            coordinates=np.array[[1./3., 1./3.]]
        )
