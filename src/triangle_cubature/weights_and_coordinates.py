from p1afempy.data_structures import CoordinatesType
import numpy as np
from dataclasses import dataclass


@dataclass
class WeightsAndIntegrationPoints:
    weights: np.ndarray
    integration_points: CoordinatesType
