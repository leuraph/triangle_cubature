from p1afempy.data_structures import CoordinatesType
import numpy as np
from dataclasses import dataclass


@dataclass
class WeightsAndCoordinates:
    weights: np.ndarray
    coordinates: CoordinatesType
