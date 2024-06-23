import numpy as np
from p1afempy.data_structures import CoordinatesType


class Monomial():
    x_exponent: int = 0
    y_exponent: int = 0
    coefficient: float = 0

    def __init__(self,
                 x_exponent: float,
                 y_exponent: float,
                 coefficient: int) -> None:
        self.x_exponent = x_exponent
        self.y_exponent = y_exponent
        self.coefficient = coefficient

    @property
    def degree(self) -> int:
        return self.x_exponent + self.y_exponent

    def eval_at(self, coordinates: CoordinatesType) -> np.ndarray:
        xs = coordinates[:, 0]
        ys = coordinates[:, 1]
        return self.coefficient * (np.power(xs, self.x_exponent)
                                   * np.power(ys, self.y_exponent))


class Polynomial():
    monomials: list[Monomial]

    def __init__(self, monomials: list[Monomial]) -> None:
        self.monomials = monomials

    def eval_at(self, coordinates: CoordinatesType) -> np.ndarray:
        # TODO get rid of this for loop
        result = np.zeros(coordinates.shape[0])
        for monomial in self.monomials:
            result += monomial.eval_at(coordinates=coordinates)
        return result

    @property
    def degree(self) -> int:
        return max(monomial.degree for monomial in self.monomials)


def get_random_polynomial(degree: int, max_coeff: float = 1.) -> Polynomial:
    monomials = []
    for x_exponent in range(degree + 1):
        for y_exponent in range(degree - x_exponent + 1):
            random_coeff = np.random.uniform(-max_coeff, max_coeff)
            monomials.append(
                Monomial(x_exponent=x_exponent,
                         y_exponent=y_exponent,
                         coefficient=random_coeff))
    return Polynomial(monomials=monomials)
