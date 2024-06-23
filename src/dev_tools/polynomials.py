import numpy as np
from p1afempy.data_structures import CoordinatesType
import sympy


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


def integrate_on_triangle(
        polynomial: Polynomial,
        vertices: np.ndarray) -> float:
    """
    integrates the polynomial on the specified triangle symbolically

    polynomial: Polynomial
        the polynomial to be integrated
    vertices: np.ndarray
        the vertices of the triangle in counter-clockwise order
    """
    x, y = sympy.symbols('x y')  # physical coordinates
    m, n = sympy.symbols('m n', integer=True)  # x-, and y-exponents
    c = sympy.symbols('c')  # coefficient of monomial
    monomial_template = c * x**m * y**n

    # creating the symbolic representation of the polynomial
    # to be integrated (in physical coordinates)
    p = 0.
    for monomial in polynomial.monomials:
        p = p + monomial_template.subs([
            (m, monomial.x_exponent),
            (n, monomial.y_exponent),
            (c, monomial.coefficient)])

    r1 = vertices[0, :]
    r2 = vertices[1, :]
    r3 = vertices[2, :]

    r_1x, r_1y = sympy.symbols('r_1x r_1y')
    r_2x, r_2y = sympy.symbols('r_2x r_2y')
    r_3x, r_3y = sympy.symbols('r_3x r_3y')
    x_hat, y_hat = sympy.symbols('x_hat y_hat')

    Phi = sympy.Matrix([
        r_1x + x_hat*(r_2x - r_1x) + y_hat*(r_3x - r_1x),
        r_1y + x_hat*(r_2y - r_1y) + y_hat*(r_3y - r_1y),
    ])
    jacobian = Phi.jacobian(sympy.Matrix([x_hat, y_hat]))
    det_DPhi = sympy.det(jacobian)

    integrand = p.subs([
        (x, Phi[0]),
        (y, Phi[1])
    ])

    result = sympy.integrate(
        integrand, (y_hat, 0, 1-x_hat), (x_hat, 0, 1)) * det_DPhi

    numerical_result = result.subs([
        (r_1x, r1[0]),
        (r_1y, r1[1]),
        (r_2x, r2[0]),
        (r_2y, r2[1]),
        (r_3x, r3[0]),
        (r_3y, r3[1])
    ]).evalf()

    return float(numerical_result)
