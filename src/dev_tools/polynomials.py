import numpy as np
from p1afempy.data_structures import CoordinatesType, ElementsType
import sympy
from tqdm import tqdm
from math import factorial, comb


def multinomial_coefficient(*args):
    """
    Provided k_1, ..., k_m,
    returns the multinomial coefficient, i.e.
    n! / (k_1! * ... * k_m!),
    where n = k_1 + ... + k_m.

    Source: https://en.wikipedia.org/wiki/Multinomial_theorem
    """
    n = sum(args)
    result = factorial(n)
    for k in args:
        result /= factorial(k)
    return result


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


def integrate_on_triangle_using_sympy(
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


def integrate_on_triangle_by_hand(
        polynomial: Polynomial,
        vertices: np.ndarray) -> float:
    """
    integrates the polynomial on the specified triangle by hand

    polynomial: Polynomial
        the polynomial to be integrated
    vertices: np.ndarray
        the vertices of the triangle in counter-clockwise order
    """
    r1 = vertices[0, :]
    r2 = vertices[1, :]
    r3 = vertices[2, :]

    r1x, r1y = r1[0], r1[1]
    r2x, r2y = r2[0], r2[1]
    r3x, r3y = r3[0], r3[1]

    d2 = r2 - r1
    d3 = r3 - r1

    dPhi = np.column_stack([d2, d3])
    det_dPhi = np.linalg.det(dPhi)

    result_by_hand = 0.
    for monomial in polynomial.monomials:
        x_exponent = monomial.x_exponent
        y_exponent = monomial.y_exponent
        coefficient = monomial.coefficient

        integral_monomial_triangle = 0
        for m1 in range(x_exponent + 1):
            for m2 in range(x_exponent - m1 + 1):
                m3 = x_exponent - m1 - m2
                for n1 in range(y_exponent + 1):
                    for n2 in range(y_exponent - n1 + 1):
                        n3 = y_exponent - n1 - n2

                        C = (
                            multinomial_coefficient(m1, m2, m3) *
                            multinomial_coefficient(n1, n2, n3) *
                            r1x**m1 * r1y**n1 *
                            (r2x - r1x)**m2 * (r2y - r1y)**n2 *
                            (r3x - r1x)**m3 * (r3y - r1y)**n3)

                        a = m2 + n2
                        b = m3 + n3

                        integral_monomial_ref_triangle = 0.
                        for k in range(b + 2):
                            integral_monomial_ref_triangle += (
                                comb(b+1, k) * (-1.)**k * (1. / (a + k + 1)))
                        integral_monomial_ref_triangle *= (1./(b+1))

                        integral_monomial_triangle += (
                            C * integral_monomial_ref_triangle)
        result_by_hand += coefficient * integral_monomial_triangle
    result_by_hand *= det_dPhi

    return result_by_hand


def integrate_on_triangle(
        polynomial: Polynomial,
        vertices: np.ndarray,
        using_sympy: bool = True) -> float:
    """
    integrates the polynomial on the specified triangle

    polynomial: Polynomial
        the polynomial to be integrated
    vertices: np.ndarray
        the vertices of the triangle in counter-clockwise order
    """
    if using_sympy:
        return integrate_on_triangle_using_sympy(
            polynomial=polynomial, vertices=vertices)
    return integrate_on_triangle_by_hand(
        polynomial=polynomial, vertices=vertices)


def integrate_on_mesh(
        polynomial: Polynomial,
        elements: ElementsType,
        vertices: CoordinatesType,
        display_progress: bool = False,
        using_sympy: bool = True) -> float:
    sum = 0.
    for element in tqdm(elements, disable=not display_progress):
        sum += integrate_on_triangle(
            polynomial=polynomial,
            vertices=vertices[element, :],
            using_sympy=using_sympy)
    return sum
