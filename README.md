# Triangle Cubature Rules
This repo serves as a collection of well-tested triangle cubature rules,
i.e. numerical integration schemes for integrals of the form

$$
\int_K f(x, y) ~\mathrm{d}x ~\mathrm{d}y,
$$

where $K \subset \mathbb{R}^2$ is a triangle.
All cubature rules are based on [1].

## (Unit) Tests
To run auto tests, you do
```sh
python -m unittest discover tests/auto/
```

> The unit tests use `sympy` to verify the degree of exactness of the
> implemented cubature rules, i.e. creates random polynomials $p_d$ of the 
> expected degree of exactness $d$ and compares the exact result of
> $\int_K p_d(x, y) ~\mathrm{d}x ~\mathrm{d}y$ to the value obtained
> with the cubature rule at hand.

## References
- [1] Stenger, Frank.
    'Approximate Calculation of Multiple Integrals (A. H. Stroud)'.
    SIAM Review 15, no. 1 (January 1973): 234-35.
    https://doi.org/10.1137/1015023. p. 306-315