import sympy as sp

from core.algebra.polynomials import (
    parse_polynomial,
    get_polynomial_degree,
    get_polynomial_coefficients,
    get_polynomial_roots,
    factor_polynomial,
    differentiate_polynomial,
    evaluate_polynomial,
)


# ============================================================
# PARSING
# ============================================================


def test_parse_polynomial():

    polynomial = parse_polynomial(
        "x**2 - 5*x + 6"
    )

    assert isinstance(
        polynomial,
        sp.Poly,
    )


# ============================================================
# DEGRE
# ============================================================


def test_polynomial_degree():

    polynomial = parse_polynomial(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    degree = get_polynomial_degree(
        polynomial
    )

    assert degree == 3


# ============================================================
# COEFFICIENTS
# ============================================================


def test_polynomial_coefficients():

    polynomial = parse_polynomial(
        "2*x**3 - 3*x**2 + 4*x - 7"
    )

    coefficients = get_polynomial_coefficients(
        polynomial
    )

    assert coefficients == [
        2,
        -3,
        4,
        -7,
    ]


# ============================================================
# RACINES
# ============================================================


def test_polynomial_roots():

    polynomial = parse_polynomial(
        "x**2 - 5*x + 6"
    )

    roots = get_polynomial_roots(
        polynomial
    )

    assert set(roots) == {
        sp.Integer(2),
        sp.Integer(3),
    }


# ============================================================
# FACTORISATION
# ============================================================


def test_polynomial_factorization():

    polynomial = parse_polynomial(
        "x**2 - 5*x + 6"
    )

    factorized = factor_polynomial(
        polynomial
    )

    x = sp.symbols("x")

    assert factorized == (
        (x - 2) * (x - 3)
    )


# ============================================================
# DERIVEE
# ============================================================


def test_polynomial_derivative():

    polynomial = parse_polynomial(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    derivative = differentiate_polynomial(
        polynomial
    )

    x = sp.symbols("x")

    expected = sp.Poly(
        3*x**2 - 12*x + 11,
        x,
    )

    assert derivative == expected


# ============================================================
# EVALUATION
# ============================================================


def test_polynomial_evaluation():

    polynomial = parse_polynomial(
        "x**2 - 5*x + 6"
    )

    result = evaluate_polynomial(
        polynomial,
        2,
    )

    assert result == 0