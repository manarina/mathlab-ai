import sympy as sp

from core.algebra.polynomials import (
    parse_polynomial,
    get_polynomial_roots,
    factor_polynomial,
    differentiate_polynomial,
)

from core.algebra.polynomial_explanations import (
    explain_polynomial_analysis,
)


# ============================================================
# PREPARATION
# ============================================================


def analyze_polynomial(expression: str):
    """
    Prépare les résultats nécessaires aux explications.
    """

    polynomial = parse_polynomial(expression)

    roots = get_polynomial_roots(
        polynomial
    )

    factorized = factor_polynomial(
        polynomial
    )

    derivative = differentiate_polynomial(
        polynomial
    )

    return (
        polynomial,
        roots,
        factorized,
        derivative,
    )


# ============================================================
# STRUCTURE GENERALE
# ============================================================


def test_polynomial_explanation_contains_main_steps():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "x**2 - 5*x + 6"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    assert len(steps) == 7

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier le polynôme" in titles
    assert "Déterminer le degré" in titles
    assert "Identifier les coefficients" in titles
    assert "Rechercher les racines" in titles
    assert "Déterminer les racines" in titles
    assert "Factoriser le polynôme" in titles
    assert "Calculer la dérivée" in titles


# ============================================================
# IDENTIFICATION
# ============================================================


def test_polynomial_explanation_identifies_expression():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "x**2 - 5*x + 6"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    assert steps[0]["title"] == (
        "Identifier le polynôme"
    )

    assert steps[0]["formula"] == (
        r"P(x) = x^{2} - 5 x + 6"
    )


# ============================================================
# DEGRE
# ============================================================


def test_polynomial_explanation_degree():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    degree_step = steps[1]

    assert degree_step["title"] == (
        "Déterminer le degré"
    )

    assert degree_step["formula"] == (
        r"\deg(P) = 3"
    )


# ============================================================
# COEFFICIENTS
# ============================================================


def test_polynomial_explanation_coefficients():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "2*x**3 - 3*x**2 + 4*x - 7"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    coefficients_step = steps[2]

    assert coefficients_step["title"] == (
        "Identifier les coefficients"
    )

    assert coefficients_step["formula"] == (
        r"a_{3} = 2,\quad "
        r"a_{2} = -3,\quad "
        r"a_{1} = 4,\quad "
        r"a_{0} = -7"
    )


# ============================================================
# RACINES
# ============================================================


def test_polynomial_explanation_roots():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "x**2 - 5*x + 6"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    roots_step = steps[4]

    assert roots_step["title"] == (
        "Déterminer les racines"
    )

    assert roots_step["formula"] == (
        r"x_{1} = 2,\quad x_{2} = 3"
    )


# ============================================================
# FACTORISATION
# ============================================================


def test_polynomial_explanation_factorization():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "x**2 - 5*x + 6"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    factorization_step = steps[5]

    assert factorization_step["title"] == (
        "Factoriser le polynôme"
    )

    assert "x - 2" in factorization_step["formula"]
    assert "x - 3" in factorization_step["formula"]


# ============================================================
# DERIVEE
# ============================================================


def test_polynomial_explanation_derivative():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    derivative_step = steps[6]

    assert derivative_step["title"] == (
        "Calculer la dérivée"
    )

    assert derivative_step["formula"] == (
        r"P'(x) = 3 x^{2} - 12 x + 11"
    )


# ============================================================
# POLYNOME SANS RACINE REELLE
# ============================================================


def test_polynomial_explanation_complex_roots():

    (
        polynomial,
        roots,
        factorized,
        derivative,
    ) = analyze_polynomial(
        "x**2 + 1"
    )

    steps = explain_polynomial_analysis(
        polynomial,
        roots,
        factorized,
        derivative,
    )

    roots_step = steps[4]

    assert roots_step["title"] == (
        "Déterminer les racines"
    )

    assert roots_step["formula"] == (
        r"x_{1} = - i,\quad x_{2} = i"
    )

    assert (
        "racines"
        in roots_step["explanation"]
    )