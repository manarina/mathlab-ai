from __future__ import annotations

import sympy as sp  # type: ignore[import-not-found]


# ============================================================
# PARSING
# ============================================================


def parse_polynomial(expression: str) -> sp.Poly:
    """
    Transforme une expression en polynôme SymPy.

    Exemple :
        "x**2 - 5*x + 6"
    """

    x = sp.symbols("x")

    expr = sp.sympify(expression)

    return sp.Poly(expr, x)


# ============================================================
# DEGRE
# ============================================================


def get_polynomial_degree(
    polynomial: sp.Poly,
) -> int:
    """
    Retourne le degré du polynôme.
    """

    return polynomial.degree()


# ============================================================
# COEFFICIENTS
# ============================================================


def get_polynomial_coefficients(
    polynomial: sp.Poly,
) -> list[sp.Expr]:
    """
    Retourne les coefficients du polynôme.
    """

    return polynomial.all_coeffs()


# ============================================================
# RACINES
# ============================================================


def get_polynomial_roots(
    polynomial: sp.Poly,
) -> list[sp.Expr]:
    """
    Retourne les racines du polynôme.
    """

    return sp.solve(
        polynomial.as_expr(),
        polynomial.gens[0],
    )


# ============================================================
# FACTORISATION
# ============================================================


def factor_polynomial(
    polynomial: sp.Poly,
) -> sp.Expr:
    """
    Retourne la forme factorisée du polynôme.
    """

    return sp.factor(
        polynomial.as_expr()
    )


# ============================================================
# DERIVEE
# ============================================================


def differentiate_polynomial(
    polynomial: sp.Poly,
) -> sp.Poly:
    """
    Calcule la dérivée du polynôme.
    """

    derivative = sp.diff(
        polynomial.as_expr(),
        polynomial.gens[0],
    )

    return sp.Poly(
        derivative,
        polynomial.gens[0],
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_polynomial(
    polynomial: sp.Poly,
    value: float,
) -> sp.Expr:
    """
    Évalue le polynôme pour x = value.
    """

    return polynomial.eval(value)