from __future__ import annotations

import sympy as sp

from core.algebra.horner import (
    horner_evaluate,
)


def parse_polynomial_equation(
    expression: str,
) -> sp.Poly:
    """
    Transforme une expression en polynôme SymPy.

    Exemple :

        x**5 - 3*x**3 + 2*x - 1
    """

    x = sp.symbols("x")

    try:
        expr = sp.sympify(expression)
    except (
        sp.SympifyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Expression polynomiale invalide."
        ) from error

    try:
        polynomial = sp.Poly(
            expr,
            x,
        )
    except (
        sp.PolynomialError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "L'expression ne définit pas "
            "un polynôme valide."
        ) from error

    if polynomial.is_zero:
        raise ValueError(
            "Le polynôme nul ne définit pas "
            "une équation polynomiale classique."
        )

    return polynomial


def get_equation_degree(
    polynomial: sp.Poly,
) -> int:
    """
    Retourne le degré du polynôme.
    """

    return polynomial.degree()


def select_solution_method(
    degree: int,
) -> str:
    """
    Sélectionne la méthode principale selon le degré.
    """

    if degree == 1:
        return "linear"

    if degree == 2:
        return "quadratic"

    if degree == 3:
        return "horner_cardano"

    if degree == 4:
        return "quartic_symbolic"

    if degree >= 5:
        return "factorization_and_numerical"

    raise ValueError(
        "Le degré doit être positif."
    )


def _calculate_numerical_roots(
    expression: sp.Expr,
) -> list[sp.Expr]:
    """
    Calcule les racines numériques d'un polynôme.

    La stratégie est volontairement robuste :

    1. factorisation sans extension complexe ;
    2. calcul numérique de chaque facteur non constant ;
    3. repli sur all_roots() si nroots() ne converge pas.

    Cette approche évite notamment les problèmes
    de convergence liés aux racines multiples.
    """

    polynomial = sp.Poly(
        expression,
        expression.free_symbols.pop(),
    )

    factors = sp.factor_list(
        polynomial.as_expr()
    )[1]

    numerical_roots: list[sp.Expr] = []

    for factor, multiplicity in factors:

        factor_poly = sp.Poly(
            factor,
            polynomial.gens[0],
        )

        if factor_poly.degree() <= 0:
            continue

        try:
            roots = factor_poly.nroots(
                maxsteps=200,
            )

        except (
            sp.NoConvergence,
            ValueError,
            sp.PolynomialError,
        ):
            roots = [
                sp.N(root)
                for root in factor_poly.all_roots()
            ]

        for root in roots:
            numerical_roots.extend(
                [root] * multiplicity
            )

    return numerical_roots


def solve_polynomial_equation(
    polynomial: sp.Poly,
) -> dict:
    """
    Analyse et résout un polynôme de degré quelconque.

    La stratégie dépend du degré :

        degré 1 → méthode linéaire
        degré 2 → méthode quadratique
        degré 3 → Horner / Cardano
        degré 4 → résolution symbolique
        degré >=5 → factorisation + numérique
    """

    degree = polynomial.degree()

    x = polynomial.gens[0]

    expression = polynomial.as_expr()

    method = select_solution_method(
        degree
    )

    factorized = sp.factor(
        expression
    )

    exact_roots = sp.solve(
        expression,
        x,
    )

    numerical_roots = []

    if degree >= 5:
        try:
            numerical_roots = (
                _calculate_numerical_roots(
                    expression
                )
            )
        except (
            sp.PolynomialError,
            ValueError,
        ):
            numerical_roots = []

    return {
        "degree": degree,
        "polynomial": polynomial,
        "method": method,
        "factorized": factorized,
        "exact_roots": exact_roots,
        "numerical_roots": numerical_roots,
        "has_exact_roots": bool(
            exact_roots
        ),
        "has_numerical_roots": bool(
            numerical_roots
        ),
    }


def evaluate_polynomial_with_horner(
    polynomial: sp.Poly,
    value: float,
) -> float:
    """
    Évalue n'importe quel polynôme avec Horner.
    """

    coefficients = [
        float(coefficient)
        for coefficient
        in polynomial.all_coeffs()
    ]

    return horner_evaluate(
        coefficients,
        value,
    )