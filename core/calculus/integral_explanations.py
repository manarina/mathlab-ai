"""
Calculus Lab - Integral Explanations
====================================

Utilities for computing and explaining integrals step by step.

This module is intentionally independent from the Streamlit UI so it can
be tested and reused by the Calculus Lab interface.

Supported:
- Indefinite integrals
- Definite integrals
- Basic polynomial, exponential and trigonometric integrals
- Step-by-step explanations
- Verification by differentiation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class IntegralResult:
    """Result returned by an integral computation."""

    expression: sp.Expr
    variable: sp.Symbol
    integral: sp.Expr
    steps: list[str]
    verification: sp.Expr
    is_definite: bool = False
    lower_bound: Optional[sp.Expr] = None
    upper_bound: Optional[sp.Expr] = None


# ============================================================
# PARSING
# ============================================================


def parse_expression(expression: str, variable: str = "x") -> tuple[sp.Expr, sp.Symbol]:
    """
    Parse a mathematical expression.

    Parameters
    ----------
    expression:
        Expression entered by the user, e.g. "x**2 + 3*x".
    variable:
        Integration variable, normally "x".

    Returns
    -------
    tuple[sp.Expr, sp.Symbol]
        Parsed SymPy expression and variable.

    Raises
    ------
    ValueError
        If the expression cannot be parsed.
    """

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("L'expression ne peut pas être vide.")

    if not isinstance(variable, str) or not variable.strip():
        raise ValueError("La variable d'intégration est obligatoire.")

    symbol = sp.Symbol(variable.strip())

    try:
        expr = sp.sympify(expression.strip())
    except (sp.SympifyError, SyntaxError) as exc:
        raise ValueError(
            f"Expression mathématique invalide : {expression}"
        ) from exc

    return expr, symbol


# ============================================================
# FORMATTING
# ============================================================


def format_expression(expression: sp.Expr) -> str:
    """
    Return a readable LaTeX representation of an expression.
    """

    return sp.latex(sp.simplify(expression))


def format_plain(expression: sp.Expr) -> str:
    """
    Return a readable textual representation of an expression.
    """

    return str(sp.simplify(expression))


# ============================================================
# CLASSIFICATION
# ============================================================


def classify_integrand(expression: sp.Expr, variable: sp.Symbol) -> str:
    """
    Classify an integrand to provide an appropriate explanation.

    Returns one of:
    - polynomial
    - constant
    - exponential
    - logarithmic
    - trigonometric
    - rational
    - general
    """

    expr = sp.expand(expression)

    if not expr.has(variable):
        return "constant"

    if expr.is_polynomial(variable):
        return "polynomial"

    if expr.has(sp.exp(variable)):
        return "exponential"

    if expr.has(sp.log(variable)):
        return "logarithmic"

    if expr.has(sp.sin(variable), sp.cos(variable), sp.tan(variable)):
        return "trigonometric"

    if sp.denom(sp.together(expr)).has(variable):
        return "rational"

    return "general"


# ============================================================
# INDEFINITE INTEGRAL
# ============================================================


def integrate_indefinite(
    expression: str,
    variable: str = "x",
) -> IntegralResult:
    """
    Compute an indefinite integral and generate an explanation.

    Example
    -------
    integrate_indefinite("x**2")

    Result:
        x**3 / 3 + C
    """

    expr, symbol = parse_expression(expression, variable)

    try:
        primitive = sp.integrate(expr, symbol)
    except Exception as exc:
        raise ValueError(
            f"Impossible de calculer l'intégrale de {expression}."
        ) from exc

    kind = classify_integrand(expr, symbol)

    steps: list[str] = []

    # --------------------------------------------------------
    # General first step
    # --------------------------------------------------------

    steps.append(
        f"On cherche une primitive de {sp.latex(expr)} par rapport à "
        f"{symbol}."
    )

    # --------------------------------------------------------
    # Constant
    # --------------------------------------------------------

    if kind == "constant":
        steps.append(
            f"L'expression est constante par rapport à {symbol}."
        )

        steps.append(
            f"On utilise : "
            f"\\int a\\,d{symbol} = a{symbol} + C."
        )

    # --------------------------------------------------------
    # Polynomial
    # --------------------------------------------------------

    elif kind == "polynomial":
        terms = sp.Add.make_args(sp.expand(expr))

        if len(terms) > 1:
            steps.append(
                "On utilise la linéarité de l'intégrale : "
                "\\int (f+g) = \\int f + \\int g."
            )

        for term in terms:
            coefficient, power = term.as_coeff_exponent(symbol)

            if power == 0:
                steps.append(
                    f"Pour le terme constant "
                    f"{sp.latex(term)}, on obtient "
                    f"{sp.latex(sp.integrate(term, symbol))}."
                )

            elif power == -1:
                steps.append(
                    f"Pour {sp.latex(term)}, on utilise : "
                    f"\\int \\frac{{1}}{{{symbol}}}\\,d{symbol}"
                    f" = \\ln|{symbol}| + C."
                )

            else:
                steps.append(
                    f"Pour {sp.latex(term)}, on utilise la règle : "
                    f"\\int {symbol}^n\\,d{symbol}"
                    f" = \\frac{{{symbol}^{{n+1}}}}{{n+1}} + C."
                )

    # --------------------------------------------------------
    # Exponential
    # --------------------------------------------------------

    elif kind == "exponential":
        steps.append(
            "On utilise la primitive de l'exponentielle : "
            f"\\int e^{{{symbol}}}\\,d{symbol}"
            f" = e^{{{symbol}}} + C."
        )

    # --------------------------------------------------------
    # Logarithmic
    # --------------------------------------------------------

    elif kind == "logarithmic":
        steps.append(
            "La présence d'un logarithme est traitée en utilisant "
            "les règles de primitives correspondantes."
        )

    # --------------------------------------------------------
    # Trigonometric
    # --------------------------------------------------------

    elif kind == "trigonometric":
        steps.append(
            "On utilise les primitives usuelles des fonctions "
            "trigonométriques."
        )

        if expr.has(sp.sin(symbol)):
            steps.append(
                f"On utilise : "
                f"\\int \\sin({symbol})\\,d{symbol}"
                f" = -\\cos({symbol}) + C."
            )

        if expr.has(sp.cos(symbol)):
            steps.append(
                f"On utilise : "
                f"\\int \\cos({symbol})\\,d{symbol}"
                f" = \\sin({symbol}) + C."
            )

    # --------------------------------------------------------
    # Rational
    # --------------------------------------------------------

    elif kind == "rational":
        steps.append(
            "L'expression est rationnelle. SymPy applique les "
            "techniques adaptées pour déterminer une primitive."
        )

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    else:
        steps.append(
            "On applique les règles générales d'intégration "
            "disponibles."
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    steps.append(
        f"Une primitive obtenue est : "
        f"{sp.latex(primitive)}."
    )

    steps.append(
        "On ajoute la constante d'intégration C."
    )

    # SymPy's integrate() returns one primitive without C.
    result_with_c = primitive + sp.Symbol("C")

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    verification = sp.simplify(sp.diff(primitive, symbol) - expr)

    steps.append(
        f"Vérification : en dérivant la primitive, "
        f"on retrouve {sp.latex(expr)}."
    )

    return IntegralResult(
        expression=expr,
        variable=symbol,
        integral=result_with_c,
        steps=steps,
        verification=verification,
        is_definite=False,
    )


# ============================================================
# DEFINITE INTEGRAL
# ============================================================


def integrate_definite(
    expression: str,
    lower_bound: str | int | float,
    upper_bound: str | int | float,
    variable: str = "x",
) -> IntegralResult:
    """
    Compute a definite integral.

    Example
    -------
    integrate_definite("x**2", 0, 2)

    Result:
        8/3
    """

    expr, symbol = parse_expression(expression, variable)

    try:
        lower = sp.sympify(lower_bound)
        upper = sp.sympify(upper_bound)
    except (sp.SympifyError, TypeError) as exc:
        raise ValueError(
            "Les bornes d'intégration sont invalides."
        ) from exc

    try:
        primitive = sp.integrate(expr, symbol)
        result = sp.integrate(expr, (symbol, lower, upper))
    except Exception as exc:
        raise ValueError(
            f"Impossible de calculer l'intégrale définie "
            f"de {expression}."
        ) from exc

    steps = [
        f"On cherche d'abord une primitive de "
        f"{sp.latex(expr)}.",
        f"Une primitive est : {sp.latex(primitive)}.",
        (
            f"On applique le théorème fondamental de l'analyse : "
            f"\\int_{{{sp.latex(lower)}}}^{{{sp.latex(upper)}}}"
            f"{sp.latex(expr)}\\,d{symbol}"
            f" = [F({symbol})]_{{{sp.latex(lower)}}}"
            f"^{{{sp.latex(upper)}}}."
        ),
        (
            f"On calcule F({sp.latex(upper)}) - "
            f"F({sp.latex(lower)})."
        ),
        f"Résultat : {sp.latex(result)}.",
    ]

    verification = sp.simplify(
        sp.diff(primitive, symbol) - expr
    )

    return IntegralResult(
        expression=expr,
        variable=symbol,
        integral=result,
        steps=steps,
        verification=verification,
        is_definite=True,
        lower_bound=lower,
        upper_bound=upper,
    )


# ============================================================
# SIMPLE API
# ============================================================


def calculate_integral(
    expression: str,
    variable: str = "x",
    lower_bound: Optional[str | int | float] = None,
    upper_bound: Optional[str | int | float] = None,
) -> IntegralResult:
    """
    Unified integral calculator.

    If lower_bound and upper_bound are provided, a definite integral
    is calculated. Otherwise an indefinite integral is calculated.
    """

    if (lower_bound is None) != (upper_bound is None):
        raise ValueError(
            "Les deux bornes doivent être fournies pour une "
            "intégrale définie."
        )

    if lower_bound is not None and upper_bound is not None:
        return integrate_definite(
            expression=expression,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            variable=variable,
        )

    return integrate_indefinite(
        expression=expression,
        variable=variable,
    )


# ============================================================
# VERIFICATION
# ============================================================


def verify_integral(
    expression: str,
    integral: sp.Expr,
    variable: str = "x",
) -> bool:
    """
    Verify an indefinite integral by differentiating it.

    The constant C is automatically ignored.
    """

    expr, symbol = parse_expression(expression, variable)

    if isinstance(integral, str):
        try:
            integral_expr = sp.sympify(integral)
        except sp.SympifyError as exc:
            raise ValueError(
                "La primitive fournie est invalide."
            ) from exc
    else:
        integral_expr = integral

    derivative = sp.diff(integral_expr, symbol)

    return sp.simplify(derivative - expr) == 0


def is_exact_integral(
    expression: str,
    integral: sp.Expr,
    variable: str = "x",
) -> bool:
    """
    Alias for verify_integral().
    """

    return verify_integral(
        expression=expression,
        integral=integral,
        variable=variable,
    )


# ============================================================
# PRESENTATION
# ============================================================


def explain_integral(
    expression: str,
    variable: str = "x",
    lower_bound: Optional[str | int | float] = None,
    upper_bound: Optional[str | int | float] = None,
) -> dict:
    """
    Return a UI-friendly dictionary for the Calculus Lab.
    """

    result = calculate_integral(
        expression=expression,
        variable=variable,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return {
        "expression": result.expression,
        "variable": result.variable,
        "integral": result.integral,
        "integral_latex": format_expression(result.integral),
        "steps": result.steps,
        "verification": result.verification,
        "verification_ok": result.verification == 0,
        "is_definite": result.is_definite,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
    }