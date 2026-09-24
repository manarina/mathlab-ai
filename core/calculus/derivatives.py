from __future__ import annotations

from typing import Any

import sympy as sp


# ============================================================
# PARSING
# ============================================================

def parse_function(expression: str) -> sp.Expr:
    """
    Transforme une expression textuelle en expression SymPy.

    Exemples
    --------
    x**2 + 3*x
    sin(x)
    exp(x)
    sqrt(x)
    """

    if not isinstance(expression, str):
        raise TypeError(
            "L'expression doit être une chaîne de caractères."
        )

    expression = expression.strip()

    if not expression:
        raise ValueError(
            "L'expression ne peut pas être vide."
        )

    x = sp.Symbol("x")

    try:

        return sp.sympify(
            expression,
            locals={
                "x": x,
                "e": sp.E,
                "pi": sp.pi,
            },
        )

    except (
        sp.SympifyError,
        TypeError,
    ) as error:

        raise ValueError(
            f"Expression invalide : {expression}"
        ) from error


# ============================================================
# DERIVEE PREMIERE
# ============================================================

def calculate_derivative(
    function: sp.Expr,
) -> sp.Expr:
    """
    Calcule la dérivée première d'une fonction.
    """

    x = sp.Symbol("x")

    return sp.simplify(
        sp.diff(
            function,
            x,
        )
    )


# ============================================================
# DERIVEE D'ORDRE N
# ============================================================

def calculate_nth_derivative(
    function: sp.Expr,
    order: int,
) -> sp.Expr:
    """
    Calcule la dérivée d'ordre n.
    """

    if not isinstance(order, int):
        raise TypeError(
            "L'ordre de dérivation doit être un entier."
        )

    if order < 0:
        raise ValueError(
            "L'ordre de dérivation doit être positif ou nul."
        )

    x = sp.Symbol("x")

    return sp.simplify(
        sp.diff(
            function,
            x,
            order,
        )
    )


# ============================================================
# DERIVEE SECONDE
# ============================================================

def calculate_second_derivative(
    function: sp.Expr,
) -> sp.Expr:
    """
    Calcule la dérivée seconde.
    """

    return calculate_nth_derivative(
        function,
        2,
    )


# ============================================================
# EVALUATION DE LA DERIVEE
# ============================================================

def evaluate_derivative(
    function: sp.Expr,
    value: Any,
) -> sp.Expr:
    """
    Calcule f'(a).
    """

    x = sp.Symbol("x")

    derivative = calculate_derivative(
        function
    )

    value = sp.sympify(
        value
    )

    return sp.simplify(
        derivative.subs(
            x,
            value,
        )
    )


# ============================================================
# PENTE DE LA TANGENTE
# ============================================================

def calculate_tangent_slope(
    function: sp.Expr,
    point: Any,
) -> sp.Expr:
    """
    Calcule la pente de la tangente à la courbe
    au point x = a.

    La pente est égale à f'(a).
    """

    return evaluate_derivative(
        function,
        point,
    )


# ============================================================
# EQUATION DE LA TANGENTE
# ============================================================

def calculate_tangent_line(
    function: sp.Expr,
    point: Any,
) -> sp.Expr:
    """
    Calcule l'équation de la tangente :

        y = f(a) + f'(a)(x - a)
    """

    x = sp.Symbol("x")

    point = sp.sympify(
        point
    )

    function_value = sp.simplify(
        function.subs(
            x,
            point,
        )
    )

    derivative_value = evaluate_derivative(
        function,
        point,
    )

    tangent = (
        function_value
        + derivative_value * (x - point)
    )

    return sp.expand(
        tangent
    )


# ============================================================
# VALEUR DE LA FONCTION
# ============================================================

def evaluate_function(
    function: sp.Expr,
    value: Any,
) -> sp.Expr:
    """
    Calcule f(a).
    """

    x = sp.Symbol("x")

    value = sp.sympify(
        value
    )

    return sp.simplify(
        function.subs(
            x,
            value,
        )
    )


# ============================================================
# ANALYSE COMPLETE
# ============================================================

def analyze_derivative(
    function: sp.Expr,
    point: Any | None = None,
) -> dict[str, Any]:
    """
    Effectue une analyse complète d'une fonction
    et de ses dérivées.
    """

    derivative = calculate_derivative(
        function
    )

    second_derivative = calculate_second_derivative(
        function
    )

    result: dict[str, Any] = {
        "function": function,
        "derivative": derivative,
        "second_derivative": second_derivative,
    }

    if point is not None:

        point = sp.sympify(
            point
        )

        function_value = evaluate_function(
            function,
            point,
        )

        derivative_value = evaluate_derivative(
            function,
            point,
        )

        tangent = calculate_tangent_line(
            function,
            point,
        )

        result.update({
            "point": point,
            "function_value": function_value,
            "derivative_value": derivative_value,
            "tangent_slope": derivative_value,
            "tangent_line": tangent,
        })

    return result