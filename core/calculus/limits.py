from __future__ import annotations

from typing import Any

import sympy as sp


def parse_function(expression: str) -> sp.Expr:
    """
    Transforme une expression textuelle en expression SymPy.

    Exemple :
        "x**2 + 3*x - 2"
    """

    if not expression or not expression.strip():
        raise ValueError(
            "L'expression de la fonction ne peut pas être vide."
        )

    x = sp.Symbol("x")

    try:
        return sp.sympify(
            expression,
            locals={"x": x},
        )
    except (sp.SympifyError, TypeError) as error:
        raise ValueError(
            "Expression mathématique invalide."
        ) from error


def calculate_limit(
    function: sp.Expr,
    point: Any,
    direction: str = "two_sided",
) -> sp.Expr | None:
    """
    Calcule une limite.

    Directions disponibles :

    - two_sided
    - left
    - right
    - plus_infinity
    - minus_infinity

    Pour une limite bilatérale inexistante, retourne None.
    """

    x = sp.Symbol("x")

    point = sp.sympify(point)

    if direction == "left":

        return sp.limit(
            function,
            x,
            point,
            dir="-",
        )

    if direction == "right":

        return sp.limit(
            function,
            x,
            point,
            dir="+",
        )

    if direction == "plus_infinity":

        return sp.limit(
            function,
            x,
            sp.oo,
        )

    if direction == "minus_infinity":

        return sp.limit(
            function,
            x,
            -sp.oo,
        )

    if direction == "two_sided":

        left_limit = sp.limit(
            function,
            x,
            point,
            dir="-",
        )

        right_limit = sp.limit(
            function,
            x,
            point,
            dir="+",
        )

        # La limite bilatérale existe uniquement
        # si les deux limites latérales sont égales.
        if left_limit == right_limit:
            return left_limit

        # Les limites gauche et droite sont différentes.
        return None

    raise ValueError(
        f"Direction inconnue : {direction}"
    )


def calculate_one_sided_limits(
    function: sp.Expr,
    point: Any,
) -> dict[str, sp.Expr]:
    """
    Calcule les limites à gauche et à droite.
    """

    return {
        "left": calculate_limit(
            function,
            point,
            direction="left",
        ),
        "right": calculate_limit(
            function,
            point,
            direction="right",
        ),
    }


def classify_limit(
    left_limit: sp.Expr,
    right_limit: sp.Expr,
) -> str:
    """
    Classifie une limite à partir des limites
    à gauche et à droite.
    """

    left_limit = sp.sympify(left_limit)
    right_limit = sp.sympify(right_limit)

    if left_limit == right_limit:
        return "finite_or_infinite"

    return "does_not_exist"


def analyze_limit(
    function: sp.Expr,
    point: Any,
) -> dict[str, Any]:
    """
    Analyse complète d'une limite en un point.
    """

    point = sp.sympify(point)

    left_limit = calculate_limit(
        function,
        point,
        direction="left",
    )

    right_limit = calculate_limit(
        function,
        point,
        direction="right",
    )

    limit_type = classify_limit(
        left_limit,
        right_limit,
    )

    if limit_type == "finite_or_infinite":

        limit = left_limit

    else:

        limit = sp.nan

    return {
        "function": function,
        "point": point,
        "left_limit": left_limit,
        "right_limit": right_limit,
        "limit": limit,
        "type": limit_type,
    }


def evaluate_function(
    function: sp.Expr,
    value: Any,
) -> sp.Expr:
    """
    Évalue une fonction en une valeur donnée.
    """

    x = sp.Symbol("x")

    return sp.simplify(
        function.subs(
            x,
            sp.sympify(value),
        )
    )