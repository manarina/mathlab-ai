from __future__ import annotations

from typing import Any

import sympy as sp


def explain_derivative(
    function: sp.Expr,
    result: dict[str, Any],
) -> list[dict]:
    """
    Génère une explication étape par étape
    de l'analyse d'une dérivée.

    Le résultat attendu contient au minimum :

    {
        "function": ...,
        "derivative": ...,
        "second_derivative": ...
    }

    Si un point est fourni :

    {
        "point": ...,
        "function_value": ...,
        "derivative_value": ...,
        "tangent_slope": ...,
        "tangent_line": ...
    }
    """

    x = sp.Symbol("x")

    function = sp.sympify(function)

    steps: list[dict] = []

    # ========================================================
    # 1. IDENTIFIER LA FONCTION
    # ========================================================

    steps.append({
        "title": "Identifier la fonction",
        "formula": (
            rf"f(x) = {sp.latex(function)}"
        ),
        "explanation": (
            "On identifie la fonction dont on souhaite "
            "étudier la dérivée."
        ),
    })

    # ========================================================
    # 2. CALCULER LA DERIVEE PREMIERE
    # ========================================================

    derivative = sp.sympify(
        result["derivative"]
    )

    steps.append({
        "title": "Calculer la dérivée première",
        "formula": (
            rf"f'(x) = {sp.latex(derivative)}"
        ),
        "explanation": (
            "On dérive la fonction par rapport à x "
            "en appliquant les règles de dérivation "
            "appropriées."
        ),
    })

    # ========================================================
    # 3. CALCULER LA DERIVEE SECONDE
    # ========================================================

    second_derivative = sp.sympify(
        result["second_derivative"]
    )

    steps.append({
        "title": "Calculer la dérivée seconde",
        "formula": (
            rf"f''(x) = {sp.latex(second_derivative)}"
        ),
        "explanation": (
            "On dérive une seconde fois la fonction "
            "pour obtenir la dérivée seconde."
        ),
    })

    # ========================================================
    # 4. ANALYSE EN UN POINT
    # ========================================================

    if "point" in result:

        point = sp.sympify(
            result["point"]
        )

        function_value = sp.sympify(
            result["function_value"]
        )

        derivative_value = sp.sympify(
            result["derivative_value"]
        )

        tangent_line = sp.sympify(
            result["tangent_line"]
        )

        # ----------------------------------------------------
        # Valeur de la fonction
        # ----------------------------------------------------

        steps.append({
            "title": "Calculer la valeur de la fonction",
            "formula": (
                rf"f({sp.latex(point)}) = "
                rf"{sp.latex(function_value)}"
            ),
            "explanation": (
                "On remplace x par la valeur du point "
                "pour déterminer les coordonnées du point "
                "de la courbe."
            ),
        })

        # ----------------------------------------------------
        # Valeur de la dérivée
        # ----------------------------------------------------

        steps.append({
            "title": "Calculer la valeur de la dérivée",
            "formula": (
                rf"f'({sp.latex(point)}) = "
                rf"{sp.latex(derivative_value)}"
            ),
            "explanation": (
                "La valeur de la dérivée au point donne "
                "la pente de la tangente à la courbe."
            ),
        })

        # ----------------------------------------------------
        # Pente de la tangente
        # ----------------------------------------------------

        steps.append({
            "title": "Déterminer la pente de la tangente",
            "formula": (
                rf"m = f'({sp.latex(point)}) = "
                rf"{sp.latex(derivative_value)}"
            ),
            "explanation": (
                "La pente de la tangente est égale à "
                "la valeur de la dérivée au point considéré."
            ),
        })

        # ----------------------------------------------------
        # Equation de la tangente
        # ----------------------------------------------------

        steps.append({
            "title": "Calculer l'équation de la tangente",
            "formula": (
                rf"y = {sp.latex(tangent_line)}"
            ),
            "explanation": (
                "On utilise la formule de la tangente "
                "y = f(a) + f'(a)(x - a) "
                "pour obtenir son équation."
            ),
        })

        # ----------------------------------------------------
        # Conclusion
        # ----------------------------------------------------

        steps.append({
            "title": "Conclusion",
            "formula": (
                rf"\boxed{{f'({sp.latex(point)}) = "
                rf"{sp.latex(derivative_value)}}}"
            ),
            "explanation": (
                "La dérivée au point étudié donne la pente "
                "de la tangente à la courbe."
            ),
        })

    return steps