from __future__ import annotations

from typing import Any

import sympy as sp


def explain_limit(
    function: sp.Expr,
    point: Any,
    result: dict,
) -> list[dict]:
    """
    Génère une explication pédagogique de l'analyse
    d'une limite.
    """

    x = sp.Symbol("x")
    point = sp.sympify(point)

    steps: list[dict] = []

    # ========================================================
    # ÉTAPE 1 — Identifier la fonction
    # ========================================================

    steps.append({
        "title": "Identifier la fonction",
        "formula": (
            rf"f(x) = "
            rf"{sp.latex(function)}"
        ),
        "explanation": (
            "On identifie la fonction dont on cherche "
            "la limite."
        ),
    })

    # ========================================================
    # ÉTAPE 2 — Identifier le point
    # ========================================================

    steps.append({
        "title": "Identifier le point",
        "formula": (
            rf"x \to {sp.latex(point)}"
        ),
        "explanation": (
            "On détermine la valeur vers laquelle "
            "x tend."
        ),
    })

    # ========================================================
    # ÉTAPE 3 — Limite à gauche
    # ========================================================

    left_limit = result["left_limit"]

    steps.append({
        "title": "Calculer la limite à gauche",
        "formula": (
            rf"\lim_{{x \to "
            rf"{sp.latex(point)}^-}} "
            rf"f(x) = "
            rf"{sp.latex(left_limit)}"
        ),
        "explanation": (
            "On étudie le comportement de la fonction "
            "lorsque x approche le point par les valeurs "
            "inférieures."
        ),
    })

    # ========================================================
    # ÉTAPE 4 — Limite à droite
    # ========================================================

    right_limit = result["right_limit"]

    steps.append({
        "title": "Calculer la limite à droite",
        "formula": (
            rf"\lim_{{x \to "
            rf"{sp.latex(point)}^+}} "
            rf"f(x) = "
            rf"{sp.latex(right_limit)}"
        ),
        "explanation": (
            "On étudie le comportement de la fonction "
            "lorsque x approche le point par les valeurs "
            "supérieures."
        ),
    })

    # ========================================================
    # ÉTAPE 5 — Comparer les limites
    # ========================================================

    if result["type"] == "does_not_exist":

        steps.append({
            "title": "Comparer les deux limites",
            "formula": (
                rf"\lim_{{x \to {sp.latex(point)}^-}} "
                rf"f(x) \ne "
                rf"\lim_{{x \to {sp.latex(point)}^+}} "
                rf"f(x)"
            ),
            "explanation": (
                "Les limites à gauche et à droite sont "
                "différentes. La limite bilatérale "
                "n'existe donc pas."
            ),
        })

        steps.append({
            "title": "Conclusion",
            "formula": (
                rf"\boxed{{\lim_{{x \to "
                rf"{sp.latex(point)}}} f(x) "
                rf"\text{{ n'existe pas}}}}"
            ),
            "explanation": (
                "Une limite en un point existe seulement "
                "si les limites à gauche et à droite "
                "sont égales."
            ),
        })

    else:

        limit = result["limit"]

        steps.append({
            "title": "Comparer les deux limites",
            "formula": (
                rf"\lim_{{x \to {sp.latex(point)}^-}} "
                rf"f(x)"
                rf" = "
                rf"\lim_{{x \to {sp.latex(point)}^+}} "
                rf"f(x)"
            ),
            "explanation": (
                "Les limites à gauche et à droite sont "
                "égales. La limite bilatérale existe."
            ),
        })

        steps.append({
            "title": "Conclusion",
            "formula": (
                rf"\boxed{{\lim_{{x \to "
                rf"{sp.latex(point)}}} f(x) "
                rf"= {sp.latex(limit)}}}"
            ),
            "explanation": (
                "La limite recherchée est égale à la "
                "valeur commune des limites à gauche "
                "et à droite."
            ),
        })

    return steps