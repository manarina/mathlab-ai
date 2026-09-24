
from __future__ import annotations

import sympy as sp


def explain_cubic_equation(
    result: dict,
) -> list[dict]:
    """
    Génère une explication pédagogique complète
    de la résolution d'une équation cubique.

    Le résultat fourni doit provenir de
    solve_cubic_equation().
    """

    polynomial = result["polynomial"]
    variable = polynomial.gens[0]
    coefficients = polynomial.all_coeffs()

    a, b, c, d = coefficients

    steps: list[dict] = []

    # ========================================================
    # IDENTIFICATION
    # ========================================================

    steps.append(
        {
            "title": "Identifier l'équation",
            "formula": (
                rf"{sp.latex(polynomial.as_expr())} = 0"
            ),
            "explanation": (
                "L'équation est un polynôme de degré 3 "
                "car le plus grand exposant de la variable "
                "est 3."
            ),
        }
    )

    # ========================================================
    # COEFFICIENTS
    # ========================================================

    steps.append(
        {
            "title": "Identifier les coefficients",
            "formula": (
                rf"a={sp.latex(a)},\quad "
                rf"b={sp.latex(b)},\quad "
                rf"c={sp.latex(c)},\quad "
                rf"d={sp.latex(d)}"
            ),
            "explanation": (
                "On identifie les quatre coefficients "
                "de l'équation cubique."
            ),
        }
    )

    # ========================================================
    # DISCRIMINANT CUBIQUE
    # ========================================================

    discriminant = result["discriminant"]

    steps.append(
        {
            "title": "Calculer le discriminant cubique",
            "formula": (
                rf"\Delta_{{cubique}} = "
                rf"{sp.latex(discriminant)}"
            ),
            "explanation": (
                "Le discriminant cubique permet de déterminer "
                "la nature des racines du polynôme."
            ),
        }
    )

    # ========================================================
    # NATURE DES RACINES
    # ========================================================

    steps.append(
        {
            "title": "Déterminer la nature des racines",
            "formula": result["type"],
            "explanation": (
                "On interprète le signe du discriminant "
                "cubique afin de déterminer la nature "
                "des racines."
            ),
        }
    )

    # ========================================================
    # HORNER
    # ========================================================

    horner = result.get(
        "horner",
        {},
    )

    if horner.get("available", True):

        candidates = horner.get(
            "tested_candidates",
            [],
        )

        # ----------------------------------------------------
        # CANDIDATS
        # ----------------------------------------------------

        if candidates:

            candidate_text = ", ".join(
                str(item["candidate"])
                for item in candidates
            )

            steps.append(
                {
                    "title": (
                        "Rechercher une racine rationnelle"
                    ),
                    "formula": (
                        rf"r \in "
                        rf"\{{{candidate_text}\}}"
                    ),
                    "explanation": (
                        "On recherche une racine rationnelle "
                        "parmi les candidats possibles."
                    ),
                }
            )

        # ----------------------------------------------------
        # RACINES TROUVÉES
        # ----------------------------------------------------

        rational_roots = horner.get(
            "rational_roots",
            [],
        )

        if rational_roots:

            roots_text = ",\quad ".join(
                sp.latex(root)
                for root in rational_roots
            )

            steps.append(
                {
                    "title": (
                        "Racine(s) trouvée(s) avec Horner"
                    ),
                    "formula": (
                        rf"x = {roots_text}"
                    ),
                    "explanation": (
                        "Une ou plusieurs racines rationnelles "
                        "ont été trouvées grâce à la méthode "
                        "de Horner."
                    ),
                }
            )

            # ------------------------------------------------
            # RÉDUCTION DU DEGRÉ
            # ------------------------------------------------

            remaining = horner.get(
                "remaining_polynomial"
            )

            if remaining is not None:

                steps.append(
                    {
                        "title": "Réduire le degré",
                        "formula": (
                            rf"Q({variable}) = "
                            rf"{sp.latex(remaining.as_expr())}"
                        ),
                        "explanation": (
                            "La division de Horner permet de "
                            "réduire le polynôme cubique à un "
                            "polynôme de degré inférieur."
                        ),
                    }
                )

    # ========================================================
    # FACTORISATION
    # ========================================================

    factorized = result["factorized"]

    steps.append(
        {
            "title": "Factoriser",
            "formula": (
                rf"P({variable}) = "
                rf"{sp.latex(factorized)}"
            ),
            "explanation": (
                "La factorisation permet d'écrire "
                "l'équation sous forme de produit "
                "de facteurs."
            ),
        }
    )

    # ========================================================
    # CARDANO — NORMALISATION
    # ========================================================

    cardano = result["cardano"]

    steps.append(
        {
            "title": "Préparer la méthode de Cardano",
            "formula": (
                r"x^3 + Ax^2 + Bx + C = 0"
            ),
            "explanation": (
                "On normalise l'équation afin de pouvoir "
                "appliquer la transformation de Cardano."
            ),
        }
    )

    # ========================================================
    # CARDANO — p ET q
    # ========================================================

    steps.append(
        {
            "title": "Calculer p et q",
            "formula": (
                rf"p = {sp.latex(cardano['p'])}"
                rf"\qquad "
                rf"q = {sp.latex(cardano['q'])}"
            ),
            "explanation": (
                "Après le changement de variable "
                "x = y - A/3, on obtient une cubique "
                "déprimée y³ + py + q = 0."
            ),
        }
    )

    # ========================================================
    # CARDANO — DISCRIMINANT
    # ========================================================

    steps.append(
        {
            "title": (
                "Calculer le discriminant de Cardano"
            ),
            "formula": (
                rf"\Delta_C = "
                rf"{sp.latex(cardano['delta'])}"
            ),
            "explanation": (
                "Le discriminant de Cardano permet "
                "d'étudier les différentes situations "
                "de la résolution cubique."
            ),
        }
    )

    # ========================================================
    # SOLUTIONS
    # ========================================================

    roots = result["roots"]

    roots_text = r",\quad ".join(
        [
            rf"x_{{{index}}} = "
            rf"{sp.latex(root)}"
            for index, root in enumerate(
                roots,
                start=1,
            )
        ]
    )

    steps.append(
        {
            "title": "Solutions",
            "formula": roots_text,
            "explanation": (
                "Les racines obtenues constituent "
                "les solutions de l'équation."
            ),
        }
    )

    return steps

