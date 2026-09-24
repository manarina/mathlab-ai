from __future__ import annotations

import sympy as sp


# ============================================================
# UTILITAIRES
# ============================================================


def _format_expression(expression: sp.Expr) -> str:
    """
    Convertit une expression SymPy en chaîne LaTeX.
    """

    return sp.latex(expression)


# ============================================================
# ANALYSE PEDAGOGIQUE DU POLYNOME
# ============================================================


def explain_polynomial_analysis(
    polynomial: sp.Poly,
    roots: list[sp.Expr],
    factorized: sp.Expr,
    derivative: sp.Poly,
) -> list[dict]:
    """
    Génère les étapes pédagogiques d'analyse d'un polynôme.

    L'analyse couvre :

        - identification du polynôme
        - degré
        - coefficients
        - racines
        - factorisation
        - dérivée

    Le calcul est effectué par polynomials.py.
    Cette fonction se limite à construire la démarche
    pédagogique.
    """

    steps: list[dict] = []

    expression = polynomial.as_expr()
    variable = polynomial.gens[0]
    degree = polynomial.degree()
    coefficients = polynomial.all_coeffs()

    # ========================================================
    # ETAPE 1 — IDENTIFIER LE POLYNOME
    # ========================================================

    steps.append(
        {
            "title": "Identifier le polynôme",
            "formula": (
                rf"P({variable}) = "
                rf"{_format_expression(expression)}"
            ),
            "explanation": (
                "On commence par identifier l'expression "
                "du polynôme que l'on souhaite analyser."
            ),
        }
    )

    # ========================================================
    # ETAPE 2 — DETERMINER LE DEGRE
    # ========================================================

    steps.append(
        {
            "title": "Déterminer le degré",
            "formula": (
                rf"\deg(P) = {degree}"
            ),
            "explanation": (
                "Le degré d'un polynôme correspond au plus "
                "grand exposant de la variable."
            ),
        }
    )

    # ========================================================
    # ETAPE 3 — IDENTIFIER LES COEFFICIENTS
    # ========================================================

    coefficient_text = ",\quad ".join(
        [
            rf"a_{{{degree - index}}} = "
            rf"{_format_expression(coefficient)}"
            for index, coefficient in enumerate(coefficients)
        ]
    )

    steps.append(
        {
            "title": "Identifier les coefficients",
            "formula": coefficient_text,
            "explanation": (
                "Les coefficients sont les nombres associés "
                "aux différentes puissances de la variable."
            ),
        }
    )

    # ========================================================
    # ETAPE 4 — RECHERCHER LES RACINES
    # ========================================================

    steps.append(
        {
            "title": "Rechercher les racines",
            "formula": (
                rf"P({variable}) = 0"
            ),
            "explanation": (
                "Les racines sont les valeurs de la variable "
                "pour lesquelles le polynôme est égal à zéro."
            ),
        }
    )

    # ========================================================
    # ETAPE 5 — RESULTAT DES RACINES
    # ========================================================

    if roots:

        roots_formula = ",\quad ".join(
            [
                rf"x_{{{index}}} = "
                rf"{_format_expression(root)}"
                for index, root in enumerate(roots, start=1)
            ]
        )

        roots_explanation = (
            "Les valeurs obtenues sont les racines "
            "du polynôme."
        )

    else:

        roots_formula = r"S = \varnothing"

        roots_explanation = (
            "Aucune racine n'a été trouvée dans "
            "le domaine considéré."
        )

    steps.append(
        {
            "title": "Déterminer les racines",
            "formula": roots_formula,
            "explanation": roots_explanation,
        }
    )

    # ========================================================
    # ETAPE 6 — FACTORISATION
    # ========================================================

    steps.append(
        {
            "title": "Factoriser le polynôme",
            "formula": (
                rf"P({variable}) = "
                rf"{_format_expression(factorized)}"
            ),
            "explanation": (
                "La factorisation permet d'écrire le polynôme "
                "comme un produit de facteurs lorsque cela "
                "est possible."
            ),
        }
    )

    # ========================================================
    # ETAPE 7 — CALCULER LA DERIVEE
    # ========================================================

    derivative_expression = derivative.as_expr()

    steps.append(
        {
            "title": "Calculer la dérivée",
            "formula": (
                rf"P'({variable}) = "
                rf"{_format_expression(derivative_expression)}"
            ),
            "explanation": (
                "La dérivée décrit notamment la variation "
                "du polynôme."
            ),
        }
    )

    return steps