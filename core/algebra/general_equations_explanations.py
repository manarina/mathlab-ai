from __future__ import annotations

import sympy as sp


def explain_general_polynomial_equation(
    result: dict,
) -> list[dict]:
    """
    Génère une explication pédagogique de l'analyse
    d'une équation polynomiale de degré quelconque.

    Le résultat doit provenir de
    solve_polynomial_equation().
    """

    polynomial = result["polynomial"]
    degree = result["degree"]
    method = result["method"]

    variable = polynomial.gens[0]
    expression = polynomial.as_expr()
    coefficients = polynomial.all_coeffs()

    steps: list[dict] = []

    # ========================================================
    # 1. IDENTIFIER L'ÉQUATION
    # ========================================================

    steps.append(
        {
            "title": "Identifier l'équation",
            "formula": (
                rf"{sp.latex(expression)} = 0"
            ),
            "explanation": (
                "On commence par identifier le polynôme "
                "qui constitue l'équation."
            ),
        }
    )

    # ========================================================
    # 2. DÉTERMINER LE DEGRÉ
    # ========================================================

    steps.append(
        {
            "title": "Déterminer le degré",
            "formula": (
                rf"\deg(P) = {degree}"
            ),
            "explanation": (
                "Le degré d'un polynôme correspond "
                "au plus grand exposant de la variable."
            ),
        }
    )

    # ========================================================
    # 3. IDENTIFIER LES COEFFICIENTS
    # ========================================================

    coefficient_text = r",\quad ".join(
        [
            rf"a_{{{degree - index}}} = "
            rf"{sp.latex(coefficient)}"
            for index, coefficient
            in enumerate(coefficients)
        ]
    )

    steps.append(
        {
            "title": "Identifier les coefficients",
            "formula": coefficient_text,
            "explanation": (
                "Les coefficients sont les nombres "
                "qui accompagnent les différentes "
                "puissances de la variable."
            ),
        }
    )

    # ========================================================
    # 4. CHOISIR LA MÉTHODE
    # ========================================================

    method_labels = {
        "linear": (
            "Méthode du premier degré"
        ),
        "quadratic": (
            "Méthode du second degré"
        ),
        "horner_cardano": (
            "Méthodes de Horner et Cardano"
        ),
        "quartic_symbolic": (
            "Résolution symbolique du quatrième degré"
        ),
        "factorization_and_numerical": (
            "Factorisation et méthodes numériques"
        ),
    }

    method_label = method_labels.get(
        method,
        method,
    )

    steps.append(
        {
            "title": "Choisir la méthode de résolution",
            "formula": (
                rf"\text{{Méthode : }} "
                rf"\text{{{method_label}}}"
            ),
            "explanation": (
                "La méthode principale est sélectionnée "
                "automatiquement en fonction du degré "
                "du polynôme."
            ),
        }
    )

    # ========================================================
    # 5. FACTORISATION
    # ========================================================

    factorized = result["factorized"]

    steps.append(
        {
            "title": "Factoriser le polynôme",
            "formula": (
                rf"P({variable}) = "
                rf"{sp.latex(factorized)}"
            ),
            "explanation": (
                "La factorisation permet de rechercher "
                "une forme produit du polynôme et peut "
                "faciliter l'identification des racines."
            ),
        }
    )

    # ========================================================
    # 6. RACINES EXACTES
    # ========================================================

    exact_roots = result.get(
        "exact_roots",
        [],
    )

    steps.append(
        {
            "title": "Rechercher les racines exactes",
            "formula": (
                rf"P({variable}) = 0"
            ),
            "explanation": (
                "On recherche les valeurs exactes de la "
                "variable qui annulent le polynôme."
            ),
        }
    )

    if exact_roots:

        roots_text = r",\quad ".join(
            [
                rf"x_{{{index}}} = "
                rf"{sp.latex(root)}"
                for index, root
                in enumerate(
                    exact_roots,
                    start=1,
                )
            ]
        )

        steps.append(
            {
                "title": "Solutions exactes",
                "formula": roots_text,
                "explanation": (
                    "Les valeurs obtenues sont les "
                    "racines exactes du polynôme."
                ),
            }
        )

    else:

        steps.append(
            {
                "title": "Solutions exactes",
                "formula": r"S = \varnothing",
                "explanation": (
                    "Aucune solution exacte n'a été "
                    "obtenue sous forme symbolique."
                ),
            }
        )

    # ========================================================
    # 7. RACINES NUMÉRIQUES
    # ========================================================

    numerical_roots = result.get(
        "numerical_roots",
        [],
    )

    if numerical_roots:

        numerical_text = r",\quad ".join(
            [
                rf"x_{{{index}}} \approx "
                rf"{sp.latex(root)}"
                for index, root
                in enumerate(
                    numerical_roots,
                    start=1,
                )
            ]
        )

        steps.append(
            {
                "title": "Calculer les solutions numériques",
                "formula": numerical_text,
                "explanation": (
                    "Pour les polynômes de degré élevé, "
                    "une méthode numérique permet "
                    "d'obtenir des approximations "
                    "des racines."
                ),
            }
        )

    # ========================================================
    # 8. CONCLUSION
    # ========================================================

    if numerical_roots:
        conclusion = (
            "Les solutions numériques fournissent "
            "une approximation des racines du polynôme."
        )
    elif exact_roots:
        conclusion = (
            "Les solutions exactes obtenues constituent "
            "les racines du polynôme."
        )
    else:
        conclusion = (
            "L'analyse ne fournit pas de solution "
            "symbolique exploitable."
        )

    steps.append(
        {
            "title": "Conclusion",
            "formula": (
                rf"S = "
                rf"\{{"
                rf"{r",\quad ".join(
                    sp.latex(root)
                    for root in (
                        numerical_roots
                        if numerical_roots
                        else exact_roots
                    )
                )}"
                rf"\}}"
                if (
                    numerical_roots
                    or exact_roots
                )
                else r"S = \varnothing"
            ),
            "explanation": conclusion,
        }
    )

    return steps