from __future__ import annotations


def explain_linear_equation(
    a: float,
    b: float,
    result: dict,
) -> list[dict]:
    """
    Génère la démarche pédagogique pour une équation :

        ax + b = 0
    """

    steps = []

    # --------------------------------------------------------
    # ÉTAPE 1 — FORME GÉNÉRALE
    # --------------------------------------------------------

    steps.append(
        {
            "title": "Identifier la forme de l'équation",
            "formula": r"ax + b = 0",
            "explanation": (
                "Une équation du premier degré peut être écrite "
                "sous la forme ax + b = 0."
            ),
        }
    )

    # --------------------------------------------------------
    # CAS PARTICULIER : a = 0
    # --------------------------------------------------------

    if result["type"] == "infinite":

        steps.append(
            {
                "title": "Remplacer les coefficients",
                "formula": rf"{a:g}x + ({b:g}) = 0",
                "explanation": (
                    "Comme a = 0 et b = 0, l'équation devient "
                    "0x + 0 = 0."
                ),
            }
        )

        steps.append(
            {
                "title": "Analyser l'équation",
                "formula": r"0 = 0",
                "explanation": (
                    "Cette égalité est vraie pour tout nombre réel."
                ),
            }
        )

        return steps

    if result["type"] == "none":

        steps.append(
            {
                "title": "Remplacer les coefficients",
                "formula": rf"{a:g}x + ({b:g}) = 0",
                "explanation": (
                    "Comme a = 0, il n'y a plus de terme contenant x."
                ),
            }
        )

        steps.append(
            {
                "title": "Analyser l'équation",
                "formula": rf"{b:g} = 0",
                "explanation": (
                    "Cette égalité est fausse. "
                    "L'équation ne possède donc aucune solution."
                ),
            }
        )

        return steps

    # --------------------------------------------------------
    # SOLUTION UNIQUE
    # --------------------------------------------------------

    steps.append(
        {
            "title": "Remplacer les coefficients",
            "formula": rf"{a:g}x + ({b:g}) = 0",
            "explanation": (
                "On remplace a et b par leurs valeurs."
            ),
        }
    )

    steps.append(
        {
            "title": "Isoler le terme contenant x",
            "formula": rf"{a:g}x = {-b:g}",
            "explanation": (
                "On soustrait b des deux côtés de l'équation."
            ),
        }
    )

    steps.append(
        {
            "title": "Diviser par le coefficient de x",
            "formula": rf"x = \frac{{{-b:g}}}{{{a:g}}}",
            "explanation": (
                "On divise les deux membres par a afin d'isoler x."
            ),
        }
    )

    solution = result["solutions"][0]

    steps.append(
        {
            "title": "Solution",
            "formula": rf"x = {solution:g}",
            "explanation": (
                "La valeur obtenue est l'unique solution réelle "
                "de l'équation."
            ),
        }
    )

    return steps


def explain_quadratic_equation(
    a: float,
    b: float,
    c: float,
    result: dict,
) -> list[dict]:
    """
    Génère la démarche pédagogique pour une équation :

        ax² + bx + c = 0
    """

    steps = []

    # --------------------------------------------------------
    # ÉTAPE 1 — FORME GÉNÉRALE
    # --------------------------------------------------------

    steps.append(
        {
            "title": "Identifier la forme de l'équation",
            "formula": r"ax^2 + bx + c = 0",
            "explanation": (
                "Une équation du second degré possède la forme "
                "ax² + bx + c = 0 avec a différent de 0."
            ),
        }
    )

    # --------------------------------------------------------
    # CAS a = 0
    # --------------------------------------------------------

    if result["degree"] == 1:

        steps.append(
            {
                "title": "Vérifier le coefficient a",
                "formula": r"a = 0",
                "explanation": (
                    "Lorsque a = 0, le terme ax² disparaît. "
                    "L'équation devient donc une équation du premier degré."
                ),
            }
        )

        steps.append(
            {
                "title": "Nouvelle forme",
                "formula": rf"{b:g}x + ({c:g}) = 0",
                "explanation": (
                    "On utilise alors la méthode de résolution "
                    "d'une équation du premier degré."
                ),
            }
        )

        return steps

    # --------------------------------------------------------
    # ÉTAPE 2 — IDENTIFICATION
    # --------------------------------------------------------

    steps.append(
        {
            "title": "Identifier les coefficients",
            "formula": (
                rf"a = {a:g},\quad b = {b:g},\quad c = {c:g}"
            ),
            "explanation": (
                "Les coefficients a, b et c sont identifiés "
                "à partir de l'équation."
            ),
        }
    )

    # --------------------------------------------------------
    # ÉTAPE 3 — DISCRIMINANT
    # --------------------------------------------------------

    discriminant = result["discriminant"]

    steps.append(
        {
            "title": "Calculer le discriminant",
            "formula": r"\Delta = b^2 - 4ac",
            "explanation": (
                "Le discriminant permet de déterminer "
                "le nombre de solutions réelles."
            ),
        }
    )

    steps.append(
        {
            "title": "Remplacer les coefficients",
            "formula": (
                rf"\Delta = ({b:g})^2 - 4({a:g})({c:g})"
            ),
            "explanation": (
                "On remplace a, b et c par leurs valeurs."
            ),
        }
    )

    steps.append(
        {
            "title": "Calculer Δ",
            "formula": rf"\Delta = {discriminant:g}",
            "explanation": (
                "Le discriminant obtenu permet de déterminer "
                "la nature des solutions."
            ),
        }
    )

    # --------------------------------------------------------
    # Δ > 0
    # --------------------------------------------------------

    if result["type"] == "two_real_solutions":

        steps.append(
            {
                "title": "Interpréter le discriminant",
                "formula": r"\Delta > 0",
                "explanation": (
                    "Lorsque Δ est positif, l'équation possède "
                    "deux solutions réelles distinctes."
                ),
            }
        )

        steps.append(
            {
                "title": "Utiliser les formules",
                "formula": (
                    r"x_1 = \frac{-b-\sqrt{\Delta}}{2a}"
                    r"\qquad"
                    r"x_2 = \frac{-b+\sqrt{\Delta}}{2a}"
                ),
                "explanation": (
                    "On applique les deux formules de résolution."
                ),
            }
        )

        x1, x2 = result["solutions"]

        steps.append(
    {
        "title": "Calculer les solutions",
        "formula": (
            rf"x_1 = {x1:g}"
            rf"\qquad "
            rf"x_2 = {x2:g}"
        ),
        "explanation": (
            "Les deux valeurs obtenues sont les solutions "
            "réelles de l'équation."
        ),
    }
    )

       

    # --------------------------------------------------------
    # Δ = 0
    # --------------------------------------------------------

    elif result["type"] == "one_real_solution":

        steps.append(
            {
                "title": "Interpréter le discriminant",
                "formula": r"\Delta = 0",
                "explanation": (
                    "Lorsque Δ est nul, l'équation possède "
                    "une solution réelle double."
                ),
            }
        )

        steps.append(
            {
                "title": "Utiliser la formule",
                "formula": r"x = \frac{-b}{2a}",
                "explanation": (
                    "Lorsque Δ = 0, une seule valeur de x est obtenue."
                ),
            }
        )

        x = result["solutions"][0]

        steps.append(
            {
                "title": "Calculer la solution",
                "formula": rf"x = {x:g}",
                "explanation": (
                    "Cette valeur est la solution réelle double."
                ),
            }
        )

    # --------------------------------------------------------
    # Δ < 0
    # --------------------------------------------------------

    else:

        steps.append(
            {
                "title": "Interpréter le discriminant",
                "formula": r"\Delta < 0",
                "explanation": (
                    "Lorsque Δ est négatif, l'équation ne possède "
                    "aucune solution dans les nombres réels."
                ),
            }
        )

        steps.append(
            {
                "title": "Conclusion",
                "formula": r"S = \varnothing",
                "explanation": (
                    "L'ensemble des solutions réelles est vide."
                ),
            }
        )

    return steps