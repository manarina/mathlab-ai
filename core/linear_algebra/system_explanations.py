from __future__ import annotations

from core.linear_algebra.linear_systems import (
    LinearSystem,
    LinearSystemSolution,
)


def _format_number(value: int | float) -> str:
    """Formate proprement un nombre pour l'affichage."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def _format_expression(
    coefficients: tuple[int | float, ...],
    variables: tuple[str, ...],
) -> str:
    """
    Transforme une ligne de coefficients en équation lisible.

    Exemple :
        (2, 1) -> "2x + y"
    """

    terms = []

    for coefficient, variable in zip(coefficients, variables):
        if coefficient == 0:
            continue

        if coefficient == 1:
            term = variable
        elif coefficient == -1:
            term = f"-{variable}"
        else:
            term = f"{_format_number(coefficient)}{variable}"

        if not terms:
            terms.append(term)
        elif coefficient > 0:
            terms.append(f"+ {term}")
        else:
            terms.append(f"- {_format_number(abs(coefficient))}{variable}")

    if not terms:
        return "0"

    return " ".join(terms)


def explain_linear_system(
    system: LinearSystem,
) -> str:
    """
    Explique la structure d'un système linéaire.
    """

    size = system.size

    variables = tuple(
        "x" if index == 0 else f"x_{index + 1}"
        for index in range(size)
    )

    lines = [
        "### Système linéaire",
        "",
        f"Le système contient **{size} équations** "
        f"et **{size} inconnues**.",
        "",
        "Sous forme matricielle :",
        "",
        r"$$AX = B$$",
        "",
        "où :",
        "",
        "- $A$ est la matrice des coefficients ;",
        "- $X$ est le vecteur des inconnues ;",
        "- $B$ est le vecteur des constantes.",
        "",
        "### Système",
        "",
    ]

    for row in range(size):
        coefficients = system.coefficients.values[row]
        constant = system.constants.values[row]

        expression = _format_expression(
            coefficients,
            variables,
        )

        lines.append(
            f"$${expression} = {_format_number(constant)}$$"
        )

    return "\n".join(lines)


def explain_gaussian_elimination(
    system: LinearSystem,
) -> str:
    """
    Explique le principe de l'élimination de Gauss.
    """

    size = system.size

    lines = [
        "### Méthode : élimination de Gauss",
        "",
        "Pour résoudre le système, on utilise "
        "**l'élimination de Gauss**.",
        "",
        "La première étape consiste à construire "
        "la matrice augmentée :",
        "",
        r"$$[A \mid B]$$",
        "",
        "On effectue ensuite des opérations élémentaires "
        "sur les lignes afin de transformer progressivement "
        "la matrice.",
        "",
        "Les opérations autorisées sont :",
        "",
        "1. échanger deux lignes ;",
        "2. multiplier une ligne par un nombre non nul ;",
        "3. ajouter à une ligne un multiple d'une autre ligne.",
        "",
        f"Le système possède **{size} équations** "
        f"et **{size} inconnues**.",
        "",
    ]

    return "\n".join(lines)


def explain_solution(
    solution: LinearSystemSolution,
) -> str:
    """
    Explique le résultat obtenu après résolution.
    """

    lines = [
        "### Résultat",
        "",
    ]

    if solution.is_unique:
        lines.extend(
            [
                "Le système possède une **solution unique**.",
                "",
            ]
        )

        if solution.values is not None:
            for index, value in enumerate(solution.values):
                variable = (
                    "x"
                    if index == 0
                    else f"x_{index + 1}"
                )

                lines.append(
                    f"$$ {variable} = {_format_number(value)} $$"
                )

    elif solution.has_no_solution:
        lines.extend(
            [
                "Le système ne possède **aucune solution**.",
                "",
                "Après élimination, on obtient une contradiction "
                "du type :",
                "",
                r"$$0 = c \qquad (c \neq 0)$$",
                "",
                "Le système est donc **incompatible**.",
            ]
        )

    elif solution.has_infinite_solutions:
        lines.extend(
            [
                "Le système possède **une infinité de solutions**.",
                "",
                "Après élimination, le nombre de pivots est "
                "inférieur au nombre d'inconnues.",
                "",
                "Au moins une inconnue est donc libre.",
            ]
        )

    return "\n".join(lines)


def explain_complete_solution(
    system: LinearSystem,
    solution: LinearSystemSolution,
) -> str:
    """
    Produit une explication complète du système
    et de sa résolution.
    """

    sections = [
        explain_linear_system(system),
        explain_gaussian_elimination(system),
        explain_solution(solution),
    ]

    return "\n\n".join(sections)