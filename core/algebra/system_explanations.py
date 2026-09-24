from __future__ import annotations

from typing import Any

import sympy as sp


def _format_equation(
    coefficients: list[Any],
    constant: Any,
    variables: list[str],
) -> str:
    """
    Construit une équation lisible à partir
    des coefficients et de la constante.
    """

    terms: list[str] = []

    for coefficient, variable in zip(
        coefficients,
        variables,
    ):
        coefficient = sp.sympify(coefficient)

        if coefficient == 0:
            continue

        if coefficient == 1:
            term = variable
        elif coefficient == -1:
            term = f"-{variable}"
        else:
            term = f"{coefficient}{variable}"

        if terms and coefficient > 0:
            term = f"+ {term}"

        elif terms and coefficient < 0:
            term = term.replace("-", "- ")

        terms.append(term)

    if not terms:
        left_side = "0"
    else:
        left_side = " ".join(terms)

    return f"{left_side} = {sp.sympify(constant)}"


def _format_matrix(
    matrix: sp.Matrix,
) -> str:
    """
    Formate une matrice SymPy pour l'affichage
    pédagogique.
    """

    return sp.latex(matrix)


def explain_system(
    matrix,
    constants,
    result: dict,
) -> list[dict]:
    """
    Génère les étapes pédagogiques de résolution
    d'un système linéaire.

    Chaque étape possède :

        title
        formula
        explanation
    """

    coefficient_matrix = sp.Matrix(matrix)
    constants_vector = sp.Matrix(constants)

    number_of_equations = coefficient_matrix.rows
    number_of_unknowns = coefficient_matrix.cols

    variables = [
        "x",
        "y",
        "z",
        "t",
        "u",
        "v",
    ][:number_of_unknowns]

    augmented_matrix = coefficient_matrix.row_join(
        constants_vector
    )

    steps: list[dict] = []

    # ========================================================
    # ÉTAPE 1 — IDENTIFIER LE SYSTÈME
    # ========================================================

    equations = []

    for row_index in range(number_of_equations):
        equations.append(
            _format_equation(
                list(
                    coefficient_matrix.row(
                        row_index
                    )
                ),
                constants_vector[row_index],
                variables,
            )
        )

    steps.append(
        {
            "title": "Identifier le système",
            "formula": "\n".join(
                equations
            ),
            "explanation": (
                "On identifie les équations du système "
                "ainsi que les inconnues à déterminer."
            ),
        }
    )

    # ========================================================
    # ÉTAPE 2 — MATRICE AUGMENTÉE
    # ========================================================

    steps.append(
        {
            "title": "Écrire la matrice augmentée",
            "formula": (
                f"$${_format_matrix(augmented_matrix)}$$"
            ),
            "explanation": (
                "On regroupe les coefficients des inconnues "
                "et les constantes dans une matrice augmentée."
            ),
        }
    )

    # ========================================================
    # ÉTAPE 3 — CLASSIFICATION
    # ========================================================

    system_type = result["type"]

    if system_type == "unique_solution":
        classification = (
            "Le système possède une solution unique."
        )

    elif system_type == "infinite_solutions":
        classification = (
            "Le système possède une infinité de solutions."
        )

    else:
        classification = (
            "Le système ne possède aucune solution."
        )

    steps.append(
        {
            "title": "Analyser le système",
            "formula": (
                f"rang(A) = {result['rank']}, "
                f"rang(A|b) = {result['augmented_rank']}"
            ),
            "explanation": classification,
        }
    )

    # ========================================================
    # ÉTAPE 4 — ÉLIMINATION DE GAUSS
    # ========================================================

    if result["type"] == "unique_solution":

        steps.append(
            {
                "title": "Appliquer l'élimination de Gauss",
                "formula": (
                    "Transformation de la matrice "
                    "augmentée vers une forme échelonnée."
                ),
                "explanation": (
                    "On choisit successivement des pivots "
                    "et on élimine les coefficients situés "
                    "sous et au-dessus des pivots."
                ),
            }
        )

        # ====================================================
        # RÉCUPÉRER LES ÉTAPES DE GAUSS
        # ====================================================

        from core.algebra.systems import (
            gaussian_elimination,
        )

        gaussian_result = gaussian_elimination(
            matrix,
            constants,
        )

        for index, step_matrix in enumerate(
            gaussian_result["steps"][1:],
            start=1,
        ):
            steps.append(
                {
                    "title": (
                        f"Opération de Gauss {index}"
                    ),
                    "formula": (
                        f"$${_format_matrix(step_matrix)}$$"
                    ),
                    "explanation": (
                        "On effectue une opération élémentaire "
                        "sur les lignes afin de simplifier "
                        "le système."
                    ),
                }
            )

    # ========================================================
    # ÉTAPE 5 — SOLUTIONS
    # ========================================================

    if result["type"] == "unique_solution":

        solution = result["solutions"][0]

        solution_lines = []

        for variable, value in zip(
            variables,
            solution,
        ):
            solution_lines.append(
                f"{variable} = {sp.simplify(value)}"
            )

        steps.append(
            {
                "title": "Lire les solutions",
                "formula": "\n".join(
                    solution_lines
                ),
                "explanation": (
                    "La forme finale de la matrice permet "
                    "de lire directement les valeurs des inconnues."
                ),
            }
        )

        # ====================================================
        # ÉTAPE 6 — VÉRIFICATION
        # ====================================================

        verification_lines = []

        for row_index in range(
            number_of_equations
        ):
            expression = sum(
                coefficient_matrix[
                    row_index,
                    column_index,
                ]
                * solution[column_index]
                for column_index in range(
                    number_of_unknowns
                )
            )

            verification_lines.append(
                (
                    f"Équation {row_index + 1} : "
                    f"{sp.simplify(expression)} = "
                    f"{constants_vector[row_index]}"
                )
            )

        steps.append(
            {
                "title": "Vérifier les solutions",
                "formula": "\n".join(
                    verification_lines
                ),
                "explanation": (
                    "On remplace les inconnues par les "
                    "solutions obtenues afin de vérifier "
                    "que toutes les équations sont satisfaites."
                ),
            }
        )

    elif result["type"] == "infinite_solutions":

        steps.append(
            {
                "title": "Décrire les solutions",
                "formula": str(
                    result["solution_set"]
                ),
                "explanation": (
                    "Le rang de la matrice est inférieur "
                    "au nombre d'inconnues. Le système possède "
                    "donc des variables libres et une infinité "
                    "de solutions."
                ),
            }
        )

    else:

        steps.append(
            {
                "title": "Conclusion",
                "formula": (
                    "rang(A) ≠ rang(A|b)"
                ),
                "explanation": (
                    "Les deux rangs sont différents. "
                    "Le système est incompatible et "
                    "ne possède aucune solution."
                ),
            }
        )

    return steps