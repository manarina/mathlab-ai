from __future__ import annotations

from typing import Sequence

import sympy as sp


def validate_system(
    matrix: Sequence[Sequence[sp.Expr]],
    constants: Sequence[sp.Expr],
) -> None:
    """
    Vérifie que la matrice des coefficients et le vecteur
    des constantes sont compatibles.

    Accepte :
        - listes Python
        - tuples
        - matrices SymPy
    """

    if matrix is None:
        raise ValueError(
            "La matrice du système ne peut pas être vide."
        )

    if constants is None:
        raise ValueError(
            "Le vecteur des constantes ne peut pas être vide."
        )

    # ========================================================
    # MATRICE DES COEFFICIENTS
    # ========================================================

    if isinstance(matrix, sp.MatrixBase):

        number_of_rows = matrix.rows
        number_of_columns = matrix.cols

        if number_of_rows == 0:
            raise ValueError(
                "La matrice du système ne peut pas être vide."
            )

        if number_of_columns == 0:
            raise ValueError(
                "La matrice doit contenir au moins une colonne."
            )

    else:

        if len(matrix) == 0:
            raise ValueError(
                "La matrice du système ne peut pas être vide."
            )

        number_of_rows = len(matrix)

        number_of_columns = len(
            matrix[0]
        )

        if number_of_columns == 0:
            raise ValueError(
                "La matrice doit contenir au moins une colonne."
            )

        for row in matrix:

            if len(row) != number_of_columns:
                raise ValueError(
                    "Toutes les lignes de la matrice "
                    "doivent avoir la même longueur."
                )

    # ========================================================
    # VECTEUR DES CONSTANTES
    # ========================================================

    if isinstance(constants, sp.MatrixBase):

        # Le vecteur doit être une matrice colonne.
        if constants.cols != 1:
            raise ValueError(
                "Le vecteur des constantes doit être "
                "une matrice colonne."
            )

        number_of_constants = constants.rows

    else:

        number_of_constants = len(
            constants
        )

    # ========================================================
    # COMPATIBILITÉ DES DIMENSIONS
    # ========================================================

    if number_of_rows != number_of_constants:
        raise ValueError(
            "Le nombre de lignes de la matrice "
            "doit correspondre au nombre de constantes."
        )


def build_matrix(
    matrix: Sequence[Sequence[sp.Expr]],
) -> sp.Matrix:
    """
    Transforme une matrice Python ou SymPy
    en matrice SymPy.
    """

    if isinstance(matrix, sp.MatrixBase):

        if matrix.rows == 0 or matrix.cols == 0:
            raise ValueError(
                "La matrice doit contenir au moins "
                "une ligne et une colonne."
            )

        return sp.Matrix(matrix)

    if not matrix:
        raise ValueError(
            "La matrice doit contenir au moins une ligne."
        )

    number_of_columns = len(
        matrix[0]
    )

    if number_of_columns == 0:
        raise ValueError(
            "La matrice doit contenir au moins une colonne."
        )

    for row in matrix:

        if len(row) != number_of_columns:
            raise ValueError(
                "Toutes les lignes de la matrice "
                "doivent avoir la même longueur."
            )

    return sp.Matrix(
        [
            [
                sp.sympify(value)
                for value in row
            ]
            for row in matrix
        ]
    )


def build_constants_vector(
    constants: Sequence[sp.Expr],
) -> sp.Matrix:
    """
    Transforme les constantes en vecteur colonne SymPy.
    """

    if constants is None:
        raise ValueError(
            "Le vecteur des constantes "
            "ne peut pas être vide."
        )

    if isinstance(constants, sp.MatrixBase):

        if constants.rows == 0:
            raise ValueError(
                "Le vecteur des constantes "
                "ne peut pas être vide."
            )

        if constants.cols != 1:
            raise ValueError(
                "Le vecteur des constantes doit être "
                "une matrice colonne."
            )

        return sp.Matrix(constants)

    if len(constants) == 0:
        raise ValueError(
            "Le vecteur des constantes "
            "ne peut pas être vide."
        )

    return sp.Matrix(
        [
            sp.sympify(value)
            for value in constants
        ]
    )


def calculate_determinant(
    matrix: sp.Matrix,
) -> sp.Expr:
    """
    Calcule le déterminant d'une matrice carrée.
    """

    if matrix.rows != matrix.cols:
        raise ValueError(
            "Le déterminant nécessite une matrice carrée."
        )

    return sp.simplify(
        matrix.det()
    )


def calculate_rank(
    matrix: sp.Matrix,
) -> int:
    """
    Calcule le rang d'une matrice.
    """

    return matrix.rank()


def classify_system(
    matrix: sp.Matrix,
    constants: sp.Matrix,
) -> dict:
    """
    Classifie un système linéaire selon les rangs :

        rang(A) = rang(A|b) = nombre d'inconnues
            -> solution unique

        rang(A) = rang(A|b) < nombre d'inconnues
            -> infinité de solutions

        rang(A) != rang(A|b)
            -> aucune solution
    """

    if matrix.rows != constants.rows:
        raise ValueError(
            "Dimensions incompatibles entre "
            "la matrice et le vecteur."
        )

    if constants.cols != 1:
        raise ValueError(
            "Le vecteur des constantes doit être "
            "une matrice colonne."
        )

    augmented_matrix = matrix.row_join(
        constants
    )

    rank_matrix = matrix.rank()

    rank_augmented = augmented_matrix.rank()

    number_of_unknowns = matrix.cols

    if rank_matrix != rank_augmented:

        return {
            "type": "no_solution",
            "rank": rank_matrix,
            "augmented_rank": rank_augmented,
            "number_of_unknowns": number_of_unknowns,
            "message": (
                "Le système n'a aucune solution."
            ),
        }

    if rank_matrix < number_of_unknowns:

        return {
            "type": "infinite_solutions",
            "rank": rank_matrix,
            "augmented_rank": rank_augmented,
            "number_of_unknowns": number_of_unknowns,
            "message": (
                "Le système possède une infinité "
                "de solutions."
            ),
        }

    return {
        "type": "unique_solution",
        "rank": rank_matrix,
        "augmented_rank": rank_augmented,
        "number_of_unknowns": number_of_unknowns,
        "message": (
            "Le système possède une solution unique."
        ),
    }


def solve_system(
    matrix: Sequence[Sequence[sp.Expr]],
    constants: Sequence[sp.Expr],
) -> dict:
    """
    Résout un système linéaire général.
    """

    validate_system(
        matrix,
        constants,
    )

    coefficient_matrix = build_matrix(
        matrix
    )

    constants_vector = build_constants_vector(
        constants
    )

    classification = classify_system(
        coefficient_matrix,
        constants_vector,
    )

    solution_set = sp.linsolve(
        (
            coefficient_matrix,
            constants_vector,
        )
    )

    solutions = list(
        solution_set
    )

    determinant = None

    if (
        coefficient_matrix.rows
        == coefficient_matrix.cols
    ):
        determinant = calculate_determinant(
            coefficient_matrix
        )

    return {
        "matrix": coefficient_matrix,
        "constants": constants_vector,
        "augmented_matrix": (
            coefficient_matrix.row_join(
                constants_vector
            )
        ),
        "determinant": determinant,
        "rank": classification["rank"],
        "augmented_rank": classification[
            "augmented_rank"
        ],
        "number_of_unknowns": classification[
            "number_of_unknowns"
        ],
        "type": classification["type"],
        "message": classification["message"],
        "solution_set": solution_set,
        "solutions": solutions,
    }


def solve_2x2_system(
    a11,
    a12,
    a21,
    a22,
    b1,
    b2,
) -> dict:
    """
    Résout un système 2×2.
    """

    return solve_system(
        [
            [a11, a12],
            [a21, a22],
        ],
        [
            b1,
            b2,
        ],
    )


def solve_3x3_system(
    matrix,
    constants,
) -> dict:
    """
    Résout un système 3×3.
    """

    if isinstance(
        matrix,
        sp.MatrixBase,
    ):
        if matrix.rows != 3:
            raise ValueError(
                "Un système 3×3 doit avoir 3 lignes."
            )

        if matrix.cols != 3:
            raise ValueError(
                "Un système 3×3 doit avoir 3 colonnes."
            )
    else:

        if len(matrix) != 3:
            raise ValueError(
                "Un système 3×3 doit avoir 3 lignes."
            )

        for row in matrix:

            if len(row) != 3:
                raise ValueError(
                    "Un système 3×3 doit avoir 3 colonnes."
                )

    if isinstance(
        constants,
        sp.MatrixBase,
    ):
        if constants.rows != 3:
            raise ValueError(
                "Un système 3×3 doit avoir "
                "3 constantes."
            )
    else:

        if len(constants) != 3:
            raise ValueError(
                "Un système 3×3 doit avoir "
                "3 constantes."
            )

    return solve_system(
        matrix,
        constants,
    )


def gaussian_elimination(
    matrix,
    constants,
) -> dict:
    """
    Effectue l'élimination de Gauss-Jordan
    et conserve chaque étape de transformation.
    """

    validate_system(
        matrix,
        constants,
    )

    coefficient_matrix = build_matrix(
        matrix
    )

    constants_vector = build_constants_vector(
        constants
    )

    augmented = coefficient_matrix.row_join(
        constants_vector
    )

    steps = [
        augmented.copy()
    ]

    row = 0

    for column in range(
        augmented.cols - 1
    ):

        pivot = None

        for candidate_row in range(
            row,
            augmented.rows,
        ):

            if augmented[
                candidate_row,
                column,
            ] != 0:

                pivot = candidate_row
                break

        if pivot is None:
            continue

        if pivot != row:

            augmented.row_swap(
                pivot,
                row,
            )

            steps.append(
                augmented.copy()
            )

        pivot_value = augmented[
            row,
            column,
        ]

        if pivot_value != 1:

            augmented.row_op(
                row,
                lambda value, _: (
                    value / pivot_value
                ),
            )

            steps.append(
                augmented.copy()
            )

        for current_row in range(
            augmented.rows
        ):

            if current_row == row:
                continue

            factor = augmented[
                current_row,
                column,
            ]

            if factor == 0:
                continue

            augmented.row_op(
                current_row,
                lambda value, index: (
                    value
                    - factor
                    * augmented[
                        row,
                        index,
                    ]
                ),
            )

            steps.append(
                augmented.copy()
            )

        row += 1

        if row >= augmented.rows:
            break

    return {
        "initial_matrix": (
            coefficient_matrix.row_join(
                constants_vector
            )
        ),
        "steps": steps,
        "final_matrix": augmented,
    }