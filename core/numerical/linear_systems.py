from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Number = int | float


@dataclass(frozen=True)
class LinearSystemResult:
    """
    Résultat de la résolution d'un système linéaire.

    Attributes:
        solution: vecteur solution du système.
        matrix: matrice triangulaire obtenue après élimination.
        vector: second membre transformé après élimination.
        swaps: nombre d'échanges de lignes effectués.
    """

    solution: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]
    vector: tuple[float, ...]
    swaps: int


def _validate_number(value: Number, name: str) -> float:
    """Valide et convertit une valeur numérique en float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} doit être un nombre.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} doit être fini.")

    return value


def _validate_matrix(matrix: Sequence[Sequence[Number]]) -> list[list[float]]:
    """Valide une matrice carrée et la convertit en float."""
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise TypeError("La matrice doit être une séquence de séquences.")

    if len(matrix) == 0:
        raise ValueError("La matrice ne peut pas être vide.")

    if not all(
        isinstance(row, Sequence) and not isinstance(row, (str, bytes))
        for row in matrix
    ):
        raise TypeError("Chaque ligne de la matrice doit être une séquence.")

    size = len(matrix)

    if any(len(row) != size for row in matrix):
        raise ValueError("La matrice doit être carrée.")

    validated_matrix: list[list[float]] = []

    for row_index, row in enumerate(matrix):
        validated_row = []

        for column_index, value in enumerate(row):
            validated_row.append(
                _validate_number(
                    value,
                    f"matrix[{row_index}][{column_index}]",
                )
            )

        validated_matrix.append(validated_row)

    return validated_matrix


def _validate_vector(
    vector: Sequence[Number],
    expected_size: int,
) -> list[float]:
    """Valide le second membre et le convertit en float."""
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise TypeError("Le vecteur doit être une séquence.")

    if len(vector) != expected_size:
        raise ValueError(
            "Le vecteur doit avoir la même dimension que la matrice."
        )

    return [
        _validate_number(value, f"vector[{index}]")
        for index, value in enumerate(vector)
    ]


def gaussian_elimination(
    matrix: Sequence[Sequence[Number]],
    vector: Sequence[Number],
    tolerance: Number = 1e-12,
) -> LinearSystemResult:
    """
    Résout un système linéaire AX = B par élimination de Gauss.

    La méthode utilise un pivot partiel afin d'améliorer la stabilité
    numérique.

    Args:
        matrix: matrice carrée A.
        vector: second membre B.
        tolerance: seuil utilisé pour détecter un pivot nul.

    Returns:
        LinearSystemResult contenant la solution et la matrice
        triangulaire obtenue.

    Raises:
        TypeError: si les données ne sont pas numériques ou mal structurées.
        ValueError: si le système est invalide, singulier ou incompatible.
    """
    tolerance_value = _validate_number(tolerance, "La tolérance")

    if tolerance_value <= 0:
        raise ValueError("La tolérance doit être strictement positive.")

    working_matrix = _validate_matrix(matrix)
    size = len(working_matrix)
    working_vector = _validate_vector(vector, size)

    swaps = 0

    # ============================================================
    # ÉLIMINATION DE GAUSS AVEC PIVOT PARTIEL
    # ============================================================

    for pivot_column in range(size):
        # Recherche du meilleur pivot sur la colonne courante.
        pivot_row = max(
            range(pivot_column, size),
            key=lambda row: abs(working_matrix[row][pivot_column]),
        )

        pivot_value = working_matrix[pivot_row][pivot_column]

        # Aucun pivot exploitable.
        if abs(pivot_value) <= tolerance_value:
            # Vérification d'une éventuelle incompatibilité.
            for row in range(pivot_column, size):
                coefficients_zero = all(
                    abs(working_matrix[row][column]) <= tolerance_value
                    for column in range(pivot_column, size)
                )

                if coefficients_zero and abs(working_vector[row]) > tolerance_value:
                    raise ValueError(
                        "Le système est incompatible : aucune solution."
                    )

            raise ValueError(
                "Le système est singulier ou ne possède pas une solution unique."
            )

        # Échange de lignes si nécessaire.
        if pivot_row != pivot_column:
            working_matrix[pivot_column], working_matrix[pivot_row] = (
                working_matrix[pivot_row],
                working_matrix[pivot_column],
            )

            working_vector[pivot_column], working_vector[pivot_row] = (
                working_vector[pivot_row],
                working_vector[pivot_column],
            )

            swaps += 1

        pivot_value = working_matrix[pivot_column][pivot_column]

        # Élimination sous le pivot.
        for row in range(pivot_column + 1, size):
            current_value = working_matrix[row][pivot_column]

            if abs(current_value) <= tolerance_value:
                working_matrix[row][pivot_column] = 0.0
                continue

            factor = current_value / pivot_value

            working_matrix[row][pivot_column] = 0.0

            for column in range(pivot_column + 1, size):
                working_matrix[row][column] -= (
                    factor * working_matrix[pivot_column][column]
                )

            working_vector[row] -= factor * working_vector[pivot_column]

    # ============================================================
    # REMONTÉE
    # ============================================================

    solution = [0.0] * size

    for row in range(size - 1, -1, -1):
        diagonal = working_matrix[row][row]

        if abs(diagonal) <= tolerance_value:
            raise ValueError(
                "Le système ne possède pas une solution unique."
            )

        known_sum = sum(
            working_matrix[row][column] * solution[column]
            for column in range(row + 1, size)
        )

        solution[row] = (working_vector[row] - known_sum) / diagonal

    return LinearSystemResult(
        solution=tuple(solution),
        matrix=tuple(tuple(row) for row in working_matrix),
        vector=tuple(working_vector),
        swaps=swaps,
    )