from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from core.linear_algebra.matrices import Matrix
from core.linear_algebra.vectors import Vector


@dataclass(frozen=True)
class LinearSystem:
    """
    Représente un système linéaire AX = B.

    A : matrice des coefficients
    B : vecteur du second membre
    """

    coefficients: Matrix
    constants: Vector

    def __post_init__(self) -> None:
        if self.coefficients.rows != self.coefficients.columns:
            raise ValueError(
                "La matrice des coefficients doit être carrée."
            )

        if self.coefficients.rows != self.constants.dimension:
            raise ValueError(
                "Le nombre d'équations doit être égal "
                "au nombre de constantes."
            )

    @property
    def size(self) -> int:
        """Nombre d'équations et d'inconnues."""
        return self.coefficients.rows


@dataclass(frozen=True)
class LinearSystemSolution:
    """
    Résultat de la résolution d'un système linéaire.
    """

    solution_type: str
    values: tuple[float, ...] | None = None

    @property
    def is_unique(self) -> bool:
        return self.solution_type == "solution unique"

    @property
    def has_no_solution(self) -> bool:
        return self.solution_type == "aucune solution"

    @property
    def has_infinite_solutions(self) -> bool:
        return self.solution_type == "une infinité de solutions"


def create_linear_system(
    coefficients: Matrix,
    constants: Vector,
) -> LinearSystem:
    """
    Crée un système linéaire AX = B.
    """
    return LinearSystem(
        coefficients=coefficients,
        constants=constants,
    )


def solve_linear_system(
    system: LinearSystem,
) -> LinearSystemSolution:
    """
    Résout un système linéaire par élimination de Gauss.
    """

    size = system.size

    augmented = [
        [
            float(system.coefficients.values[row][column])
            for column in range(size)
        ]
        + [float(system.constants.values[row])]
        for row in range(size)
    ]

    pivot_row = 0
    pivot_columns = []

    for column in range(size):
        current_pivot = max(
            range(pivot_row, size),
            key=lambda row: abs(augmented[row][column]),
        )

        pivot = augmented[current_pivot][column]

        if isclose(pivot, 0.0, abs_tol=1e-12):
            continue

        if current_pivot != pivot_row:
            augmented[pivot_row], augmented[current_pivot] = (
                augmented[current_pivot],
                augmented[pivot_row],
            )

        pivot = augmented[pivot_row][column]

        for row in range(pivot_row + 1, size):
            factor = augmented[row][column] / pivot

            for current_column in range(column, size + 1):
                augmented[row][current_column] -= (
                    factor * augmented[pivot_row][current_column]
                )

        pivot_columns.append(column)
        pivot_row += 1

        if pivot_row == size:
            break

    # Vérification des lignes du type :
    # 0 0 ... 0 | c
    # avec c != 0
    for row in range(size):
        coefficients_zero = all(
            isclose(augmented[row][column], 0.0, abs_tol=1e-10)
            for column in range(size)
        )

        constant_non_zero = not isclose(
            augmented[row][size],
            0.0,
            abs_tol=1e-10,
        )

        if coefficients_zero and constant_non_zero:
            return LinearSystemSolution(
                solution_type="aucune solution"
            )

    # Si le nombre de pivots est inférieur au nombre
    # d'inconnues, il existe une infinité de solutions.
    if len(pivot_columns) < size:
        return LinearSystemSolution(
            solution_type="une infinité de solutions"
        )

    # Remontée de Gauss pour obtenir la solution.
    solution = [0.0] * size

    for row in range(size - 1, -1, -1):
        pivot_column = None

        for column in range(size):
            if not isclose(
                augmented[row][column],
                0.0,
                abs_tol=1e-10,
            ):
                pivot_column = column
                break

        if pivot_column is None:
            continue

        value = augmented[row][size]

        for column in range(pivot_column + 1, size):
            value -= (
                augmented[row][column]
                * solution[column]
            )

        solution[pivot_column] = (
            value / augmented[row][pivot_column]
        )

    normalized_solution = tuple(
        int(round(value))
        if isclose(value, round(value), abs_tol=1e-10)
        else value
        for value in solution
    )

    return LinearSystemSolution(
        solution_type="solution unique",
        values=normalized_solution,
    )