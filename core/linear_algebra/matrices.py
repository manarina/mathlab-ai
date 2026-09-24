from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable

Number = int | float


@dataclass(frozen=True)
class Matrix:
    """
    Représente une matrice numérique de dimension finie.
    """

    values: tuple[tuple[Number, ...], ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(
                "Une matrice doit contenir au moins une ligne."
            )

        column_count = len(self.values[0])

        if column_count == 0:
            raise ValueError(
                "Une matrice doit contenir au moins une colonne."
            )

        if any(
            len(row) != column_count
            for row in self.values
        ):
            raise ValueError(
                "Toutes les lignes doivent avoir le même nombre de colonnes."
            )

        if not all(
            isinstance(value, (int, float))
            for row in self.values
            for value in row
        ):
            raise TypeError(
                "Toutes les valeurs de la matrice doivent être numériques."
            )

    # ========================================================
    # PROPRIÉTÉS
    # ========================================================

    @property
    def rows(self) -> int:
        """Nombre de lignes."""
        return len(self.values)

    @property
    def columns(self) -> int:
        """Nombre de colonnes."""
        return len(self.values[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions de la matrice : lignes × colonnes."""
        return self.rows, self.columns

    @property
    def is_square(self) -> bool:
        """Indique si la matrice est carrée."""
        return self.rows == self.columns

    # ========================================================
    # ACCÈS AUX ÉLÉMENTS
    # ========================================================

    def get(
        self,
        row: int,
        column: int,
    ) -> Number:
        """
        Retourne l'élément situé à la ligne row
        et à la colonne column.

        Les indices commencent à 0.
        """
        return self.values[row][column]

    # ========================================================
    # ADDITION
    # ========================================================

    def __add__(self, other: Matrix) -> Matrix:
        _check_same_shape(self, other)

        return Matrix(
            tuple(
                tuple(
                    a + b
                    for a, b in zip(row_a, row_b)
                )
                for row_a, row_b in zip(
                    self.values,
                    other.values,
                )
            )
        )

    # ========================================================
    # SOUSTRACTION
    # ========================================================

    def __sub__(self, other: Matrix) -> Matrix:
        _check_same_shape(self, other)

        return Matrix(
            tuple(
                tuple(
                    a - b
                    for a, b in zip(row_a, row_b)
                )
                for row_a, row_b in zip(
                    self.values,
                    other.values,
                )
            )
        )

    # ========================================================
    # MULTIPLICATION PAR UN SCALAIRE
    # ========================================================

    def __mul__(self, scalar: Number) -> Matrix:
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                "Le scalaire doit être un nombre."
            )

        return Matrix(
            tuple(
                tuple(
                    scalar * value
                    for value in row
                )
                for row in self.values
            )
        )

    def __rmul__(self, scalar: Number) -> Matrix:
        return self.__mul__(scalar)

    # ========================================================
    # MULTIPLICATION MATRICIELLE
    # ========================================================

    def __matmul__(self, other: Matrix) -> Matrix:
        return matrix_multiply(self, other)

    # ========================================================
    # TRANSPOSÉE
    # ========================================================

    def transpose(self) -> Matrix:
        """
        Retourne la matrice transposée.
        """
        return Matrix(
            tuple(
                tuple(
                    self.values[row][column]
                    for row in range(self.rows)
                )
                for column in range(self.columns)
            )
        )

    # ========================================================
    # DÉTERMINANT
    # ========================================================

    def determinant(self) -> Number:
        """
        Calcule le déterminant d'une matrice carrée.

        Utilise une élimination de Gauss avec pivot partiel.
        """
        if not self.is_square:
            raise ValueError(
                "Le déterminant est défini uniquement "
                "pour une matrice carrée."
            )

        size = self.rows

        if size == 1:
            return self.values[0][0]

        if size == 2:
            a = self.values[0][0]
            b = self.values[0][1]
            c = self.values[1][0]
            d = self.values[1][1]

            return a * d - b * c

        matrix = [
            list(row)
            for row in self.values
        ]

        determinant = 1.0
        sign = 1

        for column in range(size):
            pivot_row = max(
                range(column, size),
                key=lambda row: abs(matrix[row][column]),
            )

            pivot = matrix[pivot_row][column]

            if isclose(pivot, 0.0, abs_tol=1e-12):
                return 0

            if pivot_row != column:
                matrix[column], matrix[pivot_row] = (
                    matrix[pivot_row],
                    matrix[column],
                )
                sign *= -1

            pivot = matrix[column][column]
            determinant *= pivot

            for row in range(column + 1, size):
                factor = (
                    matrix[row][column] / pivot
                )

                for current_column in range(
                    column,
                    size,
                ):
                    matrix[row][current_column] -= (
                        factor
                        * matrix[column][current_column]
                    )

        result = sign * determinant

        if isclose(
            result,
            round(result),
            abs_tol=1e-10,
        ):
            return int(round(result))

        return result

    # ========================================================
    # INVERSE
    # ========================================================

    def inverse(self) -> Matrix:
        """
        Calcule l'inverse d'une matrice carrée
        par la méthode de Gauss-Jordan.
        """
        if not self.is_square:
            raise ValueError(
                "Seule une matrice carrée peut avoir une inverse."
            )

        determinant = self.determinant()

        if isclose(
            float(determinant),
            0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "La matrice est singulière et n'est pas inversible."
            )

        size = self.rows

        augmented = [
            list(row) + [
                1 if row_index == column else 0
                for column in range(size)
            ]
            for row_index, row in enumerate(
                self.values
            )
        ]

        for column in range(size):
            pivot_row = max(
                range(column, size),
                key=lambda row: abs(
                    augmented[row][column]
                ),
            )

            pivot = augmented[pivot_row][column]

            if isclose(
                pivot,
                0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "La matrice est singulière et n'est pas inversible."
                )

            if pivot_row != column:
                augmented[column], augmented[pivot_row] = (
                    augmented[pivot_row],
                    augmented[column],
                )

            pivot = augmented[column][column]

            augmented[column] = [
                value / pivot
                for value in augmented[column]
            ]

            for row in range(size):
                if row == column:
                    continue

                factor = augmented[row][column]

                augmented[row] = [
                    current
                    - factor * pivot_value
                    for current, pivot_value in zip(
                        augmented[row],
                        augmented[column],
                    )
                ]

        inverse_values = tuple(
            tuple(
                augmented[row][column]
                for column in range(size, 2 * size)
            )
            for row in range(size)
        )

        return Matrix(inverse_values)

    # ========================================================
    # CONVERSION
    # ========================================================

    def to_tuple(self) -> tuple[tuple[Number, ...], ...]:
        """Retourne les valeurs sous forme de tuple imbriqué."""
        return self.values


# ============================================================
# VALIDATION
# ============================================================

def _check_same_shape(
    first: Matrix,
    second: Matrix,
) -> None:
    if first.shape != second.shape:
        raise ValueError(
            "Les matrices doivent avoir les mêmes dimensions."
        )


def create_matrix(
    values: Iterable[Iterable[Number]],
) -> Matrix:
    """
    Crée une matrice à partir d'un iterable de lignes.
    """
    values_tuple = tuple(
        tuple(row)
        for row in values
    )

    return Matrix(values_tuple)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def add_matrices(
    first: Matrix,
    second: Matrix,
) -> Matrix:
    return first + second


def subtract_matrices(
    first: Matrix,
    second: Matrix,
) -> Matrix:
    return first - second


def scalar_multiply_matrix(
    matrix: Matrix,
    scalar: Number,
) -> Matrix:
    return matrix * scalar


def matrix_multiply(
    first: Matrix,
    second: Matrix,
) -> Matrix:
    """
    Multiplication matricielle classique.

    Si A est de dimension m × n et B de dimension n × p,
    alors A × B est de dimension m × p.
    """
    if first.columns != second.rows:
        raise ValueError(
            "Le nombre de colonnes de la première matrice "
            "doit être égal au nombre de lignes de la deuxième."
        )

    result = tuple(
        tuple(
            sum(
                first.values[row][index]
                * second.values[index][column]
                for index in range(first.columns)
            )
            for column in range(second.columns)
        )
        for row in range(first.rows)
    )

    return Matrix(result)


def transpose_matrix(
    matrix: Matrix,
) -> Matrix:
    return matrix.transpose()


def matrix_determinant(
    matrix: Matrix,
) -> Number:
    return matrix.determinant()


def matrix_inverse(
    matrix: Matrix,
) -> Matrix:
    return matrix.inverse()