from __future__ import annotations

import pytest

from core.linear_algebra.matrices import Matrix
from core.linear_algebra.matrix_explanations import (
    explain_matrix,
    explain_addition,
    explain_subtraction,
    explain_scalar_multiplication,
    explain_multiplication,
    explain_transpose,
    explain_determinant,
    explain_inverse,
)


# ============================================================
# explain_matrix
# ============================================================


def test_explain_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = explain_matrix(matrix)

    assert "### Matrice" in result
    assert "1, 2" in result
    assert "3, 4" in result
    assert "2 ligne(s)" in result
    assert "2 colonne(s)" in result
    assert "2 × 2" in result


def test_explain_matrix_rectangular() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    result = explain_matrix(matrix)

    assert "2 ligne(s)" in result
    assert "3 colonne(s)" in result
    assert "2 × 3" in result


# ============================================================
# explain_addition
# ============================================================


def test_explain_addition() -> None:
    first = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    second = Matrix(
        (
            (5, 6),
            (7, 8),
        )
    )

    result = first + second

    explanation = explain_addition(
        first,
        second,
        result,
    )

    assert "### Addition de matrices" in explanation
    assert "1 + 5 = 6" in explanation
    assert "2 + 6 = 8" in explanation
    assert "3 + 7 = 10" in explanation
    assert "4 + 8 = 12" in explanation
    assert "Résultat" in explanation


def test_explain_addition_dimension_mismatch() -> None:
    first = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    second = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    result = Matrix(
        (
            (2, 4),
            (6, 8),
        )
    )

    with pytest.raises(ValueError):
        explain_addition(
            first,
            second,
            result,
        )


# ============================================================
# explain_subtraction
# ============================================================


def test_explain_subtraction() -> None:
    first = Matrix(
        (
            (5, 6),
            (7, 8),
        )
    )

    second = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = first - second

    explanation = explain_subtraction(
        first,
        second,
        result,
    )

    assert "### Soustraction de matrices" in explanation
    assert "5 - 1 = 4" in explanation
    assert "6 - 2 = 4" in explanation
    assert "7 - 3 = 4" in explanation
    assert "8 - 4 = 4" in explanation
    assert "Résultat" in explanation


def test_explain_subtraction_dimension_mismatch() -> None:
    first = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    second = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    result = Matrix(
        (
            (0, 0),
            (0, 0),
        )
    )

    with pytest.raises(ValueError):
        explain_subtraction(
            first,
            second,
            result,
        )


# ============================================================
# explain_scalar_multiplication
# ============================================================


def test_explain_scalar_multiplication() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = 2 * matrix

    explanation = explain_scalar_multiplication(
        matrix,
        2,
        result,
    )

    assert "### Multiplication par un scalaire" in explanation
    assert "2 × 1 = 2" in explanation
    assert "2 × 2 = 4" in explanation
    assert "2 × 3 = 6" in explanation
    assert "2 × 4 = 8" in explanation
    assert "Résultat" in explanation


def test_explain_scalar_multiplication_negative_scalar() -> None:
    matrix = Matrix(
        (
            (1, -2),
            (3, -4),
        )
    )

    result = -2 * matrix

    explanation = explain_scalar_multiplication(
        matrix,
        -2,
        result,
    )

    assert "-2 × 1 = -2" in explanation
    assert "-2 × -2 = 4" in explanation
    assert "-2 × 3 = -6" in explanation
    assert "-2 × -4 = 8" in explanation


# ============================================================
# explain_multiplication
# ============================================================


def test_explain_multiplication() -> None:
    first = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    second = Matrix(
        (
            (5, 6),
            (7, 8),
        )
    )

    result = first @ second

    explanation = explain_multiplication(
        first,
        second,
        result,
    )

    assert "### Multiplication matricielle" in explanation
    assert "1 × 5 + 2 × 7" in explanation
    assert "1 × 6 + 2 × 8" in explanation
    assert "3 × 5 + 4 × 7" in explanation
    assert "3 × 6 + 4 × 8" in explanation
    assert "= 19" in explanation
    assert "= 22" in explanation
    assert "= 43" in explanation
    assert "= 50" in explanation
    assert "Résultat" in explanation


def test_explain_multiplication_rectangular() -> None:
    first = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    second = Matrix(
        (
            (7, 8),
            (9, 10),
            (11, 12),
        )
    )

    result = first @ second

    explanation = explain_multiplication(
        first,
        second,
        result,
    )

    assert "2 × 3" in explanation
    assert "3 × 2" in explanation
    assert "2 × 2" in explanation
    assert "1 × 7 + 2 × 9 + 3 × 11" in explanation
    assert "4 × 7 + 5 × 9 + 6 × 11" in explanation


def test_explain_multiplication_dimension_mismatch() -> None:
    first = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    second = Matrix(
        (
            (1, 2),
            (3, 4),
            (5, 6),
        )
    )

    result = Matrix(
        (
            (0, 0),
            (0, 0),
        )
    )

    with pytest.raises(ValueError):
        explain_multiplication(
            first,
            second,
            result,
        )


# ============================================================
# explain_transpose
# ============================================================


def test_explain_transpose() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    result = matrix.transpose()

    explanation = explain_transpose(
        matrix,
        result,
    )

    assert "### Transposée d'une matrice" in explanation
    assert "Aᵀ" in explanation
    assert "2 ligne(s)" in explanation
    assert "3 colonne(s)" in explanation
    assert "A(1,1) = 1" in explanation
    assert "Aᵀ(1,1) = 1" in explanation
    assert "A(1,2) = 2" in explanation
    assert "Aᵀ(2,1) = 2" in explanation
    assert "Résultat" in explanation


def test_explain_transpose_square_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = matrix.transpose()

    explanation = explain_transpose(
        matrix,
        result,
    )

    assert "A(1,2) = 2" in explanation
    assert "Aᵀ(2,1) = 2" in explanation
    assert "A(2,1) = 3" in explanation
    assert "Aᵀ(1,2) = 3" in explanation


# ============================================================
# explain_determinant
# ============================================================


def test_explain_determinant_1x1() -> None:
    matrix = Matrix(
        (
            (5,),
        )
    )

    result = matrix.determinant()

    explanation = explain_determinant(
        matrix,
        result,
    )

    assert "### Déterminant d'une matrice" in explanation
    assert "matrice 1 × 1" in explanation
    assert "det(A) = a₁₁" in explanation
    assert "det(A) = 5" in explanation


def test_explain_determinant_2x2() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = matrix.determinant()

    explanation = explain_determinant(
        matrix,
        result,
    )

    assert "### Déterminant d'une matrice" in explanation
    assert "det(A) = ad - bc" in explanation
    assert "(1 × 4)" in explanation
    assert "(2 × 3)" in explanation
    assert "4 - 6" in explanation
    assert "det(A) = -2" in explanation


def test_explain_determinant_3x3() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (0, 1, 4),
            (5, 6, 0),
        )
    )

    result = matrix.determinant()

    explanation = explain_determinant(
        matrix,
        result,
    )

    assert "### Déterminant d'une matrice" in explanation
    assert "dimension 3 × 3" in explanation
    assert "élimination" in explanation
    assert f"det(A) = {result}" in explanation


def test_explain_determinant_non_square() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    with pytest.raises(ValueError):
        explain_determinant(
            matrix,
            0,
        )


# ============================================================
# explain_inverse
# ============================================================


def test_explain_inverse_2x2() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = matrix.inverse()

    explanation = explain_inverse(
        matrix,
        result,
    )

    assert "### Inverse d'une matrice" in explanation
    assert "Gauss-Jordan" in explanation
    assert "[ A | I ]" in explanation
    assert "[ I | A⁻¹ ]" in explanation
    assert "matrice identité" in explanation
    assert "A⁻¹" in explanation
    assert "déterminant" in explanation


def test_explain_inverse_identity() -> None:
    matrix = Matrix(
        (
            (1, 0),
            (0, 1),
        )
    )

    result = matrix.inverse()

    explanation = explain_inverse(
        matrix,
        result,
    )

    assert "Gauss-Jordan" in explanation
    assert "A⁻¹" in explanation
    assert str(result.values) in explanation


def test_explain_inverse_3x3() -> None:
    matrix = Matrix(
        (
            (1, 0, 0),
            (0, 2, 0),
            (0, 0, 4),
        )
    )

    result = matrix.inverse()

    explanation = explain_inverse(
        matrix,
        result,
    )

    assert "Gauss-Jordan" in explanation
    assert "A⁻¹" in explanation
    assert str(result.values) in explanation


def test_explain_inverse_non_square() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    result = Matrix(
        (
            (1, 0),
            (0, 1),
        )
    )

    with pytest.raises(ValueError):
        explain_inverse(
            matrix,
            result,
        )