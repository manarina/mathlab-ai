from __future__ import annotations

import math

import pytest

from core.linear_algebra.matrices import (
    Matrix,
    add_matrices,
    create_matrix,
    matrix_determinant,
    matrix_inverse,
    matrix_multiply,
    scalar_multiply_matrix,
    subtract_matrices,
    transpose_matrix,
)


# ============================================================
# CRÉATION
# ============================================================


def test_create_matrix() -> None:
    matrix = create_matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    assert matrix.values == (
        (1, 2),
        (3, 4),
    )


def test_matrix_creation_directly() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    assert matrix.values == (
        (1, 2, 3),
        (4, 5, 6),
    )


def test_empty_matrix_raises_error() -> None:
    with pytest.raises(ValueError):
        Matrix(())


def test_matrix_with_empty_row_raises_error() -> None:
    with pytest.raises(ValueError):
        Matrix(
            (
                (),
            )
        )


def test_irregular_matrix_raises_error() -> None:
    with pytest.raises(ValueError):
        Matrix(
            (
                (1, 2),
                (3,),
            )
        )


def test_non_numeric_matrix_raises_error() -> None:
    with pytest.raises(TypeError):
        Matrix(
            (
                (1, 2),
                (3, "a"),
            )
        )


# ============================================================
# DIMENSIONS
# ============================================================


def test_matrix_rows() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    assert matrix.rows == 2


def test_matrix_columns() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    assert matrix.columns == 3


def test_matrix_shape() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    assert matrix.shape == (2, 3)


def test_square_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    assert matrix.is_square is True


def test_non_square_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    assert matrix.is_square is False


# ============================================================
# ACCÈS AUX ÉLÉMENTS
# ============================================================


def test_get_matrix_element() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    assert matrix.get(0, 0) == 1
    assert matrix.get(0, 2) == 3
    assert matrix.get(1, 1) == 5


# ============================================================
# ADDITION
# ============================================================


def test_matrix_addition() -> None:
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

    assert result.values == (
        (6, 8),
        (10, 12),
    )


def test_add_matrices_helper() -> None:
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

    result = add_matrices(first, second)

    assert result.values == (
        (6, 8),
        (10, 12),
    )


def test_matrix_addition_dimension_mismatch() -> None:
    first = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    second = Matrix(
        (
            (5, 6, 7),
        )
    )

    with pytest.raises(ValueError):
        first + second


# ============================================================
# SOUSTRACTION
# ============================================================


def test_matrix_subtraction() -> None:
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

    result = first - second

    assert result.values == (
        (-4, -4),
        (-4, -4),
    )


def test_subtract_matrices_helper() -> None:
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

    result = subtract_matrices(first, second)

    assert result.values == (
        (4, 4),
        (4, 4),
    )


def test_matrix_subtraction_dimension_mismatch() -> None:
    first = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    second = Matrix(
        (
            (1, 2, 3),
        )
    )

    with pytest.raises(ValueError):
        first - second


# ============================================================
# MULTIPLICATION PAR UN SCALAIRE
# ============================================================


def test_matrix_scalar_multiplication() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = matrix * 2

    assert result.values == (
        (2, 4),
        (6, 8),
    )


def test_reverse_scalar_multiplication() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = 3 * matrix

    assert result.values == (
        (3, 6),
        (9, 12),
    )


def test_scalar_multiply_matrix_helper() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = scalar_multiply_matrix(
        matrix,
        5,
    )

    assert result.values == (
        (5, 10),
        (15, 20),
    )


def test_matrix_invalid_scalar() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    with pytest.raises(TypeError):
        matrix * "2"


# ============================================================
# MULTIPLICATION MATRICIELLE
# ============================================================


def test_matrix_multiplication() -> None:
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

    assert result.values == (
        (19, 22),
        (43, 50),
    )


def test_matrix_multiplication_helper() -> None:
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

    result = matrix_multiply(
        first,
        second,
    )

    assert result.values == (
        (58, 64),
        (139, 154),
    )


def test_matrix_multiplication_rectangular_matrices() -> None:
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

    assert result.shape == (2, 2)


def test_matrix_multiplication_dimension_mismatch() -> None:
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

    with pytest.raises(ValueError):
        first @ second

# ============================================================
# TRANSPOSÉE
# ============================================================


def test_matrix_transpose() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    result = matrix.transpose()

    assert result.values == (
        (1, 4),
        (2, 5),
        (3, 6),
    )


def test_transpose_matrix_helper() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
            (5, 6),
        )
    )

    result = transpose_matrix(matrix)

    assert result.values == (
        (1, 3, 5),
        (2, 4, 6),
    )


def test_transpose_square_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = matrix.transpose()

    assert result.values == (
        (1, 3),
        (2, 4),
    )


# ============================================================
# DÉTERMINANT
# ============================================================


def test_determinant_1x1() -> None:
    matrix = Matrix(
        (
            (5,),
        )
    )

    assert matrix.determinant() == 5


def test_determinant_2x2() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    assert matrix.determinant() == -2


def test_determinant_3x3() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (0, 1, 4),
            (5, 6, 0),
        )
    )

    assert matrix.determinant() == 1


def test_determinant_4x4() -> None:
    matrix = Matrix(
        (
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (2, 6, 4, 8),
            (3, 1, 1, 2),
        )
    )

    assert matrix.determinant() == 72


def test_determinant_singular_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (2, 4),
        )
    )

    assert matrix.determinant() == 0


def test_determinant_non_square_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    with pytest.raises(ValueError):
        matrix.determinant()


def test_matrix_determinant_helper() -> None:
    matrix = Matrix(
        (
            (2, 3),
            (1, 4),
        )
    )

    assert matrix_determinant(matrix) == 5


# ============================================================
# INVERSE
# ============================================================


def test_inverse_2x2() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    result = matrix.inverse()

    assert math.isclose(
        result.values[0][0],
        -2.0,
    )

    assert math.isclose(
        result.values[0][1],
        1.0,
    )

    assert math.isclose(
        result.values[1][0],
        1.5,
    )

    assert math.isclose(
        result.values[1][1],
        -0.5,
    )


def test_inverse_identity_matrix() -> None:
    matrix = Matrix(
        (
            (1, 0),
            (0, 1),
        )
    )

    result = matrix.inverse()

    assert result.values == (
        (1.0, 0.0),
        (0.0, 1.0),
    )


def test_inverse_3x3() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (0, 1, 4),
            (5, 6, 0),
        )
    )

    inverse = matrix.inverse()

    identity = matrix @ inverse

    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0

            assert math.isclose(
                identity.values[row][column],
                expected,
                abs_tol=1e-9,
            )


def test_matrix_inverse_helper() -> None:
    matrix = Matrix(
        (
            (4, 7),
            (2, 6),
        )
    )

    result = matrix_inverse(matrix)

    assert math.isclose(
        result.values[0][0],
        0.6,
        abs_tol=1e-9,
    )

    assert math.isclose(
        result.values[0][1],
        -0.7,
        abs_tol=1e-9,
    )

    assert math.isclose(
        result.values[1][0],
        -0.2,
        abs_tol=1e-9,
    )

    assert math.isclose(
        result.values[1][1],
        0.4,
        abs_tol=1e-9,
    )


def test_inverse_singular_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (2, 4),
        )
    )

    with pytest.raises(ValueError):
        matrix.inverse()


def test_inverse_non_square_matrix() -> None:
    matrix = Matrix(
        (
            (1, 2, 3),
            (4, 5, 6),
        )
    )

    with pytest.raises(ValueError):
        matrix.inverse()


# ============================================================
# CONVERSION
# ============================================================


def test_to_tuple() -> None:
    matrix = Matrix(
        (
            (1, 2),
            (3, 4),
        )
    )

    assert matrix.to_tuple() == (
        (1, 2),
        (3, 4),
    )