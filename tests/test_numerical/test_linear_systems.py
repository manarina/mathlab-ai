from __future__ import annotations

import math

import pytest

from core.numerical.linear_systems import (
    LinearSystemResult,
    _validate_matrix,
    _validate_number,
    _validate_vector,
    gaussian_elimination,
)


# ============================================================
# FONCTIONS UTILITAIRES POUR LES TESTS
# ============================================================


def assert_solution_close(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    tolerance: float = 1e-10,
) -> None:
    """Vérifie deux vecteurs solution avec une tolérance numérique."""
    assert len(actual) == len(expected)

    for actual_value, expected_value in zip(actual, expected):
        assert actual_value == pytest.approx(
            expected_value,
            abs=tolerance,
        )


# ============================================================
# _validate_number
# ============================================================


def test_validate_number_accepts_integer():
    assert _validate_number(5, "x") == 5.0


def test_validate_number_accepts_float():
    assert _validate_number(3.14, "x") == 3.14


def test_validate_number_rejects_boolean():
    with pytest.raises(TypeError):
        _validate_number(True, "x")


def test_validate_number_rejects_string():
    with pytest.raises(TypeError):
        _validate_number("3.14", "x")


def test_validate_number_rejects_nan():
    with pytest.raises(ValueError):
        _validate_number(float("nan"), "x")


def test_validate_number_rejects_positive_infinity():
    with pytest.raises(ValueError):
        _validate_number(float("inf"), "x")


def test_validate_number_rejects_negative_infinity():
    with pytest.raises(ValueError):
        _validate_number(float("-inf"), "x")


# ============================================================
# _validate_matrix
# ============================================================


def test_validate_matrix_accepts_valid_2x2_matrix():
    matrix = _validate_matrix(
        [
            [2, 1],
            [1, -1],
        ]
    )

    assert matrix == [
        [2.0, 1.0],
        [1.0, -1.0],
    ]


def test_validate_matrix_accepts_valid_3x3_matrix():
    matrix = _validate_matrix(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 10],
        ]
    )

    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)


def test_validate_matrix_rejects_empty_matrix():
    with pytest.raises(ValueError):
        _validate_matrix([])


def test_validate_matrix_rejects_non_sequence():
    with pytest.raises(TypeError):
        _validate_matrix(123)


def test_validate_matrix_rejects_string():
    with pytest.raises(TypeError):
        _validate_matrix("invalid")


def test_validate_matrix_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        _validate_matrix(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )


def test_validate_matrix_rejects_inconsistent_rows():
    with pytest.raises(ValueError):
        _validate_matrix(
            [
                [1, 2],
                [3],
            ]
        )


def test_validate_matrix_rejects_invalid_value():
    with pytest.raises(TypeError):
        _validate_matrix(
            [
                [1, "invalid"],
                [3, 4],
            ]
        )


def test_validate_matrix_rejects_nan():
    with pytest.raises(ValueError):
        _validate_matrix(
            [
                [1, math.nan],
                [3, 4],
            ]
        )


# ============================================================
# _validate_vector
# ============================================================


def test_validate_vector_accepts_valid_vector():
    vector = _validate_vector([5, 1], 2)

    assert vector == [5.0, 1.0]


def test_validate_vector_accepts_3_elements():
    vector = _validate_vector([1, 2, 3], 3)

    assert vector == [1.0, 2.0, 3.0]


def test_validate_vector_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        _validate_vector([1, 2], 3)


def test_validate_vector_rejects_string():
    with pytest.raises(TypeError):
        _validate_vector("invalid", 7)


def test_validate_vector_rejects_invalid_value():
    with pytest.raises(TypeError):
        _validate_vector([1, "invalid"], 2)


def test_validate_vector_rejects_nan():
    with pytest.raises(ValueError):
        _validate_vector([1, math.nan], 2)


# ============================================================
# RÉSULTAT
# ============================================================


def test_linear_system_result_is_dataclass():
    result = LinearSystemResult(
        solution=(2.0, 1.0),
        matrix=((2.0, 1.0), (0.0, -1.5)),
        vector=(5.0, -1.5),
        swaps=0,
    )

    assert result.solution == (2.0, 1.0)
    assert result.swaps == 0


def test_linear_system_result_is_immutable():
    result = LinearSystemResult(
        solution=(2.0, 1.0),
        matrix=((2.0, 1.0), (0.0, -1.5)),
        vector=(5.0, -1.5),
        swaps=0,
    )

    with pytest.raises(Exception):
        result.solution = (0.0, 0.0)


# ============================================================
# GAUSS — SYSTÈME 2x2
# ============================================================


def test_gaussian_elimination_solves_simple_2x2_system():
    result = gaussian_elimination(
        [
            [2, 1],
            [1, -1],
        ],
        [5, 1],
    )

    assert_solution_close(
        result.solution,
        (2.0, 1.0),
    )


def test_gaussian_elimination_returns_correct_result_type():
    result = gaussian_elimination(
        [
            [2, 1],
            [1, -1],
        ],
        [5, 1],
    )

    assert isinstance(result, LinearSystemResult)


def test_gaussian_elimination_returns_triangular_matrix():
    result = gaussian_elimination(
        [
            [2, 1],
            [1, -1],
        ],
        [5, 1],
    )

    assert result.matrix[1][0] == pytest.approx(0.0)


def test_gaussian_elimination_returns_transformed_vector():
    result = gaussian_elimination(
        [
            [2, 1],
            [1, -1],
        ],
        [5, 1],
    )

    assert len(result.vector) == 2


# ============================================================
# GAUSS — SYSTÈME 3x3
# ============================================================


def test_gaussian_elimination_solves_3x3_system():
    result = gaussian_elimination(
        [
            [2, 1, -1],
            [-3, -1, 2],
            [-2, 1, 2],
        ],
        [8, -11, -3],
    )

    assert_solution_close(
        result.solution,
        (2.0, 3.0, -1.0),
    )


def test_gaussian_elimination_solves_another_3x3_system():
    result = gaussian_elimination(
        [
            [1, 2, 3],
            [2, -1, 1],
            [3, 1, -2],
        ],
        [14, 3, 1],
    )

    assert_solution_close(
        result.solution,
        (4 / 3, 7 / 3, 8 / 3),
    )

# ============================================================
# PIVOT NUL — ÉCHANGE DE LIGNES
# ============================================================


def test_gaussian_elimination_performs_row_swap():
    result = gaussian_elimination(
        [
            [0, 2],
            [1, 1],
        ],
        [4, 3],
    )

    assert_solution_close(
        result.solution,
        (1.0, 2.0),
    )

    assert result.swaps == 1


def test_gaussian_elimination_handles_zero_initial_pivot():
    result = gaussian_elimination(
        [
            [0, 1, 1],
            [1, 2, 3],
            [2, 1, 2],
        ],
        [3, 9, 6],
    )

    assert_solution_close(
        result.solution,
        (0.0, 0.0, 3.0),
    )

    assert result.swaps >= 1


def test_gaussian_elimination_prefers_larger_pivot():
    result = gaussian_elimination(
        [
            [1e-15, 1],
            [1, 2],
        ],
        [1, 3],
    )

    assert_solution_close(
        result.solution,
        (1.0, 1.0),
        tolerance=1e-9,
    )

    assert result.swaps == 1


# ============================================================
# SYSTÈME SINGULIER
# ============================================================


def test_gaussian_elimination_rejects_singular_matrix():
    with pytest.raises(ValueError, match="singulier"):
        gaussian_elimination(
            [
                [1, 2],
                [2, 4],
            ],
            [3, 6],
        )


def test_gaussian_elimination_rejects_singular_3x3_matrix():
    with pytest.raises(ValueError, match="singulier"):
        gaussian_elimination(
            [
                [1, 2, 3],
                [2, 4, 6],
                [3, 6, 9],
            ],
            [6, 12, 18],
        )


# ============================================================
# SYSTÈME INCOMPATIBLE
# ============================================================


def test_gaussian_elimination_rejects_incompatible_system():
    with pytest.raises(ValueError, match="incompatible"):
        gaussian_elimination(
            [
                [1, 2],
                [2, 4],
            ],
            [3, 7],
        )


def test_gaussian_elimination_rejects_incompatible_3x3_system():
    with pytest.raises(ValueError, match="incompatible"):
        gaussian_elimination(
            [
                [1, 2, 3],
                [2, 4, 6],
                [1, 2, 3],
            ],
            [6, 12, 7],
        )


# ============================================================
# TOLÉRANCE
# ============================================================


def test_gaussian_elimination_rejects_zero_tolerance():
    with pytest.raises(ValueError):
        gaussian_elimination(
            [[1, 0], [0, 1]],
            [1, 2],
            tolerance=0,
        )


def test_gaussian_elimination_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        gaussian_elimination(
            [[1, 0], [0, 1]],
            [1, 2],
            tolerance=-1e-10,
        )


def test_gaussian_elimination_rejects_nan_tolerance():
    with pytest.raises(ValueError):
        gaussian_elimination(
            [[1, 0], [0, 1]],
            [1, 2],
            tolerance=math.nan,
        )


def test_gaussian_elimination_rejects_infinite_tolerance():
    with pytest.raises(ValueError):
        gaussian_elimination(
            [[1, 0], [0, 1]],
            [1, 2],
            tolerance=math.inf,
        )


# ============================================================
# DIMENSIONS ET STRUCTURE
# ============================================================


def test_gaussian_elimination_rejects_empty_matrix():
    with pytest.raises(ValueError):
        gaussian_elimination([], [])


def test_gaussian_elimination_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        gaussian_elimination(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [7, 8],
        )


def test_gaussian_elimination_rejects_wrong_vector_dimension():
    with pytest.raises(ValueError):
        gaussian_elimination(
            [
                [1, 2],
                [3, 4],
            ],
            [5],
        )


def test_gaussian_elimination_rejects_non_numeric_matrix():
    with pytest.raises(TypeError):
        gaussian_elimination(
            [
                [1, "x"],
                [3, 4],
            ],
            [5, 6],
        )


def test_gaussian_elimination_rejects_non_numeric_vector():
    with pytest.raises(TypeError):
        gaussian_elimination(
            [
                [1, 2],
                [3, 4],
            ],
            [5, "x"],
        )


def test_gaussian_elimination_rejects_boolean_matrix_value():
    with pytest.raises(TypeError):
        gaussian_elimination(
            [
                [True, 2],
                [3, 4],
            ],
            [5, 6],
        )


def test_gaussian_elimination_rejects_boolean_vector_value():
    with pytest.raises(TypeError):
        gaussian_elimination(
            [
                [1, 2],
                [3, 4],
            ],
            [True, 6],
        )


# ============================================================
# PRÉCISION NUMÉRIQUE
# ============================================================


def test_gaussian_elimination_has_high_precision():
    result = gaussian_elimination(
        [
            [0.1, 0.2],
            [0.3, 0.7],
        ],
        [0.5, 1.7],
    )

    assert_solution_close(
        result.solution,
        (1.0, 2.0),
        tolerance=1e-10,
    )


def test_gaussian_elimination_solves_diagonal_system():
    result = gaussian_elimination(
        [
            [2, 0, 0],
            [0, 4, 0],
            [0, 0, 5],
        ],
        [6, 8, 10],
    )

    assert_solution_close(
        result.solution,
        (3.0, 2.0, 2.0),
    )


def test_gaussian_elimination_identity_matrix():
    result = gaussian_elimination(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        [4, -2, 7],
    )

    assert_solution_close(
        result.solution,
        (4.0, -2.0, 7.0),
    )


# ============================================================
# CAS PARTICULIERS
# ============================================================


def test_gaussian_elimination_single_equation():
    result = gaussian_elimination(
        [[4]],
        [12],
    )

    assert_solution_close(
        result.solution,
        (3.0,),
    )


def test_gaussian_elimination_negative_values():
    result = gaussian_elimination(
        [
            [-2, 1],
            [1, -3],
        ],
        [-5, 7],
    )

    assert_solution_close(
        result.solution,
        (1.6, -1.8),
    )


def test_gaussian_elimination_does_not_modify_input_matrix():
    matrix = [
        [2, 1],
        [1, -1],
    ]

    original = [
        [2, 1],
        [1, -1],
    ]

    gaussian_elimination(matrix, [5, 1])

    assert matrix == original


def test_gaussian_elimination_does_not_modify_input_vector():
    matrix = [
        [2, 1],
        [1, -1],
    ]

    vector = [5, 1]
    original = [5, 1]

    gaussian_elimination(matrix, vector)

    assert vector == original