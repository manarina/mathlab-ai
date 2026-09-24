import pytest

from core.linear_algebra.linear_systems import (
    LinearSystem,
    LinearSystemSolution,
    create_linear_system,
    solve_linear_system,
)
from core.linear_algebra.matrices import create_matrix
from core.linear_algebra.vectors import create_vector


# ============================================================
# CRÉATION D'UN SYSTÈME
# ============================================================


def test_create_linear_system():
    coefficients = create_matrix(
        [
            [2, 1],
            [1, -1],
        ]
    )

    constants = create_vector([5, 1])

    system = create_linear_system(
        coefficients,
        constants,
    )

    assert isinstance(system, LinearSystem)
    assert system.coefficients == coefficients
    assert system.constants == constants
    assert system.size == 2


def test_linear_system_size():
    coefficients = create_matrix(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    constants = create_vector([1, 2, 3])

    system = create_linear_system(
        coefficients,
        constants,
    )

    assert system.size == 3


# ============================================================
# SOLUTION UNIQUE — SYSTÈME 2×2
# ============================================================


def test_unique_solution_2x2():
    coefficients = create_matrix(
        [
            [2, 1],
            [1, -1],
        ]
    )

    constants = create_vector([5, 1])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert isinstance(result, LinearSystemSolution)
    assert result.solution_type == "solution unique"
    assert result.is_unique
    assert not result.has_no_solution
    assert not result.has_infinite_solutions
    assert result.values == (2, 1)


# ============================================================
# SOLUTION UNIQUE — AUTRE SYSTÈME 2×2
# ============================================================


def test_unique_solution_second_2x2():
    coefficients = create_matrix(
        [
            [1, 1],
            [2, -1],
        ]
    )

    constants = create_vector([5, 4])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.is_unique
    assert result.values == (3, 2)


# ============================================================
# SOLUTION UNIQUE — SYSTÈME 3×3
# ============================================================

def test_unique_solution_3x3():
    coefficients = create_matrix(
        [
            [1, 1, 1],
            [2, -1, 1],
            [1, 2, -1],
        ]
    )

    constants = create_vector([6, 3, 2])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.is_unique
    assert result.values == (1, 2, 3)


# ============================================================
# SOLUTION UNIQUE AVEC DES VALEURS DÉCIMALES
# ============================================================


def test_unique_decimal_solution():
    coefficients = create_matrix(
        [
            [2, 1],
            [1, 2],
        ]
    )

    constants = create_vector([5, 4])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.is_unique
    assert result.values is not None

    assert result.values[0] == pytest.approx(2)
    assert result.values[1] == pytest.approx(1)


# ============================================================
# AUCUNE SOLUTION
# ============================================================


def test_no_solution():
    coefficients = create_matrix(
        [
            [1, 1],
            [2, 2],
        ]
    )

    constants = create_vector([2, 5])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.solution_type == "aucune solution"
    assert result.has_no_solution
    assert not result.is_unique
    assert not result.has_infinite_solutions
    assert result.values is None


# ============================================================
# INFINITÉ DE SOLUTIONS
# ============================================================


def test_infinite_solutions():
    coefficients = create_matrix(
        [
            [1, 1],
            [2, 2],
        ]
    )

    constants = create_vector([2, 4])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.solution_type == "une infinité de solutions"
    assert result.has_infinite_solutions
    assert not result.is_unique
    assert not result.has_no_solution
    assert result.values is None


# ============================================================
# MATRICE DES COEFFICIENTS NON CARRÉE
# ============================================================


def test_non_square_coefficients():
    coefficients = create_matrix(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    constants = create_vector([7, 8])

    with pytest.raises(ValueError, match="carrée"):
        create_linear_system(
            coefficients,
            constants,
        )


# ============================================================
# DIMENSIONS INCOMPATIBLES
# ============================================================


def test_incompatible_dimensions():
    coefficients = create_matrix(
        [
            [1, 2],
            [3, 4],
        ]
    )

    constants = create_vector([1, 2, 3])

    with pytest.raises(ValueError, match="équations"):
        create_linear_system(
            coefficients,
            constants,
        )


# ============================================================
# SYSTÈME 1×1
# ============================================================


def test_unique_solution_1x1():
    coefficients = create_matrix(
        [
            [5],
        ]
    )

    constants = create_vector([15])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.is_unique
    assert result.values == (3,)


# ============================================================
# SOLUTION NÉGATIVE
# ============================================================


def test_negative_solution():
    coefficients = create_matrix(
        [
            [1, 1],
            [2, -1],
        ]
    )

    constants = create_vector([-1, -8])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.is_unique
    assert result.values == (-3, 2)


# ============================================================
# SOLUTION AVEC PERMUTATION DE PIVOT
# ============================================================


def test_pivot_row_swap():
    coefficients = create_matrix(
        [
            [0, 2],
            [1, 1],
        ]
    )

    constants = create_vector([4, 3])

    system = create_linear_system(
        coefficients,
        constants,
    )

    result = solve_linear_system(system)

    assert result.is_unique
    assert result.values == (1, 2)