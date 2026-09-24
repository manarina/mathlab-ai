import sympy as sp

from core.algebra.systems import (
    calculate_determinant,
    calculate_rank,
    classify_system,
    gaussian_elimination,
    solve_2x2_system,
    solve_3x3_system,
    solve_system,
)


def test_solve_2x2_unique_solution():

    result = solve_2x2_system(
        1,
        1,
        1,
        -1,
        5,
        1,
    )

    assert result["type"] == "unique_solution"

    assert result["solutions"] == [
        (3, 2)
    ]


def test_solve_2x2_infinite_solutions():

    result = solve_2x2_system(
        1,
        1,
        2,
        2,
        2,
        4,
    )

    assert result["type"] == "infinite_solutions"


def test_solve_2x2_no_solution():

    result = solve_2x2_system(
        1,
        1,
        2,
        2,
        2,
        5,
    )

    assert result["type"] == "no_solution"


def test_determinant():

    matrix = sp.Matrix(
        [
            [1, 2],
            [3, 4],
        ]
    )

    assert calculate_determinant(
        matrix
    ) == -2


def test_rank():

    matrix = sp.Matrix(
        [
            [1, 2],
            [2, 4],
        ]
    )

    assert calculate_rank(
        matrix
    ) == 1


def test_classify_unique_system():

    matrix = sp.Matrix(
        [
            [1, 0],
            [0, 1],
        ]
    )

    constants = sp.Matrix(
        [2, 3]
    )

    result = classify_system(
        matrix,
        constants,
    )

    assert result["type"] == (
        "unique_solution"
    )


def test_solve_3x3_system():

    result = solve_3x3_system(
        [
            [1, 1, 1],
            [2, -1, 1],
            [1, 2, -1],
        ],
        [
            6,
            3,
            2,
        ],
    )

    assert result["type"] == (
        "unique_solution"
    )

    assert result["solutions"] == [
        (1, 2, 3)
    ]


def test_gaussian_elimination():

    result = gaussian_elimination(
        [
            [1, 1],
            [1, -1],
        ],
        [
            5,
            1,
        ],
    )

    assert len(
        result["steps"]
    ) >= 2

    assert (
        result["final_matrix"].shape
        == (2, 3)
    )