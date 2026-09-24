import sympy as sp

from core.algebra.system_explanations import (
    explain_system,
)
from core.algebra.systems import (
    solve_2x2_system,
    solve_3x3_system,
    solve_system,
)


def test_explain_unique_2x2_system():

    matrix = [
        [1, 1],
        [1, -1],
    ]

    constants = [
        5,
        1,
    ]

    result = solve_2x2_system(
        1,
        1,
        1,
        -1,
        5,
        1,
    )

    steps = explain_system(
        matrix,
        constants,
        result,
    )

    assert len(steps) >= 6

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier le système" in titles
    assert "Écrire la matrice augmentée" in titles
    assert "Analyser le système" in titles
    assert "Appliquer l'élimination de Gauss" in titles
    assert "Lire les solutions" in titles
    assert "Vérifier les solutions" in titles


def test_explain_unique_3x3_system():

    matrix = [
        [1, 1, 1],
        [2, -1, 1],
        [1, 2, -1],
    ]

    constants = [
        6,
        3,
        2,
    ]

    result = solve_3x3_system(
        matrix,
        constants,
    )

    steps = explain_system(
        matrix,
        constants,
        result,
    )

    assert result["solutions"] == [
        (1, 2, 3)
    ]

    assert len(steps) >= 8

    solution_step = next(
        step
        for step in steps
        if step["title"] == "Lire les solutions"
    )

    assert "x = 1" in solution_step["formula"]
    assert "y = 2" in solution_step["formula"]
    assert "z = 3" in solution_step["formula"]


def test_explain_infinite_system():

    matrix = [
        [1, 1],
        [2, 2],
    ]

    constants = [
        2,
        4,
    ]

    result = solve_system(
        matrix,
        constants,
    )

    steps = explain_system(
        matrix,
        constants,
        result,
    )

    assert result["type"] == (
        "infinite_solutions"
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Décrire les solutions" in titles


def test_explain_incompatible_system():

    matrix = [
        [1, 1],
        [2, 2],
    ]

    constants = [
        2,
        5,
    ]

    result = solve_system(
        matrix,
        constants,
    )

    steps = explain_system(
        matrix,
        constants,
        result,
    )

    assert result["type"] == (
        "no_solution"
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Conclusion" in titles


def test_explain_augmented_matrix():

    matrix = sp.Matrix(
        [
            [1, 2],
            [3, 4],
        ]
    )

    constants = sp.Matrix(
        [5, 6]
    )

    result = solve_system(
        matrix,
        constants,
    )

    steps = explain_system(
        matrix,
        constants,
        result,
    )

    matrix_step = next(
        step
        for step in steps
        if step["title"]
        == "Écrire la matrice augmentée"
    )

    assert matrix_step["formula"]

    assert (
        "\\left[" in matrix_step["formula"]
        or "\\begin{bmatrix}"
        in matrix_step["formula"]
    )

    assert "5" in matrix_step["formula"]
    assert "6" in matrix_step["formula"]