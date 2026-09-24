from core.linear_algebra.linear_systems import (
    create_linear_system,
    solve_linear_system,
)
from core.linear_algebra.matrices import create_matrix
from core.linear_algebra.vectors import create_vector
from core.linear_algebra.system_explanations import (
    explain_linear_system,
    explain_gaussian_elimination,
    explain_solution,
    explain_complete_solution,
)


# ============================================================
# SYSTÈME DE TEST
# ============================================================


def create_test_system():
    coefficients = create_matrix(
        [
            [2, 1],
            [1, -1],
        ]
    )

    constants = create_vector([5, 1])

    return create_linear_system(
        coefficients,
        constants,
    )


# ============================================================
# EXPLICATION DE LA STRUCTURE DU SYSTÈME
# ============================================================


def test_explain_linear_system():
    system = create_test_system()

    explanation = explain_linear_system(system)

    assert isinstance(explanation, str)

    assert "### Système linéaire" in explanation
    assert "2 équations" in explanation
    assert "2 inconnues" in explanation

    assert "AX" in explanation

    assert "matrice des coefficients" in explanation
    assert "vecteur des inconnues" in explanation
    assert "vecteur des constantes" in explanation

    assert "2x" in explanation
    assert "5" in explanation
    assert "x_2" in explanation


# ============================================================
# EXPLICATION DE L'ÉLIMINATION DE GAUSS
# ============================================================


def test_explain_gaussian_elimination():
    system = create_test_system()

    explanation = explain_gaussian_elimination(system)

    assert isinstance(explanation, str)

    assert "### Méthode : élimination de Gauss" in explanation
    assert "élimination de Gauss" in explanation

    assert "[A" in explanation
    assert "B]" in explanation

    assert "échanger deux lignes" in explanation
    assert "multiplier une ligne" in explanation
    assert "ajouter à une ligne" in explanation


# ============================================================
# EXPLICATION D'UNE SOLUTION UNIQUE
# ============================================================


def test_explain_unique_solution():
    system = create_test_system()
    solution = solve_linear_system(system)

    explanation = explain_solution(solution)

    assert isinstance(explanation, str)

    assert "### Résultat" in explanation
    assert "solution unique" in explanation

    assert "x" in explanation
    assert "2" in explanation
    assert "1" in explanation


# ============================================================
# EXPLICATION D'UN SYSTÈME SANS SOLUTION
# ============================================================


def test_explain_no_solution():
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

    solution = solve_linear_system(system)

    explanation = explain_solution(solution)

    assert isinstance(explanation, str)

    assert "aucune solution" in explanation
    assert "incompatible" in explanation
    assert "0 = c" in explanation


# ============================================================
# EXPLICATION D'UNE INFINITÉ DE SOLUTIONS
# ============================================================

def test_explain_infinite_solutions():
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

    solution = solve_linear_system(system)

    explanation = explain_solution(solution)

    assert isinstance(explanation, str)

    assert "une infinité de solutions" in explanation
    assert "pivots" in explanation
    assert "inconnue est donc libre" in explanation


# ============================================================
# EXPLICATION COMPLÈTE
# ============================================================


def test_explain_complete_solution():
    system = create_test_system()
    solution = solve_linear_system(system)

    explanation = explain_complete_solution(
        system,
        solution,
    )

    assert isinstance(explanation, str)

    assert "### Système linéaire" in explanation
    assert "### Méthode : élimination de Gauss" in explanation
    assert "### Résultat" in explanation

    assert "solution unique" in explanation
    assert "2" in explanation
    assert "1" in explanation


# ============================================================
# EXPLICATION D'UN SYSTÈME 3×3
# ============================================================


def test_explain_3x3_system():
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

    explanation = explain_linear_system(system)

    assert isinstance(explanation, str)

    assert "3 équations" in explanation
    assert "3 inconnues" in explanation

    assert "6" in explanation
    assert "3" in explanation
    assert "2" in explanation