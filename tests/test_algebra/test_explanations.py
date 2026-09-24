from core.algebra.equations import (
    solve_linear_equation,
    solve_quadratic_equation,
)

from core.algebra.explanations import (
    explain_linear_equation,
    explain_quadratic_equation,
)


# ============================================================
# EQUATION DU PREMIER DEGRE
# ============================================================


def test_linear_explanation_contains_main_steps():
    result = solve_linear_equation(2, -10)

    steps = explain_linear_equation(
        2,
        -10,
        result,
    )

    assert len(steps) >= 4

    titles = [step["title"] for step in steps]

    assert "Identifier la forme de l'équation" in titles
    assert "Remplacer les coefficients" in titles
    assert "Isoler le terme contenant x" in titles
    assert "Solution" in titles


def test_linear_explanation_solution():
    result = solve_linear_equation(2, -10)

    steps = explain_linear_equation(
        2,
        -10,
        result,
    )

    assert steps[-1]["formula"] == r"x = 5"


def test_linear_explanation_no_solution():
    result = solve_linear_equation(0, 5)

    steps = explain_linear_equation(
        0,
        5,
        result,
    )

    assert len(steps) >= 3

    assert "aucune solution" in steps[-1]["explanation"].lower()


def test_linear_explanation_infinite_solutions():
    result = solve_linear_equation(0, 0)

    steps = explain_linear_equation(
        0,
        0,
        result,
    )

    assert len(steps) >= 3

    assert "tout nombre réel" in steps[-1]["explanation"].lower()


# ============================================================
# EQUATION DU SECOND DEGRE
# ============================================================


def test_quadratic_explanation_positive_delta():
    result = solve_quadratic_equation(
        1,
        -5,
        6,
    )

    steps = explain_quadratic_equation(
        1,
        -5,
        6,
        result,
    )

    titles = [step["title"] for step in steps]

    assert "Identifier la forme de l'équation" in titles
    assert "Identifier les coefficients" in titles
    assert "Calculer le discriminant" in titles
    assert "Interpréter le discriminant" in titles
    assert "Utiliser les formules" in titles
    assert "Calculer les solutions" in titles


def test_quadratic_explanation_positive_delta_values():
    result = solve_quadratic_equation(
        1,
        -5,
        6,
    )

    steps = explain_quadratic_equation(
        1,
        -5,
        6,
        result,
    )

    assert steps[-1]["formula"] == r"x_1 = 2\qquad x_2 = 3"


def test_quadratic_explanation_zero_delta():
    result = solve_quadratic_equation(
        1,
        -4,
        4,
    )

    steps = explain_quadratic_equation(
        1,
        -4,
        4,
        result,
    )

    titles = [step["title"] for step in steps]

    assert "Interpréter le discriminant" in titles
    assert "Utiliser la formule" in titles
    assert "Calculer la solution" in titles

    assert steps[-1]["formula"] == r"x = 2"


def test_quadratic_explanation_negative_delta():
    result = solve_quadratic_equation(
        1,
        0,
        1,
    )

    steps = explain_quadratic_equation(
        1,
        0,
        1,
        result,
    )

    assert steps[-1]["formula"] == r"S = \varnothing"


def test_quadratic_explanation_a_zero():
    result = solve_quadratic_equation(
        0,
        2,
        -10,
    )

    steps = explain_quadratic_equation(
        0,
        2,
        -10,
        result,
    )

    assert result["degree"] == 1

    titles = [step["title"] for step in steps]

    assert "Vérifier le coefficient a" in titles
    assert "Nouvelle forme" in titles