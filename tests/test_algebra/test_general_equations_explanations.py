import sympy as sp

from core.algebra.general_equations import (
    parse_polynomial_equation,
    solve_polynomial_equation,
)

from core.algebra.general_equations_explanations import (
    explain_general_polynomial_equation,
)


# ============================================================
# ÉQUATION DU PREMIER DEGRÉ
# ============================================================


def test_explain_general_linear_equation():
    polynomial = parse_polynomial_equation(
        "2*x - 4"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    assert len(steps) >= 7

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier l'équation" in titles
    assert "Déterminer le degré" in titles
    assert "Identifier les coefficients" in titles
    assert "Choisir la méthode de résolution" in titles
    assert "Factoriser le polynôme" in titles
    assert "Rechercher les racines exactes" in titles
    assert "Solutions exactes" in titles
    assert "Conclusion" in titles


def test_explain_general_linear_degree():
    polynomial = parse_polynomial_equation(
        "2*x - 4"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    degree_step = next(
        step
        for step in steps
        if step["title"] == "Déterminer le degré"
    )

    assert "1" in degree_step["formula"]


def test_explain_general_linear_solution():
    polynomial = parse_polynomial_equation(
        "2*x - 4"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    solution_step = next(
        step
        for step in steps
        if step["title"] == "Solutions exactes"
    )

    assert "2" in solution_step["formula"]


# ============================================================
# ÉQUATION DU SECOND DEGRÉ
# ============================================================


def test_explain_general_quadratic_equation():
    polynomial = parse_polynomial_equation(
        "x**2 - 5*x + 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier l'équation" in titles
    assert "Déterminer le degré" in titles
    assert "Identifier les coefficients" in titles
    assert "Choisir la méthode de résolution" in titles
    assert "Factoriser le polynôme" in titles
    assert "Solutions exactes" in titles
    assert "Conclusion" in titles


def test_explain_general_quadratic_degree():
    polynomial = parse_polynomial_equation(
        "x**2 - 5*x + 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    degree_step = next(
        step
        for step in steps
        if step["title"] == "Déterminer le degré"
    )

    assert "2" in degree_step["formula"]


def test_explain_general_quadratic_solutions():
    polynomial = parse_polynomial_equation(
        "x**2 - 5*x + 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    solution_step = next(
        step
        for step in steps
        if step["title"] == "Solutions exactes"
    )

    assert "1" in solution_step["formula"]
    assert "2" in solution_step["formula"]


# ============================================================
# ÉQUATION DU TROISIÈME DEGRÉ
# ============================================================


def test_explain_general_cubic_equation():
    polynomial = parse_polynomial_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier l'équation" in titles
    assert "Déterminer le degré" in titles
    assert "Identifier les coefficients" in titles
    assert "Choisir la méthode de résolution" in titles
    assert "Factoriser le polynôme" in titles
    assert "Solutions exactes" in titles
    assert "Conclusion" in titles


def test_explain_general_cubic_degree():
    polynomial = parse_polynomial_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    degree_step = next(
        step
        for step in steps
        if step["title"] == "Déterminer le degré"
    )

    assert "3" in degree_step["formula"]


def test_explain_general_cubic_method():
    polynomial = parse_polynomial_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    method_step = next(
        step
        for step in steps
        if step["title"]
        == "Choisir la méthode de résolution"
    )

    assert "Horner" in method_step["formula"]
    assert "Cardano" in method_step["formula"]


def test_explain_general_cubic_solutions():
    polynomial = parse_polynomial_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    solution_step = next(
        step
        for step in steps
        if step["title"] == "Solutions exactes"
    )

    assert "1" in solution_step["formula"]
    assert "2" in solution_step["formula"]
    assert "3" in solution_step["formula"]


# ============================================================
# ÉQUATION DU QUATRIÈME DEGRÉ
# ============================================================


def test_explain_general_quartic_equation():
    polynomial = parse_polynomial_equation(
        "x**4 - 5*x**2 + 4"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier l'équation" in titles
    assert "Déterminer le degré" in titles
    assert "Identifier les coefficients" in titles
    assert "Choisir la méthode de résolution" in titles
    assert "Factoriser le polynôme" in titles
    assert "Solutions exactes" in titles
    assert "Conclusion" in titles


def test_explain_general_quartic_method():
    polynomial = parse_polynomial_equation(
        "x**4 - 5*x**2 + 4"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    method_step = next(
        step
        for step in steps
        if step["title"]
        == "Choisir la méthode de résolution"
    )

    assert "quatrième degré" in method_step["formula"]


def test_explain_general_quartic_solutions():
    polynomial = parse_polynomial_equation(
        "x**4 - 5*x**2 + 4"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    solution_step = next(
        step
        for step in steps
        if step["title"] == "Solutions exactes"
    )

    assert "1" in solution_step["formula"]
    assert "-1" in solution_step["formula"]
    assert "2" in solution_step["formula"]
    assert "-2" in solution_step["formula"]


# ============================================================
# DEGRÉ SUPÉRIEUR OU ÉGAL À 5
# ============================================================


def test_explain_general_high_degree_equation():
    polynomial = parse_polynomial_equation(
        "x**5 - 5*x**4 + 4*x**3"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier l'équation" in titles
    assert "Déterminer le degré" in titles
    assert "Identifier les coefficients" in titles
    assert "Choisir la méthode de résolution" in titles
    assert "Factoriser le polynôme" in titles
    assert "Rechercher les racines exactes" in titles
    assert "Solutions exactes" in titles
    assert "Calculer les solutions numériques" in titles
    assert "Conclusion" in titles


def test_explain_general_high_degree_method():
    polynomial = parse_polynomial_equation(
        "x**5 - 5*x**4 + 4*x**3"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    method_step = next(
        step
        for step in steps
        if step["title"]
        == "Choisir la méthode de résolution"
    )

    assert (
        "Factorisation et méthodes numériques"
        in method_step["formula"]
    )


def test_explain_general_high_degree_numerical_roots():
    polynomial = parse_polynomial_equation(
        "x**5 - 5*x**4 + 4*x**3"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    numerical_step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer les solutions numériques"
    )

    assert "x_{1}" in numerical_step["formula"]


# ============================================================
# STRUCTURE DES ÉTAPES
# ============================================================


def test_explanation_steps_have_required_fields():
    polynomial = parse_polynomial_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    for step in steps:
        assert "title" in step
        assert "formula" in step
        assert "explanation" in step

        assert isinstance(
            step["title"],
            str,
        )

        assert isinstance(
            step["formula"],
            str,
        )

        assert isinstance(
            step["explanation"],
            str,
        )


def test_explanation_contains_no_empty_content():
    polynomial = parse_polynomial_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    steps = explain_general_polynomial_equation(
        result
    )

    for step in steps:
        assert step["title"].strip()
        assert step["formula"].strip()
        assert step["explanation"].strip()