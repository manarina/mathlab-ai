"""
Tests du moteur AI Math Assistant.
"""

import pytest
import sympy as sp

from core.ai.math_assistant import (
    calculate_derivative,
    calculate_expression,
    calculate_integral,
    calculate_limit,
    detect_operation,
    extract_equation,
    format_result,
    math_assistant,
    normalize_expression,
    parse_expression,
    simplify_expression,
    solve_equation,
)


# ============================================================
# VALIDATION
# ============================================================


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        math_assistant("")


def test_non_string_question_is_rejected():
    with pytest.raises(TypeError):
        math_assistant(123)


# ============================================================
# NORMALISATION
# ============================================================


def test_normalize_power():
    assert normalize_expression(
        "x^2"
    ) == "x**2"


def test_normalize_implicit_multiplication():
    assert normalize_expression(
        "2x"
    ) == "2*x"


def test_normalize_pi():
    assert normalize_expression(
        "pi"
    ) == "pi"


# ============================================================
# PARSING
# ============================================================


def test_parse_expression():
    expression = parse_expression(
        "x^2 + 2*x + 1"
    )

    x = sp.Symbol("x")

    assert expression == (
        x**2 + 2*x + 1
    )


def test_invalid_expression():
    with pytest.raises(ValueError):
        parse_expression(
            "x ++++"
        )


# ============================================================
# DÉTECTION
# ============================================================


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Résous 2*x + 5 = 15",
            "equation",
        ),
        (
            "Calcule la dérivée de x^2",
            "derivative",
        ),
        (
            "Calcule l'intégrale de x^2",
            "integral",
        ),
        (
            "Calcule la limite de sin(x)/x quand x tend vers 0",
            "limit",
        ),
        (
            "Simplifie (x+1)^2",
            "simplify",
        ),
        (
            "Calcule 2 + 3",
            "calculate",
        ),
    ],
)
def test_detect_operation(
    question,
    expected,
):
    assert detect_operation(
        question
    ) == expected


# ============================================================
# ÉQUATION
# ============================================================


def test_extract_equation():
    left, right = extract_equation(
        "Résous 2*x + 5 = 15"
    )

    assert left == "2*x + 5"
    assert right == "15"


def test_solve_linear_equation():
    result = solve_equation(
        "Résous 2*x + 5 = 15"
    )

    x = sp.Symbol("x")

    assert result["operation"] == "equation"
    assert result["result"] == [5]
    assert result["variable"] == x


def test_solve_quadratic_equation():
    result = solve_equation(
        "Résous x^2 - 5*x + 6 = 0"
    )

    assert set(result["result"]) == {
        sp.Integer(2),
        sp.Integer(3),
    }


def test_equation_contains_steps():
    result = solve_equation(
        "Résous 3*x - 6 = 0"
    )

    assert len(
        result["steps"]
    ) >= 2


# ============================================================
# DÉRIVÉE
# ============================================================


def test_derivative_polynomial():
    result = calculate_derivative(
        "Calcule la dérivée de x^3 + 2*x"
    )

    x = sp.Symbol("x")

    assert result["result"] == (
        3*x**2 + 2
    )


def test_derivative_constant():
    result = calculate_derivative(
        "Calcule la dérivée de 5"
    )

    assert result["result"] == 0


def test_derivative_trigonometric():
    result = calculate_derivative(
        "Calcule la dérivée de sin(x)"
    )

    x = sp.Symbol("x")

    assert result["result"] == sp.cos(x)


# ============================================================
# INTÉGRALE
# ============================================================


def test_indefinite_integral():
    result = calculate_integral(
        "Calcule l'intégrale de x^2"
    )

    x = sp.Symbol("x")

    assert sp.diff(
        result["result"],
        x,
    ) == x**2


def test_definite_integral():
    result = calculate_integral(
        "Calcule l'intégrale de x^2 de 0 à 2"
    )

    assert result["result"] == sp.Rational(
        8,
        3,
    )


def test_integral_contains_steps():
    result = calculate_integral(
        "Calcule l'intégrale de 2*x"
    )

    assert len(
        result["steps"]
    ) >= 2


# ============================================================
# LIMITES
# ============================================================


def test_limit_sin_x_over_x():
    result = calculate_limit(
        "Calcule la limite de sin(x)/x quand x tend vers 0"
    )

    assert result["result"] == 1


def test_limit_polynomial():
    result = calculate_limit(
        "Calcule la limite de x^2 quand x tend vers 2"
    )

    assert result["result"] == 4


def test_limit_at_infinity():
    result = calculate_limit(
        "Calcule la limite de 1/x quand x tend vers +inf"
    )

    assert result["result"] == 0


# ============================================================
# SIMPLIFICATION
# ============================================================


def test_simplify_expression():
    result = simplify_expression(
        "Simplifie (x + 1)^2"
    )

    x = sp.Symbol("x")

    assert sp.expand(
        result["result"]
    ) == (
        x**2 + 2*x + 1
    )


def test_simplify_fraction():
    result = simplify_expression(
        "Simplifie (x^2 - 1)/(x - 1)"
    )

    x = sp.Symbol("x")

    assert result["result"] == (
        x + 1
    )


# ============================================================
# CALCUL
# ============================================================


def test_calculate_numeric_expression():
    result = calculate_expression(
        "Calcule 2 + 3"
    )

    assert result["result"] == 5


def test_calculate_fraction():
    result = calculate_expression(
        "Calcule 1/2 + 1/2"
    )

    assert result["result"] == 1


# ============================================================
# MOTEUR PRINCIPAL
# ============================================================


def test_math_assistant_equation():
    result = math_assistant(
        "Résous 2*x + 5 = 15"
    )

    assert result["operation"] == "equation"
    assert result["result"] == [5]
    assert "question" in result
    assert "steps" in result
    assert "explanation" in result


def test_math_assistant_derivative():
    result = math_assistant(
        "Calcule la dérivée de x^2"
    )

    x = sp.Symbol("x")

    assert result["operation"] == "derivative"
    assert result["result"] == 2*x


def test_math_assistant_integral():
    result = math_assistant(
        "Calcule l'intégrale de x^2"
    )

    x = sp.Symbol("x")

    assert result["operation"] == "integral"

    assert sp.diff(
        result["result"],
        x,
    ) == x**2


def test_math_assistant_limit():
    result = math_assistant(
        "Calcule la limite de sin(x)/x quand x tend vers 0"
    )

    assert result["operation"] == "limit"
    assert result["result"] == 1


# ============================================================
# FORMATAGE
# ============================================================


def test_format_result():
    result = math_assistant(
        "Résous 2*x + 5 = 15"
    )

    text = format_result(
        result
    )

    assert "Équation" in text
    assert "Démarche" in text
    assert "Résultat" in text
    assert "Explication" in text


# ============================================================
# COHÉRENCE DES RÉSULTATS
# ============================================================


@pytest.mark.parametrize(
    "question",
    [
        "Résous 2*x + 5 = 15",
        "Calcule la dérivée de x^2",
        "Calcule l'intégrale de x^2",
        "Calcule la limite de sin(x)/x quand x tend vers 0",
        "Simplifie (x + 1)^2",
        "Calcule 2 + 3",
    ],
)
def test_math_assistant_returns_standard_structure(
    question,
):
    result = math_assistant(
        question
    )

    assert isinstance(
        result,
        dict,
    )

    assert "question" in result
    assert "operation" in result
    assert "result" in result
    assert "steps" in result
    assert "explanation" in result

    assert result["operation"] in (
        "equation",
        "derivative",
        "integral",
        "limit",
        "simplify",
        "calculate",
    )

