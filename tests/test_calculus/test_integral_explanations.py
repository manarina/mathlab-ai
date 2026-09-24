"""
Tests for core.calculus.integral_explanations
"""

import sympy as sp
import pytest

from core.calculus.integral_explanations import (
    calculate_integral,
    classify_integrand,
    explain_integral,
    format_expression,
    integrate_definite,
    integrate_indefinite,
    parse_expression,
    verify_integral,
)


# ============================================================
# PARSING
# ============================================================


def test_parse_expression():
    expr, x = parse_expression("x**2 + 3*x")

    assert x == sp.Symbol("x")
    assert expr == x**2 + 3 * x


def test_parse_empty_expression():
    with pytest.raises(ValueError):
        parse_expression("")


def test_parse_invalid_expression():
    with pytest.raises(ValueError):
        parse_expression("x***2")


# ============================================================
# FORMATTING
# ============================================================


def test_format_expression():
    x = sp.Symbol("x")

    result = format_expression(x**2)

    assert result == "x^{2}"


# ============================================================
# CLASSIFICATION
# ============================================================


def test_classify_constant():
    x = sp.Symbol("x")

    assert classify_integrand(sp.Integer(5), x) == "constant"


def test_classify_polynomial():
    x = sp.Symbol("x")

    assert classify_integrand(x**3 + 2*x, x) == "polynomial"


def test_classify_exponential():
    x = sp.Symbol("x")

    assert classify_integrand(sp.exp(x), x) == "exponential"


def test_classify_trigonometric():
    x = sp.Symbol("x")

    assert classify_integrand(sp.sin(x), x) == "trigonometric"


# ============================================================
# INDEFINITE INTEGRALS
# ============================================================


def test_integral_x_squared():
    result = integrate_indefinite("x**2")

    x = sp.Symbol("x")

    assert sp.simplify(
        result.integral - (x**3 / 3 + sp.Symbol("C"))
    ) == 0


def test_integral_linear():
    result = integrate_indefinite("2*x + 3")

    x = sp.Symbol("x")

    expected = x**2 + 3*x + sp.Symbol("C")

    assert sp.simplify(result.integral - expected) == 0


def test_integral_constant():
    result = integrate_indefinite("5")

    x = sp.Symbol("x")

    expected = 5*x + sp.Symbol("C")

    assert sp.simplify(result.integral - expected) == 0


def test_integral_exponential():
    result = integrate_indefinite("exp(x)")

    x = sp.Symbol("x")

    expected = sp.exp(x) + sp.Symbol("C")

    assert sp.simplify(result.integral - expected) == 0


def test_integral_sine():
    result = integrate_indefinite("sin(x)")

    x = sp.Symbol("x")

    expected = -sp.cos(x) + sp.Symbol("C")

    assert sp.simplify(result.integral - expected) == 0


def test_integral_cosine():
    result = integrate_indefinite("cos(x)")

    x = sp.Symbol("x")

    expected = sp.sin(x) + sp.Symbol("C")

    assert sp.simplify(result.integral - expected) == 0


# ============================================================
# DEFINITE INTEGRALS
# ============================================================


def test_definite_integral_x_squared():
    result = integrate_definite("x**2", 0, 2)

    assert result.integral == sp.Rational(8, 3)


def test_definite_integral_linear():
    result = integrate_definite("2*x + 1", 0, 3)

    assert result.integral == 12


def test_definite_integral_constant():
    result = integrate_definite("5", 0, 2)

    assert result.integral == 10


def test_definite_integral_sine():
    result = integrate_definite("sin(x)", 0, sp.pi)

    assert result.integral == 2


# ============================================================
# UNIFIED API
# ============================================================


def test_calculate_indefinite_integral():
    result = calculate_integral("x**2")

    assert result.is_definite is False


def test_calculate_definite_integral():
    result = calculate_integral(
        "x**2",
        lower_bound=0,
        upper_bound=2,
    )

    assert result.is_definite is True
    assert result.integral == sp.Rational(8, 3)


def test_calculate_missing_upper_bound():
    with pytest.raises(ValueError):
        calculate_integral(
            "x**2",
            lower_bound=0,
        )


def test_calculate_missing_lower_bound():
    with pytest.raises(ValueError):
        calculate_integral(
            "x**2",
            upper_bound=2,
        )


# ============================================================
# VERIFICATION
# ============================================================


def test_verify_integral_x_squared():
    x = sp.Symbol("x")

    primitive = x**3 / 3

    assert verify_integral(
        "x**2",
        primitive,
    ) is True


def test_verify_integral_sine():
    x = sp.Symbol("x")

    primitive = -sp.cos(x)

    assert verify_integral(
        "sin(x)",
        primitive,
    ) is True


def test_verify_wrong_integral():
    x = sp.Symbol("x")

    wrong_primitive = x**2

    assert verify_integral(
        "x**2",
        wrong_primitive,
    ) is False


# ============================================================
# EXPLANATION API
# ============================================================


def test_explain_indefinite_integral():
    result = explain_integral("x**2")

    assert "integral" in result
    assert "integral_latex" in result
    assert "steps" in result
    assert "verification_ok" in result

    assert result["verification_ok"] is True


def test_explain_definite_integral():
    result = explain_integral(
        "x**2",
        lower_bound=0,
        upper_bound=2,
    )

    assert result["is_definite"] is True
    assert result["integral"] == sp.Rational(8, 3)
    assert result["verification_ok"] is True


# ============================================================
# CUSTOM VARIABLE
# ============================================================


def test_custom_variable():
    result = integrate_indefinite(
        "t**2",
        variable="t",
    )

    t = sp.Symbol("t")

    expected = t**3 / 3 + sp.Symbol("C")

    assert sp.simplify(result.integral - expected) == 0


# ============================================================
# STEP GENERATION
# ============================================================


def test_steps_are_generated():
    result = integrate_indefinite("x**2 + 2*x")

    assert len(result.steps) >= 3


def test_definite_steps_are_generated():
    result = integrate_definite("x**2", 0, 2)

    assert len(result.steps) >= 4