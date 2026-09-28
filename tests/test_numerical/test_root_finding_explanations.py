from __future__ import annotations

import math

import pytest

from core.numerical.root_finding import (
    RootFindingResult,
    bisection,
    newton_raphson,
    secant,
)
from core.numerical.root_finding_explanations import (
    _format_number,
    _format_result,
    _validate_result,
    explain_bisection,
    explain_bisection_method,
    explain_newton_method,
    explain_newton_raphson,
    explain_root_finding,
    explain_root_finding_comparison,
    explain_secant,
    explain_secant_method,
)


# ============================================================
# FONCTIONS DE TEST
# ============================================================


def quadratic(x: float) -> float:
    return x**2 - 2


def quadratic_derivative(x: float) -> float:
    return 2 * x


def cubic(x: float) -> float:
    return x**3 - x - 2


def cubic_derivative(x: float) -> float:
    return 3 * x**2 - 1


# ============================================================
# _format_number
# ============================================================


def test_format_number_formats_integer():
    assert _format_number(5) == "5"


def test_format_number_formats_float():
    assert _format_number(3.1415926535) == "3.1415926535"


def test_format_number_removes_trailing_zeros():
    assert _format_number(3.1400000000) == "3.14"


def test_format_number_formats_zero():
    assert _format_number(0) == "0"


def test_format_number_formats_negative_number():
    assert _format_number(-2.5) == "-2.5"


def test_format_number_accepts_custom_decimal_count():
    assert _format_number(3.1415926535, decimals=3) == "3.142"


def test_format_number_rejects_boolean():
    with pytest.raises(TypeError):
        _format_number(True)


def test_format_number_rejects_non_numeric_value():
    with pytest.raises(TypeError):
        _format_number("3.14")


def test_format_number_rejects_nan():
    with pytest.raises(ValueError):
        _format_number(float("nan"))


def test_format_number_rejects_infinity():
    with pytest.raises(ValueError):
        _format_number(float("inf"))


# ============================================================
# _format_result
# ============================================================


def test_format_result_contains_root():
    result = RootFindingResult(
        root=math.sqrt(2),
        iterations=5,
        error=1e-8,
    )

    text = _format_result(result)

    assert "Racine approchée" in text
    assert "1.4142135624" in text


def test_format_result_contains_iterations():
    result = RootFindingResult(
        root=1.5,
        iterations=7,
        error=1e-9,
    )

    text = _format_result(result)

    assert "Nombre d'itérations : 7" in text


def test_format_result_contains_error():
    result = RootFindingResult(
        root=1.5,
        iterations=7,
        error=1e-9,
    )

    text = _format_result(result)

    assert "Erreur finale" in text
    assert "1.000e-09" in text


def test_format_result_rejects_invalid_result():
    with pytest.raises(TypeError):
        _format_result("invalid")


def test_format_result_rejects_none():
    with pytest.raises(TypeError):
        _format_result(None)


# ============================================================
# _validate_result
# ============================================================


def test_validate_result_accepts_none():
    assert _validate_result(None) is None


def test_validate_result_accepts_root_finding_result():
    result = RootFindingResult(
        root=1.414,
        iterations=5,
        error=1e-5,
    )

    assert _validate_result(result) == result


def test_validate_result_rejects_invalid_result():
    with pytest.raises(TypeError):
        _validate_result("invalid")


# ============================================================
# EXPLAIN BISECTION
# ============================================================


def test_explain_bisection_contains_title():
    text = explain_bisection(
        quadratic,
        1,
        2,
    )

    assert "Méthode de dichotomie" in text


def test_explain_bisection_contains_principle():
    text = explain_bisection(
        quadratic,
        1,
        2,
    )

    assert "changement de signe" in text
    assert "intervalle" in text


def test_explain_bisection_contains_formula():
    text = explain_bisection(
        quadratic,
        1,
        2,
    )

    assert r"m = \frac{a+b}{2}" in text


def test_explain_bisection_contains_initial_interval():
    text = explain_bisection(
        quadratic,
        1,
        2,
    )

    assert "[1, 2]" in text


def test_explain_bisection_contains_result():
    result = bisection(
        quadratic,
        1,
        2,
        tolerance=1e-8,
    )

    text = explain_bisection(
        quadratic,
        1,
        2,
        tolerance=1e-8,
        result=result,
    )

    assert "Racine approchée" in text
    assert "Nombre d'itérations" in text
    assert "Erreur finale" in text


def test_explain_bisection_accepts_precomputed_result():
    result = RootFindingResult(
        root=1.414213562,
        iterations=20,
        error=1e-8,
    )

    text = explain_bisection(
        quadratic,
        1,
        2,
        result=result,
    )

    assert "1.414213562" in text
    assert "Nombre d'itérations : 20" in text


def test_explain_bisection_rejects_non_callable_function():
    with pytest.raises(TypeError):
        explain_bisection(
            "invalid",
            1,
            2,
        )


def test_explain_bisection_rejects_invalid_result():
    with pytest.raises(TypeError):
        explain_bisection(
            quadratic,
            1,
            2,
            result="invalid",
        )


def test_explain_bisection_method_matches_main_function():
    result = bisection(
        quadratic,
        1,
        2,
    )

    text_main = explain_bisection(
        quadratic,
        1,
        2,
        result=result,
    )

    text_alias = explain_bisection_method(
        quadratic,
        1,
        2,
        result=result,
    )

    assert text_alias == text_main


# ============================================================
# EXPLAIN NEWTON-RAPHSON
# ============================================================


def test_explain_newton_raphson_contains_title():
    text = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
    )

    assert "Newton-Raphson" in text


def test_explain_newton_raphson_contains_principle():
    text = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
    )

    assert "dérivée" in text
    assert "tangente" in text


def test_explain_newton_raphson_contains_formula():
    text = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
    )

    assert r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}" in text


def test_explain_newton_raphson_contains_initial_guess():
    text = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
    )

    assert "x_0 = 1.5" in text


def test_explain_newton_raphson_contains_result():
    result = newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
        tolerance=1e-8,
    )

    text = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
        tolerance=1e-8,
        result=result,
    )

    assert "Racine approchée" in text
    assert "Nombre d'itérations" in text
    assert "Erreur finale" in text


def test_explain_newton_raphson_accepts_precomputed_result():
    result = RootFindingResult(
        root=math.sqrt(2),
        iterations=5,
        error=1e-8,
    )

    text = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
        result=result,
    )

    assert "1.4142135624" in text
    assert "Nombre d'itérations : 5" in text


def test_explain_newton_raphson_rejects_non_callable_function():
    with pytest.raises(TypeError):
        explain_newton_raphson(
            "invalid",
            quadratic_derivative,
            1.5,
        )


def test_explain_newton_raphson_rejects_non_callable_derivative():
    with pytest.raises(TypeError):
        explain_newton_raphson(
            quadratic,
            "invalid",
            1.5,
        )


def test_explain_newton_raphson_rejects_invalid_result():
    with pytest.raises(TypeError):
        explain_newton_raphson(
            quadratic,
            quadratic_derivative,
            1.5,
            result="invalid",
        )


def test_explain_newton_method_matches_main_function():
    result = newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
    )

    text_main = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
        result=result,
    )

    text_alias = explain_newton_method(
        quadratic,
        quadratic_derivative,
        1.5,
        result=result,
    )

    assert text_alias == text_main


# ============================================================
# EXPLAIN SECANT
# ============================================================


def test_explain_secant_contains_title():
    text = explain_secant(
        quadratic,
        1,
        2,
    )

    assert "Méthode de la sécante" in text


def test_explain_secant_contains_principle():
    text = explain_secant(
        quadratic,
        1,
        2,
    )

    assert "deux approximations initiales" in text
    assert "dérivée" in text


def test_explain_secant_contains_formula():
    text = explain_secant(
        quadratic,
        1,
        2,
    )

    assert r"x_{n+1}" in text
    assert r"f(x_n)-f(x_{n-1})" in text


def test_explain_secant_contains_initial_points():
    text = explain_secant(
        quadratic,
        1,
        2,
    )

    assert "(1, -1)" in text
    assert "(2, 2)" in text


def test_explain_secant_contains_result():
    result = secant(
        quadratic,
        1,
        2,
        tolerance=1e-8,
    )

    text = explain_secant(
        quadratic,
        1,
        2,
        tolerance=1e-8,
        result=result,
    )

    assert "Racine approchée" in text
    assert "Nombre d'itérations" in text
    assert "Erreur finale" in text


def test_explain_secant_accepts_precomputed_result():
    result = RootFindingResult(
        root=math.sqrt(2),
        iterations=6,
        error=1e-8,
    )

    text = explain_secant(
        quadratic,
        1,
        2,
        result=result,
    )

    assert "1.4142135624" in text
    assert "Nombre d'itérations : 6" in text


def test_explain_secant_rejects_non_callable_function():
    with pytest.raises(TypeError):
        explain_secant(
            "invalid",
            1,
            2,
        )


def test_explain_secant_rejects_invalid_result():
    with pytest.raises(TypeError):
        explain_secant(
            quadratic,
            1,
            2,
            result="invalid",
        )


def test_explain_secant_method_matches_main_function():
    result = secant(
        quadratic,
        1,
        2,
    )

    text_main = explain_secant(
        quadratic,
        1,
        2,
        result=result,
    )

    text_alias = explain_secant_method(
        quadratic,
        1,
        2,
        result=result,
    )

    assert text_alias == text_main


# ============================================================
# COMPARAISON
# ============================================================


def test_explain_root_finding_comparison_contains_title():
    text = explain_root_finding_comparison(
        quadratic,
        quadratic_derivative,
        lower=1,
        upper=2,
        initial_guess=1.5,
        first_guess=1,
        second_guess=2,
    )

    assert "Comparaison des méthodes" in text


def test_explain_root_finding_comparison_contains_all_methods():
    text = explain_root_finding_comparison(
        quadratic,
        quadratic_derivative,
        lower=1,
        upper=2,
        initial_guess=1.5,
        first_guess=1,
        second_guess=2,
    )

    assert "Dichotomie" in text
    assert "Newton-Raphson" in text
    assert "Sécante" in text


def test_explain_root_finding_comparison_contains_comparison_table():
    text = explain_root_finding_comparison(
        quadratic,
        quadratic_derivative,
        lower=1,
        upper=2,
        initial_guess=1.5,
        first_guess=1,
        second_guess=2,
    )

    assert "| Méthode |" in text
    assert "Dérivée nécessaire ?" in text


def test_explain_root_finding_comparison_contains_results():
    text = explain_root_finding_comparison(
        quadratic,
        quadratic_derivative,
        lower=1,
        upper=2,
        initial_guess=1.5,
        first_guess=1,
        second_guess=2,
    )

    assert "Racine approchée" in text
    assert "Nombre d'itérations" in text
    assert "Erreur finale" in text


def test_explain_root_finding_comparison_rejects_invalid_function():
    with pytest.raises(TypeError):
        explain_root_finding_comparison(
            "invalid",
            quadratic_derivative,
            lower=1,
            upper=2,
            initial_guess=1.5,
            first_guess=1,
            second_guess=2,
        )


def test_explain_root_finding_comparison_rejects_invalid_derivative():
    with pytest.raises(TypeError):
        explain_root_finding_comparison(
            quadratic,
            "invalid",
            lower=1,
            upper=2,
            initial_guess=1.5,
            first_guess=1,
            second_guess=2,
        )


# ============================================================
# EXPLICATION GÉNÉRALE
# ============================================================


def test_explain_root_finding_contains_title():
    text = explain_root_finding()

    assert "Recherche numérique de racines" in text


def test_explain_root_finding_contains_equation():
    text = explain_root_finding()

    assert r"f(x)=0" in text


def test_explain_root_finding_contains_all_methods():
    text = explain_root_finding()

    assert "Dichotomie" in text
    assert "Newton-Raphson" in text
    assert "Sécante" in text


def test_explain_root_finding_contains_tolerance():
    text = explain_root_finding()

    assert "tolérance" in text


def test_explain_root_finding_contains_error():
    text = explain_root_finding()

    assert "erreur" in text


# ============================================================
# COHÉRENCE NUMÉRIQUE DES EXPLICATIONS
# ============================================================


def test_explain_bisection_uses_actual_result():
    result = bisection(
        quadratic,
        1,
        2,
        tolerance=1e-6,
    )

    text = explain_bisection(
        quadratic,
        1,
        2,
        result=result,
    )

    assert _format_number(result.root) in text


def test_explain_newton_uses_actual_result():
    result = newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
        tolerance=1e-6,
    )

    text = explain_newton_raphson(
        quadratic,
        quadratic_derivative,
        1.5,
        result=result,
    )

    assert _format_number(result.root) in text


def test_explain_secant_uses_actual_result():
    result = secant(
        quadratic,
        1,
        2,
        tolerance=1e-6,
    )

    text = explain_secant(
        quadratic,
        1,
        2,
        result=result,
    )

    assert _format_number(result.root) in text