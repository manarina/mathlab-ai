
"""
Tests du module core.optimization.objectives.
"""

import math

import pytest

from core.optimization.objectives import (
    evaluate_objective,
    is_valid_objective,
    objective_name,
    validate_objective,
)


# ============================================================
# validate_objective
# ============================================================


def test_validate_objective_accepts_callable():
    function = lambda x: x**2

    result = validate_objective(function)

    assert result is function


def test_validate_objective_rejects_non_callable():
    with pytest.raises(TypeError, match="appelable"):
        validate_objective(42)


def test_validate_objective_rejects_string():
    with pytest.raises(TypeError, match="appelable"):
        validate_objective("x**2")


# ============================================================
# is_valid_objective
# ============================================================


def test_is_valid_objective_returns_true_for_function():
    assert is_valid_objective(lambda x: x**2) is True


def test_is_valid_objective_returns_true_for_callable_object():
    class Objective:
        def __call__(self, x):
            return x**2

    assert is_valid_objective(Objective()) is True


def test_is_valid_objective_returns_false_for_number():
    assert is_valid_objective(10) is False


def test_is_valid_objective_returns_false_for_string():
    assert is_valid_objective("x**2") is False


# ============================================================
# evaluate_objective — cas de base
# ============================================================


def test_evaluate_objective_returns_correct_value():
    function = lambda x: x**2

    result = evaluate_objective(function, 3)

    assert result == pytest.approx(9.0)


def test_evaluate_objective_accepts_integer():
    function = lambda x: x + 2

    result = evaluate_objective(function, 3)

    assert result == pytest.approx(5.0)


def test_evaluate_objective_accepts_float():
    function = lambda x: x**2

    result = evaluate_objective(function, 1.5)

    assert result == pytest.approx(2.25)


def test_evaluate_objective_returns_float():
    function = lambda x: x**2

    result = evaluate_objective(function, 2)

    assert isinstance(result, float)


def test_evaluate_objective_supports_negative_values():
    function = lambda x: x**2

    result = evaluate_objective(function, -4)

    assert result == pytest.approx(16.0)


# ============================================================
# Validation de x
# ============================================================


def test_evaluate_objective_rejects_non_callable():
    with pytest.raises(TypeError, match="appelable"):
        evaluate_objective(42, 2)


def test_evaluate_objective_rejects_string_x():
    function = lambda x: x**2

    with pytest.raises(TypeError, match="numérique"):
        evaluate_objective(function, "2")


def test_evaluate_objective_rejects_boolean_x():
    function = lambda x: x**2

    with pytest.raises(TypeError, match="numérique"):
        evaluate_objective(function, True)


def test_evaluate_objective_rejects_nan_x():
    function = lambda x: x**2

    with pytest.raises(ValueError, match="finie"):
        evaluate_objective(function, math.nan)


def test_evaluate_objective_rejects_positive_infinity_x():
    function = lambda x: x**2

    with pytest.raises(ValueError, match="finie"):
        evaluate_objective(function, math.inf)


def test_evaluate_objective_rejects_negative_infinity_x():
    function = lambda x: x**2

    with pytest.raises(ValueError, match="finie"):
        evaluate_objective(function, -math.inf)


# ============================================================
# Validation du résultat
# ============================================================


def test_evaluate_objective_rejects_non_numeric_result():
    function = lambda x: "invalid"

    with pytest.raises(TypeError, match="valeur numérique"):
        evaluate_objective(function, 2)


def test_evaluate_objective_rejects_boolean_result():
    function = lambda x: True

    with pytest.raises(TypeError, match="valeur numérique"):
        evaluate_objective(function, 2)


def test_evaluate_objective_rejects_nan_result():
    function = lambda x: math.nan

    with pytest.raises(ValueError, match="non finie"):
        evaluate_objective(function, 2)


def test_evaluate_objective_rejects_positive_infinity_result():
    function = lambda x: math.inf

    with pytest.raises(ValueError, match="non finie"):
        evaluate_objective(function, 2)


def test_evaluate_objective_rejects_negative_infinity_result():
    function = lambda x: -math.inf

    with pytest.raises(ValueError, match="non finie"):
        evaluate_objective(function, 2)


# ============================================================
# Gestion des erreurs d'évaluation
# ============================================================


def test_evaluate_objective_handles_function_error():
    def function(x):
        raise RuntimeError("erreur interne")

    with pytest.raises(ValueError, match="Impossible d'évaluer"):
        evaluate_objective(function, 2)


# ============================================================
# objective_name
# ============================================================


def test_objective_name_returns_function_name():
    def quadratic(x):
        return x**2

    assert objective_name(quadratic) == "quadratic"


def test_objective_name_returns_lambda_name():
    function = lambda x: x**2

    assert objective_name(function) == "<lambda>"


def test_objective_name_rejects_non_callable():
    with pytest.raises(TypeError, match="appelable"):
        objective_name(42)


def test_objective_name_supports_callable_object():
    class QuadraticObjective:
        def __call__(self, x):
            return x**2

    objective = QuadraticObjective()

    assert objective_name(objective) == "QuadraticObjective"

