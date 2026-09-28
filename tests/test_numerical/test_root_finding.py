import math

import pytest

from core.numerical.root_finding import (
    RootFindingResult,
    bisection,
    newton_raphson,
    secant,
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
# RESULT DATACLASS
# ============================================================


def test_root_finding_result_contains_expected_values():
    result = RootFindingResult(
        root=math.sqrt(2),
        iterations=5,
        error=1e-8,
    )

    assert result.root == pytest.approx(math.sqrt(2))
    assert result.iterations == 5
    assert result.error == pytest.approx(1e-8)


# ============================================================
# DICHOTOMIE — CAS NORMAUX
# ============================================================


def test_bisection_finds_square_root_of_two():
    result = bisection(
        quadratic,
        1,
        2,
        tolerance=1e-10,
    )

    assert result.root == pytest.approx(math.sqrt(2), abs=1e-9)
    assert result.iterations > 0
    assert result.error <= 1e-10


def test_bisection_finds_cubic_root():
    expected = 1.5213797068

    result = bisection(
        cubic,
        1,
        2,
        tolerance=1e-10,
    )

    assert result.root == pytest.approx(expected, abs=1e-9)
    assert result.error <= 1e-10


def test_bisection_returns_lower_endpoint_when_it_is_root():
    result = bisection(
        lambda x: x - 2,
        2,
        5,
    )

    assert result.root == 2
    assert result.iterations == 0
    assert result.error == 0


def test_bisection_returns_upper_endpoint_when_it_is_root():
    result = bisection(
        lambda x: x - 2,
        1,
        2,
    )

    assert result.root == 2
    assert result.iterations == 0
    assert result.error == 0


def test_bisection_respects_max_iterations():
    result = bisection(
        quadratic,
        1,
        2,
        tolerance=1e-20,
        max_iterations=5,
    )

    assert result.iterations == 5
    assert result.error > 0


# ============================================================
# DICHOTOMIE — ERREURS
# ============================================================


def test_bisection_rejects_non_callable_function():
    with pytest.raises(TypeError):
        bisection(
            "not a function",
            1,
            2,
        )


def test_bisection_rejects_invalid_interval():
    with pytest.raises(ValueError):
        bisection(
            quadratic,
            2,
            1,
        )


def test_bisection_rejects_equal_bounds():
    with pytest.raises(ValueError):
        bisection(
            quadratic,
            1,
            1,
        )


def test_bisection_rejects_interval_without_sign_change():
    with pytest.raises(ValueError):
        bisection(
            quadratic,
            2,
            3,
        )


def test_bisection_rejects_zero_tolerance():
    with pytest.raises(ValueError):
        bisection(
            quadratic,
            1,
            2,
            tolerance=0,
        )


def test_bisection_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        bisection(
            quadratic,
            1,
            2,
            tolerance=-1,
        )


def test_bisection_rejects_invalid_max_iterations():
    with pytest.raises(ValueError):
        bisection(
            quadratic,
            1,
            2,
            max_iterations=0,
        )


def test_bisection_rejects_non_integer_max_iterations():
    with pytest.raises(TypeError):
        bisection(
            quadratic,
            1,
            2,
            max_iterations=10.5,
        )


def test_bisection_rejects_boolean_tolerance():
    with pytest.raises(TypeError):
        bisection(
            quadratic,
            1,
            2,
            tolerance=True,
        )


def test_bisection_rejects_boolean_bound():
    with pytest.raises(TypeError):
        bisection(
            quadratic,
            True,
            2,
        )


# ============================================================
# NEWTON-RAPHSON — CAS NORMAUX
# ============================================================


def test_newton_raphson_finds_square_root_of_two():
    result = newton_raphson(
        quadratic,
        quadratic_derivative,
        initial_guess=1.5,
        tolerance=1e-10,
    )

    assert result.root == pytest.approx(math.sqrt(2), abs=1e-9)
    assert result.iterations > 0
    assert result.error <= 1e-10


def test_newton_raphson_finds_cubic_root():
    expected = 1.5213797068

    result = newton_raphson(
        cubic,
        cubic_derivative,
        initial_guess=1.5,
        tolerance=1e-10,
    )

    assert result.root == pytest.approx(expected, abs=1e-9)
    assert result.error <= 1e-10


def test_newton_raphson_returns_exact_initial_root():
    result = newton_raphson(
        lambda x: x - 3,
        lambda x: 1,
        initial_guess=3,
    )

    assert result.root == 3
    assert result.iterations == 0
    assert result.error == 0


def test_newton_raphson_respects_max_iterations():
    result = newton_raphson(
        quadratic,
        quadratic_derivative,
        initial_guess=1.5,
        tolerance=1e-30,
        max_iterations=1,
    )

    assert result.iterations == 1
    assert result.error > 0


# ============================================================
# NEWTON-RAPHSON — ERREURS
# ============================================================


def test_newton_raphson_rejects_non_callable_function():
    with pytest.raises(TypeError):
        newton_raphson(
            "not a function",
            quadratic_derivative,
            1.5,
        )


def test_newton_raphson_rejects_non_callable_derivative():
    with pytest.raises(TypeError):
        newton_raphson(
            quadratic,
            "not a derivative",
            1.5,
        )


def test_newton_raphson_rejects_zero_derivative():
    with pytest.raises(ValueError):
        newton_raphson(
            lambda x: x**2 + 1,
            lambda x: 0,
            initial_guess=1,
        )


def test_newton_raphson_rejects_invalid_initial_guess():
    with pytest.raises(TypeError):
        newton_raphson(
            quadratic,
            quadratic_derivative,
            initial_guess="1.5",
        )


def test_newton_raphson_rejects_zero_max_iterations():
    with pytest.raises(ValueError):
        newton_raphson(
            quadratic,
            quadratic_derivative,
            initial_guess=1.5,
            max_iterations=0,
        )


def test_newton_raphson_rejects_non_integer_max_iterations():
    with pytest.raises(TypeError):
        newton_raphson(
            quadratic,
            quadratic_derivative,
            initial_guess=1.5,
            max_iterations=10.5,
        )


def test_newton_raphson_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        newton_raphson(
            quadratic,
            quadratic_derivative,
            initial_guess=1.5,
            tolerance=-1,
        )


# ============================================================
# SECANTE — CAS NORMAUX
# ============================================================


def test_secant_finds_square_root_of_two():
    result = secant(
        quadratic,
        first_guess=1,
        second_guess=2,
        tolerance=1e-10,
    )

    assert result.root == pytest.approx(math.sqrt(2), abs=1e-9)
    assert result.iterations > 0
    assert result.error <= 1e-10


def test_secant_finds_cubic_root():
    expected = 1.5213797068

    result = secant(
        cubic,
        first_guess=1,
        second_guess=2,
        tolerance=1e-10,
    )

    assert result.root == pytest.approx(expected, abs=1e-9)
    assert result.error <= 1e-10


def test_secant_returns_first_exact_root():
    result = secant(
        lambda x: x - 2,
        first_guess=2,
        second_guess=5,
    )

    assert result.root == 2
    assert result.iterations == 0
    assert result.error == 0


def test_secant_returns_second_exact_root():
    result = secant(
        lambda x: x - 2,
        first_guess=1,
        second_guess=2,
    )

    assert result.root == 2
    assert result.iterations == 0
    assert result.error == 0


def test_secant_respects_max_iterations():
    result = secant(
        quadratic,
        first_guess=1,
        second_guess=2,
        tolerance=1e-30,
        max_iterations=2,
    )

    assert result.iterations == 2
    assert result.error > 0


# ============================================================
# SECANTE — ERREURS
# ============================================================


def test_secant_rejects_non_callable_function():
    with pytest.raises(TypeError):
        secant(
            "not a function",
            1,
            2,
        )


def test_secant_rejects_equal_initial_guesses():
    with pytest.raises(ValueError):
        secant(
            quadratic,
            first_guess=1,
            second_guess=1,
        )


def test_secant_rejects_invalid_first_guess():
    with pytest.raises(TypeError):
        secant(
            quadratic,
            first_guess="1",
            second_guess=2,
        )


def test_secant_rejects_invalid_second_guess():
    with pytest.raises(TypeError):
        secant(
            quadratic,
            first_guess=1,
            second_guess="2",
        )


def test_secant_rejects_zero_tolerance():
    with pytest.raises(ValueError):
        secant(
            quadratic,
            first_guess=1,
            second_guess=2,
            tolerance=0,
        )


def test_secant_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        secant(
            quadratic,
            first_guess=1,
            second_guess=2,
            tolerance=-1,
        )


def test_secant_rejects_zero_max_iterations():
    with pytest.raises(ValueError):
        secant(
            quadratic,
            first_guess=1,
            second_guess=2,
            max_iterations=0,
        )


def test_secant_rejects_non_integer_max_iterations():
    with pytest.raises(TypeError):
        secant(
            quadratic,
            first_guess=1,
            second_guess=2,
            max_iterations=5.5,
        )


# ============================================================
# VALIDATION DES VALEURS NUMÉRIQUES
# ============================================================


@pytest.mark.parametrize(
    "method",
    [
        bisection,
    ],
)
def test_bisection_rejects_non_finite_bounds(method):
    with pytest.raises(ValueError):
        method(
            quadratic,
            float("nan"),
            2,
        )

    with pytest.raises(ValueError):
        method(
            quadratic,
            1,
            float("inf"),
        )


def test_newton_raphson_rejects_non_finite_initial_guess():
    with pytest.raises(ValueError):
        newton_raphson(
            quadratic,
            quadratic_derivative,
            initial_guess=float("nan"),
        )


def test_secant_rejects_non_finite_initial_guess():
    with pytest.raises(ValueError):
        secant(
            quadratic,
            first_guess=float("inf"),
            second_guess=2,
        )


# ============================================================
# VALIDATION DES RETOURS DE FONCTION
# ============================================================


def test_bisection_rejects_non_numeric_function_result():
    with pytest.raises(TypeError):
        bisection(
            lambda x: "invalid",
            0,
            1,
        )


def test_newton_raphson_rejects_non_numeric_function_result():
    with pytest.raises(TypeError):
        newton_raphson(
            lambda x: "invalid",
            lambda x: 1,
            1,
        )


def test_secant_rejects_non_numeric_function_result():
    with pytest.raises(TypeError):
        secant(
            lambda x: "invalid",
            0,
            1,
        )


def test_bisection_rejects_non_finite_function_result():
    with pytest.raises(ValueError):
        bisection(
            lambda x: float("nan"),
            -1,
            1,
        )


def test_newton_raphson_rejects_non_finite_function_result():
    with pytest.raises(ValueError):
        newton_raphson(
            lambda x: float("inf"),
            lambda x: 1,
            1,
        )


def test_secant_rejects_non_finite_function_result():
    with pytest.raises(ValueError):
        secant(
            lambda x: float("inf"),
            0,
            1,
        )