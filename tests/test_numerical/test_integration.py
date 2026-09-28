
"""
Tests unitaires des méthodes d'intégration numérique.

Méthodes testées :
- Rectangle (point milieu)
- Trapèzes
- Simpson 1/3

Le module vérifie également :
- la validation des paramètres ;
- les fonctions scalaires et vectorisées NumPy ;
- la convergence ;
- la précision numérique ;
- certaines propriétés mathématiques.
"""

import math

import numpy as np
import pytest

from core.numerical.integration import (
    rectangle_rule,
    trapezoidal_rule,
    simpson_one_third,
)


# ============================================================
# Fonctions de test
# ============================================================


def constant_function(x):
    """f(x) = 2."""
    return 2


def linear_function(x):
    """f(x) = x."""
    return x


def quadratic_function(x):
    """f(x) = x²."""
    return x**2


def cubic_function(x):
    """f(x) = x³."""
    return x**3


def sine_function(x):
    """f(x) = sin(x)."""
    return math.sin(x)


def exponential_function(x):
    """f(x) = exp(x)."""
    return math.exp(x)


def zero_function(x):
    """f(x) = 0."""
    return 0


# ============================================================
# Rectangle Rule
# ============================================================


class TestRectangleRule:
    """Tests de la méthode des rectangles (point milieu)."""

    def test_constant_function(self):
        result = rectangle_rule(constant_function, 0, 1, 10)

        assert result == pytest.approx(2.0)

    def test_linear_function(self):
        result = rectangle_rule(linear_function, 0, 1, 100)

        assert result == pytest.approx(0.5)

    def test_quadratic_function(self):
        result = rectangle_rule(quadratic_function, 0, 1, 1000)

        assert result == pytest.approx(
            1 / 3,
            rel=1e-6,
        )

    def test_cubic_function(self):
        result = rectangle_rule(cubic_function, 0, 1, 1000)

        assert result == pytest.approx(
            0.25,
            rel=1e-6,
        )

    def test_sine_function(self):
        result = rectangle_rule(
            sine_function,
            0,
            math.pi,
            10_000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-6,
            abs=1e-8,
        )

    def test_numpy_vectorized_function(self):
        result = rectangle_rule(
            lambda x: np.sin(x),
            0,
            np.pi,
            10_000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-6,
            abs=1e-8,
        )

    def test_scalar_only_function(self):
        def scalar_function(x):
            if isinstance(x, np.ndarray):
                raise TypeError("Fonction scalaire uniquement")
            return x**2

        result = rectangle_rule(
            scalar_function,
            0,
            1,
            1000,
        )

        assert result == pytest.approx(
            1 / 3,
            rel=1e-6,
        )

    def test_convergence(self):
        result_low = rectangle_rule(
            sine_function,
            0,
            math.pi,
            100,
        )

        result_high = rectangle_rule(
            sine_function,
            0,
            math.pi,
            10_000,
        )

        error_low = abs(result_low - 2.0)
        error_high = abs(result_high - 2.0)

        assert error_high < error_low


# ============================================================
# Trapezoidal Rule
# ============================================================


class TestTrapezoidalRule:
    """Tests de la méthode des trapèzes."""

    def test_constant_function(self):
        result = trapezoidal_rule(
            constant_function,
            0,
            1,
            10,
        )

        assert result == pytest.approx(2.0)

    def test_linear_function(self):
        result = trapezoidal_rule(
            linear_function,
            0,
            1,
            10,
        )

        assert result == pytest.approx(0.5)

    def test_quadratic_function(self):
        result = trapezoidal_rule(
            quadratic_function,
            0,
            1,
            1000,
        )

        assert result == pytest.approx(
            1 / 3,
            rel=1e-6,
        )

    def test_cubic_function(self):
        result = trapezoidal_rule(
            cubic_function,
            0,
            1,
            1000,
        )

        assert result == pytest.approx(
            0.25,
            rel=1e-4,
        )

    def test_sine_function(self):
        result = trapezoidal_rule(
            sine_function,
            0,
            math.pi,
            10_000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-6,
            abs=1e-8,
        )

    def test_exponential_function(self):
        result = trapezoidal_rule(
            exponential_function,
            0,
            1,
            10_000,
        )

        exact = math.e - 1

        assert result == pytest.approx(
            exact,
            rel=1e-6,
        )

    def test_numpy_vectorized_function(self):
        result = trapezoidal_rule(
            lambda x: np.sin(x),
            0,
            np.pi,
            10_000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-6,
            abs=1e-8,
        )

    def test_convergence(self):
        result_low = trapezoidal_rule(
            sine_function,
            0,
            math.pi,
            100,
        )

        result_high = trapezoidal_rule(
            sine_function,
            0,
            math.pi,
            10_000,
        )

        error_low = abs(result_low - 2.0)
        error_high = abs(result_high - 2.0)

        assert error_high < error_low


# ============================================================
# Simpson 1/3
# ============================================================


class TestSimpsonOneThird:
    """Tests de la méthode de Simpson 1/3."""

    def test_constant_function(self):
        result = simpson_one_third(
            constant_function,
            0,
            1,
            10,
        )

        assert result == pytest.approx(2.0)

    def test_linear_function(self):
        result = simpson_one_third(
            linear_function,
            0,
            1,
            10,
        )

        assert result == pytest.approx(0.5)

    def test_quadratic_function_exact(self):
        result = simpson_one_third(
            quadratic_function,
            0,
            1,
            10,
        )

        assert result == pytest.approx(
            1 / 3,
            rel=1e-12,
            abs=1e-14,
        )

    def test_cubic_function_exact(self):
        result = simpson_one_third(
            cubic_function,
            0,
            1,
            10,
        )

        assert result == pytest.approx(
            0.25,
            rel=1e-12,
            abs=1e-14,
        )

    def test_sine_function(self):
        result = simpson_one_third(
            sine_function,
            0,
            math.pi,
            1000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-10,
            abs=1e-12,
        )

    def test_exponential_function(self):
        result = simpson_one_third(
            exponential_function,
            0,
            1,
            1000,
        )

        exact = math.e - 1

        assert result == pytest.approx(
            exact,
            rel=1e-10,
            abs=1e-12,
        )

    def test_numpy_vectorized_function(self):
        result = simpson_one_third(
            lambda x: np.sin(x),
            0,
            np.pi,
            1000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-10,
            abs=1e-12,
        )

    def test_convergence(self):
        result_low = simpson_one_third(
            sine_function,
            0,
            math.pi,
            10,
        )

        result_high = simpson_one_third(
            sine_function,
            0,
            math.pi,
            1000,
        )

        error_low = abs(result_low - 2.0)
        error_high = abs(result_high - 2.0)

        assert error_high < error_low

    def test_odd_number_of_subdivisions_raises(self):
        with pytest.raises(ValueError):
            simpson_one_third(
                quadratic_function,
                0,
                1,
                9,
            )

    def test_even_number_of_subdivisions_is_valid(self):
        result = simpson_one_third(
            quadratic_function,
            0,
            1,
            10,
        )

        assert result == pytest.approx(
            1 / 3,
            rel=1e-12,
        )


# ============================================================
# Validation commune
# ============================================================


class TestIntegrationValidation:
    """Tests de validation des paramètres."""

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_function_must_be_callable(self, method):
        with pytest.raises(TypeError):
            method(
                "not a function",
                0,
                1,
                10,
            )

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_invalid_bounds(self, method):
        with pytest.raises(ValueError):
            method(
                quadratic_function,
                1,
                0,
                10,
            )

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    @pytest.mark.parametrize(
        "n",
        [
            0,
            -1,
            -10,
        ],
    )
    def test_invalid_number_of_subdivisions(self, method, n):
        with pytest.raises(ValueError):
            method(
                quadratic_function,
                0,
                1,
                n,
            )

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_boolean_number_of_subdivisions_is_invalid(self, method):
        with pytest.raises(TypeError):
            method(
                quadratic_function,
                0,
                1,
                True,
            )

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_non_finite_bounds(self, method):
        with pytest.raises(ValueError):
            method(
                quadratic_function,
                math.nan,
                1,
                10,
            )

        with pytest.raises(ValueError):
            method(
                quadratic_function,
                0,
                math.inf,
                10,
            )

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    @pytest.mark.parametrize(
        "bad_function",
        [
            lambda x: float("nan"),
            lambda x: float("inf"),
            lambda x: float("-inf"),
        ],
    )
    def test_non_finite_function_values(
        self,
        method,
        bad_function,
    ):
        with pytest.raises(ValueError):
            method(
                bad_function,
                0,
                1,
                10,
            )

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_non_numeric_function_values(self, method):
        def invalid_function(x):
            return "invalid"

        with pytest.raises(TypeError):
            method(
                invalid_function,
                0,
                1,
                10,
            )

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_numpy_integer_subdivisions(self, method):
        result = method(
            quadratic_function,
            0,
            1,
            np.int64(100),
        )

        assert np.isfinite(result)

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_numpy_scalar_bounds(self, method):
        result = method(
            quadratic_function,
            np.float64(0),
            np.float64(1),
            100,
        )

        assert np.isfinite(result)


# ============================================================
# Propriétés mathématiques
# ============================================================


class TestMathematicalProperties:
    """Tests de propriétés générales des intégrales numériques."""

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_zero_function(self, method):
        result = method(
            zero_function,
            0,
            10,
            100,
        )

        assert result == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_positive_constant_integral(self, method):
        result = method(
            constant_function,
            0,
            5,
            100,
        )

        assert result > 0

    @pytest.mark.parametrize(
        "method",
        [
            rectangle_rule,
            trapezoidal_rule,
            simpson_one_third,
        ],
    )
    def test_linearity(self, method):
        def f(x):
            return x

        def g(x):
            return x**2

        def combined(x):
            return 2 * f(x) + 3 * g(x)

        integral_f = method(f, 0, 1, 1000)
        integral_g = method(g, 0, 1, 1000)
        integral_combined = method(combined, 0, 1, 1000)

        assert integral_combined == pytest.approx(
            2 * integral_f + 3 * integral_g,
            rel=1e-8,
            abs=1e-10,
        )


# ============================================================
# Comparaison de précision
# ============================================================


class TestMethodAccuracy:
    """Comparaison de la précision des méthodes."""

    def test_rectangle_accuracy_for_sine(self):
        result = rectangle_rule(
            sine_function,
            0,
            math.pi,
            10_000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-6,
            abs=1e-8,
        )

    def test_trapezoidal_accuracy_for_sine(self):
        result = trapezoidal_rule(
            sine_function,
            0,
            math.pi,
            10_000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-6,
            abs=1e-8,
        )

    def test_simpson_accuracy_for_sine(self):
        result = simpson_one_third(
            sine_function,
            0,
            math.pi,
            1000,
        )

        assert result == pytest.approx(
            2.0,
            rel=1e-6,
            abs=1e-8,
        )

    def test_simpson_is_more_accurate_than_trapezoidal(self):
        trapezoidal_result = trapezoidal_rule(
            sine_function,
            0,
            math.pi,
            100,
        )

        simpson_result = simpson_one_third(
            sine_function,
            0,
            math.pi,
            100,
        )

        trapezoidal_error = abs(trapezoidal_result - 2.0)
        simpson_error = abs(simpson_result - 2.0)

        assert simpson_error < trapezoidal_error


# ============================================================
# Workflow complet
# ============================================================


class TestCompleteWorkflow:
    """Test d'un workflow complet d'intégration numérique."""

    def test_complete_integration_workflow(self):
        function = quadratic_function
        a = 0
        b = 1
        n = 1000

        rectangle_result = rectangle_rule(
            function,
            a,
            b,
            n,
        )

        trapezoidal_result = trapezoidal_rule(
            function,
            a,
            b,
            n,
        )

        simpson_result = simpson_one_third(
            function,
            a,
            b,
            n,
        )

        exact_value = 1 / 3

        rectangle_error = abs(
            rectangle_result - exact_value
        )

        trapezoidal_error = abs(
            trapezoidal_result - exact_value
        )

        simpson_error = abs(
            simpson_result - exact_value
        )

        assert np.isfinite(rectangle_result)
        assert np.isfinite(trapezoidal_result)
        assert np.isfinite(simpson_result)

        assert rectangle_error >= 0
        assert trapezoidal_error >= 0
        assert simpson_error >= 0

        assert simpson_error < trapezoidal_error

