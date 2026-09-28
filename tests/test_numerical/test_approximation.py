
from __future__ import annotations

import math

import numpy as np
import pytest

from core.numerical.approximation import (
    absolute_errors,
    coefficient_of_determination,
    evaluate_polynomial,
    polynomial_approximation,
    predicted_values,
    relative_errors,
    root_mean_square_error,
)


# ============================================================
# polynomial_approximation
# ============================================================


def test_polynomial_approximation_recovers_exact_quadratic():
    """
    Les données suivent exactement :

        y = x² + x + 1
    """

    x_values = [0, 1, 2]
    y_values = [1, 3, 7]

    coefficients = polynomial_approximation(
        x_values,
        y_values,
        degree=2,
    )

    assert coefficients == pytest.approx(
        [1.0, 1.0, 1.0],
        abs=1e-10,
    )


def test_polynomial_approximation_recovers_exact_linear_model():
    """
    Les données suivent exactement :

        y = 2x + 3
    """

    x_values = [0, 1, 2, 3]
    y_values = [3, 5, 7, 9]

    coefficients = polynomial_approximation(
        x_values,
        y_values,
        degree=1,
    )

    assert coefficients == pytest.approx(
        [3.0, 2.0],
        abs=1e-10,
    )


def test_polynomial_approximation_constant_model():
    """
    Vérifie une approximation de degré 0.
    """

    x_values = [0, 1, 2, 3]
    y_values = [5, 5, 5, 5]

    coefficients = polynomial_approximation(
        x_values,
        y_values,
        degree=0,
    )

    assert coefficients == pytest.approx(
        [5.0],
        abs=1e-10,
    )


def test_polynomial_approximation_returns_expected_number_of_coefficients():
    """
    Un polynôme de degré n possède n + 1 coefficients.
    """

    x_values = [0, 1, 2, 3, 4]
    y_values = [1, 2, 5, 10, 17]

    coefficients = polynomial_approximation(
        x_values,
        y_values,
        degree=3,
    )

    assert len(coefficients) == 4


def test_polynomial_approximation_accepts_numpy_arrays():
    """
    Les données peuvent être fournies sous forme de tableaux NumPy.
    """

    x_values = np.array([0, 1, 2])
    y_values = np.array([1, 3, 7])

    coefficients = polynomial_approximation(
        x_values,
        y_values,
        degree=2,
    )

    assert coefficients == pytest.approx(
        [1.0, 1.0, 1.0],
        abs=1e-10,
    )


# ============================================================
# VALIDATION polynomial_approximation
# ============================================================


def test_polynomial_approximation_rejects_different_lengths():
    with pytest.raises(
        ValueError,
        match="même longueur",
    ):
        polynomial_approximation(
            [0, 1, 2],
            [1, 3],
            degree=1,
        )


def test_polynomial_approximation_rejects_empty_data():
    with pytest.raises(
        ValueError,
        match="vides",
    ):
        polynomial_approximation(
            [],
            [],
            degree=0,
        )


def test_polynomial_approximation_rejects_too_high_degree():
    """
    Avec 3 points, un polynôme de degré 3 ne peut pas
    être ajusté selon notre contrat API.
    """

    with pytest.raises(
        ValueError,
        match=r"degré \+ 1",
    ):
        polynomial_approximation(
            [0, 1, 2],
            [1, 3, 7],
            degree=3,
        )


def test_polynomial_approximation_rejects_negative_degree():
    with pytest.raises(
        ValueError,
        match="positif ou nul",
    ):
        polynomial_approximation(
            [0, 1, 2],
            [1, 3, 7],
            degree=-1,
        )


def test_polynomial_approximation_rejects_boolean_degree():
    with pytest.raises(
        TypeError,
        match="entier",
    ):
        polynomial_approximation(
            [0, 1],
            [1, 2],
            degree=True,
        )


def test_polynomial_approximation_rejects_non_numeric_data():
    with pytest.raises(
        TypeError,
        match="numériques",
    ):
        polynomial_approximation(
            [0, "a", 2],
            [1, 3, 7],
            degree=1,
        )


def test_polynomial_approximation_rejects_non_finite_x():
    with pytest.raises(
        ValueError,
        match="non finies",
    ):
        polynomial_approximation(
            [0, math.inf, 2],
            [1, 3, 7],
            degree=1,
        )


def test_polynomial_approximation_rejects_non_finite_y():
    with pytest.raises(
        ValueError,
        match="non finies",
    ):
        polynomial_approximation(
            [0, 1, 2],
            [1, math.nan, 7],
            degree=1,
        )


def test_polynomial_approximation_rejects_multidimensional_data():
    with pytest.raises(
        ValueError,
        match="une seule dimension",
    ):
        polynomial_approximation(
            [[0, 1], [2, 3]],
            [1, 3],
            degree=1,
        )


def test_polynomial_approximation_rejects_string_sequence():
    with pytest.raises(
        TypeError,
        match="séquence numérique",
    ):
        polynomial_approximation(
            "0,1,2",
            [1, 3, 7],
            degree=2,
        )


# ============================================================
# evaluate_polynomial
# ============================================================


def test_evaluate_polynomial_at_scalar():
    """
    P(x) = x² + x + 1

    P(2) = 7
    """

    coefficients = [1, 1, 1]

    result = evaluate_polynomial(
        coefficients,
        2,
    )

    assert result == pytest.approx(
        7.0,
        abs=1e-10,
    )


def test_evaluate_polynomial_at_zero():
    coefficients = [5, 2, 7]

    result = evaluate_polynomial(
        coefficients,
        0,
    )

    assert result == pytest.approx(5.0)


def test_evaluate_polynomial_at_negative_value():
    """
    P(x) = x² + x + 1

    P(-2) = 3
    """

    result = evaluate_polynomial(
        [1, 1, 1],
        -2,
    )

    assert result == pytest.approx(3.0)


def test_evaluate_polynomial_accepts_vector():
    coefficients = [1, 1, 1]

    result = evaluate_polynomial(
        coefficients,
        [0, 1, 2],
    )

    assert result == pytest.approx(
        [1.0, 3.0, 7.0],
        abs=1e-10,
    )


def test_evaluate_polynomial_accepts_numpy_array():
    result = evaluate_polynomial(
        [1, 1, 1],
        np.array([0, 1, 2]),
    )

    assert result == pytest.approx(
        [1.0, 3.0, 7.0],
        abs=1e-10,
    )


def test_evaluate_polynomial_constant():
    result = evaluate_polynomial(
        [5],
        [0, 1, 10],
    )

    assert result == pytest.approx(
        [5.0, 5.0, 5.0],
    )


def test_evaluate_polynomial_rejects_empty_coefficients():
    with pytest.raises(
        ValueError,
        match="vide",
    ):
        evaluate_polynomial(
            [],
            2,
        )


def test_evaluate_polynomial_rejects_non_numeric_coefficients():
    with pytest.raises(
        TypeError,
        match="numériques",
    ):
        evaluate_polynomial(
            [1, "a"],
            2,
        )


def test_evaluate_polynomial_rejects_non_finite_coefficients():
    with pytest.raises(
        ValueError,
        match="finis",
    ):
        evaluate_polynomial(
            [1, math.inf],
            2,
        )


def test_evaluate_polynomial_rejects_non_finite_x():
    with pytest.raises(
        ValueError,
        match="finie",
    ):
        evaluate_polynomial(
            [1, 1],
            math.nan,
        )


# ============================================================
# predicted_values
# ============================================================


def test_predicted_values_returns_model_values():
    coefficients = [1, 1, 1]

    result = predicted_values(
        coefficients,
        [0, 1, 2],
    )

    assert result == pytest.approx(
        [1.0, 3.0, 7.0],
        abs=1e-10,
    )


def test_predicted_values_returns_numpy_array():
    result = predicted_values(
        [2, 3],
        [0, 1, 2],
    )

    assert isinstance(
        result,
        np.ndarray,
    )


# ============================================================
# absolute_errors
# ============================================================


def test_absolute_errors():
    y_values = [1, 3, 7]
    predicted = [1, 2, 8]

    errors = absolute_errors(
        y_values,
        predicted,
    )

    assert errors == pytest.approx(
        [0.0, 1.0, 1.0],
    )


def test_absolute_errors_are_non_negative():
    errors = absolute_errors(
        [10, 20, 30],
        [11, 18, 35],
    )

    assert np.all(errors >= 0)


def test_absolute_errors_reject_different_shapes():
    with pytest.raises(
        ValueError,
        match="même forme",
    ):
        absolute_errors(
            [1, 2, 3],
            [1, 2],
        )


# ============================================================
# relative_errors
# ============================================================


def test_relative_errors():
    y_values = [10, 20]
    predicted = [11, 18]

    errors = relative_errors(
        y_values,
        predicted,
    )

    assert errors == pytest.approx(
        [0.1, 0.1],
        abs=1e-10,
    )


def test_relative_errors_returns_nan_when_actual_value_is_zero():
    errors = relative_errors(
        [0, 10],
        [1, 11],
    )

    assert np.isnan(errors[0])
    assert errors[1] == pytest.approx(0.1)


def test_relative_errors_are_non_negative():
    errors = relative_errors(
        [10, 20, 30],
        [11, 18, 35],
    )

    finite_errors = errors[
        np.isfinite(errors)
    ]

    assert np.all(
        finite_errors >= 0
    )


def test_relative_errors_reject_different_shapes():
    with pytest.raises(
        ValueError,
        match="même forme",
    ):
        relative_errors(
            [1, 2, 3],
            [1, 2],
        )


# ============================================================
# root_mean_square_error
# ============================================================


def test_root_mean_square_error():
    """
    Erreurs : [0, 1, 1]

    RMSE = sqrt((0² + 1² + 1²) / 3)
         = sqrt(2/3)
    """

    result = root_mean_square_error(
        [1, 3, 7],
        [1, 2, 8],
    )

    expected = math.sqrt(
        2 / 3
    )

    assert result == pytest.approx(
        expected,
        abs=1e-10,
    )


def test_root_mean_square_error_is_zero_for_exact_model():
    result = root_mean_square_error(
        [1, 3, 7],
        [1, 3, 7],
    )

    assert result == pytest.approx(0.0)


def test_root_mean_square_error_is_non_negative():
    result = root_mean_square_error(
        [1, 3, 7],
        [2, 4, 8],
    )

    assert result >= 0


# ============================================================
# coefficient_of_determination
# ============================================================


def test_coefficient_of_determination_is_one_for_exact_model():
    result = coefficient_of_determination(
        [1, 3, 7],
        [1, 3, 7],
    )

    assert result == pytest.approx(
        1.0,
        abs=1e-10,
    )


def test_coefficient_of_determination_for_simple_model():
    """
    y = [1, 2, 3]
    prediction = [1, 2, 2]

    SST = 2
    SSE = 1

    R² = 1 - 1/2 = 0.5
    """

    result = coefficient_of_determination(
        [1, 2, 3],
        [1, 2, 2],
    )

    assert result == pytest.approx(
        0.5,
        abs=1e-10,
    )


def test_coefficient_of_determination_can_be_negative():
    """
    Un modèle très mauvais peut produire un R² négatif.
    """

    result = coefficient_of_determination(
        [1, 2, 3],
        [10, 10, 10],
    )

    assert result < 0


def test_coefficient_of_determination_rejects_constant_observations():
    with pytest.raises(
        ValueError,
        match="identiques",
    ):
        coefficient_of_determination(
            [5, 5, 5],
            [5, 5, 5],
        )


def test_coefficient_of_determination_rejects_different_shapes():
    with pytest.raises(
        ValueError,
        match="même forme",
    ):
        coefficient_of_determination(
            [1, 2, 3],
            [1, 2],
        )


# ============================================================
# TEST D'INTÉGRATION DU MODULE
# ============================================================


def test_complete_approximation_workflow():
    """
    Vérifie le workflow complet :

    données
        ↓
    approximation polynomiale
        ↓
    prédictions
        ↓
    erreurs
        ↓
    métriques
    """

    x_values = [0, 1, 2, 3, 4]
    y_values = [1, 3, 7, 13, 21]

    coefficients = polynomial_approximation(
        x_values,
        y_values,
        degree=2,
    )

    predicted = predicted_values(
        coefficients,
        x_values,
    )

    errors = absolute_errors(
        y_values,
        predicted,
    )

    rmse = root_mean_square_error(
        y_values,
        predicted,
    )

    r_squared = coefficient_of_determination(
        y_values,
        predicted,
    )

    assert coefficients == pytest.approx(
        [1.0, 1.0, 1.0],
        abs=1e-10,
    )

    assert predicted == pytest.approx(
        y_values,
        abs=1e-10,
    )

    assert np.all(
        errors == pytest.approx(0.0)
    )

    assert rmse == pytest.approx(
        0.0,
        abs=1e-10,
    )

    assert r_squared == pytest.approx(
        1.0,
        abs=1e-10,
    )

