import math

import pytest

from core.statistics.regression import (
    coefficient_of_determination,
    linear_regression,
    predict,
    regression_intercept,
    regression_predictions,
    regression_slope,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================

X_LINEAR = [1, 2, 3, 4, 5]
Y_LINEAR = [2, 4, 6, 8, 10]


# ============================================================
# TESTS DE LA PENTE
# ============================================================


def test_regression_slope_positive():
    """Vérifie une pente positive."""

    result = regression_slope(
        X_LINEAR,
        Y_LINEAR,
    )

    assert result == pytest.approx(2.0)


def test_regression_slope_negative():
    """Vérifie une pente négative."""

    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]

    result = regression_slope(x, y)

    assert result == pytest.approx(-2.0)


def test_regression_slope_decimal():
    """Vérifie le calcul avec des valeurs décimales."""

    x = [1, 2, 3]
    y = [1.5, 2.5, 3.5]

    result = regression_slope(x, y)

    assert result == pytest.approx(1.0)


# ============================================================
# TESTS DE L'ORDONNÉE À L'ORIGINE
# ============================================================


def test_regression_intercept_zero():
    """Vérifie une droite passant par l'origine."""

    result = regression_intercept(
        X_LINEAR,
        Y_LINEAR,
    )

    assert result == pytest.approx(0.0)


def test_regression_intercept_non_zero():
    """Vérifie une ordonnée à l'origine non nulle."""

    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    result = regression_intercept(x, y)

    assert result == pytest.approx(1.0)


# ============================================================
# TESTS DE LA RÉGRESSION COMPLÈTE
# ============================================================


def test_linear_regression_perfect_positive_relation():
    """
    Vérifie une régression parfaite :

        y = 2x
    """

    slope, intercept = linear_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.0)


def test_linear_regression_non_zero_intercept():
    """
    Vérifie :

        y = 2x + 1
    """

    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    slope, intercept = linear_regression(x, y)

    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_linear_regression_returns_tuple():
    """Vérifie le type du résultat."""

    result = linear_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2


# ============================================================
# TESTS DES PRÉDICTIONS
# ============================================================


def test_predict():
    """
    Vérifie les prédictions pour :

        y = 2x
    """

    result = predict(
        X_LINEAR,
        Y_LINEAR,
        [6, 7, 8],
    )

    assert result == pytest.approx(
        [12.0, 14.0, 16.0]
    )


def test_predict_non_zero_intercept():
    """
    Vérifie les prédictions pour :

        y = 2x + 1
    """

    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    result = predict(
        x,
        y,
        [6, 7],
    )

    assert result == pytest.approx(
        [13.0, 15.0]
    )


def test_regression_predictions():
    """
    Vérifie les valeurs prédites pour les observations
    utilisées dans le modèle.
    """

    result = regression_predictions(
        X_LINEAR,
        Y_LINEAR,
    )

    assert result == pytest.approx(
        [2.0, 4.0, 6.0, 8.0, 10.0]
    )


def test_regression_predictions_imperfect_fit():
    """Vérifie les prédictions d'un modèle non parfait."""

    x = [1, 2, 3, 4]
    y = [2, 5, 5, 8]

    result = regression_predictions(x, y)

    assert len(result) == len(y)

    for value in result:
        assert isinstance(value, float)


# ============================================================
# TESTS DU COEFFICIENT R²
# ============================================================


def test_coefficient_of_determination_perfect_fit():
    """
    Pour une relation parfaitement linéaire :

        R² = 1
    """

    result = coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
    )

    assert result == pytest.approx(1.0)


def test_coefficient_of_determination_imperfect_fit():
    """Vérifie un ajustement linéaire imparfait."""

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 8, 11]

    result = coefficient_of_determination(x, y)

    assert 0 <= result <= 1
    assert result < 1


def test_coefficient_of_determination_expected_value():
    """Vérifie une valeur connue de R²."""

    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]

    result = coefficient_of_determination(x, y)

    assert result == pytest.approx(
        0.64,
        abs=1e-2,
    )


def test_coefficient_of_determination_perfect_negative_relation():
    """
    Une relation parfaitement négative est également
    parfaitement expliquée par une régression linéaire.
    """

    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]

    result = coefficient_of_determination(x, y)

    assert result == pytest.approx(1.0)


# ============================================================
# TESTS DE VALIDATION — LONGUEUR
# ============================================================


def test_different_lengths_raise_value_error():
    """X et Y doivent avoir la même longueur."""

    with pytest.raises(ValueError):
        regression_slope(
            [1, 2, 3],
            [2, 4],
        )


def test_different_lengths_intercept_raise_value_error():
    with pytest.raises(ValueError):
        regression_intercept(
            [1, 2, 3],
            [2, 4],
        )


def test_different_lengths_regression_raise_value_error():
    with pytest.raises(ValueError):
        linear_regression(
            [1, 2, 3],
            [2, 4],
        )


# ============================================================
# TESTS DE VALIDATION — SÉRIES VIDES
# ============================================================


def test_empty_x_raise_value_error():
    with pytest.raises(ValueError):
        regression_slope(
            [],
            [],
        )


def test_empty_y_raise_value_error():
    with pytest.raises(ValueError):
        regression_slope(
            [1, 2, 3],
            [],
        )


# ============================================================
# TESTS DE VALIDATION — OBSERVATIONS INSUFFISANTES
# ============================================================


def test_one_observation_raise_value_error():
    """Une seule observation ne suffit pas."""

    with pytest.raises(ValueError):
        regression_slope(
            [1],
            [2],
        )


def test_one_observation_linear_regression_raise_value_error():
    with pytest.raises(ValueError):
        linear_regression(
            [1],
            [2],
        )


# ============================================================
# TESTS DE VALIDATION — TYPES
# ============================================================


def test_nonnumeric_x_raise_type_error():
    with pytest.raises(TypeError):
        regression_slope(
            [1, "2", 3],
            [2, 4, 6],
        )


def test_nonnumeric_y_raise_type_error():
    with pytest.raises(TypeError):
        regression_slope(
            [1, 2, 3],
            [2, "4", 6],
        )


def test_boolean_x_raise_type_error():
    with pytest.raises(TypeError):
        regression_slope(
            [True, 2, 3],
            [2, 4, 6],
        )


def test_boolean_y_raise_type_error():
    with pytest.raises(TypeError):
        regression_slope(
            [1, 2, 3],
            [False, 4, 6],
        )


# ============================================================
# TESTS DE VALIDATION — VALEURS NON FINIES
# ============================================================


def test_nan_x_raise_value_error():
    with pytest.raises(ValueError):
        regression_slope(
            [1, math.nan, 3],
            [2, 4, 6],
        )


def test_nan_y_raise_value_error():
    with pytest.raises(ValueError):
        regression_slope(
            [1, 2, 3],
            [2, math.nan, 6],
        )


def test_infinity_x_raise_value_error():
    with pytest.raises(ValueError):
        regression_slope(
            [1, math.inf, 3],
            [2, 4, 6],
        )


def test_infinity_y_raise_value_error():
    with pytest.raises(ValueError):
        regression_slope(
            [1, 2, 3],
            [2, 4, math.inf],
        )


# ============================================================
# TESTS DE VALIDATION — X CONSTANT
# ============================================================


def test_constant_x_raise_value_error():
    """
    Une régression linéaire est impossible si X
    ne varie pas.
    """

    with pytest.raises(ValueError):
        regression_slope(
            [2, 2, 2],
            [1, 3, 5],
        )


def test_constant_x_intercept_raise_value_error():
    with pytest.raises(ValueError):
        regression_intercept(
            [2, 2, 2],
            [1, 3, 5],
        )


def test_constant_x_linear_regression_raise_value_error():
    with pytest.raises(ValueError):
        linear_regression(
            [2, 2, 2],
            [1, 3, 5],
        )


def test_constant_x_predict_raise_value_error():
    with pytest.raises(ValueError):
        predict(
            [2, 2, 2],
            [1, 3, 5],
            [4, 5],
        )


# ============================================================
# TESTS DES PRÉDICTIONS — VALIDATION
# ============================================================


def test_predict_empty_values_raise_value_error():
    with pytest.raises(ValueError):
        predict(
            X_LINEAR,
            Y_LINEAR,
            [],
        )


def test_predict_nonnumeric_value_raise_type_error():
    with pytest.raises(TypeError):
        predict(
            X_LINEAR,
            Y_LINEAR,
            [6, "7"],
        )


def test_predict_boolean_value_raise_type_error():
    with pytest.raises(TypeError):
        predict(
            X_LINEAR,
            Y_LINEAR,
            [6, True],
        )


def test_predict_nan_raise_value_error():
    with pytest.raises(ValueError):
        predict(
            X_LINEAR,
            Y_LINEAR,
            [6, math.nan],
        )


def test_predict_infinity_raise_value_error():
    with pytest.raises(ValueError):
        predict(
            X_LINEAR,
            Y_LINEAR,
            [6, math.inf],
        )