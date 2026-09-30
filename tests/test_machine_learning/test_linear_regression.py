
"""
Tests unitaires de la régression linéaire.

Module testé :
    core.machine_learning.linear_regression
"""

import numpy as np
import pytest

from core.machine_learning.linear_regression import (
    fit_linear_regression,
    predict_linear_regression,
    mean_squared_error,
    r_squared,
    linear_regression,
)


# ============================================================
# FIT LINEAR REGRESSION
# ============================================================


def test_fit_linear_regression_exact_line():
    """Teste une relation linéaire parfaite y = 2x + 1."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 7, 9, 11])

    intercept, slope = fit_linear_regression(x, y)

    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(2.0)


def test_fit_linear_regression_negative_slope():
    """Teste une régression avec une pente négative."""
    x = np.array([1, 2, 3, 4])
    y = np.array([10, 8, 6, 4])

    intercept, slope = fit_linear_regression(x, y)

    assert intercept == pytest.approx(12.0)
    assert slope == pytest.approx(-2.0)


def test_fit_linear_regression_non_integer_values():
    """Teste une relation linéaire avec des valeurs décimales."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.5, 3.5, 5.5, 7.5])

    intercept, slope = fit_linear_regression(x, y)

    assert intercept == pytest.approx(1.5)
    assert slope == pytest.approx(2.0)


# ============================================================
# PREDICTION
# ============================================================


def test_predict_linear_regression():
    """Teste les prédictions du modèle y = 2x + 1."""
    x = np.array([0, 1, 2, 3])

    predictions = predict_linear_regression(
        x,
        intercept=1.0,
        slope=2.0,
    )

    expected = np.array([1, 3, 5, 7])

    np.testing.assert_allclose(predictions, expected)


def test_predict_linear_regression_empty_array():
    """Un tableau vide est accepté pour la fonction de prédiction."""
    x = np.array([], dtype=float)

    predictions = predict_linear_regression(
        x,
        intercept=1.0,
        slope=2.0,
    )

    assert predictions.size == 0


def test_predict_linear_regression_invalid_dimension():
    """Teste le rejet d'un tableau multidimensionnel."""
    x = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="unidimensionnel"):
        predict_linear_regression(
            x,
            intercept=1.0,
            slope=2.0,
        )


# ============================================================
# MEAN SQUARED ERROR
# ============================================================


def test_mean_squared_error_zero():
    """Le MSE doit être nul pour des prédictions parfaites."""
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])

    mse = mean_squared_error(y_true, y_pred)

    assert mse == pytest.approx(0.0)


def test_mean_squared_error():
    """Teste le calcul classique du MSE."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 4])

    mse = mean_squared_error(y_true, y_pred)

    # ((1-2)^2 + (2-2)^2 + (3-4)^2) / 3
    # = (1 + 0 + 1) / 3
    # = 2/3
    assert mse == pytest.approx(2 / 3)


def test_mean_squared_error_different_lengths():
    """Teste le rejet de tableaux de longueurs différentes."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])

    with pytest.raises(ValueError, match="même longueur"):
        mean_squared_error(y_true, y_pred)


# ============================================================
# R-SQUARED
# ============================================================


def test_r_squared_perfect_prediction():
    """R² doit être égal à 1 pour une prédiction parfaite."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    r2 = r_squared(y_true, y_pred)

    assert r2 == pytest.approx(1.0)


def test_r_squared_known_value():
    """Teste R² sur un exemple connu."""
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 4, 3])

    r2 = r_squared(y_true, y_pred)

    # SS_res = 0 + 0 + 1 + 1 = 2
    # moyenne(y_true) = 2.5
    # SS_tot = 2.25 + 0.25 + 0.25 + 2.25 = 5
    # R² = 1 - 2/5 = 0.6
    assert r2 == pytest.approx(0.6)


def test_r_squared_constant_target_raises_error():
    """R² doit refuser une variable cible constante."""
    y_true = np.array([5, 5, 5, 5])
    y_pred = np.array([4, 5, 6, 5])

    with pytest.raises(ValueError, match="constant"):
        r_squared(y_true, y_pred)


# ============================================================
# VALIDATION DES DONNÉES
# ============================================================


def test_fit_linear_regression_different_lengths():
    """x et y doivent avoir la même longueur."""
    x = np.array([1, 2, 3])
    y = np.array([2, 4])

    with pytest.raises(ValueError, match="même longueur"):
        fit_linear_regression(x, y)


def test_fit_linear_regression_not_enough_observations():
    """Au moins deux observations sont nécessaires."""
    x = np.array([1])
    y = np.array([2])

    with pytest.raises(ValueError, match="au moins deux"):
        fit_linear_regression(x, y)


def test_fit_linear_regression_constant_x():
    """x ne peut pas être constant."""
    x = np.array([2, 2, 2, 2])
    y = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError, match="valeurs différentes"):
        fit_linear_regression(x, y)


def test_fit_linear_regression_multidimensional_x():
    """x doit être unidimensionnel."""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([2, 4])

    with pytest.raises(ValueError, match="unidimensionnel"):
        fit_linear_regression(x, y)


def test_fit_linear_regression_nan_values():
    """Les valeurs NaN doivent être rejetées."""
    x = np.array([1, 2, np.nan, 4])
    y = np.array([2, 4, 6, 8])

    with pytest.raises(ValueError, match="non finies"):
        fit_linear_regression(x, y)


def test_fit_linear_regression_infinite_values():
    """Les valeurs infinies doivent être rejetées."""
    x = np.array([1, 2, np.inf, 4])
    y = np.array([2, 4, 6, 8])

    with pytest.raises(ValueError, match="non finies"):
        fit_linear_regression(x, y)


# ============================================================
# PIPELINE COMPLET
# ============================================================


def test_linear_regression_complete():
    """
    Teste le pipeline complet sur :

        y = 3x + 2
    """
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([2, 5, 8, 11, 14])

    result = linear_regression(x, y)

    assert isinstance(result, dict)

    assert result["intercept"] == pytest.approx(2.0)
    assert result["slope"] == pytest.approx(3.0)

    np.testing.assert_allclose(
        result["predictions"],
        y,
    )

    assert result["mse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_linear_regression_result_keys():
    """Vérifie la structure du résultat final."""
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 7, 9])

    result = linear_regression(x, y)

    expected_keys = {
        "intercept",
        "slope",
        "predictions",
        "mse",
        "r2",
    }

    assert set(result.keys()) == expected_keys


def test_linear_regression_noisy_data():
    """
    Teste la régression sur des données non parfaitement linéaires.
    """
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2.1, 4.2, 5.8, 8.1, 9.9])

    result = linear_regression(x, y)

    assert isinstance(result["intercept"], float)
    assert isinstance(result["slope"], float)
    assert isinstance(result["mse"], float)
    assert isinstance(result["r2"], float)

    assert len(result["predictions"]) == len(x)

    assert result["mse"] >= 0.0
    assert result["r2"] <= 1.0

