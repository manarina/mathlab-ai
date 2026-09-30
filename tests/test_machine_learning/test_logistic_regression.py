"""
Tests de la régression logistique.

Ces tests vérifient :
    - la fonction sigmoid
    - la validation des données
    - l'entraînement du modèle
    - les probabilités prédites
    - les classes prédites
    - la Binary Cross-Entropy
    - l'accuracy
    - le pipeline complet
    - les paramètres invalides
"""

import numpy as np
import pytest

from core.machine_learning.logistic_regression import (
    sigmoid,
    fit_logistic_regression,
    predict_probability,
    predict_class,
    binary_cross_entropy,
    accuracy_score,
    logistic_regression,
)


# ============================================================
# SIGMOID
# ============================================================

def test_sigmoid_zero():
    """sigmoid(0) doit être égal à 0.5."""
    assert sigmoid(0) == pytest.approx(0.5)


def test_sigmoid_positive_value():
    """sigmoid(x) doit être supérieure à 0.5 pour x > 0."""
    result = sigmoid(2)

    assert result > 0.5
    assert result < 1.0


def test_sigmoid_negative_value():
    """sigmoid(x) doit être inférieure à 0.5 pour x < 0."""
    result = sigmoid(-2)

    assert result > 0.0
    assert result < 0.5


def test_sigmoid_array():
    """sigmoid doit fonctionner avec un tableau NumPy."""
    x = np.array([-2.0, 0.0, 2.0])

    result = sigmoid(x)

    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert result[0] < 0.5
    assert result[1] == pytest.approx(0.5)
    assert result[2] > 0.5


def test_sigmoid_output_range():
    """Les sorties de sigmoid doivent être comprises entre 0 et 1."""
    x = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])

    result = sigmoid(x)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_sigmoid_large_values_are_stable():
    """
    sigmoid doit rester stable numériquement
    pour des valeurs très grandes.
    """
    x = np.array([-1000.0, 1000.0])

    result = sigmoid(x)

    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(1.0)


def test_sigmoid_rejects_nan():
    """sigmoid doit refuser les valeurs NaN."""
    with pytest.raises(ValueError):
        sigmoid(np.nan)


# ============================================================
# FIT LOGISTIC REGRESSION
# ============================================================

def test_fit_returns_two_parameters():
    """fit doit retourner l'intercept et le coefficient."""
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    intercept, coefficient = fit_logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=2000,
    )

    assert isinstance(intercept, float)
    assert isinstance(coefficient, float)


def test_fit_learns_positive_relationship():
    """
    Pour des données où y augmente avec x,
    le coefficient doit être positif.
    """
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    intercept, coefficient = fit_logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=3000,
    )

    assert coefficient > 0


def test_fit_learns_negative_relationship():
    """
    Pour des données où y diminue avec x,
    le coefficient doit être négatif.
    """
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([1, 1, 1, 0, 0, 0], dtype=float)

    intercept, coefficient = fit_logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=3000,
    )

    assert coefficient < 0


def test_fit_returns_finite_parameters():
    """Les paramètres appris doivent être finis."""
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    intercept, coefficient = fit_logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=1000,
    )

    assert np.isfinite(intercept)
    assert np.isfinite(coefficient)


# ============================================================
# PREDICT PROBABILITY
# ============================================================

def test_predict_probability_range():
    """Les probabilités doivent être comprises entre 0 et 1."""
    x = np.array([-2, -1, 0, 1, 2], dtype=float)

    probabilities = predict_probability(
        x,
        intercept=0.0,
        coefficient=1.0,
    )

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_predict_probability_at_zero():
    """
    Avec intercept=0 et coefficient=0,
    toutes les probabilités doivent être égales à 0.5.
    """
    x = np.array([-10, -2, 0, 3, 10], dtype=float)

    probabilities = predict_probability(
        x,
        intercept=0.0,
        coefficient=0.0,
    )

    assert np.allclose(probabilities, 0.5)


def test_predict_probability_increases_with_positive_coefficient():
    """
    Avec un coefficient positif,
    la probabilité doit augmenter avec x.
    """
    x = np.array([0, 1, 2], dtype=float)

    probabilities = predict_probability(
        x,
        intercept=0.0,
        coefficient=1.0,
    )

    assert probabilities[0] < probabilities[1]
    assert probabilities[1] < probabilities[2]


def test_predict_probability_decreases_with_negative_coefficient():
    """
    Avec un coefficient négatif,
    la probabilité doit diminuer avec x.
    """
    x = np.array([0, 1, 2], dtype=float)

    probabilities = predict_probability(
        x,
        intercept=0.0,
        coefficient=-1.0,
    )

    assert probabilities[0] > probabilities[1]
    assert probabilities[1] > probabilities[2]


# ============================================================
# PREDICT CLASS
# ============================================================

def test_predict_class_default_threshold():
    """Le seuil par défaut doit être 0.5."""
    x = np.array([-2, 0, 2], dtype=float)

    predictions = predict_class(
        x,
        intercept=0.0,
        coefficient=1.0,
    )

    expected = np.array([0, 1, 1])

    assert np.array_equal(predictions, expected)


def test_predict_class_custom_threshold():
    """predict_class doit accepter un seuil personnalisé."""
    x = np.array([-2, 0, 2], dtype=float)

    predictions = predict_class(
        x,
        intercept=0.0,
        coefficient=1.0,
        threshold=0.8,
    )

    expected = np.array([0, 0, 0])

    assert np.array_equal(predictions, expected)


def test_predict_class_returns_binary_values():
    """Les prédictions doivent être uniquement 0 ou 1."""
    x = np.linspace(-10, 10, 20)

    predictions = predict_class(
        x,
        intercept=0.0,
        coefficient=1.0,
    )

    assert np.all(np.isin(predictions, [0, 1]))


# ============================================================
# BINARY CROSS-ENTROPY
# ============================================================

def test_binary_cross_entropy_perfect_predictions():
    """
    Des prédictions parfaites doivent produire
    une perte très proche de zéro.
    """
    y_true = np.array([0, 0, 1, 1], dtype=float)
    probabilities = np.array([0, 0, 1, 1], dtype=float)

    loss = binary_cross_entropy(
        y_true,
        probabilities,
    )

    assert loss == pytest.approx(0.0, abs=1e-10)


def test_binary_cross_entropy_half_predictions():
    """
    Avec p=0.5 pour chaque observation,
    BCE = -log(0.5) ≈ 0.693147.
    """
    y_true = np.array([0, 1, 0, 1], dtype=float)
    probabilities = np.array([0.5, 0.5, 0.5, 0.5])

    loss = binary_cross_entropy(
        y_true,
        probabilities,
    )

    assert loss == pytest.approx(
        -np.log(0.5),
        rel=1e-10,
    )


def test_binary_cross_entropy_is_positive():
    """La BCE doit être positive ou nulle."""
    y_true = np.array([0, 1, 0, 1], dtype=float)
    probabilities = np.array([0.2, 0.8, 0.3, 0.7])

    loss = binary_cross_entropy(
        y_true,
        probabilities,
    )

    assert loss >= 0.0


# ============================================================
# ACCURACY
# ============================================================

def test_accuracy_perfect_predictions():
    """Accuracy = 1 lorsque toutes les prédictions sont correctes."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    assert accuracy == pytest.approx(1.0)


def test_accuracy_zero_correct_predictions():
    """Accuracy = 0 lorsque toutes les prédictions sont fausses."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    assert accuracy == pytest.approx(0.0)


def test_accuracy_partial_predictions():
    """Vérifie le calcul d'une accuracy partielle."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    assert accuracy == pytest.approx(0.5)


# ============================================================
# PIPELINE COMPLET
# ============================================================

def test_logistic_regression_returns_expected_keys():
    """
    Le pipeline complet doit retourner
    toutes les informations nécessaires.
    """
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    result = logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=2000,
    )

    expected_keys = {
        "intercept",
        "coefficient",
        "probabilities",
        "predictions",
        "loss",
        "accuracy",
        "threshold",
        "learning_rate",
        "n_iterations",
    }

    assert set(result.keys()) == expected_keys


def test_logistic_regression_predictions_are_binary():
    """Le pipeline doit produire des classes 0/1."""
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    result = logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=2000,
    )

    assert np.all(
        np.isin(result["predictions"], [0, 1])
    )


def test_logistic_regression_probabilities_are_valid():
    """Les probabilités du pipeline doivent être valides."""
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    result = logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=2000,
    )

    probabilities = result["probabilities"]

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_logistic_regression_learns_simple_dataset():
    """
    Vérifie que le modèle apprend correctement
    un jeu de données simple.
    """
    x = np.array(
        [1, 2, 3, 4, 5, 6],
        dtype=float,
    )

    y = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=float,
    )

    result = logistic_regression(
        x,
        y,
        learning_rate=0.1,
        n_iterations=5000,
    )

    assert result["coefficient"] > 0
    assert result["accuracy"] >= 0.83
    assert result["loss"] >= 0.0


# ============================================================
# VALIDATION DES ENTRÉES
# ============================================================

def test_fit_rejects_multidimensional_x():
    """x doit être un tableau 1D."""
    x = np.array([
        [0, 1],
        [2, 3],
    ])

    y = np.array([0, 1])

    with pytest.raises(ValueError):
        fit_logistic_regression(x, y)


def test_fit_rejects_empty_x():
    """x vide doit être refusé."""
    x = np.array([])
    y = np.array([])

    with pytest.raises(ValueError):
        fit_logistic_regression(x, y)


def test_fit_rejects_mismatched_lengths():
    """x et y doivent avoir la même longueur."""
    x = np.array([1, 2, 3])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        fit_logistic_regression(x, y)


def test_fit_rejects_invalid_labels():
    """Les labels doivent être 0 ou 1."""
    x = np.array([1, 2, 3, 4])
    y = np.array([0, 1, 2, 1])

    with pytest.raises(ValueError):
        fit_logistic_regression(x, y)


def test_fit_rejects_nan():
    """Les données contenant NaN doivent être refusées."""
    x = np.array([1, 2, np.nan, 4])
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError):
        fit_logistic_regression(x, y)


def test_fit_rejects_infinite_values():
    """Les données infinies doivent être refusées."""
    x = np.array([1, 2, np.inf, 4])
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError):
        fit_logistic_regression(x, y)


def test_fit_rejects_invalid_learning_rate():
    """learning_rate doit être strictement positif."""
    x = np.array([0, 1, 2])
    y = np.array([0, 0, 1])

    with pytest.raises(ValueError):
        fit_logistic_regression(
            x,
            y,
            learning_rate=0,
        )


def test_fit_rejects_invalid_iterations():
    """n_iterations doit être strictement positif."""
    x = np.array([0, 1, 2])
    y = np.array([0, 0, 1])

    with pytest.raises(ValueError):
        fit_logistic_regression(
            x,
            y,
            n_iterations=0,
        )


def test_predict_class_rejects_invalid_threshold():
    """Le seuil doit être strictement compris entre 0 et 1."""
    x = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        predict_class(
            x,
            intercept=0.0,
            coefficient=1.0,
            threshold=0.0,
        )


def test_predict_class_rejects_threshold_above_one():
    """Un seuil supérieur à 1 doit être refusé."""
    x = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        predict_class(
            x,
            intercept=0.0,
            coefficient=1.0,
            threshold=1.5,
        )


def test_binary_cross_entropy_rejects_invalid_probabilities():
    """Les probabilités hors de [0, 1] doivent être refusées."""
    y_true = np.array([0, 1])

    probabilities = np.array([
        0.2,
        1.5,
    ])

    with pytest.raises(ValueError):
        binary_cross_entropy(
            y_true,
            probabilities,
        )


def test_binary_cross_entropy_rejects_mismatched_lengths():
    """Les deux tableaux doivent avoir la même longueur."""
    y_true = np.array([0, 1, 1])
    probabilities = np.array([0.2, 0.8])

    with pytest.raises(ValueError):
        binary_cross_entropy(
            y_true,
            probabilities,
        )


def test_accuracy_rejects_mismatched_lengths():
    """accuracy_score doit refuser des longueurs différentes."""
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        accuracy_score(
            y_true,
            y_pred,
        )

