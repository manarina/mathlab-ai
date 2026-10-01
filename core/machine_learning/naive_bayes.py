
"""
Naive Bayes — Gaussian Naive Bayes
==================================

Implémentation éducative de l'algorithme Gaussian Naive Bayes
avec NumPy uniquement.

Le modèle suppose que chaque caractéristique suit une loi normale
conditionnellement à la classe.

API publique
------------
gaussian_probability()
fit_naive_bayes()
predict_proba_naive_bayes()
predict_naive_bayes()
accuracy_score()
naive_bayes()
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ============================================================
# VALIDATION
# ============================================================


def _validate_input(
    X: np.ndarray,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Valide et convertit les données d'entrée.

    Parameters
    ----------
    X : array-like
        Matrice des observations.
    y : array-like, optional
        Labels correspondants.

    Returns
    -------
    X_array : np.ndarray
        Matrice numérique 2D.
    y_array : np.ndarray | None
        Labels sous forme de tableau.

    Raises
    ------
    ValueError
        Si les données sont invalides.
    """

    X_array = np.asarray(X, dtype=float)

    if X_array.ndim != 2:
        raise ValueError(
            "X doit être une matrice 2D."
        )

    if X_array.shape[0] == 0:
        raise ValueError(
            "X ne doit pas être vide."
        )

    if X_array.shape[1] == 0:
        raise ValueError(
            "X doit contenir au moins une caractéristique."
        )

    if not np.all(np.isfinite(X_array)):
        raise ValueError(
            "X doit contenir uniquement des valeurs finies."
        )

    if y is None:
        return X_array, None

    y_array = np.asarray(y)

    if y_array.ndim != 1:
        raise ValueError(
            "y doit être un vecteur 1D."
        )

    if y_array.shape[0] != X_array.shape[0]:
        raise ValueError(
            "X et y doivent contenir le même nombre "
            "d'observations."
        )

    if y_array.shape[0] == 0:
        raise ValueError(
            "y ne doit pas être vide."
        )

    return X_array, y_array


def _validate_model_parameters(
    means: np.ndarray,
    variances: np.ndarray,
    priors: np.ndarray,
    classes: np.ndarray,
) -> None:
    """Valide les paramètres appris du modèle."""

    if means.ndim != 2:
        raise ValueError(
            "means doit être une matrice 2D."
        )

    if variances.ndim != 2:
        raise ValueError(
            "variances doit être une matrice 2D."
        )

    if means.shape != variances.shape:
        raise ValueError(
            "means et variances doivent avoir la même forme."
        )

    if priors.ndim != 1:
        raise ValueError(
            "priors doit être un vecteur 1D."
        )

    if classes.ndim != 1:
        raise ValueError(
            "classes doit être un vecteur 1D."
        )

    if means.shape[0] != classes.shape[0]:
        raise ValueError(
            "Le nombre de classes doit correspondre "
            "au nombre de lignes de means."
        )

    if priors.shape[0] != classes.shape[0]:
        raise ValueError(
            "Le nombre de classes doit correspondre "
            "à la taille de priors."
        )

    if not np.all(np.isfinite(means)):
        raise ValueError(
            "means doit contenir uniquement des valeurs finies."
        )

    if not np.all(np.isfinite(variances)):
        raise ValueError(
            "variances doit contenir uniquement des valeurs finies."
        )

    if not np.all(np.isfinite(priors)):
        raise ValueError(
            "priors doit contenir uniquement des valeurs finies."
        )

    if np.any(variances <= 0):
        raise ValueError(
            "Toutes les variances doivent être strictement positives."
        )

    if np.any(priors <= 0):
        raise ValueError(
            "Toutes les probabilités a priori doivent être "
            "strictement positives."
        )

    if not np.isclose(np.sum(priors), 1.0):
        raise ValueError(
            "La somme des probabilités a priori doit être égale à 1."
        )


# ============================================================
# GAUSSIAN PROBABILITY
# ============================================================


def gaussian_probability(
    X: np.ndarray | list,
    mean: float,
    variance: float,
) -> np.ndarray:
    """
    Calcule la densité de probabilité gaussienne.

    Formule :

        p(x) =
            1 / sqrt(2πσ²)
            × exp(-(x - μ)² / (2σ²))

    Parameters
    ----------
    X : array-like
        Valeurs à évaluer.
    mean : float
        Moyenne μ.
    variance : float
        Variance σ².

    Returns
    -------
    np.ndarray
        Densités de probabilité.

    Raises
    ------
    ValueError
        Si mean/variance sont invalides.
    """

    values = np.asarray(X, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "X doit contenir uniquement des valeurs finies."
        )

    if not np.isfinite(mean):
        raise ValueError(
            "mean doit être une valeur finie."
        )

    if not np.isfinite(variance):
        raise ValueError(
            "variance doit être une valeur finie."
        )

    if variance <= 0:
        raise ValueError(
            "variance doit être strictement positive."
        )

    coefficient = 1.0 / np.sqrt(
        2.0 * np.pi * variance
    )

    exponent = -(
        (values - mean) ** 2
    ) / (2.0 * variance)

    return coefficient * np.exp(exponent)


# ============================================================
# FIT
# ============================================================


def fit_naive_bayes(
    X: np.ndarray | list,
    y: np.ndarray | list,
    variance_smoothing: float = 1e-9,
) -> dict[str, Any]:
    """
    Entraîne un modèle Gaussian Naive Bayes.

    Parameters
    ----------
    X : array-like
        Données d'entraînement.
    y : array-like
        Labels d'entraînement.
    variance_smoothing : float, default=1e-9
        Petite valeur ajoutée aux variances pour éviter
        les variances nulles.

    Returns
    -------
    dict
        Paramètres appris :

        - classes
        - priors
        - means
        - variances
        - variance_smoothing
        - n_features
        - n_classes
        - n_samples

    Raises
    ------
    ValueError
        Si les données ou paramètres sont invalides.
    """

    X_array, y_array = _validate_input(X, y)

    if variance_smoothing <= 0:
        raise ValueError(
            "variance_smoothing doit être strictement positif."
        )

    if not np.isfinite(variance_smoothing):
        raise ValueError(
            "variance_smoothing doit être une valeur finie."
        )

    classes = np.unique(y_array)

    if classes.size < 2:
        raise ValueError(
            "y doit contenir au moins deux classes."
        )

    n_samples = X_array.shape[0]
    n_features = X_array.shape[1]
    n_classes = classes.size

    means = np.zeros(
        (n_classes, n_features),
        dtype=float,
    )

    variances = np.zeros(
        (n_classes, n_features),
        dtype=float,
    )

    priors = np.zeros(
        n_classes,
        dtype=float,
    )

    # --------------------------------------------------------
    # Estimation des paramètres pour chaque classe
    # --------------------------------------------------------

    for class_index, class_value in enumerate(classes):

        X_class = X_array[y_array == class_value]

        class_count = X_class.shape[0]

        priors[class_index] = (
            class_count / n_samples
        )

        means[class_index] = np.mean(
            X_class,
            axis=0,
        )

        # ddof=0 = variance de population.
        variances[class_index] = np.var(
            X_class,
            axis=0,
            ddof=0,
        )

    # --------------------------------------------------------
    # Évite les variances nulles
    # --------------------------------------------------------

    variances = variances + variance_smoothing

    _validate_model_parameters(
        means,
        variances,
        priors,
        classes,
    )

    return {
        "classes": classes,
        "priors": priors,
        "means": means,
        "variances": variances,
        "variance_smoothing": variance_smoothing,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_samples": n_samples,
    }


# ============================================================
# PREDICT PROBABILITIES
# ============================================================


def predict_proba_naive_bayes(
    X: np.ndarray | list,
    model: dict[str, Any],
) -> np.ndarray:
    """
    Calcule les probabilités de chaque classe.

    Parameters
    ----------
    X : array-like
        Observations à classifier.
    model : dict
        Modèle retourné par fit_naive_bayes().

    Returns
    -------
    np.ndarray
        Matrice de probabilités de forme :

        (n_samples, n_classes)
    """

    X_array, _ = _validate_input(X)

    required_keys = {
        "classes",
        "priors",
        "means",
        "variances",
    }

    missing_keys = required_keys - set(model.keys())

    if missing_keys:
        raise ValueError(
            "Modèle incomplet. Clés manquantes : "
            f"{sorted(missing_keys)}"
        )

    classes = np.asarray(model["classes"])
    priors = np.asarray(model["priors"], dtype=float)
    means = np.asarray(model["means"], dtype=float)
    variances = np.asarray(
        model["variances"],
        dtype=float,
    )

    _validate_model_parameters(
        means,
        variances,
        priors,
        classes,
    )

    if X_array.shape[1] != means.shape[1]:
        raise ValueError(
            "X doit avoir le même nombre de caractéristiques "
            "que les données d'entraînement."
        )

    n_samples = X_array.shape[0]
    n_classes = classes.shape[0]

    log_probabilities = np.zeros(
        (n_samples, n_classes),
        dtype=float,
    )

    # --------------------------------------------------------
    # Calcul en logarithme pour améliorer la stabilité
    # numérique.
    # --------------------------------------------------------

    for class_index in range(n_classes):

        class_mean = means[class_index]
        class_variance = variances[class_index]

        log_prior = np.log(
            priors[class_index]
        )

        log_likelihood = np.zeros(
            n_samples,
            dtype=float,
        )

        for feature_index in range(X_array.shape[1]):

            probabilities = gaussian_probability(
                X_array[:, feature_index],
                class_mean[feature_index],
                class_variance[feature_index],
            )

            # Protection contre log(0)
            probabilities = np.maximum(
                probabilities,
                np.finfo(float).tiny,
            )

            log_likelihood += np.log(
                probabilities
            )

        log_probabilities[:, class_index] = (
            log_prior + log_likelihood
        )

    # --------------------------------------------------------
    # Softmax stable
    # --------------------------------------------------------

    max_log = np.max(
        log_probabilities,
        axis=1,
        keepdims=True,
    )

    exp_probabilities = np.exp(
        log_probabilities - max_log
    )

    probabilities = (
        exp_probabilities
        / np.sum(
            exp_probabilities,
            axis=1,
            keepdims=True,
        )
    )

    return probabilities


# ============================================================
# PREDICT
# ============================================================


def predict_naive_bayes(
    X: np.ndarray | list,
    model: dict[str, Any],
) -> np.ndarray:
    """
    Prédit la classe de chaque observation.

    Parameters
    ----------
    X : array-like
        Observations à classifier.
    model : dict
        Modèle Gaussian Naive Bayes.

    Returns
    -------
    np.ndarray
        Classes prédites.
    """

    probabilities = predict_proba_naive_bayes(
        X,
        model,
    )

    classes = np.asarray(
        model["classes"]
    )

    class_indices = np.argmax(
        probabilities,
        axis=1,
    )

    return classes[class_indices]


# ============================================================
# ACCURACY
# ============================================================


def accuracy_score(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
) -> float:
    """
    Calcule l'accuracy.

    Parameters
    ----------
    y_true : array-like
        Labels réels.
    y_pred : array-like
        Labels prédits.

    Returns
    -------
    float
        Accuracy comprise entre 0 et 1.
    """

    true = np.asarray(y_true)

    predicted = np.asarray(y_pred)

    if true.ndim != 1:
        raise ValueError(
            "y_true doit être un vecteur 1D."
        )

    if predicted.ndim != 1:
        raise ValueError(
            "y_pred doit être un vecteur 1D."
        )

    if true.shape[0] == 0:
        raise ValueError(
            "y_true ne doit pas être vide."
        )

    if true.shape[0] != predicted.shape[0]:
        raise ValueError(
            "y_true et y_pred doivent avoir la même longueur."
        )

    return float(
        np.mean(true == predicted)
    )


# ============================================================
# PIPELINE COMPLET
# ============================================================


def naive_bayes(
    X_train: np.ndarray | list,
    y_train: np.ndarray | list,
    X_test: np.ndarray | list,
    y_test: np.ndarray | list | None = None,
    variance_smoothing: float = 1e-9,
) -> dict[str, Any]:
    """
    Pipeline complet Gaussian Naive Bayes.

    Parameters
    ----------
    X_train : array-like
        Données d'entraînement.
    y_train : array-like
        Labels d'entraînement.
    X_test : array-like
        Données de test.
    y_test : array-like, optional
        Labels de test.
    variance_smoothing : float, default=1e-9
        Lissage des variances.

    Returns
    -------
    dict
        Résultats complets du modèle.
    """

    X_train_array, y_train_array = _validate_input(
        X_train,
        y_train,
    )

    X_test_array, y_test_array = _validate_input(
        X_test,
        y_test,
    )

    if X_train_array.shape[1] != X_test_array.shape[1]:
        raise ValueError(
            "X_train et X_test doivent avoir le même nombre "
            "de caractéristiques."
        )

    model = fit_naive_bayes(
        X_train_array,
        y_train_array,
        variance_smoothing=variance_smoothing,
    )

    probabilities = predict_proba_naive_bayes(
        X_test_array,
        model,
    )

    predictions = predict_naive_bayes(
        X_test_array,
        model,
    )

    accuracy = None

    if y_test_array is not None:
        accuracy = accuracy_score(
            y_test_array,
            predictions,
        )

    return {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "classes": model["classes"],
        "priors": model["priors"],
        "means": model["means"],
        "variances": model["variances"],
        "accuracy": accuracy,
        "variance_smoothing": variance_smoothing,
        "n_train_samples": X_train_array.shape[0],
        "n_test_samples": X_test_array.shape[0],
        "n_features": X_train_array.shape[1],
        "n_classes": model["n_classes"],
    }

