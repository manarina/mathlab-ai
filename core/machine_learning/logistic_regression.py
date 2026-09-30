
"""
Logistic Regression
===================

Implémentation pédagogique de la régression logistique binaire
avec NumPy uniquement.

Modèle :

    z = b0 + b1 * x

    sigmoid(z) = 1 / (1 + exp(-z))

    P(y=1 | x) = sigmoid(z)

L'apprentissage utilise la descente de gradient
avec la Binary Cross-Entropy comme fonction de coût.
"""

import numpy as np


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(x, y=None):
    """
    Valide les données d'entrée.

    Parameters
    ----------
    x : array-like
        Variables explicatives.
        Doit être un tableau 1D non vide.

    y : array-like, optional
        Variable cible binaire.
        Les valeurs doivent être 0 ou 1.

    Returns
    -------
    x_array : np.ndarray
        Tableau NumPy de type float.

    y_array : np.ndarray or None
        Tableau NumPy de type float si y est fourni.
    """
    x_array = np.asarray(x, dtype=float)

    if x_array.ndim != 1:
        raise ValueError("x doit être un tableau 1D.")

    if len(x_array) == 0:
        raise ValueError("x ne peut pas être vide.")

    if not np.all(np.isfinite(x_array)):
        raise ValueError("x doit contenir uniquement des valeurs finies.")

    if y is None:
        return x_array, None

    y_array = np.asarray(y, dtype=float)

    if y_array.ndim != 1:
        raise ValueError("y doit être un tableau 1D.")

    if len(y_array) == 0:
        raise ValueError("y ne peut pas être vide.")

    if len(x_array) != len(y_array):
        raise ValueError(
            "x et y doivent avoir le même nombre d'observations."
        )

    if not np.all(np.isfinite(y_array)):
        raise ValueError("y doit contenir uniquement des valeurs finies.")

    unique_values = np.unique(y_array)

    if not np.all(np.isin(unique_values, [0.0, 1.0])):
        raise ValueError(
            "La régression logistique binaire nécessite des labels 0 et 1."
        )

    return x_array, y_array


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(z):
    """
    Calcule la fonction sigmoïde de manière numériquement stable.

    Formule :

        sigmoid(z) = 1 / (1 + exp(-z))

    Parameters
    ----------
    z : float or array-like
        Valeur(s) à transformer.

    Returns
    -------
    float or np.ndarray
        Valeur(s) comprises entre 0 et 1.
    """
    z_array = np.asarray(z, dtype=float)

    if not np.all(np.isfinite(z_array)):
        raise ValueError("z doit contenir uniquement des valeurs finies.")

    result = np.empty_like(z_array, dtype=float)

    positive = z_array >= 0
    negative = ~positive

    # Pour z >= 0 :
    # sigmoid(z) = 1 / (1 + exp(-z))
    result[positive] = 1.0 / (
        1.0 + np.exp(-z_array[positive])
    )

    # Pour z < 0 :
    # sigmoid(z) = exp(z) / (1 + exp(z))
    # Cette forme évite les overflow de exp(-z).
    exp_z = np.exp(z_array[negative])

    result[negative] = exp_z / (1.0 + exp_z)

    # Retourner un scalaire si l'entrée était scalaire.
    if np.ndim(z) == 0:
        return float(result)

    return result


# ============================================================
# FIT
# ============================================================

def fit_logistic_regression(
    x,
    y,
    learning_rate=0.1,
    n_iterations=1000,
):
    """
    Entraîne une régression logistique binaire à une variable.

    Le modèle est :

        z = b0 + b1*x

        p = sigmoid(z)

    Les paramètres b0 et b1 sont appris par descente de gradient.

    Parameters
    ----------
    x : array-like
        Variable explicative 1D.

    y : array-like
        Labels binaires 0 ou 1.

    learning_rate : float, default=0.1
        Taux d'apprentissage.

    n_iterations : int, default=1000
        Nombre d'itérations.

    Returns
    -------
    intercept : float
        Intercept b0.

    coefficient : float
        Coefficient b1.
    """
    x_array, y_array = _validate_input(x, y)

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate doit être strictement positif."
        )

    if not np.isfinite(learning_rate):
        raise ValueError(
            "learning_rate doit être une valeur finie."
        )

    if not isinstance(n_iterations, (int, np.integer)):
        raise ValueError(
            "n_iterations doit être un entier."
        )

    if n_iterations <= 0:
        raise ValueError(
            "n_iterations doit être strictement positif."
        )

    # Initialisation des paramètres
    intercept = 0.0
    coefficient = 0.0

    n = len(x_array)

    # --------------------------------------------------------
    # Descente de gradient
    # --------------------------------------------------------

    for _ in range(n_iterations):

        # Fonction linéaire
        z = intercept + coefficient * x_array

        # Probabilités prédites
        probabilities = sigmoid(z)

        # Erreur
        error = probabilities - y_array

        # Gradient de l'intercept
        gradient_intercept = np.mean(error)

        # Gradient du coefficient
        gradient_coefficient = np.mean(
            error * x_array
        )

        # Mise à jour
        intercept -= learning_rate * gradient_intercept

        coefficient -= learning_rate * gradient_coefficient

    return float(intercept), float(coefficient)


# ============================================================
# PREDICTION DES PROBABILITÉS
# ============================================================

def predict_probability(
    x,
    intercept,
    coefficient,
):
    """
    Calcule la probabilité P(y=1 | x).

    Parameters
    ----------
    x : array-like
        Variable explicative.

    intercept : float
        Intercept du modèle.

    coefficient : float
        Coefficient du modèle.

    Returns
    -------
    np.ndarray
        Probabilités comprises entre 0 et 1.
    """
    x_array, _ = _validate_input(x)

    if not np.isfinite(intercept):
        raise ValueError("intercept doit être une valeur finie.")

    if not np.isfinite(coefficient):
        raise ValueError("coefficient doit être une valeur finie.")

    z = intercept + coefficient * x_array

    return sigmoid(z)


# ============================================================
# PREDICTION DES CLASSES
# ============================================================

def predict_class(
    x,
    intercept,
    coefficient,
    threshold=0.5,
):
    """
    Prédit les classes binaires 0 ou 1.

    Parameters
    ----------
    x : array-like
        Variable explicative.

    intercept : float
        Intercept du modèle.

    coefficient : float
        Coefficient du modèle.

    threshold : float, default=0.5
        Seuil de classification.

    Returns
    -------
    np.ndarray
        Classes prédites : 0 ou 1.
    """
    if not np.isfinite(threshold):
        raise ValueError(
            "threshold doit être une valeur finie."
        )

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold doit être compris entre 0 et 1."
        )

    probabilities = predict_probability(
        x,
        intercept,
        coefficient,
    )

    return (probabilities >= threshold).astype(int)


# ============================================================
# BINARY CROSS-ENTROPY
# ============================================================

def binary_cross_entropy(y_true, probabilities):
    """
    Calcule la Binary Cross-Entropy (log loss).

    Formule :

        BCE = -1/n * Σ [
            y log(p) + (1-y) log(1-p)
        ]

    Parameters
    ----------
    y_true : array-like
        Labels réels 0 ou 1.

    probabilities : array-like
        Probabilités prédites entre 0 et 1.

    Returns
    -------
    float
        Valeur de la Binary Cross-Entropy.
    """
    y_array = np.asarray(y_true, dtype=float)
    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if y_array.ndim != 1:
        raise ValueError(
            "y_true doit être un tableau 1D."
        )

    if probability_array.ndim != 1:
        raise ValueError(
            "probabilities doit être un tableau 1D."
        )

    if len(y_array) == 0:
        raise ValueError(
            "y_true ne peut pas être vide."
        )

    if len(y_array) != len(probability_array):
        raise ValueError(
            "y_true et probabilities doivent avoir "
            "la même longueur."
        )

    if not np.all(np.isfinite(y_array)):
        raise ValueError(
            "y_true doit contenir uniquement des valeurs finies."
        )

    if not np.all(np.isfinite(probability_array)):
        raise ValueError(
            "probabilities doit contenir uniquement "
            "des valeurs finies."
        )

    if not np.all(np.isin(y_array, [0.0, 1.0])):
        raise ValueError(
            "y_true doit contenir uniquement les valeurs 0 et 1."
        )

    if np.any(probability_array < 0) or np.any(
        probability_array > 1
    ):
        raise ValueError(
            "Les probabilités doivent être comprises entre 0 et 1."
        )

    # Évite log(0)
    epsilon = 1e-15

    probabilities_clipped = np.clip(
        probability_array,
        epsilon,
        1.0 - epsilon,
    )

    loss = -np.mean(
        y_array * np.log(probabilities_clipped)
        + (1.0 - y_array)
        * np.log(1.0 - probabilities_clipped)
    )

    return float(loss)


# ============================================================
# ACCURACY
# ============================================================

def accuracy_score(y_true, y_pred):
    """
    Calcule l'accuracy d'une classification binaire.

    Formule :

        Accuracy = nombre de prédictions correctes / n

    Parameters
    ----------
    y_true : array-like
        Labels réels 0 ou 1.

    y_pred : array-like
        Labels prédits 0 ou 1.

    Returns
    -------
    float
        Accuracy comprise entre 0 et 1.
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_true_array.ndim != 1:
        raise ValueError(
            "y_true doit être un tableau 1D."
        )

    if y_pred_array.ndim != 1:
        raise ValueError(
            "y_pred doit être un tableau 1D."
        )

    if len(y_true_array) == 0:
        raise ValueError(
            "y_true ne peut pas être vide."
        )

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            "y_true et y_pred doivent avoir la même longueur."
        )

    if not np.all(np.isin(y_true_array, [0.0, 1.0])):
        raise ValueError(
            "y_true doit contenir uniquement les valeurs 0 et 1."
        )

    if not np.all(np.isin(y_pred_array, [0.0, 1.0])):
        raise ValueError(
            "y_pred doit contenir uniquement les valeurs 0 et 1."
        )

    return float(
        np.mean(y_true_array == y_pred_array)
    )


# ============================================================
# PIPELINE COMPLET
# ============================================================

def logistic_regression(
    x,
    y,
    learning_rate=0.1,
    n_iterations=1000,
    threshold=0.5,
):
    """
    Exécute le pipeline complet de régression logistique.

    Étapes :

        1. Entraînement du modèle
        2. Calcul des probabilités
        3. Prédiction des classes
        4. Calcul de la Binary Cross-Entropy
        5. Calcul de l'accuracy

    Parameters
    ----------
    x : array-like
        Variable explicative.

    y : array-like
        Labels binaires 0 ou 1.

    learning_rate : float, default=0.1
        Taux d'apprentissage.

    n_iterations : int, default=1000
        Nombre d'itérations.

    threshold : float, default=0.5
        Seuil de classification.

    Returns
    -------
    dict
        Résultats du modèle.
    """
    x_array, y_array = _validate_input(x, y)

    intercept, coefficient = fit_logistic_regression(
        x_array,
        y_array,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
    )

    probabilities = predict_probability(
        x_array,
        intercept,
        coefficient,
    )

    predictions = predict_class(
        x_array,
        intercept,
        coefficient,
        threshold=threshold,
    )

    loss = binary_cross_entropy(
        y_array,
        probabilities,
    )

    accuracy = accuracy_score(
        y_array,
        predictions,
    )

    return {
        "intercept": intercept,
        "coefficient": coefficient,
        "probabilities": probabilities,
        "predictions": predictions,
        "loss": loss,
        "accuracy": accuracy,
        "threshold": threshold,
        "learning_rate": learning_rate,
        "n_iterations": n_iterations,
    }

