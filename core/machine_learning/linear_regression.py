
"""
Linear Regression
=================

Implémentation pédagogique de la régression linéaire simple
par la méthode des moindres carrés ordinaires (OLS).

La fonction principale permet de :
- calculer l'intercept et la pente ;
- effectuer des prédictions ;
- calculer le MSE ;
- calculer le coefficient de détermination R².

Cette implémentation utilise uniquement NumPy.
"""

import numpy as np


def _validate_input(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Valide et normalise les données d'entrée.

    Parameters
    ----------
    x : np.ndarray
        Variable indépendante.
    y : np.ndarray
        Variable dépendante.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Les tableaux x et y convertis en float.

    Raises
    ------
    ValueError
        Si les données sont invalides.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1:
        raise ValueError("x doit être un tableau unidimensionnel.")

    if y.ndim != 1:
        raise ValueError("y doit être un tableau unidimensionnel.")

    if len(x) == 0 or len(y) == 0:
        raise ValueError("x et y ne doivent pas être vides.")

    if len(x) != len(y):
        raise ValueError("x et y doivent avoir la même longueur.")

    if len(x) < 2:
        raise ValueError(
            "La régression linéaire nécessite au moins deux observations."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("x contient des valeurs non finies.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contient des valeurs non finies.")

    if np.all(x == x[0]):
        raise ValueError(
            "La variable x doit contenir au moins deux valeurs différentes."
        )

    return x, y


def fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """
    Ajuste une régression linéaire simple.

    Le modèle est :

        y = intercept + slope * x

    Parameters
    ----------
    x : np.ndarray
        Variable indépendante.
    y : np.ndarray
        Variable dépendante.

    Returns
    -------
    tuple[float, float]
        (intercept, slope)
    """
    x, y = _validate_input(x, y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return float(intercept), float(slope)


def predict_linear_regression(
    x: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    """
    Effectue des prédictions avec un modèle linéaire.

    Parameters
    ----------
    x : np.ndarray
        Valeurs d'entrée.
    intercept : float
        Ordonnée à l'origine.
    slope : float
        Pente.

    Returns
    -------
    np.ndarray
        Valeurs prédites.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x doit être un tableau unidimensionnel.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x contient des valeurs non finies.")

    return intercept + slope * x


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcule l'erreur quadratique moyenne (MSE).

    Formule :

        MSE = (1/n) * Σ(y_true - y_pred)²

    Parameters
    ----------
    y_true : np.ndarray
        Valeurs réelles.
    y_pred : np.ndarray
        Valeurs prédites.

    Returns
    -------
    float
        MSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true et y_pred doivent être unidimensionnels.")

    if len(y_true) == 0:
        raise ValueError("Les données ne doivent pas être vides.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true et y_pred doivent avoir la même longueur."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contient des valeurs non finies.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contient des valeurs non finies.")

    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcule le coefficient de détermination R².

    Formule :

        R² = 1 - SS_res / SS_tot

    Parameters
    ----------
    y_true : np.ndarray
        Valeurs réelles.
    y_pred : np.ndarray
        Valeurs prédites.

    Returns
    -------
    float
        Coefficient R².

    Raises
    ------
    ValueError
        Si y_true est constant.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true et y_pred doivent être unidimensionnels.")

    if len(y_true) == 0:
        raise ValueError("Les données ne doivent pas être vides.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true et y_pred doivent avoir la même longueur."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contient des valeurs non finies.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contient des valeurs non finies.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if np.isclose(ss_tot, 0.0):
        raise ValueError(
            "R² ne peut pas être calculé lorsque y_true est constant."
        )

    return float(1.0 - (ss_res / ss_tot))


def linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Exécute une régression linéaire complète.

    Parameters
    ----------
    x : np.ndarray
        Variable indépendante.
    y : np.ndarray
        Variable dépendante.

    Returns
    -------
    dict
        Résultats du modèle contenant :

        - intercept
        - slope
        - predictions
        - mse
        - r2
    """
    x, y = _validate_input(x, y)

    intercept, slope = fit_linear_regression(x, y)

    predictions = predict_linear_regression(
        x,
        intercept,
        slope,
    )

    mse = mean_squared_error(y, predictions)
    r2 = r_squared(y, predictions)

    return {
        "intercept": intercept,
        "slope": slope,
        "predictions": predictions,
        "mse": mse,
        "r2": r2,
    }

