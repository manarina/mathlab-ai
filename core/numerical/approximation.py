
from __future__ import annotations

"""
Méthodes d'approximation et de régression numérique.

Ce module fournit des fonctions pour :

- ajuster un polynôme aux données par la méthode des
  moindres carrés ;
- évaluer le modèle obtenu ;
- calculer les erreurs d'approximation ;
- calculer l'erreur quadratique moyenne (RMSE).

Les fonctions sont volontairement indépendantes de Streamlit.
L'interface utilisateur sera ajoutée ultérieurement dans
pages/05_Numerical.py.
"""

import math

import numpy as np


# ============================================================
# VALIDATION
# ============================================================


def _validate_data(
    x_values,
    y_values,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valide et convertit les données d'entrée.

    Parameters
    ----------
    x_values:
        Valeurs indépendantes.

    y_values:
        Valeurs observées.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Les deux séries converties en tableaux float.

    Raises
    ------
    ValueError
        Si les listes sont vides, de longueurs différentes,
        ou contiennent des valeurs non finies.
    TypeError
        Si les données ne peuvent pas être converties
        en valeurs numériques.
    """

    if isinstance(x_values, (str, bytes)):
        raise TypeError(
            "x_values doit être une séquence numérique."
        )

    if isinstance(y_values, (str, bytes)):
        raise TypeError(
            "y_values doit être une séquence numérique."
        )

    try:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Les données doivent être numériques."
        ) from error

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "Les données doivent être des séquences "
            "à une seule dimension."
        )

    if len(x) == 0 or len(y) == 0:
        raise ValueError(
            "Les données ne peuvent pas être vides."
        )

    if len(x) != len(y):
        raise ValueError(
            "x_values et y_values doivent avoir "
            "la même longueur."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "x_values contient des valeurs non finies."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y_values contient des valeurs non finies."
        )

    return x, y


def _validate_degree(
    degree: int,
    number_of_points: int,
) -> None:
    """
    Vérifie que le degré du polynôme est valide.
    """

    if isinstance(degree, bool) or not isinstance(
        degree,
        (int, np.integer),
    ):
        raise TypeError(
            "Le degré du polynôme doit être un entier."
        )

    if degree < 0:
        raise ValueError(
            "Le degré du polynôme doit être positif ou nul."
        )

    if number_of_points < degree + 1:
        raise ValueError(
            "Le nombre de points doit être supérieur "
            "ou égal à degré + 1."
        )


# ============================================================
# APPROXIMATION POLYNOMIALE
# ============================================================


def polynomial_approximation(
    x_values,
    y_values,
    degree: int,
) -> np.ndarray:
    """
    Calcule une approximation polynomiale par moindres carrés.

    Le modèle recherché est :

        P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ

    où ``degree = n``.

    Les coefficients sont retournés dans l'ordre croissant
    des puissances :

        [a₀, a₁, ..., aₙ]

    Parameters
    ----------
    x_values:
        Valeurs x des observations.

    y_values:
        Valeurs y des observations.

    degree:
        Degré du polynôme d'approximation.

    Returns
    -------
    np.ndarray
        Coefficients du polynôme.

    Examples
    --------
    >>> polynomial_approximation(
    ...     [0, 1, 2],
    ...     [1, 3, 7],
    ...     2,
    ... )
    array([1., 1., 1.])
    """

    x, y = _validate_data(
        x_values,
        y_values,
    )

    _validate_degree(
        degree,
        len(x),
    )

    # np.polyfit retourne les coefficients dans l'ordre
    # décroissant des puissances.
    coefficients_descending = np.polyfit(
        x,
        y,
        degree,
    )

    # Notre API utilise l'ordre :
    # [a₀, a₁, ..., aₙ]
    coefficients = coefficients_descending[::-1]

    return np.asarray(
        coefficients,
        dtype=float,
    )


# ============================================================
# ÉVALUATION DU MODÈLE
# ============================================================


def evaluate_polynomial(
    coefficients,
    x,
):
    """
    Évalue un polynôme à partir de ses coefficients.

    Les coefficients doivent être fournis dans l'ordre :

        [a₀, a₁, ..., aₙ]

    correspondant à :

        P(x) = a₀ + a₁x + ... + aₙxⁿ

    Parameters
    ----------
    coefficients:
        Coefficients du polynôme.

    x:
        Valeur ou ensemble de valeurs à évaluer.

    Returns
    -------
    float ou np.ndarray
        Valeur(s) du polynôme.

    Examples
    --------
    >>> evaluate_polynomial(
    ...     [1, 1, 1],
    ...     2,
    ... )
    7.0
    """

    try:
        coefficients_array = np.asarray(
            coefficients,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Les coefficients doivent être numériques."
        ) from error

    if coefficients_array.ndim != 1:
        raise ValueError(
            "Les coefficients doivent former "
            "une séquence à une dimension."
        )

    if len(coefficients_array) == 0:
        raise ValueError(
            "La liste des coefficients ne peut pas être vide."
        )

    if not np.all(
        np.isfinite(coefficients_array)
    ):
        raise ValueError(
            "Les coefficients doivent être finis."
        )

    try:
        x_array = np.asarray(
            x,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "La valeur x doit être numérique."
        ) from error

    if not np.all(np.isfinite(x_array)):
        raise ValueError(
            "La valeur x doit être finie."
        )

    # Évaluation stable du polynôme avec Horner.
    result = np.zeros_like(
        x_array,
        dtype=float,
    )

    for coefficient in coefficients_array[::-1]:
        result = (
            result * x_array
            + coefficient
        )

    if x_array.ndim == 0:
        return float(result)

    return result


# ============================================================
# VALEURS PRÉDITES
# ============================================================


def predicted_values(
    coefficients,
    x_values,
) -> np.ndarray:
    """
    Calcule les valeurs prédites par le modèle.

    Parameters
    ----------
    coefficients:
        Coefficients du polynôme.

    x_values:
        Valeurs x.

    Returns
    -------
    np.ndarray
        Valeurs P(x).
    """

    result = evaluate_polynomial(
        coefficients,
        x_values,
    )

    return np.asarray(
        result,
        dtype=float,
    )


# ============================================================
# ERREURS D'APPROXIMATION
# ============================================================


def absolute_errors(
    y_values,
    predicted,
) -> np.ndarray:
    """
    Calcule les erreurs absolues.

    Formule :

        Eᵢ = |yᵢ - ŷᵢ|

    Parameters
    ----------
    y_values:
        Valeurs observées.

    predicted:
        Valeurs prédites.

    Returns
    -------
    np.ndarray
        Erreurs absolues.
    """

    try:
        y = np.asarray(
            y_values,
            dtype=float,
        )

        y_predicted = np.asarray(
            predicted,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Les valeurs doivent être numériques."
        ) from error

    if y.shape != y_predicted.shape:
        raise ValueError(
            "y_values et predicted doivent avoir "
            "la même forme."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y_values contient des valeurs non finies."
        )

    if not np.all(
        np.isfinite(y_predicted)
    ):
        raise ValueError(
            "predicted contient des valeurs non finies."
        )

    return np.abs(
        y - y_predicted
    )


def relative_errors(
    y_values,
    predicted,
) -> np.ndarray:
    """
    Calcule les erreurs relatives.

    Formule :

        Eᵣ = |y - ŷ| / |y|

    Lorsque y = 0, l'erreur relative n'est pas définie.
    Dans ce cas, ``np.nan`` est retourné.

    Returns
    -------
    np.ndarray
        Erreurs relatives.
    """

    try:
        y = np.asarray(
            y_values,
            dtype=float,
        )

        y_predicted = np.asarray(
            predicted,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Les valeurs doivent être numériques."
        ) from error

    if y.shape != y_predicted.shape:
        raise ValueError(
            "y_values et predicted doivent avoir "
            "la même forme."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y_values contient des valeurs non finies."
        )

    if not np.all(
        np.isfinite(y_predicted)
    ):
        raise ValueError(
            "predicted contient des valeurs non finies."
        )

    errors = np.full(
        y.shape,
        np.nan,
        dtype=float,
    )

    non_zero = y != 0

    errors[non_zero] = (
        np.abs(
            y[non_zero]
            - y_predicted[non_zero]
        )
        / np.abs(y[non_zero])
    )

    return errors


# ============================================================
# RMSE
# ============================================================


def root_mean_square_error(
    y_values,
    predicted,
) -> float:
    """
    Calcule l'erreur quadratique moyenne racine (RMSE).

    Formule :

        RMSE = sqrt(
            (1/n) Σ(yᵢ - ŷᵢ)²
        )

    Returns
    -------
    float
        RMSE du modèle.
    """

    errors = absolute_errors(
        y_values,
        predicted,
    )

    return float(
        math.sqrt(
            np.mean(errors**2)
        )
    )


# ============================================================
# R²
# ============================================================


def coefficient_of_determination(
    y_values,
    predicted,
) -> float:
    """
    Calcule le coefficient de détermination R².

    Formule :

        R² = 1 - SSE / SST

    avec :

        SSE = Σ(yᵢ - ŷᵢ)²
        SST = Σ(yᵢ - ȳ)²

    Returns
    -------
    float
        Coefficient R².

    Raises
    ------
    ValueError
        Si toutes les valeurs observées sont identiques.
    """

    try:
        y = np.asarray(
            y_values,
            dtype=float,
        )

        y_predicted = np.asarray(
            predicted,
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Les valeurs doivent être numériques."
        ) from error

    if y.shape != y_predicted.shape:
        raise ValueError(
            "y_values et predicted doivent avoir "
            "la même forme."
        )

    if len(y) == 0:
        raise ValueError(
            "Les données ne peuvent pas être vides."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y_values contient des valeurs non finies."
        )

    if not np.all(
        np.isfinite(y_predicted)
    ):
        raise ValueError(
            "predicted contient des valeurs non finies."
        )

    total_sum_of_squares = np.sum(
        (y - np.mean(y)) ** 2
    )

    if total_sum_of_squares == 0:
        raise ValueError(
            "R² n'est pas défini lorsque toutes les "
            "valeurs observées sont identiques."
        )

    residual_sum_of_squares = np.sum(
        (y - y_predicted) ** 2
    )

    return float(
        1
        - (
            residual_sum_of_squares
            / total_sum_of_squares
        )
    )

