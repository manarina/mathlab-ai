
"""
Méthodes d'intégration numérique.

Ce module fournit des méthodes classiques pour approximer
l'intégrale définie d'une fonction :

    ∫_a^b f(x) dx

Méthodes disponibles :
- Méthode des rectangles (point milieu)
- Méthode des trapèzes
- Méthode de Simpson 1/3
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _validate_function(function: Callable[[float], float]) -> None:
    """Valide que l'objet fourni est appelable."""
    if not callable(function):
        raise TypeError("function doit être une fonction appelable.")


def _validate_bounds(a: float, b: float) -> None:
    """Valide les bornes d'intégration."""
    if not np.isscalar(a) or not np.isscalar(b):
        raise TypeError("Les bornes a et b doivent être des nombres.")

    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Les bornes a et b doivent être finies.")

    if a >= b:
        raise ValueError(
            "La borne inférieure a doit être strictement inférieure à b."
        )


def _validate_subdivisions(n: int) -> None:
    """Valide le nombre de subdivisions."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("n doit être un entier.")

    if n <= 0:
        raise ValueError("n doit être strictement positif.")


def _evaluate_function(
    function: Callable[[float], float],
    x_values: np.ndarray,
) -> np.ndarray:
    """
    Évalue la fonction sur un ensemble de points.

    La fonction peut être vectorisée ou accepter uniquement
    des valeurs scalaires.

    Les valeurs retournées doivent être numériques et finies.
    """

    # --------------------------------------------------------
    # 1. Tentative d'évaluation vectorisée
    # --------------------------------------------------------
    try:
        raw_values = function(x_values)
        values = np.asarray(raw_values, dtype=float)

        if values.shape == x_values.shape:
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "La fonction retourne des valeurs non finies."
                )

            return values

    except ValueError as exc:
        # Une erreur de type "valeurs non finies" doit être
        # conservée.
        if "non finies" in str(exc):
            raise

        # Une erreur de conversion vers float signifie que la
        # fonction n'est probablement pas vectorisée.
        pass

    except (TypeError, OverflowError):
        # La fonction ne supporte probablement pas les tableaux.
        pass

    # --------------------------------------------------------
    # 2. Évaluation point par point
    # --------------------------------------------------------
    values_list = []

    for x in x_values:
        try:
            value = function(float(x))
            value = float(value)

        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "La fonction doit retourner des valeurs numériques."
            ) from exc

        if not np.isfinite(value):
            raise ValueError(
                "La fonction retourne des valeurs non finies."
            )

        values_list.append(value)

    values = np.asarray(values_list, dtype=float)

    return values


def rectangle_rule(
    function: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> float:
    """
    Approxime une intégrale avec la méthode des rectangles
    au point milieu.

    Formule :

        ∫_a^b f(x) dx
        ≈ h Σ f(x_i*)

    où x_i* représente le milieu de chaque sous-intervalle.
    """
    _validate_function(function)
    _validate_bounds(a, b)
    _validate_subdivisions(n)

    a = float(a)
    b = float(b)

    h = (b - a) / n

    midpoints = a + (np.arange(n, dtype=float) + 0.5) * h

    values = _evaluate_function(function, midpoints)

    result = h * np.sum(values)

    return float(result)


def trapezoidal_rule(
    function: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> float:
    """
    Approxime une intégrale avec la méthode des trapèzes.

    Formule :

        ∫_a^b f(x) dx
        ≈ h/2 [f(a) + f(b) + 2Σf(x_i)]
    """
    _validate_function(function)
    _validate_bounds(a, b)
    _validate_subdivisions(n)

    a = float(a)
    b = float(b)

    h = (b - a) / n

    x_values = np.linspace(a, b, n + 1)

    values = _evaluate_function(function, x_values)

    result = h * (
        0.5 * values[0]
        + np.sum(values[1:-1])
        + 0.5 * values[-1]
    )

    return float(result)


def simpson_one_third(
    function: Callable[[float], float],
    a: float,
    b: float,
    n: int,
) -> float:
    """
    Approxime une intégrale avec la méthode de Simpson 1/3.

    La méthode nécessite un nombre pair de subdivisions.

    Formule :

        ∫_a^b f(x) dx
        ≈ h/3 [
            f(x_0)
            + f(x_n)
            + 4Σf(x_i) pour i impair
            + 2Σf(x_i) pour i pair
        ]
    """
    _validate_function(function)
    _validate_bounds(a, b)
    _validate_subdivisions(n)

    if n % 2 != 0:
        raise ValueError(
            "La méthode de Simpson 1/3 nécessite un nombre pair "
            "de subdivisions."
        )

    a = float(a)
    b = float(b)

    h = (b - a) / n

    x_values = np.linspace(a, b, n + 1)

    values = _evaluate_function(function, x_values)

    odd_sum = np.sum(values[1:-1:2])
    even_sum = np.sum(values[2:-1:2])

    result = (h / 3) * (
        values[0]
        + values[-1]
        + 4 * odd_sum
        + 2 * even_sum
    )

    return float(result)

