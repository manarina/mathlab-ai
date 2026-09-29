
"""
Gestion des contraintes simples pour l'Optimization Lab.

Ce module fournit des utilitaires pour gérer les contraintes de type
bornes sur une variable :

    lower_bound <= x <= upper_bound

Ces contraintes sont utilisées notamment pour les méthodes
d'optimisation à une variable définies sur un intervalle.
"""

from __future__ import annotations

import math
from numbers import Real


def _validate_bound_value(value: Real, name: str) -> float:
    """
    Valide une borne individuelle.

    Parameters
    ----------
    value : Real
        Valeur de la borne.
    name : str
        Nom de la borne pour le message d'erreur.

    Returns
    -------
    float
        Borne convertie en float.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} doit être une valeur numérique.")

    value_float = float(value)

    if not math.isfinite(value_float):
        raise ValueError(f"{name} doit être une valeur finie.")

    return value_float


def validate_bounds(
    lower_bound: Real,
    upper_bound: Real,
) -> tuple[float, float]:
    """
    Valide un intervalle de bornes.

    Parameters
    ----------
    lower_bound : Real
        Borne inférieure.
    upper_bound : Real
        Borne supérieure.

    Returns
    -------
    tuple[float, float]
        Les bornes validées sous la forme (lower_bound, upper_bound).

    Raises
    ------
    TypeError
        Si une borne n'est pas numérique.
    ValueError
        Si une borne n'est pas finie ou si lower_bound >= upper_bound.
    """
    lower = _validate_bound_value(lower_bound, "La borne inférieure")
    upper = _validate_bound_value(upper_bound, "La borne supérieure")

    if lower >= upper:
        raise ValueError(
            "La borne inférieure doit être strictement inférieure "
            "à la borne supérieure."
        )

    return lower, upper


def is_within_bounds(
    x: Real,
    lower_bound: Real,
    upper_bound: Real,
) -> bool:
    """
    Vérifie si une valeur appartient à un intervalle fermé.

    Parameters
    ----------
    x : Real
        Valeur à tester.
    lower_bound : Real
        Borne inférieure.
    upper_bound : Real
        Borne supérieure.

    Returns
    -------
    bool
        True si lower_bound <= x <= upper_bound.
    """
    lower, upper = validate_bounds(lower_bound, upper_bound)

    if isinstance(x, bool) or not isinstance(x, Real):
        raise TypeError("x doit être une valeur numérique.")

    x_value = float(x)

    if not math.isfinite(x_value):
        raise ValueError("x doit être une valeur finie.")

    return lower <= x_value <= upper


def clip_to_bounds(
    x: Real,
    lower_bound: Real,
    upper_bound: Real,
) -> float:
    """
    Ramène une valeur dans un intervalle fermé.

    Si x est déjà dans l'intervalle, sa valeur est conservée.
    Si x est inférieur à la borne minimale, la borne minimale est retournée.
    Si x est supérieur à la borne maximale, la borne maximale est retournée.

    Parameters
    ----------
    x : Real
        Valeur à contraindre.
    lower_bound : Real
        Borne inférieure.
    upper_bound : Real
        Borne supérieure.

    Returns
    -------
    float
        Valeur contrainte dans [lower_bound, upper_bound].
    """
    lower, upper = validate_bounds(lower_bound, upper_bound)

    if isinstance(x, bool) or not isinstance(x, Real):
        raise TypeError("x doit être une valeur numérique.")

    x_value = float(x)

    if not math.isfinite(x_value):
        raise ValueError("x doit être une valeur finie.")

    return max(lower, min(x_value, upper))


def bounds_width(
    lower_bound: Real,
    upper_bound: Real,
) -> float:
    """
    Calcule la largeur d'un intervalle.

    Parameters
    ----------
    lower_bound : Real
        Borne inférieure.
    upper_bound : Real
        Borne supérieure.

    Returns
    -------
    float
        Largeur de l'intervalle :

            upper_bound - lower_bound
    """
    lower, upper = validate_bounds(lower_bound, upper_bound)

    return upper - lower

