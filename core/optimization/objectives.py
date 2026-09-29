
"""
Gestion des fonctions objectif pour l'Optimization Lab.

Ce module fournit des utilitaires permettant de valider et d'évaluer
une fonction objectif f(x) utilisée dans les problèmes d'optimisation
à une variable.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Callable


ObjectiveFunction = Callable[[float], float]


def validate_objective(function: ObjectiveFunction) -> ObjectiveFunction:
    """
    Vérifie qu'une fonction objectif est appelable.

    Parameters
    ----------
    function : callable
        Fonction objectif f(x).

    Returns
    -------
    callable
        La fonction validée.

    Raises
    ------
    TypeError
        Si l'objet fourni n'est pas appelable.
    """
    if not callable(function):
        raise TypeError("La fonction objectif doit être appelable.")

    return function


def evaluate_objective(
    function: ObjectiveFunction,
    x: Real,
) -> float:
    """
    Évalue une fonction objectif en un point x.

    Parameters
    ----------
    function : callable
        Fonction objectif f(x).
    x : Real
        Point auquel évaluer la fonction.

    Returns
    -------
    float
        Valeur f(x).

    Raises
    ------
    TypeError
        Si la fonction n'est pas appelable ou si x n'est pas numérique.
    ValueError
        Si x n'est pas fini ou si f(x) produit une valeur non finie.
    """
    validate_objective(function)

    if isinstance(x, bool) or not isinstance(x, Real):
        raise TypeError("Le point x doit être une valeur numérique.")

    x_value = float(x)

    if not math.isfinite(x_value):
        raise ValueError("Le point x doit être une valeur finie.")

    try:
        result = function(x_value)
    except Exception as exc:
        raise ValueError(
            f"Impossible d'évaluer la fonction objectif en x={x_value}."
        ) from exc

    if isinstance(result, bool) or not isinstance(result, Real):
        raise TypeError(
            "La fonction objectif doit retourner une valeur numérique."
        )

    result_value = float(result)

    if not math.isfinite(result_value):
        raise ValueError(
            "La fonction objectif retourne une valeur non finie."
        )

    return result_value


def is_valid_objective(function: ObjectiveFunction) -> bool:
    """
    Indique si un objet peut être utilisé comme fonction objectif.

    Parameters
    ----------
    function : callable
        Objet à vérifier.

    Returns
    -------
    bool
        True si l'objet est appelable, False sinon.
    """
    return callable(function)


def objective_name(function: ObjectiveFunction) -> str:
    """
    Retourne un nom lisible pour une fonction objectif.

    Parameters
    ----------
    function : callable
        Fonction objectif.

    Returns
    -------
    str
        Nom de la fonction.
    """
    validate_objective(function)

    name = getattr(function, "__name__", None)

    if name:
        return str(name)

    return function.__class__.__name__

