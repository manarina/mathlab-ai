"""
Algorithmes de recherche pour MathLab AI.

Ce module contient plusieurs algorithmes classiques permettant
de rechercher une valeur dans une séquence :

- Recherche linéaire
- Recherche binaire
- Recherche par saut
- Recherche par interpolation
- Recherche exponentielle

Convention :
    - retourne l'indice de la première occurrence trouvée ;
    - retourne -1 si la valeur n'est pas trouvée.

Les algorithmes nécessitant des données triées ne trient jamais
automatiquement la séquence fournie.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence


# ============================================================
# Validation
# ============================================================


def _validate_data(data: Sequence) -> None:
    """
    Vérifie que les données sont une séquence exploitable.

    Parameters
    ----------
    data : Sequence
        Séquence dans laquelle effectuer la recherche.

    Raises
    ------
    TypeError
        Si data n'est pas une séquence.
    """
    if isinstance(data, (str, bytes)):
        raise TypeError("Les données doivent être une séquence de valeurs.")

    if not isinstance(data, Sequence):
        raise TypeError("Les données doivent être une séquence de valeurs.")


def _validate_sorted_data(data: Sequence) -> None:
    """
    Vérifie que les données sont triées dans l'ordre croissant.

    Parameters
    ----------
    data : Sequence
        Séquence à vérifier.

    Raises
    ------
    ValueError
        Si les données ne sont pas triées.
    """
    _validate_data(data)

    for index in range(len(data) - 1):
        try:
            if data[index] > data[index + 1]:
                raise ValueError(
                    "Les données doivent être triées dans l'ordre croissant."
                )
        except TypeError as exc:
            raise TypeError(
                "Les éléments de la séquence doivent être comparables."
            ) from exc


def _validate_numeric_sorted_data(data: Sequence) -> None:
    """
    Vérifie que les données sont numériques et triées.
    """
    _validate_sorted_data(data)

    for value in data:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "Les données doivent contenir uniquement des valeurs numériques."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                "Les données doivent contenir uniquement des valeurs finies."
            )


# ============================================================
# 1. Linear Search
# ============================================================


def linear_search(data: Sequence, target) -> int:
    """
    Recherche linéaire d'une valeur.

    L'algorithme parcourt les éléments un par un jusqu'à trouver
    la cible.

    Complexité temporelle :
        O(n)

    Complexité spatiale :
        O(1)

    Parameters
    ----------
    data : Sequence
        Séquence à parcourir.
    target
        Valeur recherchée.

    Returns
    -------
    int
        Indice de la première occurrence de target,
        ou -1 si target n'est pas trouvé.
    """
    _validate_data(data)

    for index, value in enumerate(data):
        if value == target:
            return index

    return -1


# ============================================================
# 2. Binary Search
# ============================================================


def binary_search(data: Sequence, target) -> int:
    """
    Recherche binaire d'une valeur dans une séquence triée.

    L'algorithme divise successivement l'espace de recherche
    en deux parties.

    Complexité temporelle :
        O(log n)

    Complexité spatiale :
        O(1)

    Parameters
    ----------
    data : Sequence
        Séquence triée dans l'ordre croissant.
    target
        Valeur recherchée.

    Returns
    -------
    int
        Indice de target,
        ou -1 si target n'est pas trouvé.
    """
    _validate_sorted_data(data)

    left = 0
    right = len(data) - 1

    while left <= right:
        middle = left + (right - left) // 2

        if data[middle] == target:
            return middle

        if data[middle] < target:
            left = middle + 1
        else:
            right = middle - 1

    return -1


# ============================================================
# 3. Jump Search
# ============================================================


def jump_search(data: Sequence, target) -> int:
    """
    Recherche par saut dans une séquence triée.

    L'algorithme avance par blocs de taille approximativement
    égale à sqrt(n), puis effectue une recherche linéaire dans
    le bloc susceptible de contenir la cible.

    Complexité temporelle :
        O(sqrt(n))

    Complexité spatiale :
        O(1)

    Parameters
    ----------
    data : Sequence
        Séquence triée dans l'ordre croissant.
    target
        Valeur recherchée.

    Returns
    -------
    int
        Indice de target,
        ou -1 si target n'est pas trouvé.
    """
    _validate_sorted_data(data)

    n = len(data)

    if n == 0:
        return -1

    jump = max(1, int(math.sqrt(n)))

    previous = 0
    current = 0

    while current < n and data[current] < target:
        previous = current
        current = min(current + jump, n - 1)

        if data[current] >= target:
            break

        if current == n - 1:
            break

    end = min(current + 1, n)

    for index in range(previous, end):
        if data[index] == target:
            return index

        if data[index] > target:
            return -1

    return -1


# ============================================================
# 4. Interpolation Search
# ============================================================


def interpolation_search(data: Sequence, target) -> int:
    """
    Recherche par interpolation dans une séquence numérique triée.

    L'algorithme estime la position probable de la cible à partir
    de sa valeur et des valeurs situées aux extrémités de
    l'intervalle de recherche.

    Il est particulièrement efficace lorsque les valeurs sont
    relativement uniformément distribuées.

    Complexité moyenne :
        O(log log n)

    Complexité dans le pire cas :
        O(n)

    Complexité spatiale :
        O(1)

    Parameters
    ----------
    data : Sequence[Real]
        Séquence numérique triée dans l'ordre croissant.
    target : Real
        Valeur numérique recherchée.

    Returns
    -------
    int
        Indice de target,
        ou -1 si target n'est pas trouvé.
    """
    _validate_numeric_sorted_data(data)

    if isinstance(target, bool) or not isinstance(target, Real):
        raise TypeError("La cible doit être une valeur numérique.")

    target_value = float(target)

    if not math.isfinite(target_value):
        raise ValueError("La cible doit être une valeur finie.")

    if not data:
        return -1

    low = 0
    high = len(data) - 1

    while (
        low <= high
        and float(data[low]) <= target_value <= float(data[high])
    ):
        low_value = float(data[low])
        high_value = float(data[high])

        # Toutes les valeurs restantes sont identiques.
        if high_value == low_value:
            if low_value == target_value:
                return low
            return -1

        position = low + int(
            (target_value - low_value)
            * (high - low)
            / (high_value - low_value)
        )

        if position < low or position > high:
            return -1

        value = float(data[position])

        if value == target_value:
            return position

        if value < target_value:
            low = position + 1
        else:
            high = position - 1

    return -1


# ============================================================
# 5. Exponential Search
# ============================================================


def exponential_search(data: Sequence, target) -> int:
    """
    Recherche exponentielle dans une séquence triée.

    L'algorithme commence avec une petite zone puis double
    progressivement la taille de la zone de recherche jusqu'à
    dépasser la cible. Une recherche binaire est ensuite utilisée
    dans la zone identifiée.

    Complexité temporelle :
        O(log n)

    Complexité spatiale :
        O(1)

    Parameters
    ----------
    data : Sequence
        Séquence triée dans l'ordre croissant.
    target
        Valeur recherchée.

    Returns
    -------
    int
        Indice de target,
        ou -1 si target n'est pas trouvé.
    """
    _validate_sorted_data(data)

    n = len(data)

    if n == 0:
        return -1

    if data[0] == target:
        return 0

    bound = 1

    while bound < n and data[bound] < target:
        bound *= 2

    left = bound // 2
    right = min(bound, n - 1)

    while left <= right:
        middle = left + (right - left) // 2

        if data[middle] == target:
            return middle

        if data[middle] < target:
            left = middle + 1
        else:
            right = middle - 1

    return -1