from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Iterable


Number = int | float


def _validate_data(data: Iterable[Number]) -> tuple[Number, ...]:
    """
    Valide et transforme les données statistiques en tuple.

    Parameters
    ----------
    data:
        Suite de valeurs numériques.

    Returns
    -------
    tuple[Number, ...]
        Données validées.

    Raises
    ------
    ValueError
        Si les données sont vides.
    TypeError
        Si une valeur n'est pas numérique.
    """
    values = tuple(data)

    if not values:
        raise ValueError(
            "Les données statistiques ne peuvent pas être vides."
        )

    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        for value in values
    ):
        raise TypeError(
            "Toutes les données doivent être numériques."
        )

    return values


def mean(data: Iterable[Number]) -> float:
    """
    Calcule la moyenne arithmétique.

    Exemple
    -------
    mean([10, 12, 14]) -> 12.0
    """
    values = _validate_data(data)

    return sum(values) / len(values)


def median(data: Iterable[Number]) -> float:
    """
    Calcule la médiane.

    Les données sont automatiquement triées.

    Exemple
    -------
    median([5, 1, 3]) -> 3.0
    median([1, 2, 3, 4]) -> 2.5
    """
    values = sorted(_validate_data(data))
    n = len(values)

    middle = n // 2

    if n % 2 == 1:
        return float(values[middle])

    return (values[middle - 1] + values[middle]) / 2


def mode(data: Iterable[Number]) -> Number | tuple[Number, ...]:
    """
    Détermine le ou les modes d'une série.

    Si une seule valeur possède la fréquence maximale,
    cette valeur est retournée.

    Si plusieurs valeurs ont la même fréquence maximale,
    elles sont retournées sous forme de tuple trié.

    Exemple
    -------
    mode([1, 2, 2, 3]) -> 2
    mode([1, 1, 2, 2, 3]) -> (1, 2)
    """
    values = _validate_data(data)
    frequencies = Counter(values)

    maximum_frequency = max(frequencies.values())

    modes = tuple(
        sorted(
            value
            for value, frequency in frequencies.items()
            if frequency == maximum_frequency
        )
    )

    if len(modes) == 1:
        return modes[0]

    return modes


def minimum(data: Iterable[Number]) -> Number:
    """
    Retourne la plus petite valeur de la série.
    """
    values = _validate_data(data)

    return min(values)


def maximum(data: Iterable[Number]) -> Number:
    """
    Retourne la plus grande valeur de la série.
    """
    values = _validate_data(data)

    return max(values)


def data_range(data: Iterable[Number]) -> Number:
    """
    Calcule l'étendue de la série.

    Étendue = maximum - minimum.
    """
    values = _validate_data(data)

    return max(values) - min(values)


def variance(data: Iterable[Number]) -> float:
    """
    Calcule la variance de population.

    Formule
    -------
    σ² = (1/n) × Σ(xᵢ - μ)²
    """
    values = _validate_data(data)

    average = mean(values)

    return sum(
        (value - average) ** 2
        for value in values
    ) / len(values)


def standard_deviation(data: Iterable[Number]) -> float:
    """
    Calcule l'écart-type de population.

    Formule
    -------
    σ = √σ²
    """
    return sqrt(variance(data))


def percentile(
    data: Iterable[Number],
    p: float,
) -> float:
    """
    Calcule un percentile par interpolation linéaire.

    Parameters
    ----------
    data:
        Données statistiques.
    p:
        Percentile compris entre 0 et 100.

    Examples
    --------
    percentile([1, 2, 3, 4, 5], 50) -> 3.0
    percentile([1, 2, 3, 4, 5], 25) -> 2.0
    """
    values = sorted(_validate_data(data))

    if not 0 <= p <= 100:
        raise ValueError(
            "Le percentile doit être compris entre 0 et 100."
        )

    if len(values) == 1:
        return float(values[0])

    position = (len(values) - 1) * (p / 100)

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(values) - 1,
    )

    fraction = position - lower_index

    lower_value = values[lower_index]
    upper_value = values[upper_index]

    return (
        lower_value
        + fraction * (upper_value - lower_value)
    )


def quartiles(
    data: Iterable[Number],
) -> tuple[float, float, float]:
    """
    Calcule les trois quartiles d'une série.

    Returns
    -------
    tuple[float, float, float]
        (Q1, Q2, Q3)

    Q1 = 25e percentile
    Q2 = 50e percentile
    Q3 = 75e percentile
    """
    values = _validate_data(data)

    return (
        percentile(values, 25),
        percentile(values, 50),
        percentile(values, 75),
    )