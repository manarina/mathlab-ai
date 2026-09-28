from __future__ import annotations

import math
from collections.abc import Sequence


Number = int | float


# ============================================================
# VALIDATION
# ============================================================


def _validate_series(
    values: Sequence[Number],
    name: str,
) -> list[float]:
    """
    Valide une série statistique et la convertit en float.

    Conditions :
    - la série ne doit pas être vide ;
    - toutes les valeurs doivent être numériques ;
    - les booléens sont refusés ;
    - les valeurs infinies ou NaN sont refusées.
    """

    if not values:
        raise ValueError(
            f"La série « {name} » ne peut pas être vide."
        )

    validated: list[float] = []

    for value in values:

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"La série « {name} » contient une "
                f"valeur non numérique : {value!r}."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"La série « {name} » contient une "
                f"valeur non finie : {value!r}."
            )

        validated.append(numeric_value)

    return validated


def _validate_pair(
    x: Sequence[Number],
    y: Sequence[Number],
) -> tuple[list[float], list[float]]:
    """
    Valide deux séries destinées à être comparées.

    Les deux séries doivent :
    - être non vides ;
    - contenir uniquement des nombres valides ;
    - avoir la même longueur.
    """

    x_values = _validate_series(x, "X")
    y_values = _validate_series(y, "Y")

    if len(x_values) != len(y_values):
        raise ValueError(
            "Les deux séries doivent avoir le même nombre "
            "de valeurs."
        )

    return x_values, y_values


# ============================================================
# MOYENNE
# ============================================================


def _mean(values: Sequence[float]) -> float:
    """
    Calcule la moyenne d'une série déjà validée.
    """

    return sum(values) / len(values)


# ============================================================
# COVARIANCE
# ============================================================


def covariance(
    x: Sequence[Number],
    y: Sequence[Number],
) -> float:
    """
    Calcule la covariance entre deux séries.

    Formule utilisée :

        Cov(X, Y) =
        Σ[(xᵢ - x̄)(yᵢ - ȳ)] / n

    Il s'agit ici de la covariance de population,
    avec une division par n.
    """

    x_values, y_values = _validate_pair(x, y)

    mean_x = _mean(x_values)
    mean_y = _mean(y_values)

    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(
            x_values,
            y_values,
        )
    ) / len(x_values)


# ============================================================
# CORRELATION DE PEARSON
# ============================================================


def pearson_correlation(
    x: Sequence[Number],
    y: Sequence[Number],
) -> float:
    """
    Calcule le coefficient de corrélation linéaire
    de Pearson entre deux séries.

    Formule :

        r =
        Cov(X,Y) / (σₓ × σᵧ)

    Le résultat appartient à l'intervalle [-1, 1].

    - r proche de 1  : relation linéaire positive
    - r proche de -1 : relation linéaire négative
    - r proche de 0  : absence de relation linéaire forte

    Une série constante ne permet pas de calculer
    une corrélation de Pearson.
    """

    x_values, y_values = _validate_pair(x, y)

    mean_x = _mean(x_values)
    mean_y = _mean(y_values)

    centered_x = [
        value - mean_x
        for value in x_values
    ]

    centered_y = [
        value - mean_y
        for value in y_values
    ]

    sum_x_squared = sum(
        value ** 2
        for value in centered_x
    )

    sum_y_squared = sum(
        value ** 2
        for value in centered_y
    )

    if sum_x_squared == 0:
        raise ValueError(
            "La corrélation de Pearson est impossible : "
            "la série X est constante."
        )

    if sum_y_squared == 0:
        raise ValueError(
            "La corrélation de Pearson est impossible : "
            "la série Y est constante."
        )

    numerator = sum(
        x_centered * y_centered
        for x_centered, y_centered in zip(
            centered_x,
            centered_y,
        )
    )

    denominator = math.sqrt(
        sum_x_squared * sum_y_squared
    )

    return numerator / denominator