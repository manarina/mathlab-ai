from __future__ import annotations

import math
from collections.abc import Sequence

from core.statistics.regression import (
    linear_regression,
    regression_predictions,
)

Number = int | float


def _validate_series(
    values: Sequence[Number],
    name: str,
) -> list[float]:
    """
    Valide une série numérique destinée à la visualisation.

    Règles :
    - la série ne doit pas être vide ;
    - les valeurs doivent être numériques ;
    - les booléens sont refusés ;
    - les valeurs infinies ou NaN sont refusées.
    """

    if not values:
        raise ValueError(
            f"La série {name} ne peut pas être vide."
        )

    validated: list[float] = []

    for value in values:

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"La série {name} doit contenir uniquement "
                "des valeurs numériques."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"La série {name} contient une valeur non finie."
            )

        validated.append(numeric_value)

    return validated


def _validate_pair(
    x: Sequence[Number],
    y: Sequence[Number],
) -> tuple[list[float], list[float]]:
    """
    Valide deux séries destinées à la visualisation
    d'une régression linéaire.
    """

    x_values = _validate_series(x, "X")
    y_values = _validate_series(y, "Y")

    if len(x_values) != len(y_values):
        raise ValueError(
            "Les séries X et Y doivent avoir le même nombre "
            "de valeurs."
        )

    if len(x_values) < 2:
        raise ValueError(
            "Au moins deux observations sont nécessaires "
            "pour visualiser une régression linéaire."
        )

    if all(value == x_values[0] for value in x_values):
        raise ValueError(
            "La visualisation de la régression est impossible "
            "lorsque toutes les valeurs de X sont identiques."
        )

    return x_values, y_values


def regression_visualization_data(
    x: Sequence[Number],
    y: Sequence[Number],
) -> list[dict[str, float]]:
    """
    Prépare les données observées et prédites pour un graphique.

    Chaque observation contient :

        {
            "x": valeur de X,
            "y": valeur observée de Y,
            "predicted_y": valeur prédite par la régression
        }

    Les données sont classées selon X afin de faciliter
    l'affichage graphique.
    """

    x_values, y_values = _validate_pair(x, y)

    predicted_values = regression_predictions(
        x_values,
        y_values,
    )

    data = [
        {
            "x": x_value,
            "y": y_value,
            "predicted_y": predicted_value,
        }
        for x_value, y_value, predicted_value in zip(
            x_values,
            y_values,
            predicted_values,
        )
    ]

    return sorted(
        data,
        key=lambda item: item["x"],
    )


def regression_line_data(
    x: Sequence[Number],
    y: Sequence[Number],
    number_of_points: int = 100,
) -> list[dict[str, float]]:
    """
    Génère les points de la droite de régression.

    Les points couvrent l'intervalle [min(X), max(X)].

    Paramètres
    ----------
    x, y :
        Données utilisées pour calculer la régression.

    number_of_points :
        Nombre de points générés pour représenter la droite.
        Minimum : 2.

    Retour
    ------
    list[dict[str, float]]
        Liste contenant :

            {
                "x": valeur de X,
                "y": valeur prédite
            }
    """

    x_values, y_values = _validate_pair(x, y)

    if not isinstance(number_of_points, int):
        raise TypeError(
            "Le nombre de points doit être un entier."
        )

    if number_of_points < 2:
        raise ValueError(
            "Le nombre de points doit être au moins égal à 2."
        )

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    x_min = min(x_values)
    x_max = max(x_values)

    if x_min == x_max:
        raise ValueError(
            "Impossible de générer une droite lorsque "
            "les valeurs de X sont identiques."
        )

    step = (x_max - x_min) / (number_of_points - 1)

    return [
        {
            "x": x_min + index * step,
            "y": slope * (x_min + index * step) + intercept,
        }
        for index in range(number_of_points)
    ]


def regression_plot_data(
    x: Sequence[Number],
    y: Sequence[Number],
    number_of_line_points: int = 100,
) -> dict[str, object]:
    """
    Prépare toutes les données nécessaires à une visualisation
    complète de la régression linéaire.

    Retour :

        {
            "observations": [...],
            "regression_line": [...],
            "slope": ...,
            "intercept": ...
        }
    """

    x_values, y_values = _validate_pair(x, y)

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    observations = regression_visualization_data(
        x_values,
        y_values,
    )

    regression_line = regression_line_data(
        x_values,
        y_values,
        number_of_line_points,
    )

    return {
        "observations": observations,
        "regression_line": regression_line,
        "slope": slope,
        "intercept": intercept,
    }