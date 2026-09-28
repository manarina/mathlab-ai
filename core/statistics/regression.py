from __future__ import annotations

import math
from collections.abc import Sequence

Number = int | float


def _validate_series(
    values: Sequence[Number],
    name: str,
) -> list[float]:
    """
    Valide et convertit une série numérique en liste de float.

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
    Valide deux séries destinées à une régression linéaire.
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
            "pour effectuer une régression linéaire."
        )

    return x_values, y_values


def _mean(values: Sequence[float]) -> float:
    """Calcule la moyenne d'une série déjà validée."""

    return sum(values) / len(values)


def _validate_x_variation(x: Sequence[float]) -> None:
    """
    Vérifie que X possède une variation suffisante.

    Une régression linéaire est impossible si toutes les
    valeurs de X sont identiques.
    """

    if all(value == x[0] for value in x):
        raise ValueError(
            "La régression linéaire est impossible lorsque "
            "toutes les valeurs de X sont identiques."
        )


def regression_slope(
    x: Sequence[Number],
    y: Sequence[Number],
) -> float:
    """
    Calcule la pente a de la droite de régression linéaire.

    Formule :

        a = Σ[(xᵢ - x̄)(yᵢ - ȳ)]
            / Σ[(xᵢ - x̄)²]
    """

    x_values, y_values = _validate_pair(x, y)

    _validate_x_variation(x_values)

    x_mean = _mean(x_values)
    y_mean = _mean(y_values)

    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(
            x_values,
            y_values,
        )
    )

    denominator = sum(
        (x_value - x_mean) ** 2
        for x_value in x_values
    )

    return numerator / denominator


def regression_intercept(
    x: Sequence[Number],
    y: Sequence[Number],
) -> float:
    """
    Calcule l'ordonnée à l'origine b.

    Formule :

        b = ȳ - ax̄
    """

    x_values, y_values = _validate_pair(x, y)

    slope = regression_slope(
        x_values,
        y_values,
    )

    x_mean = _mean(x_values)
    y_mean = _mean(y_values)

    return y_mean - slope * x_mean


def linear_regression(
    x: Sequence[Number],
    y: Sequence[Number],
) -> tuple[float, float]:
    """
    Calcule les paramètres de la droite de régression.

    Retourne :

        (a, b)

    où :

        y = ax + b
    """

    x_values, y_values = _validate_pair(x, y)

    slope = regression_slope(
        x_values,
        y_values,
    )

    intercept = regression_intercept(
        x_values,
        y_values,
    )

    return slope, intercept


def predict(
    x: Sequence[Number],
    y: Sequence[Number],
    x_values: Sequence[Number],
) -> list[float]:
    """
    Calcule les valeurs prédites par la droite de régression.

    Pour chaque valeur x :

        ŷ = ax + b
    """

    regression_x, regression_y = _validate_pair(
        x,
        y,
    )

    prediction_values = _validate_series(
        x_values,
        "X à prédire",
    )

    slope, intercept = linear_regression(
        regression_x,
        regression_y,
    )

    return [
        slope * value + intercept
        for value in prediction_values
    ]


def coefficient_of_determination(
    x: Sequence[Number],
    y: Sequence[Number],
) -> float:
    """
    Calcule le coefficient de détermination R².

    Formule :

        R² = 1 - SSE / SST

    où :

        SSE = Σ(yᵢ - ŷᵢ)²
        SST = Σ(yᵢ - ȳ)²

    R² mesure la proportion de la variabilité de Y
    expliquée par le modèle linéaire.
    """

    x_values, y_values = _validate_pair(x, y)

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    y_mean = _mean(y_values)

    predicted_values = [
        slope * x_value + intercept
        for x_value in x_values
    ]

    sse = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(
            y_values,
            predicted_values,
        )
    )

    sst = sum(
        (actual - y_mean) ** 2
        for actual in y_values
    )

    if sst == 0:
        return 1.0 if sse == 0 else 0.0

    return 1 - (sse / sst)


def regression_predictions(
    x: Sequence[Number],
    y: Sequence[Number],
) -> list[float]:
    """
    Calcule les valeurs prédites pour les observations X
    utilisées dans le modèle.
    """

    x_values, y_values = _validate_pair(x, y)

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    return [
        slope * x_value + intercept
        for x_value in x_values
    ]