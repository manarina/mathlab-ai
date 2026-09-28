from __future__ import annotations

from math import isfinite

from scipy.stats import norm, uniform


Number = int | float


# ============================================================
# VALIDATION
# ============================================================


def _validate_number(
    value: Number,
    name: str,
) -> float:
    """
    Vérifie qu'une valeur est numérique et finie.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{name} doit être un nombre."
        )

    value = float(value)

    if not isfinite(value):
        raise ValueError(
            f"{name} doit être un nombre fini."
        )

    return value


def _validate_standard_deviation(
    standard_deviation: Number,
) -> float:
    """
    Vérifie qu'un écart-type est strictement positif.
    """

    value = _validate_number(
        standard_deviation,
        "L'écart-type",
    )

    if value <= 0:
        raise ValueError(
            "L'écart-type doit être strictement positif."
        )

    return value


def _validate_probability(
    probability: Number,
) -> float:
    """
    Vérifie qu'une probabilité appartient à [0, 1].
    """

    value = _validate_number(
        probability,
        "La probabilité",
    )

    if not 0 <= value <= 1:
        raise ValueError(
            "La probabilité doit être comprise entre 0 et 1."
        )

    return value


# ============================================================
# DISTRIBUTION NORMALE
# ============================================================


def normal_pdf(
    x: Number,
    mean_value: Number = 0,
    standard_deviation: Number = 1,
) -> float:
    """
    Calcule la densité de probabilité d'une loi normale.

    Paramètres
    ----------
    x :
        Valeur à évaluer.

    mean_value :
        Moyenne de la distribution.

    standard_deviation :
        Écart-type de la distribution.
    """

    x_value = _validate_number(x, "x")
    mean_value = _validate_number(
        mean_value,
        "La moyenne",
    )
    standard_deviation = _validate_standard_deviation(
        standard_deviation
    )

    return float(
        norm.pdf(
            x_value,
            loc=mean_value,
            scale=standard_deviation,
        )
    )


def normal_cdf(
    x: Number,
    mean_value: Number = 0,
    standard_deviation: Number = 1,
) -> float:
    """
    Calcule P(X <= x) pour une loi normale.
    """

    x_value = _validate_number(x, "x")
    mean_value = _validate_number(
        mean_value,
        "La moyenne",
    )
    standard_deviation = _validate_standard_deviation(
        standard_deviation
    )

    return float(
        norm.cdf(
            x_value,
            loc=mean_value,
            scale=standard_deviation,
        )
    )


def normal_quantile(
    probability: Number,
    mean_value: Number = 0,
    standard_deviation: Number = 1,
) -> float:
    """
    Calcule le quantile d'une loi normale.

    Retourne x tel que :

        P(X <= x) = probability
    """

    probability = _validate_probability(probability)

    mean_value = _validate_number(
        mean_value,
        "La moyenne",
    )

    standard_deviation = _validate_standard_deviation(
        standard_deviation
    )

    return float(
        norm.ppf(
            probability,
            loc=mean_value,
            scale=standard_deviation,
        )
    )


def normal_probability_between(
    lower: Number,
    upper: Number,
    mean_value: Number = 0,
    standard_deviation: Number = 1,
) -> float:
    """
    Calcule :

        P(lower <= X <= upper)

    pour une loi normale.
    """

    lower_value = _validate_number(
        lower,
        "La borne inférieure",
    )

    upper_value = _validate_number(
        upper,
        "La borne supérieure",
    )

    if lower_value > upper_value:
        raise ValueError(
            "La borne inférieure doit être "
            "inférieure ou égale à la borne supérieure."
        )

    mean_value = _validate_number(
        mean_value,
        "La moyenne",
    )

    standard_deviation = _validate_standard_deviation(
        standard_deviation
    )

    return float(
        norm.cdf(
            upper_value,
            loc=mean_value,
            scale=standard_deviation,
        )
        - norm.cdf(
            lower_value,
            loc=mean_value,
            scale=standard_deviation,
        )
    )


# ============================================================
# DISTRIBUTION UNIFORME
# ============================================================


def uniform_pdf(
    x: Number,
    lower: Number = 0,
    upper: Number = 1,
) -> float:
    """
    Calcule la densité d'une loi uniforme U(lower, upper).
    """

    x_value = _validate_number(x, "x")

    lower_value = _validate_number(
        lower,
        "La borne inférieure",
    )

    upper_value = _validate_number(
        upper,
        "La borne supérieure",
    )

    if lower_value >= upper_value:
        raise ValueError(
            "La borne inférieure doit être "
            "strictement inférieure à la borne supérieure."
        )

    return float(
        uniform.pdf(
            x_value,
            loc=lower_value,
            scale=upper_value - lower_value,
        )
    )


def uniform_cdf(
    x: Number,
    lower: Number = 0,
    upper: Number = 1,
) -> float:
    """
    Calcule P(X <= x) pour une loi uniforme.
    """

    x_value = _validate_number(x, "x")

    lower_value = _validate_number(
        lower,
        "La borne inférieure",
    )

    upper_value = _validate_number(
        upper,
        "La borne supérieure",
    )

    if lower_value >= upper_value:
        raise ValueError(
            "La borne inférieure doit être "
            "strictement inférieure à la borne supérieure."
        )

    return float(
        uniform.cdf(
            x_value,
            loc=lower_value,
            scale=upper_value - lower_value,
        )
    )


def uniform_probability_between(
    lower_bound: Number,
    upper_bound: Number,
    distribution_lower: Number = 0,
    distribution_upper: Number = 1,
) -> float:
    """
    Calcule :

        P(lower_bound <= X <= upper_bound)

    pour une loi uniforme.
    """

    lower_bound = _validate_number(
        lower_bound,
        "La borne inférieure de l'intervalle",
    )

    upper_bound = _validate_number(
        upper_bound,
        "La borne supérieure de l'intervalle",
    )

    if lower_bound > upper_bound:
        raise ValueError(
            "La borne inférieure doit être "
            "inférieure ou égale à la borne supérieure."
        )

    distribution_lower = _validate_number(
        distribution_lower,
        "La borne inférieure de la distribution",
    )

    distribution_upper = _validate_number(
        distribution_upper,
        "La borne supérieure de la distribution",
    )

    if distribution_lower >= distribution_upper:
        raise ValueError(
            "La borne inférieure de la distribution doit être "
            "strictement inférieure à la borne supérieure."
        )

    return float(
        uniform.cdf(
            upper_bound,
            loc=distribution_lower,
            scale=distribution_upper - distribution_lower,
        )
        - uniform.cdf(
            lower_bound,
            loc=distribution_lower,
            scale=distribution_upper - distribution_lower,
        )
    )