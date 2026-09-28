from __future__ import annotations

import math


# ============================================================
# OUTILS DE FORMATAGE
# ============================================================


def _format_number(value: float | int) -> str:
    """
    Formate un nombre pour une présentation pédagogique.
    """

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return f"{value:.6g}"


def _format_probability(value: float) -> str:
    """
    Formate une probabilité à la fois en décimal et en pourcentage.
    """

    return (
        f"{value:.6f} "
        f"(soit {value * 100:.2f} %)"
    )


# ============================================================
# LOI NORMALE — PDF
# ============================================================


def explain_normal_pdf(
    x: float | int,
    mean_value: float | int = 0,
    standard_deviation: float | int = 1,
    result: float | None = None,
) -> str:
    """
    Explique le calcul de la densité d'une loi normale.
    """

    x = float(x)
    mean_value = float(mean_value)
    standard_deviation = float(standard_deviation)

    if standard_deviation <= 0:
        raise ValueError(
            "L'écart-type doit être strictement positif."
        )

    if result is None:
        coefficient = 1 / (
            standard_deviation * math.sqrt(2 * math.pi)
        )

        exponent = -(
            (x - mean_value) ** 2
        ) / (
            2 * standard_deviation**2
        )

        result = coefficient * math.exp(exponent)

    return f"""
### 📊 Densité de la loi normale

On considère une variable aléatoire :

\\[
X \\sim \\mathcal{{N}}(\\mu, \\sigma)
\\]

avec :

- moyenne \\(\\mu = {_format_number(mean_value)}\\)
- écart-type \\(\\sigma = {_format_number(standard_deviation)}\\)
- valeur étudiée \\(x = {_format_number(x)}\\)

La densité d'une loi normale est :

\\[
f(x) =
\\frac{{1}}{{\\sigma\\sqrt{{2\\pi}}}}
e^{{-\\frac{{(x-\\mu)^2}}{{2\\sigma^2}}}}
\\]

En remplaçant par les valeurs :

\\[
f({_format_number(x)}) =
\\frac{{1}}{{{_format_number(standard_deviation)}
\\sqrt{{2\\pi}}}}
e^{{-\\frac{{({_format_number(x)}
-{_format_number(mean_value)})^2}}
{{2({_format_number(standard_deviation)})^2}}}}
\\]

**Résultat :**

\\[
f({_format_number(x)}) \\approx {_format_number(result)}
\\]
"""


# ============================================================
# LOI NORMALE — CDF
# ============================================================


def explain_normal_cdf(
    x: float | int,
    mean_value: float | int = 0,
    standard_deviation: float | int = 1,
    result: float | None = None,
) -> str:
    """
    Explique le calcul de P(X <= x) pour une loi normale.
    """

    x = float(x)
    mean_value = float(mean_value)
    standard_deviation = float(standard_deviation)

    if standard_deviation <= 0:
        raise ValueError(
            "L'écart-type doit être strictement positif."
        )

    if result is None:
        from core.statistics.distributions import normal_cdf

        result = normal_cdf(
            x,
            mean_value,
            standard_deviation,
        )

    z = (
        x - mean_value
    ) / standard_deviation

    return f"""
### 📈 Fonction de répartition de la loi normale

On cherche :

\\[
P(X \\leq {_format_number(x)})
\\]

avec :

\\[
X \\sim \\mathcal{{N}}(
{_format_number(mean_value)},
{_format_number(standard_deviation)}
)
\\]

On standardise d'abord la valeur :

\\[
Z =
\\frac{{X-\\mu}}{{\\sigma}}
\\]

Donc :

\\[
z =
\\frac{{{_format_number(x)}
-{_format_number(mean_value)}}}
{{{_format_number(standard_deviation)}}}
=
{_format_number(z)}
\\]

On obtient alors :

\\[
P(X \\leq {_format_number(x)})
=
P(Z \\leq {_format_number(z)})
\\]

**Résultat :**

\\[
P(X \\leq {_format_number(x)})
\\approx {_format_probability(result)}
\\]
"""


# ============================================================
# LOI NORMALE — QUANTILE
# ============================================================


def explain_normal_quantile(
    probability: float | int,
    mean_value: float | int = 0,
    standard_deviation: float | int = 1,
    result: float | None = None,
) -> str:
    """
    Explique le calcul d'un quantile de loi normale.
    """

    probability = float(probability)
    mean_value = float(mean_value)
    standard_deviation = float(standard_deviation)

    if not 0 <= probability <= 1:
        raise ValueError(
            "La probabilité doit être comprise entre 0 et 1."
        )

    if standard_deviation <= 0:
        raise ValueError(
            "L'écart-type doit être strictement positif."
        )

    if result is None:
        from core.statistics.distributions import normal_quantile

        result = normal_quantile(
            probability,
            mean_value,
            standard_deviation,
        )

    return f"""
### 📐 Quantile de la loi normale

On cherche la valeur \\(x\\) telle que :

\\[
P(X \\leq x) = {_format_number(probability)}
\\]

avec :

\\[
X \\sim \\mathcal{{N}}(
{_format_number(mean_value)},
{_format_number(standard_deviation)}
)
\\]

Le quantile de la loi normale standard est noté :

\\[
z_p = \\Phi^{{-1}}(p)
\\]

Puis on revient à la variable \\(X\\) :

\\[
x = \\mu + \\sigma z_p
\\]

**Résultat :**

\\[
x \\approx {_format_number(result)}
\\]
"""


# ============================================================
# LOI NORMALE — PROBABILITÉ ENTRE DEUX VALEURS
# ============================================================


def explain_normal_probability_between(
    lower: float | int,
    upper: float | int,
    mean_value: float | int = 0,
    standard_deviation: float | int = 1,
    result: float | None = None,
) -> str:
    """
    Explique P(lower <= X <= upper) pour une loi normale.
    """

    lower = float(lower)
    upper = float(upper)
    mean_value = float(mean_value)
    standard_deviation = float(standard_deviation)

    if lower > upper:
        raise ValueError(
            "La borne inférieure doit être "
            "inférieure ou égale à la borne supérieure."
        )

    if standard_deviation <= 0:
        raise ValueError(
            "L'écart-type doit être strictement positif."
        )

    if result is None:
        from core.statistics.distributions import (
            normal_probability_between,
        )

        result = normal_probability_between(
            lower,
            upper,
            mean_value,
            standard_deviation,
        )

    z_lower = (
        lower - mean_value
    ) / standard_deviation

    z_upper = (
        upper - mean_value
    ) / standard_deviation

    return f"""
### 📊 Probabilité entre deux valeurs

On cherche :

\\[
P({_format_number(lower)}
\\leq X \\leq
{_format_number(upper)})
\\]

avec :

\\[
X \\sim \\mathcal{{N}}(
{_format_number(mean_value)},
{_format_number(standard_deviation)}
)
\\]

On standardise les deux bornes :

\\[
z_1 =
\\frac{{{_format_number(lower)}
-{_format_number(mean_value)}}}
{{{_format_number(standard_deviation)}}}
=
{_format_number(z_lower)}
\\]

\\[
z_2 =
\\frac{{{_format_number(upper)}
-{_format_number(mean_value)}}}
{{{_format_number(standard_deviation)}}}
=
{_format_number(z_upper)}
\\]

La probabilité recherchée est :

\\[
P(z_1 \\leq Z \\leq z_2)
=
\\Phi(z_2)-\\Phi(z_1)
\\]

**Résultat :**

\\[
P({_format_number(lower)}
\\leq X \\leq
{_format_number(upper)})
\\approx {_format_probability(result)}
\\]
"""


# ============================================================
# LOI UNIFORME — PDF
# ============================================================


def explain_uniform_pdf(
    x: float | int,
    lower: float | int = 0,
    upper: float | int = 1,
    result: float | None = None,
) -> str:
    """
    Explique le calcul de la densité d'une loi uniforme.
    """

    x = float(x)
    lower = float(lower)
    upper = float(upper)

    if lower >= upper:
        raise ValueError(
            "La borne inférieure doit être "
            "strictement inférieure à la borne supérieure."
        )

    if result is None:
        from core.statistics.distributions import uniform_pdf

        result = uniform_pdf(
            x,
            lower,
            upper,
        )

    density = 1 / (upper - lower)

    if lower <= x <= upper:
        position = (
            "x appartient à l'intervalle "
            f"[{_format_number(lower)}, {_format_number(upper)}]"
        )
    else:
        position = (
            "x n'appartient pas à l'intervalle de définition"
        )

    return f"""
### 📊 Densité de la loi uniforme

On considère :

\\[
X \\sim U(a,b)
\\]

avec :

- \\(a = {_format_number(lower)}\\)
- \\(b = {_format_number(upper)}\\)
- \\(x = {_format_number(x)}\\)

La densité est :

\\[
f(x) =
\\frac{{1}}{{b-a}}
\\]

pour \\(a \\leq x \\leq b\\), et 0 en dehors.

Ici :

\\[
\\frac{{1}}{{b-a}}
=
\\frac{{1}}{{{_format_number(upper)}
-{_format_number(lower)}}}
=
{_format_number(density)}
\\]

{position}.

**Résultat :**

\\[
f({_format_number(x)}) =
{_format_number(result)}
\\]
"""


# ============================================================
# LOI UNIFORME — CDF
# ============================================================


def explain_uniform_cdf(
    x: float | int,
    lower: float | int = 0,
    upper: float | int = 1,
    result: float | None = None,
) -> str:
    """
    Explique la fonction de répartition d'une loi uniforme.
    """

    x = float(x)
    lower = float(lower)
    upper = float(upper)

    if lower >= upper:
        raise ValueError(
            "La borne inférieure doit être "
            "strictement inférieure à la borne supérieure."
        )

    if result is None:
        from core.statistics.distributions import uniform_cdf

        result = uniform_cdf(
            x,
            lower,
            upper,
        )

    return f"""
### 📈 Fonction de répartition de la loi uniforme

On cherche :

\\[
P(X \\leq {_format_number(x)})
\\]

pour :

\\[
X \\sim U(
{_format_number(lower)},
{_format_number(upper)}
)
\\]

Lorsque \\(a \\leq x \\leq b\\) :

\\[
F(x) =
\\frac{{x-a}}{{b-a}}
\\]

Donc :

\\[
F({_format_number(x)})
=
\\frac{{{_format_number(x)}
-{_format_number(lower)}}}
{{{_format_number(upper)}
-{_format_number(lower)}}}
\\]

**Résultat :**

\\[
F({_format_number(x)})
\\approx {_format_probability(result)}
\\]
"""


# ============================================================
# LOI UNIFORME — PROBABILITÉ ENTRE DEUX VALEURS
# ============================================================


def explain_uniform_probability_between(
    lower_bound: float | int,
    upper_bound: float | int,
    distribution_lower: float | int = 0,
    distribution_upper: float | int = 1,
    result: float | None = None,
) -> str:
    """
    Explique P(a <= X <= b) pour une loi uniforme.
    """

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    distribution_lower = float(distribution_lower)
    distribution_upper = float(distribution_upper)

    if lower_bound > upper_bound:
        raise ValueError(
            "La borne inférieure doit être "
            "inférieure ou égale à la borne supérieure."
        )

    if distribution_lower >= distribution_upper:
        raise ValueError(
            "La borne inférieure de la distribution doit être "
            "strictement inférieure à la borne supérieure."
        )

    if result is None:
        from core.statistics.distributions import (
            uniform_probability_between,
        )

        result = uniform_probability_between(
            lower_bound,
            upper_bound,
            distribution_lower,
            distribution_upper,
        )

    return f"""
### 📊 Probabilité entre deux valeurs — loi uniforme

On cherche :

\\[
P({_format_number(lower_bound)}
\\leq X \\leq
{_format_number(upper_bound)})
\\]

avec :

\\[
X \\sim U(
{_format_number(distribution_lower)},
{_format_number(distribution_upper)}
)
\\]

Pour une loi uniforme, la probabilité correspond au
rapport entre la longueur de l'intervalle recherché et
la longueur totale de l'intervalle.

\\[
P(a \\leq X \\leq b)
=
\\frac{{b-a}}{{B-A}}
\\]

Ici :

\\[
P =
\\frac{{{_format_number(upper_bound)}
-{_format_number(lower_bound)}}}
{{{_format_number(distribution_upper)}
-{_format_number(distribution_lower)}}}
\\]

**Résultat :**

\\[
P({_format_number(lower_bound)}
\\leq X \\leq
{_format_number(upper_bound)})
\\approx {_format_probability(result)}
\\]
"""