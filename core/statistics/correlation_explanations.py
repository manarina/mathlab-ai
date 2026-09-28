from __future__ import annotations

from collections.abc import Sequence

from core.statistics.correlation import (
    covariance,
    pearson_correlation,
)


Number = int | float


# ============================================================
# OUTILS
# ============================================================


def _format_number(value: float | int) -> str:
    """
    Formate proprement un nombre.
    """

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return f"{value:.6g}"


def _format_series(
    values: Sequence[Number],
) -> str:
    """
    Formate une série statistique pour l'affichage.
    """

    return ", ".join(
        _format_number(value)
        for value in values
    )


# ============================================================
# EXPLICATION DE LA COVARIANCE
# ============================================================


def explain_covariance(
    x: Sequence[Number],
    y: Sequence[Number],
    result: float | None = None,
) -> str:
    """
    Explique étape par étape le calcul de la covariance.
    """

    if result is None:
        result = covariance(x, y)

    mean_x = sum(float(value) for value in x) / len(x)
    mean_y = sum(float(value) for value in y) / len(y)

    products = [
        (float(x_value) - mean_x)
        * (float(y_value) - mean_y)
        for x_value, y_value in zip(x, y)
    ]

    sum_products = sum(products)

    products_formatted = ", ".join(
        _format_number(value)
        for value in products
    )

    return f"""
### 📊 Covariance

**Séries étudiées**

- X = `{_format_series(x)}`
- Y = `{_format_series(y)}`

### 1. Calcul des moyennes

La moyenne de X est :

\\[
\\bar{{x}} = \\frac{{\\sum x_i}}{{n}}
= {_format_number(mean_x)}
\\]

La moyenne de Y est :

\\[
\\bar{{y}} = \\frac{{\\sum y_i}}{{n}}
= {_format_number(mean_y)}
\\]

### 2. Produits des écarts à la moyenne

Pour chaque couple de valeurs, on calcule :

\\[
(x_i - \\bar{{x}})(y_i - \\bar{{y}})
\\]

Les produits obtenus sont :

`{products_formatted}`

Leur somme vaut :

\\[
\\sum (x_i - \\bar{{x}})(y_i - \\bar{{y}})
= {_format_number(sum_products)}
\\]

### 3. Formule

La covariance de population est :

\\[
\\operatorname{{Cov}}(X,Y)
=
\\frac{{\\sum (x_i-\\bar{{x}})(y_i-\\bar{{y}})}}{{n}}
\\]

Donc :

\\[
\\operatorname{{Cov}}(X,Y)
=
{_format_number(result)}
\\]

### 💡 Interprétation

- Une covariance **positive** indique que X et Y ont tendance à évoluer dans le même sens.
- Une covariance **négative** indique qu'elles ont tendance à évoluer en sens opposés.
- Une covariance proche de **0** indique une faible association linéaire, mais ne permet pas à elle seule de conclure à l'absence de relation.

> La covariance dépend des unités de mesure des variables.
"""


# ============================================================
# EXPLICATION DE PEARSON
# ============================================================


def explain_pearson_correlation(
    x: Sequence[Number],
    y: Sequence[Number],
    result: float | None = None,
) -> str:
    """
    Explique étape par étape le calcul du coefficient
    de corrélation linéaire de Pearson.
    """

    if result is None:
        result = pearson_correlation(x, y)

    mean_x = sum(float(value) for value in x) / len(x)
    mean_y = sum(float(value) for value in y) / len(y)

    centered_x = [
        float(value) - mean_x
        for value in x
    ]

    centered_y = [
        float(value) - mean_y
        for value in y
    ]

    numerator = sum(
        x_centered * y_centered
        for x_centered, y_centered in zip(
            centered_x,
            centered_y,
        )
    )

    sum_x_squared = sum(
        value ** 2
        for value in centered_x
    )

    sum_y_squared = sum(
        value ** 2
        for value in centered_y
    )

    denominator = (
        sum_x_squared * sum_y_squared
    ) ** 0.5

    return f"""
### 📈 Coefficient de corrélation de Pearson

**Séries étudiées**

- X = `{_format_series(x)}`
- Y = `{_format_series(y)}`

Le coefficient de Pearson mesure l'intensité et le sens
d'une **relation linéaire** entre deux variables.

### 1. Calcul des moyennes

\\[
\\bar{{x}} = {_format_number(mean_x)}
\\]

\\[
\\bar{{y}} = {_format_number(mean_y)}
\\]

### 2. Calcul des écarts à la moyenne

Pour chaque valeur :

\\[
x_i - \\bar{{x}}
\\]

et

\\[
y_i - \\bar{{y}}
\\]

### 3. Numérateur

On calcule :

\\[
\\sum
(x_i-\\bar{{x}})
(y_i-\\bar{{y}})
\\]

Le résultat est :

\\[
{_format_number(numerator)}
\\]

### 4. Dénominateur

On calcule :

\\[
\\sqrt{{
\\sum (x_i-\\bar{{x}})^2
\\times
\\sum(y_i-\\bar{{y}})^2
}}
\\]

Le résultat est :

\\[
{_format_number(denominator)}
\\]

### 5. Formule de Pearson

\\[
r =
\\frac{{
\\sum(x_i-\\bar{{x}})(y_i-\\bar{{y}})
}}{{
\\sqrt{{
\\sum(x_i-\\bar{{x}})^2
\\sum(y_i-\\bar{{y}})^2
}}
}}
\\]

Donc :

\\[
\\boxed{{r = {_format_number(result)}}}
\\]

### 💡 Interprétation

Le coefficient de Pearson est compris entre **-1 et 1**.

- **r proche de 1** → relation linéaire positive forte.
- **r proche de -1** → relation linéaire négative forte.
- **r proche de 0** → faible relation linéaire.

⚠️ Une corrélation ne permet pas, à elle seule, d'établir
une relation de causalité.
"""