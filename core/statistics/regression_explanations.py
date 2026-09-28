from __future__ import annotations

from collections.abc import Sequence

from core.statistics.regression import (
    coefficient_of_determination,
    linear_regression,
    predict,
    regression_predictions,
    regression_slope,
)


Number = int | float


# ============================================================
# OUTILS DE FORMATAGE
# ============================================================


def _format_number(value: float) -> str:
    """
    Formate proprement une valeur numérique.
    """

    if value == 0:
        return "0"

    if float(value).is_integer():
        return str(int(value))

    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_series(
    values: Sequence[Number],
) -> str:
    """
    Formate une série numérique pour l'affichage.
    """

    return "[" + ", ".join(
        _format_number(float(value))
        for value in values
    ) + "]"


# ============================================================
# EXPLICATION DE LA PENTE
# ============================================================


def explain_regression_slope(
    x: Sequence[Number],
    y: Sequence[Number],
    result: float | None = None,
) -> str:
    """
    Génère une explication pédagogique du calcul de la pente.

    Formule :

        a = Σ[(xᵢ - x̄)(yᵢ - ȳ)]
            / Σ[(xᵢ - x̄)²]
    """

    x_values, y_values = list(x), list(y)

    slope = (
        regression_slope(x_values, y_values)
        if result is None
        else float(result)
    )

    x_mean = sum(float(value) for value in x_values) / len(x_values)
    y_mean = sum(float(value) for value in y_values) / len(y_values)

    numerator = sum(
        (float(x_value) - x_mean)
        * (float(y_value) - y_mean)
        for x_value, y_value in zip(
            x_values,
            y_values,
        )
    )

    denominator = sum(
        (float(x_value) - x_mean) ** 2
        for x_value in x_values
    )

    return "\n".join(
        [
            "### 📐 Calcul de la pente",
            "",
            f"**Série X :** {_format_series(x_values)}",
            f"**Série Y :** {_format_series(y_values)}",
            "",
            f"**Moyenne de X :** {_format_number(x_mean)}",
            f"**Moyenne de Y :** {_format_number(y_mean)}",
            "",
            "**Formule :**",
            "",
            "a = Σ[(xᵢ − x̄)(yᵢ − ȳ)] / Σ[(xᵢ − x̄)²]",
            "",
            f"**Numérateur :** {_format_number(numerator)}",
            f"**Dénominateur :** {_format_number(denominator)}",
            "",
            f"**Pente a = {_format_number(slope)}**",
            "",
            "La pente indique la variation moyenne de Y "
            "lorsque X augmente d'une unité.",
        ]
    )


# ============================================================
# EXPLICATION DE L'ORDONNÉE À L'ORIGINE
# ============================================================


def explain_regression_intercept(
    x: Sequence[Number],
    y: Sequence[Number],
    result: float | None = None,
) -> str:
    """
    Génère une explication pédagogique de l'ordonnée
    à l'origine.

    Formule :

        b = ȳ − ax̄
    """

    x_values = list(x)
    y_values = list(y)

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    if result is not None:
        intercept = float(result)

    x_mean = sum(float(value) for value in x_values) / len(x_values)
    y_mean = sum(float(value) for value in y_values) / len(y_values)

    return "\n".join(
        [
            "### 📍 Calcul de l'ordonnée à l'origine",
            "",
            f"**Moyenne de X :** {_format_number(x_mean)}",
            f"**Moyenne de Y :** {_format_number(y_mean)}",
            f"**Pente a :** {_format_number(slope)}",
            "",
            "**Formule :**",
            "",
            "b = ȳ − ax̄",
            "",
            (
                f"b = {_format_number(y_mean)} − "
                f"({_format_number(slope)} × "
                f"{_format_number(x_mean)})"
            ),
            "",
            f"**Ordonnée à l'origine b = {_format_number(intercept)}**",
            "",
            "L'ordonnée à l'origine correspond à la valeur "
            "prévue de Y lorsque X = 0.",
        ]
    )


# ============================================================
# EXPLICATION DE LA DROITE DE RÉGRESSION
# ============================================================


def explain_linear_regression(
    x: Sequence[Number],
    y: Sequence[Number],
    result: tuple[float, float] | None = None,
) -> str:
    """
    Génère une explication complète de la droite
    de régression linéaire.

    Modèle :

        ŷ = ax + b
    """

    x_values = list(x)
    y_values = list(y)

    if result is None:
        slope, intercept = linear_regression(
            x_values,
            y_values,
        )
    else:
        slope, intercept = (
            float(result[0]),
            float(result[1]),
        )

    slope_text = _format_number(slope)
    intercept_text = _format_number(abs(intercept))

    if intercept >= 0:
        equation = f"ŷ = {slope_text}x + {intercept_text}"
    else:
        equation = f"ŷ = {slope_text}x − {intercept_text}"

    return "\n".join(
        [
            "### 📈 Droite de régression linéaire",
            "",
            f"**Série X :** {_format_series(x_values)}",
            f"**Série Y :** {_format_series(y_values)}",
            "",
            "**Modèle :**",
            "",
            "ŷ = ax + b",
            "",
            f"**Pente :** a = {slope_text}",
            f"**Ordonnée à l'origine :** b = {_format_number(intercept)}",
            "",
            f"### Équation obtenue",
            "",
            f"**{equation}**",
            "",
            "Cette droite représente l'ajustement linéaire "
            "des observations.",
        ]
    )


# ============================================================
# EXPLICATION DES PRÉDICTIONS
# ============================================================


def explain_prediction(
    x: Sequence[Number],
    y: Sequence[Number],
    x_values: Sequence[Number],
    result: Sequence[float] | None = None,
) -> str:
    """
    Génère une explication des prédictions effectuées
    à partir de la droite de régression.
    """

    regression_x = list(x)
    regression_y = list(y)
    prediction_x = list(x_values)

    predictions = (
        predict(
            regression_x,
            regression_y,
            prediction_x,
        )
        if result is None
        else [float(value) for value in result]
    )

    slope, intercept = linear_regression(
        regression_x,
        regression_y,
    )

    lines = [
        "### 🔮 Prédictions",
        "",
        f"**Valeurs X utilisées pour la prédiction :** "
        f"{_format_series(prediction_x)}",
        "",
        f"**Pente :** a = {_format_number(slope)}",
        f"**Ordonnée à l'origine :** "
        f"b = {_format_number(intercept)}",
        "",
        "**Formule :**",
        "",
        "ŷ = ax + b",
        "",
    ]

    for x_value, prediction in zip(
        prediction_x,
        predictions,
    ):
        lines.extend(
            [
                (
                    f"Pour x = {_format_number(float(x_value))} :"
                ),
                (
                    f"ŷ = ({_format_number(slope)} × "
                    f"{_format_number(float(x_value))}) + "
                    f"{_format_number(intercept)}"
                ),
                f"ŷ = {_format_number(prediction)}",
                "",
            ]
        )

    return "\n".join(lines)


# ============================================================
# EXPLICATION DE R²
# ============================================================


def explain_coefficient_of_determination(
    x: Sequence[Number],
    y: Sequence[Number],
    result: float | None = None,
) -> str:
    """
    Génère une explication pédagogique du coefficient R².

    Formule :

        R² = 1 − SSE / SST
    """

    x_values = list(x)
    y_values = list(y)

    r_squared = (
        coefficient_of_determination(
            x_values,
            y_values,
        )
        if result is None
        else float(result)
    )

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    y_mean = sum(
        float(value)
        for value in y_values
    ) / len(y_values)

    predicted_values = regression_predictions(
        x_values,
        y_values,
    )

    sse = sum(
        (
            float(actual) - predicted
        ) ** 2
        for actual, predicted in zip(
            y_values,
            predicted_values,
        )
    )

    sst = sum(
        (
            float(actual) - y_mean
        ) ** 2
        for actual in y_values
    )

    percentage = r_squared * 100

    return "\n".join(
        [
            "### 📊 Coefficient de détermination R²",
            "",
            f"**Pente :** a = {_format_number(slope)}",
            f"**Ordonnée à l'origine :** "
            f"b = {_format_number(intercept)}",
            "",
            "**Valeurs prédites :**",
            "",
            _format_series(predicted_values),
            "",
            f"**SSE :** {_format_number(sse)}",
            f"**SST :** {_format_number(sst)}",
            "",
            "**Formule :**",
            "",
            "R² = 1 − SSE / SST",
            "",
            f"**R² = {_format_number(r_squared)}**",
            "",
            (
                f"Le modèle linéaire explique environ "
                f"{_format_number(percentage)} % "
                "de la variabilité observée de Y."
            ),
        ]
    )


# ============================================================
# INTERPRÉTATION DE R²
# ============================================================


def interpret_r_squared(
    r_squared: float,
) -> str:
    """
    Fournit une interprétation descriptive de R².

    Cette fonction ne remplace pas l'analyse statistique
    complète du modèle.
    """

    value = float(r_squared)

    if value < 0:
        return (
            "R² est négatif : le modèle linéaire explique "
            "moins bien les données que la prédiction "
            "constante basée sur la moyenne de Y."
        )

    if value <= 0.25:
        level = "faible"
    elif value <= 0.50:
        level = "modéré"
    elif value <= 0.75:
        level = "important"
    else:
        level = "très important"

    return (
        f"R² = {_format_number(value)}. "
        f"La proportion de variabilité expliquée par "
        f"le modèle est {level}."
    )


# ============================================================
# EXPLICATION COMPLÈTE
# ============================================================


def explain_regression(
    x: Sequence[Number],
    y: Sequence[Number],
) -> str:
    """
    Génère une explication complète de la régression
    linéaire.
    """

    x_values = list(x)
    y_values = list(y)

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    r_squared = coefficient_of_determination(
        x_values,
        y_values,
    )

    return "\n".join(
        [
            "## 📈 Analyse complète de la régression linéaire",
            "",
            f"**X :** {_format_series(x_values)}",
            f"**Y :** {_format_series(y_values)}",
            "",
            explain_linear_regression(
                x_values,
                y_values,
                (slope, intercept),
            ),
            "",
            explain_coefficient_of_determination(
                x_values,
                y_values,
                r_squared,
            ),
            "",
            "### 💡 Interprétation",
            "",
            interpret_r_squared(r_squared),
            "",
            "⚠️ Un R² élevé indique que le modèle linéaire "
            "s'ajuste bien aux données, mais ne prouve pas "
            "qu'une variable cause l'autre.",
        ]
    )


def explain_r_squared(
    x: Sequence[Number],
    y: Sequence[Number],
    result: float | None = None,
) -> str:
    """
    Explique le calcul du coefficient de détermination R².

    R² mesure la proportion de la variabilité de Y
    expliquée par le modèle de régression linéaire.

    Formule :

        R² = 1 - SSE / SST

    où :

        SSE = Σ(yᵢ - ŷᵢ)²
        SST = Σ(yᵢ - ȳ)²
    """

    x_values = list(x)
    y_values = list(y)

    if result is None:
        result = coefficient_of_determination(
            x_values,
            y_values,
        )

    slope, intercept = linear_regression(
        x_values,
        y_values,
    )

    y_mean = sum(float(value) for value in y_values) / len(y_values)

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

    percentage = result * 100

    return f"""
### 📐 Coefficient de détermination R²

Le coefficient de détermination mesure la proportion
de la variabilité de **Y** expliquée par le modèle
de régression linéaire.

#### 1. Données

**X :**
{_format_series(x_values)}

**Y :**
{_format_series(y_values)}

#### 2. Moyenne de Y

ȳ = {_format_number(y_mean)}

#### 3. Valeurs prédites

Les valeurs prédites sont calculées avec :

ŷ = ax + b

avec :

a = {_format_number(slope)}

b = {_format_number(intercept)}

#### 4. Somme des carrés des erreurs

SSE = Σ(yᵢ - ŷᵢ)²

SSE = {_format_number(sse)}

#### 5. Somme totale des carrés

SST = Σ(yᵢ - ȳ)²

SST = {_format_number(sst)}

#### 6. Formule de R²

R² = 1 - SSE / SST

#### 7. Résultat

**R² = {_format_number(result)}**

Soit environ **{_format_number(percentage)} %** de la
variabilité de Y expliquée par le modèle.

#### 💡 Interprétation

{interpret_r_squared(result)}

> ⚠️ Un coefficient R² élevé indique une bonne adéquation
> du modèle aux données, mais il ne permet pas à lui seul
> d'établir une relation de causalité entre X et Y.
"""