from __future__ import annotations

from typing import Iterable

from core.statistics.descriptive import (
    data_range,
    maximum,
    mean,
    median,
    minimum,
    mode,
    percentile,
    quartiles,
    standard_deviation,
    variance,
)


Number = int | float


def _format_number(value: Number) -> str:
    """
    Formate proprement une valeur numérique.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return f"{value:.6g}"


def _format_data(data: Iterable[Number]) -> str:
    """
    Formate une série de données pour l'affichage.
    """
    return ", ".join(
        _format_number(value)
        for value in data
    )


def explain_mean(data: Iterable[Number]) -> str:
    """
    Explique le calcul de la moyenne.
    """
    values = tuple(data)
    result = mean(values)

    total = sum(values)
    count = len(values)

    return (
        "### Moyenne\n\n"
        "La moyenne arithmétique est obtenue en additionnant "
        "toutes les valeurs puis en divisant la somme par "
        "le nombre de valeurs.\n\n"
        f"Données : {_format_data(values)}\n\n"
        f"Somme = {_format_number(total)}\n\n"
        f"Nombre de valeurs = {count}\n\n"
        f"Formule : moyenne = somme / nombre de valeurs\n\n"
        f"Calcul : {_format_number(total)} / {count} "
        f"= {_format_number(result)}\n\n"
        f"**La moyenne est donc {_format_number(result)}.**"
    )


def explain_median(data: Iterable[Number]) -> str:
    """
    Explique le calcul de la médiane.
    """
    values = tuple(data)
    sorted_values = sorted(values)
    result = median(values)

    count = len(sorted_values)

    explanation = (
        "### Médiane\n\n"
        "La médiane est la valeur qui partage une série "
        "ordonnée en deux parties de même effectif.\n\n"
        f"Données ordonnées : {_format_data(sorted_values)}\n\n"
    )

    if count % 2 == 1:
        middle_index = count // 2
        middle_value = sorted_values[middle_index]

        explanation += (
            f"Il y a {count} valeurs, donc une seule valeur "
            "se trouve au centre.\n\n"
            f"Valeur centrale = {_format_number(middle_value)}\n\n"
            f"**La médiane est donc {_format_number(result)}.**"
        )
    else:
        middle_right = count // 2
        middle_left = middle_right - 1

        left_value = sorted_values[middle_left]
        right_value = sorted_values[middle_right]

        explanation += (
            f"Il y a {count} valeurs, donc nous prenons "
            "la moyenne des deux valeurs centrales.\n\n"
            f"Valeurs centrales : {_format_number(left_value)} "
            f"et {_format_number(right_value)}\n\n"
            f"Calcul : ({_format_number(left_value)} + "
            f"{_format_number(right_value)}) / 2 "
            f"= {_format_number(result)}\n\n"
            f"**La médiane est donc {_format_number(result)}.**"
        )

    return explanation


def explain_mode(data: Iterable[Number]) -> str:
    """
    Explique la détermination du mode.
    """
    values = tuple(data)
    result = mode(values)

    frequencies: dict[Number, int] = {}

    for value in values:
        frequencies[value] = frequencies.get(value, 0) + 1

    frequency_text = "\n".join(
        f"- {_format_number(value)} : {frequency} occurrence(s)"
        for value, frequency in sorted(frequencies.items())
    )

    explanation = (
        "### Mode\n\n"
        "Le mode est la valeur ou l'ensemble de valeurs "
        "qui apparaît le plus souvent dans une série.\n\n"
        f"Données : {_format_data(values)}\n\n"
        "### Fréquences\n\n"
        f"{frequency_text}\n\n"
    )

    if isinstance(result, tuple):
        modes_text = ", ".join(
            _format_number(value)
            for value in result
        )

        explanation += (
            "Plusieurs valeurs possèdent la fréquence maximale.\n\n"
            f"**Les modes sont donc : {modes_text}.**"
        )
    else:
        explanation += (
            f"**Le mode est donc {_format_number(result)}.**"
        )

    return explanation


def explain_minimum(data: Iterable[Number]) -> str:
    """
    Explique la recherche du minimum.
    """
    values = tuple(data)
    result = minimum(values)

    return (
        "### Minimum\n\n"
        "Le minimum est la plus petite valeur "
        "de la série statistique.\n\n"
        f"Données : {_format_data(values)}\n\n"
        f"Plus petite valeur = {_format_number(result)}\n\n"
        f"**Le minimum est donc {_format_number(result)}.**"
    )


def explain_maximum(data: Iterable[Number]) -> str:
    """
    Explique la recherche du maximum.
    """
    values = tuple(data)
    result = maximum(values)

    return (
        "### Maximum\n\n"
        "Le maximum est la plus grande valeur "
        "de la série statistique.\n\n"
        f"Données : {_format_data(values)}\n\n"
        f"Plus grande valeur = {_format_number(result)}\n\n"
        f"**Le maximum est donc {_format_number(result)}.**"
    )


def explain_data_range(data: Iterable[Number]) -> str:
    """
    Explique le calcul de l'étendue.
    """
    values = tuple(data)

    minimum_value = minimum(values)
    maximum_value = maximum(values)
    result = data_range(values)

    return (
        "### Étendue\n\n"
        "L'étendue mesure l'écart entre la plus grande "
        "et la plus petite valeur.\n\n"
        "Formule :\n\n"
        "étendue = maximum - minimum\n\n"
        f"Maximum = {_format_number(maximum_value)}\n\n"
        f"Minimum = {_format_number(minimum_value)}\n\n"
        f"Calcul : {_format_number(maximum_value)} - "
        f"{_format_number(minimum_value)} "
        f"= {_format_number(result)}\n\n"
        f"**L'étendue est donc {_format_number(result)}.**"
    )


def explain_variance(data: Iterable[Number]) -> str:
    """
    Explique le calcul de la variance de population.
    """
    values = tuple(data)
    result = variance(values)
    average = mean(values)

    squared_deviations = [
        (value - average) ** 2
        for value in values
    ]

    deviations_text = "\n".join(
        f"- ({_format_number(value)} - "
        f"{_format_number(average)})² = "
        f"{_format_number(squared)}"
        for value, squared in zip(values, squared_deviations)
    )

    return (
        "### Variance\n\n"
        "La variance mesure la dispersion des valeurs "
        "autour de la moyenne.\n\n"
        "Nous utilisons ici la variance de population.\n\n"
        "Formule :\n\n"
        "σ² = Σ(xᵢ - μ)² / n\n\n"
        f"Moyenne = {_format_number(average)}\n\n"
        "### Écarts au carré\n\n"
        f"{deviations_text}\n\n"
        f"Somme des écarts au carré = "
        f"{_format_number(sum(squared_deviations))}\n\n"
        f"Nombre de valeurs = {len(values)}\n\n"
        f"Calcul : { _format_number(sum(squared_deviations)) } "
        f"/ {len(values)} = {_format_number(result)}\n\n"
        f"**La variance est donc {_format_number(result)}.**"
    )


def explain_standard_deviation(
    data: Iterable[Number],
) -> str:
    """
    Explique le calcul de l'écart-type.
    """
    values = tuple(data)
    variance_value = variance(values)
    result = standard_deviation(values)

    return (
        "### Écart-type\n\n"
        "L'écart-type mesure la dispersion des valeurs "
        "autour de la moyenne.\n\n"
        "Il correspond à la racine carrée de la variance.\n\n"
        "Formule :\n\n"
        "σ = √σ²\n\n"
        f"Variance = {_format_number(variance_value)}\n\n"
        f"Calcul : √{_format_number(variance_value)} "
        f"= {_format_number(result)}\n\n"
        f"**L'écart-type est donc {_format_number(result)}.**"
    )


def explain_quartiles(data: Iterable[Number]) -> str:
    """
    Explique le calcul des quartiles.
    """
    values = tuple(data)

    q1, q2, q3 = quartiles(values)

    return (
        "### Quartiles\n\n"
        "Les quartiles divisent une série ordonnée en "
        "quatre parties.\n\n"
        f"Données ordonnées : {_format_data(sorted(values))}\n\n"
        f"- Q1 = 25e percentile = {_format_number(q1)}\n"
        f"- Q2 = 50e percentile = {_format_number(q2)}\n"
        f"- Q3 = 75e percentile = {_format_number(q3)}\n\n"
        "Q2 correspond également à la médiane.\n\n"
        f"**Les quartiles sont donc "
        f"(Q1, Q2, Q3) = "
        f"({_format_number(q1)}, "
        f"{_format_number(q2)}, "
        f"{_format_number(q3)}).**"
    )


def explain_percentile(
    data: Iterable[Number],
    p: float,
) -> str:
    """
    Explique le calcul d'un percentile.
    """
    values = tuple(data)
    result = percentile(values, p)

    return (
        f"### Percentile {p:g}\n\n"
        f"Le percentile {p:g} indique une valeur sous laquelle "
        "se situe approximativement la proportion correspondante "
        "des observations dans la série ordonnée.\n\n"
        f"Données ordonnées : {_format_data(sorted(values))}\n\n"
        f"Percentile demandé : {p:g}\n\n"
        f"Résultat : {_format_number(result)}\n\n"
        f"**Le percentile {p:g} est donc "
        f"{_format_number(result)}.**"
    )


def explain_descriptive_statistics(
    data: Iterable[Number],
) -> str:
    """
    Génère une synthèse pédagogique des principales
    statistiques descriptives.
    """
    values = tuple(data)

    average = mean(values)
    median_value = median(values)
    mode_value = mode(values)
    minimum_value = minimum(values)
    maximum_value = maximum(values)
    range_value = data_range(values)
    variance_value = variance(values)
    standard_deviation_value = standard_deviation(values)
    q1, q2, q3 = quartiles(values)

    if isinstance(mode_value, tuple):
        mode_text = ", ".join(
            _format_number(value)
            for value in mode_value
        )
    else:
        mode_text = _format_number(mode_value)

    return (
        "## Statistiques descriptives\n\n"
        f"### Données\n\n"
        f"{_format_data(values)}\n\n"
        "### Résultats\n\n"
        f"- Moyenne : {_format_number(average)}\n"
        f"- Médiane : {_format_number(median_value)}\n"
        f"- Mode : {mode_text}\n"
        f"- Minimum : {_format_number(minimum_value)}\n"
        f"- Maximum : {_format_number(maximum_value)}\n"
        f"- Étendue : {_format_number(range_value)}\n"
        f"- Variance : {_format_number(variance_value)}\n"
        f"- Écart-type : "
        f"{_format_number(standard_deviation_value)}\n"
        f"- Q1 : {_format_number(q1)}\n"
        f"- Q2 : {_format_number(q2)}\n"
        f"- Q3 : {_format_number(q3)}"
    )