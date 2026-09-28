from __future__ import annotations

from collections.abc import Sequence

from core.statistics.chi_square import (
    chi_square_critical_value,
    chi_square_p_value,
    chi_square_statistic,
    degrees_of_freedom,
    expected_frequencies,
)


Number = int | float


# ============================================================
# OUTILS DE FORMATAGE
# ============================================================


def _format_number(value: Number) -> str:
    """
    Formate un nombre pour une présentation lisible.
    """

    numeric_value = float(value)

    if numeric_value == 0:
        return "0"

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:.6f}".rstrip("0").rstrip(".")


def _format_probability(value: Number) -> str:
    """
    Formate une probabilité sous forme décimale et en pourcentage.
    """

    numeric_value = float(value)

    return (
        f"{numeric_value:.6f} "
        f"({_format_number(numeric_value * 100)} %)"
    )


def _format_table(
    table: Sequence[Sequence[Number]],
) -> str:
    """
    Transforme un tableau numérique en représentation textuelle.
    """

    rows = []

    for row in table:
        formatted_row = ", ".join(
            _format_number(value)
            for value in row
        )

        rows.append(
            f"[{formatted_row}]"
        )

    return "\n".join(rows)


# ============================================================
# HYPOTHÈSES
# ============================================================


def explain_chi_square_hypotheses() -> str:
    """
    Explique les hypothèses du test du Khi-deux d'indépendance.
    """

    return """
## 🧪 Hypothèses du test du Khi-deux

Le test du Khi-deux d'indépendance permet d'étudier
s'il existe une association statistique entre deux
variables qualitatives.

### Hypothèse nulle H₀

**H₀ : les deux variables sont indépendantes.**

Autrement dit, la répartition d'une variable ne dépend
pas de l'autre variable.

### Hypothèse alternative H₁

**H₁ : les deux variables ne sont pas indépendantes.**

Il existe donc une association statistique entre
les deux variables.

### Règle de décision

On compare généralement la **p-value** au niveau
de signification α.

- Si **p-value < α** : on rejette H₀.
- Si **p-value ≥ α** : on ne rejette pas H₀.

Le test met en évidence une association statistique,
mais il ne permet pas à lui seul de conclure
à une relation de causalité.
""".strip()


# ============================================================
# EFFECTIFS ATTENDUS
# ============================================================


def explain_expected_frequencies(
    table: Sequence[Sequence[Number]],
    result: Sequence[Sequence[Number]] | None = None,
) -> str:
    """
    Explique le calcul des effectifs théoriques attendus.
    """

    expected = (
        expected_frequencies(table)
        if result is None
        else result
    )

    rows = len(table)
    columns = len(table[0])

    row_totals = [
        sum(float(value) for value in row)
        for row in table
    ]

    column_totals = [
        sum(
            float(table[row_index][column_index])
            for row_index in range(rows)
        )
        for column_index in range(columns)
    ]

    total = sum(row_totals)

    lines = [
        "## 📊 Effectifs attendus",
        "",
        "Les effectifs attendus représentent les effectifs",
        "que l'on obtiendrait si les deux variables étaient",
        "réellement indépendantes.",
        "",
        "### Formule",
        "",
        "**Eᵢⱼ = (Total ligne i × Total colonne j) / "
        "Total général**",
        "",
        f"Total général = **{_format_number(total)}**",
        "",
        "### Totaux des lignes",
        "",
    ]

    for index, row_total in enumerate(row_totals, start=1):
        lines.append(
            f"- Ligne {index} : "
            f"{_format_number(row_total)}"
        )

    lines.extend(
        [
            "",
            "### Totaux des colonnes",
            "",
        ]
    )

    for index, column_total in enumerate(
        column_totals,
        start=1,
    ):
        lines.append(
            f"- Colonne {index} : "
            f"{_format_number(column_total)}"
        )

    lines.extend(
        [
            "",
            "### Tableau des effectifs attendus",
            "",
            "```text",
            _format_table(expected),
            "```",
        ]
    )

    return "\n".join(lines)


# ============================================================
# STATISTIQUE DU KHI-DEUX
# ============================================================


def explain_chi_square_statistic(
    table: Sequence[Sequence[Number]],
    result: Number | None = None,
) -> str:
    """
    Explique le calcul de la statistique du Khi-deux.
    """

    observed = table

    statistic = (
        chi_square_statistic(observed)
        if result is None
        else float(result)
    )

    expected = expected_frequencies(observed)

    lines = [
        "## 📐 Statistique du Khi-deux",
        "",
        "La statistique du Khi-deux mesure l'écart entre",
        "les effectifs observés et les effectifs attendus.",
        "",
        "### Formule",
        "",
        "**χ² = Σ ((Oᵢⱼ − Eᵢⱼ)² / Eᵢⱼ)**",
        "",
        "avec :",
        "",
        "- **Oᵢⱼ** : effectif observé ;",
        "- **Eᵢⱼ** : effectif attendu.",
        "",
        "### Effectifs observés",
        "",
        "```text",
        _format_table(observed),
        "```",
        "",
        "### Effectifs attendus",
        "",
        "```text",
        _format_table(expected),
        "```",
        "",
        f"### Résultat",
        "",
        f"**χ² = {_format_number(statistic)}**",
    ]

    return "\n".join(lines)


# ============================================================
# DEGRÉS DE LIBERTÉ
# ============================================================
def explain_degrees_of_freedom(
    table: Sequence[Sequence[Number]],
    result: int | None = None,
) -> str:
    """
    Explique le calcul des degrés de liberté.
    """

    # Validation du tableau via le module principal.
    # Cela garantit que les erreurs sont cohérentes
    # avec les fonctions statistiques du cœur.
    calculated_df = degrees_of_freedom(table)

    rows = len(table)
    columns = len(table[0])

    df = (
        calculated_df
        if result is None
        else int(result)
    )

    return f"""
## 🔢 Degrés de liberté

Pour un tableau de contingence contenant :

- **{rows} lignes**
- **{columns} colonnes**

la formule est :

**ddl = (nombre de lignes − 1) × (nombre de colonnes − 1)**

Donc :

**ddl = ({rows} − 1) × ({columns} − 1)**

**ddl = {df}**
""".strip()


# ============================================================
# P-VALUE
# ============================================================


def explain_chi_square_p_value(
    table: Sequence[Sequence[Number]],
    result: Number | None = None,
) -> str:
    """
    Explique la p-value du test du Khi-deux.
    """

    statistic = chi_square_statistic(table)
    df = degrees_of_freedom(table)

    p_value = (
        chi_square_p_value(table)
        if result is None
        else float(result)
    )

    return f"""
## 📉 p-value

La p-value mesure la compatibilité des données observées
avec l'hypothèse nulle H₀.

Pour :

- **χ² = {_format_number(statistic)}**
- **ddl = {df}**

on obtient :

**p-value = {_format_probability(p_value)}**

Une p-value faible indique que les données observées
sont moins compatibles avec l'hypothèse d'indépendance.

La p-value doit être comparée au niveau de signification
α choisi pour le test.
""".strip()


# ============================================================
# VALEUR CRITIQUE
# ============================================================


def explain_chi_square_critical_value(
    table: Sequence[Sequence[Number]],
    significance_level: Number = 0.05,
    result: Number | None = None,
) -> str:
    """
    Explique la valeur critique du Khi-deux.
    """

    df = degrees_of_freedom(table)

    critical_value = (
        chi_square_critical_value(
            table,
            significance_level,
        )
        if result is None
        else float(result)
    )

    return f"""
## 🎯 Valeur critique

Le niveau de signification choisi est :

**α = {_format_number(significance_level)}**
({_format_number(float(significance_level) * 100)} %)

Avec :

**ddl = {df}**

la valeur critique du Khi-deux est :

**χ² critique = {_format_number(critical_value)}**

La règle de décision basée sur la valeur critique est :

- si **χ² calculé > χ² critique** → rejet de H₀ ;
- si **χ² calculé ≤ χ² critique** → non-rejet de H₀.
""".strip()


# ============================================================
# DÉCISION
# ============================================================


def explain_chi_square_decision(
    p_value: Number,
    significance_level: Number = 0.05,
) -> str:
    """
    Explique la décision statistique à partir de la p-value.
    """

    numeric_p_value = float(p_value)
    alpha = float(significance_level)

    if numeric_p_value < alpha:
        decision = "rejet de H₀"
        interpretation = (
            "Les données fournissent des éléments statistiques "
            "en faveur d'une association entre les deux variables."
        )
    else:
        decision = "non-rejet de H₀"
        interpretation = (
            "Les données ne fournissent pas suffisamment "
            "d'éléments statistiques pour rejeter l'hypothèse "
            "d'indépendance."
        )

    return f"""
## ⚖️ Décision statistique

On compare :

- **p-value = {_format_number(numeric_p_value)}**
- **α = {_format_number(alpha)}**

Comme :

**p-value {'<' if numeric_p_value < alpha else '≥'} α**

la décision est :

### {decision}

{interpretation}

Cette conclusion concerne uniquement l'association
statistique étudiée. Elle ne permet pas à elle seule
d'établir une causalité.
""".strip()


# ============================================================
# INTERPRÉTATION
# ============================================================


def interpret_chi_square(
    p_value: Number,
    significance_level: Number = 0.05,
) -> str:
    """
    Produit une interprétation courte du résultat du test.
    """

    numeric_p_value = float(p_value)
    alpha = float(significance_level)

    if numeric_p_value < alpha:
        return (
            "Le test du Khi-deux indique une association "
            "statistiquement significative entre les deux "
            "variables au niveau de signification choisi."
        )

    return (
        "Le test du Khi-deux n'indique pas d'association "
        "statistiquement significative entre les deux "
        "variables au niveau de signification choisi."
    )


# ============================================================
# EXPLICATION COMPLÈTE
# ============================================================


def explain_chi_square_test(
    table: Sequence[Sequence[Number]],
    significance_level: Number = 0.05,
    result: dict[str, object] | None = None,
) -> str:
    """
    Génère une explication complète du test du Khi-deux.

    Parameters
    ----------
    table :
        Tableau des effectifs observés.

    significance_level :
        Niveau de signification alpha.

    result :
        Résultat déjà calculé par chi_square_test().
        Si None, le résultat est calculé automatiquement.
    """

    if result is None:
        from core.statistics.chi_square import chi_square_test

        result = chi_square_test(
            table,
            significance_level,
        )

    statistic = float(result["statistic"])
    df = int(result["degrees_of_freedom"])
    p_value = float(result["p_value"])
    critical_value = float(result["critical_value"])

    decision = bool(
        result["reject_null_hypothesis"]
    )

    sections = [
        "# 🧪 Test du Khi-deux d'indépendance",
        "",
        "## 1. Objectif du test",
        "",
        "Le test du Khi-deux permet de déterminer s'il existe",
        "une association statistique entre deux variables",
        "qualitatives.",
        "",
        explain_chi_square_hypotheses(),
        "",
        explain_expected_frequencies(table),
        "",
        explain_chi_square_statistic(
            table,
            statistic,
        ),
        "",
        explain_degrees_of_freedom(
            table,
            df,
        ),
        "",
        explain_chi_square_p_value(
            table,
            p_value,
        ),
        "",
        explain_chi_square_critical_value(
            table,
            significance_level,
            critical_value,
        ),
        "",
        explain_chi_square_decision(
            p_value,
            significance_level,
        ),
        "",
        "## 📝 Interprétation finale",
        "",
        interpret_chi_square(
            p_value,
            significance_level,
        ),
        "",
        "### Résumé numérique",
        "",
        f"- **χ² calculé** : {_format_number(statistic)}",
        f"- **ddl** : {df}",
        f"- **p-value** : {_format_probability(p_value)}",
        (
            f"- **χ² critique** : "
            f"{_format_number(critical_value)}"
        ),
        (
            f"- **H₀** : "
            f"{'rejetée' if decision else 'non rejetée'}"
        ),
        "",
        "⚠️ **Attention :** une association statistique "
        "ne signifie pas nécessairement qu'une variable "
        "cause l'autre.",
    ]

    return "\n".join(sections)


# ============================================================
# ALIAS COURT
# ============================================================


def explain_chi_square(
    table: Sequence[Sequence[Number]],
    significance_level: Number = 0.05,
) -> str:
    """
    Alias pratique pour générer l'explication complète.
    """

    return explain_chi_square_test(
        table,
        significance_level,
    )