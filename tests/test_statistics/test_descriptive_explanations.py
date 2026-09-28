import pytest

from core.statistics.descriptive_explanations import (
    explain_data_range,
    explain_descriptive_statistics,
    explain_maximum,
    explain_mean,
    explain_median,
    explain_minimum,
    explain_mode,
    explain_percentile,
    explain_quartiles,
    explain_standard_deviation,
    explain_variance,
)


# ============================================================
# MOYENNE
# ============================================================


def test_explain_mean_contains_title():
    result = explain_mean([10, 12, 14, 16, 18])

    assert "Moyenne" in result


def test_explain_mean_contains_formula():
    result = explain_mean([10, 12, 14])

    assert "moyenne = somme / nombre de valeurs" in result


def test_explain_mean_contains_result():
    result = explain_mean([10, 12, 14])

    assert "La moyenne est donc 12." in result


# ============================================================
# MEDIANE
# ============================================================


def test_explain_median_with_odd_number_of_values():
    result = explain_median([5, 1, 3])

    assert "Médiane" in result
    assert "Données ordonnées : 1, 3, 5" in result
    assert "La médiane est donc 3." in result


def test_explain_median_with_even_number_of_values():
    result = explain_median([1, 2, 3, 4])

    assert "Médiane" in result
    assert "Valeurs centrales : 2 et 3" in result
    assert "Calcul : (2 + 3) / 2 = 2.5" in result
    assert "La médiane est donc 2.5." in result


# ============================================================
# MODE
# ============================================================


def test_explain_mode_with_unique_mode():
    result = explain_mode([1, 2, 2, 3])

    assert "Mode" in result
    assert "2 : 2 occurrence(s)" in result
    assert "Le mode est donc 2." in result


def test_explain_mode_with_multiple_modes():
    result = explain_mode([1, 1, 2, 2, 3])

    assert "Plusieurs valeurs possèdent la fréquence maximale." in result
    assert "Les modes sont donc : 1, 2." in result


# ============================================================
# MINIMUM
# ============================================================


def test_explain_minimum():
    result = explain_minimum([8, 3, 12, 5, 1])

    assert "Minimum" in result
    assert "Plus petite valeur = 1" in result
    assert "Le minimum est donc 1." in result


# ============================================================
# MAXIMUM
# ============================================================


def test_explain_maximum():
    result = explain_maximum([8, 3, 12, 5, 1])

    assert "Maximum" in result
    assert "Plus grande valeur = 12" in result
    assert "Le maximum est donc 12." in result


# ============================================================
# ETENDUE
# ============================================================


def test_explain_data_range():
    result = explain_data_range([2, 5, 10])

    assert "Étendue" in result
    assert "étendue = maximum - minimum" in result
    assert "Maximum = 10" in result
    assert "Minimum = 2" in result
    assert "L'étendue est donc 8." in result


# ============================================================
# VARIANCE
# ============================================================


def test_explain_variance():
    result = explain_variance([1, 2, 3, 4, 5])

    assert "Variance" in result
    assert "variance de population" in result
    assert "σ² = Σ(xᵢ - μ)² / n" in result
    assert "Moyenne = 3" in result
    assert "La variance est donc 2." in result


def test_explain_variance_contains_squared_deviations():
    result = explain_variance([1, 2, 3])

    assert "(1 - 2)² = 1" in result
    assert "(2 - 2)² = 0" in result
    assert "(3 - 2)² = 1" in result


# ============================================================
# ECART-TYPE
# ============================================================


def test_explain_standard_deviation():
    result = explain_standard_deviation([1, 2, 3, 4, 5])

    assert "Écart-type" in result
    assert "σ = √σ²" in result
    assert "Variance = 2" in result
    assert "L'écart-type est donc 1.41421." in result


def test_explain_standard_deviation_with_zero_variance():
    result = explain_standard_deviation([5, 5, 5])

    assert "Variance = 0" in result
    assert "L'écart-type est donc 0." in result


# ============================================================
# QUARTILES
# ============================================================


def test_explain_quartiles():
    result = explain_quartiles([1, 2, 3, 4, 5])

    assert "Quartiles" in result
    assert "Q1 = 25e percentile = 2" in result
    assert "Q2 = 50e percentile = 3" in result
    assert "Q3 = 75e percentile = 4" in result
    assert "Q2 correspond également à la médiane." in result


def test_explain_quartiles_contains_ordered_data():
    result = explain_quartiles([5, 1, 4, 2, 3])

    assert "Données ordonnées : 1, 2, 3, 4, 5" in result


# ============================================================
# PERCENTILE
# ============================================================


def test_explain_percentile_50():
    result = explain_percentile([1, 2, 3, 4, 5], 50)

    assert "Percentile 50" in result
    assert "Percentile demandé : 50" in result
    assert "Le percentile 50 est donc 3." in result


def test_explain_percentile_25():
    result = explain_percentile([1, 2, 3, 4, 5], 25)

    assert "Percentile 25" in result
    assert "Le percentile 25 est donc 2." in result


def test_explain_percentile_with_interpolation():
    result = explain_percentile([10, 20, 30, 40], 25)

    assert "Percentile demandé : 25" in result
    assert "Résultat : 17.5" in result


# ============================================================
# SYNTHESE DES STATISTIQUES DESCRIPTIVES
# ============================================================


def test_explain_descriptive_statistics():
    result = explain_descriptive_statistics(
        [1, 2, 3, 4, 5]
    )

    assert "Statistiques descriptives" in result
    assert "Données" in result
    assert "Moyenne : 3" in result
    assert "Médiane : 3" in result
    assert "Mode : 1, 2, 3, 4, 5" in result
    assert "Minimum : 1" in result
    assert "Maximum : 5" in result
    assert "Étendue : 4" in result
    assert "Variance : 2" in result
    assert "Écart-type : 1.41421" in result
    assert "Q1 : 2" in result
    assert "Q2 : 3" in result
    assert "Q3 : 4" in result


# ============================================================
# GESTION DES ERREURS
# ============================================================


def test_explain_mean_with_empty_data_raises_value_error():
    with pytest.raises(ValueError):
        explain_mean([])


def test_explain_median_with_empty_data_raises_value_error():
    with pytest.raises(ValueError):
        explain_median([])


def test_explain_variance_with_empty_data_raises_value_error():
    with pytest.raises(ValueError):
        explain_variance([])


def test_explain_quartiles_with_empty_data_raises_value_error():
    with pytest.raises(ValueError):
        explain_quartiles([])


def test_explain_percentile_with_invalid_percentile_raises_value_error():
    with pytest.raises(ValueError):
        explain_percentile([1, 2, 3], 101)


def test_explain_mean_with_non_numeric_data_raises_type_error():
    with pytest.raises(TypeError):
        explain_mean([1, 2, "3"])