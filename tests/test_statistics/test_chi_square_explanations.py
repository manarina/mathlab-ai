from __future__ import annotations

import math

import pytest

from core.statistics.chi_square_explanations import (
    explain_chi_square,
    explain_chi_square_critical_value,
    explain_chi_square_decision,
    explain_chi_square_hypotheses,
    explain_chi_square_p_value,
    explain_chi_square_statistic,
    explain_chi_square_test,
    explain_degrees_of_freedom,
    explain_expected_frequencies,
    interpret_chi_square,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================


TABLE_2X2 = [
    [30, 20],
    [20, 30],
]

TABLE_3X2 = [
    [10, 20],
    [20, 30],
    [30, 10],
]

PERFECT_INDEPENDENCE_TABLE = [
    [10, 10],
    [10, 10],
]


# ============================================================
# TEST explain_chi_square_hypotheses
# ============================================================


def test_explain_chi_square_hypotheses_contains_title():
    result = explain_chi_square_hypotheses()

    assert "Hypothèses" in result
    assert "Khi-deux" in result


def test_explain_chi_square_hypotheses_contains_h0():
    result = explain_chi_square_hypotheses()

    assert "H₀" in result
    assert "indépendantes" in result


def test_explain_chi_square_hypotheses_contains_h1():
    result = explain_chi_square_hypotheses()

    assert "H₁" in result
    assert "association" in result


def test_explain_chi_square_hypotheses_contains_decision_rule():
    result = explain_chi_square_hypotheses()

    assert "p-value < α" in result
    assert "p-value ≥ α" in result


def test_explain_chi_square_hypotheses_mentions_causality():
    result = explain_chi_square_hypotheses()

    assert "causalité" in result


# ============================================================
# TEST explain_expected_frequencies
# ============================================================


def test_explain_expected_frequencies_contains_title():
    result = explain_expected_frequencies(TABLE_2X2)

    assert "Effectifs attendus" in result


def test_explain_expected_frequencies_contains_formula():
    result = explain_expected_frequencies(TABLE_2X2)

    assert "Eᵢⱼ" in result
    assert "Total ligne" in result
    assert "Total colonne" in result


def test_explain_expected_frequencies_contains_general_total():
    result = explain_expected_frequencies(TABLE_2X2)

    assert "Total général" in result
    assert "100" in result


def test_explain_expected_frequencies_contains_row_totals():
    result = explain_expected_frequencies(TABLE_2X2)

    assert "Ligne 1" in result
    assert "Ligne 2" in result


def test_explain_expected_frequencies_contains_column_totals():
    result = explain_expected_frequencies(TABLE_2X2)

    assert "Colonne 1" in result
    assert "Colonne 2" in result


def test_explain_expected_frequencies_contains_expected_values():
    result = explain_expected_frequencies(TABLE_2X2)

    assert "[25, 25]" in result


def test_explain_expected_frequencies_3x2():
    result = explain_expected_frequencies(TABLE_3X2)

    assert "[15, 15]" in result
    assert "[25, 25]" in result
    assert "[20, 20]" in result


def test_explain_expected_frequencies_accepts_existing_result():
    expected = [
        [25.0, 25.0],
        [25.0, 25.0],
    ]

    result = explain_expected_frequencies(
        TABLE_2X2,
        expected,
    )

    assert "[25, 25]" in result


# ============================================================
# TEST explain_chi_square_statistic
# ============================================================


def test_explain_chi_square_statistic_contains_title():
    result = explain_chi_square_statistic(TABLE_2X2)

    assert "Statistique du Khi-deux" in result


def test_explain_chi_square_statistic_contains_formula():
    result = explain_chi_square_statistic(TABLE_2X2)

    assert "χ²" in result
    assert "Oᵢⱼ" in result
    assert "Eᵢⱼ" in result


def test_explain_chi_square_statistic_contains_observed_values():
    result = explain_chi_square_statistic(TABLE_2X2)

    assert "[30, 20]" in result
    assert "[20, 30]" in result


def test_explain_chi_square_statistic_contains_expected_values():
    result = explain_chi_square_statistic(TABLE_2X2)

    assert "[25, 25]" in result


def test_explain_chi_square_statistic_contains_result():
    result = explain_chi_square_statistic(TABLE_2X2)

    assert "χ² = 4" in result


def test_explain_chi_square_statistic_with_explicit_result():
    result = explain_chi_square_statistic(
        TABLE_2X2,
        12.5,
    )

    assert "χ² = 12.5" in result


# ============================================================
# TEST explain_degrees_of_freedom
# ============================================================


def test_explain_degrees_of_freedom_2x2():
    result = explain_degrees_of_freedom(TABLE_2X2)

    assert "Degrés de liberté" in result
    assert "2 lignes" in result
    assert "2 colonnes" in result
    assert "ddl = 1" in result


def test_explain_degrees_of_freedom_3x2():
    result = explain_degrees_of_freedom(TABLE_3X2)

    assert "3 lignes" in result
    assert "2 colonnes" in result
    assert "ddl = 2" in result


def test_explain_degrees_of_freedom_contains_formula():
    result = explain_degrees_of_freedom(TABLE_2X2)

    assert "(nombre de lignes − 1)" in result
    assert "(nombre de colonnes − 1)" in result


def test_explain_degrees_of_freedom_with_explicit_result():
    result = explain_degrees_of_freedom(
        TABLE_2X2,
        99,
    )

    assert "ddl = 99" in result


# ============================================================
# TEST explain_chi_square_p_value
# ============================================================


def test_explain_chi_square_p_value_contains_title():
    result = explain_chi_square_p_value(TABLE_2X2)

    assert "p-value" in result


def test_explain_chi_square_p_value_contains_statistic():
    result = explain_chi_square_p_value(TABLE_2X2)

    assert "χ² = 4" in result


def test_explain_chi_square_p_value_contains_degrees_of_freedom():
    result = explain_chi_square_p_value(TABLE_2X2)

    assert "ddl = 1" in result


def test_explain_chi_square_p_value_contains_result():
    result = explain_chi_square_p_value(TABLE_2X2)

    assert "p-value =" in result


def test_explain_chi_square_p_value_for_independence():
    result = explain_chi_square_p_value(
        PERFECT_INDEPENDENCE_TABLE
    )

    assert "1" in result


def test_explain_chi_square_p_value_with_explicit_result():
    result = explain_chi_square_p_value(
        TABLE_2X2,
        0.025,
    )

    assert "0.025" in result
    assert "2.5 %" in result


# ============================================================
# TEST explain_chi_square_critical_value
# ============================================================


def test_explain_chi_square_critical_value_contains_title():
    result = explain_chi_square_critical_value(
        TABLE_2X2
    )

    assert "Valeur critique" in result


def test_explain_chi_square_critical_value_default_alpha():
    result = explain_chi_square_critical_value(
        TABLE_2X2
    )

    assert "α = 0.05" in result
    assert "3.841459" in result


def test_explain_chi_square_critical_value_alpha_001():
    result = explain_chi_square_critical_value(
        TABLE_2X2,
        significance_level=0.01,
    )

    assert "α = 0.01" in result
    assert "6.634897" in result


def test_explain_chi_square_critical_value_contains_degree_of_freedom():
    result = explain_chi_square_critical_value(
        TABLE_2X2
    )

    assert "ddl = 1" in result


def test_explain_chi_square_critical_value_contains_decision_rule():
    result = explain_chi_square_critical_value(
        TABLE_2X2
    )

    assert "χ² calculé > χ² critique" in result
    assert "χ² calculé ≤ χ² critique" in result


def test_explain_chi_square_critical_value_with_explicit_result():
    result = explain_chi_square_critical_value(
        TABLE_2X2,
        significance_level=0.05,
        result=10.5,
    )

    assert "χ² critique = 10.5" in result


# ============================================================
# TEST explain_chi_square_decision
# ============================================================


def test_explain_chi_square_decision_rejects_h0():
    result = explain_chi_square_decision(
        0.01,
        0.05,
    )

    assert "rejet de H₀" in result
    assert "p-value < α" in result


def test_explain_chi_square_decision_does_not_reject_h0():
    result = explain_chi_square_decision(
        0.10,
        0.05,
    )

    assert "non-rejet de H₀" in result
    assert "p-value ≥ α" in result


def test_explain_chi_square_decision_equal_alpha():
    result = explain_chi_square_decision(
        0.05,
        0.05,
    )

    assert "non-rejet de H₀" in result


def test_explain_chi_square_decision_contains_values():
    result = explain_chi_square_decision(
        0.01,
        0.05,
    )

    assert "0.01" in result
    assert "0.05" in result


def test_explain_chi_square_decision_mentions_causality():
    result = explain_chi_square_decision(
        0.01,
        0.05,
    )

    assert "causalité" in result


# ============================================================
# TEST interpret_chi_square
# ============================================================


def test_interpret_chi_square_significant():
    result = interpret_chi_square(
        0.01,
        0.05,
    )

    assert "association" in result
    assert "significative" in result


def test_interpret_chi_square_not_significant():
    result = interpret_chi_square(
        0.10,
        0.05,
    )

    assert "association" in result
    assert "pas" in result
    assert "significative" in result


def test_interpret_chi_square_equal_alpha():
    result = interpret_chi_square(
        0.05,
        0.05,
    )

    assert "pas" in result


# ============================================================
# TEST explain_chi_square_test
# ============================================================


def test_explain_chi_square_test_contains_main_title():
    result = explain_chi_square_test(TABLE_2X2)

    assert "Test du Khi-deux d'indépendance" in result


def test_explain_chi_square_test_contains_objective():
    result = explain_chi_square_test(TABLE_2X2)

    assert "Objectif du test" in result


def test_explain_chi_square_test_contains_hypotheses():
    result = explain_chi_square_test(TABLE_2X2)

    assert "H₀" in result
    assert "H₁" in result


def test_explain_chi_square_test_contains_observed_table():
    result = explain_chi_square_test(TABLE_2X2)

    assert "[30, 20]" in result
    assert "[20, 30]" in result


def test_explain_chi_square_test_contains_expected_table():
    result = explain_chi_square_test(TABLE_2X2)

    assert "[25, 25]" in result


def test_explain_chi_square_test_contains_statistic():
    result = explain_chi_square_test(TABLE_2X2)

    assert "χ² = 4" in result


def test_explain_chi_square_test_contains_degrees_of_freedom():
    result = explain_chi_square_test(TABLE_2X2)

    assert "ddl = 1" in result


def test_explain_chi_square_test_contains_p_value():
    result = explain_chi_square_test(TABLE_2X2)

    assert "p-value =" in result


def test_explain_chi_square_test_contains_critical_value():
    result = explain_chi_square_test(TABLE_2X2)

    assert "χ² critique" in result


def test_explain_chi_square_test_contains_final_interpretation():
    result = explain_chi_square_test(TABLE_2X2)

    assert "Interprétation finale" in result


def test_explain_chi_square_test_contains_summary():
    result = explain_chi_square_test(TABLE_2X2)

    assert "Résumé numérique" in result


def test_explain_chi_square_test_contains_causality_warning():
    result = explain_chi_square_test(TABLE_2X2)

    assert "causalité" in result


def test_explain_chi_square_test_significant_result():
    result = explain_chi_square_test(TABLE_2X2)

    assert "rejetée" in result


def test_explain_chi_square_test_independence_result():
    result = explain_chi_square_test(
        PERFECT_INDEPENDENCE_TABLE
    )

    assert "non rejetée" in result


def test_explain_chi_square_test_custom_alpha():
    result = explain_chi_square_test(
        TABLE_2X2,
        significance_level=0.01,
    )

    assert "α = 0.01" in result


# ============================================================
# TEST explain_chi_square
# ============================================================


def test_explain_chi_square_is_alias():
    result = explain_chi_square(TABLE_2X2)

    assert "Test du Khi-deux d'indépendance" in result
    assert "Résumé numérique" in result


def test_explain_chi_square_matches_full_explanation():
    result_alias = explain_chi_square(
        TABLE_2X2
    )

    result_full = explain_chi_square_test(
        TABLE_2X2
    )

    assert result_alias == result_full


# ============================================================
# TESTS AVEC RESULTAT PRÉCALCULÉ
# ============================================================


def test_explain_chi_square_test_accepts_precomputed_result():
    result = {
        "observed": [
            [30.0, 20.0],
            [20.0, 30.0],
        ],
        "expected": [
            [25.0, 25.0],
            [25.0, 25.0],
        ],
        "statistic": 4.0,
        "degrees_of_freedom": 1,
        "p_value": 0.04550026389635857,
        "critical_value": 3.841458820694124,
        "significance_level": 0.05,
        "reject_null_hypothesis": True,
    }

    explanation = explain_chi_square_test(
        TABLE_2X2,
        result=result,
    )

    assert "χ² = 4" in explanation
    assert "ddl = 1" in explanation
    assert "p-value" in explanation
    assert "3.841459" in explanation


# ============================================================
# TESTS DE FORMATAGE INDIRECT
# ============================================================


def test_explanations_format_integer_without_decimal():
    result = explain_chi_square_statistic(
        TABLE_2X2,
        4,
    )

    assert "χ² = 4" in result
    assert "4.0" not in result


def test_explanations_format_decimal_without_trailing_zeroes():
    result = explain_chi_square_statistic(
        TABLE_2X2,
        4.5,
    )

    assert "χ² = 4.5" in result


def test_probability_formatting():
    result = explain_chi_square_p_value(
        TABLE_2X2,
        0.025,
    )

    assert "0.025" in result
    assert "2.5 %" in result


# ============================================================
# TESTS DE ROBUSTESSE
# ============================================================


def test_explain_expected_frequencies_with_3x2_table():
    result = explain_expected_frequencies(
        TABLE_3X2
    )

    assert "3 lignes" not in result
    assert "[15, 15]" in result
    assert "[25, 25]" in result
    assert "[20, 20]" in result


def test_explain_chi_square_statistic_perfect_independence():
    result = explain_chi_square_statistic(
        PERFECT_INDEPENDENCE_TABLE
    )

    assert "χ² = 0" in result


def test_explain_chi_square_test_perfect_independence():
    result = explain_chi_square_test(
        PERFECT_INDEPENDENCE_TABLE
    )

    assert "χ² = 0" in result
    assert "non rejetée" in result


# ============================================================
# TESTS DE VALIDATION INDIRECTE
# ============================================================


def test_explain_expected_frequencies_invalid_table():
    with pytest.raises(ValueError):
        explain_expected_frequencies([])


def test_explain_chi_square_statistic_invalid_table():
    with pytest.raises(ValueError):
        explain_chi_square_statistic([])


def test_explain_degrees_of_freedom_invalid_table():
    with pytest.raises(ValueError):
        explain_degrees_of_freedom([])


def test_explain_chi_square_p_value_invalid_table():
    with pytest.raises(ValueError):
        explain_chi_square_p_value([])


def test_explain_chi_square_critical_value_invalid_table():
    with pytest.raises(ValueError):
        explain_chi_square_critical_value([])


def test_explain_chi_square_test_invalid_table():
    with pytest.raises(ValueError):
        explain_chi_square_test([])


# ============================================================
# TESTS DE COHÉRENCE NUMÉRIQUE
# ============================================================


def test_explanation_statistic_matches_core():
    from core.statistics.chi_square import (
        chi_square_statistic,
    )

    statistic = chi_square_statistic(TABLE_2X2)

    explanation = explain_chi_square_statistic(
        TABLE_2X2,
        statistic,
    )

    assert _format_expected_value(
        statistic,
        explanation,
    )


def test_explanation_p_value_matches_core():
    from core.statistics.chi_square import (
        chi_square_p_value,
    )

    p_value = chi_square_p_value(TABLE_2X2)

    explanation = explain_chi_square_p_value(
        TABLE_2X2,
        p_value,
    )

    assert "p-value =" in explanation
    assert str(round(p_value, 6)) in explanation


# ============================================================
# OUTIL DE TEST INTERNE
# ============================================================


def _format_expected_value(
    value: float,
    text: str,
) -> bool:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")

    return formatted in text