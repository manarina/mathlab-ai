from __future__ import annotations

import math

import pytest

from core.statistics.chi_square import (
    chi_square_critical_value,
    chi_square_p_value,
    chi_square_statistic,
    chi_square_test,
    chi_square_test_scipy,
    degrees_of_freedom,
    expected_frequencies,
    reject_null_hypothesis,
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
# TESTS DES EFFECTIFS ATTENDUS
# ============================================================


def test_expected_frequencies_2x2():
    result = expected_frequencies(TABLE_2X2)

    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 2

    assert math.isclose(
        result[0][0],
        25.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[0][1],
        25.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[1][0],
        25.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[1][1],
        25.0,
        rel_tol=1e-9,
    )


def test_expected_frequencies_3x2():
    result = expected_frequencies(TABLE_3X2)

    assert len(result) == 3
    assert len(result[0]) == 2

    assert math.isclose(
        result[0][0],
        15.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[0][1],
        15.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[1][0],
        25.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[1][1],
        25.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[2][0],
        20.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        result[2][1],
        20.0,
        rel_tol=1e-9,
    )

def test_expected_frequencies_preserves_shape():
    table = [
        [5, 15, 10],
        [10, 20, 5],
    ]

    result = expected_frequencies(table)

    assert len(result) == 2
    assert len(result[0]) == 3
    assert len(result[1]) == 3


# ============================================================
# TESTS DE LA STATISTIQUE DU KHI-DEUX
# ============================================================


def test_chi_square_statistic_2x2():
    result = chi_square_statistic(TABLE_2X2)

    assert math.isclose(
        result,
        4.0,
        rel_tol=1e-9,
    )


def test_chi_square_statistic_perfect_independence():
    result = chi_square_statistic(
        PERFECT_INDEPENDENCE_TABLE
    )

    assert math.isclose(
        result,
        0.0,
        abs_tol=1e-12,
    )


def test_chi_square_statistic_is_non_negative():
    result = chi_square_statistic(TABLE_3X2)

    assert result >= 0


# ============================================================
# TESTS DES DEGRÉS DE LIBERTÉ
# ============================================================


def test_degrees_of_freedom_2x2():
    result = degrees_of_freedom(TABLE_2X2)

    assert result == 1


def test_degrees_of_freedom_3x2():
    result = degrees_of_freedom(TABLE_3X2)

    assert result == 2


def test_degrees_of_freedom_3x3():
    table = [
        [10, 20, 30],
        [20, 30, 40],
        [30, 40, 50],
    ]

    result = degrees_of_freedom(table)

    assert result == 4


# ============================================================
# TESTS DE LA P-VALUE
# ============================================================


def test_chi_square_p_value_is_between_zero_and_one():
    result = chi_square_p_value(TABLE_2X2)

    assert 0 <= result <= 1


def test_chi_square_p_value_for_perfect_independence():
    result = chi_square_p_value(
        PERFECT_INDEPENDENCE_TABLE
    )

    assert math.isclose(
        result,
        1.0,
        rel_tol=1e-9,
    )


def test_chi_square_p_value_for_significant_result():
    result = chi_square_p_value(TABLE_2X2)

    assert result < 0.05


# ============================================================
# TESTS DE LA VALEUR CRITIQUE
# ============================================================


def test_chi_square_critical_value_default_alpha():
    result = chi_square_critical_value(
        TABLE_2X2
    )

    # χ² critique à 5 % avec 1 ddl ≈ 3.841459
    assert math.isclose(
        result,
        3.841458820694124,
        rel_tol=1e-9,
    )


def test_chi_square_critical_value_alpha_001():
    result = chi_square_critical_value(
        TABLE_2X2,
        significance_level=0.01,
    )

    # χ² critique à 1 % avec 1 ddl ≈ 6.634897
    assert math.isclose(
        result,
        6.6348966010212145,
        rel_tol=1e-9,
    )


def test_chi_square_critical_value_decreases_when_alpha_increases():
    critical_01 = chi_square_critical_value(
        TABLE_2X2,
        significance_level=0.01,
    )

    critical_05 = chi_square_critical_value(
        TABLE_2X2,
        significance_level=0.05,
    )

    critical_10 = chi_square_critical_value(
        TABLE_2X2,
        significance_level=0.10,
    )

    assert critical_01 > critical_05
    assert critical_05 > critical_10


# ============================================================
# TESTS DU TEST COMPLET
# ============================================================


def test_chi_square_test_returns_expected_keys():
    result = chi_square_test(TABLE_2X2)

    expected_keys = {
        "observed",
        "expected",
        "statistic",
        "degrees_of_freedom",
        "p_value",
        "critical_value",
        "significance_level",
        "reject_null_hypothesis",
    }

    assert set(result.keys()) == expected_keys


def test_chi_square_test_observed_data():
    result = chi_square_test(TABLE_2X2)

    assert result["observed"] == [
        [30.0, 20.0],
        [20.0, 30.0],
    ]


def test_chi_square_test_expected_data():
    result = chi_square_test(TABLE_2X2)

    expected = result["expected"]

    assert math.isclose(
        expected[0][0],
        25.0,
        rel_tol=1e-9,
    )

    assert math.isclose(
        expected[0][1],
        25.0,
        rel_tol=1e-9,
    )


def test_chi_square_test_statistic():
    result = chi_square_test(TABLE_2X2)

    assert math.isclose(
        result["statistic"],
        4.0,
        rel_tol=1e-9,
    )


def test_chi_square_test_degrees_of_freedom():
    result = chi_square_test(TABLE_2X2)

    assert result["degrees_of_freedom"] == 1


def test_chi_square_test_p_value():
    result = chi_square_test(TABLE_2X2)

    assert math.isclose(
        result["p_value"],
        chi_square_p_value(TABLE_2X2),
        rel_tol=1e-9,
    )


def test_chi_square_test_critical_value():
    result = chi_square_test(TABLE_2X2)

    assert math.isclose(
        result["critical_value"],
        chi_square_critical_value(TABLE_2X2),
        rel_tol=1e-9,
    )


def test_chi_square_test_significance_level():
    result = chi_square_test(
        TABLE_2X2,
        significance_level=0.01,
    )

    assert result["significance_level"] == 0.01


def test_chi_square_test_rejects_null_hypothesis():
    result = chi_square_test(TABLE_2X2)

    assert result["reject_null_hypothesis"] is True


def test_chi_square_test_does_not_reject_null_hypothesis():
    result = chi_square_test(
        PERFECT_INDEPENDENCE_TABLE
    )

    assert result["reject_null_hypothesis"] is False


# ============================================================
# COMPARAISON AVEC SCIPY
# ============================================================


def test_chi_square_test_matches_scipy_statistic():
    result = chi_square_test(TABLE_3X2)
    scipy_result = chi_square_test_scipy(TABLE_3X2)

    assert math.isclose(
        result["statistic"],
        scipy_result["statistic"],
        rel_tol=1e-9,
    )


def test_chi_square_test_matches_scipy_p_value():
    result = chi_square_test(TABLE_3X2)
    scipy_result = chi_square_test_scipy(TABLE_3X2)

    assert math.isclose(
        result["p_value"],
        scipy_result["p_value"],
        rel_tol=1e-9,
    )


def test_chi_square_test_matches_scipy_degrees_of_freedom():
    result = chi_square_test(TABLE_3X2)
    scipy_result = chi_square_test_scipy(TABLE_3X2)

    assert (
        result["degrees_of_freedom"]
        == scipy_result["degrees_of_freedom"]
    )


def test_chi_square_test_matches_scipy_expected_frequencies():
    result = chi_square_test(TABLE_3X2)
    scipy_result = chi_square_test_scipy(TABLE_3X2)

    for result_row, scipy_row in zip(
        result["expected"],
        scipy_result["expected"],
    ):
        for result_value, scipy_value in zip(
            result_row,
            scipy_row,
        ):
            assert math.isclose(
                result_value,
                scipy_value,
                rel_tol=1e-9,
            )


# ============================================================
# TESTS DE reject_null_hypothesis
# ============================================================


def test_reject_null_hypothesis_when_p_value_is_below_alpha():
    assert (
        reject_null_hypothesis(
            0.01,
            0.05,
        )
        is True
    )


def test_reject_null_hypothesis_when_p_value_equals_alpha():
    assert (
        reject_null_hypothesis(
            0.05,
            0.05,
        )
        is False
    )


def test_reject_null_hypothesis_when_p_value_is_above_alpha():
    assert (
        reject_null_hypothesis(
            0.10,
            0.05,
        )
        is False
    )


def test_reject_null_hypothesis_with_alpha_001():
    assert (
        reject_null_hypothesis(
            0.005,
            0.01,
        )
        is True
    )

    assert (
        reject_null_hypothesis(
            0.02,
            0.01,
        )
        is False
    )


# ============================================================
# TESTS DE VALIDATION DU TABLEAU
# ============================================================


def test_empty_table_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies([])


def test_one_row_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies([[10, 20]])


def test_one_column_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies(
            [
                [10],
                [20],
            ]
        )


def test_non_rectangular_table_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies(
            [
                [10, 20],
                [30],
            ]
        )


def test_negative_frequency_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies(
            [
                [10, -5],
                [20, 30],
            ]
        )


def test_string_frequency_raises_type_error():
    with pytest.raises(TypeError):
        expected_frequencies(
            [
                [10, "20"],
                [30, 40],
            ]
        )


def test_boolean_frequency_raises_type_error():
    with pytest.raises(TypeError):
        expected_frequencies(
            [
                [True, 20],
                [30, 40],
            ]
        )


def test_infinite_frequency_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies(
            [
                [10, math.inf],
                [30, 40],
            ]
        )


def test_nan_frequency_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies(
            [
                [10, math.nan],
                [30, 40],
            ]
        )


def test_all_zero_table_raises_value_error():
    with pytest.raises(ValueError):
        expected_frequencies(
            [
                [0, 0],
                [0, 0],
            ]
        )


# ============================================================
# TESTS DU NIVEAU DE SIGNIFICATION
# ============================================================


def test_invalid_significance_level_zero():
    with pytest.raises(ValueError):
        chi_square_critical_value(
            TABLE_2X2,
            significance_level=0,
        )


def test_invalid_significance_level_one():
    with pytest.raises(ValueError):
        chi_square_critical_value(
            TABLE_2X2,
            significance_level=1,
        )


def test_negative_significance_level():
    with pytest.raises(ValueError):
        chi_square_critical_value(
            TABLE_2X2,
            significance_level=-0.05,
        )


def test_significance_level_greater_than_one():
    with pytest.raises(ValueError):
        chi_square_critical_value(
            TABLE_2X2,
            significance_level=1.5,
        )


def test_boolean_significance_level_raises_type_error():
    with pytest.raises(TypeError):
        chi_square_critical_value(
            TABLE_2X2,
            significance_level=True,
        )


def test_string_significance_level_raises_type_error():
    with pytest.raises(TypeError):
        chi_square_critical_value(
            TABLE_2X2,
            significance_level="0.05",
        )


# ============================================================
# TESTS DES P-VALUES
# ============================================================


def test_invalid_p_value_below_zero():
    with pytest.raises(ValueError):
        reject_null_hypothesis(-0.01)


def test_invalid_p_value_above_one():
    with pytest.raises(ValueError):
        reject_null_hypothesis(1.01)


def test_invalid_p_value_nan():
    with pytest.raises(ValueError):
        reject_null_hypothesis(math.nan)


def test_invalid_p_value_infinity():
    with pytest.raises(ValueError):
        reject_null_hypothesis(math.inf)


def test_boolean_p_value_raises_type_error():
    with pytest.raises(TypeError):
        reject_null_hypothesis(True)


def test_string_p_value_raises_type_error():
    with pytest.raises(TypeError):
        reject_null_hypothesis("0.05")