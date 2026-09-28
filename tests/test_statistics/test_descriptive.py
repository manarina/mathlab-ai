from math import isclose

import pytest

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


# ============================================================
# MOYENNE
# ============================================================


def test_mean():
    assert mean([10, 20, 30]) == 20.0


def test_mean_with_decimal_values():
    assert isclose(
        mean([1.5, 2.5, 3.5]),
        2.5,
    )


def test_mean_with_negative_values():
    assert mean([-10, 0, 10]) == 0.0


# ============================================================
# MEDIANE
# ============================================================


def test_median_with_odd_number_of_values():
    assert median([5, 1, 3]) == 3.0


def test_median_with_even_number_of_values():
    assert median([1, 2, 3, 4]) == 2.5


def test_median_unsorted_data():
    assert median([10, 2, 7, 4, 5]) == 5.0


# ============================================================
# MODE
# ============================================================


def test_mode_with_unique_mode():
    assert mode([1, 2, 2, 3, 4]) == 2


def test_mode_with_multiple_modes():
    assert mode([1, 1, 2, 2, 3]) == (1, 2)


def test_mode_with_all_values_different():
    assert mode([1, 2, 3, 4]) == (1, 2, 3, 4)


# ============================================================
# MINIMUM / MAXIMUM
# ============================================================


def test_minimum():
    assert minimum([8, 3, 12, 5, 1]) == 1


def test_maximum():
    assert maximum([8, 3, 12, 5, 1]) == 12


def test_minimum_with_negative_values():
    assert minimum([-5, -2, -10, 3]) == -10


def test_maximum_with_negative_values():
    assert maximum([-5, -2, -10, 3]) == 3


# ============================================================
# ETENDUE
# ============================================================


def test_data_range():
    assert data_range([2, 5, 10]) == 8


def test_data_range_with_negative_values():
    assert data_range([-5, 0, 5]) == 10


# ============================================================
# VARIANCE
# ============================================================


def test_variance():
    assert isclose(
        variance([1, 2, 3, 4, 5]),
        2.0,
    )


def test_variance_with_identical_values():
    assert variance([7, 7, 7, 7]) == 0.0


def test_variance_with_negative_values():
    assert isclose(
        variance([-2, -1, 0, 1, 2]),
        2.0,
    )


# ============================================================
# ECART-TYPE
# ============================================================


def test_standard_deviation():
    assert isclose(
        standard_deviation([1, 2, 3, 4, 5]),
        2 ** 0.5,
    )


def test_standard_deviation_with_identical_values():
    assert standard_deviation([5, 5, 5]) == 0.0


# ============================================================
# PERCENTILES
# ============================================================


def test_percentile_0():
    assert percentile([1, 2, 3, 4, 5], 0) == 1.0


def test_percentile_50():
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_percentile_100():
    assert percentile([1, 2, 3, 4, 5], 100) == 5.0


def test_percentile_25():
    assert percentile([1, 2, 3, 4, 5], 25) == 2.0


def test_percentile_with_interpolation():
    assert percentile([10, 20, 30, 40], 25) == 17.5


# ============================================================
# QUARTILES
# ============================================================


def test_quartiles():
    assert quartiles([1, 2, 3, 4, 5]) == (
        2.0,
        3.0,
        4.0,
    )


def test_quartiles_with_even_number_of_values():
    assert quartiles([1, 2, 3, 4]) == (
        1.75,
        2.5,
        3.25,
    )


# ============================================================
# VALIDATION DES DONNEES
# ============================================================


def test_empty_data_raises_value_error():
    with pytest.raises(ValueError):
        mean([])


def test_empty_data_for_median_raises_value_error():
    with pytest.raises(ValueError):
        median([])


def test_empty_data_for_variance_raises_value_error():
    with pytest.raises(ValueError):
        variance([])


def test_non_numeric_data_raises_type_error():
    with pytest.raises(TypeError):
        mean([1, 2, "3"])


def test_non_numeric_data_for_median_raises_type_error():
    with pytest.raises(TypeError):
        median([1, "2", 3])


def test_boolean_data_raises_type_error():
    with pytest.raises(TypeError):
        mean([1, True, 3])


# ============================================================
# VALIDATION DES PERCENTILES
# ============================================================


def test_percentile_below_zero_raises_value_error():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], -1)


def test_percentile_above_100_raises_value_error():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


def test_percentile_single_value():
    assert percentile([42], 25) == 42.0