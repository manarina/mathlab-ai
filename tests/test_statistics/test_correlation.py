import math

import pytest

from core.statistics.correlation import (
    covariance,
    pearson_correlation,
)


# ============================================================
# COVARIANCE
# ============================================================


def test_covariance_positive_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    assert covariance(x, y) == pytest.approx(4.0)


def test_covariance_negative_relationship():
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]

    assert covariance(x, y) == pytest.approx(-4.0)


def test_covariance_zero_relationship():
    x = [1, 2, 3]
    y = [2, 1, 2]

    assert covariance(x, y) == pytest.approx(0.0)


def test_covariance_with_decimal_values():
    x = [1.5, 2.5, 3.5]
    y = [2.0, 4.0, 6.0]

    assert covariance(x, y) == pytest.approx(1.3333333333)


# ============================================================
# PEARSON
# ============================================================


def test_pearson_perfect_positive_correlation():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    assert pearson_correlation(x, y) == pytest.approx(1.0)


def test_pearson_perfect_negative_correlation():
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]

    assert pearson_correlation(x, y) == pytest.approx(-1.0)


def test_pearson_uncorrelated_example():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 5, 4, 5]

    result = pearson_correlation(x, y)

    assert result == pytest.approx(0.0)


def test_pearson_result_is_between_minus_one_and_one():
    x = [2, 4, 6, 8, 10]
    y = [1, 3, 2, 5, 4]

    result = pearson_correlation(x, y)

    assert -1.0 <= result <= 1.0


# ============================================================
# VALIDATION
# ============================================================


def test_covariance_empty_x():
    with pytest.raises(ValueError):
        covariance([], [1, 2, 3])


def test_covariance_empty_y():
    with pytest.raises(ValueError):
        covariance([1, 2, 3], [])


def test_covariance_different_lengths():
    with pytest.raises(ValueError):
        covariance([1, 2, 3], [1, 2])


def test_pearson_different_lengths():
    with pytest.raises(ValueError):
        pearson_correlation([1, 2, 3], [1, 2])


def test_covariance_rejects_non_numeric_values():
    with pytest.raises(TypeError):
        covariance([1, 2, "a"], [1, 2, 3])


def test_pearson_rejects_boolean_values():
    with pytest.raises(TypeError):
        pearson_correlation([1, True, 3], [1, 2, 3])


def test_covariance_rejects_nan():
    with pytest.raises(ValueError):
        covariance([1, math.nan, 3], [1, 2, 3])


def test_pearson_rejects_infinity():
    with pytest.raises(ValueError):
        pearson_correlation(
            [1, math.inf, 3],
            [1, 2, 3],
        )


# ============================================================
# SERIES CONSTANTES
# ============================================================


def test_pearson_rejects_constant_x():
    with pytest.raises(ValueError):
        pearson_correlation(
            [5, 5, 5],
            [1, 2, 3],
        )


def test_pearson_rejects_constant_y():
    with pytest.raises(ValueError):
        pearson_correlation(
            [1, 2, 3],
            [5, 5, 5],
        )