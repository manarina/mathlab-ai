import pytest

from core.statistics.correlation_explanations import (
    explain_covariance,
    explain_pearson_correlation,
)


# ============================================================
# COVARIANCE
# ============================================================


def test_explain_covariance_contains_title():
    result = explain_covariance(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "Covariance" in result


def test_explain_covariance_contains_formula():
    result = explain_covariance(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "Cov" in result
    assert "n" in result


def test_explain_covariance_contains_series():
    result = explain_covariance(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "1, 2, 3" in result
    assert "2, 4, 6" in result


def test_explain_covariance_contains_means():
    result = explain_covariance(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "1" in result
    assert "4" in result


def test_explain_covariance_contains_result():
    result = explain_covariance(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "1.33333" in result


def test_explain_covariance_accepts_explicit_result():
    result = explain_covariance(
        [1, 2, 3],
        [2, 4, 6],
        result=42.0,
    )

    assert "42" in result


def test_explain_covariance_raises_for_invalid_data():
    with pytest.raises(ValueError):
        explain_covariance(
            [],
            [],
        )


# ============================================================
# PEARSON
# ============================================================


def test_explain_pearson_contains_title():
    result = explain_pearson_correlation(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "Pearson" in result


def test_explain_pearson_contains_formula():
    result = explain_pearson_correlation(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "r" in result
    assert "x_i" in result
    assert "y_i" in result


def test_explain_pearson_contains_series():
    result = explain_pearson_correlation(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "1, 2, 3" in result
    assert "2, 4, 6" in result


def test_explain_pearson_contains_result():
    result = explain_pearson_correlation(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "1" in result


def test_explain_pearson_accepts_explicit_result():
    result = explain_pearson_correlation(
        [1, 2, 3],
        [2, 4, 6],
        result=-0.75,
    )

    assert "-0.75" in result


def test_explain_pearson_contains_interpretation():
    result = explain_pearson_correlation(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "relation linéaire" in result


def test_explain_pearson_mentions_causality():
    result = explain_pearson_correlation(
        [1, 2, 3],
        [2, 4, 6],
    )

    assert "causalité" in result


def test_explain_pearson_raises_for_constant_series():
    with pytest.raises(ValueError):
        explain_pearson_correlation(
            [5, 5, 5],
            [1, 2, 3],
        )