import math

import pytest

from core.statistics.distributions import (
    normal_cdf,
    normal_pdf,
    normal_probability_between,
    normal_quantile,
    uniform_cdf,
    uniform_pdf,
    uniform_probability_between,
)


# ============================================================
# LOI NORMALE
# ============================================================


def test_normal_pdf_standard():
    result = normal_pdf(0)

    assert result == pytest.approx(
        1 / math.sqrt(2 * math.pi)
    )


def test_normal_pdf_with_parameters():
    result = normal_pdf(
        10,
        mean_value=10,
        standard_deviation=2,
    )

    assert result == pytest.approx(
        1 / (2 * math.sqrt(2 * math.pi))
    )


def test_normal_cdf_standard_at_zero():
    result = normal_cdf(0)

    assert result == pytest.approx(0.5)


def test_normal_cdf_with_parameters():
    result = normal_cdf(
        12,
        mean_value=10,
        standard_deviation=2,
    )

    assert result == pytest.approx(
        0.841344746,
        rel=1e-6,
    )


def test_normal_quantile_median():
    result = normal_quantile(0.5)

    assert result == pytest.approx(0)


def test_normal_quantile_with_parameters():
    result = normal_quantile(
        0.5,
        mean_value=10,
        standard_deviation=2,
    )

    assert result == pytest.approx(10)


def test_normal_probability_between():
    result = normal_probability_between(
        -1,
        1,
    )

    assert result == pytest.approx(
        0.682689492,
        rel=1e-6,
    )


def test_normal_probability_between_with_parameters():
    result = normal_probability_between(
        8,
        12,
        mean_value=10,
        standard_deviation=2,
    )

    assert result == pytest.approx(
        0.682689492,
        rel=1e-6,
    )


# ============================================================
# LOI UNIFORME
# ============================================================


def test_uniform_pdf_inside_interval():
    result = uniform_pdf(
        0.5,
        lower=0,
        upper=1,
    )

    assert result == pytest.approx(1)


def test_uniform_pdf_outside_interval():
    result = uniform_pdf(
        2,
        lower=0,
        upper=1,
    )

    assert result == pytest.approx(0)


def test_uniform_cdf_below_interval():
    result = uniform_cdf(
        -1,
        lower=0,
        upper=1,
    )

    assert result == pytest.approx(0)


def test_uniform_cdf_inside_interval():
    result = uniform_cdf(
        0.5,
        lower=0,
        upper=1,
    )

    assert result == pytest.approx(0.5)


def test_uniform_cdf_above_interval():
    result = uniform_cdf(
        2,
        lower=0,
        upper=1,
    )

    assert result == pytest.approx(1)


def test_uniform_probability_between():
    result = uniform_probability_between(
        0.2,
        0.7,
        distribution_lower=0,
        distribution_upper=1,
    )

    assert result == pytest.approx(0.5)


# ============================================================
# VALIDATION
# ============================================================


def test_normal_rejects_non_numeric_x():
    with pytest.raises(TypeError):
        normal_pdf("x")


def test_normal_rejects_zero_standard_deviation():
    with pytest.raises(ValueError):
        normal_pdf(
            0,
            standard_deviation=0,
        )


def test_normal_rejects_negative_standard_deviation():
    with pytest.raises(ValueError):
        normal_pdf(
            0,
            standard_deviation=-1,
        )


def test_normal_rejects_invalid_probability():
    with pytest.raises(ValueError):
        normal_quantile(1.5)


def test_normal_rejects_reversed_interval():
    with pytest.raises(ValueError):
        normal_probability_between(
            5,
            2,
        )


def test_uniform_rejects_invalid_interval():
    with pytest.raises(ValueError):
        uniform_pdf(
            0.5,
            lower=1,
            upper=1,
        )


def test_uniform_rejects_reversed_interval():
    with pytest.raises(ValueError):
        uniform_probability_between(
            0.8,
            0.2,
        )


def test_uniform_rejects_invalid_distribution_bounds():
    with pytest.raises(ValueError):
        uniform_pdf(
            0.5,
            lower=2,
            upper=1,
        )