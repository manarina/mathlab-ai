import pytest

from core.statistics.distribution_explanations import (
    explain_normal_cdf,
    explain_normal_pdf,
    explain_normal_probability_between,
    explain_normal_quantile,
    explain_uniform_cdf,
    explain_uniform_pdf,
    explain_uniform_probability_between,
)


# ============================================================
# LOI NORMALE
# ============================================================


def test_explain_normal_pdf():
    result = explain_normal_pdf(
        0,
        mean_value=0,
        standard_deviation=1,
    )

    assert "Densité de la loi normale" in result
    assert "Résultat" in result
    assert "f(0)" in result


def test_explain_normal_pdf_with_result():
    result = explain_normal_pdf(
        0,
        result=0.39894228,
    )

    assert "0.398942" in result


def test_explain_normal_pdf_rejects_invalid_standard_deviation():
    with pytest.raises(ValueError):
        explain_normal_pdf(
            0,
            standard_deviation=0,
        )


def test_explain_normal_cdf():
    result = explain_normal_cdf(
        0,
    )

    assert "Fonction de répartition" in result
    assert "P(X" in result
    assert "0.500000" in result


def test_explain_normal_cdf_with_parameters():
    result = explain_normal_cdf(
        12,
        mean_value=10,
        standard_deviation=2,
    )

    assert "12" in result
    assert "10" in result
    assert "2" in result


def test_explain_normal_quantile():
    result = explain_normal_quantile(
        0.5,
    )

    assert "Quantile de la loi normale" in result
    assert "0.5" in result
    assert "Résultat" in result


def test_explain_normal_quantile_rejects_invalid_probability():
    with pytest.raises(ValueError):
        explain_normal_quantile(1.5)


def test_explain_normal_probability_between():
    result = explain_normal_probability_between(
        -1,
        1,
    )

    assert "Probabilité entre deux valeurs" in result
    assert "-1" in result
    assert "1" in result
    assert "68.27 %" in result


def test_explain_normal_probability_between_rejects_reversed_interval():
    with pytest.raises(ValueError):
        explain_normal_probability_between(
            5,
            2,
        )


# ============================================================
# LOI UNIFORME
# ============================================================


def test_explain_uniform_pdf():
    result = explain_uniform_pdf(
        0.5,
        lower=0,
        upper=1,
    )

    assert "Densité de la loi uniforme" in result
    assert "0.5" in result
    assert "1" in result


def test_explain_uniform_pdf_outside_interval():
    result = explain_uniform_pdf(
        2,
        lower=0,
        upper=1,
    )

    assert "n'appartient pas" in result
    assert "0" in result


def test_explain_uniform_pdf_rejects_invalid_interval():
    with pytest.raises(ValueError):
        explain_uniform_pdf(
            0.5,
            lower=1,
            upper=1,
        )


def test_explain_uniform_cdf():
    result = explain_uniform_cdf(
        0.5,
        lower=0,
        upper=1,
    )

    assert "Fonction de répartition de la loi uniforme" in result
    assert "0.500000" in result


def test_explain_uniform_cdf_with_parameters():
    result = explain_uniform_cdf(
        5,
        lower=0,
        upper=10,
    )

    assert "5" in result
    assert "10" in result
    assert "0.500000" in result


def test_explain_uniform_probability_between():
    result = explain_uniform_probability_between(
        0.2,
        0.7,
        distribution_lower=0,
        distribution_upper=1,
    )

    assert "Probabilité entre deux valeurs" in result
    assert "0.500000" in result


def test_explain_uniform_probability_between_rejects_reversed_interval():
    with pytest.raises(ValueError):
        explain_uniform_probability_between(
            0.8,
            0.2,
        )


def test_explain_uniform_probability_between_rejects_invalid_distribution():
    with pytest.raises(ValueError):
        explain_uniform_probability_between(
            0.2,
            0.7,
            distribution_lower=1,
            distribution_upper=0,
        )