import pytest

from core.statistics.regression_explanations import (
    _format_number,
    _format_series,
    explain_coefficient_of_determination,
    explain_linear_regression,
    explain_prediction,
    explain_regression,
    explain_regression_intercept,
    explain_regression_slope,
    interpret_r_squared,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================

X_LINEAR = [1, 2, 3, 4, 5]
Y_LINEAR = [2, 4, 6, 8, 10]


# ============================================================
# TESTS DES OUTILS DE FORMATAGE
# ============================================================


def test_format_number_integer():
    """Vérifie le formatage d'un entier."""

    assert _format_number(5.0) == "5"


def test_format_number_zero():
    """Vérifie le formatage de zéro."""

    assert _format_number(0.0) == "0"


def test_format_number_decimal():
    """Vérifie le formatage d'un nombre décimal."""

    assert _format_number(2.5) == "2.5"


def test_format_number_removes_trailing_zeros():
    """Vérifie la suppression des zéros inutiles."""

    assert _format_number(2.500000) == "2.5"


def test_format_series():
    """Vérifie le formatage d'une série."""

    result = _format_series(
        [1, 2.5, 3],
    )

    assert result == "[1, 2.5, 3]"


# ============================================================
# TESTS DE L'EXPLICATION DE LA PENTE
# ============================================================


def test_explain_regression_slope_contains_title():
    """L'explication doit contenir son titre."""

    result = explain_regression_slope(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Calcul de la pente" in result


def test_explain_regression_slope_contains_formula():
    """L'explication doit contenir la formule."""

    result = explain_regression_slope(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Σ[(xᵢ − x̄)(yᵢ − ȳ)]" in result
    assert "Σ[(xᵢ − x̄)²]" in result


def test_explain_regression_slope_contains_means():
    """L'explication doit afficher les moyennes."""

    result = explain_regression_slope(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Moyenne de X" in result
    assert "Moyenne de Y" in result


def test_explain_regression_slope_contains_result():
    """Vérifie que la pente calculée apparaît."""

    result = explain_regression_slope(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Pente a = 2" in result


def test_explain_regression_slope_with_result():
    """
    Vérifie que la fonction accepte un résultat
    déjà calculé.
    """

    result = explain_regression_slope(
        X_LINEAR,
        Y_LINEAR,
        result=2.0,
    )

    assert "Pente a = 2" in result


# ============================================================
# TESTS DE L'ORDONNÉE À L'ORIGINE
# ============================================================


def test_explain_regression_intercept_contains_title():
    result = explain_regression_intercept(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "ordonnée à l'origine" in result.lower()


def test_explain_regression_intercept_contains_formula():
    result = explain_regression_intercept(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "b = ȳ − ax̄" in result


def test_explain_regression_intercept_contains_result():
    result = explain_regression_intercept(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Ordonnée à l'origine b = 0" in result


def test_explain_regression_intercept_non_zero():
    """Vérifie une ordonnée à l'origine différente de zéro."""

    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    result = explain_regression_intercept(
        x,
        y,
    )

    assert "Ordonnée à l'origine b = 1" in result


def test_explain_regression_intercept_with_result():
    result = explain_regression_intercept(
        X_LINEAR,
        Y_LINEAR,
        result=0.0,
    )

    assert "Ordonnée à l'origine b = 0" in result


# ============================================================
# TESTS DE LA DROITE DE RÉGRESSION
# ============================================================


def test_explain_linear_regression_contains_title():
    result = explain_linear_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Droite de régression linéaire" in result


def test_explain_linear_regression_contains_model():
    result = explain_linear_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "ŷ = ax + b" in result


def test_explain_linear_regression_contains_equation():
    result = explain_linear_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "ŷ = 2x + 0" in result


def test_explain_linear_regression_non_zero_intercept():
    """Vérifie l'affichage d'une équation avec b positif."""

    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    result = explain_linear_regression(
        x,
        y,
    )

    assert "ŷ = 2x + 1" in result


def test_explain_linear_regression_negative_intercept():
    """Vérifie l'affichage d'une ordonnée négative."""

    x = [1, 2, 3, 4, 5]
    y = [-1, 1, 3, 5, 7]

    result = explain_linear_regression(
        x,
        y,
    )

    assert "ŷ = 2x − 3" in result


def test_explain_linear_regression_with_result():
    """
    Vérifie qu'un résultat pré-calculé peut être fourni.
    """

    result = explain_linear_regression(
        X_LINEAR,
        Y_LINEAR,
        result=(2.0, 0.0),
    )

    assert "ŷ = 2x + 0" in result


# ============================================================
# TESTS DES PRÉDICTIONS
# ============================================================


def test_explain_prediction_contains_title():
    result = explain_prediction(
        X_LINEAR,
        Y_LINEAR,
        [6, 7, 8],
    )

    assert "Prédictions" in result


def test_explain_prediction_contains_formula():
    result = explain_prediction(
        X_LINEAR,
        Y_LINEAR,
        [6],
    )

    assert "ŷ = ax + b" in result


def test_explain_prediction_contains_x_values():
    result = explain_prediction(
        X_LINEAR,
        Y_LINEAR,
        [6, 7, 8],
    )

    assert "[6, 7, 8]" in result


def test_explain_prediction_contains_predictions():
    result = explain_prediction(
        X_LINEAR,
        Y_LINEAR,
        [6, 7, 8],
    )

    assert "ŷ = 12" in result
    assert "ŷ = 14" in result
    assert "ŷ = 16" in result


def test_explain_prediction_with_result():
    """
    Vérifie qu'une liste de prédictions déjà calculée
    peut être fournie.
    """

    result = explain_prediction(
        X_LINEAR,
        Y_LINEAR,
        [6, 7],
        result=[12.0, 14.0],
    )

    assert "ŷ = 12" in result
    assert "ŷ = 14" in result


def test_explain_prediction_non_zero_intercept():
    """Vérifie une prédiction avec b ≠ 0."""

    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    result = explain_prediction(
        x,
        y,
        [6],
    )

    assert "ŷ = 13" in result


# ============================================================
# TESTS DE L'EXPLICATION DE R²
# ============================================================


def test_explain_r_squared_contains_title():
    result = explain_coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Coefficient de détermination R²" in result


def test_explain_r_squared_contains_formula():
    result = explain_coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "R² = 1 − SSE / SST" in result


def test_explain_r_squared_perfect_fit():
    result = explain_coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "R² = 1" in result


def test_explain_r_squared_contains_sse():
    result = explain_coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "SSE" in result


def test_explain_r_squared_contains_sst():
    result = explain_coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "SST" in result


def test_explain_r_squared_contains_percentage():
    result = explain_coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "100 %" in result


def test_explain_r_squared_imperfect_fit():
    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]

    result = explain_coefficient_of_determination(
        x,
        y,
    )

    assert "R² = 0.64" in result
    assert "64 %" in result


def test_explain_r_squared_with_result():
    result = explain_coefficient_of_determination(
        X_LINEAR,
        Y_LINEAR,
        result=1.0,
    )

    assert "R² = 1" in result


# ============================================================
# TESTS DE L'INTERPRÉTATION DE R²
# ============================================================


def test_interpret_r_squared_zero():
    result = interpret_r_squared(0.0)

    assert "R² = 0" in result
    assert "faible" in result


def test_interpret_r_squared_moderate():
    result = interpret_r_squared(0.40)

    assert "R² = 0.4" in result
    assert "modéré" in result


def test_interpret_r_squared_important():
    result = interpret_r_squared(0.70)

    assert "R² = 0.7" in result
    assert "important" in result


def test_interpret_r_squared_high():
    result = interpret_r_squared(0.90)

    assert "R² = 0.9" in result
    assert "très important" in result


def test_interpret_r_squared_negative():
    result = interpret_r_squared(-0.10)

    assert "R² est négatif" in result


# ============================================================
# TESTS DE L'EXPLICATION COMPLÈTE
# ============================================================


def test_explain_regression_contains_title():
    result = explain_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Analyse complète de la régression linéaire" in result


def test_explain_regression_contains_x():
    result = explain_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "X :" in result
    assert "[1, 2, 3, 4, 5]" in result


def test_explain_regression_contains_y():
    result = explain_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Y :" in result
    assert "[2, 4, 6, 8, 10]" in result


def test_explain_regression_contains_equation():
    result = explain_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "ŷ = 2x + 0" in result


def test_explain_regression_contains_r_squared():
    result = explain_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "R² = 1" in result


def test_explain_regression_contains_interpretation():
    result = explain_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "Interprétation" in result


def test_explain_regression_contains_causality_warning():
    """
    Vérifie la présence de la mise en garde concernant
    la causalité.
    """

    result = explain_regression(
        X_LINEAR,
        Y_LINEAR,
    )

    assert "ne prouve pas" in result
    assert "cause" in result


# ============================================================
# TESTS DE CAS RÉELS NON PARFAITS
# ============================================================


def test_explain_regression_imperfect_model():
    """
    Vérifie que l'explication complète fonctionne
    avec une relation non parfaitement linéaire.
    """

    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]

    result = explain_regression(
        x,
        y,
    )

    assert "Analyse complète" in result
    assert "Droite de régression" in result
    assert "R² = 0.64" in result


# ============================================================
# TESTS DE VALIDATION
# ============================================================


def test_explain_regression_slope_empty_data():
    with pytest.raises(ValueError):
        explain_regression_slope(
            [],
            [],
        )


def test_explain_regression_slope_different_lengths():
    with pytest.raises(ValueError):
        explain_regression_slope(
            [1, 2, 3],
            [2, 4],
        )


def test_explain_linear_regression_constant_x():
    with pytest.raises(ValueError):
        explain_linear_regression(
            [2, 2, 2],
            [1, 3, 5],
        )


def test_explain_prediction_empty_prediction_values():
    with pytest.raises(ValueError):
        explain_prediction(
            X_LINEAR,
            Y_LINEAR,
            [],
        )


def test_explain_prediction_invalid_prediction_value():
    with pytest.raises(TypeError):
        explain_prediction(
            X_LINEAR,
            Y_LINEAR,
            [6, "7"],
        )


def test_explain_r_squared_constant_x():
    with pytest.raises(ValueError):
        explain_coefficient_of_determination(
            [2, 2, 2],
            [1, 3, 5],
        )