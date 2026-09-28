
"""
Tests des méthodes d'interpolation numérique.

Fonctions testées :
- lagrange_interpolation
- divided_differences
- newton_interpolation
- evaluate_interpolating_polynomial
"""

import pytest
import sympy as sp

from core.numerical.interpolation import (
    x,
    divided_differences,
    evaluate_interpolating_polynomial,
    lagrange_interpolation,
    newton_interpolation,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================


X_VALUES = [0, 1, 2]
Y_VALUES = [1, 3, 7]

EXPECTED_POLYNOMIAL = x**2 + x + 1


# ============================================================
# INTERPOLATION DE LAGRANGE
# ============================================================


def test_lagrange_interpolation_returns_expected_polynomial():
    """Vérifie que Lagrange construit le bon polynôme."""

    polynomial = lagrange_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    assert sp.expand(polynomial - EXPECTED_POLYNOMIAL) == 0


def test_lagrange_interpolation_passes_through_all_points():
    """Le polynôme doit passer exactement par tous les points."""

    polynomial = lagrange_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    for xi, yi in zip(X_VALUES, Y_VALUES):
        result = polynomial.subs(x, xi)

        assert sp.simplify(result - yi) == 0


def test_lagrange_interpolation_exact_polynomial():
    """Vérifie l'interpolation exacte d'un polynôme connu."""

    x_values = [-1, 0, 1, 2]
    y_values = [
        1**2 + 1,
        0**2 + 1,
        1**2 + 1,
        2**2 + 1,
    ]

    polynomial = lagrange_interpolation(
        x_values,
        y_values,
    )

    expected = x**2 + 1

    assert sp.expand(polynomial - expected) == 0


# ============================================================
# DIFFÉRENCES DIVISÉES
# ============================================================


def test_divided_differences_returns_expected_coefficients():
    """
    Vérifie les coefficients des différences divisées.

    Pour :

        f(x) = x² + x + 1

    avec les points 0, 1 et 2 :

        c0 = 1
        c1 = 2
        c2 = 1
    """

    coefficients = divided_differences(
        X_VALUES,
        Y_VALUES,
    )

    expected = [
        sp.Integer(1),
        sp.Integer(2),
        sp.Integer(1),
    ]

    assert coefficients == expected


def test_divided_differences_linear_function():
    """Vérifie les différences divisées d'une fonction linéaire."""

    x_values = [0, 2, 4]
    y_values = [3, 7, 11]

    coefficients = divided_differences(
        x_values,
        y_values,
    )

    assert coefficients[0] == 3
    assert coefficients[1] == 2
    assert coefficients[2] == 0


# ============================================================
# INTERPOLATION DE NEWTON
# ============================================================


def test_newton_interpolation_returns_expected_polynomial():
    """Vérifie que Newton construit le même polynôme que Lagrange."""

    polynomial = newton_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    assert sp.expand(polynomial - EXPECTED_POLYNOMIAL) == 0


def test_newton_and_lagrange_are_equivalent():
    """
    Les deux méthodes doivent produire mathématiquement
    le même polynôme.
    """

    lagrange_polynomial = lagrange_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    newton_polynomial = newton_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    assert sp.expand(
        lagrange_polynomial - newton_polynomial
    ) == 0


def test_newton_interpolation_passes_through_all_points():
    """Le polynôme de Newton doit passer par tous les points."""

    polynomial = newton_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    for xi, yi in zip(X_VALUES, Y_VALUES):
        result = polynomial.subs(x, xi)

        assert sp.simplify(result - yi) == 0


# ============================================================
# ÉVALUATION DU POLYNÔME
# ============================================================


def test_evaluate_interpolating_polynomial_exact_value():
    """Vérifie l'évaluation exacte en x = 1.5."""

    polynomial = lagrange_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    result = evaluate_interpolating_polynomial(
        polynomial,
        1.5,
    )

    assert sp.simplify(
        result - sp.Rational(19, 4)
    ) == 0


def test_evaluate_interpolating_polynomial_at_interpolation_points():
    """
    Vérifie que l'évaluation retourne les ordonnées originales
    aux points d'interpolation.
    """

    polynomial = newton_interpolation(
        X_VALUES,
        Y_VALUES,
    )

    for xi, yi in zip(X_VALUES, Y_VALUES):
        result = evaluate_interpolating_polynomial(
            polynomial,
            xi,
        )

        assert sp.simplify(result - yi) == 0


def test_evaluate_interpolating_polynomial_integer_value():
    """Vérifie l'évaluation sur une valeur entière."""

    polynomial = x**2 + x + 1

    result = evaluate_interpolating_polynomial(
        polynomial,
        3,
    )

    assert result == 13


# ============================================================
# CAS D'ERREUR — DONNÉES D'INTERPOLATION
# ============================================================


def test_interpolation_rejects_different_lengths():
    """x et y doivent avoir la même longueur."""

    with pytest.raises(ValueError, match="même longueur"):
        lagrange_interpolation(
            [0, 1, 2],
            [1, 3],
        )


def test_newton_rejects_different_lengths():
    """Newton doit également vérifier les dimensions."""

    with pytest.raises(ValueError, match="même longueur"):
        newton_interpolation(
            [0, 1, 2],
            [1, 3],
        )


def test_divided_differences_rejects_different_lengths():
    """Les différences divisées doivent vérifier les dimensions."""

    with pytest.raises(ValueError, match="même longueur"):
        divided_differences(
            [0, 1],
            [1],
        )


def test_interpolation_requires_at_least_two_points():
    """Une interpolation nécessite au moins deux points."""

    with pytest.raises(ValueError, match="au moins deux points"):
        lagrange_interpolation(
            [0],
            [1],
        )


def test_newton_requires_at_least_two_points():
    """Newton nécessite également au moins deux points."""

    with pytest.raises(ValueError, match="au moins deux points"):
        newton_interpolation(
            [0],
            [1],
        )


def test_divided_differences_requires_at_least_two_points():
    """Les différences divisées nécessitent au moins deux points."""

    with pytest.raises(ValueError, match="au moins deux points"):
        divided_differences(
            [0],
            [1],
        )


def test_interpolation_rejects_duplicate_x_values():
    """
    Deux abscisses identiques sont interdites car elles provoquent
    une division par zéro dans les formules d'interpolation.
    """

    with pytest.raises(ValueError, match="distinctes"):
        lagrange_interpolation(
            [0, 1, 1],
            [1, 3, 4],
        )


def test_newton_rejects_duplicate_x_values():
    """Newton doit également rejeter les abscisses identiques."""

    with pytest.raises(ValueError, match="distinctes"):
        newton_interpolation(
            [0, 1, 1],
            [1, 3, 4],
        )


def test_divided_differences_rejects_duplicate_x_values():
    """Les différences divisées doivent rejeter les doublons."""

    with pytest.raises(ValueError, match="distinctes"):
        divided_differences(
            [0, 1, 1],
            [1, 3, 4],
        )


# ============================================================
# CAS D'ERREUR — TYPES INVALIDES
# ============================================================


def test_interpolation_rejects_non_numeric_x():
    """Les abscisses doivent être numériques."""

    with pytest.raises(TypeError, match="abscisses"):
        lagrange_interpolation(
            [0, "a", 2],
            [1, 3, 7],
        )


def test_interpolation_rejects_non_numeric_y():
    """Les ordonnées doivent être numériques."""

    with pytest.raises(TypeError, match="ordonnées"):
        lagrange_interpolation(
            [0, 1, 2],
            [1, "a", 7],
        )


def test_newton_rejects_non_numeric_x():
    """Newton doit rejeter les abscisses non numériques."""

    with pytest.raises(TypeError, match="abscisses"):
        newton_interpolation(
            [0, "a", 2],
            [1, 3, 7],
        )


def test_newton_rejects_non_numeric_y():
    """Newton doit rejeter les ordonnées non numériques."""

    with pytest.raises(TypeError, match="ordonnées"):
        newton_interpolation(
            [0, 1, 2],
            [1, "a", 7],
        )


def test_interpolation_rejects_string_sequences():
    """Une chaîne ne doit pas être considérée comme une séquence de points."""

    with pytest.raises(TypeError):
        lagrange_interpolation(
            "012",
            [1, 3, 7],
        )


def test_interpolation_rejects_boolean_values():
    """Les booléens ne doivent pas être considérés comme des nombres."""

    with pytest.raises(TypeError):
        lagrange_interpolation(
            [0, True, 2],
            [1, 3, 7],
        )


# ============================================================
# CAS D'ERREUR — ÉVALUATION
# ============================================================


def test_evaluation_rejects_non_sympy_polynomial():
    """Le polynôme doit être une expression SymPy."""

    with pytest.raises(TypeError, match="expression SymPy"):
        evaluate_interpolating_polynomial(
            "x**2 + 1",
            2,
        )


def test_evaluation_rejects_non_numeric_value():
    """La valeur d'évaluation doit être numérique."""

    polynomial = x**2 + 1

    with pytest.raises(TypeError, match="numérique"):
        evaluate_interpolating_polynomial(
            polynomial,
            "abc",
        )


def test_evaluation_rejects_boolean_value():
    """Un booléen ne doit pas être accepté comme valeur de x."""

    polynomial = x**2 + 1

    with pytest.raises(TypeError, match="numérique"):
        evaluate_interpolating_polynomial(
            polynomial,
            True,
        )

