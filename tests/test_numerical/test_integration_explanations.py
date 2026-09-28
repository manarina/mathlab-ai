
"""
Tests unitaires des explications pédagogiques
des méthodes d'intégration numérique.

Le module testé est :
    core.numerical.integration_explanations

Ces tests vérifient principalement :
- le type de retour ;
- la présence des concepts importants ;
- la présence des formules essentielles ;
- le fonctionnement de l'explication des erreurs ;
- la cohérence pédagogique des textes.
"""

import pytest

from core.numerical.integration_explanations import (
    explain_exact_vs_numerical,
    explain_integration_error,
    explain_method_comparison,
    explain_number_of_subdivisions,
    explain_rectangle_rule,
    explain_simpson_one_third,
    explain_simpson_requirement,
    explain_trapezoidal_rule,
)


# ============================================================
# Tests généraux
# ============================================================


class TestExplanationReturnTypes:
    """Vérifie que toutes les fonctions retournent une chaîne."""

    @pytest.mark.parametrize(
        "explanation_function",
        [
            explain_rectangle_rule,
            explain_trapezoidal_rule,
            explain_simpson_one_third,
            explain_method_comparison,
            explain_number_of_subdivisions,
            explain_exact_vs_numerical,
            explain_simpson_requirement,
        ],
    )
    def test_explanation_returns_string(
        self,
        explanation_function,
    ):
        result = explanation_function()

        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_integration_error_returns_string(self):
        result = explain_integration_error(
            exact_value=1 / 3,
            approximate_value=0.333,
        )

        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ============================================================
# Méthode des rectangles
# ============================================================


class TestExplainRectangleRule:
    """Tests de l'explication de la méthode des rectangles."""

    def test_contains_method_name(self):
        result = explain_rectangle_rule()

        assert "Méthode des rectangles" in result

    def test_contains_rectangle_concept(self):
        result = explain_rectangle_rule()

        assert "rectangles" in result.lower()

    def test_mentions_midpoint(self):
        result = explain_rectangle_rule()

        assert "milieu" in result.lower()

    def test_contains_step_size_formula(self):
        result = explain_rectangle_rule()

        assert "b-a" in result
        assert "n" in result

    def test_contains_integral_notation(self):
        result = explain_rectangle_rule()

        assert r"\int" in result

    def test_contains_summation(self):
        result = explain_rectangle_rule()

        assert r"\sum" in result

    def test_mentions_convergence(self):
        result = explain_rectangle_rule()

        text = result.lower()

        assert "augmente" in text
        assert "approximation" in text


# ============================================================
# Méthode des trapèzes
# ============================================================


class TestExplainTrapezoidalRule:
    """Tests de l'explication de la méthode des trapèzes."""

    def test_contains_method_name(self):
        result = explain_trapezoidal_rule()

        assert "Méthode des trapèzes" in result

    def test_contains_trapezoid_concept(self):
        result = explain_trapezoidal_rule()

        assert "trapèze" in result.lower()

    def test_mentions_segments(self):
        result = explain_trapezoidal_rule()

        assert "segments" in result.lower()

    def test_contains_step_size_formula(self):
        result = explain_trapezoidal_rule()

        assert "b-a" in result
        assert "n" in result

    def test_contains_trapezoidal_formula_terms(self):
        result = explain_trapezoidal_rule()

        assert "f(a)" in result
        assert "f(b)" in result
        assert r"\sum" in result

    def test_contains_integral_notation(self):
        result = explain_trapezoidal_rule()

        assert r"\int" in result

    def test_mentions_curve_approximation(self):
        result = explain_trapezoidal_rule()

        text = result.lower()

        assert "courbe" in text
        assert "approxime" in text


# ============================================================
# Méthode de Simpson 1/3
# ============================================================


class TestExplainSimpsonOneThird:
    """Tests de l'explication de Simpson 1/3."""

    def test_contains_method_name(self):
        result = explain_simpson_one_third()

        assert "Méthode de Simpson 1/3" in result

    def test_mentions_polynomial_degree_two(self):
        result = explain_simpson_one_third()

        text = result.lower()

        assert "polynômes" in text
        assert "degré 2" in text

    def test_mentions_even_subdivisions(self):
        result = explain_simpson_one_third()

        text = result.lower()

        assert "pair" in text
        assert "n" in text

    def test_contains_simpson_coefficients(self):
        result = explain_simpson_one_third()

        assert "1" in result
        assert "4" in result
        assert "2" in result

    def test_contains_integral_notation(self):
        result = explain_simpson_one_third()

        assert r"\int" in result

    def test_contains_summations(self):
        result = explain_simpson_one_third()

        assert r"\sum" in result

    def test_contains_step_size_formula(self):
        result = explain_simpson_one_third()

        assert "b-a" in result
        assert "n" in result

    def test_mentions_accuracy(self):
        result = explain_simpson_one_third()

        text = result.lower()

        assert "précision" in text
        assert "approximation" in text


# ============================================================
# Explication de l'erreur d'intégration
# ============================================================


class TestExplainIntegrationError:
    """Tests de l'explication de l'erreur numérique."""

    def test_contains_title(self):
        result = explain_integration_error(
            exact_value=1 / 3,
            approximate_value=0.333,
        )

        assert "Erreur d'approximation" in result

    def test_contains_exact_value(self):
        result = explain_integration_error(
            exact_value=10.5,
            approximate_value=10.2,
        )

        assert "10.5" in result

    def test_contains_approximate_value(self):
        result = explain_integration_error(
            exact_value=10.5,
            approximate_value=10.2,
        )

        assert "10.2" in result

    def test_calculates_absolute_error(self):
        result = explain_integration_error(
            exact_value=10.5,
            approximate_value=10.2,
        )

        expected_error = 0.3

        assert f"{expected_error:.10g}" in result

    def test_zero_error(self):
        result = explain_integration_error(
            exact_value=5.0,
            approximate_value=5.0,
        )

        assert "0" in result

    def test_error_is_absolute(self):
        result_positive = explain_integration_error(
            exact_value=10.0,
            approximate_value=9.5,
        )

        result_negative = explain_integration_error(
            exact_value=9.5,
            approximate_value=10.0,
        )

        error_text = "0.5"

        assert error_text in result_positive
        assert error_text in result_negative

    @pytest.mark.parametrize(
        "exact_value, approximate_value",
        [
            (1.0, 0.9),
            (0.0, 0.0),
            (-2.0, -2.5),
            (100.0, 99.99),
        ],
    )
    def test_accepts_different_numeric_values(
        self,
        exact_value,
        approximate_value,
    ):
        result = explain_integration_error(
            exact_value,
            approximate_value,
        )

        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ============================================================
# Comparaison des méthodes
# ============================================================


class TestExplainMethodComparison:
    """Tests de l'explication comparative."""

    def test_contains_title(self):
        result = explain_method_comparison()

        assert "Comparaison des méthodes" in result

    def test_mentions_rectangle_method(self):
        result = explain_method_comparison()

        assert "Rectangles" in result

    def test_mentions_trapezoidal_method(self):
        result = explain_method_comparison()

        assert "Trapèzes" in result

    def test_mentions_simpson_method(self):
        result = explain_method_comparison()

        assert "Simpson 1/3" in result

    def test_mentions_approximation_types(self):
        result = explain_method_comparison()

        text = result.lower()

        assert "rectangles" in text
        assert "segments" in text
        assert "polynômes" in text

    def test_mentions_absolute_error(self):
        result = explain_method_comparison()

        assert "erreur absolue" in result.lower()

    def test_mentions_number_of_subdivisions(self):
        result = explain_method_comparison()

        assert "subdivisions" in result.lower()

    def test_mentions_approximate_value(self):
        result = explain_method_comparison()

        assert "valeur approchée" in result.lower()


# ============================================================
# Nombre de subdivisions
# ============================================================


class TestExplainNumberOfSubdivisions:
    """Tests sur l'influence du nombre de subdivisions."""

    def test_contains_title(self):
        result = explain_number_of_subdivisions()

        assert "Influence du nombre de subdivisions" in result

    def test_mentions_n(self):
        result = explain_number_of_subdivisions()

        assert "n" in result

    def test_contains_step_size_formula(self):
        result = explain_number_of_subdivisions()

        assert "b-a" in result
        assert "n" in result

    def test_mentions_smaller_intervals(self):
        result = explain_number_of_subdivisions()

        text = result.lower()

        assert "petits" in text
        assert "sous-intervalles" in text

    def test_mentions_precision(self):
        result = explain_number_of_subdivisions()

        text = result.lower()

        assert "précision" in text
        assert "erreur" in text

    def test_mentions_computational_cost(self):
        result = explain_number_of_subdivisions()

        text = result.lower()

        assert "calculs" in text
        assert "coût" in text


# ============================================================
# Intégration exacte vs numérique
# ============================================================


class TestExplainExactVsNumerical:
    """Tests de l'explication exacte vs numérique."""

    def test_contains_title(self):
        result = explain_exact_vs_numerical()

        assert "Intégration exacte vs intégration numérique" in result

    def test_mentions_exact_integration(self):
        result = explain_exact_vs_numerical()

        assert "Intégration exacte" in result

    def test_mentions_numerical_integration(self):
        result = explain_exact_vs_numerical()

        assert "Intégration numérique" in result

    def test_contains_integral_notation(self):
        result = explain_exact_vs_numerical()

        assert r"\int" in result

    def test_contains_exact_example(self):
        result = explain_exact_vs_numerical()

        assert "x^2" in result
        assert "1" in result
        assert "3" in result

    def test_mentions_approximation(self):
        result = explain_exact_vs_numerical()

        text = result.lower()

        assert "approximation" in text
        assert "points" in text

    def test_mentions_difficult_primitive(self):
        result = explain_exact_vs_numerical()

        text = result.lower()

        assert "primitive" in text
        assert "difficile" in text


# ============================================================
# Condition n pair pour Simpson
# ============================================================


class TestExplainSimpsonRequirement:
    """Tests de l'explication de la condition n pair."""

    def test_contains_title(self):
        result = explain_simpson_requirement()

        assert "Pourquoi" in result
        assert "pair" in result

    def test_mentions_two_subintervals(self):
        result = explain_simpson_requirement()

        text = result.lower()

        assert "deux sous-intervalles" in text

    def test_mentions_three_points(self):
        result = explain_simpson_requirement()

        text = result.lower()

        assert "trois points" in text

    def test_contains_points_indices(self):
        result = explain_simpson_requirement()

        assert "x_i" in result
        assert "x_{i+1}" in result
        assert "x_{i+2}" in result

    def test_contains_even_examples(self):
        result = explain_simpson_requirement()

        assert "n=2" in result
        assert "n=4" in result
        assert "n=6" in result

    def test_contains_odd_examples(self):
        result = explain_simpson_requirement()

        assert "n=3" in result
        assert "n=5" in result

    def test_contains_requirement(self):
        result = explain_simpson_requirement()

        assert "doit être pair" in result


# ============================================================
# Cohérence pédagogique globale
# ============================================================


class TestPedagogicalConsistency:
    """Tests de cohérence entre les différentes explications."""

    def test_all_main_methods_are_documented(self):
        """Vérifie que les trois méthodes principales sont documentées."""
        explanations = [
            explain_rectangle_rule(),
            explain_trapezoidal_rule(),
            explain_simpson_one_third(),
        ]

        for explanation in explanations:
            assert isinstance(explanation, str)
            assert len(explanation.strip()) > 0

            text = explanation.lower()

            # Chaque explication doit présenter la méthode
            assert any(
                keyword in text
                for keyword in (
                    "rectangle",
                    "trapèze",
                    "simpson",
                )
            )

            # Chaque explication doit contenir une formule ou une notation
            # mathématique significative.
            assert (
                r"\int" in explanation
                or "f(a)" in explanation
                or "f(b)" in explanation
                or "h" in explanation
            )

    def test_all_explanations_are_markdown_compatible(self):
        explanations = [
            explain_rectangle_rule(),
            explain_trapezoidal_rule(),
            explain_simpson_one_third(),
            explain_method_comparison(),
            explain_number_of_subdivisions(),
            explain_exact_vs_numerical(),
            explain_simpson_requirement(),
        ]

        for explanation in explanations:
            assert isinstance(explanation, str)

            # Présence d'un élément Markdown ou LaTeX.
            assert (
                "#" in explanation
                or r"\[" in explanation
                or "|" in explanation
            )

    def test_error_explanation_contains_mathematical_notation(self):
        result = explain_integration_error(
            exact_value=1.0,
            approximate_value=0.95,
        )

        assert "I" in result
        assert "I_n" in result
        assert "E" in result
        assert "|I-I_n|" in result

