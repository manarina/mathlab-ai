
"""
Tests des explications pédagogiques de l'Optimization Lab.
"""

import pytest

from core.optimization.optimization_explanations import (
    explain_golden_section,
    explain_golden_section_advantages,
    explain_golden_section_formula,
    explain_golden_section_limitations,
    explain_grid_search,
    explain_grid_search_advantages,
    explain_grid_search_formula,
    explain_grid_search_limitations,
    explain_method_comparison,
    explain_newton_advantages,
    explain_newton_convergence,
    explain_newton_formula,
    explain_newton_limitations,
    explain_newton_minimum,
    explain_newton_minimum_condition,
    explain_optimization,
    explain_optimization_workflow,
)


# ============================================================
# INTRODUCTION GÉNÉRALE
# ============================================================


def test_explain_optimization_returns_string():
    """L'introduction générale doit retourner une chaîne."""
    result = explain_optimization()

    assert isinstance(result, str)
    assert result.strip()


def test_explain_optimization_contains_key_concepts():
    """L'introduction doit présenter les notions essentielles."""
    result = explain_optimization().lower()

    assert "optimisation" in result
    assert "fonction" in result
    assert "minimum" in result
    assert "maximum" in result


# ============================================================
# GRID SEARCH
# ============================================================


def test_explain_grid_search_returns_string():
    """L'explication du balayage doit retourner une chaîne."""
    result = explain_grid_search()

    assert isinstance(result, str)
    assert result.strip()


def test_explain_grid_search_contains_key_concepts():
    """L'explication du balayage doit contenir ses notions principales."""
    result = explain_grid_search().lower()

    assert "balayage" in result
    assert "intervalle" in result
    assert "points" in result
    assert "minimum" in result


def test_explain_grid_search_formula():
    """La formule conceptuelle du Grid Search doit être présente."""
    result = explain_grid_search_formula()

    assert isinstance(result, str)
    assert "argmin" in result
    assert "[a, b]" in result


def test_explain_grid_search_advantages():
    """Les avantages du balayage doivent être expliqués."""
    result = explain_grid_search_advantages().lower()

    assert isinstance(result, str)
    assert "simplicité" in result
    assert "dérivées" in result


def test_explain_grid_search_limitations():
    """Les limites du balayage doivent être expliquées."""
    result = explain_grid_search_limitations().lower()

    assert isinstance(result, str)
    assert "précision" in result
    assert "points" in result


# ============================================================
# SECTION DORÉE
# ============================================================


def test_explain_golden_section_returns_string():
    """L'explication de la section dorée doit retourner une chaîne."""
    result = explain_golden_section()

    assert isinstance(result, str)
    assert result.strip()


def test_explain_golden_section_contains_key_concepts():
    """Les notions essentielles de la section dorée doivent être présentes."""
    result = explain_golden_section().lower()

    assert "section dorée" in result
    assert "minimum" in result
    assert "unimodale" in result
    assert "intervalle" in result


def test_explain_golden_section_formula():
    """La relation avec le nombre d'or doit être mentionnée."""
    result = explain_golden_section_formula()

    assert isinstance(result, str)
    assert "nombre d'or" in result
    assert "√5" in result
    assert "φ" in result


def test_explain_golden_section_advantages():
    """Les avantages de la section dorée doivent être expliqués."""
    result = explain_golden_section_advantages().lower()

    assert isinstance(result, str)
    assert "dérivée" in result
    assert "unimodales" in result


def test_explain_golden_section_limitations():
    """Les limites de la section dorée doivent être expliquées."""
    result = explain_golden_section_limitations().lower()

    assert isinstance(result, str)
    assert "unimodal" in result
    assert "minimum" in result


# ============================================================
# NEWTON
# ============================================================


def test_explain_newton_minimum_returns_string():
    """L'explication de Newton doit retourner une chaîne."""
    result = explain_newton_minimum()

    assert isinstance(result, str)
    assert result.strip()


def test_explain_newton_minimum_contains_key_concepts():
    """Les notions essentielles de Newton doivent être présentes."""
    result = explain_newton_minimum().lower()

    assert "newton" in result
    assert "première" in result
    assert "deuxième dérivée" in result
    assert "point stationnaire" in result


def test_explain_newton_formula():
    """La formule de Newton doit être présente."""
    result = explain_newton_formula()

    assert isinstance(result, str)
    assert "xₙ₊₁" in result
    assert "f'(xₙ)" in result
    assert "f''(xₙ)" in result


def test_explain_newton_convergence():
    """Les propriétés de convergence doivent être expliquées."""
    result = explain_newton_convergence().lower()

    assert isinstance(result, str)
    assert "locale" in result
    assert "point initial" in result
    assert "converger" in result


def test_explain_newton_minimum_condition():
    """La condition sur la dérivée seconde doit être expliquée."""
    result = explain_newton_minimum_condition()

    assert isinstance(result, str)
    assert "f''(x*) > 0" in result
    assert "minimum local" in result
    assert "maximum local" in result


def test_explain_newton_advantages():
    """Les avantages de Newton doivent être expliqués."""
    result = explain_newton_advantages().lower()

    assert isinstance(result, str)
    assert "rapidement" in result
    assert "précision" in result


def test_explain_newton_limitations():
    """Les limites de Newton doivent être expliquées."""
    result = explain_newton_limitations().lower()

    assert isinstance(result, str)
    assert "dérivées" in result
    assert "point initial" in result
    assert "dérivée seconde" in result


# ============================================================
# COMPARAISON
# ============================================================


def test_explain_method_comparison_returns_string():
    """La comparaison doit retourner une chaîne."""
    result = explain_method_comparison()

    assert isinstance(result, str)
    assert result.strip()


def test_explain_method_comparison_contains_all_methods():
    """La comparaison doit mentionner les trois méthodes."""
    result = explain_method_comparison().lower()

    assert "balayage" in result
    assert "section dorée" in result
    assert "newton" in result


def test_explain_method_comparison_contains_derivative_difference():
    """La comparaison doit distinguer les méthodes avec et sans dérivées."""
    result = explain_method_comparison().lower()

    assert "dérivée" in result


# ============================================================
# WORKFLOW
# ============================================================


def test_explain_optimization_workflow_returns_string():
    """La démarche d'optimisation doit retourner une chaîne."""
    result = explain_optimization_workflow()

    assert isinstance(result, str)
    assert result.strip()


def test_explain_optimization_workflow_contains_main_steps():
    """La démarche doit contenir les étapes principales."""
    result = explain_optimization_workflow().lower()

    assert "fonction objectif" in result
    assert "intervalle" in result
    assert "méthode" in result
    assert "tolérance" in result
    assert "solution" in result


# ============================================================
# QUALITÉ GÉNÉRALE
# ============================================================


@pytest.mark.parametrize(
    "explanation_function",
    [
        explain_optimization,
        explain_grid_search,
        explain_grid_search_formula,
        explain_grid_search_advantages,
        explain_grid_search_limitations,
        explain_golden_section,
        explain_golden_section_formula,
        explain_golden_section_advantages,
        explain_golden_section_limitations,
        explain_newton_minimum,
        explain_newton_formula,
        explain_newton_convergence,
        explain_newton_minimum_condition,
        explain_newton_advantages,
        explain_newton_limitations,
        explain_method_comparison,
        explain_optimization_workflow,
    ],
)
def test_all_explanations_are_non_empty(explanation_function):
    """Toutes les fonctions pédagogiques doivent produire un contenu."""
    result = explanation_function()

    assert isinstance(result, str)
    assert len(result.strip()) > 20

