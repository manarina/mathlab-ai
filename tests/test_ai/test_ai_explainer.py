# ============================================================
# MATHLAB AI — TESTS AI EXPLAINER
# ============================================================

import pytest

from core.ai.ai_explainer import (
    build_explanation_prompt,
)


# ============================================================
# TESTS — PROMPT DÉRIVÉE
# ============================================================

def test_build_explanation_prompt_derivative():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2 + 3*x",
        result="2*x + 3",
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "x^2 + 3*x" in prompt
    assert "2*x + 3" in prompt
    assert "dérivation" in prompt
    assert "français" in prompt


# ============================================================
# TESTS — PROMPT INTÉGRALE
# ============================================================

def test_build_explanation_prompt_integral():

    prompt = build_explanation_prompt(
        operation="integral",
        expression="x^2",
        result="x^3/3",
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "x^2" in prompt
    assert "x^3/3" in prompt
    assert "intégration" in prompt


# ============================================================
# TESTS — PROMPT LIMITE
# ============================================================

def test_build_explanation_prompt_limit():

    prompt = build_explanation_prompt(
        operation="limit",
        expression="sin(x)/x",
        result="1",
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "sin(x)/x" in prompt
    assert "1" in prompt
    assert "limite" in prompt


# ============================================================
# TESTS — PROMPT ÉQUATION
# ============================================================

def test_build_explanation_prompt_equation():

    prompt = build_explanation_prompt(
        operation="equation",
        expression="2*x + 5 = 15",
        result="[5]",
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "2*x + 5 = 15" in prompt
    assert "[5]" in prompt
    assert "équation" in prompt


# ============================================================
# TESTS — PROMPT SIMPLIFICATION
# ============================================================

def test_build_explanation_prompt_simplification():

    prompt = build_explanation_prompt(
        operation="simplify",
        expression="(x^2 - 1)/(x - 1)",
        result="x + 1",
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "(x^2 - 1)/(x - 1)" in prompt
    assert "x + 1" in prompt
    assert "simplification" in prompt


# ============================================================
# TESTS — CALCUL NUMÉRIQUE
# ============================================================

def test_build_explanation_prompt_calculation():

    prompt = build_explanation_prompt(
        operation="calculation",
        expression="2 + 3 * 4",
        result="14",
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "2 + 3 * 4" in prompt
    assert "14" in prompt
    assert "calcul numérique" in prompt


# ============================================================
# VALIDATION — OPÉRATION VIDE
# ============================================================

def test_empty_operation():

    with pytest.raises(ValueError):

        build_explanation_prompt(
            operation="",
            expression="x^2",
            result="2*x",
        )


# ============================================================
# VALIDATION — MAUVAIS TYPE OPÉRATION
# ============================================================

def test_invalid_operation_type():

    with pytest.raises(TypeError):

        build_explanation_prompt(
            operation=None,
            expression="x^2",
            result="2*x",
        )


# ============================================================
# VALIDATION — EXPRESSION VIDE
# ============================================================

def test_empty_expression():

    with pytest.raises(ValueError):

        build_explanation_prompt(
            operation="derivative",
            expression="",
            result="2*x",
        )


# ============================================================
# VALIDATION — MAUVAIS TYPE EXPRESSION
# ============================================================

def test_invalid_expression_type():

    with pytest.raises(TypeError):

        build_explanation_prompt(
            operation="derivative",
            expression=None,
            result="2*x",
        )


# ============================================================
# VALIDATION — RÉSULTAT MANQUANT
# ============================================================

def test_missing_result():

    with pytest.raises(ValueError):

        build_explanation_prompt(
            operation="derivative",
            expression="x^2",
            result=None,
        )


# ============================================================
# TESTS — ÉTAPES SIMPLES
# ============================================================

def test_prompt_with_steps():

    steps = [
        "Appliquer la règle de puissance.",
        "Dériver x^2.",
        "Dériver 3*x.",
    ]

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2 + 3*x",
        result="2*x + 3",
        steps=steps,
    )

    assert "Appliquer la règle de puissance." in prompt
    assert "Dériver x^2." in prompt
    assert "Dériver 3*x." in prompt


# ============================================================
# TESTS — ÉTAPES SOUS FORME DE DICTIONNAIRES
# ============================================================

def test_prompt_with_dictionary_steps():

    steps = [
        {
            "description": "Appliquer la règle de puissance."
        },
        {
            "step": "Dériver le terme constant."
        },
    ]

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2 + 3",
        result="2*x",
        steps=steps,
    )

    assert "Appliquer la règle de puissance." in prompt
    assert "Dériver le terme constant." in prompt


# ============================================================
# TESTS — NIVEAU D'EXPLICATION
# ============================================================

def test_prompt_level_lycee():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
        level="lycée",
    )

    assert "lycée" in prompt


def test_prompt_level_universite():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
        level="université",
    )

    assert "université" in prompt


def test_prompt_level_debutant():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
        level="débutant",
    )

    assert "débutant" in prompt


# ============================================================
# TESTS — CONTENU PÉDAGOGIQUE DU PROMPT
# ============================================================

def test_prompt_contains_teacher_role():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
    )

    assert "professeur de mathématiques" in prompt


def test_prompt_requests_french():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
    )

    assert "français" in prompt


def test_prompt_requests_explanation():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
    )

    assert "explication" in prompt.lower()


def test_prompt_contains_result_instruction():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
    )

    assert "résultat calculé" in prompt


# ============================================================
# TESTS — OPÉRATION INCONNUE
# ============================================================

def test_unknown_operation():

    prompt = build_explanation_prompt(
        operation="unknown_operation",
        expression="x^2",
        result="x^2",
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "unknown_operation" in prompt


# ============================================================
# TESTS — SANS ÉTAPES
# ============================================================

def test_prompt_without_steps():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
        steps=None,
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "x^2" in prompt
    assert "2*x" in prompt


# ============================================================
# TESTS — LISTE D'ÉTAPES VIDE
# ============================================================

def test_prompt_with_empty_steps():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
        steps=[],
    )

    assert isinstance(
        prompt,
        str,
    )

    assert "x^2" in prompt
    assert "2*x" in prompt


# ============================================================
# TESTS — PROMPT NON VIDE
# ============================================================

def test_prompt_is_not_empty():

    prompt = build_explanation_prompt(
        operation="derivative",
        expression="x^2",
        result="2*x",
    )

    assert prompt.strip() != ""


# ============================================================
# TESTS — PROMPT CONTIENT L'EXPRESSION
# ============================================================

@pytest.mark.parametrize(
    "expression,result",
    [
        ("x^2", "2*x"),
        ("sin(x)", "cos(x)"),
        ("cos(x)", "-sin(x)"),
        ("x^3 + 2*x", "3*x^2 + 2"),
        ("exp(x)", "exp(x)"),
    ],
)
def test_prompt_preserves_expression_and_result(
    expression,
    result,
):

    prompt = build_explanation_prompt(
        operation="derivative",
        expression=expression,
        result=result,
    )

    assert expression in prompt
    assert result in prompt