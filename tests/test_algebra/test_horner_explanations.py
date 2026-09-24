
from __future__ import annotations

from core.algebra.horner import (
    horner_division,
)
from core.algebra.horner_explanations import (
    explain_horner_evaluation,
    explain_horner_division,
)


# ============================================================
# TEST ÉVALUATION HORNER
# ============================================================
def test_explain_horner_evaluation():
    coefficients = [
        1,
        -6,
        11,
        -6,
    ]

    steps = explain_horner_evaluation(
        coefficients,
        2,
    )

    assert isinstance(steps, list)

    # 1 identification
    # 2 choix de x
    # 3 initialisation
    # 4 étape 1
    # 5 étape 2
    # 6 étape 3
    # 7 résultat
    assert len(steps) == 7

    assert steps[0]["title"] == (
        "Écrire le polynôme"
    )

    assert steps[1]["title"] == (
        "Choisir la valeur"
    )

    assert steps[2]["title"] == (
        "Initialiser Horner"
    )

    assert steps[-1]["title"] == (
        "Résultat"
    )

    assert steps[-1]["formula"] == (
        "P(2) = 0"
    )




def test_explain_horner_evaluation_intermediate_steps():
    coefficients = [
        1,
        -6,
        11,
        -6,
    ]

    steps = explain_horner_evaluation(
        coefficients,
        2,
    )

    assert steps[2]["formula"] == (
        "b_0 = 1"
    )

    assert steps[3]["formula"] == (
        "b_1 = b_0 × 2 + (-6) = -4"
    )

    assert steps[4]["formula"] == (
        "b_2 = b_1 × 2 + (11) = 3"
    )

    assert steps[5]["formula"] == (
        "b_3 = b_2 × 2 + (-6) = 0"
    )


def test_explain_horner_evaluation_non_root():
    coefficients = [
        1,
        -6,
        11,
        -6,
    ]

    steps = explain_horner_evaluation(
        coefficients,
        0,
    )

    assert steps[-1]["formula"] == (
        "P(0) = -6"
    )


def test_explain_horner_evaluation_empty_coefficients():
    try:
        explain_horner_evaluation(
            [],
            2,
        )
    except ValueError as error:
        assert str(error) == (
            "Les coefficients ne peuvent pas être vides."
        )
    else:
        assert False


# ============================================================
# TEST DIVISION HORNER
# ============================================================


def test_explain_horner_division():
    coefficients = [
        1,
        -6,
        11,
        -6,
    ]

    division_result = horner_division(
        coefficients,
        1,
    )

    steps = explain_horner_division(
        coefficients,
        1,
        division_result,
    )

    assert isinstance(steps, list)

    assert len(steps) == 5

    assert steps[0]["title"] == (
        "Choisir la racine"
    )

    assert steps[1]["title"] == (
        "Appliquer Horner"
    )

    assert steps[2]["title"] == (
        "Obtenir le quotient"
    )

    assert steps[3]["title"] == (
        "Calculer le reste"
    )

    assert steps[4]["title"] == (
        "Interpréter le résultat"
    )


def test_explain_horner_division_quotient():
    coefficients = [
        1,
        -6,
        11,
        -6,
    ]

    division_result = horner_division(
        coefficients,
        1,
    )

    steps = explain_horner_division(
        coefficients,
        1,
        division_result,
    )

    assert steps[2]["formula"] == (
        "Coefficients du quotient : [1, -5, 6]"
    )


def test_explain_horner_division_remainder():
    coefficients = [
        1,
        -6,
        11,
        -6,
    ]

    division_result = horner_division(
        coefficients,
        1,
    )

    steps = explain_horner_division(
        coefficients,
        1,
        division_result,
    )

    assert steps[3]["formula"] == (
        "R = 0"
    )

    assert steps[4]["formula"] == (
        "P(1) = 0"
    )

    assert "est une racine" in (
        steps[4]["explanation"]
    )


def test_explain_horner_division_non_root():
    coefficients = [
        1,
        -6,
        11,
        -6,
    ]

    division_result = horner_division(
        coefficients,
        0,
    )

    steps = explain_horner_division(
        coefficients,
        0,
        division_result,
    )

    assert steps[3]["formula"] == (
        "R = -6"
    )

    assert steps[4]["formula"] == (
        "P(0) = -6"
    )

    assert "n'est pas une racine" in (
        steps[4]["explanation"]
    )


def test_explain_horner_division_multiple_root():
    coefficients = [
        1,
        -3,
        3,
        -1,
    ]

    division_result = horner_division(
        coefficients,
        1,
    )

    steps = explain_horner_division(
        coefficients,
        1,
        division_result,
    )

    assert steps[3]["formula"] == (
        "R = 0"
    )

    assert "est une racine" in (
        steps[4]["explanation"]
    )

