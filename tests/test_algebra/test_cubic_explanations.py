
from __future__ import annotations

import sympy as sp

from core.algebra.cubic_equations import (
    solve_cubic_equation,
)

from core.algebra.cubic_explanations import (
    explain_cubic_equation,
)


# ============================================================
# TEST PRINCIPAL
# ============================================================


def test_explain_cubic_equation():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    assert isinstance(steps, list)

    assert len(steps) > 0

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier l'équation" in titles
    assert "Identifier les coefficients" in titles
    assert "Calculer le discriminant cubique" in titles
    assert "Déterminer la nature des racines" in titles
    assert "Rechercher une racine rationnelle" in titles
    assert "Racine(s) trouvée(s) avec Horner" in titles
    assert "Réduire le degré" in titles
    assert "Factoriser" in titles
    assert "Préparer la méthode de Cardano" in titles
    assert "Calculer p et q" in titles
    assert "Calculer le discriminant de Cardano" in titles
    assert "Solutions" in titles


# ============================================================
# TEST IDENTIFICATION DE L'ÉQUATION
# ============================================================


def test_cubic_explanation_identifies_equation():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"] == "Identifier l'équation"
    )

    assert "x^{3}" in step["formula"]

    assert step["explanation"] != ""


# ============================================================
# TEST DES COEFFICIENTS
# ============================================================


def test_cubic_explanation_coefficients():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"] == "Identifier les coefficients"
    )

    assert "a=1" in step["formula"]
    assert "b=-6" in step["formula"]
    assert "c=11" in step["formula"]
    assert "d=-6" in step["formula"]


# ============================================================
# TEST DU DISCRIMINANT
# ============================================================


def test_cubic_explanation_discriminant():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer le discriminant cubique"
    )

    assert "4" in step["formula"]


# ============================================================
# TEST DE LA NATURE DES RACINES
# ============================================================


def test_cubic_explanation_root_nature():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"]
        == "Déterminer la nature des racines"
    )

    assert (
        step["formula"]
        == "three_distinct_real_roots"
    )


# ============================================================
# TEST HORNER
# ============================================================


def test_cubic_explanation_horner():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    horner_step = next(
        step
        for step in steps
        if step["title"]
        == "Racine(s) trouvée(s) avec Horner"
    )

    assert "1" in horner_step["formula"]
    assert "2" in horner_step["formula"]
    assert "3" in horner_step["formula"]


def test_cubic_explanation_degree_reduction():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"] == "Réduire le degré"
    )

    assert "x" in step["formula"]

    assert step["explanation"] != ""


# ============================================================
# TEST FACTORISATION
# ============================================================


def test_cubic_explanation_factorization():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"] == "Factoriser"
    )

    assert "x" in step["formula"]

    assert (
        sp.latex(result["factorized"])
        in step["formula"]
    )


# ============================================================
# TEST CARDANO
# ============================================================


def test_cubic_explanation_cardano_parameters():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"] == "Calculer p et q"
    )

    assert "p" in step["formula"]
    assert "q" in step["formula"]

    assert "-1" in step["formula"]
    assert "0" in step["formula"]


# ============================================================
# TEST DISCRIMINANT DE CARDANO
# ============================================================


def test_cubic_explanation_cardano_discriminant():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer le discriminant de Cardano"
    )

    assert r"- \frac{1}{27}" in step["formula"]


# ============================================================
# TEST DES SOLUTIONS
# ============================================================


def test_cubic_explanation_solutions():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    steps = explain_cubic_equation(
        result
    )

    step = next(
        step
        for step in steps
        if step["title"] == "Solutions"
    )

    assert "1" in step["formula"]
    assert "2" in step["formula"]
    assert "3" in step["formula"]


# ============================================================
# TEST RACINES MULTIPLES
# ============================================================


def test_cubic_explanation_multiple_roots():
    result = solve_cubic_equation(
        1,
        -3,
        3,
        -1,
    )

    steps = explain_cubic_equation(
        result
    )

    assert isinstance(steps, list)

    solution_step = next(
        step
        for step in steps
        if step["title"] == "Solutions"
    )

    assert "1" in solution_step["formula"]


# ============================================================
# TEST UNE RACINE RÉELLE + DEUX COMPLEXES
# ============================================================


def test_cubic_explanation_complex_roots():
    result = solve_cubic_equation(
        1,
        0,
        0,
        -1,
    )

    steps = explain_cubic_equation(
        result
    )

    nature_step = next(
        step
        for step in steps
        if step["title"]
        == "Déterminer la nature des racines"
    )

    assert (
        nature_step["formula"]
        == "one_real_root_and_two_complex_roots"
    )

    solution_step = next(
        step
        for step in steps
        if step["title"] == "Solutions"
    )

    assert "1" in solution_step["formula"]

