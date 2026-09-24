import sympy as sp

from core.calculus.limits import (
    analyze_limit,
)

from core.calculus.limit_explanations import (
    explain_limit,
)


def test_explain_existing_limit():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_limit(
        function,
        2,
    )

    steps = explain_limit(
        function,
        2,
        result,
    )

    assert steps

    assert steps[0]["title"] == (
        "Identifier la fonction"
    )

    assert any(
        step["title"]
        == "Calculer la limite à gauche"
        for step in steps
    )

    assert any(
        step["title"]
        == "Calculer la limite à droite"
        for step in steps
    )

    assert any(
        step["title"]
        == "Conclusion"
        for step in steps
    )


def test_explain_non_existing_limit():

    x = sp.Symbol("x")

    function = sp.Abs(x) / x

    result = analyze_limit(
        function,
        0,
    )

    steps = explain_limit(
        function,
        0,
        result,
    )

    assert steps

    comparison_step = next(
        step
        for step in steps
        if step["title"]
        == "Comparer les deux limites"
    )

    assert comparison_step["formula"]

    conclusion_step = next(
        step
        for step in steps
        if step["title"]
        == "Conclusion"
    )

    assert (
        "n'existe pas"
        in conclusion_step["formula"]
    )


def test_explanation_contains_result():

    x = sp.Symbol("x")

    function = x**2 + 1

    result = analyze_limit(
        function,
        2,
    )

    steps = explain_limit(
        function,
        2,
        result,
    )

    conclusion_step = next(
        step
        for step in steps
        if step["title"]
        == "Conclusion"
    )

    assert "5" in conclusion_step["formula"]