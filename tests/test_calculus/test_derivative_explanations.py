import sympy as sp

from core.calculus.derivatives import (
    analyze_derivative,
)

from core.calculus.derivative_explanations import (
    explain_derivative,
)


def test_explain_derivative_without_point():

    x = sp.Symbol("x")

    function = x**3

    result = analyze_derivative(
        function
    )

    steps = explain_derivative(
        function,
        result,
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert "Identifier la fonction" in titles

    assert (
        "Calculer la dérivée première"
        in titles
    )

    assert (
        "Calculer la dérivée seconde"
        in titles
    )


def test_explain_first_derivative():

    x = sp.Symbol("x")

    function = x**3

    result = analyze_derivative(
        function
    )

    steps = explain_derivative(
        function,
        result,
    )

    derivative_step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer la dérivée première"
    )

    assert "3 x^{2}" in derivative_step["formula"]

    assert derivative_step["explanation"]


def test_explain_second_derivative():

    x = sp.Symbol("x")

    function = x**3

    result = analyze_derivative(
        function
    )

    steps = explain_derivative(
        function,
        result,
    )

    second_derivative_step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer la dérivée seconde"
    )

    assert "6 x" in (
        second_derivative_step["formula"]
    )

    assert (
        second_derivative_step["explanation"]
    )


def test_explain_derivative_with_point():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_derivative(
        function,
        2,
    )

    steps = explain_derivative(
        function,
        result,
    )

    titles = [
        step["title"]
        for step in steps
    ]

    assert (
        "Calculer la valeur de la fonction"
        in titles
    )

    assert (
        "Calculer la valeur de la dérivée"
        in titles
    )

    assert (
        "Déterminer la pente de la tangente"
        in titles
    )

    assert (
        "Calculer l'équation de la tangente"
        in titles
    )

    assert "Conclusion" in titles


def test_explain_function_value():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_derivative(
        function,
        2,
    )

    steps = explain_derivative(
        function,
        result,
    )

    value_step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer la valeur de la fonction"
    )

    assert "4" in value_step["formula"]


def test_explain_derivative_value():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_derivative(
        function,
        2,
    )

    steps = explain_derivative(
        function,
        result,
    )

    derivative_step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer la valeur de la dérivée"
    )

    assert "4" in derivative_step["formula"]


def test_explain_tangent_line():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_derivative(
        function,
        2,
    )

    steps = explain_derivative(
        function,
        result,
    )

    tangent_step = next(
        step
        for step in steps
        if step["title"]
        == "Calculer l'équation de la tangente"
    )

    assert tangent_step["formula"]

    assert "4" in tangent_step["formula"]


def test_explain_conclusion():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_derivative(
        function,
        2,
    )

    steps = explain_derivative(
        function,
        result,
    )

    conclusion = next(
        step
        for step in steps
        if step["title"]
        == "Conclusion"
    )

    assert "4" in conclusion["formula"]

    assert (
        "pente"
        in conclusion["explanation"]
    )