import sympy as sp
import pytest
import plotly.graph_objects as go

from core.calculus.plots import (
    create_function_plot,
)


def test_create_function_plot():

    x = sp.Symbol("x")

    function = x**2

    figure = create_function_plot(
        function,
        2,
    )

    assert isinstance(
        figure,
        go.Figure,
    )

    assert len(
        figure.data
    ) == 1


def test_create_function_plot_rational():

    x = sp.Symbol("x")

    function = (
        (x**2 - 1)
        / (x - 1)
    )

    figure = create_function_plot(
        function,
        1,
    )

    assert isinstance(
        figure,
        go.Figure,
    )

    assert len(
        figure.data
    ) == 1


def test_create_function_plot_rejects_infinity():

    x = sp.Symbol("x")

    function = x**2

    with pytest.raises(ValueError):

        create_function_plot(
            function,
            sp.oo,
        )