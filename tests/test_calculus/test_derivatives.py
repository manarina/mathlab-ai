import sympy as sp
import pytest

from core.calculus.derivatives import (
    analyze_derivative,
    calculate_derivative,
    calculate_nth_derivative,
    calculate_second_derivative,
    calculate_tangent_line,
    calculate_tangent_slope,
    evaluate_derivative,
    evaluate_function,
    parse_function,
)


# ============================================================
# PARSING
# ============================================================

def test_parse_function():

    function = parse_function(
        "x**2 + 3*x - 2"
    )

    x = sp.Symbol("x")

    assert function == (
        x**2 + 3*x - 2
    )


def test_parse_invalid_function():

    with pytest.raises(ValueError):

        parse_function(
            "expression_invalide_("
        )


def test_parse_empty_function():

    with pytest.raises(ValueError):

        parse_function("")


# ============================================================
# DERIVEE PREMIERE
# ============================================================

def test_calculate_derivative_polynomial():

    x = sp.Symbol("x")

    function = (
        x**3
        + 2*x**2
        + x
        + 1
    )

    result = calculate_derivative(
        function
    )

    expected = (
        3*x**2
        + 4*x
        + 1
    )

    assert result == expected


def test_calculate_derivative_trigonometric():

    x = sp.Symbol("x")

    function = sp.sin(x)

    result = calculate_derivative(
        function
    )

    assert result == sp.cos(x)


def test_calculate_derivative_exponential():

    x = sp.Symbol("x")

    function = sp.exp(x)

    result = calculate_derivative(
        function
    )

    assert result == sp.exp(x)


# ============================================================
# DERIVEE SECONDE
# ============================================================

def test_calculate_second_derivative():

    x = sp.Symbol("x")

    function = x**3

    result = calculate_second_derivative(
        function
    )

    assert result == 6*x


# ============================================================
# DERIVEE D'ORDRE N
# ============================================================

def test_calculate_nth_derivative():

    x = sp.Symbol("x")

    function = x**5

    result = calculate_nth_derivative(
        function,
        3,
    )

    assert result == 60*x**2


def test_nth_derivative_order_zero():

    x = sp.Symbol("x")

    function = x**3 + 2

    result = calculate_nth_derivative(
        function,
        0,
    )

    assert result == function


def test_nth_derivative_negative_order():

    x = sp.Symbol("x")

    function = x**2

    with pytest.raises(ValueError):

        calculate_nth_derivative(
            function,
            -1,
        )


def test_nth_derivative_invalid_order():

    x = sp.Symbol("x")

    function = x**2

    with pytest.raises(TypeError):

        calculate_nth_derivative(
            function,
            1.5,
        )


# ============================================================
# EVALUATION
# ============================================================

def test_evaluate_derivative():

    x = sp.Symbol("x")

    function = x**2

    result = evaluate_derivative(
        function,
        3,
    )

    assert result == 6


def test_evaluate_function():

    x = sp.Symbol("x")

    function = x**2 + 1

    result = evaluate_function(
        function,
        3,
    )

    assert result == 10


# ============================================================
# TANGENTE
# ============================================================

def test_calculate_tangent_slope():

    x = sp.Symbol("x")

    function = x**2

    result = calculate_tangent_slope(
        function,
        2,
    )

    assert result == 4


def test_calculate_tangent_line():

    x = sp.Symbol("x")

    function = x**2

    result = calculate_tangent_line(
        function,
        2,
    )

    expected = (
        4*x - 4
    )

    assert result == expected


# ============================================================
# ANALYSE COMPLETE
# ============================================================

def test_analyze_derivative_without_point():

    x = sp.Symbol("x")

    function = x**3

    result = analyze_derivative(
        function
    )

    assert result["function"] == x**3
    assert result["derivative"] == 3*x**2
    assert result["second_derivative"] == 6*x


def test_analyze_derivative_with_point():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_derivative(
        function,
        2,
    )

    assert result["function_value"] == 4
    assert result["derivative_value"] == 4
    assert result["tangent_slope"] == 4
    assert result["tangent_line"] == 4*x - 4