import sympy as sp
import pytest

from core.calculus.limits import (
    parse_function,
    calculate_limit,
    calculate_one_sided_limits,
    classify_limit,
    analyze_limit,
    evaluate_function,
)


def test_parse_function():

    function = parse_function(
        "x**2 + 3*x - 2"
    )

    assert function == (
        sp.Symbol("x")**2
        + 3 * sp.Symbol("x")
        - 2
    )


def test_parse_empty_function():

    with pytest.raises(ValueError):

        parse_function("")


def test_calculate_polynomial_limit():

    x = sp.Symbol("x")

    function = x**2 + 3*x

    result = calculate_limit(
        function,
        2,
    )

    assert result == 10


def test_calculate_limit_rational():

    x = sp.Symbol("x")

    function = (
        (x**2 - 1)
        / (x - 1)
    )

    result = calculate_limit(
        function,
        1,
    )

    assert result == 2


def test_calculate_left_limit():

    x = sp.Symbol("x")

    function = 1 / x

    result = calculate_limit(
        function,
        0,
        direction="left",
    )

    assert result == -sp.oo


def test_calculate_right_limit():

    x = sp.Symbol("x")

    function = 1 / x

    result = calculate_limit(
        function,
        0,
        direction="right",
    )

    assert result == sp.oo


def test_calculate_infinite_limit():

    x = sp.Symbol("x")

    function = 1 / x

    result = calculate_limit(
        function,
        0,
        direction="two_sided",
    )

    assert result is None

def test_limit_at_plus_infinity():

    x = sp.Symbol("x")

    function = (
        (2*x + 1)
        / x
    )

    result = calculate_limit(
        function,
        sp.oo,
        direction="plus_infinity",
    )

    assert result == 2


def test_limit_at_minus_infinity():

    x = sp.Symbol("x")

    function = (
        (2*x + 1)
        / x
    )

    result = calculate_limit(
        function,
        -sp.oo,
        direction="minus_infinity",
    )

    assert result == 2


def test_one_sided_limits():

    x = sp.Symbol("x")

    function = x**2

    result = calculate_one_sided_limits(
        function,
        2,
    )

    assert result["left"] == 4
    assert result["right"] == 4


def test_classify_existing_limit():

    result = classify_limit(
        sp.Integer(4),
        sp.Integer(4),
    )

    assert result == "finite_or_infinite"


def test_classify_non_existing_limit():

    result = classify_limit(
        -sp.oo,
        sp.oo,
    )

    assert result == "does_not_exist"


def test_analyze_existing_limit():

    x = sp.Symbol("x")

    function = x**2

    result = analyze_limit(
        function,
        2,
    )

    assert result["limit"] == 4
    assert result["left_limit"] == 4
    assert result["right_limit"] == 4
    assert result["type"] == "finite_or_infinite"


def test_analyze_non_existing_limit():

    x = sp.Symbol("x")

    function = sp.Abs(x) / x

    result = analyze_limit(
        function,
        0,
    )

    assert result["type"] == "does_not_exist"


def test_evaluate_function():

    x = sp.Symbol("x")

    function = x**2 + 2*x + 1

    result = evaluate_function(
        function,
        3,
    )

    assert result == 16


def test_invalid_direction():

    x = sp.Symbol("x")

    with pytest.raises(ValueError):

        calculate_limit(
            x**2,
            2,
            direction="invalid",
        )