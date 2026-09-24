import sympy as sp

from core.algebra.cubic_equations import (
    parse_cubic_equation,
    get_cubic_discriminant,
    get_cardano_parameters,
    solve_cubic_equation,
    solve_cubic_with_horner,
)


def test_parse_cubic_equation():
    polynomial = parse_cubic_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    assert polynomial.degree() == 3


def test_cubic_discriminant():
    discriminant = get_cubic_discriminant(
        1,
        -6,
        11,
        -6,
    )

    assert discriminant == 4


def test_cardano_parameters():
    result = get_cardano_parameters(
        1,
        -6,
        11,
        -6,
    )

    assert result["A"] == -6
    assert result["B"] == 11
    assert result["C"] == -6

    assert sp.simplify(
        result["p"] + 1
    ) == 0

    assert sp.simplify(
        result["q"]
    ) == 0


def test_cubic_horner():
    polynomial = parse_cubic_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_cubic_with_horner(
        polynomial
    )

    assert set(
        result["rational_roots"]
    ) == {
        sp.Integer(1),
        sp.Integer(2),
        sp.Integer(3),
    }


def test_cubic_three_real_roots():
    result = solve_cubic_equation(
        1,
        -6,
        11,
        -6,
    )

    assert result["type"] == (
        "three_distinct_real_roots"
    )

    assert set(
        result["roots"]
    ) == {
        sp.Integer(1),
        sp.Integer(2),
        sp.Integer(3),
    }


def test_cubic_multiple_root():
    result = solve_cubic_equation(
        1,
        -3,
        3,
        -1,
    )

    assert result["type"] == (
        "multiple_real_roots"
    )

    assert result["roots"] == [
        sp.Integer(1)
    ]


def test_cubic_one_real_root():
    result = solve_cubic_equation(
        1,
        0,
        0,
        -1,
    )

    assert result["type"] == (
        "one_real_root_and_two_complex_roots"
    )

    assert sp.Integer(1) in result["roots"]
    assert len(result["roots"]) == 3


def test_invalid_cubic_equation():
    try:
        parse_cubic_equation(
            "x**2 + 2*x + 1"
        )
    except ValueError:
        assert True
    else:
        assert False