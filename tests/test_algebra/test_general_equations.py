from core.algebra.general_equations import (
    parse_polynomial_equation,
    get_equation_degree,
    select_solution_method,
    solve_polynomial_equation,
)


def test_parse_polynomial_equation():
    polynomial = parse_polynomial_equation(
        "x**5 - 3*x + 1"
    )

    assert polynomial.degree() == 5


def test_equation_degree():
    polynomial = parse_polynomial_equation(
        "x**4 - 2*x**2 + 1"
    )

    assert get_equation_degree(
        polynomial
    ) == 4


def test_method_degree_one():
    assert (
        select_solution_method(1)
        == "linear"
    )


def test_method_degree_two():
    assert (
        select_solution_method(2)
        == "quadratic"
    )


def test_method_degree_three():
    assert (
        select_solution_method(3)
        == "horner_cardano"
    )


def test_method_degree_four():
    assert (
        select_solution_method(4)
        == "quartic_symbolic"
    )


def test_method_degree_five():
    assert (
        select_solution_method(5)
        == "factorization_and_numerical"
    )


def test_general_polynomial():
    polynomial = parse_polynomial_equation(
        "x**3 - 6*x**2 + 11*x - 6"
    )

    result = solve_polynomial_equation(
        polynomial
    )

    assert result["degree"] == 3
    assert result["method"] == "horner_cardano"

    assert set(
        result["exact_roots"]
    ) == {
        1,
        2,
        3,
    }