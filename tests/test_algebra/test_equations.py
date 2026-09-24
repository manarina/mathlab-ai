from math import isclose

from core.algebra.equations import (
    calculate_discriminant,
    solve_linear_equation,
    solve_quadratic_equation,
)


def test_calculate_discriminant():
    delta = calculate_discriminant(1, -5, 6)

    assert delta == 1


def test_linear_equation_unique_solution():
    result = solve_linear_equation(2, -10)

    assert result["type"] == "unique"
    assert len(result["solutions"]) == 1
    assert isclose(result["solutions"][0], 5.0)


def test_linear_equation_infinite_solutions():
    result = solve_linear_equation(0, 0)

    assert result["type"] == "infinite"
    assert result["solutions"] is None


def test_linear_equation_no_solution():
    result = solve_linear_equation(0, 5)

    assert result["type"] == "none"
    assert result["solutions"] == []


def test_quadratic_two_real_solutions():
    result = solve_quadratic_equation(1, -5, 6)

    assert result["degree"] == 2
    assert result["type"] == "two_real_solutions"
    assert result["discriminant"] == 1

    assert isclose(result["solutions"][0], 2.0)
    assert isclose(result["solutions"][1], 3.0)


def test_quadratic_one_real_solution():
    result = solve_quadratic_equation(1, -4, 4)

    assert result["degree"] == 2
    assert result["type"] == "one_real_solution"
    assert result["discriminant"] == 0

    assert isclose(result["solutions"][0], 2.0)


def test_quadratic_no_real_solution():
    result = solve_quadratic_equation(1, 0, 1)

    assert result["degree"] == 2
    assert result["type"] == "no_real_solution"
    assert result["discriminant"] == -4
    assert result["solutions"] == []


def test_quadratic_with_a_equal_zero():
    result = solve_quadratic_equation(0, 2, -10)

    assert result["degree"] == 1
    assert result["type"] == "unique"

    assert isclose(result["solutions"][0], 5.0)