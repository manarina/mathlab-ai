from core.algebra.horner import (
    horner_evaluate,
    horner_division,
    find_rational_root_candidates,
)


def test_horner_evaluate_cubic():
    coefficients = [1, -6, 11, -6]

    result = horner_evaluate(
        coefficients,
        1,
    )

    assert result == 0


def test_horner_evaluate_non_root():
    coefficients = [1, -6, 11, -6]

    result = horner_evaluate(
        coefficients,
        0,
    )

    assert result == -6


def test_horner_division():
    coefficients = [1, -6, 11, -6]

    result = horner_division(
        coefficients,
        1,
    )

    assert result["quotient"] == [
        1,
        -5,
        6,
    ]

    assert result["remainder"] == 0
    assert result["is_root"] is True


def test_horner_division_non_root():
    coefficients = [1, -6, 11, -6]

    result = horner_division(
        coefficients,
        2,
    )

    assert result["remainder"] == 0


def test_rational_root_candidates():
    candidates = find_rational_root_candidates(
        [1, -6, 11, -6]
    )

    assert 1.0 in candidates
    assert 2.0 in candidates
    assert 3.0 in candidates
    assert 6.0 in candidates