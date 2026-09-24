import pytest

from core.linear_algebra.vectors import Vector
from core.linear_algebra.vector_explanations import (
    explain_addition,
    explain_distance,
    explain_dot_product,
    explain_norm,
    explain_scalar_multiplication,
    explain_subtraction,
    explain_vector,
)


def test_explain_vector():
    vector = Vector((1, 2, 3))

    explanation = explain_vector(vector)

    assert "dimension 3" in explanation
    assert "(1, 2, 3)" in explanation


def test_explain_addition():
    first = Vector((1, 2, 3))
    second = Vector((4, 5, 6))
    result = first + second

    explanation = explain_addition(
        first,
        second,
        result,
    )

    assert "Addition de vecteurs" in explanation
    assert "1 + 4 = 5" in explanation
    assert "2 + 5 = 7" in explanation
    assert "3 + 6 = 9" in explanation
    assert "u + v" in explanation


def test_explain_subtraction():
    first = Vector((5, 7, 9))
    second = Vector((1, 2, 3))
    result = first - second

    explanation = explain_subtraction(
        first,
        second,
        result,
    )

    assert "Soustraction de vecteurs" in explanation
    assert "5 - 1 = 4" in explanation
    assert "7 - 2 = 5" in explanation
    assert "9 - 3 = 6" in explanation
    assert "u - v" in explanation


def test_explain_scalar_multiplication():
    vector = Vector((1, 2, 3))
    scalar = 2
    result = vector * scalar

    explanation = explain_scalar_multiplication(
        vector,
        scalar,
        result,
    )

    assert "Multiplication par un scalaire" in explanation
    assert "2 × 1 = 2" in explanation
    assert "2 × 2 = 4" in explanation
    assert "2 × 3 = 6" in explanation
    assert "2u" in explanation


def test_explain_dot_product():
    first = Vector((1, 2, 3))
    second = Vector((4, 5, 6))
    result = first.dot(second)

    explanation = explain_dot_product(
        first,
        second,
        result,
    )

    assert "Produit scalaire" in explanation
    assert "1 × 4" in explanation
    assert "2 × 5" in explanation
    assert "3 × 6" in explanation
    assert "4 + 10 + 18" in explanation
    assert "32" in explanation


def test_explain_norm():
    vector = Vector((3, 4))
    result = vector.norm()

    explanation = explain_norm(
        vector,
        result,
    )

    assert "Norme du vecteur" in explanation
    assert "3²" in explanation
    assert "4²" in explanation
    assert "9 + 16" in explanation
    assert "5.000000" in explanation


def test_explain_distance():
    first = Vector((1, 2))
    second = Vector((4, 6))
    result = first.distance_to(second)

    explanation = explain_distance(
        first,
        second,
        result,
    )

    assert "Distance entre deux vecteurs" in explanation
    assert "u - v" in explanation
    assert "(-3, -4)" in explanation
    assert "5.000000" in explanation


def test_explain_addition_rejects_different_dimensions():
    first = Vector((1, 2))
    second = Vector((3, 4, 5))
    result = Vector((4, 6))

    with pytest.raises(ValueError):
        explain_addition(
            first,
            second,
            result,
        )


def test_explain_subtraction_rejects_different_dimensions():
    first = Vector((5, 7))
    second = Vector((1, 2, 3))
    result = Vector((4, 5))

    with pytest.raises(ValueError):
        explain_subtraction(
            first,
            second,
            result,
        )


def test_explain_dot_product_rejects_different_dimensions():
    first = Vector((1, 2))
    second = Vector((3, 4, 5))

    with pytest.raises(ValueError):
        explain_dot_product(
            first,
            second,
            0,
        )


def test_explain_distance_rejects_different_dimensions():
    first = Vector((1, 2))
    second = Vector((3, 4, 5))

    with pytest.raises(ValueError):
        explain_distance(
            first,
            second,
            0,
        )