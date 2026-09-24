import pytest

from core.linear_algebra.vectors import (
    Vector,
    add_vectors,
    create_vector,
    dot_product,
    scalar_multiply,
    subtract_vectors,
    vector_distance,
    vector_norm,
)


def test_create_vector():
    vector = create_vector([1, 2, 3])

    assert vector.values == (1, 2, 3)
    assert vector.dimension == 3


def test_create_vector_rejects_empty_vector():
    with pytest.raises(ValueError):
        create_vector([])


def test_create_vector_rejects_non_numeric_values():
    with pytest.raises(TypeError):
        create_vector([1, "2", 3])


def test_vector_dimension():
    vector = Vector((10, 20, 30, 40))

    assert vector.dimension == 4


def test_vector_addition():
    first = Vector((1, 2, 3))
    second = Vector((4, 5, 6))

    result = first + second

    assert result.values == (5, 7, 9)


def test_add_vectors_function():
    first = Vector((1, 2))
    second = Vector((3, 4))

    result = add_vectors(first, second)

    assert result.values == (4, 6)


def test_vector_subtraction():
    first = Vector((5, 7, 9))
    second = Vector((1, 2, 3))

    result = first - second

    assert result.values == (4, 5, 6)


def test_subtract_vectors_function():
    first = Vector((5, 7))
    second = Vector((2, 3))

    result = subtract_vectors(first, second)

    assert result.values == (3, 4)


def test_scalar_multiplication():
    vector = Vector((1, 2, 3))

    result = vector * 2

    assert result.values == (2, 4, 6)


def test_reverse_scalar_multiplication():
    vector = Vector((1, 2, 3))

    result = 3 * vector

    assert result.values == (3, 6, 9)


def test_scalar_multiply_function():
    vector = Vector((2, 4, 6))

    result = scalar_multiply(vector, 0.5)

    assert result.values == (1.0, 2.0, 3.0)


def test_dot_product():
    first = Vector((1, 2, 3))
    second = Vector((4, 5, 6))

    result = first.dot(second)

    assert result == 32


def test_dot_product_function():
    first = Vector((1, 2, 3))
    second = Vector((4, 5, 6))

    result = dot_product(first, second)

    assert result == 32


def test_vector_norm():
    vector = Vector((3, 4))

    result = vector.norm()

    assert result == pytest.approx(5.0)


def test_vector_norm_function():
    vector = Vector((3, 4))

    result = vector_norm(vector)

    assert result == pytest.approx(5.0)


def test_vector_distance():
    first = Vector((1, 2))
    second = Vector((4, 6))

    result = first.distance_to(second)

    assert result == pytest.approx(5.0)


def test_vector_distance_function():
    first = Vector((1, 2))
    second = Vector((4, 6))

    result = vector_distance(first, second)

    assert result == pytest.approx(5.0)


def test_vectors_must_have_same_dimension_for_addition():
    first = Vector((1, 2))
    second = Vector((3, 4, 5))

    with pytest.raises(ValueError):
        first + second


def test_vectors_must_have_same_dimension_for_subtraction():
    first = Vector((1, 2))
    second = Vector((3, 4, 5))

    with pytest.raises(ValueError):
        first - second


def test_vectors_must_have_same_dimension_for_dot_product():
    first = Vector((1, 2))
    second = Vector((3, 4, 5))

    with pytest.raises(ValueError):
        first.dot(second)


def test_vectors_must_have_same_dimension_for_distance():
    first = Vector((1, 2))
    second = Vector((3, 4, 5))

    with pytest.raises(ValueError):
        first.distance_to(second)


def test_scalar_multiplication_rejects_non_numeric_scalar():
    vector = Vector((1, 2, 3))

    with pytest.raises(TypeError):
        vector * "2"


def test_to_tuple():
    vector = Vector((1, 2, 3))

    assert vector.to_tuple() == (1, 2, 3)