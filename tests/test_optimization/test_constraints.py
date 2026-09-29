
"""
Tests du module core.optimization.constraints.
"""

import math

import pytest

from core.optimization.constraints import (
    bounds_width,
    clip_to_bounds,
    is_within_bounds,
    validate_bounds,
)


# ============================================================
# validate_bounds — cas normaux
# ============================================================


def test_validate_bounds_returns_valid_bounds():
    result = validate_bounds(0, 10)

    assert result == (0.0, 10.0)


def test_validate_bounds_returns_floats():
    lower, upper = validate_bounds(1, 5)

    assert isinstance(lower, float)
    assert isinstance(upper, float)


def test_validate_bounds_accepts_negative_values():
    result = validate_bounds(-5, 5)

    assert result == (-5.0, 5.0)


def test_validate_bounds_accepts_float_values():
    result = validate_bounds(0.5, 2.5)

    assert result == (0.5, 2.5)


# ============================================================
# validate_bounds — erreurs
# ============================================================


def test_validate_bounds_rejects_equal_bounds():
    with pytest.raises(ValueError, match="strictement inférieure"):
        validate_bounds(2, 2)


def test_validate_bounds_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="strictement inférieure"):
        validate_bounds(5, 2)


def test_validate_bounds_rejects_non_numeric_lower_bound():
    with pytest.raises(TypeError, match="borne inférieure"):
        validate_bounds("0", 10)


def test_validate_bounds_rejects_non_numeric_upper_bound():
    with pytest.raises(TypeError, match="borne supérieure"):
        validate_bounds(0, "10")


def test_validate_bounds_rejects_boolean_lower_bound():
    with pytest.raises(TypeError, match="borne inférieure"):
        validate_bounds(True, 10)


def test_validate_bounds_rejects_boolean_upper_bound():
    with pytest.raises(TypeError, match="borne supérieure"):
        validate_bounds(0, False)


def test_validate_bounds_rejects_nan_lower_bound():
    with pytest.raises(ValueError, match="finie"):
        validate_bounds(math.nan, 10)


def test_validate_bounds_rejects_nan_upper_bound():
    with pytest.raises(ValueError, match="finie"):
        validate_bounds(0, math.nan)


def test_validate_bounds_rejects_infinite_lower_bound():
    with pytest.raises(ValueError, match="finie"):
        validate_bounds(-math.inf, 10)


def test_validate_bounds_rejects_infinite_upper_bound():
    with pytest.raises(ValueError, match="finie"):
        validate_bounds(0, math.inf)


# ============================================================
# is_within_bounds — cas normaux
# ============================================================


def test_is_within_bounds_returns_true_inside_interval():
    assert is_within_bounds(5, 0, 10) is True


def test_is_within_bounds_returns_true_at_lower_bound():
    assert is_within_bounds(0, 0, 10) is True


def test_is_within_bounds_returns_true_at_upper_bound():
    assert is_within_bounds(10, 0, 10) is True


def test_is_within_bounds_returns_false_below_interval():
    assert is_within_bounds(-1, 0, 10) is False


def test_is_within_bounds_returns_false_above_interval():
    assert is_within_bounds(11, 0, 10) is False


def test_is_within_bounds_accepts_float():
    assert is_within_bounds(2.5, 0, 5) is True


def test_is_within_bounds_supports_negative_interval():
    assert is_within_bounds(-3, -5, -1) is True


# ============================================================
# is_within_bounds — erreurs
# ============================================================


def test_is_within_bounds_rejects_boolean_x():
    with pytest.raises(TypeError, match="numérique"):
        is_within_bounds(True, 0, 10)


def test_is_within_bounds_rejects_non_numeric_x():
    with pytest.raises(TypeError, match="numérique"):
        is_within_bounds("5", 0, 10)


def test_is_within_bounds_rejects_nan_x():
    with pytest.raises(ValueError, match="finie"):
        is_within_bounds(math.nan, 0, 10)


def test_is_within_bounds_rejects_infinite_x():
    with pytest.raises(ValueError, match="finie"):
        is_within_bounds(math.inf, 0, 10)


# ============================================================
# clip_to_bounds — cas normaux
# ============================================================


def test_clip_to_bounds_keeps_value_inside_interval():
    assert clip_to_bounds(5, 0, 10) == 5.0


def test_clip_to_bounds_returns_lower_bound_below_interval():
    assert clip_to_bounds(-5, 0, 10) == 0.0


def test_clip_to_bounds_returns_upper_bound_above_interval():
    assert clip_to_bounds(15, 0, 10) == 10.0


def test_clip_to_bounds_keeps_lower_bound():
    assert clip_to_bounds(0, 0, 10) == 0.0


def test_clip_to_bounds_keeps_upper_bound():
    assert clip_to_bounds(10, 0, 10) == 10.0


def test_clip_to_bounds_supports_negative_interval():
    assert clip_to_bounds(-10, -5, -1) == -5.0


def test_clip_to_bounds_returns_float():
    result = clip_to_bounds(5, 0, 10)

    assert isinstance(result, float)


# ============================================================
# clip_to_bounds — erreurs
# ============================================================


def test_clip_to_bounds_rejects_boolean_x():
    with pytest.raises(TypeError, match="numérique"):
        clip_to_bounds(True, 0, 10)


def test_clip_to_bounds_rejects_non_numeric_x():
    with pytest.raises(TypeError, match="numérique"):
        clip_to_bounds("5", 0, 10)


def test_clip_to_bounds_rejects_nan_x():
    with pytest.raises(ValueError, match="finie"):
        clip_to_bounds(math.nan, 0, 10)


def test_clip_to_bounds_rejects_infinite_x():
    with pytest.raises(ValueError, match="finie"):
        clip_to_bounds(math.inf, 0, 10)


# ============================================================
# bounds_width
# ============================================================


def test_bounds_width_returns_correct_width():
    assert bounds_width(0, 10) == 10.0


def test_bounds_width_supports_negative_bounds():
    assert bounds_width(-5, 5) == 10.0


def test_bounds_width_supports_float_bounds():
    assert bounds_width(0.5, 2.5) == 2.0


def test_bounds_width_returns_float():
    result = bounds_width(0, 10)

    assert isinstance(result, float)


def test_bounds_width_rejects_equal_bounds():
    with pytest.raises(ValueError, match="strictement inférieure"):
        bounds_width(5, 5)


def test_bounds_width_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="strictement inférieure"):
        bounds_width(10, 5)

