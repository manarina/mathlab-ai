from __future__ import annotations

import math

import pytest

from core.statistics.regression_visualization import (
    regression_line_data,
    regression_plot_data,
    regression_visualization_data,
)


# ============================================================
# regression_visualization_data
# ============================================================


def test_regression_visualization_data_exact_linear_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    result = regression_visualization_data(x, y)

    assert result == [
        {"x": 1.0, "y": 2.0, "predicted_y": 2.0},
        {"x": 2.0, "y": 4.0, "predicted_y": 4.0},
        {"x": 3.0, "y": 6.0, "predicted_y": 6.0},
        {"x": 4.0, "y": 8.0, "predicted_y": 8.0},
        {"x": 5.0, "y": 10.0, "predicted_y": 10.0},
    ]


def test_regression_visualization_data_sorts_by_x():
    x = [3, 1, 2]
    y = [6, 2, 4]

    result = regression_visualization_data(x, y)

    assert [item["x"] for item in result] == [1.0, 2.0, 3.0]
    assert [item["y"] for item in result] == [2.0, 4.0, 6.0]


def test_regression_visualization_data_imperfect_model():
    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]

    result = regression_visualization_data(x, y)

    assert len(result) == 4

    assert result[0]["x"] == 1.0
    assert result[0]["y"] == 2.0
    assert math.isclose(
    result[0]["predicted_y"],
       2.3,
        rel_tol=1e-9,
    )


# ============================================================
# regression_line_data
# ============================================================


def test_regression_line_data_default_number_of_points():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    result = regression_line_data(x, y)

    assert len(result) == 100


def test_regression_line_data_custom_number_of_points():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    result = regression_line_data(
        x,
        y,
        number_of_points=5,
    )

    assert len(result) == 5


def test_regression_line_data_first_and_last_points():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    result = regression_line_data(
        x,
        y,
        number_of_points=5,
    )

    assert result[0] == {
        "x": 1.0,
        "y": 2.0,
    }

    assert result[-1] == {
        "x": 5.0,
        "y": 10.0,
    }


def test_regression_line_data_exact_linear_relationship():
    x = [1, 2, 3]
    y = [3, 5, 7]

    result = regression_line_data(
        x,
        y,
        number_of_points=3,
    )

    assert result == [
        {"x": 1.0, "y": 3.0},
        {"x": 2.0, "y": 5.0},
        {"x": 3.0, "y": 7.0},
    ]


def test_regression_line_data_x_values_are_in_ascending_order():
    x = [5, 1, 4, 2, 3]
    y = [10, 2, 8, 4, 6]

    result = regression_line_data(
        x,
        y,
        number_of_points=10,
    )

    x_values = [item["x"] for item in result]

    assert x_values == sorted(x_values)


# ============================================================
# regression_plot_data
# ============================================================


def test_regression_plot_data_contains_expected_keys():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    result = regression_plot_data(x, y)

    assert set(result.keys()) == {
        "observations",
        "regression_line",
        "slope",
        "intercept",
    }


def test_regression_plot_data_contains_correct_regression_parameters():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    result = regression_plot_data(x, y)

    assert math.isclose(result["slope"], 2.0)
    assert math.isclose(result["intercept"], 0.0)


def test_regression_plot_data_contains_observations():
    x = [1, 2, 3]
    y = [2, 4, 6]

    result = regression_plot_data(
        x,
        y,
        number_of_line_points=10,
    )

    observations = result["observations"]

    assert len(observations) == 3

    assert observations[0] == {
        "x": 1.0,
        "y": 2.0,
        "predicted_y": 2.0,
    }


def test_regression_plot_data_contains_regression_line():
    x = [1, 2, 3]
    y = [2, 4, 6]

    result = regression_plot_data(
        x,
        y,
        number_of_line_points=10,
    )

    regression_line = result["regression_line"]

    assert len(regression_line) == 10
    assert regression_line[0]["x"] == 1.0
    assert regression_line[-1]["x"] == 3.0


def test_regression_plot_data_custom_line_points():
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]

    result = regression_plot_data(
        x,
        y,
        number_of_line_points=20,
    )

    assert len(result["regression_line"]) == 20


# ============================================================
# Validation — données vides
# ============================================================


def test_regression_visualization_data_rejects_empty_x():
    with pytest.raises(ValueError):
        regression_visualization_data([], [1, 2, 3])


def test_regression_visualization_data_rejects_empty_y():
    with pytest.raises(ValueError):
        regression_visualization_data([1, 2, 3], [])


def test_regression_line_data_rejects_empty_x():
    with pytest.raises(ValueError):
        regression_line_data([], [1, 2, 3])


def test_regression_line_data_rejects_empty_y():
    with pytest.raises(ValueError):
        regression_line_data([1, 2, 3], [])


def test_regression_plot_data_rejects_empty_x():
    with pytest.raises(ValueError):
        regression_plot_data([], [1, 2, 3])


def test_regression_plot_data_rejects_empty_y():
    with pytest.raises(ValueError):
        regression_plot_data([1, 2, 3], [])


# ============================================================
# Validation — longueurs différentes
# ============================================================


def test_regression_visualization_data_rejects_different_lengths():
    with pytest.raises(ValueError):
        regression_visualization_data(
            [1, 2, 3],
            [2, 4],
        )


def test_regression_line_data_rejects_different_lengths():
    with pytest.raises(ValueError):
        regression_line_data(
            [1, 2, 3],
            [2, 4],
        )


def test_regression_plot_data_rejects_different_lengths():
    with pytest.raises(ValueError):
        regression_plot_data(
            [1, 2, 3],
            [2, 4],
        )


# ============================================================
# Validation — nombre insuffisant d'observations
# ============================================================


def test_regression_visualization_data_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        regression_visualization_data([1], [2])


def test_regression_line_data_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        regression_line_data([1], [2])


def test_regression_plot_data_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        regression_plot_data([1], [2])


# ============================================================
# Validation — X constant
# ============================================================


def test_regression_visualization_data_rejects_constant_x():
    with pytest.raises(ValueError):
        regression_visualization_data(
            [2, 2, 2],
            [1, 2, 3],
        )


def test_regression_line_data_rejects_constant_x():
    with pytest.raises(ValueError):
        regression_line_data(
            [2, 2, 2],
            [1, 2, 3],
        )


def test_regression_plot_data_rejects_constant_x():
    with pytest.raises(ValueError):
        regression_plot_data(
            [2, 2, 2],
            [1, 2, 3],
        )


# ============================================================
# Validation — types numériques
# ============================================================


def test_regression_visualization_data_rejects_nonnumeric_x():
    with pytest.raises(TypeError):
        regression_visualization_data(
            [1, "2", 3],
            [2, 4, 6],
        )


def test_regression_visualization_data_rejects_nonnumeric_y():
    with pytest.raises(TypeError):
        regression_visualization_data(
            [1, 2, 3],
            [2, "4", 6],
        )


def test_regression_line_data_rejects_nonnumeric_x():
    with pytest.raises(TypeError):
        regression_line_data(
            [1, "2", 3],
            [2, 4, 6],
        )


def test_regression_line_data_rejects_nonnumeric_y():
    with pytest.raises(TypeError):
        regression_line_data(
            [1, 2, 3],
            [2, "4", 6],
        )


def test_regression_plot_data_rejects_nonnumeric_x():
    with pytest.raises(TypeError):
        regression_plot_data(
            [1, "2", 3],
            [2, 4, 6],
        )


def test_regression_plot_data_rejects_nonnumeric_y():
    with pytest.raises(TypeError):
        regression_plot_data(
            [1, 2, 3],
            [2, "4", 6],
        )


# ============================================================
# Validation — booléens
# ============================================================


def test_regression_visualization_data_rejects_boolean_x():
    with pytest.raises(TypeError):
        regression_visualization_data(
            [1, True, 3],
            [2, 4, 6],
        )


def test_regression_visualization_data_rejects_boolean_y():
    with pytest.raises(TypeError):
        regression_visualization_data(
            [1, 2, 3],
            [2, True, 6],
        )


def test_regression_line_data_rejects_boolean_x():
    with pytest.raises(TypeError):
        regression_line_data(
            [1, False, 3],
            [2, 4, 6],
        )


def test_regression_line_data_rejects_boolean_y():
    with pytest.raises(TypeError):
        regression_line_data(
            [1, 2, 3],
            [2, False, 6],
        )


# ============================================================
# Validation — valeurs non finies
# ============================================================


def test_regression_visualization_data_rejects_nan_x():
    with pytest.raises(ValueError):
        regression_visualization_data(
            [1, math.nan, 3],
            [2, 4, 6],
        )


def test_regression_visualization_data_rejects_infinite_y():
    with pytest.raises(ValueError):
        regression_visualization_data(
            [1, 2, 3],
            [2, math.inf, 6],
        )


def test_regression_line_data_rejects_nan_x():
    with pytest.raises(ValueError):
        regression_line_data(
            [1, math.nan, 3],
            [2, 4, 6],
        )


def test_regression_line_data_rejects_infinite_y():
    with pytest.raises(ValueError):
        regression_line_data(
            [1, 2, 3],
            [2, math.inf, 6],
        )


def test_regression_plot_data_rejects_nan_x():
    with pytest.raises(ValueError):
        regression_plot_data(
            [1, math.nan, 3],
            [2, 4, 6],
        )


def test_regression_plot_data_rejects_infinite_y():
    with pytest.raises(ValueError):
        regression_plot_data(
            [1, 2, 3],
            [2, math.inf, 6],
        )


# ============================================================
# Validation — nombre de points de la droite
# ============================================================


def test_regression_line_data_rejects_non_integer_number_of_points():
    with pytest.raises(TypeError):
        regression_line_data(
            [1, 2, 3],
            [2, 4, 6],
            number_of_points=10.5,
        )


def test_regression_line_data_rejects_string_number_of_points():
    with pytest.raises(TypeError):
        regression_line_data(
            [1, 2, 3],
            [2, 4, 6],
            number_of_points="10",
        )


def test_regression_line_data_rejects_less_than_two_points():
    with pytest.raises(ValueError):
        regression_line_data(
            [1, 2, 3],
            [2, 4, 6],
            number_of_points=1,
        )


def test_regression_plot_data_rejects_non_integer_number_of_points():
    with pytest.raises(TypeError):
        regression_plot_data(
            [1, 2, 3],
            [2, 4, 6],
            number_of_line_points=10.5,
        )


def test_regression_plot_data_rejects_less_than_two_points():
    with pytest.raises(ValueError):
        regression_plot_data(
            [1, 2, 3],
            [2, 4, 6],
            number_of_line_points=1,
        )