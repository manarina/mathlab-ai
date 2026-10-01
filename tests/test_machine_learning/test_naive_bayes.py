
"""
Tests du module Gaussian Naive Bayes.

Fichier :
    tests/test_machine_learning/test_naive_bayes.py
"""

import numpy as np
import pytest

from core.machine_learning.naive_bayes import (
    _validate_input,
    _validate_model_parameters,
    accuracy_score,
    fit_naive_bayes,
    gaussian_probability,
    naive_bayes,
    predict_naive_bayes,
    predict_proba_naive_bayes,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def simple_dataset():
    """
    Petit dataset binaire facilement séparable.
    """

    X_train = np.array(
        [
            [1.0, 1.0],
            [1.2, 1.1],
            [0.8, 0.9],
            [1.1, 1.2],
            [5.0, 5.0],
            [5.2, 5.1],
            [4.8, 4.9],
            [5.1, 5.2],
        ]
    )

    y_train = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    X_test = np.array(
        [
            [1.0, 1.0],
            [5.0, 5.0],
            [1.1, 0.9],
            [4.9, 5.1],
        ]
    )

    y_test = np.array(
        [0, 1, 0, 1]
    )

    return X_train, y_train, X_test, y_test


@pytest.fixture
def multiclass_dataset():
    """
    Dataset simple à trois classes.
    """

    X_train = np.array(
        [
            [1.0, 1.0],
            [1.2, 1.1],
            [0.9, 1.0],
            [5.0, 5.0],
            [5.2, 5.1],
            [4.9, 5.2],
            [9.0, 1.0],
            [9.2, 1.1],
            [8.9, 0.9],
        ]
    )

    y_train = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2]
    )

    X_test = np.array(
        [
            [1.1, 1.0],
            [5.1, 5.0],
            [9.1, 1.0],
        ]
    )

    y_test = np.array(
        [0, 1, 2]
    )

    return X_train, y_train, X_test, y_test


# ============================================================
# _validate_input
# ============================================================


def test_validate_input_returns_arrays():
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    X_result, y_result = _validate_input(
        X,
        y,
    )

    assert isinstance(X_result, np.ndarray)
    assert isinstance(y_result, np.ndarray)

    assert X_result.shape == (2, 2)
    assert y_result.shape == (2,)


def test_validate_input_without_y():
    X = [[1, 2], [3, 4]]

    X_result, y_result = _validate_input(X)

    assert isinstance(X_result, np.ndarray)
    assert y_result is None


def test_validate_input_rejects_1d_X():
    X = [1, 2, 3]

    with pytest.raises(ValueError):
        _validate_input(X)


def test_validate_input_rejects_empty_X():
    X = np.empty((0, 2))

    with pytest.raises(ValueError):
        _validate_input(X)


def test_validate_input_rejects_zero_features():
    X = np.empty((3, 0))

    with pytest.raises(ValueError):
        _validate_input(X)


def test_validate_input_rejects_non_finite_values():
    X = np.array(
        [
            [1.0, 2.0],
            [np.nan, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        _validate_input(X)


def test_validate_input_rejects_wrong_y_length():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    y = np.array([0])

    with pytest.raises(ValueError):
        _validate_input(X, y)


def test_validate_input_rejects_multidimensional_y():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    y = np.array(
        [
            [0],
            [1],
        ]
    )

    with pytest.raises(ValueError):
        _validate_input(X, y)


# ============================================================
# gaussian_probability
# ============================================================


def test_gaussian_probability_at_mean():
    """
    À x = μ, la densité doit être :

        1 / sqrt(2πσ²)
    """

    result = gaussian_probability(
        np.array([0.0]),
        mean=0.0,
        variance=1.0,
    )

    expected = 1.0 / np.sqrt(2.0 * np.pi)

    assert result[0] == pytest.approx(
        expected
    )


def test_gaussian_probability_is_symmetric():
    left = gaussian_probability(
        np.array([-1.0]),
        mean=0.0,
        variance=1.0,
    )

    right = gaussian_probability(
        np.array([1.0]),
        mean=0.0,
        variance=1.0,
    )

    assert left[0] == pytest.approx(
        right[0]
    )


def test_gaussian_probability_returns_array():
    result = gaussian_probability(
        [0.0, 1.0, 2.0],
        mean=0.0,
        variance=1.0,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_gaussian_probability_values_are_positive():
    result = gaussian_probability(
        np.array(
            [-2.0, -1.0, 0.0, 1.0, 2.0]
        ),
        mean=0.0,
        variance=1.0,
    )

    assert np.all(result > 0)


def test_gaussian_probability_rejects_zero_variance():
    with pytest.raises(ValueError):
        gaussian_probability(
            [1.0],
            mean=0.0,
            variance=0.0,
        )


def test_gaussian_probability_rejects_negative_variance():
    with pytest.raises(ValueError):
        gaussian_probability(
            [1.0],
            mean=0.0,
            variance=-1.0,
        )


def test_gaussian_probability_rejects_non_finite_mean():
    with pytest.raises(ValueError):
        gaussian_probability(
            [1.0],
            mean=np.nan,
            variance=1.0,
        )


def test_gaussian_probability_rejects_non_finite_variance():
    with pytest.raises(ValueError):
        gaussian_probability(
            [1.0],
            mean=0.0,
            variance=np.inf,
        )


# ============================================================
# fit_naive_bayes
# ============================================================


def test_fit_naive_bayes_returns_expected_keys(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    expected_keys = {
        "classes",
        "priors",
        "means",
        "variances",
        "variance_smoothing",
        "n_features",
        "n_classes",
        "n_samples",
    }

    assert expected_keys.issubset(
        model.keys()
    )


def test_fit_naive_bayes_detects_classes(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    assert np.array_equal(
        model["classes"],
        np.array([0, 1]),
    )


def test_fit_naive_bayes_calculates_priors(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    assert np.allclose(
        model["priors"],
        np.array([0.5, 0.5]),
    )


def test_fit_naive_bayes_priors_sum_to_one(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    assert np.sum(
        model["priors"]
    ) == pytest.approx(1.0)


def test_fit_naive_bayes_shapes(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    assert model["means"].shape == (2, 2)
    assert model["variances"].shape == (2, 2)
    assert model["priors"].shape == (2,)


def test_fit_naive_bayes_means_are_correct(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    expected_class_0_mean = np.mean(
        X_train[y_train == 0],
        axis=0,
    )

    expected_class_1_mean = np.mean(
        X_train[y_train == 1],
        axis=0,
    )

    assert np.allclose(
        model["means"][0],
        expected_class_0_mean,
    )

    assert np.allclose(
        model["means"][1],
        expected_class_1_mean,
    )


def test_fit_naive_bayes_variances_are_positive(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    assert np.all(
        model["variances"] > 0
    )


def test_fit_naive_bayes_adds_variance_smoothing():
    X_train = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [5.0, 6.0],
            [5.0, 6.0],
        ]
    )

    y_train = np.array(
        [0, 0, 1, 1]
    )

    model = fit_naive_bayes(
        X_train,
        y_train,
        variance_smoothing=1e-6,
    )

    assert np.all(
        model["variances"] > 0
    )


def test_fit_naive_bayes_rejects_single_class():
    X_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    y_train = np.array(
        [0, 0, 0]
    )

    with pytest.raises(ValueError):
        fit_naive_bayes(
            X_train,
            y_train,
        )


def test_fit_naive_bayes_rejects_invalid_smoothing():
    X_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ]
    )

    y_train = np.array(
        [0, 0, 1, 1]
    )

    with pytest.raises(ValueError):
        fit_naive_bayes(
            X_train,
            y_train,
            variance_smoothing=0,
        )


def test_fit_naive_bayes_rejects_negative_smoothing():
    X_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ]
    )

    y_train = np.array(
        [0, 0, 1, 1]
    )

    with pytest.raises(ValueError):
        fit_naive_bayes(
            X_train,
            y_train,
            variance_smoothing=-1e-9,
        )


# ============================================================
# _validate_model_parameters
# ============================================================


def test_validate_model_parameters_accepts_valid_model():
    classes = np.array([0, 1])

    priors = np.array(
        [0.5, 0.5]
    )

    means = np.array(
        [
            [1.0, 2.0],
            [5.0, 6.0],
        ]
    )

    variances = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    _validate_model_parameters(
        means,
        variances,
        priors,
        classes,
    )


def test_validate_model_parameters_rejects_wrong_means_dimension():
    with pytest.raises(ValueError):
        _validate_model_parameters(
            np.array([1.0, 2.0]),
            np.ones((2, 2)),
            np.array([0.5, 0.5]),
            np.array([0, 1]),
        )


def test_validate_model_parameters_rejects_negative_variance():
    with pytest.raises(ValueError):
        _validate_model_parameters(
            np.ones((2, 2)),
            np.array(
                [
                    [1.0, -1.0],
                    [1.0, 1.0],
                ]
            ),
            np.array([0.5, 0.5]),
            np.array([0, 1]),
        )


def test_validate_model_parameters_rejects_invalid_priors():
    with pytest.raises(ValueError):
        _validate_model_parameters(
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.array([0.8, 0.8]),
            np.array([0, 1]),
        )


# ============================================================
# predict_proba_naive_bayes
# ============================================================


def test_predict_proba_shape(
    simple_dataset,
):
    X_train, y_train, X_test, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    probabilities = predict_proba_naive_bayes(
        X_test,
        model,
    )

    assert probabilities.shape == (
        X_test.shape[0],
        2,
    )


def test_predict_proba_sums_to_one(
    simple_dataset,
):
    X_train, y_train, X_test, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    probabilities = predict_proba_naive_bayes(
        X_test,
        model,
    )

    assert np.allclose(
        np.sum(probabilities, axis=1),
        1.0,
    )


def test_predict_proba_values_between_zero_and_one(
    simple_dataset,
):
    X_train, y_train, X_test, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    probabilities = predict_proba_naive_bayes(
        X_test,
        model,
    )

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_proba_classifies_simple_points(
    simple_dataset,
):
    X_train, y_train, X_test, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    probabilities = predict_proba_naive_bayes(
        X_test,
        model,
    )

    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]


def test_predict_proba_rejects_wrong_feature_count(
    simple_dataset,
):
    X_train, y_train, _, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    X_invalid = np.array(
        [
            [1.0, 2.0, 3.0]
        ]
    )

    with pytest.raises(ValueError):
        predict_proba_naive_bayes(
            X_invalid,
            model,
        )


def test_predict_proba_rejects_incomplete_model(
    simple_dataset,
):
    _, _, X_test, _ = simple_dataset

    incomplete_model = {
        "classes": np.array([0, 1])
    }

    with pytest.raises(ValueError):
        predict_proba_naive_bayes(
            X_test,
            incomplete_model,
        )


# ============================================================
# predict_naive_bayes
# ============================================================


def test_predict_returns_expected_classes(
    simple_dataset,
):
    X_train, y_train, X_test, y_test = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    predictions = predict_naive_bayes(
        X_test,
        model,
    )

    assert np.array_equal(
        predictions,
        y_test,
    )


def test_predict_returns_numpy_array(
    simple_dataset,
):
    X_train, y_train, X_test, _ = simple_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    predictions = predict_naive_bayes(
        X_test,
        model,
    )

    assert isinstance(
        predictions,
        np.ndarray,
    )


def test_predict_multiclass(
    multiclass_dataset,
):
    X_train, y_train, X_test, y_test = multiclass_dataset

    model = fit_naive_bayes(
        X_train,
        y_train,
    )

    predictions = predict_naive_bayes(
        X_test,
        model,
    )

    assert np.array_equal(
        predictions,
        y_test,
    )


# ============================================================
# accuracy_score
# ============================================================


def test_accuracy_perfect():
    y_true = np.array(
        [0, 1, 0, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1]
    )

    assert accuracy_score(
        y_true,
        y_pred,
    ) == pytest.approx(1.0)


def test_accuracy_zero():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [1, 1, 0, 0]
    )

    assert accuracy_score(
        y_true,
        y_pred,
    ) == pytest.approx(0.0)


def test_accuracy_partial():
    y_true = np.array(
        [0, 1, 0, 1]
    )

    y_pred = np.array(
        [0, 1, 1, 0]
    )

    assert accuracy_score(
        y_true,
        y_pred,
    ) == pytest.approx(0.5)


def test_accuracy_rejects_empty_true_labels():
    with pytest.raises(ValueError):
        accuracy_score(
            [],
            [],
        )


def test_accuracy_rejects_different_lengths():
    with pytest.raises(ValueError):
        accuracy_score(
            [0, 1],
            [0],
        )


def test_accuracy_rejects_multidimensional_true_labels():
    with pytest.raises(ValueError):
        accuracy_score(
            [[0], [1]],
            [0, 1],
        )


# ============================================================
# PIPELINE naive_bayes
# ============================================================


def test_naive_bayes_returns_expected_keys(
    simple_dataset,
):
    X_train, y_train, X_test, y_test = simple_dataset

    result = naive_bayes(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    expected_keys = {
        "model",
        "predictions",
        "probabilities",
        "classes",
        "priors",
        "means",
        "variances",
        "accuracy",
        "variance_smoothing",
        "n_train_samples",
        "n_test_samples",
        "n_features",
        "n_classes",
    }

    assert expected_keys.issubset(
        result.keys()
    )


def test_naive_bayes_predictions(
    simple_dataset,
):
    X_train, y_train, X_test, y_test = simple_dataset

    result = naive_bayes(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert np.array_equal(
        result["predictions"],
        y_test,
    )


def test_naive_bayes_accuracy(
    simple_dataset,
):
    X_train, y_train, X_test, y_test = simple_dataset

    result = naive_bayes(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert result["accuracy"] == pytest.approx(
        1.0
    )


def test_naive_bayes_accuracy_is_none_without_y_test(
    simple_dataset,
):
    X_train, y_train, X_test, _ = simple_dataset

    result = naive_bayes(
        X_train,
        y_train,
        X_test,
    )

    assert result["accuracy"] is None


def test_naive_bayes_probability_shape(
    simple_dataset,
):
    X_train, y_train, X_test, y_test = simple_dataset

    result = naive_bayes(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert result["probabilities"].shape == (
        X_test.shape[0],
        2,
    )


def test_naive_bayes_multiclass(
    multiclass_dataset,
):
    X_train, y_train, X_test, y_test = multiclass_dataset

    result = naive_bayes(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert result["n_classes"] == 3

    assert np.array_equal(
        result["predictions"],
        y_test,
    )

    assert result["accuracy"] == pytest.approx(
        1.0
    )


def test_naive_bayes_metadata(
    simple_dataset,
):
    X_train, y_train, X_test, y_test = simple_dataset

    result = naive_bayes(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert result["n_train_samples"] == 8
    assert result["n_test_samples"] == 4
    assert result["n_features"] == 2
    assert result["n_classes"] == 2


def test_naive_bayes_rejects_different_feature_counts():
    X_train = np.array(
        [
            [1.0, 2.0],
            [1.2, 2.1],
            [5.0, 6.0],
            [5.2, 6.1],
        ]
    )

    y_train = np.array(
        [0, 0, 1, 1]
    )

    X_test = np.array(
        [
            [1.0, 2.0, 3.0]
        ]
    )

    with pytest.raises(ValueError):
        naive_bayes(
            X_train,
            y_train,
            X_test,
        )

