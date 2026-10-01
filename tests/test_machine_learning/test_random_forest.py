
"""
Tests - Random Forest
=====================

Tests unitaires de l'implémentation pédagogique
du Random Forest pour la classification.
"""

import numpy as np
import pytest

from core.machine_learning.random_forest import (
    _default_max_features,
    _validate_input,
    bootstrap_sample,
    majority_vote,
    predict_forest,
    random_forest,
    select_features,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================


@pytest.fixture
def binary_dataset():
    """
    Jeu de données simple pour classification binaire.
    """

    X_train = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [6.0, 6.0],
        ]
    )

    y_train = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    X_test = np.array(
        [
            [1.5, 1.5],
            [5.5, 5.5],
        ]
    )

    y_test = np.array([0, 1])

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


@pytest.fixture
def multiclass_dataset():
    """
    Jeu de données simple à trois classes.
    """

    X_train = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [9.0, 9.0],
            [9.0, 10.0],
            [10.0, 9.0],
        ]
    )

    y_train = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
        ]
    )

    X_test = np.array(
        [
            [1.5, 1.5],
            [5.5, 5.5],
            [9.5, 9.5],
        ]
    )

    y_test = np.array(
        [0, 1, 2]
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


# ============================================================
# VALIDATION
# ============================================================


def test_validate_input_returns_numpy_arrays():
    X, y = _validate_input(
        [[1, 2], [3, 4]],
        [0, 1],
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    assert X.shape == (2, 2)
    assert y.shape == (2,)


def test_validate_input_converts_features_to_float():
    X, _ = _validate_input(
        [[1, 2], [3, 4]],
        [0, 1],
    )

    assert X.dtype.kind == "f"


def test_invalid_X_dimension():
    with pytest.raises(ValueError):
        _validate_input(
            [1, 2, 3],
            [0, 1, 2],
        )


def test_invalid_y_dimension():
    with pytest.raises(ValueError):
        _validate_input(
            [[1], [2]],
            [[0], [1]],
        )


def test_invalid_lengths():
    with pytest.raises(ValueError):
        _validate_input(
            [[1], [2], [3]],
            [0, 1],
        )


def test_empty_X():
    with pytest.raises(ValueError):
        _validate_input(
            np.empty((0, 2)),
            np.array([]),
        )


def test_zero_features():
    with pytest.raises(ValueError):
        _validate_input(
            np.empty((3, 0)),
            np.array([0, 1, 0]),
        )


def test_non_numeric_X():
    with pytest.raises(ValueError):
        _validate_input(
            [["a"], ["b"]],
            [0, 1],
        )


def test_nan_X():
    X = np.array(
        [
            [1.0, 2.0],
            [np.nan, 4.0],
        ]
    )

    with pytest.raises(ValueError):
        _validate_input(
            X,
            [0, 1],
        )


def test_inf_X():
    X = np.array(
        [
            [1.0, 2.0],
            [np.inf, 4.0],
        ]
    )

    with pytest.raises(ValueError):
        _validate_input(
            X,
            [0, 1],
        )


# ============================================================
# DEFAULT MAX FEATURES
# ============================================================


def test_default_max_features_one():
    assert _default_max_features(1) == 1


def test_default_max_features_four():
    assert _default_max_features(4) == 2


def test_default_max_features_nine():
    assert _default_max_features(9) == 3


def test_default_max_features_always_at_least_one():
    for n_features in range(1, 20):
        assert (
            _default_max_features(n_features)
            >= 1
        )


# ============================================================
# BOOTSTRAP
# ============================================================


def test_bootstrap_sample_preserves_sample_count(
    binary_dataset,
):
    X_train, y_train, _, _ = binary_dataset

    rng = np.random.default_rng(42)

    X_bootstrap, y_bootstrap, indices = (
        bootstrap_sample(
            X_train,
            y_train,
            rng,
        )
    )

    assert X_bootstrap.shape == X_train.shape
    assert y_bootstrap.shape == y_train.shape
    assert indices.shape == (len(X_train),)


def test_bootstrap_indices_are_valid(
    binary_dataset,
):
    X_train, y_train, _, _ = binary_dataset

    rng = np.random.default_rng(42)

    _, _, indices = bootstrap_sample(
        X_train,
        y_train,
        rng,
    )

    assert np.all(indices >= 0)
    assert np.all(
        indices < len(X_train)
    )


def test_bootstrap_is_reproducible(
    binary_dataset,
):
    X_train, y_train, _, _ = binary_dataset

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    X1, y1, indices1 = bootstrap_sample(
        X_train,
        y_train,
        rng1,
    )

    X2, y2, indices2 = bootstrap_sample(
        X_train,
        y_train,
        rng2,
    )

    assert np.array_equal(
        indices1,
        indices2,
    )

    assert np.array_equal(
        X1,
        X2,
    )

    assert np.array_equal(
        y1,
        y2,
    )


# ============================================================
# SÉLECTION DES FEATURES
# ============================================================


def test_select_features_returns_correct_number():
    rng = np.random.default_rng(42)

    indices = select_features(
        n_features=5,
        max_features=2,
        rng=rng,
    )

    assert len(indices) == 2


def test_select_features_indices_are_valid():
    rng = np.random.default_rng(42)

    indices = select_features(
        n_features=5,
        max_features=3,
        rng=rng,
    )

    assert np.all(indices >= 0)
    assert np.all(indices < 5)


def test_select_features_contains_no_duplicates():
    rng = np.random.default_rng(42)

    indices = select_features(
        n_features=10,
        max_features=5,
        rng=rng,
    )

    assert len(np.unique(indices)) == 5


def test_select_features_is_sorted():
    rng = np.random.default_rng(42)

    indices = select_features(
        n_features=10,
        max_features=5,
        rng=rng,
    )

    assert np.all(
        indices[:-1] <= indices[1:]
    )


def test_select_all_features():
    rng = np.random.default_rng(42)

    indices = select_features(
        n_features=4,
        max_features=4,
        rng=rng,
    )

    assert np.array_equal(
        indices,
        np.array([0, 1, 2, 3]),
    )


def test_invalid_n_features_type():
    rng = np.random.default_rng(42)

    with pytest.raises(TypeError):
        select_features(
            n_features=2.5,
            max_features=1,
            rng=rng,
        )


def test_invalid_max_features_type():
    rng = np.random.default_rng(42)

    with pytest.raises(TypeError):
        select_features(
            n_features=4,
            max_features=2.5,
            rng=rng,
        )


def test_invalid_n_features_value():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        select_features(
            n_features=0,
            max_features=1,
            rng=rng,
        )


def test_invalid_max_features_value():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        select_features(
            n_features=4,
            max_features=0,
            rng=rng,
        )


def test_max_features_larger_than_features():
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        select_features(
            n_features=3,
            max_features=4,
            rng=rng,
        )


# ============================================================
# MAJORITY VOTE
# ============================================================


def test_majority_vote_binary():
    assert (
        majority_vote(
            np.array([0, 0, 1, 0, 1])
        )
        == 0
    )


def test_majority_vote_class_one():
    assert (
        majority_vote(
            np.array([0, 1, 1, 1, 0])
        )
        == 1
    )


def test_majority_vote_multiclass():
    assert (
        majority_vote(
            np.array([0, 1, 2, 2, 2])
        )
        == 2
    )


def test_majority_vote_string_labels():
    assert (
        majority_vote(
            np.array(
                ["chat", "chien", "chat"]
            )
        )
        == "chat"
    )


def test_majority_vote_tie_is_deterministic():
    result = majority_vote(
        np.array([0, 1])
    )

    assert result == 0


def test_majority_vote_requires_1d():
    with pytest.raises(ValueError):
        majority_vote(
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ]
            )
        )


def test_majority_vote_empty():
    with pytest.raises(ValueError):
        majority_vote(
            np.array([])
        )


# ============================================================
# RANDOM FOREST - BINARY
# ============================================================


def test_random_forest_binary(
    binary_dataset,
):
    X_train, y_train, X_test, y_test = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=10,
        max_depth=5,
        random_state=42,
    )

    assert result["accuracy"] >= 0.5

    assert len(
        result["predictions"]
    ) == len(X_test)

    assert result["n_classes"] == 2


def test_random_forest_predictions_are_valid(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=10,
        random_state=42,
    )

    assert np.all(
        np.isin(
            result["predictions"],
            np.unique(y_train),
        )
    )


# ============================================================
# RANDOM FOREST - MULTICLASSE
# ============================================================


def test_random_forest_multiclass(
    multiclass_dataset,
):
    X_train, y_train, X_test, y_test = (
        multiclass_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=20,
        max_depth=5,
        random_state=42,
    )

    assert result["accuracy"] >= 0.5

    assert len(
        result["predictions"]
    ) == len(X_test)

    assert result["n_classes"] == 3


# ============================================================
# PREDICT FOREST
# ============================================================


def test_predict_forest_returns_numpy_array(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        random_state=42,
    )

    predictions = predict_forest(
        result["trees"],
        result["feature_indices"],
        X_test,
    )

    assert isinstance(
        predictions,
        np.ndarray,
    )

    assert predictions.shape == (
        len(X_test),
    )


def test_predict_forest_matches_pipeline(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=10,
        random_state=42,
    )

    predictions = predict_forest(
        result["trees"],
        result["feature_indices"],
        X_test,
    )

    assert np.array_equal(
        predictions,
        result["predictions"],
    )


def test_predict_forest_single_observation(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        random_state=42,
    )

    prediction = predict_forest(
        result["trees"],
        result["feature_indices"],
        np.array([1.5, 1.5]),
    )

    assert prediction.shape == (1,)


# ============================================================
# REPRODUCTIBILITÉ
# ============================================================


def test_random_forest_is_reproducible(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result1 = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=10,
        random_state=42,
    )

    result2 = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=10,
        random_state=42,
    )

    assert np.array_equal(
        result1["predictions"],
        result2["predictions"],
    )

    for indices1, indices2 in zip(
        result1["feature_indices"],
        result2["feature_indices"],
    ):
        assert np.array_equal(
            indices1,
            indices2,
        )


def test_random_forest_different_seed_can_change_result(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result1 = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=10,
        random_state=1,
    )

    result2 = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=10,
        random_state=99,
    )

    # Les structures aléatoires doivent être différentes.
    indices_are_different = any(
        not np.array_equal(
            a,
            b,
        )
        for a, b in zip(
            result1["feature_indices"],
            result2["feature_indices"],
        )
    )

    assert indices_are_different


# ============================================================
# PARAMÈTRES
# ============================================================


def test_n_estimators_is_preserved(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=15,
        random_state=42,
    )

    assert result["n_estimators"] == 15
    assert len(result["trees"]) == 15


def test_max_depth_is_preserved(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        max_depth=3,
        random_state=42,
    )

    assert result["max_depth"] == 3


def test_min_samples_split_is_preserved(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        min_samples_split=4,
        random_state=42,
    )

    assert (
        result["min_samples_split"] == 4
    )


def test_max_features_default(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        random_state=42,
    )

    expected = _default_max_features(
        X_train.shape[1]
    )

    assert (
        result["max_features"]
        == expected
    )


def test_max_features_custom(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        max_features=1,
        random_state=42,
    )

    assert result["max_features"] == 1

    for indices in result["feature_indices"]:
        assert len(indices) == 1


# ============================================================
# Y_TEST OPTIONNEL
# ============================================================


def test_random_forest_without_y_test(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        y_test=None,
        n_estimators=5,
        random_state=42,
    )

    assert result["accuracy"] is None


# ============================================================
# LISTES PYTHON
# ============================================================


def test_random_forest_accepts_python_lists():
    result = random_forest(
        [
            [1, 1],
            [1, 2],
            [5, 5],
            [5, 6],
        ],
        [0, 0, 1, 1],
        [
            [1.5, 1.5],
            [5.5, 5.5],
        ],
        [0, 1],
        n_estimators=5,
        random_state=42,
    )

    assert len(
        result["predictions"]
    ) == 2


# ============================================================
# TEST AVEC TROIS FEATURES
# ============================================================


def test_random_forest_three_features():
    X_train = np.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [2, 1, 1],
            [5, 5, 5],
            [5, 6, 5],
            [6, 5, 5],
            [9, 9, 9],
            [9, 10, 9],
            [10, 9, 9],
        ],
        dtype=float,
    )

    y_train = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2]
    )

    X_test = np.array(
        [
            [1.5, 1.5, 1],
            [5.5, 5.5, 5],
            [9.5, 9.5, 9],
        ],
        dtype=float,
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=15,
        random_state=42,
    )

    assert result["n_features"] == 3
    assert result["max_features"] == 1
    assert len(result["trees"]) == 15


# ============================================================
# CAS LIMITES
# ============================================================


def test_random_forest_one_tree(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=1,
        random_state=42,
    )

    assert len(result["trees"]) == 1


def test_random_forest_max_depth_zero(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        max_depth=0,
        random_state=42,
    )

    assert len(
        result["predictions"]
    ) == len(X_test)


def test_random_forest_min_samples_split(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        min_samples_split=len(X_train),
        random_state=42,
    )

    assert len(
        result["predictions"]
    ) == len(X_test)


# ============================================================
# PARAMÈTRES INVALIDES
# ============================================================


def test_invalid_n_estimators_type(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(TypeError):
        random_forest(
            X_train,
            y_train,
            X_test,
            n_estimators=2.5,
        )


def test_invalid_n_estimators_value(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            X_test,
            n_estimators=0,
        )


def test_invalid_max_depth_type(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(TypeError):
        random_forest(
            X_train,
            y_train,
            X_test,
            max_depth=2.5,
        )


def test_invalid_max_depth_value(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            X_test,
            max_depth=-1,
        )


def test_invalid_min_samples_split_type(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(TypeError):
        random_forest(
            X_train,
            y_train,
            X_test,
            min_samples_split=2.5,
        )


def test_invalid_min_samples_split_value(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            X_test,
            min_samples_split=1,
        )


def test_invalid_max_features_type(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(TypeError):
        random_forest(
            X_train,
            y_train,
            X_test,
            max_features=1.5,
        )


def test_invalid_max_features_value(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            X_test,
            max_features=0,
        )


def test_max_features_too_large(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            X_test,
            max_features=10,
        )


def test_invalid_random_state_type(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(TypeError):
        random_forest(
            X_train,
            y_train,
            X_test,
            random_state=1.5,
        )


# ============================================================
# X_TEST INVALID
# ============================================================


def test_invalid_X_test_dimension(
    binary_dataset,
):
    X_train, y_train, _, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            [[1, 2, 3]],
        )


def test_invalid_X_test_feature_count(
    binary_dataset,
):
    X_train, y_train, _, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            [[1]],
        )


def test_empty_X_test(
    binary_dataset,
):
    X_train, y_train, _, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            np.empty((0, 2)),
        )


def test_nan_X_test(
    binary_dataset,
):
    X_train, y_train, _, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            [[np.nan, 1]],
        )


def test_inf_X_test(
    binary_dataset,
):
    X_train, y_train, _, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            [[np.inf, 1]],
        )


# ============================================================
# Y_TEST INVALID
# ============================================================


def test_invalid_y_test_dimension(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            X_test,
            [[0], [1]],
        )


def test_invalid_y_test_length(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    with pytest.raises(ValueError):
        random_forest(
            X_train,
            y_train,
            X_test,
            [0],
        )


# ============================================================
# PREDICT FOREST INVALID
# ============================================================


def test_predict_forest_invalid_trees(
    binary_dataset,
):
    _, _, X_test, _ = binary_dataset

    with pytest.raises(TypeError):
        predict_forest(
            "invalid",
            [],
            X_test,
        )


def test_predict_forest_empty_trees(
    binary_dataset,
):
    _, _, X_test, _ = binary_dataset

    with pytest.raises(ValueError):
        predict_forest(
            [],
            [],
            X_test,
        )


def test_predict_forest_mismatched_lengths(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        random_state=42,
    )

    with pytest.raises(ValueError):
        predict_forest(
            result["trees"],
            [],
            X_test,
        )


# ============================================================
# STRUCTURE DU RÉSULTAT
# ============================================================


def test_random_forest_result_keys(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=5,
        random_state=42,
    )

    expected_keys = {
        "trees",
        "feature_indices",
        "predictions",
        "accuracy",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "max_features",
        "random_state",
        "n_train_samples",
        "n_test_samples",
        "n_features",
        "n_classes",
    }

    assert set(result.keys()) == expected_keys


def test_random_forest_result_dimensions(
    binary_dataset,
):
    X_train, y_train, X_test, _ = (
        binary_dataset
    )

    result = random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=7,
        random_state=42,
    )

    assert (
        result["n_train_samples"]
        == len(X_train)
    )

    assert (
        result["n_test_samples"]
        == len(X_test)
    )

    assert (
        result["n_features"]
        == X_train.shape[1]
    )

    assert len(
        result["trees"]
    ) == 7

    assert len(
        result["feature_indices"]
    ) == 7

