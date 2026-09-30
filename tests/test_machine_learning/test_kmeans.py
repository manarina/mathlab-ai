
"""
Tests for K-Means Clustering
============================

Tests unitaires et fonctionnels pour :
    core.machine_learning.kmeans
"""

import numpy as np
import pytest

from core.machine_learning.kmeans import (
    _validate_input,
    initialize_centroids,
    calculate_distances,
    assign_clusters,
    update_centroids,
    calculate_inertia,
    predict_kmeans,
    fit_kmeans,
    kmeans,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_data():
    """
    Données simples avec deux groupes bien séparés.
    """
    return np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [8.0, 8.0],
            [8.0, 9.0],
            [9.0, 8.0],
            [9.0, 9.0],
        ]
    )


@pytest.fixture
def simple_centroids():
    """
    Centroïdes connus pour les tests de distance.
    """
    return np.array(
        [
            [1.5, 1.5],
            [8.5, 8.5],
        ]
    )


# ============================================================
# TESTS DE VALIDATION
# ============================================================

def test_validate_input_accepts_valid_data(simple_data):
    result = _validate_input(
        simple_data,
        n_clusters=2,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 2)
    assert result.dtype == float


def test_validate_input_accepts_python_list():
    X = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    result = _validate_input(
        X,
        n_clusters=2,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)


def test_validate_input_rejects_1d_array():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        _validate_input(
            X,
            n_clusters=2,
        )


def test_validate_input_rejects_empty_array():
    X = np.empty((0, 2))

    with pytest.raises(ValueError):
        _validate_input(
            X,
            n_clusters=2,
        )


def test_validate_input_rejects_zero_features():
    X = np.empty((3, 0))

    with pytest.raises(ValueError):
        _validate_input(
            X,
            n_clusters=2,
        )


def test_validate_input_rejects_nan():
    X = np.array(
        [
            [1.0, 2.0],
            [np.nan, 4.0],
            [5.0, 6.0],
        ]
    )

    with pytest.raises(ValueError):
        _validate_input(
            X,
            n_clusters=2,
        )


def test_validate_input_rejects_infinite_values():
    X = np.array(
        [
            [1.0, 2.0],
            [np.inf, 4.0],
            [5.0, 6.0],
        ]
    )

    with pytest.raises(ValueError):
        _validate_input(
            X,
            n_clusters=2,
        )


def test_validate_input_rejects_non_integer_k(simple_data):
    with pytest.raises(TypeError):
        _validate_input(
            simple_data,
            n_clusters=2.5,
        )


def test_validate_input_rejects_zero_clusters(simple_data):
    with pytest.raises(ValueError):
        _validate_input(
            simple_data,
            n_clusters=0,
        )


def test_validate_input_rejects_negative_clusters(simple_data):
    with pytest.raises(ValueError):
        _validate_input(
            simple_data,
            n_clusters=-1,
        )


def test_validate_input_rejects_too_many_clusters(simple_data):
    with pytest.raises(ValueError):
        _validate_input(
            simple_data,
            n_clusters=9,
        )


# ============================================================
# TESTS INITIALISATION DES CENTROÏDES
# ============================================================

def test_initialize_centroids_shape(simple_data):
    centroids = initialize_centroids(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert centroids.shape == (2, 2)


def test_initialize_centroids_uses_data_points(simple_data):
    centroids = initialize_centroids(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    for centroid in centroids:
        assert any(
            np.allclose(centroid, point)
            for point in simple_data
        )


def test_initialize_centroids_are_unique(simple_data):
    centroids = initialize_centroids(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert not np.allclose(
        centroids[0],
        centroids[1],
    )


def test_initialize_centroids_is_reproducible(simple_data):
    centroids_1 = initialize_centroids(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    centroids_2 = initialize_centroids(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert np.allclose(
        centroids_1,
        centroids_2,
    )


def test_initialize_centroids_different_seed_can_change_result(
    simple_data,
):
    centroids_1 = initialize_centroids(
        simple_data,
        n_clusters=2,
        random_state=1,
    )

    centroids_2 = initialize_centroids(
        simple_data,
        n_clusters=2,
        random_state=100,
    )

    # Les graines différentes peuvent produire des centroïdes
    # différents. On vérifie simplement que les deux résultats
    # ont la bonne structure.
    assert centroids_1.shape == centroids_2.shape


# ============================================================
# TESTS DES DISTANCES
# ============================================================

def test_calculate_distances_shape(
    simple_data,
    simple_centroids,
):
    distances = calculate_distances(
        simple_data,
        simple_centroids,
    )

    assert distances.shape == (8, 2)


def test_calculate_distances_known_values():
    X = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ]
    )

    centroids = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ]
    )

    distances = calculate_distances(
        X,
        centroids,
    )

    expected = np.array(
        [
            [0.0, 5.0],
            [5.0, 0.0],
        ]
    )

    assert np.allclose(
        distances,
        expected,
    )


def test_calculate_distances_is_non_negative(
    simple_data,
    simple_centroids,
):
    distances = calculate_distances(
        simple_data,
        simple_centroids,
    )

    assert np.all(distances >= 0)


def test_calculate_distances_rejects_wrong_dimensions():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    centroids = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(ValueError):
        calculate_distances(
            X,
            centroids,
        )


def test_calculate_distances_rejects_1d_X():
    X = np.array([1.0, 2.0])

    centroids = np.array(
        [
            [1.0, 2.0],
        ]
    )

    with pytest.raises(ValueError):
        calculate_distances(
            X,
            centroids,
        )


# ============================================================
# TESTS ATTRIBUTION DES CLUSTERS
# ============================================================

def test_assign_clusters_shape(
    simple_data,
    simple_centroids,
):
    labels = assign_clusters(
        simple_data,
        simple_centroids,
    )

    assert labels.shape == (8,)


def test_assign_clusters_known_data(
    simple_data,
    simple_centroids,
):
    labels = assign_clusters(
        simple_data,
        simple_centroids,
    )

    # Les quatre premiers points sont proches du
    # premier centroïde et les quatre derniers du second.
    assert np.array_equal(
        labels,
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    )


def test_assign_clusters_labels_are_valid(
    simple_data,
    simple_centroids,
):
    labels = assign_clusters(
        simple_data,
        simple_centroids,
    )

    assert np.all(
        labels >= 0
    )

    assert np.all(
        labels < len(simple_centroids)
    )


# ============================================================
# TESTS RECALCUL DES CENTROÏDES
# ============================================================

def test_update_centroids_known_values(simple_data):
    labels = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    centroids = update_centroids(
        simple_data,
        labels,
        n_clusters=2,
    )

    expected = np.array(
        [
            [1.5, 1.5],
            [8.5, 8.5],
        ]
    )

    assert np.allclose(
        centroids,
        expected,
    )


def test_update_centroids_shape(simple_data):
    labels = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    centroids = update_centroids(
        simple_data,
        labels,
        n_clusters=2,
    )

    assert centroids.shape == (2, 2)


def test_update_centroids_handles_empty_cluster(
    simple_data,
):
    labels = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0]
    )

    old_centroids = np.array(
        [
            [1.0, 1.0],
            [8.0, 8.0],
        ]
    )

    centroids = update_centroids(
        simple_data,
        labels,
        n_clusters=2,
        old_centroids=old_centroids,
    )

    # Le cluster 0 est recalculé.
    assert np.allclose(
        centroids[0],
        np.mean(simple_data, axis=0),
    )

    # Le cluster 1 est vide :
    # son ancien centroïde est conservé.
    assert np.allclose(
        centroids[1],
        old_centroids[1],
    )


def test_update_centroids_rejects_wrong_label_length(
    simple_data,
):
    labels = np.array([0, 1])

    with pytest.raises(ValueError):
        update_centroids(
            simple_data,
            labels,
            n_clusters=2,
        )


def test_update_centroids_rejects_invalid_label(
    simple_data,
):
    labels = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2]
    )

    with pytest.raises(ValueError):
        update_centroids(
            simple_data,
            labels,
            n_clusters=2,
        )


# ============================================================
# TESTS INERTIE
# ============================================================

def test_calculate_inertia_known_value():
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )

    labels = np.array([0, 0])

    centroids = np.array(
        [
            [1.0, 0.0],
        ]
    )

    inertia = calculate_inertia(
        X,
        labels,
        centroids,
    )

    assert inertia == pytest.approx(2.0)


def test_calculate_inertia_is_non_negative(
    simple_data,
    simple_centroids,
):
    labels = assign_clusters(
        simple_data,
        simple_centroids,
    )

    inertia = calculate_inertia(
        simple_data,
        labels,
        simple_centroids,
    )

    assert inertia >= 0


def test_calculate_inertia_is_zero_for_perfect_centroids():
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [5.0, 5.0],
        ]
    )

    labels = np.array(
        [0, 0, 1, 1]
    )

    centroids = np.array(
        [
            [1.0, 1.0],
            [5.0, 5.0],
        ]
    )

    inertia = calculate_inertia(
        X,
        labels,
        centroids,
    )

    assert inertia == pytest.approx(0.0)


# ============================================================
# TESTS PREDICTION
# ============================================================

def test_predict_kmeans(simple_centroids):
    X_test = np.array(
        [
            [1.2, 1.3],
            [8.7, 8.8],
        ]
    )

    predictions = predict_kmeans(
        X_test,
        simple_centroids,
    )

    assert np.array_equal(
        predictions,
        np.array([0, 1]),
    )


def test_predict_kmeans_shape(simple_centroids):
    X_test = np.array(
        [
            [1.0, 1.0],
            [5.0, 5.0],
            [9.0, 9.0],
        ]
    )

    predictions = predict_kmeans(
        X_test,
        simple_centroids,
    )

    assert predictions.shape == (3,)


# ============================================================
# TESTS FIT K-MEANS
# ============================================================

def test_fit_kmeans_returns_expected_keys(simple_data):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    expected_keys = {
        "centroids",
        "labels",
        "inertia",
        "n_iterations",
        "converged",
        "n_clusters",
    }

    assert expected_keys.issubset(
        result.keys()
    )


def test_fit_kmeans_centroids_shape(simple_data):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert result["centroids"].shape == (2, 2)


def test_fit_kmeans_labels_shape(simple_data):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert result["labels"].shape == (8,)


def test_fit_kmeans_finds_two_clusters(simple_data):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    labels = result["labels"]

    assert len(np.unique(labels)) == 2


def test_fit_kmeans_converges(simple_data):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert result["converged"] is True


def test_fit_kmeans_iterations_are_valid(simple_data):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        max_iterations=100,
        random_state=42,
    )

    assert 1 <= result["n_iterations"] <= 100


def test_fit_kmeans_inertia_is_positive(simple_data):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert result["inertia"] >= 0


def test_fit_kmeans_is_reproducible(simple_data):
    result_1 = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    result_2 = fit_kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert np.allclose(
        result_1["centroids"],
        result_2["centroids"],
    )

    assert np.array_equal(
        result_1["labels"],
        result_2["labels"],
    )

    assert result_1["inertia"] == pytest.approx(
        result_2["inertia"]
    )


def test_fit_kmeans_respects_max_iterations(
    simple_data,
):
    result = fit_kmeans(
        simple_data,
        n_clusters=2,
        max_iterations=1,
        random_state=42,
    )

    assert result["n_iterations"] == 1


def test_fit_kmeans_rejects_invalid_max_iterations(
    simple_data,
):
    with pytest.raises(ValueError):
        fit_kmeans(
            simple_data,
            n_clusters=2,
            max_iterations=0,
        )


def test_fit_kmeans_rejects_negative_tolerance(
    simple_data,
):
    with pytest.raises(ValueError):
        fit_kmeans(
            simple_data,
            n_clusters=2,
            tolerance=-1,
        )


# ============================================================
# TESTS PIPELINE COMPLET K-MEANS
# ============================================================

def test_kmeans_pipeline(simple_data):
    result = kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert "centroids" in result
    assert "labels" in result
    assert "inertia" in result
    assert "n_iterations" in result
    assert "converged" in result
    assert "n_clusters" in result
    assert "n_samples" in result
    assert "n_features" in result


def test_kmeans_pipeline_metadata(simple_data):
    result = kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert result["n_clusters"] == 2
    assert result["n_samples"] == 8
    assert result["n_features"] == 2


def test_kmeans_pipeline_with_test_data(simple_data):
    X_test = np.array(
        [
            [1.5, 1.5],
            [8.5, 8.5],
        ]
    )

    result = kmeans(
        simple_data,
        n_clusters=2,
        X_test=X_test,
        random_state=42,
    )

    assert "predictions" in result
    assert result["predictions"].shape == (2,)
    assert result["n_test_samples"] == 2


def test_kmeans_pipeline_predicts_separated_points(
    simple_data,
):
    X_test = np.array(
        [
            [1.0, 1.0],
            [1.5, 1.5],
            [9.0, 9.0],
            [8.5, 8.5],
        ]
    )

    result = kmeans(
        simple_data,
        n_clusters=2,
        X_test=X_test,
        random_state=42,
    )

    predictions = result["predictions"]

    assert predictions[0] == predictions[1]
    assert predictions[2] == predictions[3]
    assert predictions[0] != predictions[2]


def test_kmeans_pipeline_rejects_wrong_test_features(
    simple_data,
):
    X_test = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        kmeans(
            simple_data,
            n_clusters=2,
            X_test=X_test,
            random_state=42,
        )


def test_kmeans_pipeline_rejects_nan_test_data(
    simple_data,
):
    X_test = np.array(
        [
            [1.0, np.nan],
        ]
    )

    with pytest.raises(ValueError):
        kmeans(
            simple_data,
            n_clusters=2,
            X_test=X_test,
            random_state=42,
        )


# ============================================================
# TESTS CAS PARTICULIERS
# ============================================================

def test_kmeans_with_one_cluster(simple_data):
    result = kmeans(
        simple_data,
        n_clusters=1,
        random_state=42,
    )

    assert result["n_clusters"] == 1
    assert len(np.unique(result["labels"])) == 1
    assert result["centroids"].shape == (1, 2)


def test_kmeans_with_one_feature():
    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [10.0],
            [11.0],
            [12.0],
        ]
    )

    result = kmeans(
        X,
        n_clusters=2,
        random_state=42,
    )

    assert result["centroids"].shape == (2, 1)
    assert len(result["labels"]) == 6
    assert len(np.unique(result["labels"])) == 2


def test_kmeans_with_three_features():
    X = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0],
            [8.0, 8.0, 8.0],
            [9.0, 8.0, 9.0],
            [8.0, 9.0, 8.0],
        ]
    )

    result = kmeans(
        X,
        n_clusters=2,
        random_state=42,
    )

    assert result["centroids"].shape == (2, 3)
    assert result["n_features"] == 3
    assert len(np.unique(result["labels"])) == 2


def test_kmeans_accepts_python_lists():
    X = [
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [8, 8],
        [8, 9],
        [9, 8],
        [9, 9],
    ]

    result = kmeans(
        X,
        n_clusters=2,
        random_state=42,
    )

    assert result["n_samples"] == 8
    assert result["n_features"] == 2
    assert result["centroids"].shape == (2, 2)


def test_kmeans_result_has_numeric_inertia(simple_data):
    result = kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert isinstance(
        result["inertia"],
        float,
    )


def test_kmeans_labels_are_integer_indices(simple_data):
    result = kmeans(
        simple_data,
        n_clusters=2,
        random_state=42,
    )

    assert np.issubdtype(
        result["labels"].dtype,
        np.integer,
    )

    assert np.all(
        result["labels"] >= 0
    )

    assert np.all(
        result["labels"] < 2
    )

