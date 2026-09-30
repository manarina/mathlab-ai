
"""
Tests unitaires pour K-Nearest Neighbors (K-NN).

Module testé :
    core.machine_learning.knn
"""

import numpy as np
import pytest

from core.machine_learning.knn import (
    _validate_input,
    accuracy_score,
    euclidean_distance,
    get_neighbors,
    knn,
    predict_knn,
)


# ============================================================================
# DONNÉES DE TEST
# ============================================================================


@pytest.fixture
def simple_dataset():
    """
    Petit jeu de données 2D facilement vérifiable.

    Classe 0 : points proches de (0, 0)
    Classe 1 : points proches de (5, 5)
    """

    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
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
        ]
    )

    return X_train, y_train


# ============================================================================
# TESTS DE LA VALIDATION
# ============================================================================


def test_validate_input_returns_numpy_arrays(simple_dataset):
    """Les données doivent être converties en tableaux NumPy."""

    X_train, y_train = simple_dataset

    X_train_valid, y_train_valid, _, _ = _validate_input(
        X_train.tolist(),
        y_train.tolist(),
        k=3,
    )

    assert isinstance(X_train_valid, np.ndarray)
    assert isinstance(y_train_valid, np.ndarray)


def test_validate_input_rejects_1d_X_train():
    """X_train doit être un tableau 2D."""

    X_train = np.array([1, 2, 3])
    y_train = np.array([0, 1, 0])

    with pytest.raises(ValueError, match="X_train doit être"):
        _validate_input(X_train, y_train, k=1)


def test_validate_input_rejects_empty_X_train():
    """X_train ne doit pas être vide."""

    X_train = np.empty((0, 2))
    y_train = np.array([])

    with pytest.raises(ValueError, match="X_train ne peut pas être vide"):
        _validate_input(X_train, y_train, k=1)


def test_validate_input_rejects_1d_y_train():
    """y_train doit être un tableau 1D."""

    X_train = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    y_train = np.array(
        [
            [0],
            [1],
        ]
    )

    with pytest.raises(ValueError, match="y_train doit être"):
        _validate_input(X_train, y_train, k=1)


def test_validate_input_rejects_different_lengths():
    """X_train et y_train doivent avoir la même longueur."""

    X_train = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    y_train = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="même nombre d'observations",
    ):
        _validate_input(X_train, y_train, k=1)


def test_validate_input_rejects_non_finite_values():
    """Les données numériques doivent être finies."""

    X_train = np.array(
        [
            [1.0, 2.0],
            [np.nan, 4.0],
        ]
    )

    y_train = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="valeurs numériques finies",
    ):
        _validate_input(X_train, y_train, k=1)


def test_validate_input_rejects_k_zero(simple_dataset):
    """k doit être au moins égal à 1."""

    X_train, y_train = simple_dataset

    with pytest.raises(
        ValueError,
        match="k doit être supérieur ou égal à 1",
    ):
        _validate_input(X_train, y_train, k=0)


def test_validate_input_rejects_k_greater_than_training_size(
    simple_dataset,
):
    """k ne peut pas dépasser le nombre d'observations."""

    X_train, y_train = simple_dataset

    with pytest.raises(
        ValueError,
        match="k ne peut pas être supérieur",
    ):
        _validate_input(
            X_train,
            y_train,
            k=len(X_train) + 1,
        )


def test_validate_input_rejects_non_integer_k(simple_dataset):
    """k doit être un entier."""

    X_train, y_train = simple_dataset

    with pytest.raises(TypeError, match="k doit être un entier"):
        _validate_input(X_train, y_train, k=2.5)


def test_validate_input_rejects_invalid_X_test(simple_dataset):
    """X_test doit avoir la même dimension de caractéristiques."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="même nombre de caractéristiques",
    ):
        _validate_input(
            X_train,
            y_train,
            X_test,
            k=3,
        )


def test_validate_input_rejects_y_test_without_X_test(simple_dataset):
    """y_test ne peut pas être fourni sans X_test."""

    X_train, y_train = simple_dataset
    y_test = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="X_test doit être fourni",
    ):
        _validate_input(
            X_train,
            y_train,
            y_test=y_test,
            k=3,
        )


# ============================================================================
# TESTS DE LA DISTANCE EUCLIDIENNE
# ============================================================================


def test_euclidean_distance_same_point():
    """La distance d'un point à lui-même doit être nulle."""

    x1 = np.array([1.0, 2.0])
    x2 = np.array([1.0, 2.0])

    distance = euclidean_distance(x1, x2)

    assert distance == pytest.approx(0.0)


def test_euclidean_distance_simple_case():
    """
    Distance entre (0,0) et (3,4) = 5.
    """

    x1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 4.0])

    distance = euclidean_distance(x1, x2)

    assert distance == pytest.approx(5.0)


def test_euclidean_distance_2d():
    """
    Distance entre (1,2) et (4,6) = 5.
    """

    x1 = np.array([1.0, 2.0])
    x2 = np.array([4.0, 6.0])

    distance = euclidean_distance(x1, x2)

    assert distance == pytest.approx(5.0)


def test_euclidean_distance_is_symmetric():
    """d(x,y) doit être égal à d(y,x)."""

    x1 = np.array([1.0, 3.0])
    x2 = np.array([5.0, 7.0])

    distance_xy = euclidean_distance(x1, x2)
    distance_yx = euclidean_distance(x2, x1)

    assert distance_xy == pytest.approx(distance_yx)


def test_euclidean_distance_rejects_different_dimensions():
    """Les vecteurs doivent avoir la même dimension."""

    x1 = np.array([1.0, 2.0])
    x2 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="même dimension",
    ):
        euclidean_distance(x1, x2)


def test_euclidean_distance_rejects_non_1d_arrays():
    """Les observations doivent être des vecteurs 1D."""

    x1 = np.array([[1.0, 2.0]])
    x2 = np.array([1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="vecteurs 1D",
    ):
        euclidean_distance(x1, x2)


# ============================================================================
# TESTS DE RECHERCHE DES VOISINS
# ============================================================================


def test_get_neighbors_returns_k_neighbors(simple_dataset):
    """get_neighbors doit retourner exactement k voisins."""

    X_train, y_train = simple_dataset

    query = np.array([0.2, 0.2])

    indices, distances = get_neighbors(
        X_train,
        y_train,
        query,
        k=3,
    )

    assert len(indices) == 3
    assert len(distances) == 3


def test_get_neighbors_returns_sorted_distances(simple_dataset):
    """Les distances doivent être triées par ordre croissant."""

    X_train, y_train = simple_dataset

    query = np.array([0.2, 0.2])

    _, distances = get_neighbors(
        X_train,
        y_train,
        query,
        k=4,
    )

    assert np.all(
        distances[:-1] <= distances[1:]
    )


def test_get_neighbors_returns_closest_point_first(
    simple_dataset,
):
    """Le voisin le plus proche doit être placé en première position."""

    X_train, y_train = simple_dataset

    query = np.array([0.1, 0.1])

    indices, distances = get_neighbors(
        X_train,
        y_train,
        query,
        k=1,
    )

    assert indices[0] == 0
    assert distances[0] == pytest.approx(
        np.sqrt(0.1**2 + 0.1**2)
    )


def test_get_neighbors_k_one(simple_dataset):
    """Avec k=1, un seul voisin doit être retourné."""

    X_train, y_train = simple_dataset

    query = np.array([5.1, 5.1])

    indices, distances = get_neighbors(
        X_train,
        y_train,
        query,
        k=1,
    )

    assert len(indices) == 1
    assert len(distances) == 1
    assert indices[0] in [3, 4, 5]


def test_get_neighbors_rejects_wrong_query_dimension(
    simple_dataset,
):
    """La requête doit avoir le même nombre de features."""

    X_train, y_train = simple_dataset

    query = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="même nombre de caractéristiques",
    ):
        get_neighbors(
            X_train,
            y_train,
            query,
            k=3,
        )


# ============================================================================
# TESTS DES PRÉDICTIONS
# ============================================================================


def test_predict_knn_returns_numpy_array(simple_dataset):
    """Les prédictions doivent être retournées sous forme NumPy."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.2, 5.2],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    assert isinstance(predictions, np.ndarray)


def test_predict_knn_predicts_class_zero(simple_dataset):
    """Un point proche du groupe 0 doit être classé 0."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.2, 0.2],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    assert predictions[0] == 0


def test_predict_knn_predicts_class_one(simple_dataset):
    """Un point proche du groupe 1 doit être classé 1."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [5.2, 5.2],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    assert predictions[0] == 1


def test_predict_knn_multiple_observations(simple_dataset):
    """K-NN doit pouvoir classifier plusieurs observations."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [0.8, 0.8],
            [5.1, 5.1],
            [5.8, 5.8],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    expected = np.array([0, 0, 1, 1])

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


def test_predict_knn_with_k_one(simple_dataset):
    """
    Avec k=1, la classe du voisin le plus proche
    doit être utilisée.
    """

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 5.1],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=1,
    )

    expected = np.array([0, 1])

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


def test_predict_knn_supports_multiclass():
    """K-NN doit fonctionner avec plusieurs classes."""

    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [10.0, 10.0],
            [10.0, 11.0],
        ]
    )

    y_train = np.array(
        [
            0,
            0,
            1,
            1,
            2,
            2,
        ]
    )

    X_test = np.array(
        [
            [0.2, 0.2],
            [5.2, 5.2],
            [10.2, 10.2],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    expected = np.array([0, 1, 2])

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


# ============================================================================
# TESTS DE L'ACCURACY
# ============================================================================


def test_accuracy_score_perfect():
    """Une classification parfaite doit donner 1.0."""

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    assert accuracy == pytest.approx(1.0)


def test_accuracy_score_zero():
    """Une classification totalement incorrecte doit donner 0.0."""

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    assert accuracy == pytest.approx(0.0)


def test_accuracy_score_partial():
    """Vérifie un cas d'accuracy partielle."""

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    assert accuracy == pytest.approx(0.5)


def test_accuracy_score_rejects_different_lengths():
    """Les deux tableaux doivent avoir la même longueur."""

    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])

    with pytest.raises(
        ValueError,
        match="même longueur",
    ):
        accuracy_score(
            y_true,
            y_pred,
        )


def test_accuracy_score_rejects_empty_arrays():
    """y_true ne doit pas être vide."""

    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(
        ValueError,
        match="ne peut pas être vide",
    ):
        accuracy_score(
            y_true,
            y_pred,
        )


# ============================================================================
# TESTS DU PIPELINE COMPLET
# ============================================================================


def test_knn_returns_dictionary(simple_dataset):
    """Le pipeline doit retourner un dictionnaire."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.2, 0.2],
            [5.2, 5.2],
        ]
    )

    y_test = np.array([0, 1])

    result = knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=3,
    )

    assert isinstance(result, dict)


def test_knn_returns_expected_keys(simple_dataset):
    """Le pipeline doit retourner toutes les informations attendues."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.2, 0.2],
            [5.2, 5.2],
        ]
    )

    y_test = np.array([0, 1])

    result = knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=3,
    )

    expected_keys = {
        "predictions",
        "accuracy",
        "k",
        "n_train_samples",
        "n_test_samples",
        "n_features",
    }

    assert set(result.keys()) == expected_keys


def test_knn_predictions(simple_dataset):
    """Le pipeline doit produire les bonnes prédictions."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 5.1],
        ]
    )

    y_test = np.array([0, 1])

    result = knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=3,
    )

    expected = np.array([0, 1])

    np.testing.assert_array_equal(
        result["predictions"],
        expected,
    )


def test_knn_accuracy(simple_dataset):
    """Le pipeline doit calculer correctement l'accuracy."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 5.1],
        ]
    )

    y_test = np.array([0, 1])

    result = knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=3,
    )

    assert result["accuracy"] == pytest.approx(1.0)


def test_knn_without_y_test_returns_none_accuracy(
    simple_dataset,
):
    """
    Si les classes réelles ne sont pas fournies,
    l'accuracy doit être None.
    """

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 5.1],
        ]
    )

    result = knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    assert result["accuracy"] is None


def test_knn_metadata(simple_dataset):
    """Vérifie les informations complémentaires du résultat."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 5.1],
        ]
    )

    result = knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    assert result["k"] == 3
    assert result["n_train_samples"] == 6
    assert result["n_test_samples"] == 2
    assert result["n_features"] == 2


# ============================================================================
# TESTS DE ROBUSTESSE
# ============================================================================


def test_knn_accepts_python_lists():
    """Le pipeline doit accepter des listes Python."""

    X_train = [
        [0.0, 0.0],
        [0.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
    ]

    y_train = [0, 0, 1, 1]

    X_test = [
        [0.2, 0.2],
        [5.2, 5.2],
    ]

    result = knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    expected = np.array([0, 1])

    np.testing.assert_array_equal(
        result["predictions"],
        expected,
    )


def test_knn_with_string_labels():
    """Les classes peuvent être représentées par des chaînes."""

    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
        ]
    )

    y_train = np.array(
        [
            "A",
            "A",
            "B",
            "B",
        ]
    )

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 5.1],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    expected = np.array(
        [
            "A",
            "B",
        ]
    )

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


def test_knn_predictions_length_matches_test_data(
    simple_dataset,
):
    """Il doit y avoir une prédiction par observation de test."""

    X_train, y_train = simple_dataset

    X_test = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [5.1, 5.1],
            [5.2, 5.2],
        ]
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=3,
    )

    assert len(predictions) == len(X_test)

