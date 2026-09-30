
"""
Tests unitaires - Decision Tree
================================

Tests de l'implémentation pédagogique du Decision Tree
pour la classification.
"""

import numpy as np
import pytest

from core.machine_learning.decision_tree import (
    TreeNode,
    accuracy_score,
    best_split,
    build_tree,
    decision_tree,
    gini_impurity,
    predict_tree,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================


@pytest.fixture
def binary_data():
    """
    Petit jeu de données de classification binaire.
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
            [1.2, 1.8],
            [5.8, 5.2],
        ]
    )

    y_test = np.array(
        [0, 1, 0, 1]
    )

    return X_train, y_train, X_test, y_test


@pytest.fixture
def multiclass_data():
    """
    Petit jeu de données de classification multiclasses.
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
            [9.0, 9.0],
            [9.0, 10.0],
            [10.0, 9.0],
            [10.0, 10.0],
        ]
    )

    y_train = np.array(
        [
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
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
        ["A", "B", "C"]
    )

    return X_train, y_train, X_test, y_test


# ============================================================
# TESTS GINI
# ============================================================


def test_gini_impurity_pure_class():
    """
    Une population composée d'une seule classe
    doit avoir une impureté de Gini égale à 0.
    """

    y = np.array([1, 1, 1, 1])

    result = gini_impurity(y)

    assert result == pytest.approx(0.0)


def test_gini_impurity_binary_balanced():
    """
    Deux classes équilibrées donnent Gini = 0.5.
    """

    y = np.array([0, 0, 1, 1])

    result = gini_impurity(y)

    assert result == pytest.approx(0.5)


def test_gini_impurity_binary_unbalanced():
    """
    Vérifie le calcul pour une distribution 3/1.
    """

    y = np.array([0, 0, 0, 1])

    result = gini_impurity(y)

    expected = 1.0 - (
        (3 / 4) ** 2
        + (1 / 4) ** 2
    )

    assert result == pytest.approx(expected)


def test_gini_impurity_multiclass():
    """
    Vérifie le calcul avec trois classes.
    """

    y = np.array(
        [0, 0, 1, 1, 2, 2]
    )

    result = gini_impurity(y)

    assert result == pytest.approx(
        2 / 3
    )


def test_gini_impurity_empty():
    """
    Un tableau vide doit retourner 0.
    """

    result = gini_impurity(
        np.array([])
    )

    assert result == pytest.approx(0.0)


def test_gini_impurity_requires_1d():
    """
    Gini doit refuser une matrice 2D.
    """

    with pytest.raises(ValueError):
        gini_impurity(
            np.array(
                [
                    [0, 1],
                    [1, 0],
                ]
            )
        )


# ============================================================
# TESTS BEST SPLIT
# ============================================================


def test_best_split_returns_dictionary(binary_data):
    """
    best_split doit retourner un dictionnaire
    contenant les informations du meilleur split.
    """

    X_train, y_train, _, _ = binary_data

    result = best_split(
        X_train,
        y_train,
    )

    assert result is not None
    assert isinstance(result, dict)

    assert "feature_index" in result
    assert "threshold" in result
    assert "gini" in result


def test_best_split_finds_valid_feature(binary_data):
    """
    Le meilleur split doit utiliser une caractéristique
    valide.
    """

    X_train, y_train, _, _ = binary_data

    result = best_split(
        X_train,
        y_train,
    )

    assert result is not None
    assert result["feature_index"] in (0, 1)


def test_best_split_has_valid_threshold(binary_data):
    """
    Le seuil doit être situé entre deux valeurs
    distinctes de la caractéristique choisie.
    """

    X_train, y_train, _, _ = binary_data

    result = best_split(
        X_train,
        y_train,
    )

    assert result is not None

    feature_index = result["feature_index"]
    threshold = result["threshold"]

    values = np.unique(
        X_train[:, feature_index]
    )

    assert values.min() < threshold < values.max()


def test_best_split_has_non_negative_gini(binary_data):
    """
    L'impureté de Gini doit être comprise entre 0 et 1.
    """

    X_train, y_train, _, _ = binary_data

    result = best_split(
        X_train,
        y_train,
    )

    assert result is not None

    assert 0.0 <= result["gini"] <= 1.0


def test_best_split_pure_partition():
    """
    Pour des données parfaitement séparables,
    le meilleur split doit produire Gini = 0.
    """

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

    y = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    result = best_split(X, y)

    assert result is not None
    assert result["gini"] == pytest.approx(0.0)


def test_best_split_returns_none_for_pure_labels():
    """
    Si toutes les observations appartiennent à la même classe,
    aucun split n'est nécessaire.
    """

    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = np.array(
        [1, 1, 1]
    )

    result = best_split(X, y)

    assert result is None


def test_best_split_returns_none_for_one_sample():
    """
    Une seule observation ne peut pas être séparée.
    """

    X = np.array(
        [[1.0]]
    )

    y = np.array(
        [0]
    )

    result = best_split(X, y)

    assert result is None


# ============================================================
# TESTS BUILD TREE
# ============================================================


def test_build_tree_returns_tree_node(binary_data):
    """
    build_tree doit retourner un TreeNode.
    """

    X_train, y_train, _, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
    )

    assert isinstance(
        tree,
        TreeNode,
    )


def test_build_tree_root_is_split(binary_data):
    """
    Avec des données séparables, la racine doit
    normalement être un nœud de décision.
    """

    X_train, y_train, _, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
        max_depth=5,
    )

    assert tree.feature_index is not None
    assert tree.threshold is not None
    assert tree.left is not None
    assert tree.right is not None


def test_build_tree_stops_at_max_depth(binary_data):
    """
    Avec max_depth=0, la racine doit être directement
    une feuille.
    """

    X_train, y_train, _, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
        max_depth=0,
    )

    assert tree.feature_index is None
    assert tree.left is None
    assert tree.right is None


def test_build_tree_respects_max_depth(binary_data):
    """
    La profondeur réelle ne doit jamais dépasser max_depth.
    """

    X_train, y_train, _, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
        max_depth=2,
    )

    def check_depth(node):
        if node is None:
            return

        assert node.depth <= 2

        check_depth(node.left)
        check_depth(node.right)

    check_depth(tree)


def test_build_tree_with_min_samples_split(binary_data):
    """
    Un min_samples_split élevé peut empêcher certaines
    séparations.
    """

    X_train, y_train, _, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
        max_depth=5,
        min_samples_split=100,
    )

    assert tree.feature_index is None


def test_build_tree_pure_dataset_is_leaf():
    """
    Un dataset contenant une seule classe doit
    produire directement une feuille.
    """

    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )

    y = np.array(
        [1, 1, 1]
    )

    tree = build_tree(
        X,
        y,
    )

    assert tree.feature_index is None
    assert tree.prediction == 1


# ============================================================
# TESTS PREDICT TREE
# ============================================================


def test_predict_tree_binary(binary_data):
    """
    Vérifie les prédictions sur un problème binaire.
    """

    X_train, y_train, X_test, y_test = binary_data

    tree = build_tree(
        X_train,
        y_train,
        max_depth=5,
    )

    predictions = predict_tree(
        tree,
        X_test,
    )

    assert predictions.shape == y_test.shape
    assert np.array_equal(
        predictions,
        y_test,
    )


def test_predict_tree_multiclass(multiclass_data):
    """
    Vérifie les prédictions multiclasses.
    """

    X_train, y_train, X_test, y_test = multiclass_data

    tree = build_tree(
        X_train,
        y_train,
        max_depth=5,
    )

    predictions = predict_tree(
        tree,
        X_test,
    )

    assert predictions.shape == y_test.shape
    assert np.array_equal(
        predictions,
        y_test,
    )


def test_predict_tree_single_observation(binary_data):
    """
    Une observation 1D doit être acceptée.
    """

    X_train, y_train, _, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
    )

    prediction = predict_tree(
        tree,
        np.array([1.5, 1.5]),
    )

    assert prediction.shape == (1,)
    assert prediction[0] == 0


def test_predict_tree_returns_numpy_array(binary_data):
    """
    Les prédictions doivent être retournées sous forme
    de tableau NumPy.
    """

    X_train, y_train, X_test, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
    )

    predictions = predict_tree(
        tree,
        X_test,
    )

    assert isinstance(
        predictions,
        np.ndarray,
    )


def test_predict_tree_accepts_python_lists(binary_data):
    """
    predict_tree doit accepter une liste Python.
    """

    X_train, y_train, _, _ = binary_data

    tree = build_tree(
        X_train,
        y_train,
    )

    predictions = predict_tree(
        tree,
        [
            [1.5, 1.5],
            [5.5, 5.5],
        ],
    )

    assert len(predictions) == 2
    assert predictions[0] == 0
    assert predictions[1] == 1


# ============================================================
# TESTS ACCURACY
# ============================================================


def test_accuracy_score_perfect():
    """
    Accuracy parfaite = 1.0.
    """

    y_true = np.array(
        [0, 1, 0, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1]
    )

    result = accuracy_score(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(1.0)


def test_accuracy_score_zero():
    """
    Aucune prédiction correcte = 0.0.
    """

    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [1, 1, 0, 0]
    )

    result = accuracy_score(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(0.0)


def test_accuracy_score_partial():
    """
    Vérifie une accuracy partielle.
    """

    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 1, 0]
    )

    result = accuracy_score(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(0.5)


# ============================================================
# TESTS PIPELINE
# ============================================================


def test_decision_tree_returns_expected_keys(binary_data):
    """
    Le pipeline doit retourner toutes les informations
    nécessaires à l'interface.
    """

    X_train, y_train, X_test, y_test = binary_data

    result = decision_tree(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    expected_keys = {
        "tree",
        "predictions",
        "accuracy",
        "max_depth",
        "min_samples_split",
        "n_train_samples",
        "n_test_samples",
        "n_features",
        "tree_depth",
        "n_nodes",
        "n_classes",
    }

    assert expected_keys.issubset(
        result.keys()
    )


def test_decision_tree_binary(binary_data):
    """
    Vérifie le pipeline complet sur une classification binaire.
    """

    X_train, y_train, X_test, y_test = binary_data

    result = decision_tree(
        X_train,
        y_train,
        X_test,
        y_test,
        max_depth=5,
    )

    assert isinstance(
        result["tree"],
        TreeNode,
    )

    assert np.array_equal(
        result["predictions"],
        y_test,
    )

    assert result["accuracy"] == pytest.approx(1.0)


def test_decision_tree_multiclass(multiclass_data):
    """
    Vérifie le pipeline complet en classification multiclasses.
    """

    X_train, y_train, X_test, y_test = multiclass_data

    result = decision_tree(
        X_train,
        y_train,
        X_test,
        y_test,
        max_depth=5,
    )

    assert result["n_classes"] == 3

    assert np.array_equal(
        result["predictions"],
        y_test,
    )

    assert result["accuracy"] == pytest.approx(1.0)


def test_decision_tree_without_y_test(binary_data):
    """
    y_test étant optionnel, accuracy doit être None
    lorsqu'il n'est pas fourni.
    """

    X_train, y_train, X_test, _ = binary_data

    result = decision_tree(
        X_train,
        y_train,
        X_test,
    )

    assert result["accuracy"] is None
    assert len(result["predictions"]) == len(X_test)


def test_decision_tree_accepts_python_lists(binary_data):
    """
    Le pipeline doit accepter des listes Python.
    """

    X_train, y_train, X_test, y_test = binary_data

    result = decision_tree(
        X_train.tolist(),
        y_train.tolist(),
        X_test.tolist(),
        y_test.tolist(),
    )

    assert result["accuracy"] == pytest.approx(1.0)


def test_decision_tree_three_features():
    """
    Vérifie le fonctionnement avec trois caractéristiques.
    """

    X_train = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
            [5.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
            [6.0, 5.0, 6.0],
            [6.0, 6.0, 6.0],
        ]
    )

    y_train = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    X_test = np.array(
        [
            [1.5, 1.5, 1.5],
            [5.5, 5.5, 5.5],
        ]
    )

    y_test = np.array(
        [0, 1]
    )

    result = decision_tree(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert result["n_features"] == 3
    assert result["accuracy"] == pytest.approx(1.0)


# ============================================================
# TESTS PARAMÈTRES
# ============================================================


def test_decision_tree_max_depth_zero(binary_data):
    """
    max_depth=0 doit produire un arbre composé
    uniquement d'une feuille.
    """

    X_train, y_train, X_test, _ = binary_data

    result = decision_tree(
        X_train,
        y_train,
        X_test,
        max_depth=0,
    )

    assert result["tree"].feature_index is None
    assert result["tree_depth"] == 0
    assert result["n_nodes"] == 1


def test_decision_tree_max_depth_one(binary_data):
    """
    max_depth=1 ne doit pas produire de nœud
    à une profondeur supérieure à 1.
    """

    X_train, y_train, X_test, _ = binary_data

    result = decision_tree(
        X_train,
        y_train,
        X_test,
        max_depth=1,
    )

    assert result["tree_depth"] <= 1


def test_decision_tree_min_samples_split(binary_data):
    """
    Vérifie que min_samples_split est conservé
    dans les résultats.
    """

    X_train, y_train, X_test, _ = binary_data

    result = decision_tree(
        X_train,
        y_train,
        X_test,
        min_samples_split=4,
    )

    assert result["min_samples_split"] == 4


# ============================================================
# TESTS VALIDATION
# ============================================================


def test_invalid_X_dimension():
    """
    X doit être 2D.
    """

    X = np.array(
        [1.0, 2.0, 3.0]
    )

    y = np.array(
        [0, 1, 0]
    )

    with pytest.raises(ValueError):
        build_tree(X, y)


def test_invalid_y_dimension():
    """
    y doit être 1D.
    """

    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = np.array(
        [
            [0],
            [1],
            [0],
        ]
    )

    with pytest.raises(ValueError):
        build_tree(X, y)


def test_mismatched_X_y_lengths():
    """
    X et y doivent avoir le même nombre d'observations.
    """

    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = np.array(
        [0, 1]
    )

    with pytest.raises(ValueError):
        build_tree(X, y)


def test_empty_X():
    """
    X vide doit être refusé.
    """

    X = np.empty(
        (0, 2)
    )

    y = np.array([])

    with pytest.raises(ValueError):
        build_tree(X, y)


def test_non_numeric_X():
    """
    Les caractéristiques doivent être numériques.
    """

    X = np.array(
        [
            ["a", "b"],
            ["c", "d"],
        ]
    )

    y = np.array(
        [0, 1]
    )

    with pytest.raises(ValueError):
        build_tree(X, y)


def test_nan_in_X():
    """
    Les valeurs NaN doivent être refusées.
    """

    X = np.array(
        [
            [1.0],
            [np.nan],
            [3.0],
        ]
    )

    y = np.array(
        [0, 1, 0]
    )

    with pytest.raises(ValueError):
        build_tree(X, y)


def test_infinite_value_in_X():
    """
    Les valeurs infinies doivent être refusées.
    """

    X = np.array(
        [
            [1.0],
            [np.inf],
            [3.0],
        ]
    )

    y = np.array(
        [0, 1, 0]
    )

    with pytest.raises(ValueError):
        build_tree(X, y)


def test_invalid_max_depth_type(binary_data):
    """
    max_depth doit être un entier.
    """

    X_train, y_train, X_test, _ = binary_data

    with pytest.raises(TypeError):
        decision_tree(
            X_train,
            y_train,
            X_test,
            max_depth=2.5,
        )


def test_invalid_max_depth_value(binary_data):
    """
    max_depth ne peut pas être négatif.
    """

    X_train, y_train, X_test, _ = binary_data

    with pytest.raises(ValueError):
        decision_tree(
            X_train,
            y_train,
            X_test,
            max_depth=-1,
        )


def test_invalid_min_samples_split_type(binary_data):
    """
    min_samples_split doit être un entier.
    """

    X_train, y_train, X_test, _ = binary_data

    with pytest.raises(TypeError):
        decision_tree(
            X_train,
            y_train,
            X_test,
            min_samples_split=2.5,
        )


def test_invalid_min_samples_split_value(binary_data):
    """
    min_samples_split doit être >= 2.
    """

    X_train, y_train, X_test, _ = binary_data

    with pytest.raises(ValueError):
        decision_tree(
            X_train,
            y_train,
            X_test,
            min_samples_split=1,
        )


def test_invalid_X_test_feature_count(binary_data):
    """
    X_test doit avoir le même nombre de caractéristiques
    que X_train.
    """

    X_train, y_train, _, _ = binary_data

    X_test = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    with pytest.raises(ValueError):
        decision_tree(
            X_train,
            y_train,
            X_test,
        )


def test_invalid_y_test_length(binary_data):
    """
    y_test doit avoir la même longueur que X_test.
    """

    X_train, y_train, X_test, _ = binary_data

    y_test = np.array(
        [0]
    )

    with pytest.raises(ValueError):
        decision_tree(
            X_train,
            y_train,
            X_test,
            y_test,
        )


def test_predict_tree_invalid_tree():
    """
    predict_tree doit refuser un objet qui n'est pas un TreeNode.
    """

    X = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    with pytest.raises(TypeError):
        predict_tree(
            "not a tree",
            X,
        )


def test_accuracy_mismatched_lengths():
    """
    accuracy_score doit refuser deux tableaux
    de longueurs différentes.
    """

    y_true = np.array(
        [0, 1, 0]
    )

    y_pred = np.array(
        [0, 1]
    )

    with pytest.raises(ValueError):
        accuracy_score(
            y_true,
            y_pred,
        )


def test_accuracy_empty():
    """
    accuracy_score doit refuser un tableau vide.
    """

    with pytest.raises(ValueError):
        accuracy_score(
            np.array([]),
            np.array([]),
        )

