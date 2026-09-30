
"""
Decision Tree - Classification
==============================

Implémentation pédagogique d'un arbre de décision pour la classification.

Caractéristiques :
- NumPy uniquement
- Classification binaire et multiclasses
- Critère d'impureté de Gini
- Recherche du meilleur seuil
- Construction récursive de l'arbre
- Profondeur maximale configurable
- Nombre minimal d'observations configurable
- Prédiction sur de nouvelles observations
- Calcul de l'accuracy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ============================================================
# STRUCTURE D'UN NŒUD
# ============================================================


@dataclass
class TreeNode:
    """
    Représente un nœud de l'arbre de décision.

    Un nœud interne contient :
        - feature_index
        - threshold
        - left
        - right

    Une feuille contient :
        - prediction
    """

    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    prediction: Any = None
    depth: int = 0


# ============================================================
# VALIDATION
# ============================================================


def _validate_input(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valide les données d'entrée X et y.

    Parameters
    ----------
    X : array-like
        Matrice de forme (n_samples, n_features).

    y : array-like
        Tableau des labels de forme (n_samples,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Données converties en tableaux NumPy.

    Raises
    ------
    ValueError
        Si les données sont invalides.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(
            "X doit être une matrice 2D "
            "(n_samples, n_features)."
        )

    if y.ndim != 1:
        raise ValueError(
            "y doit être un tableau 1D."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X et y doivent avoir le même nombre "
            "d'observations."
        )

    if X.shape[0] == 0:
        raise ValueError(
            "Les données ne peuvent pas être vides."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "X doit contenir au moins une caractéristique."
        )

    try:
        X = X.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Les caractéristiques de X doivent être numériques."
        ) from exc

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X ne doit pas contenir de valeurs NaN ou infinies."
        )

    if y.size == 0:
        raise ValueError(
            "y ne peut pas être vide."
        )

    return X, y


# ============================================================
# GINI IMPURITY
# ============================================================


def gini_impurity(y: np.ndarray) -> float:
    """
    Calcule l'impureté de Gini.

    Formule :

        Gini = 1 - Σ p_i²

    où p_i est la proportion de la classe i.

    Parameters
    ----------
    y : array-like
        Labels.

    Returns
    -------
    float
        Impureté de Gini.

    Examples
    --------
    >>> gini_impurity([0, 0, 1, 1])
    0.5

    >>> gini_impurity([1, 1, 1])
    0.0
    """

    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(
            "y doit être un tableau 1D."
        )

    if y.size == 0:
        return 0.0

    _, counts = np.unique(
        y,
        return_counts=True,
    )

    probabilities = counts / y.size

    return float(
        1.0 - np.sum(probabilities ** 2)
    )


# ============================================================
# CLASSE MAJORITAIRE
# ============================================================


def _majority_class(y: np.ndarray) -> Any:
    """
    Retourne la classe majoritaire.

    En cas d'égalité, le résultat est déterministe
    grâce à np.unique().
    """

    values, counts = np.unique(
        y,
        return_counts=True,
    )

    return values[np.argmax(counts)]


# ============================================================
# GINI PONDÉRÉ
# ============================================================


def _weighted_gini(
    y_left: np.ndarray,
    y_right: np.ndarray,
) -> float:
    """
    Calcule l'impureté de Gini pondérée
    après une séparation.
    """

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_left == 0 or n_right == 0:
        return float("inf")

    left_weight = n_left / n_total
    right_weight = n_right / n_total

    return float(
        left_weight * gini_impurity(y_left)
        + right_weight * gini_impurity(y_right)
    )


# ============================================================
# MEILLEURE SÉPARATION
# ============================================================


def best_split(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any] | None:
    """
    Recherche la meilleure séparation de X et y.

    Pour chaque caractéristique, les seuils possibles
    sont placés entre deux valeurs distinctes.

    Parameters
    ----------
    X : array-like
        Matrice des caractéristiques.

    y : array-like
        Labels.

    Returns
    -------
    dict | None
        Dictionnaire contenant :

            {
                "feature_index": int,
                "threshold": float,
                "gini": float,
            }

        Retourne None lorsqu'aucune séparation
        valide n'est possible.
    """

    X, y = _validate_input(X, y)

    n_samples, n_features = X.shape

    if n_samples < 2:
        return None

    if np.unique(y).size <= 1:
        return None

    best_gini = float("inf")
    best_feature = None
    best_threshold = None

    for feature_index in range(n_features):

        feature_values = X[:, feature_index]

        unique_values = np.unique(feature_values)

        if unique_values.size < 2:
            continue

        thresholds = (
            unique_values[:-1]
            + unique_values[1:]
        ) / 2.0

        for threshold in thresholds:

            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold

            if not np.any(left_mask):
                continue

            if not np.any(right_mask):
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            current_gini = _weighted_gini(
                y_left,
                y_right,
            )

            if current_gini < best_gini:
                best_gini = current_gini
                best_feature = feature_index
                best_threshold = float(threshold)

    if best_feature is None:
        return None

    return {
        "feature_index": int(best_feature),
        "threshold": float(best_threshold),
        "gini": float(best_gini),
    }


# ============================================================
# CONSTRUCTION DE L'ARBRE
# ============================================================


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    min_samples_split: int = 2,
    depth: int = 0,
) -> TreeNode:
    """
    Construit récursivement un arbre de décision.

    Parameters
    ----------
    X : array-like
        Données d'entraînement.

    y : array-like
        Labels d'entraînement.

    max_depth : int, default=5
        Profondeur maximale de l'arbre.

    min_samples_split : int, default=2
        Nombre minimal d'observations nécessaire
        pour effectuer une séparation.

    depth : int, default=0
        Profondeur actuelle du nœud.

    Returns
    -------
    TreeNode
        Nœud racine du sous-arbre.
    """

    X, y = _validate_input(X, y)

    if not isinstance(max_depth, (int, np.integer)):
        raise TypeError(
            "max_depth doit être un entier."
        )

    if not isinstance(min_samples_split, (int, np.integer)):
        raise TypeError(
            "min_samples_split doit être un entier."
        )

    if not isinstance(depth, (int, np.integer)):
        raise TypeError(
            "depth doit être un entier."
        )

    if max_depth < 0:
        raise ValueError(
            "max_depth doit être supérieur ou égal à 0."
        )

    if min_samples_split < 2:
        raise ValueError(
            "min_samples_split doit être supérieur ou égal à 2."
        )

    if depth < 0:
        raise ValueError(
            "depth doit être supérieur ou égal à 0."
        )

    # Classe majoritaire utilisée par la feuille.
    prediction = _majority_class(y)

    # --------------------------------------------------------
    # CONDITION D'ARRÊT 1 :
    # toutes les observations ont la même classe
    # --------------------------------------------------------

    if np.unique(y).size == 1:
        return TreeNode(
            prediction=prediction,
            depth=int(depth),
        )

    # --------------------------------------------------------
    # CONDITION D'ARRÊT 2 :
    # profondeur maximale atteinte
    # --------------------------------------------------------

    if depth >= max_depth:
        return TreeNode(
            prediction=prediction,
            depth=int(depth),
        )

    # --------------------------------------------------------
    # CONDITION D'ARRÊT 3 :
    # pas assez d'observations
    # --------------------------------------------------------

    if len(y) < min_samples_split:
        return TreeNode(
            prediction=prediction,
            depth=int(depth),
        )

    # --------------------------------------------------------
    # RECHERCHE DU MEILLEUR SPLIT
    # --------------------------------------------------------

    split = best_split(X, y)

    if split is None:
        return TreeNode(
            prediction=prediction,
            depth=int(depth),
        )

    feature_index = split["feature_index"]
    threshold = split["threshold"]

    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold

    # Sécurité supplémentaire.
    if not np.any(left_mask) or not np.any(right_mask):
        return TreeNode(
            prediction=prediction,
            depth=int(depth),
        )

    # --------------------------------------------------------
    # CONSTRUCTION DU SOUS-ARBRE GAUCHE
    # --------------------------------------------------------

    left_node = build_tree(
        X[left_mask],
        y[left_mask],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        depth=depth + 1,
    )

    # --------------------------------------------------------
    # CONSTRUCTION DU SOUS-ARBRE DROIT
    # --------------------------------------------------------

    right_node = build_tree(
        X[right_mask],
        y[right_mask],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        depth=depth + 1,
    )

    return TreeNode(
        feature_index=int(feature_index),
        threshold=float(threshold),
        left=left_node,
        right=right_node,
        prediction=prediction,
        depth=int(depth),
    )


# ============================================================
# PRÉDICTION D'UNE OBSERVATION
# ============================================================


def _predict_single(
    node: TreeNode,
    x: np.ndarray,
) -> Any:
    """
    Prédit la classe d'une seule observation.
    """

    # Si le nœud est une feuille.
    if node.feature_index is None:
        return node.prediction

    if node.left is None or node.right is None:
        return node.prediction

    if x[node.feature_index] <= node.threshold:
        return _predict_single(
            node.left,
            x,
        )

    return _predict_single(
        node.right,
        x,
    )


# ============================================================
# PRÉDICTION
# ============================================================


def predict_tree(
    tree: TreeNode,
    X: np.ndarray,
) -> np.ndarray:
    """
    Prédit les classes pour plusieurs observations.

    Parameters
    ----------
    tree : TreeNode
        Arbre entraîné.

    X : array-like
        Observations à classifier.

    Returns
    -------
    np.ndarray
        Classes prédites.
    """

    if not isinstance(tree, TreeNode):
        raise TypeError(
            "tree doit être une instance de TreeNode."
        )

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError(
            "X doit être une matrice 2D."
        )

    if X.shape[0] == 0:
        raise ValueError(
            "X ne peut pas être vide."
        )

    try:
        X = X.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Les caractéristiques doivent être numériques."
        ) from exc

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X ne doit pas contenir de valeurs NaN ou infinies."
        )

    # Trouve l'index de caractéristique maximum utilisé
    # par l'arbre afin de vérifier la dimension de X.
    max_feature_index = _max_feature_index(tree)

    if (
        max_feature_index is not None
        and max_feature_index >= X.shape[1]
    ):
        raise ValueError(
            "X ne contient pas suffisamment de caractéristiques."
        )

    predictions = [
        _predict_single(tree, observation)
        for observation in X
    ]

    return np.asarray(predictions)


# ============================================================
# CARACTÉRISTIQUE MAXIMALE UTILISÉE
# ============================================================


def _max_feature_index(
    node: TreeNode,
) -> int | None:
    """
    Retourne l'index maximal de caractéristique
    utilisé dans l'arbre.
    """

    if node.feature_index is None:
        return None

    indices = [node.feature_index]

    if node.left is not None:
        left_index = _max_feature_index(node.left)

        if left_index is not None:
            indices.append(left_index)

    if node.right is not None:
        right_index = _max_feature_index(node.right)

        if right_index is not None:
            indices.append(right_index)

    return max(indices)


# ============================================================
# ACCURACY
# ============================================================


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcule l'accuracy.

    Formule :

        Accuracy =
            nombre de prédictions correctes
            / nombre total de prédictions

    Parameters
    ----------
    y_true : array-like
        Labels réels.

    y_pred : array-like
        Labels prédits.

    Returns
    -------
    float
        Accuracy comprise entre 0 et 1.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError(
            "y_true doit être un tableau 1D."
        )

    if y_pred.ndim != 1:
        raise ValueError(
            "y_pred doit être un tableau 1D."
        )

    if y_true.size == 0:
        raise ValueError(
            "y_true ne peut pas être vide."
        )

    if y_true.size != y_pred.size:
        raise ValueError(
            "y_true et y_pred doivent avoir "
            "la même longueur."
        )

    return float(
        np.mean(y_true == y_pred)
    )


# ============================================================
# PROFONDEUR RÉELLE DE L'ARBRE
# ============================================================


def _tree_depth(
    node: TreeNode,
) -> int:
    """
    Retourne la profondeur maximale de l'arbre.
    """

    if node.feature_index is None:
        return int(node.depth)

    depths = [node.depth]

    if node.left is not None:
        depths.append(
            _tree_depth(node.left)
        )

    if node.right is not None:
        depths.append(
            _tree_depth(node.right)
        )

    return int(max(depths))


# ============================================================
# NOMBRE DE NŒUDS
# ============================================================


def _count_nodes(
    node: TreeNode,
) -> int:
    """
    Compte le nombre total de nœuds.
    """

    count = 1

    if node.left is not None:
        count += _count_nodes(node.left)

    if node.right is not None:
        count += _count_nodes(node.right)

    return int(count)


# ============================================================
# PIPELINE COMPLET
# ============================================================


def decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray | None = None,
    max_depth: int = 5,
    min_samples_split: int = 2,
) -> dict[str, Any]:
    """
    Entraîne un arbre de décision et effectue les prédictions.

    Parameters
    ----------
    X_train : array-like
        Données d'entraînement.

    y_train : array-like
        Labels d'entraînement.

    X_test : array-like
        Données de test.

    y_test : array-like, optional
        Labels réels des données de test.

    max_depth : int, default=5
        Profondeur maximale.

    min_samples_split : int, default=2
        Nombre minimal d'observations nécessaire
        pour effectuer une séparation.

    Returns
    -------
    dict
        Résultats du modèle :

        {
            "tree": TreeNode,
            "predictions": np.ndarray,
            "accuracy": float | None,
            "max_depth": int,
            "min_samples_split": int,
            "n_train_samples": int,
            "n_test_samples": int,
            "n_features": int,
            "tree_depth": int,
            "n_nodes": int,
            "n_classes": int,
        }
    """

    # --------------------------------------------------------
    # VALIDATION DES DONNÉES D'ENTRAÎNEMENT
    # --------------------------------------------------------

    X_train, y_train = _validate_input(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # VALIDATION DES PARAMÈTRES
    # --------------------------------------------------------

    if not isinstance(max_depth, (int, np.integer)):
        raise TypeError(
            "max_depth doit être un entier."
        )

    if not isinstance(min_samples_split, (int, np.integer)):
        raise TypeError(
            "min_samples_split doit être un entier."
        )

    if max_depth < 0:
        raise ValueError(
            "max_depth doit être supérieur ou égal à 0."
        )

    if min_samples_split < 2:
        raise ValueError(
            "min_samples_split doit être supérieur ou égal à 2."
        )

    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)

    # --------------------------------------------------------
    # VALIDATION DE X_TEST
    # --------------------------------------------------------

    X_test = np.asarray(X_test)

    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    if X_test.ndim != 2:
        raise ValueError(
            "X_test doit être une matrice 2D."
        )

    if X_test.shape[0] == 0:
        raise ValueError(
            "X_test ne peut pas être vide."
        )

    try:
        X_test = X_test.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Les caractéristiques de X_test "
            "doivent être numériques."
        ) from exc

    if not np.all(np.isfinite(X_test)):
        raise ValueError(
            "X_test ne doit pas contenir "
            "de valeurs NaN ou infinies."
        )

    # --------------------------------------------------------
    # VÉRIFICATION DU NOMBRE DE CARACTÉRISTIQUES
    # --------------------------------------------------------

    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError(
            "X_train et X_test doivent avoir "
            "le même nombre de caractéristiques."
        )

    # --------------------------------------------------------
    # CONSTRUCTION DE L'ARBRE
    # --------------------------------------------------------

    tree = build_tree(
        X_train,
        y_train,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )

    # --------------------------------------------------------
    # PRÉDICTIONS
    # --------------------------------------------------------

    predictions = predict_tree(
        tree,
        X_test,
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = None

    if y_test is not None:

        y_test = np.asarray(y_test)

        if y_test.ndim != 1:
            raise ValueError(
                "y_test doit être un tableau 1D."
            )

        if len(y_test) != len(X_test):
            raise ValueError(
                "y_test et X_test doivent avoir "
                "le même nombre d'observations."
            )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

    # --------------------------------------------------------
    # RÉSULTAT
    # --------------------------------------------------------

    return {
        "tree": tree,
        "predictions": predictions,
        "accuracy": accuracy,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "tree_depth": _tree_depth(tree),
        "n_nodes": _count_nodes(tree),
        "n_classes": int(
            np.unique(y_train).size
        ),
    }
