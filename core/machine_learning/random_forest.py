"""
Random Forest - Classification
==============================

Implémentation pédagogique d'une forêt aléatoire
pour la classification.

Caractéristiques :
- NumPy uniquement
- Classification binaire et multiclasses
- Utilisation de l'implémentation TreeNode / Decision Tree
- Échantillonnage bootstrap
- Sélection aléatoire des caractéristiques
- Vote majoritaire
- Nombre d'arbres configurable
- Profondeur maximale configurable
- Nombre minimal d'observations configurable
- Prédiction sur de nouvelles observations
- Calcul de l'accuracy
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.machine_learning.decision_tree import (
    TreeNode,
    accuracy_score,
    build_tree,
    predict_tree,
)


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
        Labels de forme (n_samples,).

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
# VALIDATION DES PARAMÈTRES
# ============================================================


def _validate_parameters(
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    max_features: int | None,
    random_state: int | None,
) -> None:
    """
    Valide les paramètres du Random Forest.
    """

    if not isinstance(
        n_estimators,
        (int, np.integer),
    ):
        raise TypeError(
            "n_estimators doit être un entier."
        )

    if n_estimators < 1:
        raise ValueError(
            "n_estimators doit être supérieur ou égal à 1."
        )

    if not isinstance(
        max_depth,
        (int, np.integer),
    ):
        raise TypeError(
            "max_depth doit être un entier."
        )

    if max_depth < 0:
        raise ValueError(
            "max_depth doit être supérieur ou égal à 0."
        )

    if not isinstance(
        min_samples_split,
        (int, np.integer),
    ):
        raise TypeError(
            "min_samples_split doit être un entier."
        )

    if min_samples_split < 2:
        raise ValueError(
            "min_samples_split doit être supérieur ou égal à 2."
        )

    if max_features is not None:

        if not isinstance(
            max_features,
            (int, np.integer),
        ):
            raise TypeError(
                "max_features doit être un entier ou None."
            )

        if max_features < 1:
            raise ValueError(
                "max_features doit être supérieur ou égal à 1."
            )

    if random_state is not None:

        if not isinstance(
            random_state,
            (int, np.integer),
        ):
            raise TypeError(
                "random_state doit être un entier ou None."
            )


# ============================================================
# NOMBRE DE CARACTÉRISTIQUES
# ============================================================


def _default_max_features(
    n_features: int,
) -> int:
    """
    Calcule le nombre par défaut de caractéristiques
    utilisées par arbre.

    Pour une classification, on utilise :

        sqrt(n_features)

    avec un minimum de 1.
    """

    return max(
        1,
        int(np.sqrt(n_features)),
    )


# ============================================================
# ÉCHANTILLONNAGE BOOTSTRAP
# ============================================================


def bootstrap_sample(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un échantillon bootstrap.

    Chaque arbre reçoit un échantillon de même taille
    que le jeu de données original, avec remise.

    Parameters
    ----------
    X : array-like
        Données d'entraînement.

    y : array-like
        Labels.

    rng : np.random.Generator
        Générateur aléatoire.

    Returns
    -------
    tuple
        X_bootstrap, y_bootstrap, indices
    """

    X, y = _validate_input(X, y)

    n_samples = X.shape[0]

    indices = rng.integers(
        0,
        n_samples,
        size=n_samples,
    )

    return (
        X[indices],
        y[indices],
        indices,
    )


# ============================================================
# SÉLECTION DES CARACTÉRISTIQUES
# ============================================================


def select_features(
    n_features: int,
    max_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sélectionne aléatoirement des caractéristiques.

    Parameters
    ----------
    n_features : int
        Nombre total de caractéristiques.

    max_features : int
        Nombre de caractéristiques à sélectionner.

    rng : np.random.Generator
        Générateur aléatoire.

    Returns
    -------
    np.ndarray
        Indices des caractéristiques sélectionnées.
    """

    if not isinstance(
        n_features,
        (int, np.integer),
    ):
        raise TypeError(
            "n_features doit être un entier."
        )

    if not isinstance(
        max_features,
        (int, np.integer),
    ):
        raise TypeError(
            "max_features doit être un entier."
        )

    if n_features < 1:
        raise ValueError(
            "n_features doit être supérieur ou égal à 1."
        )

    if max_features < 1:
        raise ValueError(
            "max_features doit être supérieur ou égal à 1."
        )

    if max_features > n_features:
        raise ValueError(
            "max_features ne peut pas être supérieur "
            "au nombre de caractéristiques."
        )

    return np.sort(
        rng.choice(
            n_features,
            size=max_features,
            replace=False,
        )
    )


# ============================================================
# VOTE MAJORITAIRE
# ============================================================


def majority_vote(
    predictions: np.ndarray,
) -> Any:
    """
    Retourne la classe majoritaire.

    Parameters
    ----------
    predictions : array-like
        Prédictions des différents arbres.

    Returns
    -------
    Any
        Classe majoritaire.
    """

    predictions = np.asarray(predictions)

    if predictions.ndim != 1:
        raise ValueError(
            "predictions doit être un tableau 1D."
        )

    if predictions.size == 0:
        raise ValueError(
            "predictions ne peut pas être vide."
        )

    values, counts = np.unique(
        predictions,
        return_counts=True,
    )

    return values[np.argmax(counts)]


# ============================================================
# PRÉDICTION PAR VOTE
# ============================================================


def predict_forest(
    trees: list[TreeNode],
    feature_indices: list[np.ndarray],
    X: np.ndarray,
) -> np.ndarray:
    """
    Effectue les prédictions de la forêt.

    Chaque arbre prédit avec son propre sous-ensemble
    de caractéristiques.

    Parameters
    ----------
    trees : list[TreeNode]
        Arbres entraînés.

    feature_indices : list[np.ndarray]
        Caractéristiques utilisées par chaque arbre.

    X : array-like
        Observations à classifier.

    Returns
    -------
    np.ndarray
        Classes prédites.
    """

    if not isinstance(trees, list):
        raise TypeError(
            "trees doit être une liste."
        )

    if len(trees) == 0:
        raise ValueError(
            "La forêt doit contenir au moins un arbre."
        )

    if len(trees) != len(feature_indices):
        raise ValueError(
            "trees et feature_indices doivent avoir "
            "la même longueur."
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

    all_predictions = []

    for tree, indices in zip(
        trees,
        feature_indices,
    ):

        indices = np.asarray(indices)

        X_selected = X[:, indices]

        predictions = predict_tree(
            tree,
            X_selected,
        )

        all_predictions.append(
            predictions
        )

    all_predictions = np.asarray(
        all_predictions
    )

    final_predictions = [
        majority_vote(
            all_predictions[:, sample_index]
        )
        for sample_index in range(X.shape[0])
    ]

    return np.asarray(final_predictions)


# ============================================================
# PIPELINE COMPLET
# ============================================================


def random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray | None = None,
    n_estimators: int = 10,
    max_depth: int = 5,
    min_samples_split: int = 2,
    max_features: int | None = None,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """
    Entraîne une forêt aléatoire et effectue
    les prédictions.

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

    n_estimators : int, default=10
        Nombre d'arbres.

    max_depth : int, default=5
        Profondeur maximale de chaque arbre.

    min_samples_split : int, default=2
        Nombre minimal d'observations nécessaire
        pour effectuer une séparation.

    max_features : int or None, default=None
        Nombre de caractéristiques utilisées par arbre.

        Si None :
            sqrt(n_features)

    random_state : int or None, default=42
        Graine aléatoire pour rendre les résultats
        reproductibles.

    Returns
    -------
    dict
        Résultats du Random Forest.

    Exemple
    -------
    >>> result = random_forest(
    ...     [[1], [2], [5], [6]],
    ...     [0, 0, 1, 1],
    ...     [[1.5], [5.5]],
    ... )
    """

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    X_train, y_train = _validate_input(
        X_train,
        y_train,
    )

    _validate_parameters(
        n_estimators,
        max_depth,
        min_samples_split,
        max_features,
        random_state,
    )

    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)

    n_train_samples = X_train.shape[0]
    n_features = X_train.shape[1]

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

    if X_test.shape[1] != n_features:
        raise ValueError(
            "X_train et X_test doivent avoir "
            "le même nombre de caractéristiques."
        )

    # --------------------------------------------------------
    # VALIDATION DE Y_TEST
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NOMBRE DE FEATURES
    # --------------------------------------------------------

    if max_features is None:
        max_features = _default_max_features(
            n_features
        )
    else:
        max_features = int(max_features)

        if max_features > n_features:
            raise ValueError(
                "max_features ne peut pas être supérieur "
                "au nombre de caractéristiques."
            )

    # --------------------------------------------------------
    # GÉNÉRATEUR ALÉATOIRE
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # CONSTRUCTION DE LA FORÊT
    # --------------------------------------------------------

    trees: list[TreeNode] = []
    feature_indices: list[np.ndarray] = []

    for _ in range(n_estimators):

        # ----------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------

        X_bootstrap, y_bootstrap, _ = (
            bootstrap_sample(
                X_train,
                y_train,
                rng,
            )
        )

        # ----------------------------------------------------
        # Sélection aléatoire des caractéristiques
        # ----------------------------------------------------

        selected_features = select_features(
            n_features,
            max_features,
            rng,
        )

        X_bootstrap_selected = (
            X_bootstrap[:, selected_features]
        )

        # ----------------------------------------------------
        # Construction de l'arbre
        # ----------------------------------------------------

        tree = build_tree(
            X_bootstrap_selected,
            y_bootstrap,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        trees.append(tree)
        feature_indices.append(
            selected_features
        )

    # --------------------------------------------------------
    # PRÉDICTIONS
    # --------------------------------------------------------

    predictions = predict_forest(
        trees,
        feature_indices,
        X_test,
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = None

    if y_test is not None:
        accuracy = accuracy_score(
            y_test,
            predictions,
        )

    # --------------------------------------------------------
    # RÉSULTAT
    # --------------------------------------------------------

    return {
        "trees": trees,
        "feature_indices": feature_indices,
        "predictions": predictions,
        "accuracy": accuracy,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "random_state": random_state,
        "n_train_samples": int(n_train_samples),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(n_features),
        "n_classes": int(
            np.unique(y_train).size
        ),
    }

