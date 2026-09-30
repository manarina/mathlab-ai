
"""
K-Nearest Neighbors (K-NN)
==========================

Implémentation pédagogique de l'algorithme K-NN pour la classification.

Principe
--------
Pour une observation donnée :

1. Calculer la distance entre l'observation et toutes les observations
   d'entraînement.
2. Sélectionner les K observations les plus proches.
3. Examiner leurs classes.
4. Attribuer à l'observation la classe majoritaire.

Distance euclidienne
--------------------
Pour deux observations x et y :

    d(x, y) = sqrt(sum((x_j - y_j)^2))

L'implémentation utilise uniquement NumPy afin de garder le fonctionnement
du modèle transparent et facilement explicable mathématiquement.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


# ============================================================================
# VALIDATION
# ============================================================================


def _validate_input(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Valide et convertit les données utilisées par K-NN.

    Parameters
    ----------
    X_train : np.ndarray
        Données d'entraînement de forme (n_samples, n_features).

    y_train : np.ndarray
        Classes associées aux données d'entraînement.

    X_test : np.ndarray | None, optional
        Données sur lesquelles effectuer les prédictions.

    y_test : np.ndarray | None, optional
        Classes réelles des données de test.

    k : int, default=3
        Nombre de voisins utilisés pour la classification.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)

    Raises
    ------
    ValueError
        Si les données sont invalides.

    TypeError
        Si k n'est pas un entier.
    """

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    if X_test is not None:
        X_test = np.asarray(X_test, dtype=float)

    if y_test is not None:
        y_test = np.asarray(y_test)

    # ------------------------------------------------------------------
    # Validation de X_train
    # ------------------------------------------------------------------

    if X_train.ndim != 2:
        raise ValueError(
            "X_train doit être un tableau 2D de forme "
            "(n_samples, n_features)."
        )

    if X_train.shape[0] == 0:
        raise ValueError("X_train ne peut pas être vide.")

    if X_train.shape[1] == 0:
        raise ValueError(
            "X_train doit contenir au moins une caractéristique."
        )

    # ------------------------------------------------------------------
    # Validation de y_train
    # ------------------------------------------------------------------

    if y_train.ndim != 1:
        raise ValueError("y_train doit être un tableau 1D.")

    if len(y_train) != len(X_train):
        raise ValueError(
            "X_train et y_train doivent contenir le même nombre "
            "d'observations."
        )

    if len(y_train) == 0:
        raise ValueError("y_train ne peut pas être vide.")

    # ------------------------------------------------------------------
    # Validation des valeurs numériques de X_train
    # ------------------------------------------------------------------

    if not np.all(np.isfinite(X_train)):
        raise ValueError(
            "X_train doit contenir uniquement des valeurs numériques "
            "finies."
        )

    # ------------------------------------------------------------------
    # Validation de k
    # ------------------------------------------------------------------

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k doit être un entier.")

    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")

    if k > len(X_train):
        raise ValueError(
            "k ne peut pas être supérieur au nombre d'observations "
            "d'entraînement."
        )

    # ------------------------------------------------------------------
    # Validation de X_test
    # ------------------------------------------------------------------

    if X_test is not None:

        if X_test.ndim != 2:
            raise ValueError(
                "X_test doit être un tableau 2D de forme "
                "(n_samples, n_features)."
            )

        if X_test.shape[0] == 0:
            raise ValueError("X_test ne peut pas être vide.")

        if X_test.shape[1] != X_train.shape[1]:
            raise ValueError(
                "X_train et X_test doivent avoir le même nombre "
                "de caractéristiques."
            )

        if not np.all(np.isfinite(X_test)):
            raise ValueError(
                "X_test doit contenir uniquement des valeurs "
                "numériques finies."
            )

    # ------------------------------------------------------------------
    # Validation de y_test
    # ------------------------------------------------------------------

    if y_test is not None:

        if y_test.ndim != 1:
            raise ValueError("y_test doit être un tableau 1D.")

        if X_test is None:
            raise ValueError(
                "X_test doit être fourni lorsque y_test est fourni."
            )

        if len(y_test) != len(X_test):
            raise ValueError(
                "X_test et y_test doivent contenir le même nombre "
                "d'observations."
            )

    return X_train, y_train, X_test, y_test


# ============================================================================
# DISTANCE EUCLIDIENNE
# ============================================================================


def euclidean_distance(
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    """
    Calcule la distance euclidienne entre deux observations.

    Formule
    -------
    d(x1, x2) = sqrt(sum((x1 - x2)^2))

    Parameters
    ----------
    x1 : np.ndarray
        Première observation.

    x2 : np.ndarray
        Deuxième observation.

    Returns
    -------
    float
        Distance euclidienne.

    Raises
    ------
    ValueError
        Si les observations ne sont pas des vecteurs 1D,
        n'ont pas la même dimension ou contiennent des valeurs
        non finies.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError(
            "x1 et x2 doivent être des vecteurs 1D."
        )

    if x1.shape != x2.shape:
        raise ValueError(
            "x1 et x2 doivent avoir la même dimension."
        )

    if not np.all(np.isfinite(x1)):
        raise ValueError(
            "x1 doit contenir uniquement des valeurs numériques finies."
        )

    if not np.all(np.isfinite(x2)):
        raise ValueError(
            "x2 doit contenir uniquement des valeurs numériques finies."
        )

    distance = np.sqrt(np.sum((x1 - x2) ** 2))

    return float(distance)


# ============================================================================
# RECHERCHE DES VOISINS
# ============================================================================


def get_neighbors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recherche les K plus proches voisins d'une observation.

    Parameters
    ----------
    X_train : np.ndarray
        Données d'entraînement de forme (n_samples, n_features).

    y_train : np.ndarray
        Classes des observations d'entraînement.

    x_query : np.ndarray
        Observation à classifier.

    k : int, default=3
        Nombre de voisins à sélectionner.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Deux tableaux :

        - indices des K voisins les plus proches
        - distances correspondantes

        Les voisins sont retournés dans l'ordre croissant
        de leur distance.
    """

    X_train, y_train, _, _ = _validate_input(
        X_train,
        y_train,
        k=k,
    )

    x_query = np.asarray(x_query, dtype=float)

    if x_query.ndim != 1:
        raise ValueError(
            "x_query doit être un vecteur 1D."
        )

    if x_query.shape[0] != X_train.shape[1]:
        raise ValueError(
            "x_query doit avoir le même nombre de caractéristiques "
            "que X_train."
        )

    if not np.all(np.isfinite(x_query)):
        raise ValueError(
            "x_query doit contenir uniquement des valeurs "
            "numériques finies."
        )

    # Calcul vectorisé des distances
    distances = np.sqrt(
        np.sum((X_train - x_query) ** 2, axis=1)
    )

    # Tri des indices selon la distance.
    # np.argsort est stable afin d'obtenir un comportement déterministe
    # lorsque plusieurs distances sont identiques.
    sorted_indices = np.argsort(
        distances,
        kind="stable",
    )

    neighbor_indices = sorted_indices[:k]
    neighbor_distances = distances[neighbor_indices]

    return neighbor_indices, neighbor_distances


# ============================================================================
# PRÉDICTION K-NN
# ============================================================================


def _majority_vote(labels: np.ndarray) -> object:
    """
    Détermine la classe majoritaire parmi les voisins.

    En cas d'égalité, la classe du premier voisin rencontré
    dans l'ordre des distances est conservée. Cela garantit
    un comportement déterministe.

    Parameters
    ----------
    labels : np.ndarray
        Labels des K voisins.

    Returns
    -------
    object
        Classe prédite.
    """

    counts = Counter(labels.tolist())

    max_count = max(counts.values())

    # Les labels sont parcourus dans l'ordre des voisins.
    # Le premier label ayant le nombre maximal d'occurrences
    # est retenu en cas d'égalité.
    for label in labels:
        if counts[label] == max_count:
            return label

    # Cette ligne ne devrait jamais être atteinte.
    raise RuntimeError("Impossible de déterminer la classe majoritaire.")


def predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Prédit les classes des observations de test avec K-NN.

    Parameters
    ----------
    X_train : np.ndarray
        Données d'entraînement.

    y_train : np.ndarray
        Classes des données d'entraînement.

    X_test : np.ndarray
        Données à classifier.

    k : int, default=3
        Nombre de voisins.

    Returns
    -------
    np.ndarray
        Classes prédites pour chaque observation de X_test.
    """

    X_train, y_train, X_test, _ = _validate_input(
        X_train,
        y_train,
        X_test,
        k=k,
    )

    predictions = []

    for x_query in X_test:

        neighbor_indices, _ = get_neighbors(
            X_train,
            y_train,
            x_query,
            k=k,
        )

        neighbor_labels = y_train[neighbor_indices]

        prediction = _majority_vote(neighbor_labels)

        predictions.append(prediction)

    return np.asarray(predictions)


# ============================================================================
# ACCURACY
# ============================================================================


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcule la précision de classification.

    Formule
    -------
    Accuracy = nombre de prédictions correctes / nombre total
               de prédictions

    Parameters
    ----------
    y_true : np.ndarray
        Classes réelles.

    y_pred : np.ndarray
        Classes prédites.

    Returns
    -------
    float
        Score d'accuracy compris entre 0 et 1.

    Raises
    ------
    ValueError
        Si les tableaux sont invalides ou de tailles différentes.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError("y_true doit être un tableau 1D.")

    if y_pred.ndim != 1:
        raise ValueError("y_pred doit être un tableau 1D.")

    if len(y_true) == 0:
        raise ValueError(
            "y_true ne peut pas être vide."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true et y_pred doivent avoir la même longueur."
        )

    return float(np.mean(y_true == y_pred))


# ============================================================================
# PIPELINE COMPLET
# ============================================================================


def knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray | None = None,
    k: int = 3,
) -> dict:
    """
    Pipeline complet K-NN.

    Parameters
    ----------
    X_train : np.ndarray
        Données d'entraînement.

    y_train : np.ndarray
        Classes des données d'entraînement.

    X_test : np.ndarray
        Données à classifier.

    y_test : np.ndarray | None, optional
        Classes réelles des données de test.
        Si elles sont fournies, l'accuracy est calculée.

    k : int, default=3
        Nombre de voisins.

    Returns
    -------
    dict
        Résultats du modèle contenant :

        - predictions
        - accuracy
        - k
        - n_train_samples
        - n_test_samples
        - n_features
    """

    X_train, y_train, X_test, y_test = _validate_input(
        X_train,
        y_train,
        X_test,
        y_test,
        k=k,
    )

    predictions = predict_knn(
        X_train,
        y_train,
        X_test,
        k=k,
    )

    if y_test is not None:
        accuracy = accuracy_score(
            y_test,
            predictions,
        )
    else:
        accuracy = None

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "k": int(k),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    }

