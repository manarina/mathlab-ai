
"""
K-Means Clustering
==================

Implémentation pédagogique de l'algorithme K-Means
pour MathLab AI.

Fonctionnalités :
- Validation des données
- Initialisation des centroïdes
- Calcul des distances euclidiennes
- Attribution des observations aux clusters
- Recalcul des centroïdes
- Détection de convergence
- Calcul de l'inertie
- Prédiction de nouveaux points
- Pipeline complet K-Means

Dépendance :
- NumPy uniquement
"""

from __future__ import annotations

import numpy as np


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(
    X: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Valide les données d'entrée du K-Means.

    Parameters
    ----------
    X : np.ndarray
        Matrice de données de forme (n_samples, n_features).
    n_clusters : int
        Nombre de clusters.

    Returns
    -------
    np.ndarray
        Données converties en float.

    Raises
    ------
    ValueError
        Si les données sont invalides.
    TypeError
        Si n_clusters n'est pas un entier.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X doit être un tableau 2D "
            "(n_samples, n_features)."
        )

    if X.shape[0] == 0:
        raise ValueError(
            "X ne peut pas être vide."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "X doit contenir au moins une caractéristique."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X doit contenir uniquement des valeurs finies."
        )

    if not isinstance(n_clusters, (int, np.integer)):
        raise TypeError(
            "n_clusters doit être un entier."
        )

    if n_clusters < 1:
        raise ValueError(
            "n_clusters doit être supérieur ou égal à 1."
        )

    if n_clusters > X.shape[0]:
        raise ValueError(
            "n_clusters ne peut pas être supérieur "
            "au nombre d'observations."
        )

    return X


# ============================================================
# INITIALISATION DES CENTROÏDES
# ============================================================

def initialize_centroids(
    X: np.ndarray,
    n_clusters: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Initialise les centroïdes en sélectionnant aléatoirement
    des observations distinctes de X.

    Parameters
    ----------
    X : np.ndarray
        Données de forme (n_samples, n_features).
    n_clusters : int
        Nombre de clusters.
    random_state : int | None
        Graine aléatoire pour rendre le résultat reproductible.

    Returns
    -------
    np.ndarray
        Centroïdes de forme (n_clusters, n_features).
    """

    X = _validate_input(X, n_clusters)

    rng = np.random.default_rng(random_state)

    indices = rng.choice(
        X.shape[0],
        size=n_clusters,
        replace=False,
    )

    return X[indices].copy()


# ============================================================
# DISTANCES
# ============================================================

def calculate_distances(
    X: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Calcule les distances euclidiennes entre chaque observation
    et chaque centroïde.

    Parameters
    ----------
    X : np.ndarray
        Données de forme (n_samples, n_features).
    centroids : np.ndarray
        Centroïdes de forme (n_clusters, n_features).

    Returns
    -------
    np.ndarray
        Matrice des distances de forme
        (n_samples, n_clusters).
    """

    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    if X.ndim != 2:
        raise ValueError("X doit être un tableau 2D.")

    if centroids.ndim != 2:
        raise ValueError(
            "centroids doit être un tableau 2D."
        )

    if X.shape[1] != centroids.shape[1]:
        raise ValueError(
            "X et centroids doivent avoir le même "
            "nombre de caractéristiques."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X doit contenir uniquement des valeurs finies."
        )

    if not np.all(np.isfinite(centroids)):
        raise ValueError(
            "centroids doit contenir uniquement "
            "des valeurs finies."
        )

    differences = (
        X[:, np.newaxis, :]
        - centroids[np.newaxis, :, :]
    )

    distances = np.sqrt(
        np.sum(differences ** 2, axis=2)
    )

    return distances


# ============================================================
# ATTRIBUTION DES CLUSTERS
# ============================================================

def assign_clusters(
    X: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Attribue chaque observation au centroïde le plus proche.

    Parameters
    ----------
    X : np.ndarray
        Données de forme (n_samples, n_features).
    centroids : np.ndarray
        Centroïdes de forme (n_clusters, n_features).

    Returns
    -------
    np.ndarray
        Indice du cluster associé à chaque observation.
    """

    distances = calculate_distances(
        X,
        centroids,
    )

    return np.argmin(distances, axis=1)


# ============================================================
# RECALCUL DES CENTROÏDES
# ============================================================

def update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    old_centroids: np.ndarray | None = None,
) -> np.ndarray:
    """
    Recalcule les centroïdes à partir des clusters.

    Si un cluster devient vide, son ancien centroïde est conservé.

    Parameters
    ----------
    X : np.ndarray
        Données de forme (n_samples, n_features).
    labels : np.ndarray
        Labels de cluster pour chaque observation.
    n_clusters : int
        Nombre de clusters.
    old_centroids : np.ndarray | None
        Anciens centroïdes, utilisés pour les clusters vides.

    Returns
    -------
    np.ndarray
        Nouveaux centroïdes.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X doit être un tableau 2D.")

    if labels.ndim != 1:
        raise ValueError(
            "labels doit être un tableau 1D."
        )

    if len(labels) != X.shape[0]:
        raise ValueError(
            "labels doit contenir un label par observation."
        )

    if not isinstance(n_clusters, (int, np.integer)):
        raise TypeError(
            "n_clusters doit être un entier."
        )

    if n_clusters < 1:
        raise ValueError(
            "n_clusters doit être supérieur ou égal à 1."
        )

    if np.any(labels < 0) or np.any(labels >= n_clusters):
        raise ValueError(
            "Les labels doivent être compris entre "
            "0 et n_clusters - 1."
        )

    if old_centroids is not None:
        old_centroids = np.asarray(
            old_centroids,
            dtype=float,
        )

        if old_centroids.shape != (
            n_clusters,
            X.shape[1],
        ):
            raise ValueError(
                "old_centroids possède une forme invalide."
            )

        centroids = old_centroids.copy()

    else:
        centroids = np.zeros(
            (n_clusters, X.shape[1]),
            dtype=float,
        )

    for cluster_index in range(n_clusters):
        cluster_points = X[
            labels == cluster_index
        ]

        if len(cluster_points) > 0:
            centroids[cluster_index] = np.mean(
                cluster_points,
                axis=0,
            )

    return centroids


# ============================================================
# INERTIE
# ============================================================

def calculate_inertia(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """
    Calcule l'inertie du modèle K-Means.

    L'inertie correspond à la somme des distances
    euclidiennes au carré entre chaque observation
    et le centroïde de son cluster.

    Parameters
    ----------
    X : np.ndarray
        Données de forme (n_samples, n_features).
    labels : np.ndarray
        Labels de cluster.
    centroids : np.ndarray
        Centroïdes.

    Returns
    -------
    float
        Inertie totale.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centroids = np.asarray(
        centroids,
        dtype=float,
    )

    if X.ndim != 2:
        raise ValueError("X doit être un tableau 2D.")

    if labels.ndim != 1:
        raise ValueError(
            "labels doit être un tableau 1D."
        )

    if len(labels) != X.shape[0]:
        raise ValueError(
            "labels doit contenir un label par observation."
        )

    if centroids.ndim != 2:
        raise ValueError(
            "centroids doit être un tableau 2D."
        )

    if centroids.shape[1] != X.shape[1]:
        raise ValueError(
            "X et centroids doivent avoir le même "
            "nombre de caractéristiques."
        )

    if np.any(labels < 0) or np.any(
        labels >= centroids.shape[0]
    ):
        raise ValueError(
            "Les labels ne correspondent pas "
            "aux centroïdes fournis."
        )

    selected_centroids = centroids[labels]

    squared_distances = np.sum(
        (X - selected_centroids) ** 2,
        axis=1,
    )

    return float(np.sum(squared_distances))


# ============================================================
# PRÉDICTION
# ============================================================

def predict_kmeans(
    X: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Prédit le cluster de nouvelles observations.

    Parameters
    ----------
    X : np.ndarray
        Nouvelles observations.
    centroids : np.ndarray
        Centroïdes appris.

    Returns
    -------
    np.ndarray
        Labels de cluster prédits.
    """

    return assign_clusters(
        X,
        centroids,
    )


# ============================================================
# FIT
# ============================================================

def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    random_state: int | None = 42,
) -> dict:
    """
    Entraîne le modèle K-Means.

    Parameters
    ----------
    X : np.ndarray
        Données d'entraînement.
    n_clusters : int, default=3
        Nombre de clusters.
    max_iterations : int, default=100
        Nombre maximal d'itérations.
    tolerance : float, default=1e-4
        Seuil de convergence.
    random_state : int | None, default=42
        Graine aléatoire.

    Returns
    -------
    dict
        Résultats de l'entraînement :
        - centroids
        - labels
        - inertia
        - n_iterations
        - converged
        - n_clusters
    """

    X = _validate_input(
        X,
        n_clusters,
    )

    if not isinstance(
        max_iterations,
        (int, np.integer),
    ):
        raise TypeError(
            "max_iterations doit être un entier."
        )

    if max_iterations < 1:
        raise ValueError(
            "max_iterations doit être supérieur ou égal à 1."
        )

    if not isinstance(
        tolerance,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "tolerance doit être numérique."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance doit être supérieur ou égal à 0."
        )

    centroids = initialize_centroids(
        X,
        n_clusters,
        random_state=random_state,
    )

    converged = False
    n_iterations = 0

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        labels = assign_clusters(
            X,
            centroids,
        )

        new_centroids = update_centroids(
            X,
            labels,
            n_clusters,
            old_centroids=centroids,
        )

        centroid_shift = np.linalg.norm(
            new_centroids - centroids
        )

        centroids = new_centroids
        n_iterations = iteration

        if centroid_shift <= tolerance:
            converged = True
            break

    # Réaffectation finale après convergence
    labels = assign_clusters(
        X,
        centroids,
    )

    inertia = calculate_inertia(
        X,
        labels,
        centroids,
    )

    return {
        "centroids": centroids,
        "labels": labels,
        "inertia": inertia,
        "n_iterations": n_iterations,
        "converged": converged,
        "n_clusters": n_clusters,
    }


# ============================================================
# PIPELINE COMPLET
# ============================================================

def kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    random_state: int | None = 42,
    X_test: np.ndarray | None = None,
) -> dict:
    """
    Pipeline complet K-Means.

    Parameters
    ----------
    X : np.ndarray
        Données d'entraînement.
    n_clusters : int, default=3
        Nombre de clusters.
    max_iterations : int, default=100
        Nombre maximal d'itérations.
    tolerance : float, default=1e-4
        Seuil de convergence.
    random_state : int | None, default=42
        Graine aléatoire.
    X_test : np.ndarray | None
        Nouvelles observations à classifier.

    Returns
    -------
    dict
        Résultats complets du K-Means :
        - centroids
        - labels
        - inertia
        - n_iterations
        - converged
        - n_clusters
        - predictions (si X_test est fourni)
        - n_samples
        - n_features
    """

    X = _validate_input(
        X,
        n_clusters,
    )

    result = fit_kmeans(
        X,
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        tolerance=tolerance,
        random_state=random_state,
    )

    if X_test is not None:
        X_test = np.asarray(
            X_test,
            dtype=float,
        )

        if X_test.ndim != 2:
            raise ValueError(
                "X_test doit être un tableau 2D."
            )

        if X_test.shape[1] != X.shape[1]:
            raise ValueError(
                "X_test doit avoir le même nombre "
                "de caractéristiques que X."
            )

        if not np.all(np.isfinite(X_test)):
            raise ValueError(
                "X_test doit contenir uniquement "
                "des valeurs finies."
            )

        result["predictions"] = predict_kmeans(
            X_test,
            result["centroids"],
        )

        result["n_test_samples"] = X_test.shape[0]

    result["n_samples"] = X.shape[0]
    result["n_features"] = X.shape[1]

    return result

