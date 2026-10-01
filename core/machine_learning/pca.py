
"""
Principal Component Analysis (PCA)
=================================

Implémentation pédagogique de l'Analyse en Composantes Principales
avec NumPy uniquement.

Le PCA permet notamment de :
    - centrer et éventuellement standardiser les données ;
    - calculer la matrice de covariance ;
    - déterminer les composantes principales ;
    - projeter les données dans un espace de dimension réduite ;
    - calculer la variance expliquée ;
    - calculer la variance expliquée cumulée.

API principale :
    standardize_data()
    covariance_matrix()
    compute_principal_components()
    project_data()
    pca()
"""

from __future__ import annotations

import numpy as np


# ============================================================
# VALIDATION
# ============================================================


def _validate_input(X) -> np.ndarray:
    """
    Valide et convertit les données d'entrée en tableau NumPy 2D.

    Parameters
    ----------
    X : array-like
        Données de forme (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Tableau 2D de type float.

    Raises
    ------
    ValueError
        Si les données sont invalides.
    """

    X_array = np.asarray(X, dtype=float)

    if X_array.ndim != 2:
        raise ValueError(
            "X doit être un tableau 2D "
            "(n_samples, n_features)."
        )

    n_samples, n_features = X_array.shape

    if n_samples < 2:
        raise ValueError(
            "X doit contenir au moins 2 observations."
        )

    if n_features < 1:
        raise ValueError(
            "X doit contenir au moins une caractéristique."
        )

    if not np.all(np.isfinite(X_array)):
        raise ValueError(
            "X doit contenir uniquement des valeurs finies."
        )

    return X_array


def _validate_n_components(
    n_components,
    n_features: int,
) -> int:
    """
    Valide le nombre de composantes principales demandé.
    """

    if isinstance(n_components, bool) or not isinstance(
        n_components, (int, np.integer)
    ):
        raise ValueError(
            "n_components doit être un entier."
        )

    n_components = int(n_components)

    if n_components < 1:
        raise ValueError(
            "n_components doit être supérieur ou égal à 1."
        )

    if n_components > n_features:
        raise ValueError(
            "n_components ne peut pas dépasser "
            "le nombre de caractéristiques."
        )

    return n_components


# ============================================================
# STANDARDISATION
# ============================================================


def standardize_data(
    X,
    ddof: int = 0,
):
    """
    Centre et standardise les données.

    Pour chaque caractéristique :

        z = (x - moyenne) / écart-type

    Les caractéristiques constantes sont conservées à 0 après
    standardisation afin d'éviter une division par zéro.

    Parameters
    ----------
    X : array-like
        Données de forme (n_samples, n_features).

    ddof : int, default=0
        Degrés de liberté utilisés pour l'écart-type.

    Returns
    -------
    tuple
        (X_standardized, means, standard_deviations)
    """

    X_array = _validate_input(X)

    if isinstance(ddof, bool) or not isinstance(
        ddof, (int, np.integer)
    ):
        raise ValueError(
            "ddof doit être un entier."
        )

    if ddof < 0 or ddof >= X_array.shape[0]:
        raise ValueError(
            "ddof doit être compris entre 0 et n_samples - 1."
        )

    means = np.mean(X_array, axis=0)
    standard_deviations = np.std(
        X_array,
        axis=0,
        ddof=ddof,
    )

    X_standardized = np.zeros_like(X_array)

    non_constant = standard_deviations > 0

    X_standardized[:, non_constant] = (
        X_array[:, non_constant]
        - means[non_constant]
    ) / standard_deviations[non_constant]

    return (
        X_standardized,
        means,
        standard_deviations,
    )


# ============================================================
# MATRICE DE COVARIANCE
# ============================================================


def covariance_matrix(
    X,
    centered: bool = False,
):
    """
    Calcule la matrice de covariance.

    La covariance est calculée selon :

        C = X_c^T X_c / (n - 1)

    où X_c représente les données centrées.

    Parameters
    ----------
    X : array-like
        Données de forme (n_samples, n_features).

    centered : bool, default=False
        Si True, considère que X est déjà centrée.

    Returns
    -------
    np.ndarray
        Matrice de covariance de forme
        (n_features, n_features).
    """

    X_array = _validate_input(X)

    if not isinstance(centered, (bool, np.bool_)):
        raise ValueError(
            "centered doit être un booléen."
        )

    if centered:
        X_centered = X_array
    else:
        X_centered = (
            X_array
            - np.mean(X_array, axis=0)
        )

    return (
        X_centered.T @ X_centered
        / (X_centered.shape[0] - 1)
    )


# ============================================================
# COMPOSANTES PRINCIPALES
# ============================================================


def compute_principal_components(
    covariance,
):
    """
    Calcule les composantes principales à partir
    d'une matrice de covariance.

    Les valeurs propres sont triées par ordre décroissant.

    Parameters
    ----------
    covariance : array-like
        Matrice de covariance carrée.

    Returns
    -------
    tuple
        (
            eigenvalues,
            eigenvectors,
        )

    eigenvalues :
        Valeurs propres triées décroissantes.

    eigenvectors :
        Vecteurs propres correspondants,
        placés par colonnes.
    """

    covariance_array = np.asarray(
        covariance,
        dtype=float,
    )

    if covariance_array.ndim != 2:
        raise ValueError(
            "La matrice de covariance doit être 2D."
        )

    if (
        covariance_array.shape[0]
        != covariance_array.shape[1]
    ):
        raise ValueError(
            "La matrice de covariance doit être carrée."
        )

    if covariance_array.shape[0] < 1:
        raise ValueError(
            "La matrice de covariance ne peut pas être vide."
        )

    if not np.all(np.isfinite(covariance_array)):
        raise ValueError(
            "La matrice de covariance doit contenir "
            "uniquement des valeurs finies."
        )

    if not np.allclose(
        covariance_array,
        covariance_array.T,
        atol=1e-10,
    ):
        raise ValueError(
            "La matrice de covariance doit être symétrique."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_array
    )

    # Tri décroissant des valeurs propres.
    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Les petites erreurs numériques peuvent produire
    # de très petites valeurs propres négatives.
    eigenvalues = np.where(
        eigenvalues < 0,
        np.where(
            np.isclose(
                eigenvalues,
                0,
                atol=1e-12,
            ),
            0.0,
            eigenvalues,
        ),
        eigenvalues,
    )

    return eigenvalues, eigenvectors


# ============================================================
# VARIANCE EXPLIQUÉE
# ============================================================


def explained_variance_ratio(
    eigenvalues,
):
    """
    Calcule la proportion de variance expliquée
    par chaque composante principale.

    Formule :

        ratio_i = lambda_i / somme(lambda)

    Parameters
    ----------
    eigenvalues : array-like
        Valeurs propres.

    Returns
    -------
    np.ndarray
        Ratio de variance expliquée.
    """

    eigenvalues_array = np.asarray(
        eigenvalues,
        dtype=float,
    )

    if eigenvalues_array.ndim != 1:
        raise ValueError(
            "eigenvalues doit être un tableau 1D."
        )

    if eigenvalues_array.size == 0:
        raise ValueError(
            "eigenvalues ne peut pas être vide."
        )

    if not np.all(np.isfinite(eigenvalues_array)):
        raise ValueError(
            "eigenvalues doit contenir uniquement "
            "des valeurs finies."
        )

    if np.any(eigenvalues_array < -1e-12):
        raise ValueError(
            "Les valeurs propres ne peuvent pas être négatives."
        )

    total_variance = np.sum(eigenvalues_array)

    if np.isclose(
        total_variance,
        0.0,
        atol=1e-12,
    ):
        return np.zeros_like(
            eigenvalues_array
        )

    return (
        eigenvalues_array
        / total_variance
    )


def cumulative_explained_variance(
    eigenvalues,
):
    """
    Calcule la variance expliquée cumulée.

    Parameters
    ----------
    eigenvalues : array-like
        Valeurs propres.

    Returns
    -------
    np.ndarray
        Variance expliquée cumulée.
    """

    ratios = explained_variance_ratio(
        eigenvalues
    )

    return np.cumsum(ratios)


# ============================================================
# PROJECTION
# ============================================================


def project_data(
    X,
    components,
    mean=None,
):
    """
    Projette les données dans l'espace des composantes principales.

    Formule :

        Z = X_c W

    où :
        X_c = données centrées
        W   = matrice des composantes principales

    Parameters
    ----------
    X : array-like
        Données de forme (n_samples, n_features).

    components : array-like
        Composantes principales de forme
        (n_features, n_components).

    mean : array-like, optional
        Moyennes utilisées pour centrer les données.
        Si None, elles sont recalculées à partir de X.

    Returns
    -------
    np.ndarray
        Données projetées de forme
        (n_samples, n_components).
    """

    X_array = _validate_input(X)

    components_array = np.asarray(
        components,
        dtype=float,
    )

    if components_array.ndim != 2:
        raise ValueError(
            "components doit être un tableau 2D."
        )

    if (
        components_array.shape[0]
        != X_array.shape[1]
    ):
        raise ValueError(
            "Le nombre de lignes de components doit "
            "correspondre au nombre de caractéristiques de X."
        )

    if not np.all(np.isfinite(components_array)):
        raise ValueError(
            "components doit contenir uniquement "
            "des valeurs finies."
        )

    if mean is None:
        mean_array = np.mean(
            X_array,
            axis=0,
        )
    else:
        mean_array = np.asarray(
            mean,
            dtype=float,
        )

        if mean_array.ndim != 1:
            raise ValueError(
                "mean doit être un tableau 1D."
            )

        if mean_array.shape[0] != X_array.shape[1]:
            raise ValueError(
                "mean doit contenir une valeur par caractéristique."
            )

        if not np.all(np.isfinite(mean_array)):
            raise ValueError(
                "mean doit contenir uniquement "
                "des valeurs finies."
            )

    X_centered = (
        X_array - mean_array
    )

    return X_centered @ components_array


# ============================================================
# PIPELINE PCA
# ============================================================


def pca(
    X,
    n_components=2,
    standardize: bool = True,
):
    """
    Effectue une Analyse en Composantes Principales complète.

    Étapes :

        1. Validation des données
        2. Centrage / standardisation
        3. Matrice de covariance
        4. Valeurs et vecteurs propres
        5. Sélection des composantes
        6. Projection des données
        7. Calcul de la variance expliquée

    Parameters
    ----------
    X : array-like
        Données de forme (n_samples, n_features).

    n_components : int, default=2
        Nombre de composantes principales à conserver.

    standardize : bool, default=True
        Si True, standardise les caractéristiques avant le PCA.
        Si False, centre uniquement les données.

    Returns
    -------
    dict
        Résultats du PCA :

        - transformed_data
        - components
        - eigenvalues
        - explained_variance_ratio
        - cumulative_explained_variance
        - covariance_matrix
        - mean
        - standard_deviations
        - standardize
        - n_components
        - n_samples
        - n_features
    """

    X_array = _validate_input(X)

    n_samples, n_features = X_array.shape

    n_components = _validate_n_components(
        n_components,
        n_features,
    )

    if not isinstance(
        standardize,
        (bool, np.bool_),
    ):
        raise ValueError(
            "standardize doit être un booléen."
        )

    # --------------------------------------------------------
    # Préparation des données
    # --------------------------------------------------------

    if standardize:
        (
            X_processed,
            means,
            standard_deviations,
        ) = standardize_data(X_array)

    else:
        means = np.mean(
            X_array,
            axis=0,
        )

        standard_deviations = np.std(
            X_array,
            axis=0,
            ddof=0,
        )

        X_processed = (
            X_array - means
        )

    # --------------------------------------------------------
    # Matrice de covariance
    # --------------------------------------------------------

    covariance = covariance_matrix(
        X_processed,
        centered=True,
    )

    # --------------------------------------------------------
    # Décomposition spectrale
    # --------------------------------------------------------

    eigenvalues, eigenvectors = (
        compute_principal_components(
            covariance
        )
    )

    # --------------------------------------------------------
    # Sélection des composantes
    # --------------------------------------------------------

    selected_components = eigenvectors[
        :, :n_components
    ]

    selected_eigenvalues = eigenvalues[
        :n_components
    ]

    # --------------------------------------------------------
    # Projection
    # --------------------------------------------------------

    transformed_data = (
        X_processed
        @ selected_components
    )

    # --------------------------------------------------------
    # Variance expliquée
    # --------------------------------------------------------

    variance_ratio = (
        explained_variance_ratio(
            eigenvalues
        )
    )

    cumulative_variance = (
        cumulative_explained_variance(
            eigenvalues
        )
    )

    return {
        "transformed_data": transformed_data,
        "components": selected_components,
        "eigenvalues": selected_eigenvalues,
        "all_eigenvalues": eigenvalues,
        "explained_variance_ratio": variance_ratio[
            :n_components
        ],
        "all_explained_variance_ratio": variance_ratio,
        "cumulative_explained_variance": cumulative_variance[
            :n_components
        ],
        "all_cumulative_explained_variance": cumulative_variance,
        "covariance_matrix": covariance,
        "mean": means,
        "standard_deviations": standard_deviations,
        "standardize": standardize,
        "n_components": n_components,
        "n_samples": n_samples,
        "n_features": n_features,
    }

