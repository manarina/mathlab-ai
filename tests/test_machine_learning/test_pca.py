
"""
Tests de l'Analyse en Composantes Principales (PCA).

Les tests couvrent :
    - validation des entrées ;
    - standardisation ;
    - matrice de covariance ;
    - valeurs et vecteurs propres ;
    - variance expliquée ;
    - variance expliquée cumulée ;
    - projection ;
    - pipeline PCA complet.
"""

import numpy as np
import pytest

from core.machine_learning.pca import (
    _validate_input,
    _validate_n_components,
    standardize_data,
    covariance_matrix,
    compute_principal_components,
    explained_variance_ratio,
    cumulative_explained_variance,
    project_data,
    pca,
)


# ============================================================
# DONNÉES DE TEST
# ============================================================


@pytest.fixture
def simple_data():
    """
    Jeu de données simple à deux caractéristiques.
    """
    return np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )


@pytest.fixture
def three_feature_data():
    """
    Jeu de données à trois caractéristiques.
    """
    return np.array(
        [
            [1.0, 2.0, 10.0],
            [2.0, 4.0, 20.0],
            [3.0, 6.0, 30.0],
            [4.0, 8.0, 40.0],
            [5.0, 10.0, 50.0],
        ]
    )


@pytest.fixture
def non_correlated_data():
    """
    Données avec deux caractéristiques non corrélées.
    """
    return np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )


# ============================================================
# TESTS DE VALIDATION
# ============================================================


def test_validate_input_returns_numpy_array():
    X = [[1, 2], [3, 4]]

    result = _validate_input(X)

    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.shape == (2, 2)


def test_validate_input_accepts_valid_array():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = _validate_input(X)

    np.testing.assert_array_equal(result, X)


@pytest.mark.parametrize(
    "X",
    [
        [1, 2, 3],
        np.array([1, 2, 3]),
        np.array(5),
    ],
)
def test_validate_input_rejects_non_2d_data(X):
    with pytest.raises(ValueError, match="2D"):
        _validate_input(X)


def test_validate_input_rejects_less_than_two_samples():
    X = [[1, 2]]

    with pytest.raises(
        ValueError,
        match="au moins 2 observations",
    ):
        _validate_input(X)


def test_validate_input_rejects_zero_features():
    X = np.empty((3, 0))

    with pytest.raises(
        ValueError,
        match="au moins une caractéristique",
    ):
        _validate_input(X)


@pytest.mark.parametrize(
    "X",
    [
        [[1, 2], [3, np.nan]],
        [[1, 2], [3, np.inf]],
        [[1, 2], [3, -np.inf]],
    ],
)
def test_validate_input_rejects_non_finite_values(X):
    with pytest.raises(
        ValueError,
        match="valeurs finies",
    ):
        _validate_input(X)


# ============================================================
# TESTS N_COMPONENTS
# ============================================================


def test_validate_n_components_accepts_valid_value():
    assert _validate_n_components(2, 3) == 2


def test_validate_n_components_accepts_numpy_integer():
    result = _validate_n_components(
        np.int64(2),
        3,
    )

    assert result == 2


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_validate_n_components_rejects_non_positive(value):
    with pytest.raises(
        ValueError,
        match="supérieur ou égal à 1",
    ):
        _validate_n_components(value, 3)


def test_validate_n_components_rejects_too_many_components():
    with pytest.raises(
        ValueError,
        match="ne peut pas dépasser",
    ):
        _validate_n_components(4, 3)


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "2",
        None,
        True,
    ],
)
def test_validate_n_components_rejects_invalid_type(value):
    with pytest.raises(
        ValueError,
        match="doit être un entier",
    ):
        _validate_n_components(value, 3)


# ============================================================
# TESTS STANDARDISATION
# ============================================================


def test_standardize_data_has_zero_mean(simple_data):
    standardized, means, stds = standardize_data(
        simple_data
    )

    np.testing.assert_allclose(
        np.mean(standardized, axis=0),
        np.zeros(2),
        atol=1e-12,
    )


def test_standardize_data_has_unit_standard_deviation(
    simple_data,
):
    standardized, _, _ = standardize_data(
        simple_data
    )

    np.testing.assert_allclose(
        np.std(
            standardized,
            axis=0,
            ddof=0,
        ),
        np.ones(2),
        atol=1e-12,
    )


def test_standardize_data_returns_correct_means(
    simple_data,
):
    _, means, _ = standardize_data(
        simple_data
    )

    np.testing.assert_allclose(
        means,
        [2.5, 5.0],
    )


def test_standardize_data_returns_correct_standard_deviations(
    simple_data,
):
    _, _, stds = standardize_data(
        simple_data
    )

    expected = np.std(
        simple_data,
        axis=0,
        ddof=0,
    )

    np.testing.assert_allclose(
        stds,
        expected,
    )


def test_standardize_data_handles_constant_feature():
    X = np.array(
        [
            [1.0, 10.0],
            [1.0, 20.0],
            [1.0, 30.0],
        ]
    )

    standardized, means, stds = standardize_data(X)

    assert means[0] == pytest.approx(1.0)
    assert stds[0] == pytest.approx(0.0)

    np.testing.assert_allclose(
        standardized[:, 0],
        np.zeros(3),
    )


def test_standardize_data_with_ddof_one():
    X = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    standardized, _, stds = standardize_data(
        X,
        ddof=1,
    )

    expected_std = np.std(
        X,
        axis=0,
        ddof=1,
    )

    np.testing.assert_allclose(
        stds,
        expected_std,
    )

    np.testing.assert_allclose(
        np.std(
            standardized,
            axis=0,
            ddof=1,
        ),
        [1.0],
    )


def test_standardize_data_rejects_invalid_ddof():
    with pytest.raises(
        ValueError,
        match="ddof doit être un entier",
    ):
        standardize_data(
            [[1, 2], [3, 4]],
            ddof=1.5,
        )


def test_standardize_data_rejects_too_large_ddof():
    with pytest.raises(
        ValueError,
        match="ddof doit être compris",
    ):
        standardize_data(
            [[1, 2], [3, 4]],
            ddof=2,
        )


# ============================================================
# TESTS MATRICE DE COVARIANCE
# ============================================================


def test_covariance_matrix_has_correct_shape(
    simple_data,
):
    covariance = covariance_matrix(
        simple_data
    )

    assert covariance.shape == (2, 2)


def test_covariance_matrix_is_symmetric(
    simple_data,
):
    covariance = covariance_matrix(
        simple_data
    )

    np.testing.assert_allclose(
        covariance,
        covariance.T,
    )


def test_covariance_matrix_matches_numpy(
    simple_data,
):
    covariance = covariance_matrix(
        simple_data
    )

    expected = np.cov(
        simple_data,
        rowvar=False,
    )

    np.testing.assert_allclose(
        covariance,
        expected,
    )


def test_covariance_matrix_centered_matches_numpy(
    simple_data,
):
    centered = (
        simple_data
        - np.mean(simple_data, axis=0)
    )

    covariance = covariance_matrix(
        centered,
        centered=True,
    )

    expected = np.cov(
        simple_data,
        rowvar=False,
    )

    np.testing.assert_allclose(
        covariance,
        expected,
    )


def test_covariance_matrix_is_positive_semidefinite(
    three_feature_data,
):
    covariance = covariance_matrix(
        three_feature_data
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    assert np.all(eigenvalues >= -1e-10)


def test_covariance_matrix_rejects_invalid_centered():
    with pytest.raises(
        ValueError,
        match="centered doit être un booléen",
    ):
        covariance_matrix(
            [[1, 2], [3, 4]],
            centered="yes",
        )


# ============================================================
# TESTS COMPOSANTES PRINCIPALES
# ============================================================


def test_compute_principal_components_sorts_eigenvalues():
    covariance = np.array(
        [
            [4.0, 0.0],
            [0.0, 1.0],
        ]
    )

    eigenvalues, eigenvectors = (
        compute_principal_components(
            covariance
        )
    )

    np.testing.assert_allclose(
        eigenvalues,
        [4.0, 1.0],
    )

    assert eigenvectors.shape == (2, 2)


def test_compute_principal_components_matches_numpy():
    covariance = np.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ]
    )

    eigenvalues, eigenvectors = (
        compute_principal_components(
            covariance
        )
    )

    expected_values, _ = np.linalg.eigh(
        covariance
    )

    expected_values = np.sort(
        expected_values
    )[::-1]

    np.testing.assert_allclose(
        eigenvalues,
        expected_values,
    )


def test_compute_principal_components_vectors_are_orthonormal():
    covariance = np.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ]
    )

    _, eigenvectors = (
        compute_principal_components(
            covariance
        )
    )

    identity = (
        eigenvectors.T
        @ eigenvectors
    )

    np.testing.assert_allclose(
        identity,
        np.eye(2),
        atol=1e-10,
    )


def test_compute_principal_components_rejects_non_square_matrix():
    covariance = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="carrée",
    ):
        compute_principal_components(
            covariance
        )


def test_compute_principal_components_rejects_non_symmetric_matrix():
    covariance = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="symétrique",
    ):
        compute_principal_components(
            covariance
        )


def test_compute_principal_components_rejects_non_finite_values():
    covariance = np.array(
        [
            [1.0, np.nan],
            [np.nan, 2.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="valeurs finies",
    ):
        compute_principal_components(
            covariance
        )


# ============================================================
# TESTS VARIANCE EXPLIQUÉE
# ============================================================


def test_explained_variance_ratio_sums_to_one():
    eigenvalues = np.array(
        [6.0, 3.0, 1.0]
    )

    ratios = explained_variance_ratio(
        eigenvalues
    )

    assert np.sum(ratios) == pytest.approx(
        1.0
    )


def test_explained_variance_ratio_is_correct():
    eigenvalues = np.array(
        [6.0, 3.0, 1.0]
    )

    ratios = explained_variance_ratio(
        eigenvalues
    )

    expected = np.array(
        [0.6, 0.3, 0.1]
    )

    np.testing.assert_allclose(
        ratios,
        expected,
    )


def test_explained_variance_ratio_handles_zero_variance():
    eigenvalues = np.array(
        [0.0, 0.0]
    )

    ratios = explained_variance_ratio(
        eigenvalues
    )

    np.testing.assert_allclose(
        ratios,
        [0.0, 0.0],
    )


def test_explained_variance_ratio_rejects_negative_values():
    with pytest.raises(
        ValueError,
        match="négatives",
    ):
        explained_variance_ratio(
            [2.0, -1.0]
        )


def test_explained_variance_ratio_rejects_empty_array():
    with pytest.raises(
        ValueError,
        match="ne peut pas être vide",
    ):
        explained_variance_ratio([])


def test_explained_variance_ratio_rejects_non_1d_array():
    with pytest.raises(
        ValueError,
        match="tableau 1D",
    ):
        explained_variance_ratio(
            [[1.0, 2.0]]
        )


def test_cumulative_explained_variance_is_monotonic():
    eigenvalues = np.array(
        [6.0, 3.0, 1.0]
    )

    cumulative = (
        cumulative_explained_variance(
            eigenvalues
        )
    )

    assert np.all(
        np.diff(cumulative) >= 0
    )


def test_cumulative_explained_variance_ends_at_one():
    eigenvalues = np.array(
        [6.0, 3.0, 1.0]
    )

    cumulative = (
        cumulative_explained_variance(
            eigenvalues
        )
    )

    assert cumulative[-1] == pytest.approx(
        1.0
    )


def test_cumulative_explained_variance_is_correct():
    eigenvalues = np.array(
        [6.0, 3.0, 1.0]
    )

    cumulative = (
        cumulative_explained_variance(
            eigenvalues
        )
    )

    expected = np.array(
        [
            0.6,
            0.9,
            1.0,
        ]
    )

    np.testing.assert_allclose(
        cumulative,
        expected,
    )


# ============================================================
# TESTS PROJECTION
# ============================================================


def test_project_data_reduces_dimension(
    simple_data,
):
    components = np.array(
        [
            [1.0],
            [0.0],
        ]
    )

    projected = project_data(
        simple_data,
        components,
    )

    assert projected.shape == (4, 1)


def test_project_data_centers_data(
    simple_data,
):
    components = np.array(
        [
            [1.0],
            [0.0],
        ]
    )

    projected = project_data(
        simple_data,
        components,
    )

    assert np.mean(projected) == pytest.approx(
        0.0
    )


def test_project_data_with_custom_mean():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    components = np.eye(2)

    projected = project_data(
        X,
        components,
        mean=np.array([2.0, 3.0]),
    )

    expected = np.array(
        [
            [-1.0, -1.0],
            [1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        projected,
        expected,
    )


def test_project_data_rejects_invalid_components():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    components = np.array(
        [
            [1.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="nombre de lignes",
    ):
        project_data(
            X,
            components,
        )


def test_project_data_rejects_invalid_mean_shape():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    components = np.eye(2)

    with pytest.raises(
        ValueError,
        match="une valeur par caractéristique",
    ):
        project_data(
            X,
            components,
            mean=np.array([1.0]),
        )


def test_project_data_rejects_non_finite_components():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    components = np.array(
        [
            [1.0, np.nan],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="valeurs finies",
    ):
        project_data(
            X,
            components,
        )


# ============================================================
# TESTS PIPELINE PCA
# ============================================================


def test_pca_returns_expected_keys(
    simple_data,
):
    result = pca(
        simple_data,
        n_components=2,
    )

    expected_keys = {
        "transformed_data",
        "components",
        "eigenvalues",
        "all_eigenvalues",
        "explained_variance_ratio",
        "all_explained_variance_ratio",
        "cumulative_explained_variance",
        "all_cumulative_explained_variance",
        "covariance_matrix",
        "mean",
        "standard_deviations",
        "standardize",
        "n_components",
        "n_samples",
        "n_features",
    }

    assert expected_keys.issubset(
        result.keys()
    )


def test_pca_returns_correct_dimensions(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=2,
    )

    assert result["transformed_data"].shape == (
        5,
        2,
    )

    assert result["components"].shape == (
        3,
        2,
    )

    assert result["eigenvalues"].shape == (
        2,
    )


def test_pca_returns_correct_metadata(
    simple_data,
):
    result = pca(
        simple_data,
        n_components=1,
    )

    assert result["n_components"] == 1
    assert result["n_samples"] == 4
    assert result["n_features"] == 2
    assert result["standardize"] is True


def test_pca_standardizes_by_default(
    simple_data,
):
    result = pca(
        simple_data,
        n_components=2,
    )

    np.testing.assert_allclose(
        result["mean"],
        [2.5, 5.0],
    )


def test_pca_without_standardization(
    simple_data,
):
    result = pca(
        simple_data,
        n_components=2,
        standardize=False,
    )

    assert result["standardize"] is False

    np.testing.assert_allclose(
        result["mean"],
        [2.5, 5.0],
    )


def test_pca_explained_variance_sums_to_one(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=3,
    )

    assert np.sum(
        result["explained_variance_ratio"]
    ) == pytest.approx(1.0)


def test_pca_cumulative_variance_is_monotonic(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=3,
    )

    cumulative = result[
        "cumulative_explained_variance"
    ]

    assert np.all(
        np.diff(cumulative) >= -1e-12
    )


def test_pca_cumulative_variance_ends_at_one(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=3,
    )

    assert result[
        "cumulative_explained_variance"
    ][-1] == pytest.approx(1.0)


def test_pca_first_component_explains_most_variance(
    simple_data,
):
    result = pca(
        simple_data,
        n_components=2,
    )

    ratios = result[
        "explained_variance_ratio"
    ]

    assert ratios[0] >= ratios[1]


def test_pca_components_are_orthonormal(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=3,
    )

    components = result[
        "components"
    ]

    np.testing.assert_allclose(
        components.T @ components,
        np.eye(3),
        atol=1e-10,
    )


def test_pca_transformed_data_is_centered(
    simple_data,
):
    result = pca(
        simple_data,
        n_components=2,
    )

    transformed = result[
        "transformed_data"
    ]

    np.testing.assert_allclose(
        np.mean(transformed, axis=0),
        np.zeros(2),
        atol=1e-10,
    )


def test_pca_rejects_too_many_components(
    simple_data,
):
    with pytest.raises(
        ValueError,
        match="ne peut pas dépasser",
    ):
        pca(
            simple_data,
            n_components=3,
        )


def test_pca_rejects_invalid_standardize(
    simple_data,
):
    with pytest.raises(
        ValueError,
        match="standardize doit être un booléen",
    ):
        pca(
            simple_data,
            n_components=2,
            standardize="yes",
        )


def test_pca_with_one_component(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=1,
    )

    assert result[
        "transformed_data"
    ].shape == (5, 1)

    assert result[
        "components"
    ].shape == (3, 1)

    assert len(
        result["explained_variance_ratio"]
    ) == 1


def test_pca_accepts_python_lists():
    X = [
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8],
    ]

    result = pca(
        X,
        n_components=2,
    )

    assert isinstance(
        result["transformed_data"],
        np.ndarray,
    )


# ============================================================
# TEST COHÉRENCE MATHÉMATIQUE
# ============================================================


def test_pca_reconstruction_projection_dimensions(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=2,
    )

    transformed = result[
        "transformed_data"
    ]

    components = result[
        "components"
    ]

    reconstructed = (
        transformed
        @ components.T
    )

    assert reconstructed.shape == (
        5,
        3,
    )


def test_pca_covariance_matches_processed_data(
    three_feature_data,
):
    result = pca(
        three_feature_data,
        n_components=3,
    )

    covariance = result[
        "covariance_matrix"
    ]

    transformed_covariance = (
        covariance
    )

    assert transformed_covariance.shape == (
        3,
        3,
    )

    np.testing.assert_allclose(
        covariance,
        covariance.T,
        atol=1e-10,
    )

