from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2, chi2_contingency

Number = int | float


# ============================================================
# VALIDATION
# ============================================================


def _validate_contingency_table(
    table: Sequence[Sequence[Number]],
) -> np.ndarray:
    """
    Valide et convertit un tableau de contingence.

    Le tableau doit :
    - contenir au moins 2 lignes ;
    - contenir au moins 2 colonnes ;
    - être rectangulaire ;
    - contenir uniquement des nombres ;
    - contenir uniquement des effectifs >= 0 ;
    - contenir au moins un effectif strictement positif.

    Retourne
    --------
    numpy.ndarray
        Tableau numérique de type float.
    """

    if not table:
        raise ValueError(
            "Le tableau de contingence ne peut pas être vide."
        )

    if not isinstance(table, Sequence):
        raise TypeError(
            "Le tableau de contingence doit être une séquence "
            "de lignes numériques."
        )

    rows = list(table)

    if len(rows) < 2:
        raise ValueError(
            "Le tableau doit contenir au moins deux lignes."
        )

    if any(
        isinstance(row, (str, bytes))
        or not isinstance(row, Sequence)
        for row in rows
    ):
        raise TypeError(
            "Chaque ligne du tableau doit être une séquence "
            "de valeurs numériques."
        )

    column_count = len(rows[0])

    if column_count < 2:
        raise ValueError(
            "Le tableau doit contenir au moins deux colonnes."
        )

    if any(len(row) != column_count for row in rows):
        raise ValueError(
            "Toutes les lignes du tableau doivent avoir "
            "le même nombre de colonnes."
        )

    validated_rows: list[list[float]] = []

    for row in rows:
        validated_row: list[float] = []

        for value in row:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.number)
            ):
                raise TypeError(
                    "Les effectifs doivent être des valeurs numériques."
                )

            numeric_value = float(value)

            if not np.isfinite(numeric_value):
                raise ValueError(
                    "Les effectifs doivent être des valeurs finies."
                )

            if numeric_value < 0:
                raise ValueError(
                    "Les effectifs ne peuvent pas être négatifs."
                )

            validated_row.append(numeric_value)

        validated_rows.append(validated_row)

    array = np.asarray(validated_rows, dtype=float)

    if np.sum(array) <= 0:
        raise ValueError(
            "Le tableau doit contenir au moins un effectif "
            "strictement positif."
        )

    return array


def _validate_significance_level(
    significance_level: Number,
) -> float:
    """
    Valide un niveau de signification alpha.
    """

    if isinstance(significance_level, bool) or not isinstance(
        significance_level,
        (int, float),
    ):
        raise TypeError(
            "Le niveau de signification doit être numérique."
        )

    alpha = float(significance_level)

    if not np.isfinite(alpha):
        raise ValueError(
            "Le niveau de signification doit être fini."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "Le niveau de signification doit être compris "
            "strictement entre 0 et 1."
        )

    return alpha


# ============================================================
# EFFECTIFS ATTENDUS
# ============================================================


def expected_frequencies(
    table: Sequence[Sequence[Number]],
) -> list[list[float]]:
    """
    Calcule les effectifs théoriques attendus d'un test du Khi-deux.

    Formule :

        E_ij = (total ligne i × total colonne j) / total général

    Paramètres
    ----------
    table :
        Tableau des effectifs observés.

    Retourne
    --------
    list[list[float]]
        Tableau des effectifs attendus.
    """

    observed = _validate_contingency_table(table)

    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    expected = (row_totals @ column_totals) / total

    return expected.tolist()


# ============================================================
# STATISTIQUE DU KHI-DEUX
# ============================================================


def chi_square_statistic(
    table: Sequence[Sequence[Number]],
) -> float:
    """
    Calcule la statistique du test du Khi-deux.

    Formule :

        χ² = Σ ((O_ij - E_ij)² / E_ij)

    où :

        O_ij = effectif observé
        E_ij = effectif attendu
    """

    observed = _validate_contingency_table(table)
    expected = np.asarray(
        expected_frequencies(observed.tolist()),
        dtype=float,
    )

    statistic = np.sum(
        (observed - expected) ** 2 / expected
    )

    return float(statistic)


# ============================================================
# DEGRÉS DE LIBERTÉ
# ============================================================


def degrees_of_freedom(
    table: Sequence[Sequence[Number]],
) -> int:
    """
    Calcule les degrés de liberté du test du Khi-deux.

    Formule :

        ddl = (nombre de lignes - 1)
              × (nombre de colonnes - 1)
    """

    observed = _validate_contingency_table(table)

    rows, columns = observed.shape

    return int((rows - 1) * (columns - 1))


# ============================================================
# P-VALUE
# ============================================================


def chi_square_p_value(
    table: Sequence[Sequence[Number]],
) -> float:
    """
    Calcule la p-value associée à la statistique du Khi-deux.
    """

    statistic = chi_square_statistic(table)
    df = degrees_of_freedom(table)

    return float(chi2.sf(statistic, df))


# ============================================================
# VALEUR CRITIQUE
# ============================================================


def chi_square_critical_value(
    table: Sequence[Sequence[Number]],
    significance_level: Number = 0.05,
) -> float:
    """
    Calcule la valeur critique du Khi-deux.

    Paramètres
    ----------
    table :
        Tableau de contingence.

    significance_level :
        Niveau de signification alpha.
        Par défaut : 0.05.
    """

    alpha = _validate_significance_level(
        significance_level
    )

    df = degrees_of_freedom(table)

    return float(chi2.ppf(1 - alpha, df))


# ============================================================
# TEST COMPLET DU KHI-DEUX
# ============================================================


def chi_square_test(
    table: Sequence[Sequence[Number]],
    significance_level: Number = 0.05,
) -> dict[str, object]:
    """
    Effectue un test complet du Khi-deux d'indépendance.

    Paramètres
    ----------
    table :
        Tableau des effectifs observés.

    significance_level :
        Niveau de signification alpha.
        Par défaut : 0.05.

    Retourne
    --------
    dict
        Dictionnaire contenant :

        - observed
        - expected
        - statistic
        - degrees_of_freedom
        - p_value
        - critical_value
        - significance_level
        - reject_null_hypothesis
    """

    observed = _validate_contingency_table(table)

    alpha = _validate_significance_level(
        significance_level
    )

    expected = np.asarray(
        expected_frequencies(observed.tolist()),
        dtype=float,
    )

    statistic = float(
        np.sum(
            (observed - expected) ** 2
            / expected
        )
    )

    df = int(
        (observed.shape[0] - 1)
        * (observed.shape[1] - 1)
    )

    p_value = float(
        chi2.sf(statistic, df)
    )

    critical_value = float(
        chi2.ppf(1 - alpha, df)
    )

    reject_null_hypothesis = p_value < alpha

    return {
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "statistic": statistic,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "critical_value": critical_value,
        "significance_level": alpha,
        "reject_null_hypothesis": reject_null_hypothesis,
    }


# ============================================================
# TEST AVEC SCIPY
# ============================================================


def chi_square_test_scipy(
    table: Sequence[Sequence[Number]],
) -> dict[str, object]:
    """
    Effectue le test du Khi-deux directement avec SciPy.

    Cette fonction est utile pour comparer les résultats
    du calcul manuel avec ceux de scipy.stats.
    """

    observed = _validate_contingency_table(table)

    statistic, p_value, degrees_of_freedom_value, expected = (
        chi2_contingency(observed)
    )

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom_value),
        "expected": expected.tolist(),
    }


# ============================================================
# DÉCISION STATISTIQUE
# ============================================================


def reject_null_hypothesis(
    p_value: Number,
    significance_level: Number = 0.05,
) -> bool:
    """
    Détermine si l'hypothèse nulle doit être rejetée.

    Règle :

        si p-value < alpha :
            rejet de H0

        sinon :
            non-rejet de H0
    """

    if isinstance(p_value, bool) or not isinstance(
        p_value,
        (int, float),
    ):
        raise TypeError(
            "La p-value doit être numérique."
        )

    numeric_p_value = float(p_value)

    if not np.isfinite(numeric_p_value):
        raise ValueError(
            "La p-value doit être finie."
        )

    if not 0 <= numeric_p_value <= 1:
        raise ValueError(
            "La p-value doit être comprise entre 0 et 1."
        )

    alpha = _validate_significance_level(
        significance_level
    )

    return numeric_p_value < alpha