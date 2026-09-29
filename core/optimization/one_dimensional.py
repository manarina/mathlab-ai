
"""
Algorithmes d'optimisation à une variable.

Ce module contient les premières méthodes numériques permettant
de rechercher un minimum ou un maximum d'une fonction réelle.

Méthodes disponibles :
    - Recherche du minimum par balayage (Grid Search)
    - Recherche du minimum par section dorée (Golden Section)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


# ============================================================
# VALIDATION
# ============================================================


def _validate_function(function: Callable[[float], float]) -> None:
    """Vérifie que l'objectif fourni est appelable."""
    if not callable(function):
        raise TypeError("La fonction objectif doit être appelable.")


def _validate_interval(a: float, b: float) -> tuple[float, float]:
    """Valide et normalise un intervalle [a, b]."""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Les bornes de l'intervalle doivent être numériques."
        ) from exc

    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError(
            "Les bornes de l'intervalle doivent être finies."
        )

    if a >= b:
        raise ValueError(
            "La borne inférieure doit être strictement inférieure "
            "à la borne supérieure."
        )

    return a, b


def _validate_points(points: int) -> int:
    """Valide le nombre de points utilisés pour le balayage."""
    if isinstance(points, bool):
        raise TypeError(
            "Le nombre de points doit être un entier."
        )

    if not isinstance(points, (int, np.integer)):
        raise TypeError(
            "Le nombre de points doit être un entier."
        )

    if points < 2:
        raise ValueError(
            "Le nombre de points doit être supérieur ou égal à 2."
        )

    return int(points)


def _validate_tolerance(tolerance: float) -> float:
    """Valide la tolérance numérique."""
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "La tolérance doit être numérique."
        ) from exc

    if not np.isfinite(tolerance):
        raise ValueError(
            "La tolérance doit être finie."
        )

    if tolerance <= 0:
        raise ValueError(
            "La tolérance doit être strictement positive."
        )

    return tolerance


def _validate_max_iterations(max_iterations: int) -> int:
    """Valide le nombre maximal d'itérations."""
    if isinstance(max_iterations, bool):
        raise TypeError(
            "Le nombre maximal d'itérations doit être un entier."
        )

    if not isinstance(max_iterations, (int, np.integer)):
        raise TypeError(
            "Le nombre maximal d'itérations doit être un entier."
        )

    if max_iterations <= 0:
        raise ValueError(
            "Le nombre maximal d'itérations doit être strictement positif."
        )

    return int(max_iterations)


def _evaluate_scalar(
    function: Callable[[float], float],
    x: float,
) -> float:
    """
    Évalue une fonction en un point scalaire.

    La valeur retournée doit être numérique et finie.
    """
    try:
        value = float(function(float(x)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "La fonction doit retourner des valeurs numériques."
        ) from exc

    if not np.isfinite(value):
        raise ValueError(
            "La fonction retourne des valeurs non finies."
        )

    return value


def _evaluate_function(
    function: Callable[[float], float],
    x_values: np.ndarray,
) -> np.ndarray:
    """
    Évalue la fonction sur les points fournis.

    La fonction peut accepter directement un tableau NumPy
    ou uniquement une valeur scalaire.
    """
    try:
        raw_values = function(x_values)
    except (TypeError, ValueError, OverflowError):
        raw_values = None
    else:
        try:
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError, OverflowError):
            values = None
        else:
            if values.shape == x_values.shape:
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        "La fonction retourne des valeurs non finies."
                    )

                return values

    values_list = []

    for x in x_values:
        values_list.append(_evaluate_scalar(function, float(x)))

    return np.asarray(values_list, dtype=float)


# ============================================================
# GRID SEARCH — MINIMUM
# ============================================================


def grid_search_minimum(
    function: Callable[[float], float],
    a: float,
    b: float,
    points: int = 1000,
) -> tuple[float, float]:
    """
    Recherche approximativement le minimum d'une fonction
    sur un intervalle donné par balayage.

    La méthode évalue la fonction sur un ensemble de points
    régulièrement espacés entre a et b, puis sélectionne le
    point donnant la plus petite valeur.

    Parameters
    ----------
    function:
        Fonction objectif f(x) à minimiser.

    a:
        Borne inférieure de l'intervalle.

    b:
        Borne supérieure de l'intervalle.

    points:
        Nombre de points utilisés pour le balayage.
        La valeur par défaut est 1000.

    Returns
    -------
    tuple[float, float]
        Un tuple contenant :

        - x_min : position approximative du minimum
        - f_min : valeur approximative de la fonction au minimum

    Examples
    --------
    >>> f = lambda x: (x - 2) ** 2 + 1
    >>> x_min, f_min = grid_search_minimum(f, 0, 4, points=1001)
    >>> round(x_min, 3)
    2.0
    >>> round(f_min, 3)
    1.0
    """
    _validate_function(function)

    a, b = _validate_interval(a, b)

    points = _validate_points(points)

    x_values = np.linspace(a, b, points)

    y_values = _evaluate_function(function, x_values)

    minimum_index = int(np.argmin(y_values))

    x_min = float(x_values[minimum_index])
    f_min = float(y_values[minimum_index])

    return x_min, f_min


# ============================================================
# GOLDEN SECTION — MINIMUM
# ============================================================


def golden_section_minimum(
    function: Callable[[float], float],
    a: float,
    b: float,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[float, float]:
    """
    Recherche approximativement le minimum d'une fonction
    unimodale sur un intervalle par la méthode de la section dorée.

    La méthode réduit progressivement l'intervalle de recherche
    en conservant les points nécessaires à la comparaison des
    valeurs de la fonction.

    Parameters
    ----------
    function:
        Fonction objectif f(x) à minimiser.

    a:
        Borne inférieure de l'intervalle.

    b:
        Borne supérieure de l'intervalle.

    tolerance:
        Tolérance sur la largeur de l'intervalle.
        La valeur par défaut est 1e-6.

    max_iterations:
        Nombre maximal d'itérations autorisées.
        La valeur par défaut est 1000.

    Returns
    -------
    tuple[float, float]
        Un tuple contenant :

        - x_min : position approximative du minimum
        - f_min : valeur de la fonction au minimum.

    Notes
    -----
    La méthode suppose que la fonction possède un minimum
    unimodal sur l'intervalle considéré.

    Examples
    --------
    >>> f = lambda x: (x - 2) ** 2 + 1
    >>> x_min, f_min = golden_section_minimum(f, 0, 4)
    >>> round(x_min, 5)
    2.0
    >>> round(f_min, 5)
    1.0
    """
    _validate_function(function)

    a, b = _validate_interval(a, b)

    tolerance = _validate_tolerance(tolerance)

    max_iterations = _validate_max_iterations(max_iterations)

    # --------------------------------------------------------
    # Constante de la section dorée
    # --------------------------------------------------------

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    # --------------------------------------------------------
    # Deux points internes
    # --------------------------------------------------------

    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    f_c = _evaluate_scalar(function, c)
    f_d = _evaluate_scalar(function, d)

    # --------------------------------------------------------
    # Réduction progressive de l'intervalle
    # --------------------------------------------------------

    for _ in range(max_iterations):
        if abs(b - a) <= tolerance:
            break

        if f_c < f_d:
            # Le minimum se trouve dans [a, d]
            b = d
            d = c
            f_d = f_c

            c = b - (b - a) / golden_ratio
            f_c = _evaluate_scalar(function, c)

        else:
            # Le minimum se trouve dans [c, b]
            a = c
            c = d
            f_c = f_d

            d = a + (b - a) / golden_ratio
            f_d = _evaluate_scalar(function, d)

    # --------------------------------------------------------
    # Estimation finale
    # --------------------------------------------------------

    x_min = (a + b) / 2.0
    f_min = _evaluate_scalar(function, x_min)

    return float(x_min), float(f_min)


# ============================================================
# NEWTON — MINIMUM
# ============================================================


def newton_minimum(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    second_derivative: Callable[[float], float],
    x0: float,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[float, float]:
    """
    Recherche approximativement le minimum d'une fonction
    par la méthode de Newton.

    La méthode repose sur l'itération :

        x_(n+1) = x_n - f'(x_n) / f''(x_n)

    Un point stationnaire obtenu par Newton correspond à un
    minimum local lorsque la dérivée seconde est positive.

    Parameters
    ----------
    function:
        Fonction objectif f(x).

    derivative:
        Première dérivée f'(x).

    second_derivative:
        Deuxième dérivée f''(x).

    x0:
        Point initial de l'algorithme.

    tolerance:
        Tolérance utilisée pour le critère de convergence.

    max_iterations:
        Nombre maximal d'itérations autorisées.

    Returns
    -------
    tuple[float, float]
        Un tuple contenant :

        - x_min : approximation du point minimisant la fonction
        - f_min : valeur de la fonction en ce point

    Raises
    ------
    TypeError
        Si l'objectif ou les dérivées ne sont pas appelables,
        ou si une valeur attendue n'est pas numérique.

    ValueError
        Si le point initial, la tolérance ou une valeur calculée
        n'est pas finie, ou si la dérivée seconde devient nulle.

    Notes
    -----
    La méthode de Newton est une méthode locale. Elle dépend
    fortement du choix du point initial et peut ne pas converger
    vers le minimum recherché.

    Pour obtenir un minimum local, la dérivée seconde au point
    obtenu doit être positive.

    Examples
    --------
    >>> f = lambda x: (x - 2) ** 2 + 1
    >>> df = lambda x: 2 * (x - 2)
    >>> d2f = lambda x: 2
    >>> x_min, f_min = newton_minimum(f, df, d2f, 0)
    >>> round(x_min, 5)
    2.0
    >>> round(f_min, 5)
    1.0
    """
    # --------------------------------------------------------
    # Validation des fonctions
    # --------------------------------------------------------

    _validate_function(function)
    _validate_function(derivative)
    _validate_function(second_derivative)

    # --------------------------------------------------------
    # Validation du point initial
    # --------------------------------------------------------

    try:
        x = float(x0)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Le point initial doit être numérique."
        ) from exc

    if not np.isfinite(x):
        raise ValueError(
            "Le point initial doit être fini."
        )

    # --------------------------------------------------------
    # Validation des paramètres numériques
    # --------------------------------------------------------

    tolerance = _validate_tolerance(tolerance)

    max_iterations = _validate_max_iterations(max_iterations)

    # --------------------------------------------------------
    # Itérations de Newton
    # --------------------------------------------------------

    for _ in range(max_iterations):
        first_derivative = _evaluate_scalar(
            derivative,
            x,
        )

        second_derivative_value = _evaluate_scalar(
            second_derivative,
            x,
        )

        # ----------------------------------------------------
        # Vérification de la dérivée seconde
        # ----------------------------------------------------

        if np.isclose(second_derivative_value, 0.0):
            raise ValueError(
                "La dérivée seconde est nulle ou trop proche de zéro."
            )

        # ----------------------------------------------------
        # Itération de Newton
        # ----------------------------------------------------

        x_new = (
            x
            - first_derivative / second_derivative_value
        )

        if not np.isfinite(x_new):
            raise ValueError(
                "La méthode de Newton produit une valeur non finie."
            )

        # ----------------------------------------------------
        # Critère de convergence
        # ----------------------------------------------------

        if abs(x_new - x) <= tolerance:
            x = x_new
            break

        x = x_new

    # --------------------------------------------------------
    # Vérification du point final
    # --------------------------------------------------------

    f_min = _evaluate_scalar(
        function,
        x,
    )

    return float(x), float(f_min)


