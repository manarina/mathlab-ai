
"""
Tests des algorithmes d'optimisation à une variable.
"""

import numpy as np
import pytest

from core.optimization.one_dimensional import (
    golden_section_minimum,
    grid_search_minimum,
    newton_minimum,
)


# ============================================================
# GRID SEARCH — TESTS DE BASE
# ============================================================


def test_grid_search_minimum_quadratic():
    """Trouve le minimum d'une fonction quadratique simple."""
    function = lambda x: (x - 2) ** 2 + 1

    x_min, f_min = grid_search_minimum(
        function,
        0,
        4,
        points=1001,
    )

    assert x_min == pytest.approx(2.0, abs=0.004)
    assert f_min == pytest.approx(1.0, abs=1e-12)


def test_grid_search_minimum_another_quadratic():
    """Teste une fonction quadratique avec un minimum différent."""
    function = lambda x: (x + 3) ** 2 - 5

    x_min, f_min = grid_search_minimum(
        function,
        -6,
        0,
        points=1201,
    )

    assert x_min == pytest.approx(-3.0, abs=0.005)
    assert f_min == pytest.approx(-5.0, abs=1e-12)


def test_grid_search_minimum_linear():
    """Teste une fonction dont le minimum est sur une borne."""
    function = lambda x: 2 * x + 1

    x_min, f_min = grid_search_minimum(
        function,
        0,
        5,
        points=101,
    )

    assert x_min == pytest.approx(0.0)
    assert f_min == pytest.approx(1.0)


def test_grid_search_minimum_decreasing_function():
    """Teste une fonction décroissante dont le minimum est à droite."""
    function = lambda x: -x

    x_min, f_min = grid_search_minimum(
        function,
        0,
        10,
        points=101,
    )

    assert x_min == pytest.approx(10.0)
    assert f_min == pytest.approx(-10.0)


# ============================================================
# GRID SEARCH — PRÉCISION
# ============================================================


def test_grid_search_minimum_precision_depends_on_grid():
    """
    Vérifie que l'augmentation du nombre de points
    améliore la localisation du minimum.
    """
    function = lambda x: (x - 1.234) ** 2

    x_min_coarse, _ = grid_search_minimum(
        function,
        0,
        2,
        points=101,
    )

    x_min_fine, _ = grid_search_minimum(
        function,
        0,
        2,
        points=10001,
    )

    coarse_error = abs(x_min_coarse - 1.234)
    fine_error = abs(x_min_fine - 1.234)

    assert fine_error < coarse_error
    assert fine_error <= 0.0002


# ============================================================
# GRID SEARCH — FONCTION VECTORISÉE
# ============================================================


def test_grid_search_minimum_vectorized_function():
    """Vérifie la compatibilité avec une fonction NumPy vectorisée."""
    function = lambda x: np.sin(x) ** 2

    x_min, f_min = grid_search_minimum(
        function,
        2.5,
        3.8,
        points=1001,
    )

    assert x_min == pytest.approx(np.pi, abs=0.002)
    assert f_min == pytest.approx(0.0, abs=1e-5)


# ============================================================
# GRID SEARCH — TYPES DE RETOUR
# ============================================================


def test_grid_search_minimum_returns_floats():
    """Vérifie que le résultat contient deux nombres flottants."""
    function = lambda x: (x - 1) ** 2

    x_min, f_min = grid_search_minimum(
        function,
        0,
        2,
        points=101,
    )

    assert isinstance(x_min, float)
    assert isinstance(f_min, float)


# ============================================================
# GRID SEARCH — VALIDATION DE LA FONCTION
# ============================================================


def test_grid_search_minimum_rejects_non_callable_function():
    """Une fonction non appelable doit provoquer une erreur."""
    with pytest.raises(TypeError, match="appelable"):
        grid_search_minimum(
            42,
            0,
            1,
        )


def test_grid_search_minimum_rejects_function_returning_non_numeric():
    """Une fonction retournant du texte doit être rejetée."""
    function = lambda x: "invalid"

    with pytest.raises(TypeError, match="valeurs numériques"):
        grid_search_minimum(
            function,
            0,
            1,
            points=10,
        )


def test_grid_search_minimum_rejects_non_finite_values():
    """Une fonction retournant NaN ou inf doit être rejetée."""
    function = lambda x: np.nan

    with pytest.raises(ValueError, match="non finies"):
        grid_search_minimum(
            function,
            0,
            1,
            points=10,
        )


# ============================================================
# GRID SEARCH — VALIDATION DES BORNES
# ============================================================


def test_grid_search_minimum_rejects_equal_bounds():
    """Les deux bornes ne peuvent pas être identiques."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="strictement inférieure"):
        grid_search_minimum(
            function,
            1,
            1,
        )


def test_grid_search_minimum_rejects_reversed_bounds():
    """La borne inférieure doit être plus petite que la borne supérieure."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="strictement inférieure"):
        grid_search_minimum(
            function,
            5,
            1,
        )


def test_grid_search_minimum_rejects_non_finite_bounds():
    """Les bornes infinies ou NaN sont refusées."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="finies"):
        grid_search_minimum(
            function,
            -np.inf,
            1,
        )

    with pytest.raises(ValueError, match="finies"):
        grid_search_minimum(
            function,
            0,
            np.inf,
        )


# ============================================================
# GRID SEARCH — VALIDATION DU NOMBRE DE POINTS
# ============================================================


def test_grid_search_minimum_rejects_too_few_points():
    """Au moins deux points sont nécessaires."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="supérieur ou égal à 2"):
        grid_search_minimum(
            function,
            0,
            1,
            points=1,
        )


def test_grid_search_minimum_rejects_zero_points():
    """Un nombre nul de points doit être refusé."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="supérieur ou égal à 2"):
        grid_search_minimum(
            function,
            0,
            1,
            points=0,
        )


def test_grid_search_minimum_rejects_negative_points():
    """Un nombre négatif de points doit être refusé."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="supérieur ou égal à 2"):
        grid_search_minimum(
            function,
            0,
            1,
            points=-10,
        )


def test_grid_search_minimum_rejects_non_integer_points():
    """Le nombre de points doit être un entier."""
    function = lambda x: x**2

    with pytest.raises(TypeError, match="entier"):
        grid_search_minimum(
            function,
            0,
            1,
            points=100.5,
        )


def test_grid_search_minimum_rejects_boolean_points():
    """Un booléen ne doit pas être accepté comme nombre de points."""
    function = lambda x: x**2

    with pytest.raises(TypeError, match="entier"):
        grid_search_minimum(
            function,
            0,
            1,
            points=True,
        )


# ============================================================
# SECTION DORÉE — TESTS DE BASE
# ============================================================


def test_golden_section_minimum_quadratic():
    """Trouve le minimum d'une fonction quadratique simple."""
    function = lambda x: (x - 2) ** 2 + 1

    x_min, f_min = golden_section_minimum(
        function,
        0,
        4,
    )

    assert x_min == pytest.approx(2.0, abs=1e-5)
    assert f_min == pytest.approx(1.0, abs=1e-10)


def test_golden_section_minimum_another_quadratic():
    """Teste une fonction quadratique avec un minimum négatif."""
    function = lambda x: (x + 3) ** 2 - 5

    x_min, f_min = golden_section_minimum(
        function,
        -6,
        0,
    )

    assert x_min == pytest.approx(-3.0, abs=1e-5)
    assert f_min == pytest.approx(-5.0, abs=1e-10)


def test_golden_section_minimum_non_symmetric_interval():
    """Teste un intervalle non symétrique autour du minimum."""
    function = lambda x: (x - 1.5) ** 2 + 2

    x_min, f_min = golden_section_minimum(
        function,
        -4,
        7,
    )

    assert x_min == pytest.approx(1.5, abs=1e-5)
    assert f_min == pytest.approx(2.0, abs=1e-10)


def test_golden_section_minimum_shifted_quadratic():
    """Teste une fonction quadratique avec un minimum différent."""
    function = lambda x: 3 * (x - 4.25) ** 2 - 7

    x_min, f_min = golden_section_minimum(
        function,
        0,
        8,
    )

    assert x_min == pytest.approx(4.25, abs=1e-5)
    assert f_min == pytest.approx(-7.0, abs=1e-9)


# ============================================================
# SECTION DORÉE — MINIMUM PROCHE D'UNE BORNE
# ============================================================


def test_golden_section_minimum_near_left_bound():
    """Teste un minimum situé près de la borne gauche."""
    function = lambda x: (x - 0.1) ** 2 + 3

    x_min, f_min = golden_section_minimum(
        function,
        0,
        10,
    )

    assert x_min == pytest.approx(0.1, abs=1e-5)
    assert f_min == pytest.approx(3.0, abs=1e-9)


def test_golden_section_minimum_near_right_bound():
    """Teste un minimum situé près de la borne droite."""
    function = lambda x: (x - 9.9) ** 2 + 3

    x_min, f_min = golden_section_minimum(
        function,
        0,
        10,
    )

    assert x_min == pytest.approx(9.9, abs=1e-5)
    assert f_min == pytest.approx(3.0, abs=1e-9)


# ============================================================
# SECTION DORÉE — FONCTION SCALAIRE
# ============================================================


def test_golden_section_minimum_scalar_only_function():
    """
    Vérifie qu'une fonction acceptant uniquement des scalaires
    fonctionne correctement.
    """

    def function(x):
        if isinstance(x, np.ndarray):
            raise TypeError("scalar only")

        return (x - 3) ** 2 + 2

    x_min, f_min = golden_section_minimum(
        function,
        0,
        6,
    )

    assert x_min == pytest.approx(3.0, abs=1e-5)
    assert f_min == pytest.approx(2.0, abs=1e-9)


# ============================================================
# SECTION DORÉE — FONCTION TRIGONOMÉTRIQUE
# ============================================================


def test_golden_section_minimum_trigonometric_function():
    """Teste une fonction unimodale sur l'intervalle choisi."""
    function = lambda x: np.cos(x)

    x_min, f_min = golden_section_minimum(
        function,
        2.5,
        3.8,
    )

    assert x_min == pytest.approx(np.pi, abs=1e-5)
    assert f_min == pytest.approx(-1.0, abs=1e-10)


# ============================================================
# SECTION DORÉE — PRÉCISION
# ============================================================


def test_golden_section_minimum_precision():
    """Vérifie que la tolérance contrôle la précision."""
    function = lambda x: (x - 1.234567) ** 2

    x_min, f_min = golden_section_minimum(
        function,
        0,
        2,
        tolerance=1e-10,
    )

    assert x_min == pytest.approx(1.234567, abs=1e-8)
    assert f_min == pytest.approx(0.0, abs=1e-14)


def test_golden_section_minimum_tighter_tolerance():
    """
    Vérifie qu'une tolérance plus fine permet une localisation
    au moins aussi précise du minimum.
    """
    function = lambda x: (x - 1.35791) ** 2

    x_min_coarse, _ = golden_section_minimum(
        function,
        0,
        2,
        tolerance=1e-3,
    )

    x_min_fine, _ = golden_section_minimum(
        function,
        0,
        2,
        tolerance=1e-8,
    )

    coarse_error = abs(x_min_coarse - 1.35791)
    fine_error = abs(x_min_fine - 1.35791)

    assert fine_error <= coarse_error
    assert fine_error < 1e-6


# ============================================================
# SECTION DORÉE — TYPES DE RETOUR
# ============================================================


def test_golden_section_minimum_returns_floats():
    """Vérifie que le résultat contient deux nombres flottants."""
    function = lambda x: (x - 1) ** 2

    x_min, f_min = golden_section_minimum(
        function,
        0,
        2,
    )

    assert isinstance(x_min, float)
    assert isinstance(f_min, float)


# ============================================================
# SECTION DORÉE — VALIDATION DE LA FONCTION
# ============================================================


def test_golden_section_minimum_rejects_non_callable_function():
    """Une fonction non appelable doit être rejetée."""
    with pytest.raises(TypeError, match="appelable"):
        golden_section_minimum(
            42,
            0,
            1,
        )


def test_golden_section_minimum_rejects_non_numeric_output():
    """Une fonction retournant du texte doit être rejetée."""
    function = lambda x: "invalid"

    with pytest.raises(TypeError, match="valeurs numériques"):
        golden_section_minimum(
            function,
            0,
            1,
        )


def test_golden_section_minimum_rejects_nan_output():
    """Une fonction retournant NaN doit être rejetée."""
    function = lambda x: np.nan

    with pytest.raises(ValueError, match="non finies"):
        golden_section_minimum(
            function,
            0,
            1,
        )


def test_golden_section_minimum_rejects_infinite_output():
    """Une fonction retournant l'infini doit être rejetée."""
    function = lambda x: np.inf

    with pytest.raises(ValueError, match="non finies"):
        golden_section_minimum(
            function,
            0,
            1,
        )


# ============================================================
# SECTION DORÉE — VALIDATION DES BORNES
# ============================================================


def test_golden_section_minimum_rejects_equal_bounds():
    """Les deux bornes ne peuvent pas être identiques."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="strictement inférieure"):
        golden_section_minimum(
            function,
            1,
            1,
        )


def test_golden_section_minimum_rejects_reversed_bounds():
    """La borne inférieure doit être plus petite."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="strictement inférieure"):
        golden_section_minimum(
            function,
            5,
            1,
        )


def test_golden_section_minimum_rejects_non_finite_bounds():
    """Les bornes infinies ou NaN sont refusées."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="finies"):
        golden_section_minimum(
            function,
            -np.inf,
            1,
        )

    with pytest.raises(ValueError, match="finies"):
        golden_section_minimum(
            function,
            0,
            np.inf,
        )


# ============================================================
# SECTION DORÉE — VALIDATION DE LA TOLÉRANCE
# ============================================================


def test_golden_section_minimum_rejects_zero_tolerance():
    """Une tolérance nulle doit être refusée."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="strictement positive"):
        golden_section_minimum(
            function,
            0,
            1,
            tolerance=0,
        )


def test_golden_section_minimum_rejects_negative_tolerance():
    """Une tolérance négative doit être refusée."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="strictement positive"):
        golden_section_minimum(
            function,
            0,
            1,
            tolerance=-1e-6,
        )


def test_golden_section_minimum_rejects_non_finite_tolerance():
    """Une tolérance infinie ou NaN doit être refusée."""
    function = lambda x: x**2

    with pytest.raises(ValueError, match="finie"):
        golden_section_minimum(
            function,
            0,
            1,
            tolerance=np.inf,
        )

    with pytest.raises(ValueError, match="finie"):
        golden_section_minimum(
            function,
            0,
            1,
            tolerance=np.nan,
        )


def test_golden_section_minimum_rejects_non_numeric_tolerance():
    """Une tolérance non numérique doit être rejetée."""
    function = lambda x: x**2

    with pytest.raises(TypeError, match="numérique"):
        golden_section_minimum(
            function,
            0,
            1,
            tolerance="invalid",
        )


# ============================================================
# SECTION DORÉE — VALIDATION DES ITÉRATIONS
# ============================================================


def test_golden_section_minimum_rejects_zero_iterations():
    """Le nombre d'itérations doit être strictement positif."""
    function = lambda x: x**2

    with pytest.raises(
        ValueError,
        match="strictement positif",
    ):
        golden_section_minimum(
            function,
            0,
            1,
            max_iterations=0,
        )


def test_golden_section_minimum_rejects_negative_iterations():
    """Un nombre négatif d'itérations doit être refusé."""
    function = lambda x: x**2

    with pytest.raises(
        ValueError,
        match="strictement positif",
    ):
        golden_section_minimum(
            function,
            0,
            1,
            max_iterations=-10,
        )


def test_golden_section_minimum_rejects_non_integer_iterations():
    """Le nombre d'itérations doit être un entier."""
    function = lambda x: x**2

    with pytest.raises(TypeError, match="entier"):
        golden_section_minimum(
            function,
            0,
            1,
            max_iterations=100.5,
        )


def test_golden_section_minimum_rejects_boolean_iterations():
    """Un booléen ne doit pas être accepté."""
    function = lambda x: x**2

    with pytest.raises(TypeError, match="entier"):
        golden_section_minimum(
            function,
            0,
            1,
            max_iterations=True,
        )


def test_golden_section_minimum_respects_max_iterations():
    """
    Vérifie que l'algorithme peut s'arrêter avant d'atteindre
    la tolérance lorsque le nombre maximal d'itérations est faible.
    """
    function = lambda x: (x - 2) ** 2

    x_min, f_min = golden_section_minimum(
        function,
        0,
        4,
        tolerance=1e-15,
        max_iterations=1,
    )

    assert 0 <= x_min <= 4
    assert np.isfinite(f_min)


# ============================================================
# NEWTON — TESTS DE BASE
# ============================================================


def test_newton_minimum_quadratic():
    """Trouve le minimum d'une fonction quadratique simple."""
    function = lambda x: (x - 2) ** 2 + 1
    derivative = lambda x: 2 * (x - 2)
    second_derivative = lambda x: 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=0,
    )

    assert x_min == pytest.approx(2.0, abs=1e-10)
    assert f_min == pytest.approx(1.0, abs=1e-10)


def test_newton_minimum_another_quadratic():
    """Teste une fonction quadratique avec un minimum négatif."""
    function = lambda x: (x + 3) ** 2 - 5
    derivative = lambda x: 2 * (x + 3)
    second_derivative = lambda x: 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=0,
    )

    assert x_min == pytest.approx(-3.0, abs=1e-10)
    assert f_min == pytest.approx(-5.0, abs=1e-10)


def test_newton_minimum_shifted_scaled_quadratic():
    """Teste une fonction quadratique décalée et mise à l'échelle."""
    function = lambda x: 3 * (x - 4.25) ** 2 - 7
    derivative = lambda x: 6 * (x - 4.25)
    second_derivative = lambda x: 6

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=0,
    )

    assert x_min == pytest.approx(4.25, abs=1e-10)
    assert f_min == pytest.approx(-7.0, abs=1e-10)


# ============================================================
# NEWTON — POINTS INITIAUX
# ============================================================


def test_newton_minimum_from_left_initial_point():
    """Teste Newton avec un point initial situé à gauche du minimum."""
    function = lambda x: (x - 2) ** 2 + 1
    derivative = lambda x: 2 * (x - 2)
    second_derivative = lambda x: 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=-10,
    )

    assert x_min == pytest.approx(2.0, abs=1e-10)
    assert f_min == pytest.approx(1.0, abs=1e-10)


def test_newton_minimum_from_right_initial_point():
    """Teste Newton avec un point initial situé à droite du minimum."""
    function = lambda x: (x - 2) ** 2 + 1
    derivative = lambda x: 2 * (x - 2)
    second_derivative = lambda x: 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=10,
    )

    assert x_min == pytest.approx(2.0, abs=1e-10)
    assert f_min == pytest.approx(1.0, abs=1e-10)


# ============================================================
# NEWTON — FONCTION NON LINÉAIRE
# ============================================================


def test_newton_minimum_nonlinear_function():
    """
    Vérifie Newton sur une fonction non quadratique nécessitant
    plusieurs itérations.
    """
    function = lambda x: (x - 1) ** 4 + (x - 1) ** 2
    derivative = lambda x: 4 * (x - 1) ** 3 + 2 * (x - 1)
    second_derivative = lambda x: 12 * (x - 1) ** 2 + 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=3,
    )

    assert x_min == pytest.approx(1.0, abs=1e-8)
    assert f_min == pytest.approx(0.0, abs=1e-12)


# ============================================================
# NEWTON — TYPES DE RETOUR
# ============================================================


def test_newton_minimum_returns_floats():
    """Vérifie que le résultat contient deux nombres flottants."""
    function = lambda x: (x - 1) ** 2
    derivative = lambda x: 2 * (x - 1)
    second_derivative = lambda x: 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=0,
    )

    assert isinstance(x_min, float)
    assert isinstance(f_min, float)


# ============================================================
# NEWTON — VALIDATION DES FONCTIONS
# ============================================================


def test_newton_minimum_rejects_non_callable_function():
    """L'objectif doit être appelable."""
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="appelable"):
        newton_minimum(
            42,
            derivative,
            second_derivative,
            x0=0,
        )


def test_newton_minimum_rejects_non_callable_derivative():
    """La première dérivée doit être appelable."""
    function = lambda x: x**2
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="appelable"):
        newton_minimum(
            function,
            42,
            second_derivative,
            x0=0,
        )


def test_newton_minimum_rejects_non_callable_second_derivative():
    """La deuxième dérivée doit être appelable."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x

    with pytest.raises(TypeError, match="appelable"):
        newton_minimum(
            function,
            derivative,
            42,
            x0=0,
        )


def test_newton_minimum_rejects_non_numeric_derivative():
    """La première dérivée doit retourner une valeur numérique."""
    function = lambda x: x**2
    derivative = lambda x: "invalid"
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="valeurs numériques"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=0,
        )


def test_newton_minimum_rejects_non_numeric_second_derivative():
    """La deuxième dérivée doit retourner une valeur numérique."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: "invalid"

    with pytest.raises(TypeError, match="valeurs numériques"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=0,
        )


def test_newton_minimum_rejects_non_numeric_function_output():
    """L'objectif doit retourner une valeur numérique."""
    function = lambda x: "invalid"
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="valeurs numériques"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=0,
        )


# ============================================================
# NEWTON — VALIDATION DES VALEURS FINIES
# ============================================================


def test_newton_minimum_rejects_non_finite_x0():
    """Le point initial doit être fini."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(ValueError, match="point initial"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=np.inf,
        )

    with pytest.raises(ValueError, match="point initial"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=np.nan,
        )


def test_newton_minimum_rejects_non_numeric_x0():
    """Le point initial doit être numérique."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="point initial"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0="invalid",
        )


def test_newton_minimum_rejects_non_finite_derivative():
    """Une dérivée non finie doit être rejetée."""
    function = lambda x: x**2
    derivative = lambda x: np.nan
    second_derivative = lambda x: 2

    with pytest.raises(ValueError, match="non finies"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
        )


def test_newton_minimum_rejects_non_finite_second_derivative():
    """Une deuxième dérivée non finie doit être rejetée."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: np.inf

    with pytest.raises(ValueError, match="non finies"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
        )


def test_newton_minimum_rejects_non_finite_function_output():
    """L'objectif ne doit pas retourner NaN ou inf."""
    function = lambda x: np.nan
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(ValueError, match="non finies"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=0,
        )


# ============================================================
# NEWTON — DÉRIVÉE SECONDE
# ============================================================


def test_newton_minimum_rejects_zero_second_derivative():
    """Une deuxième dérivée nulle doit être refusée."""
    function = lambda x: x**3
    derivative = lambda x: 3 * x**2
    second_derivative = lambda x: 0

    with pytest.raises(
        ValueError,
        match="dérivée seconde",
    ):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
        )


def test_newton_minimum_rejects_near_zero_second_derivative():
    """Une deuxième dérivée trop proche de zéro doit être refusée."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 1e-20

    with pytest.raises(
        ValueError,
        match="dérivée seconde",
    ):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
        )


# ============================================================
# NEWTON — VALIDATION DE LA TOLÉRANCE
# ============================================================


def test_newton_minimum_rejects_zero_tolerance():
    """Une tolérance nulle doit être refusée."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(
        ValueError,
        match="strictement positive",
    ):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            tolerance=0,
        )


def test_newton_minimum_rejects_negative_tolerance():
    """Une tolérance négative doit être refusée."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(
        ValueError,
        match="strictement positive",
    ):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            tolerance=-1e-6,
        )


def test_newton_minimum_rejects_non_finite_tolerance():
    """Une tolérance infinie ou NaN doit être refusée."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(ValueError, match="finie"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            tolerance=np.inf,
        )

    with pytest.raises(ValueError, match="finie"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            tolerance=np.nan,
        )


def test_newton_minimum_rejects_non_numeric_tolerance():
    """Une tolérance non numérique doit être rejetée."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="numérique"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            tolerance="invalid",
        )


# ============================================================
# NEWTON — VALIDATION DU NOMBRE D'ITÉRATIONS
# ============================================================


def test_newton_minimum_rejects_zero_iterations():
    """Le nombre maximal d'itérations doit être positif."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(
        ValueError,
        match="strictement positif",
    ):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            max_iterations=0,
        )


def test_newton_minimum_rejects_negative_iterations():
    """Un nombre négatif d'itérations doit être refusé."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(
        ValueError,
        match="strictement positif",
    ):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            max_iterations=-10,
        )


def test_newton_minimum_rejects_non_integer_iterations():
    """Le nombre maximal d'itérations doit être entier."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="entier"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            max_iterations=10.5,
        )


def test_newton_minimum_rejects_boolean_iterations():
    """Un booléen ne doit pas être accepté."""
    function = lambda x: x**2
    derivative = lambda x: 2 * x
    second_derivative = lambda x: 2

    with pytest.raises(TypeError, match="entier"):
        newton_minimum(
            function,
            derivative,
            second_derivative,
            x0=1,
            max_iterations=True,
        )


def test_newton_minimum_respects_max_iterations():
    """
    Vérifie que Newton peut s'arrêter après un nombre limité
    d'itérations sans produire de valeur non finie.
    """
    function = lambda x: (x - 2) ** 2 + 1
    derivative = lambda x: 2 * (x - 2)
    second_derivative = lambda x: 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=10,
        tolerance=1e-15,
        max_iterations=1,
    )

    assert np.isfinite(x_min)
    assert np.isfinite(f_min)


# ============================================================
# NEWTON — PRÉCISION
# ============================================================


def test_newton_minimum_precision():
    """Vérifie que Newton atteint une bonne précision."""
    function = lambda x: (x - 1.234567) ** 2
    derivative = lambda x: 2 * (x - 1.234567)
    second_derivative = lambda x: 2

    x_min, f_min = newton_minimum(
        function,
        derivative,
        second_derivative,
        x0=0,
        tolerance=1e-10,
    )

    assert x_min == pytest.approx(1.234567, abs=1e-10)
    assert f_min == pytest.approx(0.0, abs=1e-15)

