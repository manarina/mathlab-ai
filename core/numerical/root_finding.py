from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


Number = int | float


@dataclass(frozen=True)
class RootFindingResult:
    """
    Résultat d'une méthode de recherche de racine.

    Attributes:
        root: approximation de la racine.
        iterations: nombre d'itérations effectuées.
        error: erreur finale estimée.
    """

    root: float
    iterations: int
    error: float


def _validate_function(function: Callable[[float], float]) -> None:
    """Vérifie que l'objet fourni est bien appelable."""
    if not callable(function):
        raise TypeError("La fonction doit être appelable.")


def _validate_tolerance(tolerance: Number) -> float:
    """Valide et retourne la tolérance numérique."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError("La tolérance doit être un nombre.")

    tolerance_value = float(tolerance)

    if not math.isfinite(tolerance_value):
        raise ValueError("La tolérance doit être finie.")

    if tolerance_value <= 0:
        raise ValueError("La tolérance doit être strictement positive.")

    return tolerance_value


def _validate_max_iterations(max_iterations: int) -> int:
    """Valide le nombre maximal d'itérations."""
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("Le nombre maximal d'itérations doit être un entier.")

    if max_iterations <= 0:
        raise ValueError(
            "Le nombre maximal d'itérations doit être strictement positif."
        )

    return max_iterations


def _evaluate(function: Callable[[float], float], x: float) -> float:
    """Évalue une fonction et vérifie que le résultat est numérique et fini."""
    try:
        value = function(x)
    except Exception as exc:
        raise ValueError(
            f"Impossible d'évaluer la fonction en x = {x}."
        ) from exc

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "La fonction doit retourner une valeur numérique."
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"La fonction retourne une valeur non finie en x = {x}."
        )

    return value


def bisection(
    function: Callable[[float], float],
    lower: Number,
    upper: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
) -> RootFindingResult:
    """
    Recherche une racine par la méthode de dichotomie.

    La méthode nécessite que la fonction change de signe entre
    lower et upper.

    Args:
        function: fonction f(x).
        lower: borne inférieure.
        upper: borne supérieure.
        tolerance: tolérance d'arrêt.
        max_iterations: nombre maximal d'itérations.

    Returns:
        RootFindingResult contenant la racine approchée,
        le nombre d'itérations et l'erreur finale.
    """
    _validate_function(function)

    tolerance_value = _validate_tolerance(tolerance)
    max_iterations_value = _validate_max_iterations(max_iterations)

    if isinstance(lower, bool) or not isinstance(lower, (int, float)):
        raise TypeError("La borne inférieure doit être un nombre.")

    if isinstance(upper, bool) or not isinstance(upper, (int, float)):
        raise TypeError("La borne supérieure doit être un nombre.")

    a = float(lower)
    b = float(upper)

    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Les bornes doivent être finies.")

    if a >= b:
        raise ValueError(
            "La borne inférieure doit être strictement inférieure "
            "à la borne supérieure."
        )

    f_a = _evaluate(function, a)
    f_b = _evaluate(function, b)

    if f_a == 0:
        return RootFindingResult(
            root=a,
            iterations=0,
            error=0.0,
        )

    if f_b == 0:
        return RootFindingResult(
            root=b,
            iterations=0,
            error=0.0,
        )

    if f_a * f_b > 0:
        raise ValueError(
            "La fonction doit changer de signe sur l'intervalle."
        )

    midpoint = (a + b) / 2
    error = abs(b - a) / 2

    for iteration in range(1, max_iterations_value + 1):
        midpoint = (a + b) / 2
        f_midpoint = _evaluate(function, midpoint)

        error = abs(b - a) / 2

        if f_midpoint == 0 or error <= tolerance_value:
            return RootFindingResult(
                root=midpoint,
                iterations=iteration,
                error=error,
            )

        if f_a * f_midpoint < 0:
            b = midpoint
            f_b = f_midpoint
        else:
            a = midpoint
            f_a = f_midpoint

    return RootFindingResult(
        root=midpoint,
        iterations=max_iterations_value,
        error=error,
    )


def newton_raphson(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
) -> RootFindingResult:
    """
    Recherche une racine par la méthode de Newton-Raphson.

    Formule :
        x_(n+1) = x_n - f(x_n) / f'(x_n)

    Args:
        function: fonction f(x).
        derivative: dérivée f'(x).
        initial_guess: valeur initiale.
        tolerance: tolérance d'arrêt.
        max_iterations: nombre maximal d'itérations.

    Returns:
        RootFindingResult contenant la racine approchée,
        le nombre d'itérations et l'erreur finale.
    """
    _validate_function(function)
    _validate_function(derivative)

    tolerance_value = _validate_tolerance(tolerance)
    max_iterations_value = _validate_max_iterations(max_iterations)

    if isinstance(initial_guess, bool) or not isinstance(
        initial_guess, (int, float)
    ):
        raise TypeError("L'approximation initiale doit être un nombre.")

    current = float(initial_guess)

    if not math.isfinite(current):
        raise ValueError(
            "L'approximation initiale doit être finie."
        )

    current_value = _evaluate(function, current)

    if current_value == 0:
        return RootFindingResult(
            root=current,
            iterations=0,
            error=0.0,
        )

    error = math.inf

    for iteration in range(1, max_iterations_value + 1):
        derivative_value = _evaluate(derivative, current)

        if derivative_value == 0:
            raise ValueError(
                "La dérivée est nulle pendant l'itération de Newton-Raphson."
            )

        next_value = current - current_value / derivative_value

        if not math.isfinite(next_value):
            raise ValueError(
                "La méthode de Newton-Raphson a produit une valeur non finie."
            )

        error = abs(next_value - current)

        next_function_value = _evaluate(function, next_value)

        if next_function_value == 0 or error <= tolerance_value:
            return RootFindingResult(
                root=next_value,
                iterations=iteration,
                error=error,
            )

        current = next_value
        current_value = next_function_value

    return RootFindingResult(
        root=current,
        iterations=max_iterations_value,
        error=error,
    )


def secant(
    function: Callable[[float], float],
    first_guess: Number,
    second_guess: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
) -> RootFindingResult:
    """
    Recherche une racine par la méthode de la sécante.

    Formule :
        x_(n+1) =
        x_n - f(x_n) * (x_n - x_(n-1))
        / (f(x_n) - f(x_(n-1)))

    Args:
        function: fonction f(x).
        first_guess: première approximation.
        second_guess: deuxième approximation.
        tolerance: tolérance d'arrêt.
        max_iterations: nombre maximal d'itérations.

    Returns:
        RootFindingResult contenant la racine approchée,
        le nombre d'itérations et l'erreur finale.
    """
    _validate_function(function)

    tolerance_value = _validate_tolerance(tolerance)
    max_iterations_value = _validate_max_iterations(max_iterations)

    if isinstance(first_guess, bool) or not isinstance(
        first_guess, (int, float)
    ):
        raise TypeError("La première approximation doit être un nombre.")

    if isinstance(second_guess, bool) or not isinstance(
        second_guess, (int, float)
    ):
        raise TypeError("La deuxième approximation doit être un nombre.")

    previous = float(first_guess)
    current = float(second_guess)

    if not math.isfinite(previous) or not math.isfinite(current):
        raise ValueError(
            "Les approximations initiales doivent être finies."
        )

    if previous == current:
        raise ValueError(
            "Les deux approximations initiales doivent être différentes."
        )

    previous_value = _evaluate(function, previous)
    current_value = _evaluate(function, current)

    if previous_value == 0:
        return RootFindingResult(
            root=previous,
            iterations=0,
            error=0.0,
        )

    if current_value == 0:
        return RootFindingResult(
            root=current,
            iterations=0,
            error=0.0,
        )

    error = abs(current - previous)

    for iteration in range(1, max_iterations_value + 1):
        denominator = current_value - previous_value

        if denominator == 0:
            raise ValueError(
                "Impossible de poursuivre la méthode de la sécante : "
                "le dénominateur est nul."
            )

        next_value = (
            current
            - current_value * (current - previous) / denominator
        )

        if not math.isfinite(next_value):
            raise ValueError(
                "La méthode de la sécante a produit une valeur non finie."
            )

        error = abs(next_value - current)

        next_function_value = _evaluate(function, next_value)

        if next_function_value == 0 or error <= tolerance_value:
            return RootFindingResult(
                root=next_value,
                iterations=iteration,
                error=error,
            )

        previous = current
        previous_value = current_value

        current = next_value
        current_value = next_function_value

    return RootFindingResult(
        root=current,
        iterations=max_iterations_value,
        error=error,
    )