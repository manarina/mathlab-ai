from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from core.numerical.root_finding import (
    RootFindingResult,
    bisection,
    newton_raphson,
    secant,
)


Number = int | float


# ============================================================
# OUTILS DE FORMATAGE
# ============================================================


def _format_number(value: Number, decimals: int = 10) -> str:
    """
    Formate une valeur numérique pour les explications.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("La valeur doit être numérique.")

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError("La valeur doit être finie.")

    if numeric_value == 0:
        return "0"

    return f"{numeric_value:.{decimals}f}".rstrip("0").rstrip(".")


def _format_result(result: RootFindingResult) -> str:
    """
    Formate un résultat de recherche de racine.
    """
    if not isinstance(result, RootFindingResult):
        raise TypeError(
            "Le résultat doit être une instance de RootFindingResult."
        )

    return (
        f"Racine approchée : "
        f"{_format_number(result.root)}\n"
        f"Nombre d'itérations : {result.iterations}\n"
        f"Erreur finale : {result.error:.3e}"
    )


def _validate_result(
    result: RootFindingResult | None,
) -> RootFindingResult | None:
    """
    Valide un résultat optionnel.
    """
    if result is not None and not isinstance(result, RootFindingResult):
        raise TypeError(
            "Le résultat doit être une instance de RootFindingResult."
        )

    return result


# ============================================================
# DICHOTOMIE
# ============================================================


def explain_bisection(
    function: Callable[[float], float],
    lower: Number,
    upper: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
    result: RootFindingResult | None = None,
) -> str:
    """
    Explique la méthode de dichotomie.
    """
    if not callable(function):
        raise TypeError("La fonction doit être appelable.")

    _validate_result(result)

    if result is None:
        result = bisection(
            function,
            lower,
            upper,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    f_lower = function(float(lower))
    f_upper = function(float(upper))

    return (
        "### 🔵 Méthode de dichotomie\n\n"
        "La méthode de dichotomie recherche une racine d'une fonction "
        "continue sur un intervalle où la fonction change de signe.\n\n"
        "#### Principe\n\n"
        "On part d'un intervalle "
        f"\\([a,b] = [{_format_number(lower)}, "
        f"{_format_number(upper)}]\\).\n\n"
        f"On calcule les valeurs "
        f"\\(f(a) = {_format_number(f_lower)}\\) et "
        f"\\(f(b) = {_format_number(f_upper)}\\).\n\n"
        "Si \\(f(a)\\) et \\(f(b)\\) sont de signes opposés, "
        "le théorème des valeurs intermédiaires garantit l'existence "
        "d'au moins une racine dans l'intervalle si la fonction est continue.\n\n"
        "À chaque étape, on calcule le milieu :\n\n"
        "\\[\n"
        "m = \\frac{a+b}{2}\n"
        "\\]\n\n"
        "Puis on conserve la moitié de l'intervalle contenant "
        "le changement de signe.\n\n"
        "#### Critère d'arrêt\n\n"
        "L'algorithme s'arrête lorsque la largeur de l'intervalle "
        "devient suffisamment petite ou lorsque la racine est trouvée "
        "exactement.\n\n"
        f"#### Résultat\n\n"
        f"{_format_result(result)}"
    )


# ============================================================
# NEWTON-RAPHSON
# ============================================================


def explain_newton_raphson(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
    result: RootFindingResult | None = None,
) -> str:
    """
    Explique la méthode de Newton-Raphson.
    """
    if not callable(function):
        raise TypeError("La fonction doit être appelable.")

    if not callable(derivative):
        raise TypeError("La dérivée doit être appelable.")

    _validate_result(result)

    if result is None:
        result = newton_raphson(
            function,
            derivative,
            initial_guess,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    initial_value = function(float(initial_guess))

    return (
        "### 🟢 Méthode de Newton-Raphson\n\n"
        "La méthode de Newton-Raphson utilise la dérivée de la fonction "
        "pour approcher rapidement une racine de l'équation "
        "\\(f(x)=0\\).\n\n"
        "#### Principe géométrique\n\n"
        "À partir d'une approximation \\(x_n\\), on considère la tangente "
        "à la courbe de \\(f\\). Le point où cette tangente coupe l'axe "
        "des abscisses fournit l'approximation suivante.\n\n"
        "La formule d'itération est :\n\n"
        "\\[\n"
        "x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}\n"
        "\\]\n\n"
        f"On commence avec "
        f"\\(x_0 = {_format_number(initial_guess)}\\).\n\n"
        f"Pour cette valeur, "
        f"\\(f(x_0) = {_format_number(initial_value)}\\).\n\n"
        "#### Critère d'arrêt\n\n"
        "On arrête généralement les calculs lorsque la différence "
        "entre deux approximations successives devient inférieure "
        "à la tolérance choisie.\n\n"
        "#### Attention\n\n"
        "La méthode nécessite que la dérivée ne soit pas nulle "
        "au cours des itérations. Le choix de l'approximation initiale "
        "peut également influencer la convergence.\n\n"
        f"#### Résultat\n\n"
        f"{_format_result(result)}"
    )


# ============================================================
# SÉCANTE
# ============================================================


def explain_secant(
    function: Callable[[float], float],
    first_guess: Number,
    second_guess: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
    result: RootFindingResult | None = None,
) -> str:
    """
    Explique la méthode de la sécante.
    """
    if not callable(function):
        raise TypeError("La fonction doit être appelable.")

    _validate_result(result)

    if result is None:
        result = secant(
            function,
            first_guess,
            second_guess,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    first_value = function(float(first_guess))
    second_value = function(float(second_guess))

    return (
        "### 🟠 Méthode de la sécante\n\n"
        "La méthode de la sécante recherche une racine de "
        "\\(f(x)=0\\) en utilisant deux approximations initiales.\n\n"
        "Contrairement à Newton-Raphson, elle n'utilise pas directement "
        "la dérivée de la fonction.\n\n"
        "#### Principe géométrique\n\n"
        "On construit une droite passant par les deux points :\n\n"
        f"\\(({_format_number(first_guess)}, "
        f"{_format_number(first_value)})\\) et "
        f"\\(({_format_number(second_guess)}, "
        f"{_format_number(second_value)})\\).\n\n"
        "L'intersection de cette droite avec l'axe des abscisses "
        "donne l'approximation suivante.\n\n"
        "La formule est :\n\n"
        "\\[\n"
        "x_{n+1} = x_n - "
        "\\frac{f(x_n)(x_n-x_{n-1})}"
        "{f(x_n)-f(x_{n-1})}\n"
        "\\]\n\n"
        "#### Critère d'arrêt\n\n"
        "Les itérations sont arrêtées lorsque la différence entre "
        "deux approximations successives devient inférieure "
        "à la tolérance choisie.\n\n"
        "#### Attention\n\n"
        "La méthode ne nécessite pas explicitement la dérivée, "
        "mais son dénominateur ne doit pas devenir nul.\n\n"
        f"#### Résultat\n\n"
        f"{_format_result(result)}"
    )


# ============================================================
# COMPARAISON DES MÉTHODES
# ============================================================


def explain_root_finding_comparison(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    lower: Number,
    upper: Number,
    initial_guess: Number,
    first_guess: Number,
    second_guess: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
) -> str:
    """
    Compare les trois méthodes de recherche de racine.
    """
    if not callable(function):
        raise TypeError("La fonction doit être appelable.")

    if not callable(derivative):
        raise TypeError("La dérivée doit être appelable.")

    bisection_result = bisection(
        function,
        lower,
        upper,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    newton_result = newton_raphson(
        function,
        derivative,
        initial_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    secant_result = secant(
        function,
        first_guess,
        second_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    return (
        "### 📊 Comparaison des méthodes\n\n"
        "Les trois méthodes permettent d'obtenir une approximation "
        "d'une racine de \\(f(x)=0\\), mais elles utilisent des principes "
        "différents.\n\n"
        "| Méthode | Principe | Dérivée nécessaire ? |\n"
        "|---|---|---|\n"
        "| Dichotomie | Réduction successive d'un intervalle | Non |\n"
        "| Newton-Raphson | Tangente à la courbe | Oui |\n"
        "| Sécante | Droite passant par deux approximations | Non |\n\n"
        "#### Résultats numériques\n\n"
        f"**Dichotomie**\n\n"
        f"{_format_result(bisection_result)}\n\n"
        f"**Newton-Raphson**\n\n"
        f"{_format_result(newton_result)}\n\n"
        f"**Sécante**\n\n"
        f"{_format_result(secant_result)}\n\n"
        "Les nombres d'itérations et les erreurs permettent de comparer "
        "le comportement numérique des méthodes sur le problème étudié."
    )


# ============================================================
# EXPLICATION GÉNÉRALE
# ============================================================


def explain_root_finding() -> str:
    """
    Présente les principes généraux de la recherche numérique de racines.
    """
    return (
        "## 🔬 Recherche numérique de racines\n\n"
        "La recherche de racines consiste à déterminer une valeur "
        "\\(x\\) telle que :\n\n"
        "\\[\n"
        "f(x)=0\n"
        "\\]\n\n"
        "Lorsque cette équation ne peut pas être résolue facilement "
        "de manière analytique, les méthodes numériques permettent "
        "de calculer une approximation de la solution.\n\n"
        "### Méthodes disponibles\n\n"
        "- **Dichotomie** : méthode robuste basée sur un changement de signe.\n"
        "- **Newton-Raphson** : méthode utilisant la dérivée.\n"
        "- **Sécante** : méthode utilisant deux approximations sans "
        "calculer explicitement la dérivée.\n\n"
        "### Notion de tolérance\n\n"
        "La tolérance détermine le niveau de précision recherché. "
        "Plus la tolérance est petite, plus l'approximation recherchée "
        "est précise, mais le nombre d'itérations peut augmenter.\n\n"
        "### Important\n\n"
        "Une solution numérique est généralement une approximation. "
        "Il faut donc toujours examiner l'erreur et le critère "
        "de convergence."
    )


# ============================================================
# ALIAS
# ============================================================


def explain_bisection_method(
    function: Callable[[float], float],
    lower: Number,
    upper: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
    result: RootFindingResult | None = None,
) -> str:
    """
    Alias explicite de explain_bisection().
    """
    return explain_bisection(
        function,
        lower,
        upper,
        tolerance=tolerance,
        max_iterations=max_iterations,
        result=result,
    )


def explain_newton_method(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
    result: RootFindingResult | None = None,
) -> str:
    """
    Alias explicite de explain_newton_raphson().
    """
    return explain_newton_raphson(
        function,
        derivative,
        initial_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
        result=result,
    )


def explain_secant_method(
    function: Callable[[float], float],
    first_guess: Number,
    second_guess: Number,
    tolerance: Number = 1e-10,
    max_iterations: int = 100,
    result: RootFindingResult | None = None,
) -> str:
    """
    Alias explicite de explain_secant().
    """
    return explain_secant(
        function,
        first_guess,
        second_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
        result=result,
    )