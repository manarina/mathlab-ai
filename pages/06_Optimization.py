
"""
MathLab AI — Optimization Lab

Interface Streamlit pour l'optimisation numérique d'une fonction
réelle à une variable.

Méthodes disponibles :
- Grid Search
- Section dorée
- Newton

Version initiale :
- Fonction objectif
- Intervalle [a, b]
- Minimum / maximum
- Résultat numérique
- Explication pédagogique
"""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from core.optimization.constraints import validate_bounds
from core.optimization.objectives import evaluate_objective
from core.optimization.one_dimensional import (
    golden_section_minimum,
    grid_search_minimum,
    newton_minimum,
)
from core.optimization.optimization_explanations import (
    explain_golden_section,
    explain_grid_search,
    explain_newton_minimum,
    explain_optimization,
)


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="MathLab AI — Optimization Lab",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# Fonctions utilitaires
# ============================================================


def _parse_function(expression: str):
    """
    Transforme une expression mathématique saisie par l'utilisateur
    en fonction Python.

    L'expression utilise x comme variable.

    Exemples :
        x**2 - 4*x + 5
        sin(x)
        exp(x)
        x**3 - 2*x
    """

    expression = expression.strip()

    if not expression:
        raise ValueError("Veuillez saisir une fonction objectif.")

    allowed_names = {
        "x": None,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }

    try:
        compiled = compile(expression, "<objective>", "eval")
    except SyntaxError as exc:
        raise ValueError("L'expression mathématique est invalide.") from exc

    forbidden_names = {
        name
        for name in compiled.co_names
        if name not in allowed_names
    }

    if forbidden_names:
        names = ", ".join(sorted(forbidden_names))
        raise ValueError(
            f"Nom(s) non autorisé(s) dans la fonction : {names}."
        )

    def function(x):
        local_names = dict(allowed_names)
        local_names["x"] = x

        try:
            return eval(
                compiled,
                {"__builtins__": {}},
                local_names,
            )
        except Exception as exc:
            raise ValueError(
                f"Impossible d'évaluer la fonction en x={x}."
            ) from exc

    return function


def _numerical_derivative(function, x: float) -> float:
    """
    Approximation numérique de la dérivée première.

    Cette fonction est utilisée uniquement pour permettre à Newton
    de fonctionner avec une fonction saisie par l'utilisateur.
    """
    h = 1e-5

    try:
        value = (
            evaluate_objective(function, x + h)
            - evaluate_objective(function, x - h)
        ) / (2.0 * h)
    except Exception as exc:
        raise ValueError(
            "Impossible de calculer la dérivée numérique."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            "La dérivée numérique produit une valeur non finie."
        )

    return float(value)


def _numerical_second_derivative(function, x: float) -> float:
    """
    Approximation numérique de la dérivée seconde.
    """
    h = 1e-4

    try:
        f_plus = evaluate_objective(function, x + h)
        f_current = evaluate_objective(function, x)
        f_minus = evaluate_objective(function, x - h)

        value = (f_plus - 2.0 * f_current + f_minus) / (h**2)
    except Exception as exc:
        raise ValueError(
            "Impossible de calculer la dérivée seconde numérique."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            "La dérivée seconde numérique produit une valeur non finie."
        )

    return float(value)


def _run_optimization(
    function,
    lower_bound: float,
    upper_bound: float,
    objective_type: str,
    method: str,
    points: int,
    tolerance: float,
    x0: float,
):
    """
    Exécute la méthode d'optimisation sélectionnée.
    """

    # --------------------------------------------------------
    # Pour un maximum, on optimise -f(x).
    # --------------------------------------------------------

    if objective_type == "Maximum":
        optimization_function = lambda x: -evaluate_objective(function, x)
    else:
        optimization_function = function

    if method == "Grid Search":
        x_opt, optimized_value = grid_search_minimum(
            optimization_function,
            lower_bound,
            upper_bound,
            points=points,
        )

    elif method == "Section dorée":
        x_opt, optimized_value = golden_section_minimum(
            optimization_function,
            lower_bound,
            upper_bound,
            tolerance=tolerance,
        )

    elif method == "Newton":
        derivative = lambda x: _numerical_derivative(
            optimization_function,
            x,
        )
        second_derivative = lambda x: _numerical_second_derivative(
            optimization_function,
            x,
        )

        x_opt, optimized_value = newton_minimum(
            optimization_function,
            derivative,
            second_derivative,
            x0,
            tolerance=tolerance,
        )

    else:
        raise ValueError("Méthode d'optimisation inconnue.")

    if objective_type == "Maximum":
        objective_value = -optimized_value
    else:
        objective_value = optimized_value

    return float(x_opt), float(objective_value)


def _create_plot(
    function,
    lower_bound: float,
    upper_bound: float,
    x_opt: float,
):
    """
    Crée les données du graphique de la fonction et du point optimal.
    """
    x_values = np.linspace(lower_bound, upper_bound, 500)

    y_values = []

    for x in x_values:
        try:
            y = evaluate_objective(function, float(x))
        except (TypeError, ValueError):
            y = np.nan

        y_values.append(y)

    chart_data = {
        "x": x_values,
        "f(x)": np.array(y_values, dtype=float),
    }

    st.line_chart(
        chart_data,
        x="x",
        y="f(x)",
        height=400,
    )

    optimal_value = evaluate_objective(function, x_opt)

    st.caption(
        f"Point optimal : x* = {x_opt:.6f} — "
        f"f(x*) = {optimal_value:.6f}"
    )


# ============================================================
# En-tête
# ============================================================

st.title("MathLab AI — Optimization Lab")

st.markdown(
    """
    Explorez l'**optimisation numérique d'une fonction réelle à une
    variable**.

    L'objectif est de rechercher numériquement un **minimum** ou un
    **maximum** de la fonction sur un intervalle donné.
    """
)

st.divider()


# ============================================================
# Introduction
# ============================================================

with st.expander("📘 Comprendre l'optimisation", expanded=False):
    st.markdown(explain_optimization())


# ============================================================
# Configuration du problème
# ============================================================

st.subheader("🎯 Définition du problème")

col1, col2 = st.columns([2, 1])

with col1:
    expression = st.text_input(
        "Fonction objectif f(x)",
        value="x**2 - 4*x + 5",
        help=(
            "Exemples : x**2 - 4*x + 5, "
            "sin(x), exp(x), x**3 - 2*x"
        ),
    )

with col2:
    objective_type = st.radio(
        "Objectif",
        options=["Minimum", "Maximum"],
        horizontal=True,
    )


# ============================================================
# Intervalle
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    lower_bound = st.number_input(
        "Borne inférieure a",
        value=0.0,
        step=0.5,
    )

with col2:
    upper_bound = st.number_input(
        "Borne supérieure b",
        value=5.0,
        step=0.5,
    )

with col3:
    points = st.number_input(
        "Nombre de points",
        min_value=2,
        max_value=10000,
        value=100,
        step=10,
    )


# ============================================================
# Méthode
# ============================================================

st.subheader("⚙️ Méthode numérique")

method = st.selectbox(
    "Choisir une méthode",
    options=[
        "Grid Search",
        "Section dorée",
        "Newton",
    ],
)

if method == "Grid Search":
    tolerance = 1e-6
    x0 = (lower_bound + upper_bound) / 2.0

    st.info(
        "Le Grid Search évalue la fonction sur un ensemble de points "
        "régulièrement espacés et sélectionne le meilleur point."
    )

elif method == "Section dorée":
    tolerance = st.number_input(
        "Tolérance",
        min_value=1e-12,
        max_value=1.0,
        value=1e-6,
        format="%.2e",
    )

    x0 = (lower_bound + upper_bound) / 2.0

    st.info(
        "La section dorée réduit progressivement l'intervalle de "
        "recherche afin de localiser l'optimum."
    )

else:
    tolerance = st.number_input(
        "Tolérance",
        min_value=1e-12,
        max_value=1.0,
        value=1e-6,
        format="%.2e",
    )

    x0 = st.number_input(
        "Point initial x₀",
        value=float((lower_bound + upper_bound) / 2.0),
        step=0.1,
    )

    st.info(
        "La méthode de Newton utilise les dérivées pour rechercher "
        "un point stationnaire rapidement."
    )


# ============================================================
# Calcul
# ============================================================

st.divider()

calculate = st.button(
    "🚀 Calculer l'optimum",
    type="primary",
    use_container_width=True,
)


if calculate:

    try:
        # ----------------------------------------------------
        # Validation de la fonction
        # ----------------------------------------------------

        function = _parse_function(expression)

        # ----------------------------------------------------
        # Validation des bornes
        # ----------------------------------------------------

        lower, upper = validate_bounds(
            lower_bound,
            upper_bound,
        )

        # ----------------------------------------------------
        # Validation du point initial pour Newton
        # ----------------------------------------------------

        if method == "Newton":
            if not math.isfinite(float(x0)):
                raise ValueError(
                    "Le point initial doit être une valeur finie."
                )

            if not lower <= float(x0) <= upper:
                raise ValueError(
                    "Le point initial x₀ doit appartenir à l'intervalle "
                    "[a, b]."
                )

        # ----------------------------------------------------
        # Test de la fonction sur l'intervalle
        # ----------------------------------------------------

        evaluate_objective(function, lower)
        evaluate_objective(function, upper)

        # ----------------------------------------------------
        # Exécution
        # ----------------------------------------------------

        with st.spinner("Calcul de l'optimum..."):
            x_opt, objective_value = _run_optimization(
                function=function,
                lower_bound=lower,
                upper_bound=upper,
                objective_type=objective_type,
                method=method,
                points=int(points),
                tolerance=float(tolerance),
                x0=float(x0),
            )

        # ----------------------------------------------------
        # Résultats
        # ----------------------------------------------------

        st.success("Optimisation terminée avec succès.")

        st.subheader("📊 Résultat")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Type",
                objective_type,
            )

        with result_col2:
            st.metric(
                "x*",
                f"{x_opt:.6f}",
            )

        with result_col3:
            st.metric(
                "f(x*)",
                f"{objective_value:.6f}",
            )

        st.caption(
            f"Méthode utilisée : **{method}**"
        )

        # ----------------------------------------------------
        # Graphique
        # ----------------------------------------------------

        st.subheader("📈 Visualisation")

        _create_plot(
            function,
            lower,
            upper,
            x_opt,
        )

        # ----------------------------------------------------
        # Explication pédagogique
        # ----------------------------------------------------

        st.subheader("📚 Explication")

        if method == "Grid Search":
            st.markdown(explain_grid_search())

        elif method == "Section dorée":
            st.markdown(explain_golden_section())

        elif method == "Newton":
            st.markdown(explain_newton_minimum())

    except (TypeError, ValueError, ZeroDivisionError) as exc:
        st.error(f"⚠️ {exc}")

    except Exception as exc:
        st.error(
            "Une erreur inattendue s'est produite pendant le calcul."
        )
        st.exception(exc)


# ============================================================
# Aide
# ============================================================

with st.expander("💡 Exemples de fonctions"):
    st.markdown(
        """
        **Fonction quadratique :**
        ```text
        x**2 - 4*x + 5
        ```

        **Fonction cubique :**
        ```text
        x**3 - 3*x + 1
        ```

        **Fonction trigonométrique :**
        ```text
        sin(x)
        ```

        **Fonction exponentielle :**
        ```text
        exp(x)
        ```

        **Fonction avec racine :**
        ```text
        sqrt(x + 1)
        ```

        La variable utilisée est toujours **x**.
        """
    )

