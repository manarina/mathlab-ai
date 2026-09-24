from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import sympy as sp


# ============================================================
# GRAPHIQUE D'UNE FONCTION
# ============================================================

def create_function_plot(
    function: sp.Expr,
    point: Any,
    window: float = 5.0,
    samples: int = 1000,
) -> go.Figure:
    """
    Crée un graphique interactif d'une fonction
    autour d'un point donné.

    Parameters
    ----------
    function:
        Expression SymPy de la fonction.

    point:
        Point autour duquel afficher le graphique.

    window:
        Demi-largeur de la fenêtre graphique.

    samples:
        Nombre de points utilisés pour le tracé.

    Returns
    -------
    go.Figure
        Figure Plotly interactive.

    Raises
    ------
    ValueError
        Si le point est infini ou non réel.
    """

    x = sp.Symbol("x")

    point = sp.sympify(point)

    # ========================================================
    # VERIFICATION DU POINT
    # ========================================================

    if point in (
        sp.oo,
        -sp.oo,
    ):

        raise ValueError(
            "Le graphique local nécessite un point réel."
        )

    if point.is_real is False:

        raise ValueError(
            "Le graphique nécessite un point réel."
        )

    # ========================================================
    # CONVERSION NUMERIQUE
    # ========================================================

    try:

        point_float = float(point)

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            "Le point doit être une valeur réelle."
        ) from error

    # ========================================================
    # INTERVALLE
    # ========================================================

    x_min = point_float - window
    x_max = point_float + window

    x_values = np.linspace(
        x_min,
        x_max,
        samples,
    )

    # ========================================================
    # CONVERSION DE LA FONCTION
    # ========================================================

    function_lambda = sp.lambdify(
        x,
        function,
        modules=["numpy"],
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):

        y_values = function_lambda(
            x_values
        )

    # ========================================================
    # CONVERSION NUMPY
    # ========================================================

    y_values = np.asarray(
        y_values,
        dtype=float,
    )

    # ========================================================
    # VALEURS NON FINIES
    # ========================================================

    y_values[
        ~np.isfinite(y_values)
    ] = np.nan

    # ========================================================
    # FIGURE
    # ========================================================

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name="f(x)",
            connectgaps=False,
        )
    )

    # ========================================================
    # POINT ETUDIE
    # ========================================================

    figure.add_vline(
        x=point_float,
        line_dash="dash",
        annotation_text=f"x = {point}",
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    figure.update_layout(
        title="Analyse graphique de la fonction",
        xaxis_title="x",
        yaxis_title="f(x)",
        hovermode="x unified",
    )

    return figure