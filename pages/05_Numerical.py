
from __future__ import annotations

import math

import streamlit as st
import sympy as sp

from core.numerical.approximation import (
    absolute_errors,
    coefficient_of_determination,
    evaluate_polynomial,
    polynomial_approximation,
    predicted_values,
    relative_errors,
    root_mean_square_error,
)

from core.numerical.integration import (
    rectangle_rule,
    trapezoidal_rule,
    simpson_one_third,
)

from core.numerical.integration_explanations import (
    explain_rectangle_rule,
    explain_trapezoidal_rule,
    explain_simpson_one_third,
    explain_integration_error,
    explain_method_comparison,
    explain_number_of_subdivisions,
    explain_exact_vs_numerical,
    explain_simpson_requirement,
)

from core.numerical.interpolation import (
    divided_differences,
    evaluate_interpolating_polynomial,
    lagrange_interpolation,
    newton_interpolation,
)
from core.numerical.linear_systems import gaussian_elimination
from core.numerical.root_finding import (
    bisection,
    newton_raphson,
    secant,
)
from core.numerical.root_finding_explanations import (
    explain_bisection,
    explain_newton_raphson,
    explain_secant,
    explain_root_finding_comparison,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI — Numerical Lab",
    page_icon="∫",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    # Recherche de racines
    "numerical_function": "x**2 - 2",
    "numerical_a": 1.0,
    "numerical_b": 2.0,
    "numerical_initial_guess": 1.5,
    "numerical_second_guess": 2.0,
    "numerical_tolerance": 1e-10,
    "numerical_max_iterations": 100,
    "numerical_matrix_size": 2,

    # Interpolation
    "interpolation_x_values": "0, 1, 2",
    "interpolation_y_values": "1, 3, 7",
    "interpolation_evaluation_point": 1.5,

    # Approximation et régression
    "approximation_x_values": "0, 1, 2, 3, 4",
    "approximation_y_values": "1.1, 2.9, 5.2, 6.8, 9.1",
    "approximation_degree": 2,

    
    # Intégration numérique
    "integration_function": "x**2",
    "integration_a": 0.0,
    "integration_b": 1.0,
    "integration_subdivisions": 10,



}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================


def set_numerical_function(expression: str) -> None:
    """
    Met à jour uniquement la fonction sélectionnée.

    Utilisation via callback Streamlit afin d'éviter
    StreamlitWidgetAlreadyInstantiatedError.
    """
    st.session_state["numerical_function"] = expression


def set_numerical_example(
    expression: str,
    a: float,
    b: float,
    initial_guess: float,
    second_guess: float,
) -> None:
    """
    Configure complètement un exemple numérique.
    """
    st.session_state["numerical_function"] = expression
    st.session_state["numerical_a"] = a
    st.session_state["numerical_b"] = b
    st.session_state["numerical_initial_guess"] = initial_guess
    st.session_state["numerical_second_guess"] = second_guess


def parse_function(expression: str):
    """
    Convertit une expression textuelle en fonction numérique.
    """
    expression = expression.strip()

    if not expression:
        raise ValueError(
            "La fonction ne peut pas être vide."
        )

    allowed_names = {
        "x": None,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }

    def function(x: float) -> float:
        names = dict(allowed_names)
        names["x"] = x

        try:
            value = eval(
                expression,
                {"__builtins__": {}},
                names,
            )
        except Exception as error:
            raise ValueError(
                f"Impossible d'évaluer f(x) = {expression}."
            ) from error

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                "La fonction doit retourner une valeur numérique."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                "La fonction retourne une valeur non finie."
            )

        return float(value)

    return function


def parse_derivative(expression: str):
    """
    Construit la dérivée des expressions pédagogiques disponibles.
    """
    derivative_map = {
        "x**2 - 2": lambda x: 2 * x,
        "x**3 - 2": lambda x: 3 * x**2,
        "x**3 - x - 2": lambda x: 3 * x**2 - 1,
        "x**2 - 4": lambda x: 2 * x,
        "sin(x)": lambda x: math.cos(x),
        "cos(x)": lambda x: -math.sin(x),
        "exp(x) - 2": lambda x: math.exp(x),
    }

    expression = expression.strip()

    if expression in derivative_map:
        return derivative_map[expression]

    raise ValueError(
        "La dérivée automatique n'est pas disponible pour "
        "cette expression. Utilisez l'un des exemples proposés."
    )


def format_number(value) -> str:
    """
    Formatage lisible des résultats numériques.
    """
    if value is None:
        return "—"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(number):
        return str(number)

    return f"{number:.12g}"


def show_root_result(result, method_name: str) -> None:
    """
    Affiche les informations principales d'un résultat numérique.
    """
    st.subheader(f"Résultat — {method_name}")

    root = getattr(result, "root", None)
    iterations = getattr(result, "iterations", None)
    error = getattr(result, "error", None)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Racine approchée",
            format_number(root),
        )

    with col2:
        st.metric(
            "Itérations",
            str(iterations) if iterations is not None else "—",
        )

    with col3:
        st.metric(
            "Erreur",
            format_number(error),
        )

    st.success("Calcul numérique terminé.")


def display_explanation(explanation: str) -> None:
    """
    Affiche une explication produite par
    root_finding_explanations.py.
    """
    if not isinstance(explanation, str):
        raise TypeError(
            "L'explication doit être une chaîne de caractères."
        )

    st.markdown(explanation)


def validate_bisection_interval(
    function,
    a: float,
    b: float,
) -> None:
    """
    Vérifie que l'intervalle est compatible avec la dichotomie.
    """
    if a >= b:
        raise ValueError(
            "La borne inférieure a doit être strictement "
            "inférieure à b."
        )

    fa = function(a)
    fb = function(b)

    if fa == 0:
        return

    if fb == 0:
        return

    if fa * fb > 0:
        raise ValueError(
            "La fonction doit changer de signe sur l'intervalle. "
            f"f({format_number(a)}) = {format_number(fa)}, "
            f"f({format_number(b)}) = {format_number(fb)}."
        )


# ============================================================
# UTILITAIRES — INTERPOLATION
# ============================================================


def parse_interpolation_values(
    values: str,
    name: str,
) -> list[float]:
    """
    Convertit une liste de nombres saisie par l'utilisateur
    en liste de flottants.
    """
    if not isinstance(values, str):
        raise TypeError(
            f"Les valeurs de {name} doivent être saisies sous forme de texte."
        )

    cleaned = values.replace(";", ",").strip()

    if not cleaned:
        raise ValueError(
            f"La liste {name} ne peut pas être vide."
        )

    parts = [
        part.strip()
        for part in cleaned.split(",")
        if part.strip()
    ]

    if len(parts) == 1 and " " in parts[0]:
        parts = [
            part.strip()
            for part in parts[0].split()
            if part.strip()
        ]

    try:
        result = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(
            f"Les valeurs de {name} doivent être numériques."
        ) from error

    if not result:
        raise ValueError(
            f"La liste {name} ne peut pas être vide."
        )

    if not all(math.isfinite(value) for value in result):
        raise ValueError(
            f"Les valeurs de {name} doivent être finies."
        )

    return result


def set_interpolation_example(
    x_values: str,
    y_values: str,
    evaluation_point: float,
) -> None:
    """
    Configure un exemple d'interpolation.
    """
    st.session_state["interpolation_x_values"] = x_values
    st.session_state["interpolation_y_values"] = y_values
    st.session_state["interpolation_evaluation_point"] = (
        evaluation_point
    )


def display_interpolation_polynomial(
    polynomial: sp.Expr,
) -> None:
    """
    Affiche le polynôme interpolateur sous forme lisible.
    """
    st.markdown("### Polynôme interpolateur")

    st.latex(
        sp.latex(polynomial)
    )

    with st.expander("Voir l'expression SymPy"):
        st.code(
            str(polynomial),
            language="text",
        )


def display_interpolation_plot(
    x_values: list[float],
    y_values: list[float],
    polynomial: sp.Expr,
) -> None:
    """
    Affiche les points d'interpolation et la courbe
    du polynôme interpolateur.
    """
    import numpy as np
    import plotly.graph_objects as go

    x_min = min(x_values)
    x_max = max(x_values)

    span = x_max - x_min

    if span == 0:
        span = 1.0

    plot_min = x_min - 0.1 * span
    plot_max = x_max + 0.1 * span

    x_curve = np.linspace(
        plot_min,
        plot_max,
        400,
    )

    y_curve = []

    for value in x_curve:
        evaluated = polynomial.subs(
            sp.Symbol("x"),
            float(value),
        )

        y_curve.append(
            float(sp.N(evaluated))
        )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x_curve,
            y=y_curve,
            mode="lines",
            name="Polynôme interpolateur",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            name="Points d'interpolation",
            marker={
                "size": 10,
            },
        )
    )

    figure.update_layout(
        title="Visualisation de l'interpolation",
        xaxis_title="x",
        yaxis_title="P(x)",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


# ============================================================
# UTILITAIRES — APPROXIMATION ET RÉGRESSION
# ============================================================


def parse_approximation_values(
    values: str,
    name: str,
) -> list[float]:
    """
    Convertit une série de valeurs saisie dans l'interface
    en liste de nombres flottants.

    Séparateurs acceptés :
    - virgule
    - point-virgule
    - espace
    """
    if not isinstance(values, str):
        raise TypeError(
            f"Les valeurs de {name} doivent être saisies sous forme de texte."
        )

    cleaned = values.replace(";", ",").strip()

    if not cleaned:
        raise ValueError(
            f"La liste {name} ne peut pas être vide."
        )

    parts = [
        part.strip()
        for part in cleaned.split(",")
        if part.strip()
    ]

    if len(parts) == 1 and " " in parts[0]:
        parts = [
            part.strip()
            for part in parts[0].split()
            if part.strip()
        ]

    try:
        result = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(
            f"Les valeurs de {name} doivent être numériques."
        ) from error

    if not result:
        raise ValueError(
            f"La liste {name} ne peut pas être vide."
        )

    if not all(math.isfinite(value) for value in result):
        raise ValueError(
            f"Les valeurs de {name} doivent être finies."
        )

    return result


def set_approximation_example(
    x_values: str,
    y_values: str,
    degree: int,
) -> None:
    """
    Configure un exemple d'approximation.
    """
    st.session_state["approximation_x_values"] = x_values
    st.session_state["approximation_y_values"] = y_values
    st.session_state["approximation_degree"] = degree


def display_approximation_polynomial(
    coefficients,
) -> None:
    """
    Transforme les coefficients numériques en polynôme SymPy
    et l'affiche sous forme lisible.

    Les coefficients du module approximation sont stockés
    dans l'ordre des puissances croissantes :
        [a0, a1, a2, ..., an]
    """
    x = sp.Symbol("x")

    polynomial = sum(
        sp.Float(coefficient) * x**power
        for power, coefficient in enumerate(coefficients)
    )

    polynomial = sp.expand(polynomial)

    st.markdown("### Modèle polynomial")

    st.latex(
        rf"\hat{{y}} = {sp.latex(polynomial)}"
    )

    with st.expander("Voir l'expression du modèle"):
        st.code(
            str(polynomial),
            language="text",
        )

    return polynomial


def display_approximation_plot(
    x_values,
    y_values,
    coefficients,
) -> None:
    """
    Compare graphiquement les données observées et
    le modèle polynomial approximatif.
    """
    import numpy as np
    import plotly.graph_objects as go

    x_array = np.asarray(
        x_values,
        dtype=float,
    )

    y_array = np.asarray(
        y_values,
        dtype=float,
    )

    x_min = float(np.min(x_array))
    x_max = float(np.max(x_array))

    span = x_max - x_min

    if span == 0:
        span = 1.0

    plot_min = x_min - 0.1 * span
    plot_max = x_max + 0.1 * span

    x_curve = np.linspace(
        plot_min,
        plot_max,
        400,
    )

    y_curve = evaluate_polynomial(
        coefficients,
        x_curve,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x_array,
            y=y_array,
            mode="markers",
            name="Données observées",
            marker={
                "size": 10,
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x_curve,
            y=y_curve,
            mode="lines",
            name="Modèle polynomial",
        )
    )

    figure.update_layout(
        title="Données observées et modèle polynomial",
        xaxis_title="x",
        yaxis_title="y",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def set_integration_example(
    expression: str,
    a: float,
    b: float,
    subdivisions: int,
) -> None:
    """
    Configure un exemple d'intégration numérique.
    """
    st.session_state["integration_function"] = expression
    st.session_state["integration_a"] = a
    st.session_state["integration_b"] = b
    st.session_state["integration_subdivisions"] = subdivisions



# ============================================================
# TITRE
# ============================================================

st.title("∫ Numerical Lab")

st.markdown(
    """
    Explorez les méthodes numériques fondamentales pour résoudre
    des équations, des systèmes et des problèmes d'approximation.

    **Méthodes disponibles :**

    - Dichotomie
    - Newton-Raphson
    - Sécante
    - Élimination de Gauss
    - Interpolation de Lagrange
    - Interpolation de Newton
    - Approximation et régression polynomiale
    - Comparaison des méthodes
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Numerical Lab")

    analysis_type = st.radio(
        "Type d'analyse",
        [
            "Recherche de racines",
            "Systèmes linéaires",
            "Interpolation numérique",
            "Approximation et régression numérique",
            "Intégration numérique",
            "Comparaison",
        ],
    )

    st.divider()

    if analysis_type == "Recherche de racines":

        st.markdown(
            """
            ### Recherche de racines

            Méthodes disponibles :

            - Dichotomie
            - Newton-Raphson
            - Sécante
            """
        )

    elif analysis_type == "Systèmes linéaires":

        st.markdown(
            """
            ### Systèmes linéaires

            Résolution par :

            **Élimination de Gauss**

            avec pivotage partiel.
            """
        )

    elif analysis_type == "Interpolation numérique":

        st.markdown(
            """
            ### Interpolation numérique

            Méthodes disponibles :

            - Lagrange
            - Newton
            - Différences divisées
            - Évaluation du polynôme
            - Visualisation
            """
        )

    elif analysis_type == "Approximation et régression numérique":

        st.markdown(
            """
            ### Approximation et régression

            Analysez des données numériques avec :

            - Approximation polynomiale
            - Moindres carrés
            - Erreurs d'approximation
            - RMSE
            - Coefficient R²
            - Visualisation
            """
        )

        



# ============================================================
# RECHERCHE DE RACINES
# ============================================================

if analysis_type == "Recherche de racines":

    st.subheader(
        "Recherche numérique d'une racine"
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        expression = st.text_input(
            "Fonction f(x)",
            key="numerical_function",
            help=(
                "Exemples : x**2 - 2, x**3 - 2, "
                "sin(x), exp(x) - 2"
            ),
        )

    with col2:

        method = st.selectbox(
            "Méthode",
            [
                "Dichotomie",
                "Newton-Raphson",
                "Sécante",
            ],
        )

    st.divider()

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.markdown("### Paramètres numériques")

    if method == "Dichotomie":

        col1, col2 = st.columns(2)

        with col1:

            a = st.number_input(
                "Borne inférieure a",
                key="numerical_a",
                format="%.10f",
            )

        with col2:

            b = st.number_input(
                "Borne supérieure b",
                key="numerical_b",
                format="%.10f",
            )

    elif method == "Newton-Raphson":

        col1, col2 = st.columns(2)

        with col1:

            initial_guess = st.number_input(
                "Approximation initiale x₀",
                key="numerical_initial_guess",
                format="%.10f",
            )

        with col2:

            st.info(
                "La dérivée est construite à partir "
                "des exemples disponibles."
            )

    else:

        col1, col2 = st.columns(2)

        with col1:

            first_guess = st.number_input(
                "Première approximation x₀",
                key="numerical_initial_guess",
                format="%.10f",
            )

        with col2:

            second_guess = st.number_input(
                "Deuxième approximation x₁",
                key="numerical_second_guess",
                format="%.10f",
            )

    col1, col2 = st.columns(2)

    with col1:

        tolerance = st.number_input(
            "Tolérance",
            min_value=1e-14,
            max_value=1.0,
            value=float(
                st.session_state["numerical_tolerance"]
            ),
            format="%.1e",
            key="numerical_tolerance",
        )

    with col2:

        max_iterations = st.number_input(
            "Nombre maximal d'itérations",
            min_value=1,
            max_value=10000,
            value=int(
                st.session_state["numerical_max_iterations"]
            ),
            step=1,
            key="numerical_max_iterations",
        )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### Exemples")

    example_columns = st.columns(4)

    examples = [
        (
            "√2",
            "x**2 - 2",
            1.0,
            2.0,
            1.5,
            2.0,
        ),
        (
            "Racine cubique",
            "x**3 - 2",
            1.0,
            2.0,
            1.5,
            2.0,
        ),
        (
            "Polynôme",
            "x**3 - x - 2",
            1.0,
            2.0,
            1.5,
            2.0,
        ),
        (
            "Trigonométrique",
            "sin(x)",
            -1.0,
            1.0,
            1.0,
            2.0,
        ),
    ]

    example_keys = [
        "sqrt2",
        "cubic_root",
        "polynomial",
        "trigonometric",
    ]

    for column, example_data, key_name in zip(
        example_columns,
        examples,
        example_keys,
    ):

        (
            name,
            example,
            example_a,
            example_b,
            example_initial,
            example_second,
        ) = example_data

        with column:

            st.button(
                name,
                key=f"numerical_example_{key_name}",
                use_container_width=True,
                on_click=set_numerical_example,
                args=(
                    example,
                    example_a,
                    example_b,
                    example_initial,
                    example_second,
                ),
            )

    st.divider()

    # ========================================================
    # CALCUL
    # ========================================================

    if st.button(
        "Calculer la racine",
        type="primary",
        use_container_width=True,
    ):

        try:

            function = parse_function(expression)

            # ==================================================
            # DICHOTOMIE
            # ==================================================

            if method == "Dichotomie":

                validate_bisection_interval(
                    function,
                    a,
                    b,
                )

                result = bisection(
                    function,
                    a,
                    b,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )

                show_root_result(
                    result,
                    "Dichotomie",
                )

                st.divider()

                st.subheader(
                    "🧠 Explication étape par étape"
                )

                explanation = explain_bisection(
                    function,
                    a,
                    b,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    result=result,
                )

                display_explanation(
                    explanation
                )

            # ==================================================
            # NEWTON-RAPHSON
            # ==================================================

            elif method == "Newton-Raphson":

                derivative = parse_derivative(
                    expression
                )

                result = newton_raphson(
                    function,
                    derivative,
                    initial_guess,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )

                show_root_result(
                    result,
                    "Newton-Raphson",
                )

                st.divider()

                st.subheader(
                    "🧠 Explication étape par étape"
                )

                explanation = explain_newton_raphson(
                    function,
                    derivative,
                    initial_guess,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    result=result,
                )

                display_explanation(
                    explanation
                )

            # ==================================================
            # SÉCANTE
            # ==================================================

            else:

                result = secant(
                    function,
                    first_guess,
                    second_guess,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )

                show_root_result(
                    result,
                    "Sécante",
                )

                st.divider()

                st.subheader(
                    "🧠 Explication étape par étape"
                )

                explanation = explain_secant(
                    function,
                    first_guess,
                    second_guess,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )

                display_explanation(
                    explanation
                )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur de calcul : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "la recherche de la racine."
            )

            st.exception(error)


# ============================================================
# SYSTÈMES LINÉAIRES
# ============================================================

elif analysis_type == "Systèmes linéaires":

    st.subheader(
        "Résolution d'un système linéaire"
    )

    st.markdown(
        """
        Résolution du système :

        **A X = B**

        par **élimination de Gauss avec pivotage partiel**.
        """
    )

    matrix_size = st.selectbox(
        "Dimension du système",
        [2, 3],
        index=0,
        key="linear_matrix_size",
    )

    st.divider()

    # ========================================================
    # MATRICE A
    # ========================================================

    st.markdown("### Matrice A")

    matrix = []

    for i in range(matrix_size):

        columns = st.columns(matrix_size)

        row = []

        for j, column in enumerate(columns):

            with column:

                value = st.number_input(
                    f"a{i + 1}{j + 1}",
                    value=(
                        1.0
                        if i == j
                        else 0.0
                    ),
                    key=f"matrix_{matrix_size}_{i}_{j}",
                    format="%.6f",
                )

                row.append(value)

        matrix.append(row)

    st.divider()

    # ========================================================
    # VECTEUR B
    # ========================================================

    st.markdown("### Vecteur B")

    vector = []

    columns = st.columns(matrix_size)

    for i, column in enumerate(columns):

        with column:

            value = st.number_input(
                f"b{i + 1}",
                value=1.0,
                key=f"vector_{matrix_size}_{i}",
                format="%.6f",
            )

            vector.append(value)

    st.divider()

    # ========================================================
    # EXEMPLE CLASSIQUE
    # ========================================================

    st.markdown("### Exemple classique")

    st.code(
        """
2x + y = 5
x - y = 1
        """.strip()
    )

    st.caption(
        "Solution : x = 2, y = 1"
    )

    st.divider()

    # ========================================================
    # CALCUL GAUSS
    # ========================================================

    if st.button(
        "Résoudre le système",
        type="primary",
        use_container_width=True,
    ):

        try:

            result = gaussian_elimination(
                matrix,
                vector,
            )

            st.success(
                "Système résolu avec succès."
            )

            # ==================================================
            # SOLUTION
            # ==================================================

            st.subheader("Solution")

            solution = result.solution

            solution_columns = st.columns(
                len(solution)
            )

            for index, column in enumerate(
                solution_columns
            ):

                with column:

                    st.metric(
                        f"x{index + 1}",
                        format_number(
                            solution[index]
                        ),
                    )

            st.divider()

            # ==================================================
            # MATRICE TRIANGULAIRE
            # ==================================================

            st.subheader(
                "Matrice triangulaire"
            )

            st.dataframe(
                result.matrix,
                use_container_width=True,
            )

            # ==================================================
            # VECTEUR TRANSFORMÉ
            # ==================================================

            st.subheader(
                "Vecteur transformé"
            )

            st.dataframe(
                result.vector,
                use_container_width=True,
            )

            # ==================================================
            # ÉCHANGES DE LIGNES
            # ==================================================

            swaps = getattr(
                result,
                "swaps",
                None,
            )

            if swaps is not None:

                st.info(
                    f"Échanges de lignes : {swaps}"
                )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "la résolution du système."
            )

            st.exception(error)


# ============================================================
# INTERPOLATION NUMÉRIQUE
# ============================================================

elif analysis_type == "Interpolation numérique":

    st.subheader(
        "Interpolation numérique"
    )

    st.markdown(
        """
        Construisez un **polynôme interpolateur** à partir d'un
        ensemble de points numériques.

        Deux méthodes sont disponibles :

        - **Interpolation de Lagrange**
        - **Interpolation de Newton**
        """
    )

    st.divider()

    # ========================================================
    # DONNÉES
    # ========================================================

    st.markdown("### 📊 Points d'interpolation")

    col1, col2 = st.columns(2)

    with col1:

        x_values_text = st.text_input(
            "Valeurs de x",
            key="interpolation_x_values",
            help=(
                "Exemple : 0, 1, 2"
            ),
        )

    with col2:

        y_values_text = st.text_input(
            "Valeurs de y",
            key="interpolation_y_values",
            help=(
                "Exemple : 1, 3, 7"
            ),
        )

    st.caption(
        "Les valeurs peuvent être séparées par des virgules, "
        "des points-virgules ou des espaces."
    )

    st.divider()

    # ========================================================
    # MÉTHODE ET ÉVALUATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        interpolation_method = st.selectbox(
            "Méthode d'interpolation",
            [
                "Lagrange",
                "Newton",
            ],
            key="interpolation_method",
        )

    with col2:

        evaluation_point = st.number_input(
            "Point à évaluer",
            key="interpolation_evaluation_point",
            format="%.10f",
        )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### 💡 Exemples")

    example_columns = st.columns(4)

    interpolation_examples = [
        (
            "Polynôme x²+x+1",
            "0, 1, 2",
            "1, 3, 7",
            1.5,
        ),
        (
            "Polynôme x²",
            "-1, 0, 1, 2",
            "1, 0, 1, 4",
            1.5,
        ),
        (
            "Fonction linéaire",
            "0, 1, 2",
            "2, 5, 8",
            3.0,
        ),
        (
            "Fonction cubique",
            "0, 1, 2, 3",
            "1, 2, 9, 28",
            1.5,
        ),
    ]

    interpolation_example_keys = [
        "quadratic",
        "square",
        "linear",
        "cubic",
    ]

    for column, example_data, key_name in zip(
        example_columns,
        interpolation_examples,
        interpolation_example_keys,
    ):

        (
            name,
            example_x,
            example_y,
            example_point,
        ) = example_data

        with column:

            st.button(
                name,
                key=f"interpolation_example_{key_name}",
                use_container_width=True,
                on_click=set_interpolation_example,
                args=(
                    example_x,
                    example_y,
                    example_point,
                ),
            )

    st.divider()

    # ========================================================
    # CALCUL
    # ========================================================

    if st.button(
        "Construire le polynôme interpolateur",
        type="primary",
        use_container_width=True,
    ):

        try:

            x_values = parse_interpolation_values(
                x_values_text,
                "x",
            )

            y_values = parse_interpolation_values(
                y_values_text,
                "y",
            )

            if len(x_values) != len(y_values):
                raise ValueError(
                    "Les listes x et y doivent avoir "
                    "la même longueur."
                )

            if len(x_values) < 2:
                raise ValueError(
                    "L'interpolation nécessite au moins "
                    "deux points."
                )

            if interpolation_method == "Lagrange":

                polynomial = lagrange_interpolation(
                    x_values,
                    y_values,
                )

            else:

                polynomial = newton_interpolation(
                    x_values,
                    y_values,
                )

            st.success(
                "Polynôme interpolateur construit avec succès."
            )

            display_interpolation_polynomial(
                polynomial
            )

            st.divider()

            st.subheader(
                "📍 Évaluation du polynôme"
            )

            evaluated_value = (
                evaluate_interpolating_polynomial(
                    polynomial,
                    evaluation_point,
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "x",
                    format_number(
                        evaluation_point
                    ),
                )

            with col2:

                st.metric(
                    "P(x)",
                    format_number(
                        evaluated_value
                    ),
                )

            st.latex(
                rf"P({evaluation_point:g}) = "
                rf"{sp.latex(evaluated_value)}"
            )

            st.divider()

            st.subheader(
                "📐 Différences divisées"
            )

            coefficients = divided_differences(
                x_values,
                y_values,
            )

            coefficient_data = {
                "Ordre": list(
                    range(len(coefficients))
                ),
                "Coefficient": [
                    str(coefficient)
                    for coefficient in coefficients
                ],
            }

            st.dataframe(
                coefficient_data,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Les coefficients correspondent aux termes "
                "successifs du polynôme de Newton."
            )

            st.divider()

            st.subheader(
                "📋 Données d'interpolation"
            )

            points_data = {
                "x": x_values,
                "y": y_values,
            }

            st.dataframe(
                points_data,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader(
                "📈 Visualisation"
            )

            display_interpolation_plot(
                x_values,
                y_values,
                polynomial,
            )

            st.caption(
                "Les points représentent les données originales. "
                "La courbe représente le polynôme interpolateur."
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur d'interpolation : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "la construction du polynôme interpolateur."
            )

            st.exception(error)


# ============================================================
# APPROXIMATION ET RÉGRESSION NUMÉRIQUE
# ============================================================

elif analysis_type == "Approximation et régression numérique":

    st.subheader(
        "Approximation et régression numérique"
    )

    st.markdown(
        """
        Construisez un **modèle polynomial approximatif** à partir
        de données numériques.

        Contrairement à l'interpolation, le modèle n'est pas
        nécessairement obligé de passer exactement par tous les
        points. L'objectif est de trouver une approximation qui
        représente au mieux les données.
        """
    )

    st.divider()

    # ========================================================
    # DONNÉES
    # ========================================================

    st.markdown("### 📊 Données expérimentales")

    col1, col2 = st.columns(2)

    with col1:

        x_values_text = st.text_input(
            "Valeurs de x",
            key="approximation_x_values",
            help=(
                "Exemple : 0, 1, 2, 3, 4"
            ),
        )

    with col2:

        y_values_text = st.text_input(
            "Valeurs de y",
            key="approximation_y_values",
            help=(
                "Exemple : 1.1, 2.9, 5.2, 6.8, 9.1"
            ),
        )

    st.caption(
        "Les valeurs peuvent être séparées par des virgules, "
        "des points-virgules ou des espaces."
    )

    st.divider()

    # ========================================================
    # DEGRÉ DU MODÈLE
    # ========================================================

    st.markdown("### 📐 Modèle polynomial")

    degree = st.number_input(
        "Degré du polynôme",
        min_value=0,
        max_value=10,
        value=int(
            st.session_state["approximation_degree"]
        ),
        step=1,
        key="approximation_degree",
        help=(
            "Le degré doit être strictement inférieur "
            "au nombre de points disponibles."
        ),
    )

    st.caption(
        "Exemple : degré 1 → modèle linéaire ; "
        "degré 2 → modèle quadratique ; "
        "degré 3 → modèle cubique."
    )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### 💡 Exemples")

    example_columns = st.columns(4)

    approximation_examples = [
        (
            "Quadratique",
            "0, 1, 2, 3, 4",
            "1, 3, 7, 13, 21",
            2,
        ),
        (
            "Linéaire",
            "0, 1, 2, 3, 4",
            "1.1, 2.9, 5.2, 6.8, 9.1",
            1,
        ),
        (
            "Bruit expérimental",
            "0, 1, 2, 3, 4, 5",
            "1.2, 2.7, 5.1, 8.9, 12.4, 15.8",
            2,
        ),
        (
            "Cubique",
            "0, 1, 2, 3, 4",
            "1, 2, 9, 28, 65",
            3,
        ),
    ]

    approximation_example_keys = [
        "quadratic",
        "linear",
        "experimental",
        "cubic",
    ]

    for column, example_data, key_name in zip(
        example_columns,
        approximation_examples,
        approximation_example_keys,
    ):

        (
            name,
            example_x,
            example_y,
            example_degree,
        ) = example_data

        with column:

            st.button(
                name,
                key=f"approximation_example_{key_name}",
                use_container_width=True,
                on_click=set_approximation_example,
                args=(
                    example_x,
                    example_y,
                    example_degree,
                ),
            )

    st.divider()

    # ========================================================
    # CALCUL
    # ========================================================

    if st.button(
        "Calculer l'approximation",
        type="primary",
        use_container_width=True,
    ):

        try:

            # ------------------------------------------------
            # Conversion
            # ------------------------------------------------

            x_values = parse_approximation_values(
                x_values_text,
                "x",
            )

            y_values = parse_approximation_values(
                y_values_text,
                "y",
            )

            # ------------------------------------------------
            # Validation interface
            # ------------------------------------------------

            if len(x_values) != len(y_values):
                raise ValueError(
                    "Les listes x et y doivent avoir "
                    "la même longueur."
                )

            if len(x_values) < 2:
                raise ValueError(
                    "L'approximation nécessite au moins "
                    "deux points."
                )

            if degree >= len(x_values):
                raise ValueError(
                    "Le degré du polynôme doit être strictement "
                    "inférieur au nombre de points."
                )

            # ------------------------------------------------
            # Approximation polynomiale
            # ------------------------------------------------

            coefficients = polynomial_approximation(
                x_values,
                y_values,
                degree,
            )

            st.success(
                "Approximation polynomiale calculée avec succès."
            )

            # ------------------------------------------------
            # Modèle polynomial
            # ------------------------------------------------

            polynomial = display_approximation_polynomial(
                coefficients
            )

            st.divider()

            # ------------------------------------------------
            # Valeurs prédites
            # ------------------------------------------------

            st.subheader(
                "📊 Valeurs prédites"
            )

            predicted = predicted_values(
                coefficients,
                x_values,
            )

            absolute_error = absolute_errors(
                y_values,
                predicted,
            )

            relative_error = relative_errors(
                y_values,
                predicted,
            )

            data = {
                "x": x_values,
                "y observé": y_values,
                "ŷ prédit": predicted,
                "Erreur absolue": absolute_error,
                "Erreur relative": relative_error,
            }

            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # ------------------------------------------------
            # Indicateurs d'erreur
            # ------------------------------------------------

            st.subheader(
                "📏 Analyse de l'erreur"
            )

            rmse = root_mean_square_error(
                y_values,
                predicted,
            )

            r_squared = coefficient_of_determination(
                y_values,
                predicted,
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "RMSE",
                    format_number(rmse),
                )

            with col2:

                st.metric(
                    "R²",
                    format_number(r_squared),
                )

            with col3:

                st.metric(
                    "Degré",
                    str(degree),
                )

            st.markdown(
                """
                **Interprétation :**

                - **Erreur absolue** :
                  différence entre la valeur observée et la
                  valeur prédite.
                - **Erreur relative** :
                  erreur absolue rapportée à la valeur observée.
                - **RMSE** :
                  mesure globale de l'écart entre les observations
                  et les prédictions.
                - **R²** :
                  mesure de la part de la variabilité des données
                  expliquée par le modèle dans le cadre de cette
                  régression.
                """
            )

            st.divider()

            # ------------------------------------------------
            # Visualisation
            # ------------------------------------------------

            st.subheader(
                "📈 Comparaison données / modèle"
            )

            display_approximation_plot(
                x_values,
                y_values,
                coefficients,
            )

            st.caption(
                "Les points représentent les données observées. "
                "La courbe représente le modèle polynomial approximatif."
            )

            st.divider()

            # ------------------------------------------------
            # Informations mathématiques
            # ------------------------------------------------

            st.subheader(
                "🧠 Comprendre le modèle"
            )

            st.markdown(
                f"""
                Le modèle polynomial obtenu est de degré **{degree}**.

                Les coefficients sont calculés par une méthode
                d'approximation polynomiale basée sur les **moindres
                carrés**.

                Le modèle cherche les coefficients :

                $$
                \\hat{{y}} =
                a_0 + a_1x + a_2x^2 + \\cdots + a_nx^n
                $$

                qui minimisent l'écart global entre les valeurs
                observées $y_i$ et les valeurs prédites
                $\\hat{{y}}_i$.

                L'erreur quadratique totale est basée sur :

                $$
                \\sum_i (y_i - \\hat{{y}}_i)^2
                $$

                **Attention :** une approximation polynomiale ne
                signifie pas nécessairement que la courbe passe
                exactement par tous les points.
                """
            )

            with st.expander(
                "Voir les coefficients numériques"
            ):

                coefficient_data = {
                    "Puissance": list(
                        range(len(coefficients))
                    ),
                    "Coefficient": [
                        format_number(coefficient)
                        for coefficient in coefficients
                    ],
                }

                st.dataframe(
                    coefficient_data,
                    use_container_width=True,
                    hide_index=True,
                )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur d'approximation : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "le calcul de l'approximation."
            )

            st.exception(error)

# ============================================================
# INTÉGRATION NUMÉRIQUE
# ============================================================

elif analysis_type == "Intégration numérique":

    st.subheader(
        "Intégration numérique"
    )

    st.markdown(
        """
        Approximez une intégrale définie

        $$

        I = \\int_a^b f(x)\\,dx

        $$

        à l'aide de différentes méthodes numériques.

        Les méthodes disponibles sont :

        - **Rectangles**
        - **Trapèzes**
        - **Simpson 1/3**
        """
    )

    st.divider()

    # ========================================================
    # FONCTION ET INTERVALLE
    # ========================================================

    st.markdown("### 📐 Fonction et intervalle")

    col1, col2 = st.columns(2)

    with col1:

        integration_function = st.text_input(
            "Fonction f(x)",
            key="integration_function",
            help=(
                "Exemples : x**2, sin(x), exp(x), "
                "x**3 + 2*x"
            ),
        )

    with col2:

        integration_method = st.selectbox(
            "Méthode d'intégration",
            [
                "Rectangles",
                "Trapèzes",
                "Simpson 1/3",
            ],
            key="integration_method",
        )

    col1, col2 = st.columns(2)

    with col1:

        integration_a = st.number_input(
            "Borne inférieure a",
            key="integration_a",
            format="%.10f",
        )

    with col2:

        integration_b = st.number_input(
            "Borne supérieure b",
            key="integration_b",
            format="%.10f",
        )

    # ========================================================
    # NOMBRE DE SUBDIVISIONS
    # ========================================================

    integration_subdivisions = st.number_input(
        "Nombre de subdivisions n",
        min_value=1,
        max_value=100000,
        value=int(
            st.session_state[
                "integration_subdivisions"
            ]
        ),
        step=1,
        key="integration_subdivisions",
        help=(
            "Un nombre plus élevé de subdivisions "
            "améliore généralement la précision."
        ),
    )

    if integration_method == "Simpson 1/3":

        if integration_subdivisions % 2 != 0:

            st.warning(
                "La méthode de Simpson 1/3 nécessite "
                "un nombre pair de subdivisions."
            )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### 💡 Exemples")

    example_columns = st.columns(4)

    integration_examples = [
        (
            "∫ x²",
            "x**2",
            0.0,
            1.0,
            10,
        ),
        (
            "∫ sin(x)",
            "sin(x)",
            0.0,
            math.pi,
            100,
        ),
        (
            "∫ x³",
            "x**3",
            0.0,
            1.0,
            10,
        ),
        (
            "∫ exp(x)",
            "exp(x)",
            0.0,
            1.0,
            100,
        ),
    ]

    integration_example_keys = [
        "square",
        "sine",
        "cubic",
        "exponential",
    ]

    for column, example_data, key_name in zip(
        example_columns,
        integration_examples,
        integration_example_keys,
    ):

        (
            name,
            example_function,
            example_a,
            example_b,
            example_n,
        ) = example_data

        with column:

            st.button(
                name,
                key=f"integration_example_{key_name}",
                use_container_width=True,
                on_click=set_integration_example,
                args=(
                    example_function,
                    example_a,
                    example_b,
                    example_n,
                ),
            )

    st.divider()

    # ========================================================
    # CALCUL
    # ========================================================

    if st.button(
        "Calculer l'intégrale",
        type="primary",
        use_container_width=True,
    ):

        try:

            # ------------------------------------------------
            # Fonction numérique
            # ------------------------------------------------

            function = parse_function(
                integration_function
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if integration_a >= integration_b:

                raise ValueError(
                    "La borne inférieure a doit être "
                    "strictement inférieure à b."
                )

            if integration_subdivisions < 1:

                raise ValueError(
                    "Le nombre de subdivisions doit être "
                    "au moins égal à 1."
                )

            if (
                integration_method == "Simpson 1/3"
                and integration_subdivisions % 2 != 0
            ):

                raise ValueError(
                    "La méthode de Simpson 1/3 nécessite "
                    "un nombre pair de subdivisions."
                )

            # ------------------------------------------------
            # Calcul
            # ------------------------------------------------

            if integration_method == "Rectangles":

                approximate_value = rectangle_rule(
                    function,
                    integration_a,
                    integration_b,
                    integration_subdivisions,
                )

                explanation = explain_rectangle_rule()

            elif integration_method == "Trapèzes":

                approximate_value = trapezoidal_rule(
                    function,
                    integration_a,
                    integration_b,
                    integration_subdivisions,
                )

                explanation = explain_trapezoidal_rule()

            else:

                approximate_value = simpson_one_third(
                    function,
                    integration_a,
                    integration_b,
                    integration_subdivisions,
                )

                explanation = explain_simpson_one_third()

            # ------------------------------------------------
            # Résultat
            # ------------------------------------------------

            st.success(
                "Intégration numérique effectuée avec succès."
            )

            st.subheader(
                "📊 Résultat numérique"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Valeur approchée",
                    format_number(
                        approximate_value
                    ),
                )

            with col2:

                st.metric(
                    "Subdivisions",
                    str(
                        integration_subdivisions
                    ),
                )

            with col3:

                st.metric(
                    "Méthode",
                    integration_method,
                )

            st.latex(
                rf"I \approx "
                rf"{format_number(approximate_value)}"
            )

            st.divider()

            # ------------------------------------------------
            # Explication pédagogique
            # ------------------------------------------------

            st.subheader(
                "🧠 Comprendre la méthode"
            )

            display_explanation(
                explanation
            )

            st.divider()

            # ------------------------------------------------
            # Analyse du nombre de subdivisions
            # ------------------------------------------------

            st.subheader(
                "🔢 Influence du nombre de subdivisions"
            )

            display_explanation(
                explain_number_of_subdivisions()
            )

            st.divider()

            # ------------------------------------------------
            # Comparaison avec les autres méthodes
            # ------------------------------------------------

            st.subheader(
                "📈 Comparaison des méthodes"
            )

            comparison_results = {}

            # Rectangles

            comparison_results["Rectangles"] = (
                rectangle_rule(
                    function,
                    integration_a,
                    integration_b,
                    integration_subdivisions,
                )
            )

            # Trapèzes

            comparison_results["Trapèzes"] = (
                trapezoidal_rule(
                    function,
                    integration_a,
                    integration_b,
                    integration_subdivisions,
                )
            )

            # Simpson

            if integration_subdivisions % 2 == 0:

                comparison_results["Simpson 1/3"] = (
                    simpson_one_third(
                        function,
                        integration_a,
                        integration_b,
                        integration_subdivisions,
                    )
                )

            else:

                comparison_results[
                    "Simpson 1/3"
                ] = None

            comparison_data = {
                "Méthode": list(
                    comparison_results.keys()
                ),
                "Valeur approchée": [
                    (
                        format_number(value)
                        if value is not None
                        else "Non disponible"
                    )
                    for value in comparison_results.values()
                ],
            }

            st.dataframe(
                comparison_data,
                use_container_width=True,
                hide_index=True,
            )

            display_explanation(
                explain_method_comparison()
            )

            st.divider()

            # ------------------------------------------------
            # Valeur exacte optionnelle
            # ------------------------------------------------

            st.subheader(
                "🎯 Comparaison avec une valeur exacte"
            )

            exact_value_text = st.text_input(
                "Valeur exacte de l'intégrale "
                "(optionnelle)",
                value="",
                key="integration_exact_value",
                help=(
                    "Exemple : pour ∫₀¹ x² dx, "
                    "la valeur exacte est 1/3."
                ),
            )

            if exact_value_text.strip():

                try:

                    exact_value = float(
                        sp.N(
                            sp.sympify(
                                exact_value_text
                            )
                        )
                    )

                    error = abs(
                        exact_value
                        - approximate_value
                    )

                    relative_error_value = (
                        error / abs(exact_value)
                        if exact_value != 0
                        else math.nan
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Valeur exacte",
                            format_number(
                                exact_value
                            ),
                        )

                    with col2:

                        st.metric(
                            "Erreur absolue",
                            format_number(
                                error
                            ),
                        )

                    with col3:

                        st.metric(
                            "Erreur relative",
                            (
                                format_number(
                                    relative_error_value
                                )
                                if math.isfinite(
                                    relative_error_value
                                )
                                else "—"
                            ),
                        )

                    st.latex(
                        rf"E = |I - I_n| "
                        rf"\approx "
                        rf"{format_number(error)}"
                    )

                    display_explanation(
                        explain_integration_error(
                            exact_value,
                            approximate_value,
                        )
                    )

                    display_explanation(
                        explain_exact_vs_numerical()
                    )

                except (
                    ValueError,
                    TypeError,
                    sp.SympifyError,
                ) as error:

                    st.error(
                        "La valeur exacte doit être "
                        "une expression numérique valide."
                    )

                    st.caption(
                        f"Détail : {error}"
                    )

            st.divider()

            # ------------------------------------------------
            # Rappel Simpson
            # ------------------------------------------------

            if integration_method == "Simpson 1/3":

                st.subheader(
                    "📚 Condition de Simpson 1/3"
                )

                display_explanation(
                    explain_simpson_requirement()
                )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur d'intégration : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "le calcul de l'intégrale."
            )

            st.exception(error)





# ============================================================
# COMPARAISON DES MÉTHODES
# ============================================================

else:

    st.subheader(
        "Comparaison des méthodes de recherche de racines"
    )

    st.markdown(
        """
        Cette section applique plusieurs méthodes à la même
        équation afin de comparer leurs résultats numériques.
        """
    )

    expression = st.text_input(
        "Fonction f(x)",
        value="x**2 - 2",
        key="comparison_function",
    )

    col1, col2 = st.columns(2)

    with col1:

        a = st.number_input(
            "Borne / première approximation",
            value=1.0,
            format="%.10f",
            key="comparison_a",
        )

    with col2:

        b = st.number_input(
            "Borne / deuxième approximation",
            value=2.0,
            format="%.10f",
            key="comparison_b",
        )

    col1, col2 = st.columns(2)

    with col1:

        tolerance = st.number_input(
            "Tolérance",
            min_value=1e-14,
            max_value=1.0,
            value=1e-10,
            format="%.1e",
            key="comparison_tolerance",
        )

    with col2:

        max_iterations = st.number_input(
            "Nombre maximal d'itérations",
            min_value=1,
            max_value=10000,
            value=100,
            step=1,
            key="comparison_max_iterations",
        )

    st.divider()

    # ========================================================
    # CALCUL COMPARATIF
    # ========================================================

    if st.button(
        "Comparer les méthodes",
        type="primary",
        use_container_width=True,
    ):

        try:

            function = parse_function(
                expression
            )

            derivative = parse_derivative(
                expression
            )

            # ==================================================
            # VALIDATION DICHOTOMIE
            # ==================================================

            validate_bisection_interval(
                function,
                a,
                b,
            )

            # ==================================================
            # DICHOTOMIE
            # ==================================================

            bisection_result = bisection(
                function,
                a,
                b,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

            # ==================================================
            # NEWTON-RAPHSON
            # ==================================================

            newton_result = newton_raphson(
                function,
                derivative,
                b,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

            # ==================================================
            # SÉCANTE
            # ==================================================

            secant_result = secant(
                function,
                a,
                b,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

            # ==================================================
            # TABLEAU DES RÉSULTATS
            # ==================================================

            st.success(
                "Comparaison effectuée avec succès."
            )

            st.subheader(
                "Résultats numériques"
            )

            results = {
                "Méthode": [
                    "Dichotomie",
                    "Newton-Raphson",
                    "Sécante",
                ],
                "Racine": [
                    format_number(
                        bisection_result.root
                    ),
                    format_number(
                        newton_result.root
                    ),
                    format_number(
                        secant_result.root
                    ),
                ],
                "Itérations": [
                    bisection_result.iterations,
                    newton_result.iterations,
                    secant_result.iterations,
                ],
                "Erreur": [
                    format_number(
                        bisection_result.error
                    ),
                    format_number(
                        newton_result.error
                    ),
                    format_number(
                        secant_result.error
                    ),
                ],
            }

            st.table(results)

            # ==================================================
            # EXPLICATION COMPARATIVE
            # ==================================================

            st.divider()

            st.subheader(
                "🧠 Analyse comparative"
            )

            comparison_explanation = (
                explain_root_finding_comparison(
                    function=function,
                    derivative=derivative,
                    lower=a,
                    upper=b,
                    initial_guess=b,
                    first_guess=a,
                    second_guess=b,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            )

            display_explanation(
                comparison_explanation
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            st.error(
                f"Erreur : {error}"
            )

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "la comparaison."
            )

            st.exception(error)

