from __future__ import annotations

import sympy as sp
import streamlit as st

from core.calculus.limits import (
    analyze_limit,
    parse_function,
)

from core.calculus.limit_explanations import (
    explain_limit,
)

from core.calculus.derivatives import (
    analyze_derivative,
    calculate_nth_derivative,
    parse_function as parse_derivative_function,
)

from core.calculus.derivative_explanations import (
    explain_derivative,
)

from core.calculus.plots import (
    create_function_plot,
)

from core.calculus.integral_explanations import (
    calculate_integral,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI — Calculus Lab",
    page_icon="∫",
    layout="wide",
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_calculus_example(
    expression: str,
    point: str,
) -> None:
    """
    Charge un exemple dans les champs du Calculus Lab.

    Cette fonction est appelée par un callback Streamlit,
    avant la reconstruction des widgets.
    """

    st.session_state["calculus_expression"] = expression
    st.session_state["calculus_point"] = point


def load_derivative_example(
    expression: str,
    point: str,
) -> None:
    """
    Charge un exemple dans les champs du module Dérivées.
    """

    st.session_state["derivative_expression"] = expression
    st.session_state["derivative_point"] = point

def load_integral_example(
    expression: str,
    lower_bound: str,
    upper_bound: str,
) -> None:
    """
    Charge un exemple dans les champs du module Intégrales.
    """

    st.session_state["integral_expression"] = expression
    st.session_state["integral_lower_bound"] = lower_bound
    st.session_state["integral_upper_bound"] = upper_bound

# ============================================================
# INITIALISATION SESSION STATE
# ============================================================

if "calculus_expression" not in st.session_state:
    st.session_state["calculus_expression"] = (
        "(x**2 - 1)/(x - 1)"
    )

if "calculus_point" not in st.session_state:
    st.session_state["calculus_point"] = "1"

if "derivative_expression" not in st.session_state:
    st.session_state["derivative_expression"] = "x**2"

if "derivative_point" not in st.session_state:
    st.session_state["derivative_point"] = "2"

if "integral_expression" not in st.session_state:
    st.session_state["integral_expression"] = "x**2"

if "integral_lower_bound" not in st.session_state:
    st.session_state["integral_lower_bound"] = "0"

if "integral_upper_bound" not in st.session_state:
    st.session_state["integral_upper_bound"] = "2"

# ============================================================
# TITRE
# ============================================================
# ============================================================
# TITRE
# ============================================================

st.title("∫ Calculus Lab")

st.markdown(
    """
    Explorez les limites, les dérivées et les intégrales d'une fonction
    avec une résolution mathématique, une explication étape par étape
    et une analyse graphique interactive.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Calculus Lab")

    analysis_type = st.radio(
        "Type d'analyse",
        [
            "Limites",
            "Dérivées",
            "Intégrales",
        ],
    )

    st.divider()

    # --------------------------------------------------------
    # MODULE LIMITES
    # --------------------------------------------------------

    if analysis_type == "Limites":

        st.markdown(
            """
            ### Module actuel

            **Limites**

            - Limite bilatérale
            - Limite à gauche
            - Limite à droite
            - Limite à l'infini
            - Explication pédagogique
            - Analyse graphique
            """
        )

    # --------------------------------------------------------
    # MODULE DÉRIVÉES
    # --------------------------------------------------------

    elif analysis_type == "Dérivées":

        st.markdown(
            """
            ### Module actuel

            **Dérivées**

            - Dérivée première
            - Dérivée seconde
            - Dérivée d'ordre n
            - Valeur de la dérivée en un point
            - Pente de la tangente
            - Équation de la tangente
            - Explication pédagogique
            - Analyse graphique
            """
        )

    # --------------------------------------------------------
    # MODULE INTÉGRALES
    # --------------------------------------------------------

    elif analysis_type == "Intégrales":

        st.markdown(
            """
            ### Module actuel

            **Intégrales**

            - Primitive d'une fonction
            - Intégrale indéfinie
            - Intégrale définie
            - Constante d'intégration
            - Résolution étape par étape
            - Vérification par dérivation
            """
        )


# ============================================================
# LIMITES
# ============================================================

if analysis_type == "Limites":

    st.subheader("Analyse d'une limite")

    # ========================================================
    # SAISIE
    # ========================================================

    col1, col2 = st.columns([2, 1])

    with col1:

        st.text_input(
            "Fonction f(x)",
            key="calculus_expression",
            help=(
                "Utilisez la syntaxe SymPy : "
                "x**2, sin(x), exp(x), sqrt(x), etc."
            ),
        )

    with col2:

        st.text_input(
            "Point x →",
            key="calculus_point",
        )

    direction = st.selectbox(
        "Direction de la limite",
        [
            "Bilatérale",
            "À gauche",
            "À droite",
            "+∞",
            "-∞",
        ],
    )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### Exemples")

    example_columns = st.columns(4)

    examples = [
        (
            "Polynôme",
            "x**2 + 3*x - 2",
            "2",
        ),
        (
            "Rationnelle",
            "(x**2 - 1)/(x - 1)",
            "1",
        ),
        (
            "Exponentielle",
            "exp(x)",
            "0",
        ),
        (
            "Trigonométrique",
            "sin(x)/x",
            "0",
        ),
    ]

    for column, example in zip(
        example_columns,
        examples,
    ):

        name, example_function, example_point = example

        with column:

            st.button(
                name,
                key=f"example_{name}",
                use_container_width=True,
                on_click=load_calculus_example,
                args=(
                    example_function,
                    example_point,
                ),
            )

    st.divider()

    # ========================================================
    # CALCUL
    # ========================================================

    if st.button(
        "Calculer la limite",
        type="primary",
        use_container_width=True,
    ):

        try:

            # =================================================
            # RECUPERATION DES VALEURS
            # =================================================

            expression = st.session_state[
                "calculus_expression"
            ]

            point = st.session_state[
                "calculus_point"
            ]

            # =================================================
            # PARSING
            # =================================================

            function = parse_function(
                expression
            )

            point_value = sp.sympify(
                point
            )

            x = sp.Symbol("x")

            # =================================================
            # LIMITE BILATERALE
            # =================================================

            if direction == "Bilatérale":

                result = analyze_limit(
                    function,
                    point_value,
                )

            # =================================================
            # LIMITE A GAUCHE
            # =================================================

            elif direction == "À gauche":

                left_limit = sp.limit(
                    function,
                    x,
                    point_value,
                    dir="-",
                )

                result = {
                    "function": function,
                    "point": point_value,
                    "left_limit": left_limit,
                    "right_limit": None,
                    "limit": left_limit,
                    "type": "finite_or_infinite",
                }

            # =================================================
            # LIMITE A DROITE
            # =================================================

            elif direction == "À droite":

                right_limit = sp.limit(
                    function,
                    x,
                    point_value,
                    dir="+",
                )

                result = {
                    "function": function,
                    "point": point_value,
                    "left_limit": None,
                    "right_limit": right_limit,
                    "limit": right_limit,
                    "type": "finite_or_infinite",
                }

            # =================================================
            # LIMITE EN +∞
            # =================================================

            elif direction == "+∞":

                infinity_limit = sp.limit(
                    function,
                    x,
                    sp.oo,
                )

                result = {
                    "function": function,
                    "point": sp.oo,
                    "left_limit": None,
                    "right_limit": None,
                    "limit": infinity_limit,
                    "type": "finite_or_infinite",
                }

            # =================================================
            # LIMITE EN -∞
            # =================================================

            else:

                infinity_limit = sp.limit(
                    function,
                    x,
                    -sp.oo,
                )

                result = {
                    "function": function,
                    "point": -sp.oo,
                    "left_limit": None,
                    "right_limit": None,
                    "limit": infinity_limit,
                    "type": "finite_or_infinite",
                }

            # =================================================
            # RESULTAT
            # =================================================

            st.success(
                "Calcul effectué avec succès."
            )

            st.subheader("Résultat")

            result_col1, result_col2 = st.columns(2)

            # =================================================
            # FONCTION
            # =================================================

            with result_col1:

                st.markdown("**Fonction**")

                st.latex(
                    rf"f(x) = {sp.latex(function)}"
                )

            # =================================================
            # RESULTAT DE LA LIMITE
            # =================================================

            with result_col2:

                st.markdown(
                    "**Résultat de la limite**"
                )

                if result["limit"] is None:

                    st.error(
                        "La limite n'existe pas."
                    )

                else:

                    st.latex(
                        rf"\boxed{{{sp.latex(result['limit'])}}}"
                    )

            # =================================================
            # LIMITES LATERALES
            # =================================================

            if direction == "Bilatérale":

                st.divider()

                st.subheader(
                    "Limites latérales"
                )

                left_col, right_col = st.columns(2)

                # ------------------------------------------------
                # GAUCHE
                # ------------------------------------------------

                with left_col:

                    st.markdown(
                        "**Limite à gauche**"
                    )

                    st.latex(
                        rf"""
                        \lim_{{x \to {sp.latex(point_value)}^-}}
                        f(x)
                        =
                        {sp.latex(result["left_limit"])}
                        """
                    )

                # ------------------------------------------------
                # DROITE
                # ------------------------------------------------

                with right_col:

                    st.markdown(
                        "**Limite à droite**"
                    )

                    st.latex(
                        rf"""
                        \lim_{{x \to {sp.latex(point_value)}^+}}
                        f(x)
                        =
                        {sp.latex(result["right_limit"])}
                        """
                    )

                # =================================================
                # CLASSIFICATION
                # =================================================

                if result["type"] == "does_not_exist":

                    st.warning(
                        "Les deux limites latérales sont "
                        "différentes : la limite bilatérale "
                        "n'existe pas."
                    )

                else:

                    st.success(
                        "Les deux limites latérales sont "
                        "égales : la limite bilatérale existe."
                    )

            # =================================================
            # EXPLICATION PEDAGOGIQUE
            # =================================================

            st.divider()

            st.subheader(
                "🧠 Explication étape par étape"
            )

            if direction == "Bilatérale":

                steps = explain_limit(
                    function,
                    point_value,
                    result,
                )

                for index, step in enumerate(
                    steps,
                    start=1,
                ):

                    with st.expander(
                        f"Étape {index} — {step['title']}",
                        expanded=index == 1,
                    ):

                        if step["formula"]:

                            st.latex(
                                step["formula"]
                            )

                        st.write(
                            step["explanation"]
                        )

            else:

                st.info(
                    "L'explication détaillée des limites "
                    "latérales et des limites à l'infini "
                    "sera enrichie dans une prochaine version."
                )

            # =================================================
            # ANALYSE GRAPHIQUE
            # =================================================

            if direction == "Bilatérale":

                st.divider()

                st.subheader(
                    "📈 Analyse graphique"
                )

                st.caption(
                    "Le graphique représente la fonction "
                    "autour du point étudié."
                )

                try:

                    figure = create_function_plot(
                        function,
                        point_value,
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                    )

                except ValueError as error:

                    st.warning(
                        str(error)
                    )

        # =====================================================
        # ERREURS DE SAISIE
        # =====================================================

        except (
            ValueError,
            TypeError,
            sp.SympifyError,
        ) as error:

            st.error(
                f"Expression invalide : {error}"
            )

        # =====================================================
        # AUTRES ERREURS
        # =====================================================

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "le calcul de la limite."
            )

            st.exception(error)


# ============================================================
# DERIVEES
# ============================================================

elif analysis_type == "Dérivées":

    st.subheader("Analyse d'une dérivée")

    # ========================================================
    # SAISIE
    # ========================================================

    col1, col2 = st.columns([2, 1])

    with col1:

        st.text_input(
            "Fonction f(x)",
            key="derivative_expression",
            help=(
                "Utilisez la syntaxe SymPy : "
                "x**2, sin(x), exp(x), sqrt(x), etc."
            ),
        )

    with col2:

        st.text_input(
            "Point x =",
            key="derivative_point",
            help=(
                "Point utilisé pour calculer f(a), "
                "f'(a) et la tangente."
            ),
        )

    derivative_order = st.number_input(
        "Ordre de dérivation",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Permet de calculer la dérivée d'ordre n.",
    )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### Exemples")

    example_columns = st.columns(4)

    derivative_examples = [
        (
            "Polynôme",
            "x**3 + 2*x**2 + x + 1",
            "2",
        ),
        (
            "Trigonométrique",
            "sin(x)",
            "0",
        ),
        (
            "Exponentielle",
            "exp(x)",
            "1",
        ),
        (
            "Rationnelle",
            "1/x",
            "2",
        ),
    ]

    for column, example in zip(
        example_columns,
        derivative_examples,
    ):

        name, example_function, example_point = example

        with column:

            st.button(
                name,
                key=f"derivative_example_{name}",
                use_container_width=True,
                on_click=load_derivative_example,
                args=(
                    example_function,
                    example_point,
                ),
            )

    st.divider()

    # ========================================================
    # CALCUL
    # ========================================================

    if st.button(
        "Calculer la dérivée",
        type="primary",
        use_container_width=True,
    ):

        try:

            # =================================================
            # RECUPERATION DES VALEURS
            # =================================================

            expression = st.session_state[
                "derivative_expression"
            ]

            point = st.session_state[
                "derivative_point"
            ]

            # =================================================
            # PARSING
            # =================================================

            function = parse_derivative_function(
                expression
            )

            point_value = sp.sympify(
                point
            )

            # =================================================
            # ANALYSE
            # =================================================

            result = analyze_derivative(
                function,
                point_value,
            )

            nth_derivative = calculate_nth_derivative(
                function,
                int(derivative_order),
            )

            # =================================================
            # RESULTAT
            # =================================================

            st.success(
                "Calcul effectué avec succès."
            )

            st.subheader("Résultat")

            result_col1, result_col2 = st.columns(2)

            # =================================================
            # FONCTION
            # =================================================

            with result_col1:

                st.markdown("**Fonction**")

                st.latex(
                    rf"f(x) = {sp.latex(function)}"
                )

            # =================================================
            # DERIVEE
            # =================================================

            with result_col2:

                if int(derivative_order) == 1:

                    st.markdown(
                        "**Dérivée première**"
                    )

                    st.latex(
                        rf"\boxed{{f'(x) = "
                        rf"{sp.latex(result['derivative'])}}}"
                    )

                elif int(derivative_order) == 2:

                    st.markdown(
                        "**Dérivée seconde**"
                    )

                    st.latex(
                        rf"\boxed{{f''(x) = "
                        rf"{sp.latex(nth_derivative)}}}"
                    )

                else:

                    st.markdown(
                        f"**Dérivée d'ordre {int(derivative_order)}**"
                    )

                    st.latex(
                        rf"\boxed{{f^{{({int(derivative_order)})}}(x) = "
                        rf"{sp.latex(nth_derivative)}}}"
                    )

            # =================================================
            # DERIVEES PRINCIPALES
            # =================================================

            st.divider()

            st.subheader(
                "Dérivées principales"
            )

            derivative_col1, derivative_col2 = st.columns(2)

            with derivative_col1:

                st.markdown(
                    "**Dérivée première**"
                )

                st.latex(
                    rf"f'(x) = "
                    rf"{sp.latex(result['derivative'])}"
                )

            with derivative_col2:

                st.markdown(
                    "**Dérivée seconde**"
                )

                st.latex(
                    rf"f''(x) = "
                    rf"{sp.latex(result['second_derivative'])}"
                )

            # =================================================
            # ANALYSE AU POINT
            # =================================================

            st.divider()

            st.subheader(
                f"Analyse au point x = {sp.latex(point_value)}"
            )

            point_col1, point_col2, point_col3 = st.columns(3)

            with point_col1:

                st.metric(
                    "f(a)",
                    str(
                        result["function_value"]
                    ),
                )

            with point_col2:

                st.metric(
                    "f'(a)",
                    str(
                        result["derivative_value"]
                    ),
                )

            with point_col3:

                st.metric(
                    "Pente de la tangente",
                    str(
                        result["tangent_slope"]
                    ),
                )

            st.markdown(
                "**Équation de la tangente**"
            )

            st.latex(
                rf"y = {sp.latex(result['tangent_line'])}"
            )

            # =================================================
            # EXPLICATION PEDAGOGIQUE
            # =================================================

            st.divider()

            st.subheader(
                "🧠 Explication étape par étape"
            )

            steps = explain_derivative(
                function,
                result,
            )

            for index, step in enumerate(
                steps,
                start=1,
            ):

                with st.expander(
                    f"Étape {index} — {step['title']}",
                    expanded=index == 1,
                ):

                    if step["formula"]:

                        st.latex(
                            step["formula"]
                        )

                    st.write(
                        step["explanation"]
                    )

            # =================================================
            # DERIVEE D'ORDRE N
            # =================================================

            if int(derivative_order) not in (1, 2):

                st.divider()

                st.subheader(
                    f"Dérivée d'ordre {int(derivative_order)}"
                )

                st.latex(
                    rf"f^{{({int(derivative_order)})}}(x) = "
                    rf"{sp.latex(nth_derivative)}"
                )

            # =================================================
            # ANALYSE GRAPHIQUE
            # =================================================

            st.divider()

            st.subheader(
                "📈 Analyse graphique"
            )

            st.caption(
                "Le graphique représente la fonction "
                "autour du point étudié."
            )

            try:

                figure = create_function_plot(
                    function,
                    point_value,
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

            except ValueError as error:

                st.warning(
                    str(error)
                )

        # =====================================================
        # ERREURS DE SAISIE
        # =====================================================

        except (
            ValueError,
            TypeError,
            sp.SympifyError,
        ) as error:

            st.error(
                f"Expression invalide : {error}"
            )

        # =====================================================
        # AUTRES ERREURS
        # =====================================================

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "le calcul de la dérivée."
            )

            st.exception(error)

# ============================================================
# INTEGRALES
# ============================================================

elif analysis_type == "Intégrales":

    st.subheader("Analyse d'une intégrale")

    # ========================================================
    # SAISIE
    # ========================================================

    st.markdown("### Fonction à intégrer")

    st.text_input(
        "Fonction f(x)",
        key="integral_expression",
        help=(
            "Utilisez la syntaxe SymPy : "
            "x**2, 2*x + 1, sin(x), exp(x), etc."
        ),
    )

    integral_type = st.radio(
        "Type d'intégrale",
        [
            "Indéfinie",
            "Définie",
        ],
        horizontal=True,
    )

    lower_bound = None
    upper_bound = None

    if integral_type == "Définie":

        col1, col2 = st.columns(2)

        with col1:

            st.text_input(
                "Borne inférieure",
                key="integral_lower_bound",
            )

        with col2:

            st.text_input(
                "Borne supérieure",
                key="integral_upper_bound",
            )

    st.divider()

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### Exemples")

    example_columns = st.columns(4)

    integral_examples = [
        (
            "Polynôme",
            "x**2",
            "0",
            "2",
        ),
        (
            "Linéaire",
            "2*x + 1",
            "0",
            "3",
        ),
        (
            "Exponentielle",
            "exp(x)",
            "0",
            "1",
        ),
        (
            "Trigonométrique",
            "sin(x)",
            "0",
            "pi",
        ),
    ]

    for column, example in zip(
        example_columns,
        integral_examples,
    ):

        name, expression, lower, upper = example

        with column:

            st.button(
                name,
                key=f"integral_example_{name}",
                use_container_width=True,
                on_click=load_integral_example,
                args=(
                    expression,
                    lower,
                    upper,
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

            # =================================================
            # RECUPERATION DES VALEURS
            # =================================================

            expression = st.session_state[
                "integral_expression"
            ]

            if integral_type == "Définie":

                lower_bound = st.session_state[
                    "integral_lower_bound"
                ]

                upper_bound = st.session_state[
                    "integral_upper_bound"
                ]

            # =================================================
            # CALCUL
            # =================================================

            result = calculate_integral(
                expression=expression,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

            # =================================================
            # RESULTAT
            # =================================================

            st.success(
                "Calcul effectué avec succès."
            )

            st.subheader("Résultat")

            result_col1, result_col2 = st.columns(2)

            # =================================================
            # FONCTION
            # =================================================

            with result_col1:

                st.markdown("**Fonction**")

                st.latex(
                    rf"f(x) = {sp.latex(result.expression)}"
                )

            # =================================================
            # INTEGRALE
            # =================================================

            with result_col2:

                st.markdown(
                    "**Résultat de l'intégrale**"
                )

                st.latex(
                    rf"""
                    \boxed{{
                        {sp.latex(result.integral)}
                    }}
                    """
                )

            # =================================================
            # EXPRESSION COMPLETE
            # =================================================

            st.divider()

            st.subheader(
                "Expression mathématique"
            )

            if result.is_definite:

                st.latex(
                    rf"""
                    \int_{{{sp.latex(result.lower_bound)}}}^
                    {{{sp.latex(result.upper_bound)}}}
                    {sp.latex(result.expression)}
                    \,dx
                    =
                    {sp.latex(result.integral)}
                    """
                )

            else:

                st.latex(
                    rf"""
                    \int {sp.latex(result.expression)}
                    \,dx
                    =
                    {sp.latex(result.integral)}
                    """
                )

            # =================================================
            # EXPLICATION PEDAGOGIQUE
            # =================================================

            st.divider()

            st.subheader(
                "🧠 Explication étape par étape"
            )

            for index, step in enumerate(
                result.steps,
                start=1,
            ):

                with st.expander(
                    f"Étape {index}",
                    expanded=index == 1,
                ):

                    st.write(step)

            # =================================================
            # VERIFICATION
            # =================================================

            st.divider()

            st.subheader(
                "✅ Vérification"
            )

            if result.verification == 0:

                st.success(
                    "La primitive est correcte : "
                    "sa dérivée redonne la fonction "
                    "à intégrer."
                )

            else:

                st.warning(
                    "La vérification symbolique "
                    "n'a pas permis de confirmer "
                    "directement le résultat."
                )

            # =================================================
            # INTERPRETATION
            # =================================================

            if result.is_definite:

                st.divider()

                st.subheader(
                    "📐 Interprétation"
                )

                st.info(
                    "La valeur d'une intégrale définie "
                    "représente l'aire algébrique sous "
                    "la courbe entre les deux bornes."
                )

        # =====================================================
        # ERREURS DE SAISIE
        # =====================================================

        except (
            ValueError,
            TypeError,
            sp.SympifyError,
        ) as error:

            st.error(
                f"Expression invalide : {error}"
            )

        # =====================================================
        # AUTRES ERREURS
        # =====================================================

        except Exception as error:

            st.error(
                "Une erreur est survenue pendant "
                "le calcul de l'intégrale."
            )

            st.exception(error)