from __future__ import annotations

import re

import streamlit as st

from core.statistics.descriptive import (
    data_range,
    maximum,
    mean,
    median,
    minimum,
    mode,
    percentile,
    quartiles,
    standard_deviation,
    variance,
)

from core.statistics.descriptive_explanations import (
    explain_data_range,
    explain_maximum,
    explain_mean,
    explain_median,
    explain_minimum,
    explain_mode,
    explain_percentile,
    explain_quartiles,
    explain_standard_deviation,
    explain_variance,
)

# ============================================================
# DISTRIBUTIONS
# ============================================================

from core.statistics.distributions import (
    normal_cdf,
    normal_pdf,
    normal_probability_between,
    normal_quantile,
    uniform_cdf,
    uniform_pdf,
    uniform_probability_between,
)

from core.statistics.distribution_explanations import (
    explain_normal_cdf,
    explain_normal_pdf,
    explain_normal_probability_between,
    explain_normal_quantile,
    explain_uniform_cdf,
    explain_uniform_pdf,
    explain_uniform_probability_between,
)

# ============================================================
# CORRÉLATION
# ============================================================

from core.statistics.correlation import (
    covariance,
    pearson_correlation,
)

from core.statistics.correlation_explanations import (
    explain_covariance,
    explain_pearson_correlation,
)

# ============================================================
# RÉGRESSION LINÉAIRE — NOUVEAU MODULE
# ============================================================

from core.statistics.regression import (
    coefficient_of_determination,
    linear_regression,
    predict,
    regression_predictions,
    regression_slope,
    regression_intercept,
)

from core.statistics.regression_explanations import (
    explain_regression_slope,
    explain_regression_intercept,
    explain_linear_regression,
    explain_prediction,
    explain_r_squared,
    explain_regression,
    interpret_r_squared,
)

# ============================================================
# VISUALISATION DE LA RÉGRESSION
# ============================================================

from core.statistics.regression_visualization import (
    regression_plot_data,
)

# ============================================================
# KHI-DEUX (χ²)
# ============================================================

from core.statistics.chi_square import (
    chi_square_critical_value,
    chi_square_p_value,
    chi_square_statistic,
    chi_square_test,
    degrees_of_freedom,
    expected_frequencies,
)

from core.statistics.chi_square_explanations import (
    explain_chi_square,
    explain_chi_square_critical_value,
    explain_chi_square_decision,
    explain_chi_square_hypotheses,
    explain_chi_square_p_value,
    explain_chi_square_statistic,
    explain_chi_square_test,
    explain_degrees_of_freedom,
    explain_expected_frequencies,
    interpret_chi_square,
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI — Statistics Lab",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# OUTILS
# ============================================================


def parse_data(data_text: str) -> list[float]:
    """
    Transforme une saisie utilisateur en liste de nombres.

    Formats acceptés :

        1, 2, 3, 4
        1; 2; 3; 4
        1 2 3 4

    ou une valeur par ligne :

        1
        2
        3
        4

    Les nombres décimaux utilisent le point :
        1.5, 2.5, 3.5
    """

    text = data_text.strip()

    if not text:
        raise ValueError(
            "Veuillez saisir au moins une valeur."
        )

    normalized = (
        text
        .replace(";", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    raw_values = re.split(r"[,\s]+", normalized)

    values: list[float] = []

    for raw_value in raw_values:
        value_text = raw_value.strip()

        if not value_text:
            continue

        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(
                f"Valeur invalide : « {value_text} »."
            ) from exc

        values.append(value)

    if not values:
        raise ValueError(
            "Aucune valeur numérique valide n'a été trouvée."
        )

    return values


def format_number(value: float | int) -> str:
    """
    Formate proprement un nombre pour l'affichage.
    """

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return f"{value:.6g}"


def format_mode(value: float | int | tuple) -> str:
    """
    Formate le résultat du mode.
    """

    if isinstance(value, tuple):
        return ", ".join(
            format_number(item)
            for item in value
        )

    return format_number(value)


def display_error(message: str) -> None:
    """
    Affiche une erreur utilisateur de manière cohérente.
    """

    st.error(f"❌ {message}")


def display_probability(value: float) -> None:
    """
    Affiche une probabilité sous forme décimale et en pourcentage.
    """

    st.metric(
        "Probabilité",
        f"{value:.6f}",
    )

    st.caption(
        f"Soit **{value * 100:.2f} %**"
    )


# ============================================================
# TITRE
# ============================================================

st.title("📊 MathLab AI — Statistics Lab")

st.markdown(
    """
    Analysez une **série statistique** et explorez les
    principales lois de probabilité à l'aide des outils
    de MathLab AI.

    Choisissez une section, effectuez vos calculs et
    consultez ensuite les explications mathématiques.
    """
)

st.divider()


# ============================================================
# MENU
# ============================================================

section = st.selectbox(
    "📚 Choisissez une section",
    [
        "Statistiques descriptives",
        "Distributions",
        "Corrélation",
        "Régression linéaire",
        "Khi-deux (χ²)",
    ],
)


# ============================================================
# STATISTIQUES DESCRIPTIVES
# ============================================================

if section == "Statistiques descriptives":

    st.header("📋 Statistiques descriptives")

    st.markdown(
        """
        Les statistiques descriptives permettent de résumer
        une série de données à l'aide de mesures de
        **tendance centrale**, de **dispersion** et de
        **position**.
        """
    )

    # ========================================================
    # SAISIE DES DONNÉES
    # ========================================================

    st.subheader("📝 Saisie des données")

    data_text = st.text_area(
        "Entrez votre série de données",
        value="10, 12, 14, 16, 18",
        height=120,
        key="statistics_data",
        help=(
            "Séparez les valeurs par une virgule, un point-"
            "virgule, un espace ou placez une valeur par ligne."
        ),
    )

    st.caption(
        "Exemples : `10, 12, 14, 16, 18`  •  "
        "`10; 12; 14; 16; 18`  •  "
        "`10 12 14 16 18`"
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        percentile_value = st.slider(
            "📈 Percentile à calculer",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="statistics_percentile",
        )

    with col2:

        st.metric(
            "Percentile sélectionné",
            f"P{percentile_value}",
        )

    calculate_button = st.button(
        "📊 Calculer les statistiques",
        type="primary",
        use_container_width=True,
        key="calculate_descriptive_statistics",
    )

    # ========================================================
    # CALCUL
    # ========================================================

    if calculate_button:

        try:

            values = parse_data(data_text)

            average = mean(values)
            median_value = median(values)
            mode_value = mode(values)

            minimum_value = minimum(values)
            maximum_value = maximum(values)
            range_value = data_range(values)

            variance_value = variance(values)
            standard_deviation_value = standard_deviation(values)

            q1, q2, q3 = quartiles(values)

            percentile_result = percentile(
                values,
                percentile_value,
            )

            sorted_values = sorted(values)

            # ==================================================
            # EN-TÊTE DES RÉSULTATS
            # ==================================================

            st.divider()

            st.subheader("📊 Résultats")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Nombre de valeurs",
                    len(values),
                )

            with col2:
                st.metric(
                    "Minimum",
                    format_number(minimum_value),
                )

            with col3:
                st.metric(
                    "Maximum",
                    format_number(maximum_value),
                )

            with col4:
                st.metric(
                    "Étendue",
                    format_number(range_value),
                )

            # ==================================================
            # SERIE
            # ==================================================

            with st.expander(
                "📋 Voir la série de données",
                expanded=True,
            ):

                st.write(
                    "**Série saisie :**"
                )

                st.code(
                    ", ".join(
                        format_number(value)
                        for value in values
                    ),
                    language="text",
                )

                st.write(
                    "**Série ordonnée :**"
                )

                st.code(
                    ", ".join(
                        format_number(value)
                        for value in sorted_values
                    ),
                    language="text",
                )

            # ==================================================
            # ONGLETS DE RESULTATS
            # ==================================================

            tab_central, tab_dispersion, tab_position = st.tabs(
                [
                    "📊 Tendance centrale",
                    "📐 Dispersion",
                    "📦 Position",
                ]
            )

            # ==================================================
            # TENDANCE CENTRALE
            # ==================================================

            with tab_central:

                st.subheader(
                    "📊 Mesures de tendance centrale"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Moyenne",
                        format_number(average),
                    )

                    st.caption(
                        "Somme des valeurs ÷ nombre de valeurs"
                    )

                with col2:

                    st.metric(
                        "Médiane",
                        format_number(median_value),
                    )

                    st.caption(
                        "Valeur centrale de la série ordonnée"
                    )

                with col3:

                    st.metric(
                        "Mode",
                        format_mode(mode_value),
                    )

                    st.caption(
                        "Valeur(s) la/les plus fréquente(s)"
                    )

            # ==================================================
            # DISPERSION
            # ==================================================

            with tab_dispersion:

                st.subheader(
                    "📐 Mesures de dispersion"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Variance",
                        format_number(variance_value),
                    )

                    st.caption(
                        "Mesure de la dispersion autour de la moyenne"
                    )

                with col2:

                    st.metric(
                        "Écart-type",
                        format_number(
                            standard_deviation_value
                        ),
                    )

                    st.caption(
                        "Racine carrée de la variance"
                    )

                st.divider()

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Minimum",
                        format_number(minimum_value),
                    )

                with col2:

                    st.metric(
                        "Maximum",
                        format_number(maximum_value),
                    )

                with col3:

                    st.metric(
                        "Étendue",
                        format_number(range_value),
                    )

            # ==================================================
            # POSITION
            # ==================================================

            with tab_position:

                st.subheader(
                    "📦 Mesures de position"
                )

                st.markdown(
                    "Les quartiles divisent la série ordonnée "
                    "en quatre parties."
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Q1 — 25 %",
                        format_number(q1),
                    )

                with col2:

                    st.metric(
                        "Q2 — 50 %",
                        format_number(q2),
                    )

                with col3:

                    st.metric(
                        "Q3 — 75 %",
                        format_number(q3),
                    )

                st.divider()

                st.subheader(
                    f"📈 Percentile P{percentile_value}"
                )

                st.metric(
                    f"P{percentile_value}",
                    format_number(percentile_result),
                )

            # ==================================================
            # EXPLICATIONS
            # ==================================================

            st.divider()

            st.subheader(
                "💡 Comprendre la démarche"
            )

            explanation_choice = st.selectbox(
                "Choisissez une notion à comprendre",
                [
                    "Synthèse",
                    "Moyenne",
                    "Médiane",
                    "Mode",
                    "Minimum",
                    "Maximum",
                    "Étendue",
                    "Variance",
                    "Écart-type",
                    "Quartiles",
                    "Percentile",
                ],
                key="statistics_explanation_choice",
            )

            if explanation_choice == "Synthèse":

                st.info(
                    """
                    Les statistiques descriptives permettent de
                    résumer une série de données.

                    **Tendance centrale**
                    - Moyenne
                    - Médiane
                    - Mode

                    **Dispersion**
                    - Minimum
                    - Maximum
                    - Étendue
                    - Variance
                    - Écart-type

                    **Position**
                    - Quartiles
                    - Percentiles
                    """
                )

            elif explanation_choice == "Moyenne":

                st.markdown(
                    explain_mean(values)
                )

            elif explanation_choice == "Médiane":

                st.markdown(
                    explain_median(values)
                )

            elif explanation_choice == "Mode":

                st.markdown(
                    explain_mode(values)
                )

            elif explanation_choice == "Minimum":

                st.markdown(
                    explain_minimum(values)
                )

            elif explanation_choice == "Maximum":

                st.markdown(
                    explain_maximum(values)
                )

            elif explanation_choice == "Étendue":

                st.markdown(
                    explain_data_range(values)
                )

            elif explanation_choice == "Variance":

                st.markdown(
                    explain_variance(values)
                )

            elif explanation_choice == "Écart-type":

                st.markdown(
                    explain_standard_deviation(values)
                )

            elif explanation_choice == "Quartiles":

                st.markdown(
                    explain_quartiles(values)
                )

            elif explanation_choice == "Percentile":

                st.markdown(
                    explain_percentile(
                        values,
                        percentile_value,
                    )
                )

        except (ValueError, TypeError) as exc:

            display_error(str(exc))


# ============================================================
# DISTRIBUTIONS
# ============================================================

elif section == "Distributions":

    st.header("📈 Distributions de probabilité")

    st.markdown(
        """
        Explorez les principales fonctions des lois de
        probabilité continues.

        Cette section permet actuellement d'étudier :

        - **Loi normale**
        - **Loi uniforme**
        """
    )

    st.divider()

    distribution_type = st.radio(
        "📊 Choisissez une distribution",
        [
            "Loi normale",
            "Loi uniforme",
        ],
        horizontal=True,
        key="statistics_distribution_type",
    )

    # ========================================================
    # LOI NORMALE
    # ========================================================

    if distribution_type == "Loi normale":

        st.subheader("🔔 Loi normale")

        st.markdown(
            """
            La loi normale est définie par deux paramètres :

            - **μ** : moyenne
            - **σ** : écart-type, avec σ > 0
            """
        )

        operation = st.selectbox(
            "🔧 Choisissez une opération",
            [
                "Densité PDF",
                "Fonction de répartition CDF",
                "Quantile",
                "Probabilité entre deux valeurs",
            ],
            key="normal_operation",
        )

        st.divider()

        # ====================================================
        # PDF NORMALE
        # ====================================================

        if operation == "Densité PDF":

            st.markdown(
                "### 📐 Densité de probabilité"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                mean_value = st.number_input(
                    "Moyenne μ",
                    value=0.0,
                    key="normal_pdf_mean",
                )

            with col2:

                standard_deviation = st.number_input(
                    "Écart-type σ",
                    min_value=0.0001,
                    value=1.0,
                    step=0.1,
                    key="normal_pdf_std",
                )

            with col3:

                x = st.number_input(
                    "Valeur x",
                    value=0.0,
                    key="normal_pdf_x",
                )

            calculate_normal_pdf = st.button(
                "📈 Calculer la densité",
                type="primary",
                use_container_width=True,
                key="calculate_normal_pdf",
            )

            if calculate_normal_pdf:

                try:

                    result = normal_pdf(
                        x,
                        mean_value,
                        standard_deviation,
                    )

                    st.divider()

                    st.subheader("📊 Résultat")

                    st.metric(
                        "Densité f(x)",
                        f"{result:.6f}",
                    )

                    st.caption(
                        f"Pour x = {format_number(x)}, "
                        f"μ = {format_number(mean_value)}, "
                        f"σ = {format_number(standard_deviation)}"
                    )

                    st.divider()

                    st.subheader(
                        "💡 Comprendre la démarche"
                    )

                    st.markdown(
                        explain_normal_pdf(
                            x,
                            mean_value,
                            standard_deviation,
                            result,
                        )
                    )

                except (ValueError, TypeError) as exc:

                    display_error(str(exc))

        # ====================================================
        # CDF NORMALE
        # ====================================================

        elif operation == "Fonction de répartition CDF":

            st.markdown(
                "### 📊 Fonction de répartition"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                mean_value = st.number_input(
                    "Moyenne μ",
                    value=0.0,
                    key="normal_cdf_mean",
                )

            with col2:

                standard_deviation = st.number_input(
                    "Écart-type σ",
                    min_value=0.0001,
                    value=1.0,
                    step=0.1,
                    key="normal_cdf_std",
                )

            with col3:

                x = st.number_input(
                    "Valeur x",
                    value=0.0,
                    key="normal_cdf_x",
                )

            calculate_normal_cdf = st.button(
                "📈 Calculer la probabilité",
                type="primary",
                use_container_width=True,
                key="calculate_normal_cdf",
            )

            if calculate_normal_cdf:

                try:

                    result = normal_cdf(
                        x,
                        mean_value,
                        standard_deviation,
                    )

                    st.divider()

                    st.subheader("📊 Résultat")

                    display_probability(result)

                    st.caption(
                        f"Interprétation : "
                        f"P(X ≤ {format_number(x)})"
                    )

                    st.divider()

                    st.subheader(
                        "💡 Comprendre la démarche"
                    )

                    st.markdown(
                        explain_normal_cdf(
                            x,
                            mean_value,
                            standard_deviation,
                            result,
                        )
                    )

                except (ValueError, TypeError) as exc:

                    display_error(str(exc))

        # ====================================================
        # QUANTILE NORMAL
        # ====================================================

        elif operation == "Quantile":

            st.markdown(
                "### 📦 Quantile de la loi normale"
            )

            probability = st.slider(
                "Probabilité p",
                min_value=0.01,
                max_value=0.99,
                value=0.50,
                step=0.01,
                key="normal_quantile_probability",
            )

            col1, col2 = st.columns(2)

            with col1:

                mean_value = st.number_input(
                    "Moyenne μ",
                    value=0.0,
                    key="normal_quantile_mean",
                )

            with col2:

                standard_deviation = st.number_input(
                    "Écart-type σ",
                    min_value=0.0001,
                    value=1.0,
                    step=0.1,
                    key="normal_quantile_std",
                )

            calculate_normal_quantile = st.button(
                "📈 Calculer le quantile",
                type="primary",
                use_container_width=True,
                key="calculate_normal_quantile",
            )

            if calculate_normal_quantile:

                try:

                    result = normal_quantile(
                        probability,
                        mean_value,
                        standard_deviation,
                    )

                    st.divider()

                    st.subheader("📊 Résultat")

                    st.metric(
                        "Quantile",
                        format_number(result),
                    )

                    st.caption(
                        f"P(X ≤ x) = {probability:.2f}"
                    )

                    st.divider()

                    st.subheader(
                        "💡 Comprendre la démarche"
                    )

                    st.markdown(
                        explain_normal_quantile(
                            probability,
                            mean_value,
                            standard_deviation,
                            result,
                        )
                    )

                except (ValueError, TypeError) as exc:

                    display_error(str(exc))

        # ====================================================
        # PROBABILITE ENTRE DEUX VALEURS
        # ====================================================

        elif operation == "Probabilité entre deux valeurs":

            st.markdown(
                "### 📏 Probabilité entre deux valeurs"
            )

            col1, col2 = st.columns(2)

            with col1:

                lower = st.number_input(
                    "Borne inférieure",
                    value=-1.0,
                    key="normal_probability_lower",
                )

            with col2:

                upper = st.number_input(
                    "Borne supérieure",
                    value=1.0,
                    key="normal_probability_upper",
                )

            col1, col2 = st.columns(2)

            with col1:

                mean_value = st.number_input(
                    "Moyenne μ",
                    value=0.0,
                    key="normal_probability_mean",
                )

            with col2:

                standard_deviation = st.number_input(
                    "Écart-type σ",
                    min_value=0.0001,
                    value=1.0,
                    step=0.1,
                    key="normal_probability_std",
                )

            calculate_normal_probability = st.button(
                "📈 Calculer la probabilité",
                type="primary",
                use_container_width=True,
                key="calculate_normal_probability",
            )

            if calculate_normal_probability:

                try:

                    result = normal_probability_between(
                        lower,
                        upper,
                        mean_value,
                        standard_deviation,
                    )

                    st.divider()

                    st.subheader("📊 Résultat")

                    display_probability(result)

                    st.caption(
                        f"Interprétation : "
                        f"P({format_number(lower)} ≤ X ≤ "
                        f"{format_number(upper)})"
                    )

                    st.divider()

                    st.subheader(
                        "💡 Comprendre la démarche"
                    )

                    st.markdown(
                        explain_normal_probability_between(
                            lower,
                            upper,
                            mean_value,
                            standard_deviation,
                            result,
                        )
                    )

                except (ValueError, TypeError) as exc:

                    display_error(str(exc))

    # ========================================================
    # LOI UNIFORME
    # ========================================================

    elif distribution_type == "Loi uniforme":

        st.subheader("📏 Loi uniforme")

        st.markdown(
            """
            La loi uniforme est définie sur un intervalle :

            - **a** : borne inférieure
            - **b** : borne supérieure
            - avec **a < b**
            """
        )

        operation = st.selectbox(
            "🔧 Choisissez une opération",
            [
                "Densité PDF",
                "Fonction de répartition CDF",
                "Probabilité entre deux valeurs",
            ],
            key="uniform_operation",
        )

        st.divider()

        # ====================================================
        # PDF UNIFORME
        # ====================================================

        if operation == "Densité PDF":

            st.markdown(
                "### 📐 Densité de probabilité"
            )

            col1, col2 = st.columns(2)

            with col1:

                distribution_lower = st.number_input(
                    "Borne inférieure a",
                    value=0.0,
                    key="uniform_pdf_lower",
                )

            with col2:

                distribution_upper = st.number_input(
                    "Borne supérieure b",
                    value=10.0,
                    key="uniform_pdf_upper",
                )

            x = st.number_input(
                "Valeur x",
                value=5.0,
                key="uniform_pdf_x",
            )

            calculate_uniform_pdf = st.button(
                "📈 Calculer la densité",
                type="primary",
                use_container_width=True,
                key="calculate_uniform_pdf",
            )

            if calculate_uniform_pdf:

                try:

                    result = uniform_pdf(
                        x,
                        distribution_lower,
                        distribution_upper,
                    )

                    st.divider()

                    st.subheader("📊 Résultat")

                    st.metric(
                        "Densité f(x)",
                        f"{result:.6f}",
                    )

                    st.caption(
                        f"Pour x = {format_number(x)}, "
                        f"a = {format_number(distribution_lower)}, "
                        f"b = {format_number(distribution_upper)}"
                    )

                    st.divider()

                    st.subheader(
                        "💡 Comprendre la démarche"
                    )

                    st.markdown(
                        explain_uniform_pdf(
                            x,
                            distribution_lower,
                            distribution_upper,
                            result,
                        )
                    )

                except (ValueError, TypeError) as exc:

                    display_error(str(exc))

        # ====================================================
        # CDF UNIFORME
        # ====================================================

        elif operation == "Fonction de répartition CDF":

            st.markdown(
                "### 📊 Fonction de répartition"
            )

            col1, col2 = st.columns(2)

            with col1:

                distribution_lower = st.number_input(
                    "Borne inférieure a",
                    value=0.0,
                    key="uniform_cdf_lower",
                )

            with col2:

                distribution_upper = st.number_input(
                    "Borne supérieure b",
                    value=10.0,
                    key="uniform_cdf_upper",
                )

            x = st.number_input(
                "Valeur x",
                value=5.0,
                key="uniform_cdf_x",
            )

            calculate_uniform_cdf = st.button(
                "📈 Calculer la probabilité",
                type="primary",
                use_container_width=True,
                key="calculate_uniform_cdf",
            )

            if calculate_uniform_cdf:

                try:

                    result = uniform_cdf(
                        x,
                        distribution_lower,
                        distribution_upper,
                    )

                    st.divider()

                    st.subheader("📊 Résultat")

                    display_probability(result)

                    st.caption(
                        f"Interprétation : "
                        f"P(X ≤ {format_number(x)})"
                    )

                    st.divider()

                    st.subheader(
                        "💡 Comprendre la démarche"
                    )

                    st.markdown(
                        explain_uniform_cdf(
                            x,
                            distribution_lower,
                            distribution_upper,
                            result,
                        )
                    )

                except (ValueError, TypeError) as exc:

                    display_error(str(exc))

        # ====================================================
        # PROBABILITE ENTRE DEUX VALEURS
        # ====================================================

        elif operation == "Probabilité entre deux valeurs":

            st.markdown(
                "### 📏 Probabilité entre deux valeurs"
            )

            col1, col2 = st.columns(2)

            with col1:

                lower_bound = st.number_input(
                    "Borne inférieure de l'événement",
                    value=2.0,
                    key="uniform_probability_lower_bound",
                )

            with col2:

                upper_bound = st.number_input(
                    "Borne supérieure de l'événement",
                    value=7.0,
                    key="uniform_probability_upper_bound",
                )

            st.markdown(
                "#### Paramètres de la loi uniforme"
            )

            col1, col2 = st.columns(2)

            with col1:

                distribution_lower = st.number_input(
                    "Borne de la loi a",
                    value=0.0,
                    key="uniform_probability_distribution_lower",
                )

            with col2:

                distribution_upper = st.number_input(
                    "Borne de la loi b",
                    value=10.0,
                    key="uniform_probability_distribution_upper",
                )

            calculate_uniform_probability = st.button(
                "📈 Calculer la probabilité",
                type="primary",
                use_container_width=True,
                key="calculate_uniform_probability",
            )

            if calculate_uniform_probability:

                try:

                    result = uniform_probability_between(
                        lower_bound,
                        upper_bound,
                        distribution_lower,
                        distribution_upper,
                    )

                    st.divider()

                    st.subheader("📊 Résultat")

                    display_probability(result)

                    st.caption(
                        f"Interprétation : "
                        f"P({format_number(lower_bound)} ≤ X ≤ "
                        f"{format_number(upper_bound)})"
                    )

                    st.divider()

                    st.subheader(
                        "💡 Comprendre la démarche"
                    )

                    st.markdown(
                        explain_uniform_probability_between(
                            lower_bound,
                            upper_bound,
                            distribution_lower,
                            distribution_upper,
                            result,
                        )
                    )

                except (ValueError, TypeError) as exc:

                    display_error(str(exc))


# ============================================================
# CORRÉLATION
# ============================================================

elif section == "Corrélation":

    st.header("🔗 Corrélation")

    st.markdown(
        """
        La corrélation permet d'étudier la relation entre
        deux séries quantitatives.

        Cette section permet actuellement de calculer :

        - **la covariance**
        - **le coefficient de corrélation de Pearson**
        """
    )

    st.divider()

    # ========================================================
    # SAISIE DES DONNÉES
    # ========================================================

    st.subheader("📝 Saisie des deux séries")

    col1, col2 = st.columns(2)

    with col1:

        x_text = st.text_area(
            "Série X",
            value="1, 2, 3, 4, 5",
            height=120,
            key="correlation_x_data",
            help=(
                "Séparez les valeurs par une virgule, "
                "un point-virgule, un espace ou une ligne."
            ),
        )

    with col2:

        y_text = st.text_area(
            "Série Y",
            value="2, 4, 6, 8, 10",
            height=120,
            key="correlation_y_data",
            help=(
                "Séparez les valeurs par une virgule, "
                "un point-virgule, un espace ou une ligne."
            ),
        )

    st.caption(
        "Les deux séries doivent contenir le même nombre de valeurs."
    )

    operation = st.radio(
        "🔧 Choisissez une opération",
        [
            "Covariance",
            "Corrélation de Pearson",
        ],
        horizontal=True,
        key="correlation_operation",
    )

    st.divider()

    # ========================================================
    # CALCUL DE LA COVARIANCE
    # ========================================================

    if operation == "Covariance":

        st.subheader("📐 Covariance")

        st.markdown(
            """
            La covariance mesure la manière dont deux variables
            varient simultanément.

            - covariance **positive** : les variables ont tendance
              à augmenter ensemble ;
            - covariance **négative** : l'une a tendance à augmenter
              lorsque l'autre diminue ;
            - covariance proche de **0** : absence de relation
              linéaire évidente.
            """
        )

        calculate_covariance = st.button(
            "📊 Calculer la covariance",
            type="primary",
            use_container_width=True,
            key="calculate_covariance",
        )

        if calculate_covariance:

            try:

                x_values = parse_data(x_text)
                y_values = parse_data(y_text)

                result = covariance(
                    x_values,
                    y_values,
                )

                st.divider()

                st.subheader("📊 Résultat")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Covariance",
                        format_number(result),
                    )

                with col2:

                    st.metric(
                        "Nombre de valeurs X",
                        len(x_values),
                    )

                with col3:

                    st.metric(
                        "Nombre de valeurs Y",
                        len(y_values),
                    )

                st.divider()

                with st.expander(
                    "📋 Voir les séries",
                    expanded=True,
                ):

                    st.write("**Série X :**")

                    st.code(
                        ", ".join(
                            format_number(value)
                            for value in x_values
                        ),
                        language="text",
                    )

                    st.write("**Série Y :**")

                    st.code(
                        ", ".join(
                            format_number(value)
                            for value in y_values
                        ),
                        language="text",
                    )

                st.divider()

                st.subheader(
                    "💡 Comprendre la démarche"
                )

                st.markdown(
                    explain_covariance(
                        x_values,
                        y_values,
                        result,
                    )
                )

            except (ValueError, TypeError) as exc:

                display_error(str(exc))

    # ========================================================
    # CORRÉLATION DE PEARSON
    # ========================================================

    elif operation == "Corrélation de Pearson":

        st.subheader("📈 Coefficient de corrélation de Pearson")

        st.markdown(
            """
            Le coefficient de corrélation de Pearson, noté **r**,
            mesure l'intensité et le sens d'une relation linéaire
            entre deux variables.

            Sa valeur est comprise entre **−1 et +1**.
            """
        )

        calculate_pearson = st.button(
            "📈 Calculer la corrélation",
            type="primary",
            use_container_width=True,
            key="calculate_pearson_correlation",
        )

        if calculate_pearson:

            try:

                x_values = parse_data(x_text)
                y_values = parse_data(y_text)

                result = pearson_correlation(
                    x_values,
                    y_values,
                )

                st.divider()

                st.subheader("📊 Résultat")

                st.metric(
                    "Coefficient de Pearson r",
                    f"{result:.6f}",
                )

                if result > 0:

                    if result >= 0.8:
                        interpretation = (
                            "Forte corrélation linéaire positive."
                        )

                    elif result >= 0.5:
                        interpretation = (
                            "Corrélation linéaire positive modérée."
                        )

                    else:
                        interpretation = (
                            "Faible corrélation linéaire positive."
                        )

                elif result < 0:

                    if result <= -0.8:
                        interpretation = (
                            "Forte corrélation linéaire négative."
                        )

                    elif result <= -0.5:
                        interpretation = (
                            "Corrélation linéaire négative modérée."
                        )

                    else:
                        interpretation = (
                            "Faible corrélation linéaire négative."
                        )

                else:

                    interpretation = (
                        "Aucune corrélation linéaire."
                    )

                st.info(
                    f"**Interprétation :** {interpretation}"
                )

                st.divider()

                with st.expander(
                    "📋 Voir les séries",
                    expanded=True,
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write("**Série X :**")

                        st.code(
                            ", ".join(
                                format_number(value)
                                for value in x_values
                            ),
                            language="text",
                        )

                    with col2:

                        st.write("**Série Y :**")

                        st.code(
                            ", ".join(
                                format_number(value)
                                for value in y_values
                            ),
                            language="text",
                        )

                st.divider()

                st.subheader(
                    "💡 Comprendre la démarche"
                )

                st.markdown(
                    explain_pearson_correlation(
                        x_values,
                        y_values,
                        result,
                    )
                )

            except (ValueError, TypeError) as exc:

                display_error(str(exc))


# ============================================================
# RÉGRESSION LINÉAIRE
# ============================================================

elif section == "Régression linéaire":

    st.header("📈 Régression linéaire")

    st.markdown(
        """
        La régression linéaire permet de modéliser la relation
        entre une variable explicative **X** et une variable
        à expliquer **Y** à l'aide d'une droite.

        Le modèle est de la forme :

        **ŷ = ax + b**

        où :

        - **a** est la pente de la droite ;
        - **b** est l'ordonnée à l'origine ;
        - **ŷ** est la valeur prédite de Y.
        """
    )

    st.divider()

    # ========================================================
    # SAISIE DES DONNÉES
    # ========================================================

    st.subheader("📝 Saisie des données")

    col1, col2 = st.columns(2)

    with col1:

        regression_x_text = st.text_area(
            "Série X — variable explicative",
            value="1, 2, 3, 4, 5",
            height=120,
            key="regression_x_data",
            help=(
                "Entrez les valeurs de X séparées par une "
                "virgule, un point-virgule, un espace ou une ligne."
            ),
        )

    with col2:

        regression_y_text = st.text_area(
            "Série Y — variable à expliquer",
            value="2, 4, 6, 8, 10",
            height=120,
            key="regression_y_data",
            help=(
                "Entrez les valeurs de Y séparées par une "
                "virgule, un point-virgule, un espace ou une ligne."
            ),
        )

    st.caption(
        "Les deux séries doivent contenir le même nombre de valeurs "
        "et X doit présenter une variation."
    )

    # ========================================================
    # CHOIX DE L'OPÉRATION
    # ========================================================

    regression_operation = st.selectbox(
    "🔧 Choisissez une opération",
    [
        "Régression linéaire",
        "Prédiction",
        "Coefficient de détermination R²",
        "Visualisation de la régression",
    ],
    key="regression_operation",
)

    st.divider()

    # ========================================================
    # RÉGRESSION LINÉAIRE
    # ========================================================

    if regression_operation == "Régression linéaire":

        st.subheader("📐 Calcul de la droite de régression")

        st.markdown(
            """
            La droite de régression est calculée sous la forme :

            **ŷ = ax + b**
            """
        )

        calculate_regression = st.button(
            "📈 Calculer la régression",
            type="primary",
            use_container_width=True,
            key="calculate_linear_regression",
        )

        if calculate_regression:

            try:

                x_values = parse_data(
                    regression_x_text
                )

                y_values = parse_data(
                    regression_y_text
                )

                slope, intercept = linear_regression(
                    x_values,
                    y_values,
                )

                r_squared = coefficient_of_determination(
                    x_values,
                    y_values,
                )

                st.divider()

                st.subheader("📊 Résultats")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Pente a",
                        format_number(slope),
                    )

                with col2:

                    st.metric(
                        "Ordonnée à l'origine b",
                        format_number(intercept),
                    )

                with col3:

                    st.metric(
                        "R²",
                        f"{r_squared:.6f}",
                    )

                st.divider()

                # ==================================================
                # ÉQUATION
                # ==================================================

                st.subheader("📐 Équation de la droite")

                if intercept >= 0:

                    equation = (
                        f"ŷ = {format_number(slope)}x + "
                        f"{format_number(intercept)}"
                    )

                else:

                    equation = (
                        f"ŷ = {format_number(slope)}x − "
                        f"{format_number(abs(intercept))}"
                    )

                st.latex(
                    equation.replace("ŷ", r"\hat{y}")
                )

                st.success(
                    f"**Modèle obtenu :** {equation}"
                )

                # ==================================================
                # VALEURS AJUSTÉES
                # ==================================================

                fitted_values = regression_predictions(
                    x_values,
                    y_values,
                )

                with st.expander(
                    "📋 Voir les valeurs observées et ajustées",
                    expanded=True,
                ):

                    for index, (
                        x_value,
                        y_value,
                        predicted_value,
                    ) in enumerate(
                        zip(
                            x_values,
                            y_values,
                            fitted_values,
                        ),
                        start=1,
                    ):

                        st.write(
                            f"**Observation {index} :** "
                            f"X = {format_number(x_value)}, "
                            f"Y = {format_number(y_value)}, "
                            f"Ŷ = {format_number(predicted_value)}"
                        )

                st.divider()

                # ==================================================
                # EXPLICATION DE LA PENTE
                # ==================================================

                st.subheader(
                    "💡 Comprendre la pente"
                )

                st.markdown(
                    explain_regression_slope(
                        x_values,
                        y_values,
                        slope,
                    )
                )

                st.divider()

                # ==================================================
                # EXPLICATION DE L'ORDONNÉE
                # ==================================================

                st.subheader(
                    "💡 Comprendre l'ordonnée à l'origine"
                )

                st.markdown(
                    explain_regression_intercept(
                        x_values,
                        y_values,
                        intercept,
                    )
                )

                st.divider()

                # ==================================================
                # EXPLICATION DU MODÈLE
                # ==================================================

                st.subheader(
                    "💡 Comprendre le modèle"
                )

                st.markdown(
                    explain_linear_regression(
                        x_values,
                        y_values,
                        (slope, intercept),
                    )
                )

            except (ValueError, TypeError) as exc:

                display_error(str(exc))

    # ========================================================
    # PRÉDICTION
    # ========================================================

    elif regression_operation == "Prédiction":

        st.subheader("🔮 Prédiction de nouvelles valeurs")

        st.markdown(
            """
            Après avoir déterminé la droite de régression,
            nous pouvons utiliser le modèle pour prédire une
            valeur de **Y** à partir d'une nouvelle valeur de **X**.
            """
        )

        prediction_text = st.text_area(
            "Nouvelles valeurs de X à prédire",
            value="6, 7, 8",
            height=100,
            key="regression_prediction_data",
            help=(
                "Entrez une ou plusieurs nouvelles valeurs de X."
            ),
        )

        calculate_prediction = st.button(
            "🔮 Calculer les prédictions",
            type="primary",
            use_container_width=True,
            key="calculate_regression_prediction",
        )

        if calculate_prediction:

            try:

                x_values = parse_data(
                    regression_x_text
                )

                y_values = parse_data(
                    regression_y_text
                )

                new_x_values = parse_data(
                    prediction_text
                )

                slope, intercept = linear_regression(
                    x_values,
                    y_values,
                )

                predicted_values = predict(
                    x_values,
                    y_values,
                    new_x_values,
                )

                st.divider()

                st.subheader("📊 Résultats")

                st.metric(
                    "Pente a",
                    format_number(slope),
                )

                st.metric(
                    "Ordonnée à l'origine b",
                    format_number(intercept),
                )

                st.divider()

                st.subheader("🔮 Valeurs prédites")

                for x_value, predicted_value in zip(
                    new_x_values,
                    predicted_values,
                ):

                    st.write(
                        f"Pour **X = {format_number(x_value)}** "
                        f"→ **Ŷ = {format_number(predicted_value)}**"
                    )

                st.divider()

                st.subheader(
                    "💡 Comprendre la prédiction"
                )

                st.markdown(
                    explain_prediction(
                        x_values,
                        y_values,
                        new_x_values,
                        predicted_values,
                        (slope, intercept),
                    )
                )

            except (ValueError, TypeError) as exc:

                display_error(str(exc))

    # ========================================================
    # R²
    # ========================================================

    elif regression_operation == "Coefficient de détermination R²":

        st.subheader(
            "📊 Coefficient de détermination R²"
        )

        st.markdown(
            """
            Le coefficient de détermination **R²** mesure la
            proportion de la variabilité de Y expliquée par
            le modèle linéaire.

            Plus R² est proche de **1**, plus les données observées
            sont proches de la relation linéaire ajustée.
            """
        )

        calculate_r_squared = st.button(
            "📊 Calculer R²",
            type="primary",
            use_container_width=True,
            key="calculate_regression_r_squared",
        )

        if calculate_r_squared:

            try:

                x_values = parse_data(
                    regression_x_text
                )

                y_values = parse_data(
                    regression_y_text
                )

                result = coefficient_of_determination(
                    x_values,
                    y_values,
                )

                st.divider()

                st.subheader("📊 Résultat")

                st.metric(
                    "Coefficient R²",
                    f"{result:.6f}",
                )

                st.caption(
                    f"Soit environ **{result * 100:.2f} %** "
                    "de variabilité expliquée par le modèle."
                )

                st.divider()

                st.subheader(
                    "💡 Comprendre R²"
                )

                st.markdown(
                    explain_r_squared(
                        x_values,
                        y_values,
                        result,
                    )
                )

            except (ValueError, TypeError) as exc:

                display_error(str(exc))


    # ========================================================
    # VISUALISATION DE LA RÉGRESSION
    # ========================================================

    elif regression_operation == "Visualisation de la régression":

        st.subheader("📈 Visualisation de la régression")

        st.markdown(
            """
            Cette visualisation représente :

            - les **observations réelles** sous forme de points ;
            - la **droite de régression linéaire** ;
            - les valeurs prédites par le modèle.

            Elle permet de visualiser graphiquement la relation
            entre **X** et **Y** ainsi que l'ajustement du modèle.
            """
        )

        calculate_regression_visualization = st.button(
            "📊 Générer la visualisation",
            type="primary",
            use_container_width=True,
            key="calculate_regression_visualization",
        )

        if calculate_regression_visualization:

            try:

                x_values = parse_data(
                    regression_x_text
                )

                y_values = parse_data(
                    regression_y_text
                )

                plot_data = regression_plot_data(
                    x_values,
                    y_values,
                )

                observations = plot_data["observations"]
                regression_line = plot_data["regression_line"]
                slope = plot_data["slope"]
                intercept = plot_data["intercept"]

                st.divider()

                # ==================================================
                # RÉSULTATS DU MODÈLE
                # ==================================================

                st.subheader("📊 Paramètres du modèle")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Pente a",
                        format_number(slope),
                    )

                with col2:
                    st.metric(
                        "Ordonnée à l'origine b",
                        format_number(intercept),
                    )

                # ==================================================
                # ÉQUATION
                # ==================================================

                st.divider()

                st.subheader("📐 Équation de la droite")

                if intercept >= 0:
                    equation = (
                        f"ŷ = {format_number(slope)}x + "
                        f"{format_number(intercept)}"
                    )
                else:
                    equation = (
                        f"ŷ = {format_number(slope)}x − "
                        f"{format_number(abs(intercept))}"
                    )

                st.success(
                    f"**Modèle obtenu :** {equation}"
                )

                # ==================================================
                # GRAPHIQUE
                # ==================================================

                st.divider()

                st.subheader(
                    "📈 Nuage de points et droite de régression"
                )

                try:

                    import plotly.graph_objects as go

                    figure = go.Figure()

                    # ----------------------------------------------
                    # OBSERVATIONS
                    # ----------------------------------------------

                    figure.add_trace(
                        go.Scatter(
                            x=[
                                item["x"]
                                for item in observations
                            ],
                            y=[
                                item["y"]
                                for item in observations
                            ],
                            mode="markers",
                            name="Observations",
                            text=[
                                (
                                    f"X = {format_number(item['x'])}"
                                    f"<br>Y = {format_number(item['y'])}"
                                    f"<br>Ŷ = "
                                    f"{format_number(item['predicted_y'])}"
                                )
                                for item in observations
                            ],
                            hovertemplate=(
                                "%{text}<extra></extra>"
                            ),
                        )
                    )

                    # ----------------------------------------------
                    # DROITE DE RÉGRESSION
                    # ----------------------------------------------

                    figure.add_trace(
                        go.Scatter(
                            x=[
                                item["x"]
                                for item in regression_line
                            ],
                            y=[
                                item["y"]
                                for item in regression_line
                            ],
                            mode="lines",
                            name="Droite de régression",
                            hovertemplate=(
                                "X = %{x:.4g}"
                                "<br>Ŷ = %{y:.4g}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    figure.update_layout(
                        title="Régression linéaire",
                        xaxis_title="X — variable explicative",
                        yaxis_title="Y — variable à expliquer",
                        hovermode="closest",
                        height=550,
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                    )

                except ImportError:

                    display_error(
                        "Plotly n'est pas installé. "
                        "Installez-le avec : pip install plotly"
                    )

                # ==================================================
                # TABLEAU DES OBSERVATIONS
                # ==================================================

                st.divider()

                st.subheader(
                    "📋 Valeurs observées et ajustées"
                )

                for index, item in enumerate(
                    observations,
                    start=1,
                ):

                    residual = (
                        item["y"]
                        - item["predicted_y"]
                    )

                    st.write(
                        f"**Observation {index} :** "
                        f"X = {format_number(item['x'])}, "
                        f"Y = {format_number(item['y'])}, "
                        f"Ŷ = {format_number(item['predicted_y'])}, "
                        f"résidu = {format_number(residual)}"
                    )

                # ==================================================
                # EXPLICATION
                # ==================================================

                st.divider()

                st.subheader(
                    "💡 Comprendre la visualisation"
                )

                st.markdown(
                    """
                    Chaque **point** représente une observation
                    réelle `(X, Y)`.

                    La **droite** représente le modèle de régression :

                    **ŷ = ax + b**

                    La distance verticale entre une observation `Y`
                    et sa valeur ajustée `Ŷ` correspond au **résidu** :

                    **e = Y − Ŷ**

                    Plus les observations sont proches de la droite,
                    plus l'ajustement linéaire est visuellement proche
                    des données.
                    """
                )

            except (ValueError, TypeError) as exc:

                display_error(str(exc))

# ============================================================
# KHI-DEUX (χ²)
# ============================================================

elif section == "Khi-deux (χ²)":

    st.header("🧪 Test du Khi-deux (χ²)")

    st.markdown(
        """
        Le test du **Khi-deux d'indépendance** permet d'étudier
        s'il existe une association statistique entre deux
        variables qualitatives.

        Le calcul compare les **effectifs observés** aux
        **effectifs attendus** sous l'hypothèse d'indépendance.
        """
    )

    st.divider()

    # ========================================================
    # HYPOTHÈSES
    # ========================================================

    with st.expander(
        "🧠 Comprendre les hypothèses du test",
        expanded=False,
    ):

        st.markdown(
            explain_chi_square_hypotheses()
        )

    # ========================================================
    # SAISIE DU TABLEAU
    # ========================================================

    st.subheader("📝 Tableau de contingence")

    st.markdown(
        """
        Entrez les **effectifs observés** du tableau.

        Utilisez une ligne par catégorie et séparez les valeurs
        par des virgules.

        **Exemple :**

        ```text
        30, 20
        20, 30
        ```

        Ce tableau correspond à un tableau de contingence **2 × 2**.
        """
    )

    table_text = st.text_area(
        "Effectifs observés",
        value="30, 20\n20, 30",
        height=140,
        key="chi_square_table",
        help=(
            "Une ligne par ligne du tableau. "
            "Séparez les effectifs par des virgules."
        ),
    )

    st.caption(
        "Exemple 3 × 2 : `10, 20` / `20, 30` / `30, 10`"
    )

    # ========================================================
    # NIVEAU DE SIGNIFICATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        significance_level = st.selectbox(
            "🎯 Niveau de signification α",
            [
                0.10,
                0.05,
                0.01,
            ],
            index=1,
            format_func=lambda value: (
                f"α = {value:.2f} "
                f"({value * 100:.0f} %)"
            ),
            key="chi_square_significance_level",
        )

    with col2:

        st.metric(
            "Niveau α sélectionné",
            f"{significance_level:.2f}",
        )

    st.divider()

    # ========================================================
    # CALCUL
    # ========================================================

    calculate_chi_square = st.button(
        "🧪 Effectuer le test du Khi-deux",
        type="primary",
        use_container_width=True,
        key="calculate_chi_square",
    )

    if calculate_chi_square:

        try:

            # ------------------------------------------------
            # PARSING DU TABLEAU
            # ------------------------------------------------

            rows = table_text.strip().splitlines()

            if not rows:

                raise ValueError(
                    "Veuillez saisir un tableau de contingence."
                )

            table: list[list[float]] = []

            for row_index, row_text in enumerate(
                rows,
                start=1,
            ):

                row_text = row_text.strip()

                if not row_text:
                    continue

                try:

                    row = parse_data(row_text)

                except ValueError as exc:

                    raise ValueError(
                        f"Erreur dans la ligne {row_index} : "
                        f"{exc}"
                    ) from exc

                table.append(row)

            if not table:

                raise ValueError(
                    "Aucune ligne valide n'a été trouvée."
                )

            # ------------------------------------------------
            # TEST COMPLET
            # ------------------------------------------------

            result = chi_square_test(
                table,
                significance_level,
            )

            observed = result["observed"]
            expected = result["expected"]
            statistic = result["statistic"]
            df = result["degrees_of_freedom"]
            p_value = result["p_value"]
            critical_value = result["critical_value"]
            reject_h0 = result["reject_null_hypothesis"]

            st.divider()

            # =================================================
            # RÉSULTATS PRINCIPAUX
            # =================================================

            st.subheader("📊 Résultats du test")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "χ² calculé",
                    format_number(statistic),
                )

            with col2:

                st.metric(
                    "Degrés de liberté",
                    df,
                )

            with col3:

                st.metric(
                    "p-value",
                    f"{p_value:.6f}",
                )

            with col4:

                st.metric(
                    "χ² critique",
                    format_number(critical_value),
                )

            st.divider()

            # =================================================
            # DÉCISION
            # =================================================

            if reject_h0:

                st.error(
                    "❌ **Décision : rejet de H₀**"
                )

                st.markdown(
                    """
                    Les données fournissent des éléments
                    statistiques indiquant une **association**
                    entre les deux variables au niveau de
                    signification choisi.
                    """
                )

            else:

                st.success(
                    "✅ **Décision : non-rejet de H₀**"
                )

                st.markdown(
                    """
                    Les données ne fournissent pas suffisamment
                    d'éléments statistiques pour rejeter
                    l'hypothèse d'indépendance au niveau de
                    signification choisi.
                    """
                )

            st.divider()

            # =================================================
            # TABLEAUX
            # =================================================

            tab_observed, tab_expected = st.tabs(
                [
                    "📋 Effectifs observés",
                    "📊 Effectifs attendus",
                ]
            )

            with tab_observed:

                st.subheader(
                    "📋 Tableau des effectifs observés"
                )

                st.dataframe(
                    observed,
                    use_container_width=True,
                )

            with tab_expected:

                st.subheader(
                    "📊 Tableau des effectifs attendus"
                )

                st.dataframe(
                    expected,
                    use_container_width=True,
                )

            st.divider()

            # =================================================
            # INFORMATIONS STATISTIQUES
            # =================================================

            st.subheader(
                "📐 Informations statistiques"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    explain_expected_frequencies(
                        table,
                        expected,
                    )
                )

            with col2:

                st.markdown(
                    explain_degrees_of_freedom(
                        table,
                        df,
                    )
                )

                st.markdown(
                    explain_chi_square_p_value(
                        table,
                        p_value,
                    )
                )

            st.divider()

            # =================================================
            # STATISTIQUE DU KHI-DEUX
            # =================================================

            st.subheader(
                "📐 Calcul de la statistique χ²"
            )

            st.markdown(
                explain_chi_square_statistic(
                    table,
                    statistic,
                )
            )

            st.divider()

            # =================================================
            # VALEUR CRITIQUE
            # =================================================

            st.subheader(
                "🎯 Valeur critique"
            )

            st.markdown(
                explain_chi_square_critical_value(
                    table,
                    significance_level,
                    critical_value,
                )
            )

            st.divider()

            # =================================================
            # DÉCISION
            # =================================================

            st.subheader(
                "⚖️ Décision statistique"
            )

            st.markdown(
                explain_chi_square_decision(
                    p_value,
                    significance_level,
                )
            )

            st.divider()

            # =================================================
            # INTERPRÉTATION
            # =================================================

            st.subheader(
                "📝 Interprétation"
            )

            st.info(
                interpret_chi_square(
                    p_value,
                    significance_level,
                )
            )

            st.divider()

            # =================================================
            # EXPLICATION COMPLÈTE
            # =================================================

            with st.expander(
                "💡 Voir l'explication complète du test",
                expanded=True,
            ):

                st.markdown(
                    explain_chi_square_test(
                        table,
                        significance_level,
                        result,
                    )
                )

        except (ValueError, TypeError) as exc:

            display_error(str(exc))


# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Statistics Lab — Statistiques descriptives, "
    "Distributions, Corrélation & Régression linéaire"
)