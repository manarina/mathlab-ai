import streamlit as st
import sympy as sp

from core.algebra.equations import (
    solve_linear_equation,
    solve_quadratic_equation,
)

from core.algebra.systems import solve_system
from core.algebra.system_explanations import explain_system

from core.algebra.explanations import (
    explain_linear_equation,
    explain_quadratic_equation,
)

from core.algebra.polynomials import (
    parse_polynomial,
    get_polynomial_degree,
    get_polynomial_coefficients,
    get_polynomial_roots,
    factor_polynomial,
    differentiate_polynomial,
    evaluate_polynomial,
)

from core.algebra.polynomial_explanations import (
    explain_polynomial_analysis,
)

from core.algebra.cubic_equations import (
    solve_cubic_equation,
)

from core.algebra.cubic_explanations import (
    explain_cubic_equation,
)

from core.algebra.general_equations import (
    parse_polynomial_equation,
    solve_polynomial_equation,
    evaluate_polynomial_with_horner,
)

from core.algebra.general_equations_explanations import (
    explain_general_polynomial_equation,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Algebra Lab | MathLab AI",
    page_icon="🧮",
    layout="wide",
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧮 MathLab AI")

    st.caption(
        "Mathematics × Computing"
    )

    st.divider()

    equation_type = st.radio(
        "Algebra Lab",
        [
            "Équation du premier degré",
            "Équation du second degré",
            "Équation du troisième degré",
            "Système algébrique",
            "Analyseur de polynôme",
            "Équation polynomiale générale",
        ],
    )


# ============================================================
# HEADER
# ============================================================

st.title("Algebra Lab")

st.markdown(
    "Résolvez et analysez des équations algébriques "
    "et des polynômes de manière interactive."
)

st.divider()


# ============================================================
# FONCTION AFFICHAGE DEMARCHE
# ============================================================


def display_explanation(
    steps: list[dict],
    title: str = "📘 Comprendre la démarche",
    caption: str = (
        "Découvrez étape par étape comment "
        "le problème est résolu."
    ),
) -> None:
    """
    Affiche les étapes pédagogiques d'une résolution.
    """

    st.subheader(title)

    st.caption(caption)

    for index, step in enumerate(
        steps,
        start=1,
    ):

        with st.expander(
            f"Étape {index} — {step['title']}",
            expanded=(index == 1),
        ):

            st.latex(
                step["formula"]
            )

            st.write(
                step["explanation"]
            )


# ============================================================
# EQUATION DU PREMIER DEGRE
# ============================================================

if equation_type == "Équation du premier degré":

    st.subheader(
        "Équation du premier degré"
    )

    st.latex(
        r"ax + b = 0"
    )

    st.caption(
        "Le solveur gère automatiquement les cas : "
        "solution unique, aucune solution ou infinité "
        "de solutions."
    )

    col1, col2 = st.columns(2)

    with col1:

        a = st.number_input(
            "Coefficient a",
            value=2.0,
            step=1.0,
        )

    with col2:

        b = st.number_input(
            "Coefficient b",
            value=-10.0,
            step=1.0,
        )

    if st.button(
        "Résoudre",
        type="primary",
        use_container_width=True,
    ):

        result = solve_linear_equation(
            a,
            b,
        )

        st.divider()

        st.subheader(
            "Résultat"
        )

        st.latex(
            rf"{a:g}x + ({b:g}) = 0"
        )

        # ----------------------------------------------------
        # SOLUTION UNIQUE
        # ----------------------------------------------------

        if result["type"] == "unique":

            solution = result["solutions"][0]

            st.success(
                result["message"]
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Nombre de solutions",
                    "1",
                )

            with col2:

                st.metric(
                    "Solution",
                    f"x = {solution:g}",
                )

        # ----------------------------------------------------
        # AUCUNE SOLUTION
        # ----------------------------------------------------

        elif result["type"] == "none":

            st.error(
                result["message"]
            )

            st.metric(
                "Nombre de solutions",
                "0",
            )

        # ----------------------------------------------------
        # INFINITE SOLUTIONS
        # ----------------------------------------------------

        else:

            st.info(
                result["message"]
            )

            st.metric(
                "Nombre de solutions",
                "∞",
            )

        # ----------------------------------------------------
        # DEMARCHE PEDAGOGIQUE
        # ----------------------------------------------------

        st.divider()

        explanation_steps = (
            explain_linear_equation(
                a,
                b,
                result,
            )
        )

        display_explanation(
            explanation_steps,
            title="📘 Comprendre la démarche",
            caption=(
                "Découvrez étape par étape comment "
                "l'équation est résolue."
            ),
        )


# ============================================================
# EQUATION DU SECOND DEGRE
# ============================================================

elif equation_type == "Équation du second degré":

    st.subheader(
        "Équation du second degré"
    )

    st.latex(
        r"ax^2 + bx + c = 0"
    )

    st.caption(
        "Le solveur calcule le discriminant Δ et détermine "
        "automatiquement les solutions réelles."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        a = st.number_input(
            "Coefficient a",
            value=1.0,
            step=1.0,
        )

    with col2:

        b = st.number_input(
            "Coefficient b",
            value=-5.0,
            step=1.0,
        )

    with col3:

        c = st.number_input(
            "Coefficient c",
            value=6.0,
            step=1.0,
        )

    if st.button(
        "Analyser et résoudre",
        type="primary",
        use_container_width=True,
    ):

        result = solve_quadratic_equation(
            a,
            b,
            c,
        )

        st.divider()

        st.subheader(
            "Résultat"
        )

        st.latex(
            rf"{a:g}x^2 + ({b:g})x + ({c:g}) = 0"
        )

        # ----------------------------------------------------
        # CAS DEGRE 1
        # ----------------------------------------------------

        if result["degree"] == 1:

            st.warning(
                "a = 0 : l'équation devient une équation "
                "du premier degré."
            )

            if result["type"] == "unique":

                solution = result["solutions"][0]

                st.success(
                    result["message"]
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Degré détecté",
                        "1",
                    )

                with col2:

                    st.metric(
                        "Solution",
                        f"x = {solution:g}",
                    )

            elif result["type"] == "none":

                st.error(
                    result["message"]
                )

            else:

                st.info(
                    result["message"]
                )

            # ------------------------------------------------
            # DEMARCHE
            # ------------------------------------------------

            st.divider()

            explanation_steps = (
                explain_quadratic_equation(
                    a,
                    b,
                    c,
                    result,
                )
            )

            display_explanation(
                explanation_steps,
                title="📘 Comprendre la démarche",
                caption=(
                    "Découvrez étape par étape comment "
                    "l'équation est résolue."
                ),
            )

        # ----------------------------------------------------
        # CAS DEGRE 2
        # ----------------------------------------------------

        else:

            discriminant = (
                result["discriminant"]
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Discriminant Δ",
                    f"{discriminant:g}",
                )

            with col2:

                if discriminant > 0:

                    nature = "Δ > 0"

                elif discriminant == 0:

                    nature = "Δ = 0"

                else:

                    nature = "Δ < 0"

                st.metric(
                    "Nature",
                    nature,
                )

            st.divider()

            # ------------------------------------------------
            # DEUX SOLUTIONS
            # ------------------------------------------------

            if result["type"] == "two_real_solutions":

                x1, x2 = (
                    result["solutions"]
                )

                st.success(
                    result["message"]
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Solution x₁",
                        f"{x1:g}",
                    )

                with col2:

                    st.metric(
                        "Solution x₂",
                        f"{x2:g}",
                    )

            # ------------------------------------------------
            # SOLUTION DOUBLE
            # ------------------------------------------------

            elif result["type"] == "one_real_solution":

                x = result["solutions"][0]

                st.success(
                    result["message"]
                )

                st.metric(
                    "Solution double",
                    f"x = {x:g}",
                )

            # ------------------------------------------------
            # AUCUNE SOLUTION REELLE
            # ------------------------------------------------

            else:

                st.warning(
                    result["message"]
                )

                st.info(
                    "Aucune solution réelle."
                )

            # ------------------------------------------------
            # DEMARCHE PEDAGOGIQUE
            # ------------------------------------------------

            st.divider()

            explanation_steps = (
                explain_quadratic_equation(
                    a,
                    b,
                    c,
                    result,
                )
            )

            display_explanation(
                explanation_steps,
                title="📘 Comprendre la démarche",
                caption=(
                    "Découvrez étape par étape comment "
                    "l'équation est résolue."
                ),
            )


# ============================================================
# EQUATION DU TROISIEME DEGRE
# ============================================================

elif equation_type == "Équation du troisième degré":

    st.subheader(
        "Équation du troisième degré"
    )

    st.latex(
        r"ax^3 + bx^2 + cx + d = 0"
    )

    st.caption(
        "Résolution d'une équation cubique avec analyse "
        "du discriminant, recherche de racines rationnelles, "
        "méthode de Horner et paramètres de Cardano."
    )

    # --------------------------------------------------------
    # COEFFICIENTS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        a = st.number_input(
            "Coefficient a",
            value=1.0,
            step=1.0,
            key="cubic_a",
        )

    with col2:

        b = st.number_input(
            "Coefficient b",
            value=-6.0,
            step=1.0,
            key="cubic_b",
        )

    with col3:

        c = st.number_input(
            "Coefficient c",
            value=11.0,
            step=1.0,
            key="cubic_c",
        )

    with col4:

        d = st.number_input(
            "Coefficient d",
            value=-6.0,
            step=1.0,
            key="cubic_d",
        )

    if st.button(
        "Analyser et résoudre",
        type="primary",
        use_container_width=True,
        key="solve_cubic",
    ):

        try:

            # =================================================
            # CALCUL
            # =================================================

            result = solve_cubic_equation(
                a,
                b,
                c,
                d,
            )

            polynomial = result["polynomial"]

            st.divider()

            st.subheader(
                "Résultat de l'analyse"
            )

            st.latex(
                rf"{sp.latex(polynomial.as_expr())} = 0"
            )

            # =================================================
            # INFORMATIONS PRINCIPALES
            # =================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Degré",
                    "3",
                )

            with col2:

                st.metric(
                    "Discriminant cubique",
                    str(result["discriminant"]),
                )

            with col3:

                st.metric(
                    "Nature des racines",
                    result["type"],
                )

            st.divider()

            # =================================================
            # RACINES
            # =================================================

            st.subheader(
                "Solutions"
            )

            roots = result["roots"]

            if roots:

                for index, root in enumerate(
                    roots,
                    start=1,
                ):

                    st.latex(
                        rf"x_{{{index}}} = "
                        rf"{sp.latex(root)}"
                    )

            else:

                st.info(
                    "Aucune racine n'a été déterminée."
                )

            # =================================================
            # HORNER
            # =================================================

            horner = result.get(
                "horner",
                {},
            )

            if horner:

                st.divider()

                st.subheader(
                    "Méthode de Horner"
                )

                rational_roots = horner.get(
                    "rational_roots",
                    [],
                )

                if rational_roots:

                    st.success(
                        "Racine(s) rationnelle(s) trouvée(s)."
                    )

                    roots_text = r",\quad ".join(
                        sp.latex(root)
                        for root in rational_roots
                    )

                    st.latex(
                        rf"x = {roots_text}"
                    )

                else:

                    st.info(
                        "Aucune racine rationnelle n'a été "
                        "trouvée parmi les candidats testés."
                    )

                candidates = horner.get(
                    "tested_candidates",
                    [],
                )

                if candidates:

                    with st.expander(
                        "Voir les candidats testés"
                    ):

                        for candidate in candidates:

                            st.write(
                                f"r = {candidate['candidate']} "
                                f"→ reste = "
                                f"{candidate['remainder']}"
                            )

                remaining = horner.get(
                    "remaining_polynomial"
                )

                if remaining is not None:

                    st.markdown(
                        "**Polynôme après réduction du degré :**"
                    )

                    st.latex(
                        rf"Q(x) = "
                        rf"{sp.latex(remaining.as_expr())}"
                    )

            # =================================================
            # CARDANO
            # =================================================

            cardano = result["cardano"]

            st.divider()

            st.subheader(
                "Méthode de Cardano"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "p",
                    str(cardano["p"]),
                )

            with col2:

                st.metric(
                    "q",
                    str(cardano["q"]),
                )

            st.latex(
                rf"\Delta_C = "
                rf"{sp.latex(cardano['delta'])}"
            )

            st.caption(
                "Le discriminant de Cardano est distinct "
                "du discriminant cubique classique."
            )

            # =================================================
            # FACTORISATION
            # =================================================

            st.divider()

            st.subheader(
                "Factorisation"
            )

            st.latex(
                rf"P(x) = "
                rf"{sp.latex(result['factorized'])}"
            )

            # =================================================
            # DEMARCHE PEDAGOGIQUE
            # =================================================

            st.divider()

            explanation_steps = (
                explain_cubic_equation(
                    result
                )
            )

            display_explanation(
                explanation_steps,
                title="📘 Comprendre la résolution cubique",
                caption=(
                    "Découvrez étape par étape comment "
                    "Horner, la factorisation et Cardano "
                    "interviennent dans la résolution."
                ),
            )

        except Exception as error:

            st.error(
                "Impossible de résoudre cette équation cubique."
            )

            st.info(
                "Vérifiez les coefficients. "
                "Le coefficient a doit être différent de zéro."
            )

            with st.expander(
                "Détails techniques"
            ):

                st.write(
                    str(error)
                )

# ============================================================
# SYSTEME ALGEBRIQUE
# ============================================================

elif equation_type == "Système algébrique":

    st.subheader(
        "Système algébrique"
    )

    st.latex(
        r"A X = B"
    )

    st.caption(
        "Résolvez un système linéaire 2×2 ou 3×3 et "
        "analysez sa matrice, son déterminant, ses rangs "
        "et sa classification."
    )

    # --------------------------------------------------------
    # CHOIX DE LA TAILLE
    # --------------------------------------------------------

    system_size = st.radio(
        "Taille du système",
        [
            "Système 2×2",
            "Système 3×3",
        ],
        horizontal=True,
    )

    if system_size == "Système 2×2":

        # ----------------------------------------------------
        # SAISIE 2×2
        # ----------------------------------------------------

        st.markdown("### Coefficients du système")

        col1, col2, col3 = st.columns(3)

        with col1:
            a11 = st.number_input(
                "a₁₁",
                value=1.0,
                step=1.0,
                key="system_2x2_a11",
            )

            a21 = st.number_input(
                "a₂₁",
                value=3.0,
                step=1.0,
                key="system_2x2_a21",
            )

        with col2:
            a12 = st.number_input(
                "a₁₂",
                value=2.0,
                step=1.0,
                key="system_2x2_a12",
            )

            a22 = st.number_input(
                "a₂₂",
                value=4.0,
                step=1.0,
                key="system_2x2_a22",
            )

        with col3:
            b1 = st.number_input(
                "b₁",
                value=5.0,
                step=1.0,
                key="system_2x2_b1",
            )

            b2 = st.number_input(
                "b₂",
                value=6.0,
                step=1.0,
                key="system_2x2_b2",
            )

        matrix = [
            [a11, a12],
            [a21, a22],
        ]

        constants = [
            b1,
            b2,
        ]

        variables = ["x", "y"]

    else:

        # ----------------------------------------------------
        # SAISIE 3×3
        # ----------------------------------------------------

        st.markdown("### Coefficients du système")

        st.markdown("**Équation 1**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            a11 = st.number_input(
                "a₁₁",
                value=1.0,
                step=1.0,
                key="system_3x3_a11",
            )

        with col2:
            a12 = st.number_input(
                "a₁₂",
                value=1.0,
                step=1.0,
                key="system_3x3_a12",
            )

        with col3:
            a13 = st.number_input(
                "a₁₃",
                value=1.0,
                step=1.0,
                key="system_3x3_a13",
            )

        with col4:
            b1 = st.number_input(
                "b₁",
                value=6.0,
                step=1.0,
                key="system_3x3_b1",
            )

        st.markdown("**Équation 2**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            a21 = st.number_input(
                "a₂₁",
                value=2.0,
                step=1.0,
                key="system_3x3_a21",
            )

        with col2:
            a22 = st.number_input(
                "a₂₂",
                value=-1.0,
                step=1.0,
                key="system_3x3_a22",
            )

        with col3:
            a23 = st.number_input(
                "a₂₃",
                value=1.0,
                step=1.0,
                key="system_3x3_a23",
            )

        with col4:
            b2 = st.number_input(
                "b₂",
                value=3.0,
                step=1.0,
                key="system_3x3_b2",
            )

        st.markdown("**Équation 3**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            a31 = st.number_input(
                "a₃₁",
                value=1.0,
                step=1.0,
                key="system_3x3_a31",
            )

        with col2:
            a32 = st.number_input(
                "a₃₂",
                value=2.0,
                step=1.0,
                key="system_3x3_a32",
            )

        with col3:
            a33 = st.number_input(
                "a₃₃",
                value=-1.0,
                step=1.0,
                key="system_3x3_a33",
            )

        with col4:
            b3 = st.number_input(
                "b₃",
                value=2.0,
                step=1.0,
                key="system_3x3_b3",
            )

        matrix = [
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33],
        ]

        constants = [
            b1,
            b2,
            b3,
        ]

        variables = [
            "x",
            "y",
            "z",
        ]

    # --------------------------------------------------------
    # APERÇU DU SYSTEME
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Système saisi"
    )

    for row_index, row in enumerate(matrix):

        terms = []

        for coefficient, variable in zip(
            row,
            variables,
        ):

            coefficient = sp.sympify(
                coefficient
            )

            if coefficient == 0:
                continue

            if coefficient == 1:
                term = variable

            elif coefficient == -1:
                term = f"-{variable}"

            else:
                term = f"{coefficient}{variable}"

            if terms and coefficient > 0:
                term = f"+ {term}"

            elif terms and coefficient < 0:
                term = term.replace(
                    "-",
                    "- ",
                    1,
                )

            terms.append(term)

        left_side = (
            " ".join(terms)
            if terms
            else "0"
        )

        st.latex(
            rf"{left_side} = {sp.sympify(constants[row_index])}"
        )

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    if st.button(
        "Analyser et résoudre",
        type="primary",
        use_container_width=True,
        key="solve_system",
    ):

        try:

            # =================================================
            # CALCUL
            # =================================================

            result = solve_system(
                matrix,
                constants,
            )

            st.divider()

            st.subheader(
                "Résultat de l'analyse"
            )

            # =================================================
            # MATRICES
            # =================================================

            coefficient_matrix = sp.Matrix(
                matrix
            )

            constants_vector = sp.Matrix(
                constants
            )

            augmented_matrix = (
                coefficient_matrix.row_join(
                    constants_vector
                )
            )

            st.markdown(
                "**Matrice des coefficients A**"
            )

            st.latex(
                rf"{sp.latex(coefficient_matrix)}"
            )

            st.markdown(
                "**Vecteur des constantes B**"
            )

            st.latex(
                rf"{sp.latex(constants_vector)}"
            )

            st.markdown(
                "**Matrice augmentée (A | B)**"
            )

            st.latex(
                rf"{sp.latex(augmented_matrix)}"
            )

            # =================================================
            # INFORMATIONS
            # =================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Rang de A",
                    result["rank"],
                )

            with col2:

                st.metric(
                    "Rang de (A | B)",
                    result["augmented_rank"],
                )

            with col3:

                determinant = result.get(
                    "determinant"
                )

                if determinant is not None:

                    st.metric(
                        "Déterminant",
                        str(determinant),
                    )

                else:

                    st.metric(
                        "Déterminant",
                        "Non défini",
                    )

            # =================================================
            # CLASSIFICATION
            # =================================================

            st.divider()

            st.subheader(
                "Classification"
            )

            system_type = result["type"]

            if system_type == "unique_solution":

                st.success(
                    "Le système possède une solution unique."
                )

            elif system_type == "infinite_solutions":

                st.info(
                    "Le système possède une infinité de solutions."
                )

            else:

                st.error(
                    "Le système ne possède aucune solution."
                )

            # =================================================
            # SOLUTIONS
            # =================================================

            if system_type == "unique_solution":

                st.divider()

                st.subheader(
                    "Solutions"
                )

                solution = result[
                    "solutions"
                ][0]

                solution_columns = st.columns(
                    len(variables)
                )

                for index, (
                    variable,
                    value,
                ) in enumerate(
                    zip(
                        variables,
                        solution,
                    )
                ):

                    with solution_columns[index]:

                        st.metric(
                            variable,
                            str(
                                sp.simplify(
                                    value
                                )
                            ),
                        )

            elif system_type == "infinite_solutions":

                st.divider()

                st.subheader(
                    "Ensemble des solutions"
                )

                st.latex(
                    rf"{sp.latex(result['solution_set'])}"
                )

            # =================================================
            # DEMARCHE DE GAUSS
            # =================================================

            st.divider()

            explanation_steps = explain_system(
                matrix,
                constants,
                result,
            )

            display_explanation(
                explanation_steps,
                title="📘 Comprendre la résolution du système",
                caption=(
                    "Découvrez étape par étape comment "
                    "la matrice est analysée et comment "
                    "la méthode de Gauss permet de résoudre "
                    "le système."
                ),
            )

        except Exception as error:

            st.error(
                "Impossible d'analyser ce système."
            )

            st.info(
                "Vérifiez les coefficients et les constantes "
                "du système."
            )

            with st.expander(
                "Détails techniques"
            ):

                st.write(
                    str(error)
                )


# ============================================================
# ANALYSEUR DE POLYNOME
# ============================================================

elif equation_type == "Analyseur de polynôme":

    st.subheader(
        "Analyseur de polynôme"
    )

    st.latex(
        r"P(x)"
    )

    st.caption(
        "Entrez un polynôme pour analyser son degré, "
        "ses coefficients, ses racines, sa factorisation, "
        "sa dérivée et sa valeur pour un x donné."
    )

    # --------------------------------------------------------
    # SAISIE DU POLYNOME
    # --------------------------------------------------------

    expression = st.text_input(
        "Polynôme P(x)",
        value="x**2 - 5*x + 6",
        help=(
            "Utilisez la syntaxe SymPy. "
            "Exemple : x**2 - 5*x + 6"
        ),
    )

    evaluation_value = st.number_input(
        "Valeur de x pour P(x)",
        value=2.0,
        step=1.0,
    )

    if st.button(
        "Analyser le polynôme",
        type="primary",
        use_container_width=True,
    ):

        try:

            # =================================================
            # CALCULS
            # =================================================

            polynomial = parse_polynomial(
                expression
            )

            degree = get_polynomial_degree(
                polynomial
            )

            coefficients = (
                get_polynomial_coefficients(
                    polynomial
                )
            )

            roots = get_polynomial_roots(
                polynomial
            )

            factorized = factor_polynomial(
                polynomial
            )

            derivative = differentiate_polynomial(
                polynomial
            )

            evaluation = evaluate_polynomial(
                polynomial,
                evaluation_value,
            )

            # =================================================
            # RESULTAT
            # =================================================

            st.divider()

            st.subheader(
                "Résultat de l'analyse"
            )

            st.latex(
                rf"P(x) = "
                rf"{sp.latex(polynomial.as_expr())}"
            )

            # =================================================
            # INFORMATIONS PRINCIPALES
            # =================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Degré",
                    degree,
                )

            with col2:

                st.metric(
                    "Nombre de coefficients",
                    len(coefficients),
                )

            with col3:

                st.metric(
                    "Nombre de racines",
                    len(roots),
                )

            st.divider()

            # =================================================
            # COEFFICIENTS
            # =================================================

            st.subheader(
                "Coefficients"
            )

            coefficient_text = ", ".join(
                [
                    str(coefficient)
                    for coefficient in coefficients
                ]
            )

            st.write(
                coefficient_text
            )

            # =================================================
            # RACINES
            # =================================================

            st.subheader(
                "Racines"
            )

            if roots:

                for index, root in enumerate(
                    roots,
                    start=1,
                ):

                    st.latex(
                        rf"x_{{{index}}} = "
                        rf"{sp.latex(root)}"
                    )

            else:

                st.info(
                    "Aucune racine n'a été trouvée."
                )

            # =================================================
            # FACTORISATION
            # =================================================

            st.subheader(
                "Factorisation"
            )

            st.latex(
                rf"P(x) = "
                rf"{sp.latex(factorized)}"
            )

            # =================================================
            # DERIVEE
            # =================================================

            st.subheader(
                "Dérivée"
            )

            st.latex(
                rf"P'(x) = "
                rf"{sp.latex(derivative.as_expr())}"
            )

            # =================================================
            # EVALUATION
            # =================================================

            st.subheader(
                "Évaluation"
            )

            st.latex(
                rf"P({evaluation_value:g}) = "
                rf"{sp.latex(evaluation)}"
            )

            # =================================================
            # EXPLICATION PEDAGOGIQUE
            # =================================================

            st.divider()

            explanation_steps = (
                explain_polynomial_analysis(
                    polynomial,
                    roots,
                    factorized,
                    derivative,
                )
            )

            display_explanation(
                explanation_steps,
                title=(
                    "📘 Comprendre l'analyse du polynôme"
                ),
                caption=(
                    "Découvrez étape par étape comment "
                    "le polynôme est analysé."
                ),
            )

        except Exception:

            st.error(
                "Impossible d'analyser ce polynôme."
            )

            st.info(
                "Vérifiez la syntaxe de votre expression. "
                "Exemple : x**2 - 5*x + 6"
            )


# ============================================================
# EQUATION POLYNOMIALE GENERALE
# ============================================================

elif equation_type == "Équation polynomiale générale":

    st.subheader(
        "Équation polynomiale générale"
    )

    st.latex(
        r"P(x) = 0"
    )

    st.caption(
        "Entrez une équation polynomiale de degré quelconque. "
        "MathLab AI sélectionne automatiquement une stratégie "
        "de résolution adaptée au degré."
    )

    # --------------------------------------------------------
    # SAISIE
    # --------------------------------------------------------

    expression = st.text_input(
        "Polynôme P(x)",
        value="x**5 - 5*x**4 + 4*x**3",
        help=(
            "Utilisez la syntaxe SymPy. "
            "Exemple : x**5 - 5*x**4 + 4*x**3"
        ),
        key="general_polynomial_expression",
    )

    evaluation_value = st.number_input(
        "Valeur de x pour l'évaluation par Horner",
        value=2.0,
        step=1.0,
        key="general_evaluation_value",
    )

    if st.button(
        "Analyser et résoudre",
        type="primary",
        use_container_width=True,
        key="solve_general_polynomial",
    ):

        try:

            # =================================================
            # PARSING
            # =================================================

            polynomial = parse_polynomial_equation(
                expression
            )

            # =================================================
            # RESOLUTION
            # =================================================

            result = solve_polynomial_equation(
                polynomial
            )

            degree = result["degree"]

            st.divider()

            st.subheader(
                "Résultat de l'analyse"
            )

            st.latex(
                rf"P(x) = "
                rf"{sp.latex(polynomial.as_expr())} = 0"
            )

            # =================================================
            # INFORMATIONS PRINCIPALES
            # =================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Degré",
                    degree,
                )

            with col2:

                st.metric(
                    "Méthode",
                    result["method"],
                )

            with col3:

                st.metric(
                    "Racines exactes",
                    len(result["exact_roots"]),
                )

            # =================================================
            # STRATEGIE
            # =================================================

            st.divider()

            st.subheader(
                "Stratégie de résolution"
            )

            method_descriptions = {
                "linear": (
                    "Équation du premier degré"
                ),
                "quadratic": (
                    "Équation du second degré"
                ),
                "horner_cardano": (
                    "Horner + Cardano"
                ),
                "quartic_symbolic": (
                    "Résolution symbolique du quatrième degré"
                ),
                "factorization_and_numerical": (
                    "Factorisation et méthodes numériques"
                ),
            }

            method_label = method_descriptions.get(
                result["method"],
                result["method"],
            )

            st.success(
                method_label
            )

            # =================================================
            # FACTORISATION
            # =================================================

            st.divider()

            st.subheader(
                "Factorisation"
            )

            st.latex(
                rf"P(x) = "
                rf"{sp.latex(result['factorized'])}"
            )

            # =================================================
            # RACINES EXACTES
            # =================================================

            st.subheader(
                "Solutions exactes"
            )

            exact_roots = result["exact_roots"]

            if exact_roots:

                for index, root in enumerate(
                    exact_roots,
                    start=1,
                ):

                    st.latex(
                        rf"x_{{{index}}} = "
                        rf"{sp.latex(root)}"
                    )

            else:

                st.info(
                    "Aucune solution exacte n'a été déterminée."
                )

            # =================================================
            # SOLUTIONS NUMERIQUES
            # =================================================

            numerical_roots = result[
                "numerical_roots"
            ]

            if degree >= 5:

                st.divider()

                st.subheader(
                    "Solutions numériques"
                )

                if numerical_roots:

                    for index, root in enumerate(
                        numerical_roots,
                        start=1,
                    ):

                        st.latex(
                            rf"x_{{{index}}} \approx "
                            rf"{sp.latex(root)}"
                        )

                else:

                    st.info(
                        "Aucune solution numérique n'a pu "
                        "être calculée."
                    )

            # =================================================
            # EVALUATION HORNER
            # =================================================

            st.divider()

            st.subheader(
                "Évaluation par la méthode de Horner"
            )

            horner_value = (
                evaluate_polynomial_with_horner(
                    polynomial,
                    evaluation_value,
                )
            )

            st.latex(
                rf"P({evaluation_value:g}) "
                rf"\approx {horner_value:g}"
            )

            st.caption(
                "Horner permet d'évaluer efficacement "
                "un polynôme de n'importe quel degré."
            )

            # =================================================
            # DEMARCHE PEDAGOGIQUE
            # =================================================

            st.divider()

            explanation_steps = (
                explain_general_polynomial_equation(
                    result
                )
            )

            display_explanation(
                explanation_steps,
                title=(
                    "📘 Comprendre la résolution"
                ),
                caption=(
                    "Découvrez étape par étape comment "
                    "MathLab AI choisit et applique "
                    "une stratégie de résolution."
                ),
            )

        except Exception as error:

            st.error(
                "Impossible d'analyser cette équation polynomiale."
            )

            st.info(
                "Vérifiez la syntaxe de votre expression. "
                "Exemple : x**5 - 5*x**4 + 4*x**3"
            )

            with st.expander(
                "Détails techniques"
            ):

                st.write(
                    str(error)
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MathLab AI · Algebra Lab · v0.1 · Mathematics × Computing"
)