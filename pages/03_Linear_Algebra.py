from __future__ import annotations

import streamlit as st

from core.linear_algebra.vectors import (
    Vector,
    add_vectors,
    subtract_vectors,
    scalar_multiply,
    dot_product,
    vector_norm,
    vector_distance,
)

from core.linear_algebra.vector_explanations import (
    explain_vector,
    explain_addition,
    explain_subtraction,
    explain_scalar_multiplication,
    explain_dot_product,
    explain_norm,
    explain_distance,
)

from core.linear_algebra.matrices import (
    Matrix,
    create_matrix,
    add_matrices,
    subtract_matrices,
    scalar_multiply_matrix,
    matrix_multiply,
    transpose_matrix,
    matrix_determinant,
    matrix_inverse,
)

from core.linear_algebra.matrix_explanations import (
    explain_matrix,
    explain_addition as explain_matrix_addition,
    explain_subtraction as explain_matrix_subtraction,
    explain_scalar_multiplication as explain_matrix_scalar_multiplication,
    explain_multiplication,
    explain_transpose,
    explain_determinant,
    explain_inverse,
)


from core.linear_algebra.linear_systems import (
    LinearSystem,
    create_linear_system,
    solve_linear_system,
)

from core.linear_algebra.system_explanations import (
    explain_linear_system,
    explain_gaussian_elimination,
    explain_solution,
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Linear Algebra Lab",
    page_icon="📐",
    layout="wide",
)


# ============================================================
# FONCTIONS UTILITAIRES — VECTEURS
# ============================================================

def parse_vector(value: str) -> Vector:
    """
    Convertit une chaîne comme :

        1, 2, 3

    en :

        Vector((1, 2, 3))
    """
    if not value.strip():
        raise ValueError(
            "Veuillez saisir les composantes du vecteur."
        )

    try:
        values = tuple(
            float(component.strip())
            for component in value.split(",")
            if component.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "Les composantes doivent être des nombres "
            "séparés par des virgules."
        ) from exc

    if not values:
        raise ValueError(
            "Le vecteur doit contenir au moins une composante."
        )

    normalized = tuple(
        int(value) if value.is_integer() else value
        for value in values
    )

    return Vector(normalized)


def format_vector(vector: Vector) -> str:
    """Formatage lisible d'un vecteur."""
    return "(" + ", ".join(
        str(value)
        for value in vector.values
    ) + ")"


# ============================================================
# FONCTIONS UTILITAIRES — MATRICES
# ============================================================

def parse_matrix(value: str) -> Matrix:
    """
    Convertit une chaîne représentant une matrice.

    Formats acceptés :

        1, 2
        3, 4

    ou :

        1, 2; 3, 4

    ou encore :

        1, 2, 3
        4, 5, 6
    """

    if not value.strip():
        raise ValueError(
            "Veuillez saisir les valeurs de la matrice."
        )

    cleaned_value = value.strip()

    # Autoriser les deux syntaxes :
    #
    # 1, 2
    # 3, 4
    #
    # et :
    #
    # 1, 2; 3, 4
    #
    cleaned_value = cleaned_value.replace(";", "\n")

    rows_text = [
        row.strip()
        for row in cleaned_value.splitlines()
        if row.strip()
    ]

    if not rows_text:
        raise ValueError(
            "La matrice doit contenir au moins une ligne."
        )

    rows = []

    for row_number, row_text in enumerate(
        rows_text,
        start=1,
    ):
        components = [
            component.strip()
            for component in row_text.split(",")
            if component.strip()
        ]

        if not components:
            raise ValueError(
                f"La ligne {row_number} est vide."
            )

        try:
            values = tuple(
                float(component)
                for component in components
            )
        except ValueError as exc:
            raise ValueError(
                "Toutes les valeurs de la matrice doivent "
                "être des nombres."
            ) from exc

        normalized = tuple(
            int(value) if value.is_integer() else value
            for value in values
        )

        rows.append(normalized)

    try:
        return create_matrix(rows)

    except (ValueError, TypeError) as exc:
        raise ValueError(str(exc)) from exc


def format_matrix(matrix: Matrix) -> str:
    """
    Formatage lisible d'une matrice.
    """

    lines = []

    for row in matrix.values:
        lines.append(
            "["
            + ", ".join(
                str(value)
                for value in row
            )
            + "]"
        )

    return "\n".join(lines)


# ============================================================
# FONCTION ERREUR
# ============================================================

def display_error(message: str) -> None:
    """Affiche une erreur utilisateur."""
    st.error(message)


# ============================================================
# TITRE
# ============================================================

st.title("📐 Linear Algebra Lab")

st.markdown(
    """
    Explorez les principaux concepts d'algèbre linéaire
    avec des calculs détaillés et des explications étape par étape.
    """
)

st.divider()


# ============================================================
# MENU
# ============================================================

analysis_type = st.selectbox(
    "Choisissez une opération",
    [
        # ====================================================
        # VECTEURS
        # ====================================================
        "Création d'un vecteur",
        "Addition",
        "Soustraction",
        "Multiplication par un scalaire",
        "Produit scalaire",
        "Norme",
        "Distance",

        # ====================================================
        # MATRICES
        # ====================================================
        "Création d'une matrice",
        "Addition de matrices",
        "Soustraction de matrices",
        "Multiplication d'une matrice par un scalaire",
        "Multiplication matricielle",
        "Transposée",
        "Déterminant",
        "Inverse",

        # ====================================================
        # SYSTÈMES LINÉAIRES
        # ====================================================
        "Création d'un système linéaire",
        "Résolution d'un système linéaire",
    ],
)


# ============================================================
# CRÉATION D'UN VECTEUR
# ============================================================

if analysis_type == "Création d'un vecteur":

    st.subheader("📌 Création d'un vecteur")

    st.markdown(
        "Saisissez les composantes du vecteur séparées par des virgules."
    )

    vector_input = st.text_input(
        "Vecteur",
        value="1, 2, 3",
        key="vector_creation_input",
    )

    if st.button(
        "Créer le vecteur",
        type="primary",
        key="create_vector_button",
    ):
        try:
            vector = parse_vector(vector_input)

            st.success("Vecteur créé avec succès.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Dimension",
                    vector.dimension,
                )

            with col2:
                st.metric(
                    "Nombre de composantes",
                    len(vector.values),
                )

            st.markdown("### Résultat")

            st.code(
                f"v = {format_vector(vector)}",
                language="text",
            )

            st.markdown("### Explication")

            st.markdown(
                explain_vector(vector)
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# ADDITION DE VECTEURS
# ============================================================

elif analysis_type == "Addition":

    st.subheader("➕ Addition de vecteurs")

    col1, col2 = st.columns(2)

    with col1:
        first_input = st.text_input(
            "Vecteur u",
            value="1, 2, 3",
            key="addition_first",
        )

    with col2:
        second_input = st.text_input(
            "Vecteur v",
            value="4, 5, 6",
            key="addition_second",
        )

    if st.button(
        "Calculer u + v",
        type="primary",
        key="addition_button",
    ):
        try:
            first = parse_vector(first_input)
            second = parse_vector(second_input)

            result = add_vectors(
                first,
                second,
            )

            st.success("Addition effectuée.")

            st.markdown("### Résultat")

            st.code(
                f"{format_vector(first)} + "
                f"{format_vector(second)} = "
                f"{format_vector(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_addition(
                    first,
                    second,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# SOUSTRACTION DE VECTEURS
# ============================================================

elif analysis_type == "Soustraction":

    st.subheader("➖ Soustraction de vecteurs")

    col1, col2 = st.columns(2)

    with col1:
        first_input = st.text_input(
            "Vecteur u",
            value="1, 2, 3",
            key="subtraction_first",
        )

    with col2:
        second_input = st.text_input(
            "Vecteur v",
            value="4, 5, 6",
            key="subtraction_second",
        )

    if st.button(
        "Calculer u - v",
        type="primary",
        key="subtraction_button",
    ):
        try:
            first = parse_vector(first_input)
            second = parse_vector(second_input)

            result = subtract_vectors(
                first,
                second,
            )

            st.success("Soustraction effectuée.")

            st.markdown("### Résultat")

            st.code(
                f"{format_vector(first)} - "
                f"{format_vector(second)} = "
                f"{format_vector(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_subtraction(
                    first,
                    second,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# MULTIPLICATION D'UN VECTEUR PAR UN SCALAIRE
# ============================================================

elif analysis_type == "Multiplication par un scalaire":

    st.subheader("✖️ Multiplication par un scalaire")

    col1, col2 = st.columns(2)

    with col1:
        vector_input = st.text_input(
            "Vecteur u",
            value="1, 2, 3",
            key="scalar_vector",
        )

    with col2:
        scalar_input = st.number_input(
            "Scalaire",
            value=2.0,
            step=1.0,
            key="scalar_value",
        )

    if st.button(
        "Calculer",
        type="primary",
        key="scalar_button",
    ):
        try:
            vector = parse_vector(vector_input)

            scalar = (
                int(scalar_input)
                if scalar_input.is_integer()
                else scalar_input
            )

            result = scalar_multiply(
                vector,
                scalar,
            )

            st.success("Multiplication effectuée.")

            st.markdown("### Résultat")

            st.code(
                f"{scalar} × {format_vector(vector)} = "
                f"{format_vector(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_scalar_multiplication(
                    vector,
                    scalar,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# PRODUIT SCALAIRE
# ============================================================

elif analysis_type == "Produit scalaire":

    st.subheader("🔢 Produit scalaire")

    col1, col2 = st.columns(2)

    with col1:
        first_input = st.text_input(
            "Vecteur u",
            value="1, 2, 3",
            key="dot_first",
        )

    with col2:
        second_input = st.text_input(
            "Vecteur v",
            value="4, 5, 6",
            key="dot_second",
        )

    if st.button(
        "Calculer u · v",
        type="primary",
        key="dot_button",
    ):
        try:
            first = parse_vector(first_input)
            second = parse_vector(second_input)

            result = dot_product(
                first,
                second,
            )

            st.success("Produit scalaire calculé.")

            st.markdown("### Résultat")

            st.code(
                f"{format_vector(first)} · "
                f"{format_vector(second)} = {result}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_dot_product(
                    first,
                    second,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# NORME
# ============================================================

elif analysis_type == "Norme":

    st.subheader("📏 Norme d'un vecteur")

    vector_input = st.text_input(
        "Vecteur u",
        value="1, 2, 3",
        key="norm_vector",
    )

    if st.button(
        "Calculer la norme",
        type="primary",
        key="norm_button",
    ):
        try:
            vector = parse_vector(vector_input)

            result = vector_norm(vector)

            st.success("Norme calculée.")

            st.metric(
                "‖u‖",
                f"{result:.6f}",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_norm(
                    vector,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# DISTANCE
# ============================================================

elif analysis_type == "Distance":

    st.subheader("📐 Distance entre deux vecteurs")

    col1, col2 = st.columns(2)

    with col1:
        first_input = st.text_input(
            "Vecteur u",
            value="1, 2, 3",
            key="distance_first",
        )

    with col2:
        second_input = st.text_input(
            "Vecteur v",
            value="4, 5, 6",
            key="distance_second",
        )

    if st.button(
        "Calculer la distance",
        type="primary",
        key="distance_button",
    ):
        try:
            first = parse_vector(first_input)
            second = parse_vector(second_input)

            result = vector_distance(
                first,
                second,
            )

            st.success("Distance calculée.")

            st.metric(
                "d(u, v)",
                f"{result:.6f}",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_distance(
                    first,
                    second,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# CRÉATION D'UNE MATRICE
# ============================================================

elif analysis_type == "Création d'une matrice":

    st.subheader("📌 Création d'une matrice")

    st.markdown(
        """
        Saisissez les valeurs de la matrice.

        **Format :**
        - une ligne par ligne ;
        - les colonnes sont séparées par des virgules.

        **Exemple :**

        `1, 2`

        `3, 4`

        Vous pouvez également utiliser `;` pour séparer les lignes :

        `1, 2; 3, 4`
        """
    )

    matrix_input = st.text_area(
        "Matrice",
        value="1, 2\n3, 4",
        height=120,
        key="matrix_creation_input",
    )

    if st.button(
        "Créer la matrice",
        type="primary",
        key="create_matrix_button",
    ):
        try:
            matrix = parse_matrix(matrix_input)

            st.success(
                "Matrice créée avec succès."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Lignes",
                    matrix.rows,
                )

            with col2:
                st.metric(
                    "Colonnes",
                    matrix.columns,
                )

            with col3:
                st.metric(
                    "Dimension",
                    f"{matrix.rows} × {matrix.columns}",
                )

            st.markdown("### Résultat")

            st.code(
                f"A =\n{format_matrix(matrix)}",
                language="text",
            )

            st.markdown("### Explication")

            st.markdown(
                explain_matrix(matrix)
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# ADDITION DE MATRICES
# ============================================================

elif analysis_type == "Addition de matrices":

    st.subheader("➕ Addition de matrices")

    col1, col2 = st.columns(2)

    with col1:
        first_input = st.text_area(
            "Matrice A",
            value="1, 2\n3, 4",
            height=120,
            key="matrix_addition_first",
        )

    with col2:
        second_input = st.text_area(
            "Matrice B",
            value="5, 6\n7, 8",
            height=120,
            key="matrix_addition_second",
        )

    if st.button(
        "Calculer A + B",
        type="primary",
        key="matrix_addition_button",
    ):
        try:
            first = parse_matrix(first_input)
            second = parse_matrix(second_input)

            result = add_matrices(
                first,
                second,
            )

            st.success(
                "Addition effectuée."
            )

            st.markdown("### Résultat")

            st.code(
                f"A + B =\n{format_matrix(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_matrix_addition(
                    first,
                    second,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# SOUSTRACTION DE MATRICES
# ============================================================

elif analysis_type == "Soustraction de matrices":

    st.subheader("➖ Soustraction de matrices")

    col1, col2 = st.columns(2)

    with col1:
        first_input = st.text_area(
            "Matrice A",
            value="5, 6\n7, 8",
            height=120,
            key="matrix_subtraction_first",
        )

    with col2:
        second_input = st.text_area(
            "Matrice B",
            value="1, 2\n3, 4",
            height=120,
            key="matrix_subtraction_second",
        )

    if st.button(
        "Calculer A - B",
        type="primary",
        key="matrix_subtraction_button",
    ):
        try:
            first = parse_matrix(first_input)
            second = parse_matrix(second_input)

            result = subtract_matrices(
                first,
                second,
            )

            st.success(
                "Soustraction effectuée."
            )

            st.markdown("### Résultat")

            st.code(
                f"A - B =\n{format_matrix(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_matrix_subtraction(
                    first,
                    second,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# MULTIPLICATION D'UNE MATRICE PAR UN SCALAIRE
# ============================================================

elif analysis_type == "Multiplication d'une matrice par un scalaire":

    st.subheader(
        "✖️ Multiplication d'une matrice par un scalaire"
    )

    col1, col2 = st.columns(2)

    with col1:
        matrix_input = st.text_area(
            "Matrice A",
            value="1, 2\n3, 4",
            height=120,
            key="matrix_scalar_input",
        )

    with col2:
        scalar_input = st.number_input(
            "Scalaire",
            value=2.0,
            step=1.0,
            key="matrix_scalar_value",
        )

    if st.button(
        "Calculer",
        type="primary",
        key="matrix_scalar_button",
    ):
        try:
            matrix = parse_matrix(
                matrix_input
            )

            scalar = (
                int(scalar_input)
                if scalar_input.is_integer()
                else scalar_input
            )

            result = scalar_multiply_matrix(
                matrix,
                scalar,
            )

            st.success(
                "Multiplication effectuée."
            )

            st.markdown("### Résultat")

            st.code(
                f"{scalar}A =\n{format_matrix(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_matrix_scalar_multiplication(
                    matrix,
                    scalar,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# MULTIPLICATION MATRICIELLE
# ============================================================

elif analysis_type == "Multiplication matricielle":

    st.subheader("✖️ Multiplication matricielle")

    st.markdown(
        """
        Pour calculer **A × B**, le nombre de colonnes de A
        doit être égal au nombre de lignes de B.

        Exemple :

        - A : `2 × 2`
        - B : `2 × 2`
        - A × B : `2 × 2`
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        first_input = st.text_area(
            "Matrice A",
            value="1, 2\n3, 4",
            height=120,
            key="matrix_multiplication_first",
        )

    with col2:
        second_input = st.text_area(
            "Matrice B",
            value="5, 6\n7, 8",
            height=120,
            key="matrix_multiplication_second",
        )

    if st.button(
        "Calculer A × B",
        type="primary",
        key="matrix_multiplication_button",
    ):
        try:
            first = parse_matrix(
                first_input
            )

            second = parse_matrix(
                second_input
            )

            result = matrix_multiply(
                first,
                second,
            )

            st.success(
                "Multiplication matricielle effectuée."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Dimension de A",
                    f"{first.rows} × {first.columns}",
                )

            with col2:
                st.metric(
                    "Dimension de B",
                    f"{second.rows} × {second.columns}",
                )

            with col3:
                st.metric(
                    "Dimension du résultat",
                    f"{result.rows} × {result.columns}",
                )

            st.markdown("### Résultat")

            st.code(
                f"A × B =\n{format_matrix(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_multiplication(
                    first,
                    second,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# TRANSPOSÉE
# ============================================================

elif analysis_type == "Transposée":

    st.subheader("🔄 Transposée d'une matrice")

    st.markdown(
        """
        La transposée d'une matrice échange ses lignes
        et ses colonnes.

        Exemple :

        A : `2 × 3`

        Aᵀ : `3 × 2`
        """
    )

    matrix_input = st.text_area(
        "Matrice A",
        value="1, 2, 3\n4, 5, 6",
        height=120,
        key="matrix_transpose_input",
    )

    if st.button(
        "Calculer Aᵀ",
        type="primary",
        key="matrix_transpose_button",
    ):
        try:
            matrix = parse_matrix(
                matrix_input
            )

            result = transpose_matrix(
                matrix
            )

            st.success(
                "Transposée calculée."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Dimension de A",
                    f"{matrix.rows} × {matrix.columns}",
                )

            with col2:
                st.metric(
                    "Dimension de Aᵀ",
                    f"{result.rows} × {result.columns}",
                )

            st.markdown("### Résultat")

            st.code(
                f"Aᵀ =\n{format_matrix(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_transpose(
                    matrix,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# DÉTERMINANT
# ============================================================

elif analysis_type == "Déterminant":

    st.subheader(
        "🔢 Déterminant d'une matrice"
    )

    st.markdown(
        """
        Le déterminant est défini uniquement pour
        une **matrice carrée**.

        Exemple :

        `1, 2`

        `3, 4`
        """
    )

    matrix_input = st.text_area(
        "Matrice carrée A",
        value="1, 2\n3, 4",
        height=120,
        key="matrix_determinant_input",
    )

    if st.button(
        "Calculer le déterminant",
        type="primary",
        key="matrix_determinant_button",
    ):
        try:
            matrix = parse_matrix(
                matrix_input
            )

            result = matrix_determinant(
                matrix
            )

            st.success(
                "Déterminant calculé."
            )

            st.metric(
                "det(A)",
                str(result),
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_determinant(
                    matrix,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))


# ============================================================
# INVERSE
# ============================================================

elif analysis_type == "Inverse":

    st.subheader(
        "🔄 Inverse d'une matrice"
    )

    st.markdown(
        """
        L'inverse est définie uniquement pour une matrice
        carrée dont le déterminant est différent de zéro.

        Exemple :

        `1, 2`

        `3, 4`
        """
    )

    matrix_input = st.text_area(
        "Matrice carrée A",
        value="1, 2\n3, 4",
        height=120,
        key="matrix_inverse_input",
    )

    if st.button(
        "Calculer A⁻¹",
        type="primary",
        key="matrix_inverse_button",
    ):
        try:
            matrix = parse_matrix(
                matrix_input
            )

            result = matrix_inverse(
                matrix
            )

            st.success(
                "Inverse calculée."
            )

            st.markdown("### Résultat")

            st.code(
                f"A⁻¹ =\n{format_matrix(result)}",
                language="text",
            )

            st.markdown(
                "### Explication étape par étape"
            )

            st.markdown(
                explain_inverse(
                    matrix,
                    result,
                )
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))

# ============================================================
# CRÉATION D'UN SYSTÈME LINÉAIRE
# ============================================================

elif analysis_type == "Création d'un système linéaire":

    st.subheader("📐 Création d'un système linéaire")

    st.markdown(
        """
        Un système linéaire peut être écrit sous la forme :

        $$AX = B$$

        où :

        - **A** est la matrice des coefficients ;
        - **X** est le vecteur des inconnues ;
        - **B** est le vecteur des constantes.

        ### Exemple

        Le système :

        $$x + 2y = 5$$

        $$3x + 4y = 11$$

        correspond à :

        **A**
        ```
        1, 2
        3, 4
        ```

        **B**
        ```
        5, 11
        ```
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        coefficients_input = st.text_area(
            "Matrice des coefficients A",
            value="1, 2\n3, 4",
            height=140,
            key="system_creation_coefficients",
        )

    with col2:
        constants_input = st.text_input(
            "Vecteur des constantes B",
            value="5, 11",
            key="system_creation_constants",
        )

    if st.button(
        "Créer le système",
        type="primary",
        key="create_linear_system_button",
    ):
        try:
            coefficients = parse_matrix(
                coefficients_input
            )

            constants = parse_vector(
                constants_input
            )

            system = create_linear_system(
                coefficients,
                constants,
            )

            st.success(
                "Système linéaire créé avec succès."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Nombre d'équations",
                    system.size,
                )

            with col2:
                st.metric(
                    "Nombre d'inconnues",
                    system.size,
                )

            with col3:
                st.metric(
                    "Dimension de A",
                    f"{coefficients.rows} × "
                    f"{coefficients.columns}",
                )

            st.markdown("### Matrice A")

            st.code(
                format_matrix(coefficients),
                language="text",
            )

            st.markdown("### Vecteur B")

            st.code(
                format_vector(constants),
                language="text",
            )

            st.markdown("### Explication")

            st.markdown(
                explain_linear_system(system)
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))

# ============================================================
# RÉSOLUTION D'UN SYSTÈME LINÉAIRE
# ============================================================

elif analysis_type == "Résolution d'un système linéaire":

    st.subheader("🧮 Résolution d'un système linéaire")

    st.markdown(
        """
        Résolvez un système linéaire de la forme :

        $$AX = B$$

        avec la méthode de **l'élimination de Gauss**.

        Le système peut avoir :

        - une **solution unique** ;
        - **aucune solution** ;
        - une **infinité de solutions**.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        coefficients_input = st.text_area(
            "Matrice des coefficients A",
            value="1, 2\n3, 4",
            height=140,
            key="system_solver_coefficients",
        )

    with col2:
        constants_input = st.text_input(
            "Vecteur des constantes B",
            value="5, 11",
            key="system_solver_constants",
        )

    if st.button(
        "Résoudre le système",
        type="primary",
        key="solve_linear_system_button",
    ):
        try:
            coefficients = parse_matrix(
                coefficients_input
            )

            constants = parse_vector(
                constants_input
            )

            system = create_linear_system(
                coefficients,
                constants,
            )

            solution = solve_linear_system(
                system
            )

            st.success(
                "Système analysé avec succès."
            )

            # ------------------------------------------------
            # INFORMATIONS SUR LE SYSTÈME
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Équations",
                    system.size,
                )

            with col2:
                st.metric(
                    "Inconnues",
                    system.size,
                )

            with col3:
                st.metric(
                    "Type de résultat",
                    solution.solution_type,
                )

            # ------------------------------------------------
            # SYSTÈME
            # ------------------------------------------------

            st.markdown("### Système")

            st.markdown(
                explain_linear_system(system)
            )

            # ------------------------------------------------
            # MÉTHODE DE GAUSS
            # ------------------------------------------------

            st.markdown(
                "### Méthode de résolution"
            )

            st.markdown(
                explain_gaussian_elimination(system)
            )

            # ------------------------------------------------
            # RÉSULTAT
            # ------------------------------------------------

            st.markdown(
                "### Résultat"
            )

            if solution.is_unique:

                st.success(
                    "Le système possède une solution unique."
                )

                if solution.values is not None:

                    result_lines = []

                    for index, value in enumerate(
                        solution.values
                    ):
                        variable = (
                            "x"
                            if index == 0
                            else f"x_{index + 1}"
                        )

                        result_lines.append(
                            f"{variable} = {value}"
                        )

                    st.code(
                        "\n".join(result_lines),
                        language="text",
                    )

            elif solution.has_no_solution:

                st.warning(
                    "Le système ne possède aucune solution."
                )

            elif solution.has_infinite_solutions:

                st.info(
                    "Le système possède une infinité de solutions."
                )

            # ------------------------------------------------
            # EXPLICATION DU RÉSULTAT
            # ------------------------------------------------

            st.markdown(
                "### Explication du résultat"
            )

            st.markdown(
                explain_solution(solution)
            )

        except (ValueError, TypeError) as exc:
            display_error(str(exc))

# ============================================================
# INFORMATIONS
# ============================================================

st.divider()

st.caption(
    "Linear Algebra Lab — Modules Vecteurs et Matrices"
)