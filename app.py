import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI",
    page_icon="🧮",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🧮 MathLab AI")

st.subheader("Mathematics × Computing")

st.write(
    """
    Une plateforme interactive pour explorer les mathématiques
    à travers le calcul, la visualisation, l'analyse de données
    et l'informatique.
    """
)

st.divider()


# ============================================================
# PRESENTATION
# ============================================================

st.header("Explorez les mathématiques autrement")

st.write(
    """
    MathLab AI transforme des concepts mathématiques en outils
    interactifs permettant de calculer, analyser, visualiser
    et expérimenter.
    """
)


# ============================================================
# MODULES
# ============================================================

st.header("Labs")


# ------------------------------------------------------------
# LIGNE 1
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📐 Algebra Lab")
    st.caption(
        "Équations, polynômes et systèmes algébriques."
    )

with col2:
    st.markdown("### ∫ Calculus Lab")
    st.caption(
        "Dérivées, intégrales, limites et fonctions."
    )

with col3:
    st.markdown("### 🔢 Linear Algebra Lab")
    st.caption(
        "Vecteurs, matrices et systèmes linéaires."
    )


# ------------------------------------------------------------
# LIGNE 2
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Statistics Lab")
    st.caption(
        "Statistiques, distributions et analyse de données."
    )

with col2:
    st.markdown("### 🔬 Numerical Lab")
    st.caption(
        "Méthodes numériques et résolution approchée."
    )

with col3:
    st.markdown("### 🎯 Optimization Lab")
    st.caption(
        "Optimisation mathématique et recherche de solutions."
    )


# ------------------------------------------------------------
# LIGNE 3
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💻 Algorithms Lab")
    st.caption(
        "Algorithmes, complexité et résolution de problèmes."
    )

with col2:
    st.markdown("### 🤖 Machine Learning Lab")
    st.caption(
        "Mathématiques appliquées au machine learning."
    )

with col3:
    st.markdown("### 🧠 AI Math Assistant")
    st.caption(
        "Assistant intelligent pour résoudre, expliquer "
        "et explorer les mathématiques."
    )

    st.success("✅ Disponible")


# ============================================================
# PROJECT STATUS
# ============================================================

st.divider()

st.header("Projet")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Version",
        "v0.9",
    )

with col2:
    st.metric(
        "Labs disponibles",
        "9",
    )

with col3:
    st.metric(
        "Assistant IA",
        "Disponible",
    )


st.caption(
    "Algebra · Calculus · Linear Algebra · Statistics · "
    "Numerical · Optimization · Algorithms · Machine Learning · "
    "AI Math Assistant"
)


# ============================================================
# PROJECT PROGRESS
# ============================================================

st.subheader("Progression du projet")

st.progress(
    1.0,
    text="9 modules sur 9 disponibles",
)

st.write(
    """
    **Modules disponibles :**

    Algebra · Calculus · Linear Algebra · Statistics ·
    Numerical · Optimization · Algorithms · Machine Learning ·
    AI Math Assistant

    **État actuel :**

    🟢 Tous les modules principaux sont disponibles.
    """
)


# ============================================================
# TECHNOLOGIES
# ============================================================

st.subheader("Technologies")

st.write(
    """
    **Python · Streamlit · SymPy · NumPy · SciPy · Pandas ·
    Plotly · Pytest**
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MathLab AI · Mathematics × Computing"
)