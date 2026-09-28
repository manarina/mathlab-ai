
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
    à travers le calcul, la visualisation et l'informatique. 
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
        "Un assistant pour explorer et comprendre les mathématiques." 
    ) 


# ============================================================ 
# PROJECT STATUS 
# ============================================================ 

st.divider() 

st.header("Projet") 

col1, col2 = st.columns(2) 

with col1: 
    st.metric( 
        "Version", 
        "v0.5", 
    ) 

with col2: 
    st.metric( 
        "Modules disponibles", 
        "5", 
    )

st.caption(
    "Algebra · Calculus · Linear Algebra · Statistics · Numerical"
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

