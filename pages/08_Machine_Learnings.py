
"""
MathLab AI — Machine Learning Lab
=================================

Interface Streamlit dédiée aux algorithmes de Machine Learning.

Modules disponibles :
    - Linear Regression
    - Logistic Regression
    - K-Nearest Neighbors (K-NN)
    - K-Means Clustering
    - Decision Tree
    - Random Forest
    - Naive Bayes
    - Principal Component Analysis (PCA)

Les calculs sont effectués dans :
    core.machine_learning.linear_regression
    core.machine_learning.logistic_regression
"""

import streamlit as st
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from core.machine_learning.linear_regression import (
    linear_regression,
)

from core.machine_learning.logistic_regression import (
    logistic_regression,
)

from core.machine_learning.knn import (
    knn
)

from core.machine_learning.kmeans import (
    kmeans,
)

from core.machine_learning.decision_tree import (
    decision_tree,
)

from core.machine_learning.random_forest import (
    random_forest,
)

from core.machine_learning.naive_bayes import (
    naive_bayes,
)    

from core.machine_learning.pca import (
    pca,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI — Machine Learning Lab",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 MathLab AI — Machine Learning Lab")

st.markdown(
    """
    Explorez les principaux concepts du **Machine Learning**
    à travers des implémentations mathématiques explicables.

    ### Module actuel
    **📈 Linear Regression / 🎯 Logistic Regression**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🤖 Machine Learning")

    algorithm = st.selectbox(
        "Choisir un algorithme",
        [
            "📈 Linear Regression",
            "🎯 Logistic Regression",
            "👥 K-NN",
            "🧩 K-Means",
            "🌳 Decision Tree",
            "🌲 Random Forest",
            "🎲 Naive Bayes",
            "📐 PCA",
        ],
        index=0,
        key="ml_algorithm",
    )

    st.divider()

    st.info(
        "Les algorithmes seront ajoutés progressivement. "
        "Linear Regression et Logistic Regression sont actuellement disponibles."
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

if algorithm == "📈 Linear Regression":

    st.header("📈 Linear Regression")

    st.markdown(
        """
        La régression linéaire cherche une relation linéaire entre
        une variable indépendante **X** et une variable dépendante **Y**.

        Le modèle est :

        $$
        \\hat{y} = b_0 + b_1x
        $$

        où :

        - $b_0$ : intercept
        - $b_1$ : pente
        - $\\hat{y}$ : valeur prédite
        """
    )

    # ========================================================
    # EXAMPLE DATA
    # ========================================================

    st.subheader("📊 Données")

    example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Exemple simple",
            "Exemple avec bruit",
            "Exemple croissance",
        ],
        key="linear_regression_example",
    )

    if example == "Exemple simple":
        default_x = "1, 2, 3, 4, 5"
        default_y = "3, 5, 7, 9, 11"

    elif example == "Exemple avec bruit":
        default_x = "1, 2, 3, 4, 5, 6, 7, 8"
        default_y = "2.1, 4.2, 5.8, 8.1, 9.9, 12.2, 13.7, 16.1"

    elif example == "Exemple croissance":
        default_x = "1, 2, 3, 4, 5, 6"
        default_y = "2, 4.1, 6.2, 7.8, 10.1, 12.2"

    else:
        default_x = "1, 2, 3, 4, 5"
        default_y = "2, 4, 6, 8, 10"

    col1, col2 = st.columns(2)

    with col1:
        x_input = st.text_area(
            "Variable indépendante X",
            value=default_x,
            height=100,
            help="Entrez les valeurs séparées par des virgules.",
            key="linear_regression_x",
        )

    with col2:
        y_input = st.text_area(
            "Variable dépendante Y",
            value=default_y,
            height=100,
            help="Entrez les valeurs séparées par des virgules.",
            key="linear_regression_y",
        )

    # ========================================================
    # PARSING
    # ========================================================

    def parse_values(value: str) -> np.ndarray:
        """Convertit une chaîne de nombres en tableau NumPy."""
        try:
            values = [
                float(item.strip())
                for item in value.split(",")
                if item.strip()
            ]

            return np.asarray(values, dtype=float)

        except ValueError as exc:
            raise ValueError(
                "Les données doivent contenir uniquement des nombres "
                "séparés par des virgules."
            ) from exc

    # ========================================================
    # EXECUTION
    # ========================================================

    if st.button(
        "🚀 Exécuter la régression",
        type="primary",
        use_container_width=True,
        key="linear_regression_execute",
    ):

        try:
            x = parse_values(x_input)
            y = parse_values(y_input)

            # ------------------------------------------------
            # VALIDATION MINIMALE
            # ------------------------------------------------

            if len(x) == 0 or len(y) == 0:
                raise ValueError(
                    "Les données X et Y ne doivent pas être vides."
                )

            if len(x) != len(y):
                raise ValueError(
                    "X et Y doivent contenir le même nombre de valeurs."
                )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            result = linear_regression(x, y)

            st.session_state["linear_regression_result"] = result
            st.session_state["linear_regression_x_data"] = x
            st.session_state["linear_regression_y_data"] = y

        except ValueError as exc:
            st.error(f"❌ {exc}")


# ============================================================
# LINEAR REGRESSION RESULTS
# ============================================================

if "linear_regression_result" in st.session_state:

    result = st.session_state["linear_regression_result"]
    x = st.session_state["linear_regression_x_data"]
    y = st.session_state["linear_regression_y_data"]

    st.divider()

    st.header("📊 Résultats")

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Intercept (b₀)",
            f"{result['intercept']:.4f}",
        )

    with col2:
        st.metric(
            "Slope (b₁)",
            f"{result['slope']:.4f}",
        )

    with col3:
        st.metric(
            "MSE",
            f"{result['mse']:.6f}",
        )

    with col4:
        st.metric(
            "R²",
            f"{result['r2']:.4f}",
        )

    # ========================================================
    # EQUATION
    # ========================================================

    st.subheader("📐 Équation du modèle")

    intercept = result["intercept"]
    slope = result["slope"]

    if slope >= 0:
        equation = (
            f"ŷ = {intercept:.4f} + {slope:.4f}x"
        )
    else:
        equation = (
            f"ŷ = {intercept:.4f} - {abs(slope):.4f}x"
        )

    st.latex(
        rf"\hat{{y}} = {intercept:.4f} "
        rf"{'+' if slope >= 0 else '-'} "
        rf"{abs(slope):.4f}x"
    )

    st.code(
        equation,
        language="text",
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.subheader("📈 Visualisation")

    predictions = result["predictions"]

    if plt is None:
        st.warning(
            "La visualisation nécessite Matplotlib. "
            "Installez-le avec `pip install matplotlib`."
        )

    else:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(
            x,
            y,
            label="Données observées",
        )

        sort_index = np.argsort(x)

        ax.plot(
            x[sort_index],
            predictions[sort_index],
            label="Droite de régression",
            linewidth=2,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Régression linéaire")

        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    # ========================================================
    # PREDICTIONS TABLE
    # ========================================================

    st.subheader("🔢 Prédictions")

    prediction_df = pd.DataFrame(
        {
            "X": x,
            "Y réel": y,
            "Y prédit": predictions,
            "Résidu": y - predictions,
        }
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("💡 Interprétation")

    if result["r2"] >= 0.9:
        r2_description = (
            "Les données sont très fortement expliquées "
            "par le modèle linéaire."
        )

    elif result["r2"] >= 0.7:
        r2_description = (
            "Le modèle explique une part importante "
            "de la variabilité des données."
        )

    elif result["r2"] >= 0.5:
        r2_description = (
            "Le modèle présente une relation linéaire "
            "modérée avec les données."
        )

    else:
        r2_description = (
            "Le modèle linéaire explique relativement peu "
            "de la variabilité observée."
        )

    st.markdown(
        f"""
        **Pente :** `{slope:.4f}`

        Une augmentation d'une unité de X est associée,
        dans le modèle, à une variation de **{slope:.4f}**
        unité de Y.

        **Intercept :** `{intercept:.4f}`

        Lorsque X vaut 0, le modèle prédit une valeur de Y
        égale à **{intercept:.4f}**.

        **R² :** `{result["r2"]:.4f}`

        {r2_description}

        **MSE :** `{result["mse"]:.6f}`

        Le MSE mesure l'erreur quadratique moyenne entre les
        valeurs observées et les valeurs prédites.
        """
    )

    # ========================================================
    # MATHEMATICAL EXPLANATION
    # ========================================================

    with st.expander("📚 Comprendre la méthode mathématique"):

        st.markdown(
            """
            ### 1. Moyennes

            On calcule d'abord :

            $$
            \\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n}x_i
            $$

            $$
            \\bar{y} = \\frac{1}{n}\\sum_{i=1}^{n}y_i
            $$

            ### 2. Pente

            La pente est donnée par :

            $$
            b_1 =
            \\frac{
            \\sum (x_i-\\bar{x})(y_i-\\bar{y})
            }{
            \\sum (x_i-\\bar{x})^2
            }
            $$

            ### 3. Intercept

            Ensuite :

            $$
            b_0 = \\bar{y} - b_1\\bar{x}
            $$

            ### 4. Prédiction

            Pour une nouvelle valeur $x$ :

            $$
            \\hat{y}=b_0+b_1x
            $$

            ### 5. Erreur quadratique moyenne

            $$
            MSE =
            \\frac{1}{n}
            \\sum (y_i-\\hat{y}_i)^2
            $$

            ### 6. Coefficient R²

            $$
            R^2 =
            1 -
            \\frac{
            \\sum(y_i-\\hat{y}_i)^2
            }{
            \\sum(y_i-\\bar{y})^2
            }
            $$
            """
        )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

elif algorithm == "🎯 Logistic Regression":

    st.header("🎯 Logistic Regression")

    st.markdown(
        """
        La **régression logistique** est utilisée pour résoudre
        un problème de **classification binaire**.

        Contrairement à la régression linéaire, la variable cible
        appartient ici à deux classes :

        - `0` → classe négative
        - `1` → classe positive

        Le modèle commence par calculer :

        $$
        z = b_0 + b_1x
        $$

        puis transforme ce score en probabilité avec la fonction
        sigmoïde :

        $$
        \\sigma(z) =
        \\frac{1}{1+e^{-z}}
        $$

        La probabilité obtenue est comprise entre **0 et 1**.
        """
    )

    # ========================================================
    # EXAMPLE DATA
    # ========================================================

    st.subheader("📊 Données de classification")

    logistic_example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Exemple simple",
            "Exemple réussite",
            "Exemple admission",
        ],
        key="logistic_regression_example",
    )

    if logistic_example == "Exemple simple":

        default_logistic_x = (
            "1, 2, 3, 4, 5, 6"
        )

        default_logistic_y = (
            "0, 0, 0, 1, 1, 1"
        )

    elif logistic_example == "Exemple réussite":

        default_logistic_x = (
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
        )

        default_logistic_y = (
            "0, 0, 0, 0, 0, 1, 1, 1, 1, 1"
        )

    elif logistic_example == "Exemple admission":

        default_logistic_x = (
            "30, 35, 40, 45, 50, 55, 60, 65, 70, 75"
        )

        default_logistic_y = (
            "0, 0, 0, 0, 0, 1, 1, 1, 1, 1"
        )

    else:

        default_logistic_x = (
            "1, 2, 3, 4, 5, 6"
        )

        default_logistic_y = (
            "0, 0, 1, 0, 1, 1"
        )

    col1, col2 = st.columns(2)

    with col1:

        logistic_x_input = st.text_area(
            "Variable indépendante X",
            value=default_logistic_x,
            height=100,
            help=(
                "Entrez les valeurs numériques séparées "
                "par des virgules."
            ),
            key="logistic_regression_x",
        )

    with col2:

        logistic_y_input = st.text_area(
            "Classe Y",
            value=default_logistic_y,
            height=100,
            help=(
                "Entrez uniquement des classes 0 et 1 "
                "séparées par des virgules."
            ),
            key="logistic_regression_y",
        )

    # ========================================================
    # PARAMETERS
    # ========================================================

    st.subheader("⚙️ Paramètres d'apprentissage")

    parameter_col1, parameter_col2, parameter_col3 = st.columns(3)

    with parameter_col1:

        logistic_learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0001,
            max_value=10.0,
            value=0.1,
            step=0.01,
            format="%.4f",
            key="logistic_regression_learning_rate",
        )

    with parameter_col2:

        logistic_iterations = st.number_input(
            "Nombre d'itérations",
            min_value=1,
            max_value=100000,
            value=2000,
            step=100,
            key="logistic_regression_iterations",
        )

    with parameter_col3:

        logistic_threshold = st.slider(
            "Seuil de classification",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            key="logistic_regression_threshold",
        )

    # ========================================================
    # PARSING
    # ========================================================

    def parse_logistic_values(value: str) -> np.ndarray:
        """Convertit une chaîne de nombres en tableau NumPy."""

        try:

            values = [
                float(item.strip())
                for item in value.split(",")
                if item.strip()
            ]

            return np.asarray(
                values,
                dtype=float,
            )

        except ValueError as exc:

            raise ValueError(
                "Les données doivent contenir uniquement "
                "des nombres séparés par des virgules."
            ) from exc

    # ========================================================
    # EXECUTION
    # ========================================================

    if st.button(
        "🚀 Exécuter la régression logistique",
        type="primary",
        use_container_width=True,
        key="logistic_regression_execute",
    ):

        try:

            x = parse_logistic_values(
                logistic_x_input
            )

            y = parse_logistic_values(
                logistic_y_input
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if len(x) == 0 or len(y) == 0:

                raise ValueError(
                    "Les données X et Y ne doivent pas être vides."
                )

            if len(x) != len(y):

                raise ValueError(
                    "X et Y doivent contenir le même nombre "
                    "de valeurs."
                )

            if len(x) < 2:

                raise ValueError(
                    "La régression logistique nécessite "
                    "au moins deux observations."
                )

            if not np.all(np.isfinite(x)):

                raise ValueError(
                    "X contient des valeurs non finies."
                )

            if not np.all(np.isfinite(y)):

                raise ValueError(
                    "Y contient des valeurs non finies."
                )

            unique_y = np.unique(y)

            if not np.all(
                np.isin(unique_y, [0.0, 1.0])
            ):

                raise ValueError(
                    "Y doit contenir uniquement les classes "
                    "0 et 1."
                )

            if len(unique_y) < 2:

                raise ValueError(
                    "Y doit contenir les deux classes 0 et 1 "
                    "pour entraîner une régression logistique."
                )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            result = logistic_regression(
                x,
                y,
                learning_rate=logistic_learning_rate,
                n_iterations=int(logistic_iterations),
                threshold=logistic_threshold,
            )

            st.session_state[
                "logistic_regression_result"
            ] = result

            st.session_state[
                "logistic_regression_x_data"
            ] = x

            st.session_state[
                "logistic_regression_y_data"
            ] = y

        except ValueError as exc:

            st.error(
                f"❌ {exc}"
            )


# ============================================================
# LOGISTIC REGRESSION RESULTS
# ============================================================

if "logistic_regression_result" in st.session_state:

    result = st.session_state[
        "logistic_regression_result"
    ]

    x = st.session_state[
        "logistic_regression_x_data"
    ]

    y = st.session_state[
        "logistic_regression_y_data"
    ]

    st.divider()

    st.header("📊 Résultats — Logistic Regression")

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Intercept (b₀)",
            f"{result['intercept']:.4f}",
        )

    with col2:

        st.metric(
            "Coefficient (b₁)",
            f"{result['coefficient']:.4f}",
        )

    with col3:

        st.metric(
            "Binary Cross-Entropy",
            f"{result['loss']:.6f}",
        )

    with col4:

        st.metric(
            "Accuracy",
            f"{result['accuracy']:.2%}",
        )

    # ========================================================
    # MODEL EQUATION
    # ========================================================

    st.subheader("📐 Équation du modèle")

    intercept = result["intercept"]
    coefficient = result["coefficient"]

    sign = "+" if coefficient >= 0 else "-"

    st.latex(
        rf"""
        z = {intercept:.4f}
        {sign}
        {abs(coefficient):.4f}x
        """
    )

    st.latex(
        rf"""
        P(y=1\mid x)
        =
        \frac{{1}}{{1+e^{{-({intercept:.4f}
        {sign}
        {abs(coefficient):.4f}x)}}}}
        """
    )

    st.code(
        (
            f"z = {intercept:.4f} "
            f"{sign} "
            f"{abs(coefficient):.4f}x\n"
            f"P(y=1|x) = 1 / (1 + exp(-z))"
        ),
        language="text",
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.subheader("📈 Visualisation")

    if plt is None:

        st.warning(
            "La visualisation nécessite Matplotlib. "
            "Installez-le avec `pip install matplotlib`."
        )

    else:

        probabilities = result["probabilities"]
        predictions = result["predictions"]

        # ----------------------------------------------------
        # DATA + SIGMOID
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.scatter(
            x[y == 0],
            probabilities[y == 0],
            label="Classe 0",
        )

        ax.scatter(
            x[y == 1],
            probabilities[y == 1],
            label="Classe 1",
        )

        # Domaine de la courbe
        x_min = float(np.min(x))
        x_max = float(np.max(x))

        if x_min == x_max:

            x_curve = np.array([
                x_min - 1,
                x_max + 1,
            ])

        else:

            x_curve = np.linspace(
                x_min,
                x_max,
                300,
            )

        z_curve = (
            intercept
            + coefficient * x_curve
        )

        probability_curve = (
            1.0
            / (
                1.0
                + np.exp(-z_curve)
            )
        )

        ax.plot(
            x_curve,
            probability_curve,
            label="Fonction sigmoïde",
            linewidth=2,
        )

        ax.axhline(
            result["threshold"],
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Seuil = {result['threshold']:.2f}"
            ),
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Probabilité P(y=1)")
        ax.set_title(
            "Régression logistique — Fonction sigmoïde"
        )

        ax.set_ylim(
            -0.05,
            1.05,
        )

        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    # ========================================================
    # PREDICTIONS TABLE
    # ========================================================

    st.subheader("🔢 Prédictions")

    probabilities = result["probabilities"]
    predictions = result["predictions"]

    prediction_df = pd.DataFrame(
        {
            "X": x,
            "Classe réelle": y.astype(int),
            "Probabilité P(Y=1)": probabilities,
            "Classe prédite": predictions,
            "Correct": (
                y.astype(int) == predictions
            ),
        }
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("💡 Interprétation")

    if coefficient > 0:

        coefficient_description = (
            "Le coefficient est positif. "
            "Dans le modèle, une augmentation de X "
            "est associée à une augmentation de la "
            "probabilité d'appartenir à la classe 1."
        )

    elif coefficient < 0:

        coefficient_description = (
            "Le coefficient est négatif. "
            "Dans le modèle, une augmentation de X "
            "est associée à une diminution de la "
            "probabilité d'appartenir à la classe 1."
        )

    else:

        coefficient_description = (
            "Le coefficient est proche de zéro. "
            "Le modèle utilise donc peu la variable X "
            "pour différencier les deux classes."
        )

    if result["accuracy"] >= 0.9:

        accuracy_description = (
            "Sur les données utilisées pour l'apprentissage, "
            "le modèle classe correctement au moins 90 % "
            "des observations."
        )

    elif result["accuracy"] >= 0.7:

        accuracy_description = (
            "Sur les données utilisées pour l'apprentissage, "
            "le modèle classe correctement une majorité "
            "des observations."
        )

    else:

        accuracy_description = (
            "L'accuracy obtenue sur ces données est relativement "
            "faible ; les observations peuvent être difficiles "
            "à séparer avec ce modèle à une seule variable."
        )

    st.markdown(
        f"""
        **Coefficient :** `{coefficient:.4f}`

        {coefficient_description}

        **Intercept :** `{intercept:.4f}`

        L'intercept correspond au score $z$ lorsque
        **X = 0**.

        **Seuil :** `{result["threshold"]:.2f}`

        Une probabilité supérieure ou égale à ce seuil
        est classée dans la **classe 1**. Sinon,
        l'observation est classée dans la **classe 0**.

        **Accuracy :** `{result["accuracy"]:.2%}`

        {accuracy_description}

        **Binary Cross-Entropy :** `{result["loss"]:.6f}`

        La Binary Cross-Entropy mesure l'écart entre les
        probabilités prédites et les classes réelles.
        """
    )

    # ========================================================
    # MATHEMATICAL EXPLANATION
    # ========================================================

    with st.expander(
        "📚 Comprendre la méthode mathématique"
    ):

        st.markdown(
            """
            ### 1. Modèle linéaire

            La première étape consiste à calculer un score :

            $$
            z = b_0 + b_1x
            $$

            où :

            - $b_0$ est l'intercept ;
            - $b_1$ est le coefficient ;
            - $x$ est la variable d'entrée.

            ---

            ### 2. Fonction sigmoïde

            Le score $z$ est transformé en probabilité :

            $$
            \\sigma(z)
            =
            \\frac{1}{1+e^{-z}}
            $$

            Cette fonction transforme n'importe quel score réel
            en une valeur comprise entre 0 et 1.

            ---

            ### 3. Classification

            On choisit ensuite un seuil.

            Par défaut :

            $$
            threshold = 0.5
            $$

            Si :

            $$
            P(y=1|x) \\geq 0.5
            $$

            alors :

            $$
            \\hat{y}=1
            $$

            Sinon :

            $$
            \\hat{y}=0
            $$

            ---

            ### 4. Fonction de coût

            La régression logistique utilise la
            **Binary Cross-Entropy** :

            $$
            BCE =
            -\\frac{1}{n}
            \\sum_{i=1}^{n}
            \\left[
            y_i\\log(p_i)
            +
            (1-y_i)\\log(1-p_i)
            \\right]
            $$

            où $p_i$ représente la probabilité prédite.

            ---

            ### 5. Descente de gradient

            Les paramètres $b_0$ et $b_1$ sont initialisés,
            puis mis à jour progressivement afin de réduire
            la fonction de coût.

            Pour l'intercept :

            $$
            b_0
            \\leftarrow
            b_0
            -
            \\alpha
            \\frac{1}{n}
            \\sum(p_i-y_i)
            $$

            Pour le coefficient :

            $$
            b_1
            \\leftarrow
            b_1
            -
            \\alpha
            \\frac{1}{n}
            \\sum
            (p_i-y_i)x_i
            $$

            où $\\alpha$ représente le **learning rate**.

            ---

            ### 6. Accuracy

            Enfin :

            $$
            Accuracy =
            \\frac{
            \\text{nombre de prédictions correctes}
            }{
            n
            }
            $$

            L'accuracy indique la proportion d'observations
            correctement classées sur les données utilisées.
            """
        )


# ============================================================
# K-NEAREST NEIGHBORS (K-NN)
# ============================================================

if algorithm == "👥 K-NN":

    st.header("👥 K-Nearest Neighbors (K-NN)")

    st.markdown(
        """
        Le **K-Nearest Neighbors (K-NN)** est un algorithme de
        Machine Learning utilisé notamment pour la **classification**.

        Pour classer une nouvelle observation, K-NN :

        1. calcule sa distance avec les observations d'entraînement ;
        2. sélectionne les **K voisins les plus proches** ;
        3. examine leurs classes ;
        4. attribue la classe majoritaire.

        La distance utilisée ici est la **distance euclidienne** :

        $$
        d(x,x_i)
        =
        \\sqrt{
        \\sum_{j=1}^{p}
        (x_j-x_{ij})^2
        }
        $$

        **Plus deux points sont proches, plus ils sont considérés
        comme similaires.**
        """
    )

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.subheader("📊 Données de classification")

    knn_example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Classification simple",
            "Classification en deux groupes",
            "Classification multiclasses",
        ],
        key="knn_example",
    )

    if knn_example == "Classification simple":

        default_knn_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6"
        )

        default_knn_y = (
            "0,0,0,0,1,1,1,1"
        )

    elif knn_example == "Classification en deux groupes":

        default_knn_x = (
            "1,2; "
            "1,3; "
            "2,2; "
            "2,3; "
            "7,7; "
            "7,8; "
            "8,7; "
            "8,8"
        )

        default_knn_y = (
            "0,0,0,0,1,1,1,1"
        )

    elif knn_example == "Classification multiclasses":

        default_knn_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "5,5; "
            "5,6; "
            "6,5; "
            "9,9; "
            "9,10; "
            "10,9"
        )

        default_knn_y = (
            "0,0,0,1,1,1,2,2,2"
        )

    else:

        default_knn_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6"
        )

        default_knn_y = (
            "0,0,0,0,1,1,1,1"
        )

    # ========================================================
    # EXPLICATION DU FORMAT
    # ========================================================

    st.info(
        """
        **Format de X :**

        Chaque observation est séparée par `;` et les caractéristiques
        d'une observation par `,`.

        Exemple :

        `1,1; 1,2; 2,1; 2,2`

        correspond à :

        `[[1,1], [1,2], [2,1], [2,2]]`

        **Format de Y :**

        Les classes sont séparées par des virgules.

        Exemple :

        `0,0,0,1,1,1`
        """
    )

    # ========================================================
    # DONNÉES D'ENTRAÎNEMENT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        knn_x_input = st.text_area(
            "Caractéristiques X",
            value=default_knn_x,
            height=150,
            help=(
                "Séparez les observations par ';' et les "
                "caractéristiques par ','."
            ),
            key="knn_x",
        )

    with col2:

        knn_y_input = st.text_area(
            "Classes Y",
            value=default_knn_y,
            height=150,
            help=(
                "Entrez une classe pour chaque observation, "
                "séparée par des virgules."
            ),
            key="knn_y",
        )

    # ========================================================
    # DONNÉES DE TEST
    # ========================================================

    st.subheader("🔍 Observations à classifier")

    knn_test_input = st.text_area(
        "Nouvelles observations X",
        value="1.5,1.5; 5.5,5.5",
        height=100,
        help=(
            "Entrez les observations à classifier avec le même "
            "format que X."
        ),
        key="knn_test_x",
    )

    # ========================================================
    # PARAMÈTRE K
    # ========================================================

    st.subheader("⚙️ Paramètres")

    k_value = st.number_input(
        "Nombre de voisins K",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
        key="knn_k",
        help=(
            "K représente le nombre de voisins utilisés pour "
            "déterminer la classe."
        ),
    )

    # ========================================================
    # PARSING X
    # ========================================================

    def parse_knn_features(value: str) -> np.ndarray:
        """
        Convertit une chaîne de caractères en matrice NumPy.

        Format attendu :

        1,2; 2,3; 4,5

        devient :

        [[1, 2],
         [2, 3],
         [4, 5]]
        """

        if not value.strip():
            raise ValueError(
                "Les données X ne doivent pas être vides."
            )

        try:

            rows = []

            for row in value.split(";"):

                row = row.strip()

                if not row:
                    continue

                values = [
                    float(item.strip())
                    for item in row.split(",")
                    if item.strip()
                ]

                if not values:
                    raise ValueError(
                        "Une observation X est vide."
                    )

                rows.append(values)

            if not rows:
                raise ValueError(
                    "Aucune observation valide n'a été trouvée."
                )

            n_features = len(rows[0])

            if n_features == 0:
                raise ValueError(
                    "Chaque observation doit contenir "
                    "au moins une caractéristique."
                )

            for row in rows:

                if len(row) != n_features:
                    raise ValueError(
                        "Toutes les observations X doivent avoir "
                        "le même nombre de caractéristiques."
                    )

            return np.asarray(
                rows,
                dtype=float,
            )

        except ValueError as exc:

            if str(exc).startswith(
                (
                    "Une observation",
                    "Aucune observation",
                    "Chaque observation",
                    "Toutes les observations",
                )
            ):
                raise

            raise ValueError(
                "X doit contenir uniquement des nombres. "
                "Utilisez ';' entre les observations et ',' "
                "entre les caractéristiques."
            ) from exc

    # ========================================================
    # PARSING Y
    # ========================================================

    def parse_knn_labels(value: str) -> np.ndarray:
        """
        Convertit les classes Y en tableau NumPy.

        Les labels numériques sont convertis en nombres.
        Les labels textuels sont conservés.
        """

        if not value.strip():
            raise ValueError(
                "Les classes Y ne doivent pas être vides."
            )

        labels = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

        if not labels:
            raise ValueError(
                "Aucune classe valide n'a été trouvée."
            )

        # ----------------------------------------------------
        # Tentative de conversion numérique
        # ----------------------------------------------------

        try:

            numeric_labels = [
                float(label)
                for label in labels
            ]

            # Convertit 0.0 en 0, 1.0 en 1, etc.
            if all(
                value.is_integer()
                for value in numeric_labels
            ):
                return np.asarray(
                    numeric_labels,
                    dtype=int,
                )

            return np.asarray(
                numeric_labels,
                dtype=float,
            )

        except ValueError:

            # Les labels textuels restent des chaînes.
            return np.asarray(
                labels,
                dtype=str,
            )

    # ========================================================
    # EXECUTION
    # ========================================================

    if st.button(
        "🚀 Exécuter K-NN",
        type="primary",
        use_container_width=True,
        key="knn_execute",
    ):

        try:

            X_train = parse_knn_features(
                knn_x_input
            )

            y_train = parse_knn_labels(
                knn_y_input
            )

            X_test = parse_knn_features(
                knn_test_input
            )

            # ------------------------------------------------
            # VALIDATION INTERFACE
            # ------------------------------------------------

            if len(X_train) != len(y_train):

                raise ValueError(
                    "X et Y doivent contenir le même nombre "
                    "d'observations."
                )

            if X_train.shape[1] != X_test.shape[1]:

                raise ValueError(
                    "Les données d'entraînement et les données "
                    "de test doivent avoir le même nombre "
                    "de caractéristiques."
                )

            if int(k_value) > len(X_train):

                raise ValueError(
                    "K ne peut pas être supérieur au nombre "
                    "d'observations d'entraînement."
                )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            result = knn(
                X_train,
                y_train,
                X_test,
                k=int(k_value),
            )

            # ------------------------------------------------
            # SESSION STATE
            # ------------------------------------------------

            st.session_state[
                "knn_result"
            ] = result

            st.session_state[
                "knn_X_train"
            ] = X_train

            st.session_state[
                "knn_y_train"
            ] = y_train

            st.session_state[
                "knn_X_test"
            ] = X_test

            st.session_state[
                "knn_k_value"
            ] = int(k_value)

        except (ValueError, TypeError) as exc:

            st.error(
                f"❌ {exc}"
            )


# ============================================================
# K-NN RESULTS
# ============================================================

if (
    algorithm == "👥 K-NN"
    and "knn_result" in st.session_state
):

    result = st.session_state[
        "knn_result"
    ]

    X_train = st.session_state[
        "knn_X_train"
    ]

    y_train = st.session_state[
        "knn_y_train"
    ]

    X_test = st.session_state[
        "knn_X_test"
    ]

    k_value = st.session_state[
        "knn_k_value"
    ]

    st.divider()

    st.header("📊 Résultats — K-NN")

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "K",
            str(result["k"]),
        )

    with col2:

        st.metric(
            "Observations entraînement",
            str(result["n_train_samples"]),
        )

    with col3:

        st.metric(
            "Observations test",
            str(result["n_test_samples"]),
        )

    with col4:

        st.metric(
            "Caractéristiques",
            str(result["n_features"]),
        )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.subheader("🔢 Prédictions")

    predictions = result[
        "predictions"
    ]

    prediction_rows = []

    for i, (features, prediction) in enumerate(
        zip(X_test, predictions)
    ):

        row = {
            "Observation": i + 1,
        }

        for feature_index, feature_value in enumerate(
            features
        ):

            row[
                f"X{feature_index + 1}"
            ] = feature_value

        row["Classe prédite"] = prediction

        prediction_rows.append(row)

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    st.subheader("📈 Visualisation des données")

    if plt is None:

        st.warning(
            "La visualisation nécessite Matplotlib. "
            "Installez-le avec `pip install matplotlib`."
        )

    elif X_train.shape[1] == 2:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        # ----------------------------------------------------
        # Classes présentes
        # ----------------------------------------------------

        classes = np.unique(
            y_train
        )

        for class_value in classes:

            mask = (
                y_train == class_value
            )

            ax.scatter(
                X_train[mask, 0],
                X_train[mask, 1],
                label=f"Classe {class_value}",
                s=70,
            )

        # ----------------------------------------------------
        # Observations de test
        # ----------------------------------------------------

        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            marker="*",
            s=180,
            label="Nouvelles observations",
        )

        for i, prediction in enumerate(
            predictions
        ):

            ax.annotate(
                f"→ {prediction}",
                (
                    X_test[i, 0],
                    X_test[i, 1],
                ),
                xytext=(8, 8),
                textcoords="offset points",
            )

        ax.set_xlabel(
            "Caractéristique X₁"
        )

        ax.set_ylabel(
            "Caractéristique X₂"
        )

        ax.set_title(
            f"K-NN — K = {k_value}"
        )

        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    else:

        st.info(
            "La visualisation graphique 2D est disponible "
            "lorsque X contient exactement deux caractéristiques."
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("💡 Interprétation")

    st.markdown(
        f"""
        **Nombre de voisins :** `{k_value}`

        Pour chaque nouvelle observation, K-NN recherche les
        **{k_value} observations d'entraînement les plus proches**
        selon la distance euclidienne.

        La classe prédite correspond à la **classe majoritaire**
        parmi ces voisins.

        **Nombre d'observations d'entraînement :**
        `{result["n_train_samples"]}`

        **Nombre d'observations à classifier :**
        `{result["n_test_samples"]}`

        **Nombre de caractéristiques :**
        `{result["n_features"]}`
        """
    )

    # ========================================================
    # MATHEMATICAL EXPLANATION
    # ========================================================

    with st.expander(
        "📚 Comprendre la méthode mathématique"
    ):

        st.markdown(
            r"""
            ### 1. Distance euclidienne

            Pour une nouvelle observation $x$ et une observation
            d'entraînement $x_i$ :

            $$
            d(x,x_i)
            =
            \sqrt{
            \sum_{j=1}^{p}
            (x_j-x_{ij})^2
            }
            $$

            Pour deux caractéristiques :

            $$
            d(x,x_i)
            =
            \sqrt{
            (x_1-x_{i1})^2
            +
            (x_2-x_{i2})^2
            }
            $$

            ---

            ### 2. Sélection des voisins

            Après avoir calculé toutes les distances, on les trie
            de la plus petite à la plus grande.

            On conserve uniquement les **K plus petites distances**.

            ---

            ### 3. Vote majoritaire

            Supposons que les trois voisins les plus proches aient
            les classes :

            $$
            [0,0,1]
            $$

            La classe `0` apparaît deux fois.

            Donc :

            $$
            \hat{y}=0
            $$

            ---

            ### 4. Exemple

            Pour :

            $$
            K=3
            $$

            et les classes des voisins :

            $$
            [1,1,0]
            $$

            on obtient :

            $$
            \text{classe 1} = 2
            $$

            $$
            \text{classe 0} = 1
            $$

            La prédiction est donc :

            $$
            \boxed{\hat{y}=1}
            $$

            ---

            ### 5. Influence de K

            Un petit $K$ rend le modèle très sensible aux observations
            proches.

            Un grand $K$ prend en compte davantage d'observations.

            Le choix de $K$ peut donc modifier les prédictions.

            ---

            ### 6. Remarque importante

            K-NN repose directement sur les distances.

            Les variables ayant des échelles très différentes peuvent
            donc influencer fortement le calcul de distance.

            En pratique, une **normalisation ou standardisation des
            données** est souvent utilisée avant K-NN.
            """
        )


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

if algorithm == "🧩 K-Means":

    st.header("🧩 K-Means Clustering")

    st.markdown(
        """
        Le **K-Means** est un algorithme d'apprentissage
        **non supervisé** utilisé pour regrouper les observations
        en différents clusters.

        L'objectif est de répartir les observations en **K groupes**
        de manière à ce que les observations d'un même groupe soient
        aussi proches que possible de leur centroïde.

        Le fonctionnement général est :

        1. choisir le nombre de clusters $K$ ;
        2. initialiser les centroïdes ;
        3. affecter chaque observation au centroïde le plus proche ;
        4. recalculer les centroïdes ;
        5. répéter jusqu'à convergence.

        La distance utilisée est la **distance euclidienne** :

        $$
        d(x,c_j)
        =
        \\sqrt{
        \\sum_{k=1}^{p}
        (x_k-c_{jk})^2
        }
        $$

        où $x$ représente une observation et $c_j$ un centroïde.
        """
    )

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.subheader("📊 Données")

    kmeans_example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Deux groupes",
            "Trois groupes",
            "Données avec trois caractéristiques",
        ],
        key="kmeans_example",
    )

    if kmeans_example == "Deux groupes":

        default_kmeans_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "8,8; "
            "8,9; "
            "9,8; "
            "9,9"
        )

        default_kmeans_clusters = 2

    elif kmeans_example == "Trois groupes":

        default_kmeans_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6; "
            "9,1; "
            "9,2; "
            "10,1; "
            "10,2"
        )

        default_kmeans_clusters = 3

    elif kmeans_example == "Données avec trois caractéristiques":

        default_kmeans_x = (
            "1,1,1; "
            "1,2,1; "
            "2,1,2; "
            "2,2,1; "
            "8,8,8; "
            "8,9,8; "
            "9,8,9; "
            "9,9,8"
        )

        default_kmeans_clusters = 2

    else:

        default_kmeans_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "8,8; "
            "8,9; "
            "9,8; "
            "9,9"
        )

        default_kmeans_clusters = 2

    # ========================================================
    # FORMAT DES DONNÉES
    # ========================================================

    st.info(
        """
        **Format de X :**

        Séparez les observations par `;` et les caractéristiques
        par `,`.

        Exemple :

        `1,1; 1,2; 2,1; 2,2`

        correspond à :

        `[[1,1], [1,2], [2,1], [2,2]]`

        Pour trois caractéristiques :

        `1,1,2; 2,3,1; 5,5,6`
        """
    )

    # ========================================================
    # DONNÉES D'ENTRÉE
    # ========================================================

    st.subheader("📥 Données d'apprentissage")

    kmeans_x_input = st.text_area(
        "Observations X",
        value=default_kmeans_x,
        height=150,
        help=(
            "Séparez les observations par ';' et les "
            "caractéristiques par ','."
        ),
        key="kmeans_x",
    )

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.subheader("⚙️ Paramètres K-Means")

    parameter_col1, parameter_col2, parameter_col3 = st.columns(3)

    with parameter_col1:

        kmeans_n_clusters = st.number_input(
            "Nombre de clusters K",
            min_value=1,
            max_value=100,
            value=default_kmeans_clusters,
            step=1,
            key="kmeans_n_clusters",
            help=(
                "Nombre de groupes que l'algorithme "
                "doit créer."
            ),
        )

    with parameter_col2:

        kmeans_max_iterations = st.number_input(
            "Nombre maximal d'itérations",
            min_value=1,
            max_value=10000,
            value=100,
            step=10,
            key="kmeans_max_iterations",
        )

    with parameter_col3:

        kmeans_tolerance = st.number_input(
            "Tolérance de convergence",
            min_value=0.0,
            max_value=10.0,
            value=0.0001,
            step=0.0001,
            format="%.6f",
            key="kmeans_tolerance",
            help=(
                "L'algorithme s'arrête lorsque le déplacement "
                "des centroïdes devient inférieur à cette valeur."
            ),
        )

    kmeans_random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=100000,
        value=42,
        step=1,
        key="kmeans_random_state",
        help=(
            "Permet d'obtenir des résultats reproductibles."
        ),
    )

    # ========================================================
    # PARSING
    # ========================================================

    def parse_kmeans_features(value: str) -> np.ndarray:
        """
        Convertit les données saisies dans Streamlit
        en matrice NumPy.

        Format :

        1,2; 2,3; 4,5

        devient :

        [[1, 2],
         [2, 3],
         [4, 5]]
        """

        if not value.strip():

            raise ValueError(
                "Les données X ne doivent pas être vides."
            )

        try:

            rows = []

            for row in value.split(";"):

                row = row.strip()

                if not row:
                    continue

                values = [
                    float(item.strip())
                    for item in row.split(",")
                    if item.strip()
                ]

                if not values:

                    raise ValueError(
                        "Une observation X est vide."
                    )

                rows.append(values)

            if not rows:

                raise ValueError(
                    "Aucune observation valide n'a été trouvée."
                )

            n_features = len(rows[0])

            if n_features == 0:

                raise ValueError(
                    "Chaque observation doit contenir "
                    "au moins une caractéristique."
                )

            for row in rows:

                if len(row) != n_features:

                    raise ValueError(
                        "Toutes les observations doivent avoir "
                        "le même nombre de caractéristiques."
                    )

            X = np.asarray(
                rows,
                dtype=float,
            )

            if not np.all(np.isfinite(X)):

                raise ValueError(
                    "Les données contiennent des valeurs "
                    "non finies."
                )

            return X

        except ValueError as exc:

            message = str(exc)

            if message.startswith(
                (
                    "Une observation",
                    "Aucune observation",
                    "Chaque observation",
                    "Toutes les observations",
                    "Les données contiennent",
                )
            ):

                raise

            raise ValueError(
                "X doit contenir uniquement des nombres. "
                "Utilisez ';' entre les observations et ',' "
                "entre les caractéristiques."
            ) from exc


    # ========================================================
    # EXECUTION
    # ========================================================

    if st.button(
        "🚀 Exécuter K-Means",
        type="primary",
        use_container_width=True,
        key="kmeans_execute",
    ):

        try:

            X = parse_kmeans_features(
                kmeans_x_input
            )

            n_clusters = int(
                kmeans_n_clusters
            )

            max_iterations = int(
                kmeans_max_iterations
            )

            tolerance = float(
                kmeans_tolerance
            )

            random_state = int(
                kmeans_random_state
            )

            # ------------------------------------------------
            # VALIDATION INTERFACE
            # ------------------------------------------------

            if len(X) < n_clusters:

                raise ValueError(
                    "Le nombre de clusters K ne peut pas "
                    "être supérieur au nombre d'observations."
                )

            if n_clusters < 1:

                raise ValueError(
                    "K doit être supérieur ou égal à 1."
                )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            result = kmeans(
                X,
                n_clusters=n_clusters,
                max_iterations=max_iterations,
                tolerance=tolerance,
                random_state=random_state,
            )

            # ------------------------------------------------
            # SESSION STATE
            # ------------------------------------------------

            st.session_state[
                "kmeans_result"
            ] = result

            st.session_state[
                "kmeans_X"
            ] = X

        except (ValueError, TypeError) as exc:

            st.error(
                f"❌ {exc}"
            )






# ============================================================
# K-MEANS RESULTS
# ============================================================

if (
    algorithm == "🧩 K-Means"
    and "kmeans_result" in st.session_state
):

    result = st.session_state[
        "kmeans_result"
    ]

    X = st.session_state[
        "kmeans_X"
    ]

    n_clusters = st.session_state[
        "kmeans_n_clusters"
    ]

    st.divider()

    st.header("📊 Résultats — K-Means")

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Clusters K",
            str(result["n_clusters"]),
        )

    with col2:

        st.metric(
            "Observations",
            str(result["n_samples"]),
        )

    with col3:

        st.metric(
            "Itérations",
            str(result["n_iterations"]),
        )

    with col4:

        st.metric(
            "Inertie",
            f"{result['inertia']:.6f}",
        )

    # ========================================================
    # CONVERGENCE
    # ========================================================

    if result["converged"]:

        st.success(
            "✅ L'algorithme a convergé."
        )

    else:

        st.warning(
            "⚠️ L'algorithme a atteint le nombre maximal "
            "d'itérations sans satisfaire le critère "
            "de convergence."
        )

    # ========================================================
    # CENTROÏDES
    # ========================================================

    st.subheader("🎯 Centroïdes")

    centroids = result[
        "centroids"
    ]

    centroid_data = {
        "Cluster": [
            f"Cluster {i}"
            for i in range(n_clusters)
        ]
    }

    for feature_index in range(
        centroids.shape[1]
    ):

        centroid_data[
            f"X{feature_index + 1}"
        ] = centroids[
            :, feature_index
        ]

    centroid_df = pd.DataFrame(
        centroid_data
    )

    st.dataframe(
        centroid_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # LABELS DES OBSERVATIONS
    # ========================================================

    st.subheader("🔢 Attribution des clusters")

    labels = result[
        "labels"
    ]

    cluster_data = {
        "Observation": np.arange(
            1,
            len(X) + 1,
        ),
    }

    for feature_index in range(
        X.shape[1]
    ):

        cluster_data[
            f"X{feature_index + 1}"
        ] = X[
            :, feature_index
        ]

    cluster_data[
        "Cluster"
    ] = labels

    cluster_df = pd.DataFrame(
        cluster_data
    )

    st.dataframe(
        cluster_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # NOMBRE D'OBSERVATIONS PAR CLUSTER
    # ========================================================

    st.subheader("📦 Taille des clusters")

    cluster_counts = np.bincount(
        labels,
        minlength=n_clusters,
    )

    cluster_count_df = pd.DataFrame(
        {
            "Cluster": [
                f"Cluster {i}"
                for i in range(n_clusters)
            ],
            "Nombre d'observations": cluster_counts,
        }
    )

    st.dataframe(
        cluster_count_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # VISUALISATION 2D
    # ========================================================

    st.subheader("📈 Visualisation des clusters")

    if plt is None:

        st.warning(
            "La visualisation nécessite Matplotlib. "
            "Installez-le avec `pip install matplotlib`."
        )

    elif X.shape[1] == 2:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        # ----------------------------------------------------
        # OBSERVATIONS
        # ----------------------------------------------------

        for cluster_index in range(
            n_clusters
        ):

            mask = (
                labels == cluster_index
            )

            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                s=80,
                label=(
                    f"Cluster {cluster_index}"
                ),
            )

        # ----------------------------------------------------
        # CENTROÏDES
        # ----------------------------------------------------

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=220,
            label="Centroïdes",
        )

        # ----------------------------------------------------
        # LABELS DES CENTROÏDES
        # ----------------------------------------------------

        for cluster_index, centroid in enumerate(
            centroids
        ):

            ax.annotate(
                f"C{cluster_index}",
                (
                    centroid[0],
                    centroid[1],
                ),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xlabel(
            "Caractéristique X₁"
        )

        ax.set_ylabel(
            "Caractéristique X₂"
        )

        ax.set_title(
            f"K-Means — {n_clusters} clusters"
        )

        ax.legend()

        ax.grid(
            alpha=0.3
        )

        st.pyplot(fig)

    else:

        st.info(
            "La visualisation graphique 2D est disponible "
            "lorsque les données contiennent exactement "
            "deux caractéristiques."
        )

    # ========================================================
    # INTERPRÉTATION
    # ========================================================

    st.subheader("💡 Interprétation")

    st.markdown(
        f"""
        **Nombre de clusters :** `{n_clusters}`

        K-Means cherche à répartir les `{result["n_samples"]}`
        observations en **{n_clusters} groupes**.

        **Inertie :** `{result["inertia"]:.6f}`

        L'inertie correspond à la somme des distances
        **au carré** entre chaque observation et le centroïde
        du cluster auquel elle appartient.

        Une observation est affectée au cluster dont le
        centroïde est le plus proche selon la distance
        euclidienne.

        **Nombre d'itérations :**
        `{result["n_iterations"]}`

        **Convergence :**
        `{"Oui" if result["converged"] else "Non"}`
        """
    )

    # ========================================================
    # EXPLICATION MATHÉMATIQUE
    # ========================================================

    with st.expander(
        "📚 Comprendre la méthode mathématique"
    ):

        st.markdown(
            r"""
            ### 1. Initialisation

            On choisit un nombre de clusters :

            $$
            K
            $$

            Puis on initialise $K$ centroïdes.

            Chaque centroïde possède le même nombre de
            caractéristiques que les observations.

            ---

            ### 2. Distance euclidienne

            Pour une observation $x_i$ et un centroïde $c_j$ :

            $$
            d(x_i,c_j)
            =
            \sqrt{
            \sum_{k=1}^{p}
            (x_{ik}-c_{jk})^2
            }
            $$

            où :

            - $p$ est le nombre de caractéristiques ;
            - $x_{ik}$ est la caractéristique $k$ de
              l'observation $i$ ;
            - $c_{jk}$ est la caractéristique $k$ du
              centroïde $j$.

            ---

            ### 3. Attribution des observations

            Chaque observation est affectée au centroïde
            le plus proche :

            $$
            label_i
            =
            \arg\min_j d(x_i,c_j)
            $$

            ---

            ### 4. Recalcul des centroïdes

            Pour chaque cluster, le nouveau centroïde
            correspond à la moyenne de ses observations :

            $$
            c_j
            =
            \frac{1}{|C_j|}
            \sum_{x_i \in C_j}x_i
            $$

            ---

            ### 5. Fonction objectif

            K-Means cherche à minimiser l'inertie :

            $$
            J
            =
            \sum_{j=1}^{K}
            \sum_{x_i\in C_j}
            \|x_i-c_j\|^2
            $$

            Cette quantité mesure la dispersion des observations
            autour de leur centroïde.

            ---

            ### 6. Convergence

            L'algorithme s'arrête lorsque le déplacement
            des centroïdes devient suffisamment petit :

            $$
            \|C_{nouveau}-C_{ancien}\|
            \leq tolerance
            $$

            ou lorsque le nombre maximal d'itérations
            est atteint.

            ---

            ### 7. Attention au choix de K

            Le nombre de clusters $K$ doit être choisi
            avant l'exécution de K-Means.

            Une méthode courante pour étudier ce choix est
            la **méthode du coude (Elbow Method)**, qui consiste
            à observer l'évolution de l'inertie pour plusieurs
            valeurs de $K$.

            ---

            ### 8. Importance de l'échelle

            K-Means utilise des distances.

            Si les caractéristiques ont des échelles très
            différentes, certaines variables peuvent donc
            avoir une influence disproportionnée sur les
            distances.

            Une standardisation des données peut alors être
            utilisée avant l'application de K-Means.
            """
        )


# ============================================================
# DECISION TREE
# ============================================================

if algorithm == "🌳 Decision Tree":

    st.header("🌳 Decision Tree")

    st.markdown(
        """
        Le **Decision Tree** est un algorithme de Machine Learning
        supervisé utilisé pour la **classification**.

        L'algorithme construit progressivement un arbre de décision
        en divisant les observations selon les caractéristiques
        qui permettent de mieux séparer les classes.

        À chaque nœud, l'algorithme recherche une division qui
        minimise l'impureté de **Gini**.

        Le principe général est :

        1. choisir une caractéristique ;
        2. choisir un seuil ;
        3. séparer les observations ;
        4. calculer l'impureté de Gini ;
        5. conserver la meilleure séparation ;
        6. répéter récursivement jusqu'à atteindre une condition d'arrêt.
        """
    )

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.subheader("📊 Données de classification")

    decision_tree_example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Classification simple",
            "Deux groupes",
            "Classification multiclasses",
        ],
        key="decision_tree_example",
    )

    if decision_tree_example == "Classification simple":

        default_dt_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6"
        )

        default_dt_y = "0,0,0,0,1,1,1,1"
        default_dt_test_x = "1.5,1.5; 5.5,5.5"
        default_dt_max_depth = 3
        default_dt_min_samples_split = 2

    elif decision_tree_example == "Deux groupes":

        default_dt_x = (
            "1,2; "
            "1,3; "
            "2,2; "
            "2,3; "
            "7,7; "
            "7,8; "
            "8,7; "
            "8,8"
        )

        default_dt_y = "0,0,0,0,1,1,1,1"
        default_dt_test_x = "1.5,2.5; 7.5,7.5"
        default_dt_max_depth = 3
        default_dt_min_samples_split = 2

    elif decision_tree_example == "Classification multiclasses":

        default_dt_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6; "
            "9,9; "
            "9,10; "
            "10,9; "
            "10,10"
        )

        default_dt_y = "0,0,0,0,1,1,1,1,2,2,2,2"
        default_dt_test_x = "1.5,1.5; 5.5,5.5; 9.5,9.5"
        default_dt_max_depth = 4
        default_dt_min_samples_split = 2

    else:

        default_dt_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6"
        )

        default_dt_y = "0,0,0,0,1,1,1,1"
        default_dt_test_x = "1.5,1.5; 5.5,5.5"
        default_dt_max_depth = 3
        default_dt_min_samples_split = 2

    # ========================================================
    # FORMAT DES DONNÉES
    # ========================================================

    st.info(
        """
        **Format de X :**

        Séparez les observations par `;` et les caractéristiques
        par `,`.

        Exemple :

        `1,1; 1,2; 2,1; 2,2`

        correspond à :

        `[[1,1], [1,2], [2,1], [2,2]]`

        **Format de Y :**

        Une classe par observation, séparée par des virgules.

        Exemple :

        `0,0,0,0,1,1,1,1`

        Les classes peuvent être numériques ou textuelles.
        """
    )

    # ========================================================
    # DONNÉES D'ENTRAÎNEMENT
    # ========================================================

    st.subheader("📥 Données d'entraînement")

    col1, col2 = st.columns(2)

    with col1:

        decision_tree_x_input = st.text_area(
            "Caractéristiques X",
            value=default_dt_x,
            height=150,
            help=(
                "Séparez les observations par ';' et les "
                "caractéristiques par ','."
            ),
            key="decision_tree_x",
        )

    with col2:

        decision_tree_y_input = st.text_area(
            "Classes Y",
            value=default_dt_y,
            height=150,
            help=(
                "Entrez une classe pour chaque observation, "
                "séparée par des virgules."
            ),
            key="decision_tree_y",
        )

    # ========================================================
    # DONNÉES DE TEST
    # ========================================================

    st.subheader("🔍 Observations à classifier")

    decision_tree_test_input = st.text_area(
        "Nouvelles observations X",
        value=default_dt_test_x,
        height=100,
        help=(
            "Entrez les nouvelles observations avec le même "
            "nombre de caractéristiques que X."
        ),
        key="decision_tree_test_x",
    )

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.subheader("⚙️ Paramètres de l'arbre")

    parameter_col1, parameter_col2 = st.columns(2)

    with parameter_col1:

        decision_tree_max_depth = st.number_input(
            "Profondeur maximale",
            min_value=0,
            max_value=100,
            value=default_dt_max_depth,
            step=1,
            key="decision_tree_max_depth",
            help=(
                "Limite la profondeur maximale de l'arbre. "
                "0 correspond à un arbre contenant uniquement "
                "la racine."
            ),
        )

    with parameter_col2:

        decision_tree_min_samples_split = st.number_input(
            "Minimum d'observations pour diviser un nœud",
            min_value=2,
            max_value=1000,
            value=default_dt_min_samples_split,
            step=1,
            key="decision_tree_min_samples_split",
            help=(
                "Un nœud doit contenir au moins ce nombre "
                "d'observations pour pouvoir être divisé."
            ),
        )

    # ========================================================
    # PARSING X
    # ========================================================

    def parse_decision_tree_features(value: str) -> np.ndarray:

        if not value.strip():

            raise ValueError(
                "Les données X ne doivent pas être vides."
            )

        try:

            rows = []

            for row in value.split(";"):

                row = row.strip()

                if not row:
                    continue

                values = [
                    float(item.strip())
                    for item in row.split(",")
                    if item.strip()
                ]

                if not values:

                    raise ValueError(
                        "Une observation X est vide."
                    )

                rows.append(values)

            if not rows:

                raise ValueError(
                    "Aucune observation valide n'a été trouvée."
                )

            n_features = len(rows[0])

            if n_features == 0:

                raise ValueError(
                    "Chaque observation doit contenir "
                    "au moins une caractéristique."
                )

            for row in rows:

                if len(row) != n_features:

                    raise ValueError(
                        "Toutes les observations X doivent avoir "
                        "le même nombre de caractéristiques."
                    )

            X = np.asarray(
                rows,
                dtype=float,
            )

            if not np.all(np.isfinite(X)):

                raise ValueError(
                    "Les données contiennent des valeurs "
                    "non finies."
                )

            return X

        except ValueError as exc:

            message = str(exc)

            if message.startswith(
                (
                    "Une observation",
                    "Aucune observation",
                    "Chaque observation",
                    "Toutes les observations",
                    "Les données contiennent",
                )
            ):

                raise

            raise ValueError(
                "X doit contenir uniquement des nombres. "
                "Utilisez ';' entre les observations et ',' "
                "entre les caractéristiques."
            ) from exc

    # ========================================================
    # PARSING Y
    # ========================================================

    def parse_decision_tree_labels(value: str) -> np.ndarray:

        if not value.strip():

            raise ValueError(
                "Les classes Y ne doivent pas être vides."
            )

        labels = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

        if not labels:

            raise ValueError(
                "Aucune classe valide n'a été trouvée."
            )

        # ----------------------------------------------------
        # Tentative de conversion numérique
        # ----------------------------------------------------

        try:

            numeric_labels = [
                float(label)
                for label in labels
            ]

            if all(
                value.is_integer()
                for value in numeric_labels
            ):

                return np.asarray(
                    numeric_labels,
                    dtype=int,
                )

            return np.asarray(
                numeric_labels,
                dtype=float,
            )

        except ValueError:

            return np.asarray(
                labels,
                dtype=str,
            )

    # ========================================================
    # EXECUTION
    # ========================================================

    if st.button(
        "🚀 Construire l'arbre",
        type="primary",
        use_container_width=True,
        key="decision_tree_execute",
    ):

        try:

            # ------------------------------------------------
            # PARSING
            # ------------------------------------------------

            X_train = parse_decision_tree_features(
                decision_tree_x_input
            )

            y_train = parse_decision_tree_labels(
                decision_tree_y_input
            )

            X_test = parse_decision_tree_features(
                decision_tree_test_input
            )

            max_depth = int(
                decision_tree_max_depth
            )

            min_samples_split = int(
                decision_tree_min_samples_split
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if len(X_train) != len(y_train):

                raise ValueError(
                    "X et Y doivent contenir le même nombre "
                    "d'observations."
                )

            if X_train.shape[1] != X_test.shape[1]:

                raise ValueError(
                    "Les données d'entraînement et les données "
                    "de test doivent avoir le même nombre "
                    "de caractéristiques."
                )

            if len(X_train) == 0:

                raise ValueError(
                    "Les données d'entraînement ne doivent "
                    "pas être vides."
                )

            if len(y_train) == 0:

                raise ValueError(
                    "Au moins une classe est nécessaire."
                )

            # ------------------------------------------------
            # CONSTRUCTION DU MODÈLE
            # ------------------------------------------------

            result = decision_tree(
                X_train,
                y_train,
                X_test,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )

            # ------------------------------------------------
            # SAUVEGARDE
            # ------------------------------------------------

            st.session_state[
                "decision_tree_result"
            ] = result

            st.session_state[
                "decision_tree_X_train"
            ] = X_train

            st.session_state[
                "decision_tree_y_train"
            ] = y_train

            st.session_state[
                "decision_tree_X_test"
            ] = X_test

            st.success(
                "✅ Arbre de décision construit avec succès."
            )

        except (ValueError, TypeError) as exc:

            st.error(
                f"❌ {exc}"
            )


# ============================================================
# FONCTION D'AFFICHAGE DE L'ARBRE
# ============================================================

def format_decision_tree(node, depth=0):
    """
    Formate récursivement l'arbre de décision.

    Compatible avec TreeNode.

    Attributs utilisés :
        - prediction
        - feature_index
        - threshold
        - left
        - right
    """

    indent = "    " * depth

    # ========================================================
    # FEUILLE
    # ========================================================

    if node.prediction is not None:

        return (
            f"{indent}└── Feuille : "
            f"classe = {node.prediction}\n"
        )

    # ========================================================
    # NŒUD INTERNE
    # ========================================================

    text = (
        f"{indent}├── "
        f"X[{node.feature_index}] "
        f"<= {node.threshold:.4f}\n"
    )

    # ========================================================
    # BRANCHE GAUCHE
    # ========================================================

    if node.left is not None:

        text += (
            f"{indent}│   ├── Gauche :\n"
        )

        text += format_decision_tree(
            node.left,
            depth + 2,
        )

    # ========================================================
    # BRANCHE DROITE
    # ========================================================

    if node.right is not None:

        text += (
            f"{indent}│   └── Droite :\n"
        )

        text += format_decision_tree(
            node.right,
            depth + 2,
        )

    return text


# ============================================================
# DECISION TREE RESULTS
# ============================================================

if (
    algorithm == "🌳 Decision Tree"
    and "decision_tree_result" in st.session_state
):

    result = st.session_state[
        "decision_tree_result"
    ]

    X_train = st.session_state[
        "decision_tree_X_train"
    ]

    y_train = st.session_state[
        "decision_tree_y_train"
    ]

    X_test = st.session_state[
        "decision_tree_X_test"
    ]

    st.divider()

    st.header(
        "📊 Résultats — Decision Tree"
    )

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            (
                f"{result['accuracy']:.2%}"
                if result["accuracy"] is not None
                else "N/A"
            ),
        )

    with col2:

        st.metric(
            "Profondeur",
            str(result["tree_depth"]),
        )

    with col3:

        st.metric(
            "Nombre de nœuds",
            str(result["n_nodes"]),
        )

    with col4:

        st.metric(
            "Classes",
            str(result["n_classes"]),
        )

    with col5:

        st.metric(
            "Caractéristiques",
            str(result["n_features"]),
        )

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.subheader(
        "⚙️ Paramètres utilisés"
    )

    parameter_df = pd.DataFrame(
        {
            "Paramètre": [
                "Profondeur maximale",
                "Minimum observations / split",
                "Observations entraînement",
                "Observations test",
            ],
            "Valeur": [
                result["max_depth"],
                result["min_samples_split"],
                result["n_train_samples"],
                result["n_test_samples"],
            ],
        }
    )

    st.dataframe(
        parameter_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.subheader("🔢 Prédictions")

    predictions = result[
        "predictions"
    ]

    prediction_rows = []

    for i, (features, prediction) in enumerate(
        zip(X_test, predictions)
    ):

        row = {
            "Observation": i + 1,
        }

        for feature_index, feature_value in enumerate(
            features
        ):

            row[
                f"X{feature_index + 1}"
            ] = feature_value

        row[
            "Classe prédite"
        ] = prediction

        prediction_rows.append(row)

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # VISUALISATION
    # ========================================================

    st.subheader(
        "📈 Visualisation des données"
    )

    if plt is None:

        st.warning(
            "La visualisation nécessite Matplotlib. "
            "Installez-le avec `pip install matplotlib`."
        )

    elif X_train.shape[1] == 2:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        classes = np.unique(
            y_train
        )

        for class_value in classes:

            mask = (
                y_train == class_value
            )

            ax.scatter(
                X_train[mask, 0],
                X_train[mask, 1],
                s=80,
                label=f"Classe {class_value}",
            )

        # ----------------------------------------------------
        # OBSERVATIONS DE TEST
        # ----------------------------------------------------

        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            marker="*",
            s=200,
            label="Nouvelles observations",
        )

        for i, prediction in enumerate(
            predictions
        ):

            ax.annotate(
                f"→ {prediction}",
                (
                    X_test[i, 0],
                    X_test[i, 1],
                ),
                xytext=(8, 8),
                textcoords="offset points",
            )

        ax.set_xlabel(
            "Caractéristique X₁"
        )

        ax.set_ylabel(
            "Caractéristique X₂"
        )

        ax.set_title(
            "Decision Tree — Classification"
        )

        ax.legend()

        ax.grid(
            alpha=0.3
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "La visualisation graphique 2D est disponible "
            "lorsque les données contiennent exactement "
            "deux caractéristiques."
        )

    # ========================================================
    # STRUCTURE DE L'ARBRE
    # ========================================================

    st.subheader(
        "🌳 Structure de l'arbre"
    )

    tree_text = format_decision_tree(
        result["tree"]
    )

    st.code(
        tree_text,
        language="text",
    )

    # ========================================================
    # INTERPRÉTATION
    # ========================================================

    st.subheader(
        "💡 Interprétation"
    )

    st.markdown(
        f"""
        **Profondeur réelle de l'arbre :**

        `{result["tree_depth"]}`

        L'arbre contient **{result["n_nodes"]} nœuds**.

        **Nombre de classes :**

        `{result["n_classes"]}`

        Le Decision Tree construit des règles successives
        permettant de séparer les différentes classes.

        À chaque nœud interne, une caractéristique et un seuil
        sont utilisés pour diviser les observations.

        **Condition de séparation :**

        Une observation est envoyée vers une branche lorsque :

        $$
        X_j \\leq seuil
        $$

        et vers l'autre branche lorsque :

        $$
        X_j > seuil
        $$
        """
    )

    if result["accuracy"] is not None:

        st.markdown(
            f"""
            **Accuracy sur les données d'entraînement :**

            `{result["accuracy"]:.2%}`

            Cette valeur représente la proportion d'observations
            d'entraînement correctement classées par l'arbre.
            """
        )

    # ========================================================
    # EXPLICATION MATHÉMATIQUE
    # ========================================================

    with st.expander(
        "📚 Comprendre la méthode mathématique"
    ):

        st.markdown(
            r"""
            ### 1. Impureté de Gini

            Pour un ensemble de données contenant plusieurs classes,
            l'impureté de Gini est :

            $$
            Gini =
            1 -
            \sum_{k=1}^{K}p_k^2
            $$

            où $p_k$ représente la proportion d'observations
            appartenant à la classe $k$.

            ---

            ### 2. Cas d'une classe pure

            Si toutes les observations appartiennent à la même
            classe :

            $$
            p_1=1
            $$

            alors :

            $$
            Gini = 1-1^2=0
            $$

            Le nœud est donc parfaitement pur.

            ---

            ### 3. Recherche d'une séparation

            L'arbre teste différentes caractéristiques et différents
            seuils.

            Pour une séparation donnée, on calcule :

            $$
            Gini_{split}
            =
            \frac{n_L}{n}Gini_L
            +
            \frac{n_R}{n}Gini_R
            $$

            où :

            - $n_L$ est le nombre d'observations à gauche ;
            - $n_R$ est le nombre d'observations à droite ;
            - $n$ est le nombre total d'observations ;
            - $Gini_L$ est l'impureté du groupe gauche ;
            - $Gini_R$ est l'impureté du groupe droit.

            ---

            ### 4. Meilleure séparation

            L'algorithme recherche la séparation qui minimise :

            $$
            Gini_{split}
            $$

            Cela permet de créer des groupes contenant autant que
            possible des observations appartenant aux mêmes classes.

            ---

            ### 5. Construction récursive

            Après avoir trouvé la meilleure séparation, le même
            processus est appliqué aux deux sous-groupes :

            $$
            X_j \leq seuil
            $$

            et :

            $$
            X_j > seuil
            $$

            Le processus continue récursivement.

            ---

            ### 6. Conditions d'arrêt

            La construction peut s'arrêter lorsque :

            - le nœud contient une seule classe ;
            - la profondeur maximale est atteinte ;
            - le nombre d'observations est insuffisant pour effectuer
              une nouvelle séparation ;
            - aucune séparation valide n'est trouvée.

            ---

            ### 7. Prédiction

            Pour une nouvelle observation, on descend dans l'arbre
            suivant les différentes règles.

            Lorsque l'on atteint une feuille, la classe associée
            à cette feuille devient la prédiction :

            $$
            \hat{y}=classe_{feuille}
            $$

            ---

            ### 8. Accuracy

            L'accuracy est calculée par :

            $$
            Accuracy =
            \frac{
            \text{nombre de prédictions correctes}
            }{
            \text{nombre total d'observations}
            }
            $$

            Elle mesure la proportion d'observations correctement
            classées.
            """
        )

     

# ============================================================
# RANDOM FOREST
# ============================================================

if algorithm == "🌲 Random Forest":

    st.header("🌲 Random Forest")

    st.markdown(
        """
        Le **Random Forest** est un algorithme de Machine Learning
        supervisé utilisé notamment pour la **classification**.

        Il combine plusieurs **arbres de décision** afin d'obtenir
        une prédiction collective plus robuste.

        Le principe général est :

        1. créer plusieurs échantillons bootstrap des données ;
        2. construire un arbre de décision pour chaque échantillon ;
        3. sélectionner aléatoirement un sous-ensemble de caractéristiques
           pour chaque arbre ;
        4. faire une prédiction avec chaque arbre ;
        5. utiliser un **vote majoritaire** pour obtenir la prédiction finale.

        Cette approche constitue un exemple d'**ensemble learning**.
        """
    )

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.subheader("📊 Données de classification")

    random_forest_example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Classification simple",
            "Deux groupes",
            "Classification multiclasses",
        ],
        key="random_forest_example",
    )

    if random_forest_example == "Classification simple":

        default_rf_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6"
        )

        default_rf_y = "0,0,0,0,1,1,1,1"
        default_rf_test_x = "1.5,1.5; 5.5,5.5"

    elif random_forest_example == "Deux groupes":

        default_rf_x = (
            "1,2; "
            "1,3; "
            "2,2; "
            "2,3; "
            "7,7; "
            "7,8; "
            "8,7; "
            "8,8"
        )

        default_rf_y = "0,0,0,0,1,1,1,1"
        default_rf_test_x = "1.5,2.5; 7.5,7.5"

    elif random_forest_example == "Classification multiclasses":

        default_rf_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6; "
            "9,9; "
            "9,10; "
            "10,9; "
            "10,10"
        )

        default_rf_y = (
            "0,0,0,0,1,1,1,1,2,2,2,2"
        )

        default_rf_test_x = (
            "1.5,1.5; "
            "5.5,5.5; "
            "9.5,9.5"
        )

    else:

        default_rf_x = (
            "1,1; "
            "1,2; "
            "2,1; "
            "2,2; "
            "5,5; "
            "5,6; "
            "6,5; "
            "6,6"
        )

        default_rf_y = "0,0,0,0,1,1,1,1"
        default_rf_test_x = "1.5,1.5; 5.5,5.5"

    # ========================================================
    # FORMAT DES DONNÉES
    # ========================================================

    st.info(
        """
        **Format de X :**

        Séparez les observations par `;` et les caractéristiques
        par `,`.

        Exemple :

        `1,1; 1,2; 2,1; 2,2`

        correspond à :

        `[[1,1], [1,2], [2,1], [2,2]]`

        **Format de Y :**

        Une classe par observation, séparée par des virgules.

        Exemple :

        `0,0,0,0,1,1,1,1`

        Les classes peuvent être numériques ou textuelles.
        """
    )

    # ========================================================
    # DONNÉES D'ENTRAÎNEMENT
    # ========================================================

    st.subheader("📥 Données d'entraînement")

    col1, col2 = st.columns(2)

    with col1:

        random_forest_x_input = st.text_area(
            "Caractéristiques X",
            value=default_rf_x,
            height=150,
            help=(
                "Séparez les observations par ';' et les "
                "caractéristiques par ','."
            ),
            key="random_forest_x",
        )

    with col2:

        random_forest_y_input = st.text_area(
            "Classes Y",
            value=default_rf_y,
            height=150,
            help=(
                "Entrez une classe pour chaque observation, "
                "séparée par des virgules."
            ),
            key="random_forest_y",
        )

    # ========================================================
    # DONNÉES DE TEST
    # ========================================================

    st.subheader("🔍 Observations à classifier")

    random_forest_test_input = st.text_area(
        "Nouvelles observations X",
        value=default_rf_test_x,
        height=100,
        help=(
            "Entrez les nouvelles observations avec le même "
            "nombre de caractéristiques que X."
        ),
        key="random_forest_test_x",
    )

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.subheader("⚙️ Paramètres du Random Forest")

    parameter_col1, parameter_col2, parameter_col3 = st.columns(3)

    with parameter_col1:

        random_forest_n_estimators = st.number_input(
            "Nombre d'arbres",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
            key="random_forest_n_estimators",
            help=(
                "Nombre d'arbres de décision constituant "
                "la forêt."
            ),
        )

    with parameter_col2:

        random_forest_max_depth = st.number_input(
            "Profondeur maximale",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
            key="random_forest_max_depth",
            help=(
                "Profondeur maximale de chaque arbre."
            ),
        )

    with parameter_col3:

        random_forest_min_samples_split = st.number_input(
            "Minimum d'observations / split",
            min_value=2,
            max_value=1000,
            value=2,
            step=1,
            key="random_forest_min_samples_split",
            help=(
                "Nombre minimal d'observations nécessaires "
                "pour diviser un nœud."
            ),
        )

    # ========================================================
    # MAX FEATURES
    # ========================================================

    random_forest_max_features_mode = st.selectbox(
        "Nombre de caractéristiques utilisées par arbre",
        [
            "sqrt (automatique)",
            "Toutes les caractéristiques",
            "Nombre personnalisé",
        ],
        key="random_forest_max_features_mode",
        help=(
            "Le mode sqrt utilise √p caractéristiques "
            "par arbre."
        ),
    )

    if random_forest_max_features_mode == "sqrt (automatique)":

        random_forest_max_features = None

    elif random_forest_max_features_mode == "Toutes les caractéristiques":

        random_forest_max_features = "all"

    else:

        random_forest_max_features = st.number_input(
            "Max features",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key="random_forest_max_features_custom",
        )

    # ========================================================
    # RANDOM STATE
    # ========================================================

    random_forest_random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=100000,
        value=42,
        step=1,
        key="random_forest_random_state",
        help=(
            "Permet d'obtenir des résultats reproductibles."
        ),
    )

    # ========================================================
    # PARSING X
    # ========================================================

    def parse_random_forest_features(
        value: str,
    ) -> np.ndarray:
        """
        Convertit les données saisies en matrice NumPy.

        Format :

        1,2; 2,3; 4,5

        devient :

        [[1, 2],
         [2, 3],
         [4, 5]]
        """

        if not value.strip():

            raise ValueError(
                "Les données X ne doivent pas être vides."
            )

        try:

            rows = []

            for row in value.split(";"):

                row = row.strip()

                if not row:
                    continue

                values = [
                    float(item.strip())
                    for item in row.split(",")
                    if item.strip()
                ]

                if not values:

                    raise ValueError(
                        "Une observation X est vide."
                    )

                rows.append(values)

            if not rows:

                raise ValueError(
                    "Aucune observation valide n'a été trouvée."
                )

            n_features = len(rows[0])

            if n_features == 0:

                raise ValueError(
                    "Chaque observation doit contenir "
                    "au moins une caractéristique."
                )

            for row in rows:

                if len(row) != n_features:

                    raise ValueError(
                        "Toutes les observations X doivent "
                        "avoir le même nombre de caractéristiques."
                    )

            X = np.asarray(
                rows,
                dtype=float,
            )

            if not np.all(np.isfinite(X)):

                raise ValueError(
                    "Les données contiennent des valeurs "
                    "non finies."
                )

            return X

        except ValueError as exc:

            message = str(exc)

            if message.startswith(
                (
                    "Une observation",
                    "Aucune observation",
                    "Chaque observation",
                    "Toutes les observations",
                    "Les données contiennent",
                )
            ):

                raise

            raise ValueError(
                "X doit contenir uniquement des nombres. "
                "Utilisez ';' entre les observations et ',' "
                "entre les caractéristiques."
            ) from exc

    # ========================================================
    # PARSING Y
    # ========================================================

    def parse_random_forest_labels(
        value: str,
    ) -> np.ndarray:
        """
        Convertit les classes Y en tableau NumPy.

        Les labels numériques sont convertis en nombres.
        Les labels textuels sont conservés.
        """

        if not value.strip():

            raise ValueError(
                "Les classes Y ne doivent pas être vides."
            )

        labels = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

        if not labels:

            raise ValueError(
                "Aucune classe valide n'a été trouvée."
            )

        try:

            numeric_labels = [
                float(label)
                for label in labels
            ]

            if all(
                value.is_integer()
                for value in numeric_labels
            ):

                return np.asarray(
                    numeric_labels,
                    dtype=int,
                )

            return np.asarray(
                numeric_labels,
                dtype=float,
            )

        except ValueError:

            return np.asarray(
                labels,
                dtype=str,
            )

    # ========================================================
    # EXECUTION
    # ========================================================

    if st.button(
        "🚀 Entraîner le Random Forest",
        type="primary",
        use_container_width=True,
        key="random_forest_execute",
    ):

        try:

            # ------------------------------------------------
            # PARSING
            # ------------------------------------------------

            X_train = parse_random_forest_features(
                random_forest_x_input
            )

            y_train = parse_random_forest_labels(
                random_forest_y_input
            )

            X_test = parse_random_forest_features(
                random_forest_test_input
            )

            # ------------------------------------------------
            # PARAMÈTRES
            # ------------------------------------------------

            n_estimators = int(
                random_forest_n_estimators
            )

            max_depth = int(
                random_forest_max_depth
            )

            min_samples_split = int(
                random_forest_min_samples_split
            )

            random_state = int(
                random_forest_random_state
            )

            # ------------------------------------------------
            # MAX FEATURES
            # ------------------------------------------------

            if random_forest_max_features == "all":

                max_features = X_train.shape[1]

            else:

                max_features = random_forest_max_features

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if len(X_train) != len(y_train):

                raise ValueError(
                    "X et Y doivent contenir le même nombre "
                    "d'observations."
                )

            if X_train.shape[1] != X_test.shape[1]:

                raise ValueError(
                    "Les données d'entraînement et les données "
                    "de test doivent avoir le même nombre "
                    "de caractéristiques."
                )

            if len(X_train) < 2:

                raise ValueError(
                    "Le Random Forest nécessite au moins "
                    "deux observations d'entraînement."
                )

            if max_features is not None:

                if max_features < 1:

                    raise ValueError(
                        "max_features doit être supérieur ou égal à 1."
                    )

                if max_features > X_train.shape[1]:

                    raise ValueError(
                        "max_features ne peut pas être supérieur "
                        "au nombre de caractéristiques."
                    )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            result = random_forest(
                X_train,
                y_train,
                X_test,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                random_state=random_state,
            )

            # ------------------------------------------------
            # SESSION STATE
            # ------------------------------------------------

            st.session_state[
                "random_forest_result"
            ] = result

            st.session_state[
                "random_forest_X_train"
            ] = X_train

            st.session_state[
                "random_forest_y_train"
            ] = y_train

            st.session_state[
                "random_forest_X_test"
            ] = X_test

            st.success(
                "✅ Random Forest entraîné avec succès."
            )

        except (ValueError, TypeError) as exc:

            st.error(
                f"❌ {exc}"
            )


# ============================================================
# RANDOM FOREST RESULTS
# ============================================================

if (
    algorithm == "🌲 Random Forest"
    and "random_forest_result" in st.session_state
):

    result = st.session_state[
        "random_forest_result"
    ]

    X_train = st.session_state[
        "random_forest_X_train"
    ]

    y_train = st.session_state[
        "random_forest_y_train"
    ]

    X_test = st.session_state[
        "random_forest_X_test"
    ]

    st.divider()

    st.header(
        "📊 Résultats — Random Forest"
    )

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            (
                f"{result['accuracy']:.2%}"
                if result["accuracy"] is not None
                else "N/A"
            ),
        )

    with col2:

        st.metric(
            "Nombre d'arbres",
            str(result["n_estimators"]),
        )

    with col3:

        st.metric(
            "Observations train",
            str(result["n_train_samples"]),
        )

    with col4:

        st.metric(
            "Observations test",
            str(result["n_test_samples"]),
        )

    with col5:

        st.metric(
            "Caractéristiques",
            str(result["n_features"]),
        )

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.subheader(
        "⚙️ Paramètres utilisés"
    )

    max_features_result = result[
        "max_features"
    ]

    if max_features_result is None:

        max_features_display = "sqrt automatique"

    else:

        max_features_display = str(
            max_features_result
        )

    parameter_df = pd.DataFrame(
        {
            "Paramètre": [
                "Nombre d'arbres",
                "Profondeur maximale",
                "Minimum observations / split",
                "Max features",
                "Random state",
            ],
            "Valeur": [
                result["n_estimators"],
                result["max_depth"],
                result["min_samples_split"],
                max_features_display,
                result["random_state"],
            ],
        }
    )

    st.dataframe(
        parameter_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.subheader(
        "🔢 Prédictions"
    )

    predictions = result[
        "predictions"
    ]

    prediction_rows = []

    for i, (features, prediction) in enumerate(
        zip(X_test, predictions)
    ):

        row = {
            "Observation": i + 1,
        }

        for feature_index, feature_value in enumerate(
            features
        ):

            row[
                f"X{feature_index + 1}"
            ] = feature_value

        row[
            "Classe prédite"
        ] = prediction

        prediction_rows.append(
            row
        )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # FEATURES UTILISÉES PAR CHAQUE ARBRE
    # ========================================================

    st.subheader(
        "🌳 Caractéristiques utilisées par arbre"
    )

    feature_indices = result[
        "feature_indices"
    ]

    feature_rows = []

    for tree_index, indices in enumerate(
        feature_indices
    ):

        feature_rows.append(
            {
                "Arbre": tree_index + 1,
                "Caractéristiques": ", ".join(
                    f"X{i + 1}"
                    for i in indices
                ),
                "Nombre de caractéristiques": len(
                    indices
                ),
            }
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # VISUALISATION 2D
    # ========================================================

    st.subheader(
        "📈 Visualisation des données"
    )

    if plt is None:

        st.warning(
            "La visualisation nécessite Matplotlib. "
            "Installez-le avec `pip install matplotlib`."
        )

    elif X_train.shape[1] == 2:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        classes = np.unique(
            y_train
        )

        for class_value in classes:

            mask = (
                y_train == class_value
            )

            ax.scatter(
                X_train[mask, 0],
                X_train[mask, 1],
                s=80,
                label=f"Classe {class_value}",
            )

        # ----------------------------------------------------
        # OBSERVATIONS DE TEST
        # ----------------------------------------------------

        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            marker="*",
            s=220,
            label="Nouvelles observations",
        )

        # ----------------------------------------------------
        # ANNOTATIONS
        # ----------------------------------------------------

        for i, prediction in enumerate(
            predictions
        ):

            ax.annotate(
                f"→ {prediction}",
                (
                    X_test[i, 0],
                    X_test[i, 1],
                ),
                xytext=(8, 8),
                textcoords="offset points",
            )

        ax.set_xlabel(
            "Caractéristique X₁"
        )

        ax.set_ylabel(
            "Caractéristique X₂"
        )

        ax.set_title(
            "Random Forest — Classification"
        )

        ax.legend()

        ax.grid(
            alpha=0.3
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "La visualisation graphique 2D est disponible "
            "lorsque les données contiennent exactement "
            "deux caractéristiques."
        )

    # ========================================================
    # INTERPRÉTATION
    # ========================================================

    st.subheader(
        "💡 Interprétation"
    )

    st.markdown(
        f"""
        **Nombre d'arbres :**
        `{result["n_estimators"]}`

        Le Random Forest construit une forêt composée de
        **{result["n_estimators"]} arbres de décision**.

        Chaque arbre est entraîné sur un échantillon bootstrap
        des données d'entraînement.

        En plus du rééchantillonnage des observations, chaque arbre
        utilise un sous-ensemble aléatoire des caractéristiques.

        **Nombre de caractéristiques :**
        `{result["n_features"]}`

        **Nombre d'observations d'entraînement :**
        `{result["n_train_samples"]}`

        Pour chaque nouvelle observation, chaque arbre produit
        une prédiction.

        La prédiction finale est obtenue par **vote majoritaire**
        entre les arbres.

        **Accuracy :**
        `{f"{result['accuracy']:.2%}" if result["accuracy"] is not None else "N/A"}`

        Cette accuracy correspond à la proportion d'observations
        correctement classées sur les données d'entraînement
        lorsqu'un jeu de labels de test n'est pas fourni.
        """
    )

    
    # ========================================================
    # EXPLICATION MATHÉMATIQUE
    # ========================================================

    accuracy = result["accuracy"]

    if accuracy is not None:
        accuracy_display = f"{accuracy:.2%}"
    else:
        accuracy_display = "N/A"

    st.markdown(
        f"""
        ### 📐 Explication mathématique

        Le **Random Forest** repose sur un ensemble de plusieurs
        arbres de décision.

        Chaque arbre est entraîné sur un échantillon bootstrap
        des données d'entraînement.

        En plus du rééchantillonnage des observations, chaque arbre
        utilise un sous-ensemble aléatoire des caractéristiques.

        **Nombre de caractéristiques :**
        `{result["n_features"]}`

        **Nombre d'observations d'entraînement :**
        `{result["n_train_samples"]}`

        Pour chaque nouvelle observation, chaque arbre produit
        une prédiction.

        La prédiction finale est obtenue par **vote majoritaire**
        entre les arbres.

        **Accuracy :**
        `{accuracy_display}`

        Cette accuracy correspond à la proportion d'observations
        correctement classées sur les données de test lorsqu'un
        jeu de labels de test est fourni.
        """
    )

    
# ============================================================
# 🎲 NAIVE BAYES
# ============================================================

elif algorithm == "🎲 Naive Bayes":

    st.subheader("🎲 Gaussian Naive Bayes")

    st.markdown(
        """
        Le **Naive Bayes** est un algorithme de classification
        probabiliste basé sur le théorème de Bayes.

        Ici, nous utilisons la variante **Gaussian Naive Bayes**,
        adaptée aux caractéristiques numériques.
        """
    )

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### 📊 Jeu de données")

    naive_bayes_example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Deux groupes",
            "Trois groupes",
            "Classification multiclasses",
        ],
        key="naive_bayes_example",
    )

    if naive_bayes_example == "Deux groupes":

        default_x = (
            "1,1; 1.2,1.1; 0.8,0.9; 1.1,1.2; "
            "5,5; 5.2,5.1; 4.8,4.9; 5.1,5.2"
        )

        default_y = "0,0,0,0,1,1,1,1"

        default_test = (
            "1,1; 5,5; 1.1,0.9; 4.9,5.1"
        )

    elif naive_bayes_example == "Trois groupes":

        default_x = (
            "1,1; 1.2,1.1; 0.9,1; "
            "5,5; 5.2,5.1; 4.9,5.2; "
            "9,1; 9.2,1.1; 8.9,0.9"
        )

        default_y = "0,0,0,1,1,1,2,2,2"

        default_test = (
            "1.1,1; 5.1,5; 9.1,1"
        )

    elif naive_bayes_example == "Classification multiclasses":

        default_x = (
            "1,1; 1.1,1.2; 0.9,1.1; "
            "5,5; 5.1,4.9; 4.9,5.2; "
            "9,1; 9.2,1.1; 8.8,0.9"
        )

        default_y = "0,0,0,1,1,1,2,2,2"

        default_test = (
            "1,1.1; 5,5; 9,1"
        )

    else:

        default_x = (
            "1,1; 1.2,1.1; 0.8,0.9; "
            "5,5; 5.2,5.1; 4.8,4.9"
        )

        default_y = "0,0,0,1,1,1"

        default_test = "1,1; 5,5"

    # ========================================================
    # DONNÉES D'ENTRAÎNEMENT
    # ========================================================

    naive_bayes_x = st.text_area(
        "X — données d'entraînement",
        value=default_x,
        key="naive_bayes_x",
        help=(
            "Séparez les observations par ';' "
            "et les caractéristiques par ','."
        ),
    )

    naive_bayes_y = st.text_input(
        "Y — classes",
        value=default_y,
        key="naive_bayes_y",
        help=(
            "Entrez les classes séparées par des virgules."
        ),
    )

    # ========================================================
    # DONNÉES DE TEST
    # ========================================================

    naive_bayes_x_test = st.text_area(
        "X — observations à prédire",
        value=default_test,
        key="naive_bayes_x_test",
        help=(
            "Entrez les observations à prédire "
            "avec le même nombre de caractéristiques."
        ),
    )

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.markdown("### ⚙️ Paramètres")

    naive_bayes_smoothing = st.number_input(
        "Variance smoothing",
        min_value=1e-12,
        max_value=1.0,
        value=1e-9,
        step=1e-9,
        format="%.10f",
        key="naive_bayes_smoothing",
        help=(
            "Petite valeur ajoutée aux variances "
            "pour éviter les variances nulles."
        ),
    )

    # ========================================================
    # PARSERS
    # ========================================================

    def parse_naive_bayes_features(text):
        """Convertit '1,2; 3,4' en matrice NumPy."""

        try:
            rows = [
                row.strip()
                for row in text.split(";")
                if row.strip()
            ]

            if not rows:
                raise ValueError(
                    "Aucune observation fournie."
                )

            data = []

            for row in rows:

                values = [
                    float(value.strip())
                    for value in row.split(",")
                ]

                if not values:
                    raise ValueError(
                        "Une observation est vide."
                    )

                data.append(values)

            n_features = len(data[0])

            if n_features == 0:
                raise ValueError(
                    "Chaque observation doit contenir "
                    "au moins une caractéristique."
                )

            if any(
                len(row) != n_features
                for row in data
            ):
                raise ValueError(
                    "Toutes les observations doivent avoir "
                    "le même nombre de caractéristiques."
                )

            return np.asarray(
                data,
                dtype=float,
            )

        except ValueError as exc:

            raise ValueError(
                f"Format des données invalide : {exc}"
            ) from exc

    def parse_naive_bayes_labels(text):
        """Convertit '0,1,0,1' en vecteur de classes."""

        values = [
            value.strip()
            for value in text.split(",")
            if value.strip()
        ]

        if not values:
            raise ValueError(
                "Aucune classe fournie."
            )

        parsed = []

        for value in values:

            try:
                number = float(value)

                if number.is_integer():
                    parsed.append(int(number))
                else:
                    parsed.append(number)

            except ValueError:

                parsed.append(value)

        return np.asarray(parsed)

    # ========================================================
    # EXÉCUTION
    # ========================================================

    if st.button(
        "▶️ Entraîner le Naive Bayes",
        key="run_naive_bayes",
        type="primary",
    ):

        try:

            X_train = parse_naive_bayes_features(
                naive_bayes_x
            )

            y_train = parse_naive_bayes_labels(
                naive_bayes_y
            )

            X_test = parse_naive_bayes_features(
                naive_bayes_x_test
            )

            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(
                    "Le nombre d'observations X doit être "
                    "égal au nombre de classes Y."
                )

            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(
                    "X_train et X_test doivent avoir "
                    "le même nombre de caractéristiques."
                )

            result = naive_bayes(
                X_train,
                y_train,
                X_test,
                variance_smoothing=naive_bayes_smoothing,
            )

            st.session_state[
                "naive_bayes_result"
            ] = result

            st.session_state[
                "naive_bayes_X_train"
            ] = X_train

            st.session_state[
                "naive_bayes_y_train"
            ] = y_train

            st.session_state[
                "naive_bayes_X_test"
            ] = X_test

            st.success(
                "✅ Naive Bayes entraîné avec succès."
            )

        except Exception as exc:

            st.error(
                f"❌ Erreur Naive Bayes : {exc}"
            )


# ============================================================
# RÉSULTATS NAIVE BAYES
# ============================================================

if (
    algorithm == "🎲 Naive Bayes"
    and "naive_bayes_result" in st.session_state
):

    result = st.session_state[
        "naive_bayes_result"
    ]

    X_train = st.session_state[
        "naive_bayes_X_train"
    ]

    y_train = st.session_state[
        "naive_bayes_y_train"
    ]

    X_test = st.session_state[
        "naive_bayes_X_test"
    ]

    st.markdown("---")

    st.subheader("📈 Résultats — Naive Bayes")

    # ========================================================
    # MÉTRIQUES
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        accuracy = result["accuracy"]

        if accuracy is not None:
            accuracy_display = f"{accuracy:.2%}"
        else:
            accuracy_display = "N/A"

        st.metric(
            "Accuracy",
            accuracy_display,
        )

    with col2:
        st.metric(
            "Classes",
            result["n_classes"],
        )

    with col3:
        st.metric(
            "Features",
            result["n_features"],
        )

    with col4:
        st.metric(
            "Observations test",
            result["n_test_samples"],
        )

    # ========================================================
    # PROBABILITÉS A PRIORI
    # ========================================================

    st.markdown("### 📊 Probabilités a priori")

    prior_data = {
        "Classe": result["classes"],
        "Prior": [
            f"{value:.4f}"
            for value in result["priors"]
        ],
    }

    st.dataframe(
        prior_data,
        use_container_width=True,
    )

    # ========================================================
    # MOYENNES
    # ========================================================

    st.markdown("### 📐 Moyennes par classe")

    mean_data = {
        f"Feature {index + 1}":
        result["means"][:, index]
        for index in range(
            result["n_features"]
        )
    }

    mean_data["Classe"] = result["classes"]

    st.dataframe(
        mean_data,
        use_container_width=True,
    )

    # ========================================================
    # VARIANCES
    # ========================================================

    st.markdown("### 📏 Variances par classe")

    variance_data = {
        f"Feature {index + 1}":
        result["variances"][:, index]
        for index in range(
            result["n_features"]
        )
    }

    variance_data["Classe"] = result["classes"]

    st.dataframe(
        variance_data,
        use_container_width=True,
    )

    # ========================================================
    # PRÉDICTIONS
    # ========================================================

    st.markdown("### 🎯 Prédictions")

    predictions = result["predictions"]
    probabilities = result["probabilities"]
    classes = result["classes"]

    prediction_rows = []

    for index, prediction in enumerate(
        predictions
    ):

        row = {
            "Observation": index + 1,
            "Classe prédite": prediction,
        }

        for class_index, class_value in enumerate(
            classes
        ):
            row[
                f"P(classe {class_value})"
            ] = f"{probabilities[index, class_index]:.2%}"

        prediction_rows.append(row)

    st.dataframe(
        prediction_rows,
        use_container_width=True,
    )

    # ========================================================
    # VISUALISATION 2D
    # ========================================================

    if X_train.shape[1] >= 2:

        st.markdown("### 📊 Visualisation 2D")

        fig, ax = plt.subplots()

        for class_value in np.unique(y_train):

            mask = y_train == class_value

            ax.scatter(
                X_train[mask, 0],
                X_train[mask, 1],
                label=f"Classe {class_value}",
            )

        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            marker="x",
            s=100,
            label="Observations à prédire",
        )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(
            "Naive Bayes — Classification"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    # ========================================================
    # INTERPRÉTATION
    # ========================================================

    st.markdown("### 💡 Interprétation")

    st.markdown(
        f"""
        Le modèle **Gaussian Naive Bayes** a appris
        **{result["n_classes"]} classes** à partir de
        **{result["n_train_samples"]} observations**.

        Chaque classe possède :

        - une probabilité a priori ;
        - une moyenne pour chaque caractéristique ;
        - une variance pour chaque caractéristique.

        Pour une nouvelle observation, le modèle calcule
        la probabilité qu'elle appartienne à chaque classe.

        La classe possédant la probabilité la plus élevée
        est sélectionnée comme prédiction.

        **Nombre de caractéristiques :**
        `{result["n_features"]}`

        **Nombre d'observations testées :**
        `{result["n_test_samples"]}`

        **Variance smoothing :**
        `{result["variance_smoothing"]}`
        """
    )

    # ========================================================
    # EXPLICATION MATHÉMATIQUE
    # ========================================================

    st.markdown("### 📐 Explication mathématique")

    st.markdown(
        """
        Le Naive Bayes utilise le théorème de Bayes.

        Pour une observation `x` et une classe `C`, le modèle
        compare les probabilités :

        `P(C | x)`

        En supposant que les caractéristiques sont
        conditionnellement indépendantes, la probabilité
        conditionnelle peut être décomposée en produit des
        probabilités de chaque caractéristique.

        Pour une caractéristique numérique, Gaussian Naive Bayes
        utilise une distribution normale caractérisée par :

        - une moyenne `μ` ;
        - une variance `σ²`.

        Le modèle sélectionne finalement la classe ayant
        la probabilité postérieure la plus élevée.

        Cette hypothèse d'indépendance entre caractéristiques
        est à l'origine du terme **« Naive »**.
        """
    )

   

# ============================================================
# PCA
# ============================================================

elif algorithm == "📐 PCA":

    st.subheader("📐 Analyse en Composantes Principales (PCA)")

    st.markdown(
        """
        Le **PCA (Principal Component Analysis)** permet de réduire
        la dimension des données tout en conservant une grande partie
        de leur variance.

        Le PCA transforme les caractéristiques originales en nouvelles
        composantes principales, classées selon la quantité de variance
        qu'elles expliquent.
        """
    )

    # ========================================================
    # EXEMPLES
    # ========================================================

    st.markdown("### 📚 Exemple de données")

    pca_example = st.selectbox(
        "Choisir un exemple",
        [
            "Données personnalisées",
            "Données 2D",
            "Données 3D",
            "Données multivariées",
        ],
        key="pca_example",
    )

    pca_examples = {
        "Données 2D": {
            "X": (
                "2,4; "
                "3,6; "
                "4,8; "
                "5,10; "
                "6,12; "
                "7,14"
            ),
            "n_components": 1,
        },
        "Données 3D": {
            "X": (
                "1,2,10; "
                "2,4,20; "
                "3,6,30; "
                "4,8,40; "
                "5,10,50; "
                "6,12,60"
            ),
            "n_components": 2,
        },
        "Données multivariées": {
            "X": (
                "2,10,100,5; "
                "3,12,110,6; "
                "4,15,120,7; "
                "5,18,130,8; "
                "6,20,140,9; "
                "7,22,150,10; "
                "8,25,160,11; "
                "9,27,170,12"
            ),
            "n_components": 2,
        },
        "Données personnalisées": {
            "X": "",
            "n_components": 2,
        },
    }

    selected_example = pca_examples[
        pca_example
    ]

    # Ne pas modifier la session_state d'un widget déjà créé.
    if (
        pca_example != "Données personnalisées"
        and st.session_state.get("pca_last_example")
        != pca_example
    ):
        st.session_state["pca_x"] = (
            selected_example["X"]
        )
        st.session_state["pca_n_components"] = (
            selected_example["n_components"]
        )
        st.session_state["pca_last_example"] = (
            pca_example
        )

    # ========================================================
    # DONNÉES
    # ========================================================

    st.markdown("### 📊 Données")

    pca_x = st.text_area(
        "Données X",
        value=st.session_state.get(
            "pca_x",
            "",
        ),
        height=120,
        key="pca_x",
        help=(
            "Séparez les observations par ';' "
            "et les caractéristiques par ','. "
            "Exemple : 1,2,3; 2,4,6; 3,6,9"
        ),
    )

    st.caption(
        "Format : observation1; observation2; ... "
        "avec les caractéristiques séparées par des virgules."
    )

    # ========================================================
    # PARAMÈTRES
    # ========================================================

    st.markdown("### ⚙️ Paramètres")

    col1, col2 = st.columns(2)

    with col1:
        pca_n_components = st.number_input(
            "Nombre de composantes",
            min_value=1,
            value=int(
                st.session_state.get(
                    "pca_n_components",
                    2,
                )
            ),
            step=1,
            key="pca_n_components",
            help=(
                "Nombre de composantes principales "
                "à conserver."
            ),
        )

    with col2:
        pca_standardize = st.checkbox(
            "Standardiser les données",
            value=True,
            key="pca_standardize",
            help=(
                "Recommandé lorsque les caractéristiques "
                "ont des échelles différentes."
            ),
        )

    # ========================================================
    # PARSERS
    # ========================================================

    def parse_pca_features(text):
        """
        Transforme le texte de l'utilisateur en matrice NumPy.
        """

        if not text.strip():
            raise ValueError(
                "Les données X ne peuvent pas être vides."
            )

        rows = []

        for row in text.split(";"):
            row = row.strip()

            if not row:
                continue

            values = [
                value.strip()
                for value in row.split(",")
                if value.strip()
            ]

            if not values:
                continue

            try:
                rows.append(
                    [
                        float(value)
                        for value in values
                    ]
                )
            except ValueError as exc:
                raise ValueError(
                    "Toutes les caractéristiques doivent "
                    "être numériques."
                ) from exc

        if not rows:
            raise ValueError(
                "Aucune observation valide n'a été trouvée."
            )

        n_features = len(rows[0])

        if n_features == 0:
            raise ValueError(
                "Chaque observation doit contenir "
                "au moins une caractéristique."
            )

        for row in rows:
            if len(row) != n_features:
                raise ValueError(
                    "Toutes les observations doivent avoir "
                    "le même nombre de caractéristiques."
                )

        return np.array(
            rows,
            dtype=float,
        )

    # ========================================================
    # EXÉCUTION
    # ========================================================

    if st.button(
        "▶️ Appliquer le PCA",
        type="primary",
        key="run_pca",
    ):

        try:
            X = parse_pca_features(
                pca_x
            )

            result = pca(
                X,
                n_components=int(
                    pca_n_components
                ),
                standardize=pca_standardize,
            )

            st.session_state[
                "pca_result"
            ] = result

            st.session_state[
                "pca_X"
            ] = X

            st.success(
                "✅ PCA calculé avec succès."
            )

        except ValueError as exc:
            st.error(
                f"❌ Erreur PCA : {exc}"
            )

        except Exception as exc:
            st.error(
                f"❌ Une erreur inattendue est survenue : {exc}"
            )


# ============================================================
# RÉSULTATS PCA
# ============================================================

if (
    algorithm == "📐 PCA"
    and "pca_result" in st.session_state
):

    result = st.session_state[
        "pca_result"
    ]

    X = st.session_state[
        "pca_X"
    ]

    st.markdown("---")

    st.subheader("📊 Résultats du PCA")

    # ========================================================
    # MÉTRIQUES PRINCIPALES
    # ========================================================

    variance_ratio = result[
        "explained_variance_ratio"
    ]

    cumulative_variance = result[
        "cumulative_explained_variance"
    ]

    total_explained = float(
        cumulative_variance[-1]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Observations",
            result["n_samples"],
        )

    with col2:
        st.metric(
            "Caractéristiques",
            result["n_features"],
        )

    with col3:
        st.metric(
            "Composantes",
            result["n_components"],
        )

    with col4:
        st.metric(
            "Variance conservée",
            f"{total_explained:.2%}",
        )

    # ========================================================
    # VARIANCE EXPLIQUÉE
    # ========================================================

    st.markdown("### 📈 Variance expliquée")

    variance_data = {
        "Composante": [
            f"PC{i + 1}"
            for i in range(
                len(variance_ratio)
            )
        ],
        "Variance expliquée": [
            f"{value:.2%}"
            for value in variance_ratio
        ],
        "Variance cumulée": [
            f"{value:.2%}"
            for value in cumulative_variance
        ],
    }

    st.dataframe(
        variance_data,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # VISUALISATION DE LA VARIANCE
    # ========================================================

    st.markdown(
        "### 📊 Variance expliquée par composante"
    )

    fig_variance, ax_variance = plt.subplots(
        figsize=(8, 4)
    )

    component_numbers = np.arange(
        1,
        len(variance_ratio) + 1,
    )

    ax_variance.bar(
        component_numbers,
        variance_ratio,
    )

    ax_variance.set_xlabel(
        "Composante principale"
    )

    ax_variance.set_ylabel(
        "Variance expliquée"
    )

    ax_variance.set_title(
        "Variance expliquée par composante"
    )

    ax_variance.set_xticks(
        component_numbers
    )

    ax_variance.set_ylim(
        0,
        max(
            1.0,
            float(
                np.max(variance_ratio)
            ) * 1.15,
        ),
    )

    ax_variance.grid(
        axis="y",
        alpha=0.3,
    )

    st.pyplot(
        fig_variance,
        clear_figure=True,
    )

    # ========================================================
    # COMPOSANTES PRINCIPALES
    # ========================================================

    st.markdown(
        "### 🧭 Composantes principales"
    )

    components = result[
        "components"
    ]

    component_columns = [
        f"PC{i + 1}"
        for i in range(
            components.shape[1]
        )
    ]

    feature_names = [
        f"X{i + 1}"
        for i in range(
            components.shape[0]
        )
    ]

    components_data = pd.DataFrame(
        components,
        index=feature_names,
        columns=component_columns,
    )

    st.dataframe(
        components_data,
        use_container_width=True,
    )

    # ========================================================
    # VALEURS PROPRES
    # ========================================================

    st.markdown(
        "### 🔢 Valeurs propres"
    )

    eigenvalues = result[
        "eigenvalues"
    ]

    eigenvalue_data = pd.DataFrame(
        {
            "Composante": component_columns,
            "Valeur propre": eigenvalues,
        }
    )

    st.dataframe(
        eigenvalue_data,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DONNÉES TRANSFORMÉES
    # ========================================================

    st.markdown(
        "### 🔄 Données transformées"
    )

    transformed = result[
        "transformed_data"
    ]

    transformed_columns = [
        f"PC{i + 1}"
        for i in range(
            transformed.shape[1]
        )
    ]

    transformed_data = pd.DataFrame(
        transformed,
        columns=transformed_columns,
    )

    st.dataframe(
        transformed_data,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # VISUALISATION 2D
    # ========================================================

    if (
        transformed.shape[1] >= 2
    ):

        st.markdown(
            "### 🗺️ Projection sur les deux premières composantes"
        )

        fig_pca, ax_pca = plt.subplots(
            figsize=(8, 5)
        )

        ax_pca.scatter(
            transformed[:, 0],
            transformed[:, 1],
            s=70,
        )

        ax_pca.set_xlabel(
            "PC1"
        )

        ax_pca.set_ylabel(
            "PC2"
        )

        ax_pca.set_title(
            "Projection des observations dans l'espace PCA"
        )

        ax_pca.axhline(
            0,
            linewidth=0.8,
            alpha=0.4,
        )

        ax_pca.axvline(
            0,
            linewidth=0.8,
            alpha=0.4,
        )

        ax_pca.grid(
            alpha=0.3
        )

        st.pyplot(
            fig_pca,
            clear_figure=True,
        )

    else:

        st.info(
            "ℹ️ Une seule composante a été sélectionnée. "
            "Une projection 2D nécessite au moins deux composantes."
        )

    # ========================================================
    # INTERPRÉTATION
    # ========================================================

    st.markdown(
        "### 💡 Interprétation"
    )

    first_variance = float(
        variance_ratio[0]
    )

    st.info(
        f"""
        **PC1** explique environ **{first_variance:.2%}**
        de la variance totale conservée.

        Le PCA a transformé les **{result["n_features"]}**
        caractéristiques originales en
        **{result["n_components"]} composantes principales**.

        La variance cumulée des composantes conservées est
        de **{total_explained:.2%}**.
        """
    )

    # ========================================================
    # EXPLICATION MATHÉMATIQUE
    # ========================================================

    with st.expander(
        "📐 Comprendre la démarche mathématique"
    ):

        st.markdown(
            """
            ### 1. Centrage / standardisation

            Les données sont d'abord centrées autour de leur moyenne.

            Si la standardisation est activée :

            `Z = (X - μ) / σ`

            ### 2. Matrice de covariance

            On calcule ensuite la matrice de covariance :

            `C = XᵀX / (n - 1)`

            Cette matrice décrit les relations entre les
            caractéristiques.

            ### 3. Valeurs propres et vecteurs propres

            On cherche les solutions de :

            `Cv = λv`

            où :

            - `λ` représente une valeur propre ;
            - `v` représente un vecteur propre.

            Les vecteurs propres associés aux plus grandes valeurs
            propres correspondent aux principales directions de variance.

            ### 4. Sélection des composantes

            Les composantes sont classées selon leurs valeurs propres.

            La première composante explique donc la plus grande
            quantité de variance possible.

            ### 5. Projection

            Les données sont ensuite projetées sur les composantes
            sélectionnées :

            `Z = XW`

            où `W` contient les vecteurs propres sélectionnés.

            Le résultat est une représentation des données dans un
            espace de dimension réduite.
            """
        )




# ============================================================
# FUTURE ALGORITHMS
# ============================================================

elif algorithm != "📈 Linear Regression" and algorithm != "🎯 Logistic Regression":

    st.header(algorithm)

    st.info(
        "🚧 Ce module sera implémenté dans une prochaine étape."
    )

