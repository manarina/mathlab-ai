# ============================================================
# MATHLAB AI — AI MATH ASSISTANT
# ============================================================

import streamlit as st

from core.ai.math_assistant import (
    math_assistant,
    format_result,
)

from core.ai.ai_explainer import (
    ai_explain,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MathLab AI — AI Math Assistant",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EN-TÊTE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Math Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Assistant mathématique intelligent pour résoudre,
        comprendre et explorer les mathématiques.
    </div>
    """,
    unsafe_allow_html=True,
)


st.info(
    """
    💡 **Posez votre question en langage naturel.**

    Par exemple :

    > Calcule la dérivée de x² + 3x

    L'assistant identifie automatiquement l'opération
    mathématique et affiche la démarche ainsi que le résultat.

    🧠 **Une explication IA peut ensuite être générée
    à partir du résultat calculé par SymPy.**
    """
)


# ============================================================
# EXEMPLES
# ============================================================

st.markdown("### 💬 Exemples de questions")

example_columns = st.columns(3)

examples = [
    "Calcule la dérivée de x^2 + 3*x",
    "Calcule l'intégrale de x^2 de 0 à 2",
    "Calcule la limite de sin(x)/x quand x tend vers 0",
    "Résous 2*x + 5 = 15",
    "Simplifie (x^2 - 1)/(x - 1)",
    "Calcule 2 + 3 * 4",
]


for index, example in enumerate(examples):

    column = example_columns[index % 3]

    with column:

        if st.button(
            example,
            use_container_width=True,
            key=f"ai_example_{index}",
        ):

            st.session_state["ai_math_question"] = example


st.divider()


# ============================================================
# SAISIE
# ============================================================

st.markdown("### ✏️ Votre question")

question = st.text_area(
    "Entrez une expression ou une question mathématique",
    value=st.session_state.get(
        "ai_math_question",
        "",
    ),
    height=120,
    placeholder=(
        "Exemple : Calcule la dérivée de x^2 + 3*x"
    ),
    key="ai_math_question_input",
)


# ============================================================
# BOUTON D'EXÉCUTION
# ============================================================

execute = st.button(
    "🚀 Résoudre",
    type="primary",
    use_container_width=True,
    key="execute_ai_math",
)


# ============================================================
# TRAITEMENT
# ============================================================

if execute:

    if not question.strip():

        st.warning(
            "Veuillez saisir une question mathématique."
        )

    else:

        try:

            # ====================================================
            # APPEL DU MOTEUR MATHÉMATIQUE
            # ====================================================

            result = math_assistant(
                question.strip()
            )

            # ====================================================
            # VÉRIFICATION
            # ====================================================

            if not isinstance(result, dict):

                raise TypeError(
                    "Le moteur mathématique doit retourner "
                    "un dictionnaire."
                )

            # ====================================================
            # STOCKAGE
            # ====================================================

            st.session_state["ai_math_result"] = result

            st.session_state["ai_math_question_result"] = (
                question.strip()
            )

            # ====================================================
            # RÉINITIALISATION DE L'EXPLICATION IA
            # ====================================================

            st.session_state["ai_math_ai_explanation"] = None

            st.session_state[
                "ai_math_ai_explanation_error"
            ] = None

        except Exception as error:

            st.session_state["ai_math_result"] = None

            st.session_state[
                "ai_math_ai_explanation"
            ] = None

            st.error(
                f"""
                Une erreur est survenue pendant
                le traitement de la question.

                **Détail :** {error}
                """
            )


# ============================================================
# AFFICHAGE DU RÉSULTAT
# ============================================================

result = st.session_state.get(
    "ai_math_result"
)


if result is not None:

    st.divider()

    st.markdown(
        "## 📊 Résultat"
    )

    # ========================================================
    # INFORMATIONS GÉNÉRALES
    # ========================================================

    operation = result.get(
        "operation",
        "unknown",
    )

    expression = result.get(
        "expression",
        None,
    )

    steps = result.get(
        "steps",
        [],
    )

    explanation = result.get(
        "explanation",
        None,
    )

    answer = result.get(
        "result",
        result.get(
            "solution",
            None,
        ),
    )

    latex = result.get(
        "latex",
        None,
    )


    # ========================================================
    # CARTES PRINCIPALES
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🔎 Opération détectée"
        )

        operation_labels = {
            "equation": "Équation",
            "derivative": "Dérivée",
            "integral": "Intégrale",
            "limit": "Limite",
            "simplify": "Simplification",
            "calculation": "Calcul",
            "unknown": "Inconnue",
        }

        operation_label = operation_labels.get(
            operation,
            str(operation).replace(
                "_",
                " ",
            ).title(),
        )

        st.success(
            operation_label
        )


    with col2:

        st.markdown(
            "### 📐 Expression mathématique"
        )

        if expression is not None:

            st.code(
                str(expression),
                language="text",
            )

        else:

            st.info(
                "Expression non disponible."
            )


    # ========================================================
    # RÉSULTAT
    # ========================================================

    st.markdown(
        "### ✅ Résultat"
    )

    if answer is not None:

        st.success(
            str(answer)
        )

        # ----------------------------------------------------
        # REPRESENTATION LATEX
        # ----------------------------------------------------

        if latex:

            st.markdown(
                "#### 📐 Représentation mathématique"
            )

            try:

                st.latex(
                    latex
                )

            except Exception:

                st.code(
                    str(latex),
                    language="text",
                )

    else:

        st.warning(
            "Aucun résultat n'a été retourné."
        )


    st.divider()


    # ========================================================
    # DÉMARCHE
    # ========================================================

    st.markdown(
        "### 📝 Démarche"
    )

    if steps:

        for index, step in enumerate(
            steps,
            start=1,
        ):

            if isinstance(step, dict):

                step_text = step.get(
                    "description",
                    step.get(
                        "step",
                        str(step),
                    ),
                )

            else:

                step_text = str(step)

            st.markdown(
                f"**Étape {index}.** {step_text}"
            )

    else:

        st.info(
            "Aucune étape détaillée n'a été fournie."
        )


    # ========================================================
    # EXPLICATION CLASSIQUE
    # ========================================================

    st.markdown(
        "### 💡 Explication"
    )

    if explanation:

        st.markdown(
            str(explanation)
        )

    else:

        # Explication générique selon l'opération

        explanations = {

            "derivative":
                """
                La dérivée mesure le taux de variation
                d'une fonction par rapport à sa variable.
                """,

            "integral":
                """
                L'intégrale permet notamment de calculer
                une primitive ou une accumulation sur un intervalle.
                """,

            "limit":
                """
                Une limite décrit la valeur vers laquelle
                une expression tend lorsqu'une variable
                approche une valeur donnée.
                """,

            "equation":
                """
                Résoudre une équation consiste à déterminer
                les valeurs de la variable qui rendent
                l'égalité vraie.
                """,

            "simplify":
                """
                La simplification consiste à transformer
                une expression en une forme mathématiquement
                équivalente mais plus simple.
                """,

            "calculation":
                """
                L'expression numérique a été évaluée
                en respectant les règles usuelles
                de priorité des opérations.
                """,
        }

        st.markdown(
            explanations.get(
                operation,
                "Le moteur MathLab AI a analysé la question mathématique.",
            )
        )


    # ========================================================
    # EXPLICATION IA
    # ========================================================

    st.divider()

    st.markdown(
        "### 🧠 Explication IA"
    )

    st.markdown(
        """
        Le calcul est effectué par **SymPy**.
        L'API OpenAI est utilisée uniquement pour produire
        une explication pédagogique du résultat déjà calculé.
        """
    )


    # ========================================================
    # BOUTON EXPLICATION IA
    # ========================================================

    generate_ai_explanation = st.button(
        "🧠 Générer une explication IA",
        use_container_width=True,
        key="generate_ai_math_explanation",
    )


    if generate_ai_explanation:

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if expression is None:

            st.warning(
                "Impossible de générer une explication : "
                "l'expression mathématique est indisponible."
            )

        elif answer is None:

            st.warning(
                "Impossible de générer une explication : "
                "aucun résultat mathématique n'est disponible."
            )

        else:

            try:

                with st.spinner(
                    "🧠 Génération de l'explication IA..."
                ):

                    ai_explanation = ai_explain(
                        operation=operation,
                        expression=str(expression),
                        result=answer,
                        steps=steps,
                        level="lycée",
                    )

                # ------------------------------------------------
                # STOCKAGE
                # ------------------------------------------------

                st.session_state[
                    "ai_math_ai_explanation"
                ] = ai_explanation

                st.session_state[
                    "ai_math_ai_explanation_error"
                ] = None

            except Exception as error:

                st.session_state[
                    "ai_math_ai_explanation"
                ] = None

                st.session_state[
                    "ai_math_ai_explanation_error"
                ] = str(error)


    # ========================================================
    # AFFICHAGE EXPLICATION IA
    # ========================================================

    ai_explanation = st.session_state.get(
        "ai_math_ai_explanation"
    )

    ai_explanation_error = st.session_state.get(
        "ai_math_ai_explanation_error"
    )


    if ai_explanation:

        st.success(
            "Explication IA générée."
        )

        st.markdown(
            ai_explanation
        )


    elif ai_explanation_error:

        error_text = str(
            ai_explanation_error
        )

        # ----------------------------------------------------
        # CAS : CRÉDITS API ÉPUISÉS
        # ----------------------------------------------------

        if (
            "credit_balance_exhausted"
            in error_text
            or "no credits remaining"
            in error_text.lower()
            or "insufficient_quota"
            in error_text
        ):

            st.warning(
                """
                ### 💳 Crédits API OpenAI indisponibles

                L'explication IA n'a pas pu être générée
                car le compte API OpenAI ne dispose actuellement
                plus de crédits disponibles.

                Le calcul mathématique reste entièrement
                fonctionnel grâce à **SymPy**.

                Vous pouvez ajouter des crédits API depuis
                les paramètres de facturation OpenAI.
                """
            )

            with st.expander(
                "🔎 Détails techniques"
            ):

                st.code(
                    error_text,
                    language="text",
                )


        # ----------------------------------------------------
        # CAS : CLÉ API ABSENTE
        # ----------------------------------------------------

        elif (
            "OPENAI_API_KEY"
            in error_text
        ):

            st.warning(
                """
                ### 🔑 Clé API OpenAI non configurée

                Vérifiez que votre fichier `.env` contient :

                `OPENAI_API_KEY=...`

                et que `python-dotenv` est installé.
                """
            )

            with st.expander(
                "🔎 Détails techniques"
            ):

                st.code(
                    error_text,
                    language="text",
                )


        # ----------------------------------------------------
        # AUTRE ERREUR API
        # ----------------------------------------------------

        else:

            st.error(
                """
                ### ⚠️ L'explication IA est indisponible

                Le calcul mathématique reste disponible.
                L'erreur concerne uniquement le service
                d'explication IA.
                """
            )

            with st.expander(
                "🔎 Détails techniques"
            ):

                st.code(
                    error_text,
                    language="text",
                )


    else:

        st.info(
            """
            Cliquez sur **🧠 Générer une explication IA**
            pour obtenir une explication pédagogique du résultat.
            """
        )


    # ========================================================
    # RÉSULTAT BRUT — DEBUG / DÉVELOPPEMENT
    # ========================================================

    with st.expander(
        "🔧 Voir les données retournées par le moteur"
    ):

        st.json(
            result
        )


# ============================================================
# AIDE
# ============================================================

st.divider()

with st.expander(
    "📚 Types d'opérations supportées"
):

    st.markdown(
        """
        ### 🔢 Calcul

        Exemples :

        - `Calcule 2 + 3`
        - `Calcule 5 * 8`

        ### 📐 Équations

        Exemples :

        - `Résous 2*x + 5 = 15`
        - `Résous x^2 - 4 = 0`

        ### ∂ Dérivées

        Exemples :

        - `Calcule la dérivée de x^2`
        - `Calcule la dérivée de sin(x)`
        - `Calcule la dérivée de x^2 + 3*x`

        ### ∫ Intégrales

        Exemples :

        - `Calcule l'intégrale de x^2`
        - `Calcule l'intégrale de x^2 de 0 à 2`

        ### ∞ Limites

        Exemples :

        - `Calcule la limite de sin(x)/x quand x tend vers 0`
        - `Calcule la limite de x^2 quand x tend vers l'infini`

        ### 🔄 Simplification

        Exemples :

        - `Simplifie (x^2 - 1)/(x - 1)`
        - `Simplifie x + x + x`

        ---

        Le moteur utilise **SymPy** pour effectuer les calculs
        symboliques.

        L'API OpenAI intervient uniquement pour générer
        une explication pédagogique du résultat.
        """
    )


# ============================================================
# PIED DE PAGE
# ============================================================

st.caption(
    "MathLab AI — Mathematics × Computing"
)