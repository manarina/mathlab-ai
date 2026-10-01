# ============================================================
# MATHLAB AI — AI EXPLAINER
# ============================================================
"""
Module responsable de la génération d'explications
mathématiques à l'aide d'une API d'IA externe.

IMPORTANT :
- Ce module ne réalise aucun calcul mathématique.
- Les calculs restent effectués par math_assistant.py / SymPy.
- L'API externe sert uniquement à produire une explication
  pédagogique du résultat obtenu.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv


# ============================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "gpt-5-mini"


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(
    operation: str,
    expression: str,
    result: Any,
) -> None:
    """
    Vérifie les données nécessaires à l'explication.
    """

    if not isinstance(operation, str):
        raise TypeError(
            "operation doit être une chaîne de caractères."
        )

    if not operation.strip():
        raise ValueError(
            "operation ne peut pas être vide."
        )

    if not isinstance(expression, str):
        raise TypeError(
            "expression doit être une chaîne de caractères."
        )

    if not expression.strip():
        raise ValueError(
            "expression ne peut pas être vide."
        )

    if result is None:
        raise ValueError(
            "result ne peut pas être None."
        )


# ============================================================
# CONSTRUCTION DU PROMPT
# ============================================================

def build_explanation_prompt(
    operation: str,
    expression: str,
    result: Any,
    steps: list[Any] | None = None,
    level: str = "lycée",
) -> str:
    """
    Construit le prompt envoyé à l'API d'IA.

    L'API reçoit le résultat calculé par le moteur mathématique
    et doit uniquement produire une explication pédagogique.
    """

    _validate_input(
        operation=operation,
        expression=expression,
        result=result,
    )

    operation_labels = {
        "equation": "résolution d'équation",
        "derivative": "dérivation",
        "integral": "intégration",
        "limit": "calcul de limite",
        "simplify": "simplification",
        "calculation": "calcul numérique",
    }

    operation_label = operation_labels.get(
        operation,
        operation,
    )

    steps_text = ""

    if steps:

        formatted_steps = []

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

            formatted_steps.append(
                f"{index}. {step_text}"
            )

        steps_text = "\n".join(
            formatted_steps
        )

    prompt = f"""
Tu es un professeur de mathématiques.

Ton rôle est d'expliquer clairement un résultat
mathématique qui a déjà été calculé par un moteur
mathématique fiable.

Ne recalcule pas le résultat et ne le modifie pas.

Opération :
{operation_label}

Expression :
{expression}

Résultat calculé :
{result}

Niveau demandé :
{level}
"""

    if steps_text:

        prompt += f"""

Démarche fournie par le moteur mathématique :
{steps_text}
"""

    prompt += """

Produis une explication pédagogique en français.

L'explication doit :

1. rappeler brièvement le problème ;
2. expliquer la règle mathématique utilisée ;
3. expliquer les principales étapes ;
4. confirmer le résultat calculé ;
5. rester concise et facile à comprendre.

Utilise la notation mathématique lorsque cela est utile.

Ne donne pas de réponse contradictoire avec le résultat fourni.
"""

    return prompt.strip()


# ============================================================
# CLIENT OPENAI
# ============================================================

def _get_openai_client():
    """
    Initialise le client OpenAI.

    La clé API est récupérée depuis la variable
    d'environnement OPENAI_API_KEY chargée depuis .env.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "La variable d'environnement "
            "OPENAI_API_KEY n'est pas configurée. "
            "Vérifiez votre fichier .env."
        )

    try:

        from openai import OpenAI

    except ImportError as error:

        raise ImportError(
            "Le package 'openai' n'est pas installé. "
            "Installez-le avec : pip install openai"
        ) from error

    return OpenAI(
        api_key=api_key
    )


# ============================================================
# APPEL API
# ============================================================

def explain_math_result(
    operation: str,
    expression: str,
    result: Any,
    steps: list[Any] | None = None,
    level: str = "lycée",
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Génère une explication pédagogique à partir
    d'un résultat mathématique déjà calculé.

    Parameters
    ----------
    operation:
        Opération mathématique détectée.

    expression:
        Expression mathématique originale.

    result:
        Résultat produit par math_assistant.py.

    steps:
        Étapes éventuellement produites par le moteur.

    level:
        Niveau d'explication.

    model:
        Modèle OpenAI utilisé.

    Returns
    -------
    str
        Explication générée par l'IA.
    """

    prompt = build_explanation_prompt(
        operation=operation,
        expression=expression,
        result=result,
        steps=steps,
        level=level,
    )

    client = _get_openai_client()

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    explanation = response.output_text

    if not explanation:

        raise RuntimeError(
            "L'API n'a retourné aucune explication."
        )

    return explanation.strip()


# ============================================================
# API SIMPLIFIÉE
# ============================================================

def ai_explain(
    operation: str,
    expression: str,
    result: Any,
    steps: list[Any] | None = None,
    level: str = "lycée",
) -> str:
    """
    Interface simplifiée pour l'application MathLab AI.

    Exemple
    -------
    explanation = ai_explain(
        operation="derivative",
        expression="x^2 + 3*x",
        result="2*x + 3",
    )
    """

    return explain_math_result(
        operation=operation,
        expression=expression,
        result=result,
        steps=steps,
        level=level,
    )