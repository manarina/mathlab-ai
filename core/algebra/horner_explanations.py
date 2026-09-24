from __future__ import annotations

from typing import Sequence


def explain_horner_evaluation(
    coefficients: Sequence[float],
    value: float,
) -> list[dict]:
    """
    Génère une explication pédagogique de l'évaluation
    d'un polynôme par la méthode de Horner.
    """

    if not coefficients:
        raise ValueError(
            "Les coefficients ne peuvent pas être vides."
        )

    steps = []

    degree = len(coefficients) - 1

    steps.append(
        {
            "title": "Écrire le polynôme",
            "formula": (
                f"P(x) avec un degré {degree}"
            ),
            "explanation": (
                "On commence par identifier les coefficients "
                "du polynôme, du terme de plus haut degré "
                "jusqu'au terme constant."
            ),
        }
    )

    steps.append(
        {
            "title": "Choisir la valeur",
            "formula": f"x = {value}",
            "explanation": (
                "Nous allons calculer la valeur du polynôme "
                f"pour x = {value}."
            ),
        }
    )

    current = coefficients[0]

    steps.append(
        {
            "title": "Initialiser Horner",
            "formula": f"b_0 = {current}",
            "explanation": (
                "La première valeur intermédiaire est "
                "égale au premier coefficient."
            ),
        }
    )

    for index, coefficient in enumerate(
        coefficients[1:],
        start=1,
    ):
        previous = current
        current = previous * value + coefficient

        steps.append(
            {
                "title": f"Étape {index}",
                "formula": (
                    f"b_{index} = "
                    f"b_{index - 1} × {value} "
                    f"+ ({coefficient})"
                    f" = {current}"
                ),
                "explanation": (
                    "On multiplie le résultat précédent "
                    "par x puis on ajoute le coefficient "
                    "suivant."
                ),
            }
        )

    steps.append(
        {
            "title": "Résultat",
            "formula": f"P({value}) = {current}",
            "explanation": (
                "Le dernier résultat obtenu par Horner "
                "est la valeur du polynôme."
            ),
        }
    )

    return steps


def explain_horner_division(
    coefficients: Sequence[float],
    root: float,
    division_result: dict,
) -> list[dict]:
    """
    Génère les étapes pédagogiques d'une division
    de Horner par (x - root).
    """

    steps = []

    quotient = division_result["quotient"]
    remainder = division_result["remainder"]

    steps.append(
        {
            "title": "Choisir la racine",
            "formula": f"x = {root}",
            "explanation": (
                f"On cherche à diviser le polynôme "
                f"par (x - {root})."
            ),
        }
    )

    steps.append(
        {
            "title": "Appliquer Horner",
            "formula": (
                "Chaque coefficient intermédiaire "
                "est obtenu en multipliant le résultat "
                "précédent par la racine puis en ajoutant "
                "le coefficient suivant."
            ),
            "explanation": (
                "La méthode de Horner permet ici "
                "d'effectuer efficacement la division "
                "du polynôme par (x - r)."
            ),
        }
    )

    steps.append(
        {
            "title": "Obtenir le quotient",
            "formula": (
                f"Coefficients du quotient : {quotient}"
            ),
            "explanation": (
                "Les valeurs intermédiaires, sauf la "
                "dernière, forment les coefficients "
                "du polynôme quotient."
            ),
        }
    )

    steps.append(
        {
            "title": "Calculer le reste",
            "formula": f"R = {remainder}",
            "explanation": (
                "Le dernier résultat de Horner correspond "
                "au reste de la division."
            ),
        }
    )

    if remainder == 0:
        explanation = (
            f"Le reste est nul. Donc {root} est une "
            "racine du polynôme."
        )
    else:
        explanation = (
            f"Le reste n'est pas nul. Donc {root} "
            "n'est pas une racine du polynôme."
        )

    steps.append(
        {
            "title": "Interpréter le résultat",
            "formula": (
                f"P({root}) = {remainder}"
            ),
            "explanation": explanation,
        }
    )

    return steps