from __future__ import annotations

from typing import Sequence

import sympy as sp


def horner_evaluate(
    coefficients: Sequence[float],
    value: float,
) -> float:
    """
    Évalue un polynôme avec la méthode de Horner.

    Pour :

        P(x) = a_n x^n + ... + a_1 x + a_0

    coefficients doit être fourni dans l'ordre :

        [a_n, ..., a_1, a_0]

    Exemple :

        P(x) = x³ - 6x² + 11x - 6

        coefficients = [1, -6, 11, -6]
    """

    if not coefficients:
        raise ValueError(
            "La liste des coefficients ne peut pas être vide."
        )

    result = coefficients[0]

    for coefficient in coefficients[1:]:
        result = result * value + coefficient

    return result


def horner_steps(
    coefficients: Sequence[float],
    value: float,
) -> list[dict]:
    """
    Retourne les étapes intermédiaires de Horner.

    Chaque étape contient :
        - coefficient
        - valeur précédente
        - calcul
        - résultat
    """

    if not coefficients:
        raise ValueError(
            "La liste des coefficients ne peut pas être vide."
        )

    steps = []

    result = coefficients[0]

    steps.append(
        {
            "index": 0,
            "coefficient": coefficients[0],
            "previous": None,
            "calculation": str(coefficients[0]),
            "result": result,
        }
    )

    for index, coefficient in enumerate(
        coefficients[1:],
        start=1,
    ):
        previous = result
        result = previous * value + coefficient

        steps.append(
            {
                "index": index,
                "coefficient": coefficient,
                "previous": previous,
                "calculation": (
                    f"({previous}) × ({value}) + "
                    f"({coefficient})"
                ),
                "result": result,
            }
        )

    return steps


def horner_division(
    coefficients: Sequence[int],
    root: float,
) -> dict:
    """
    Effectue une division synthétique de Horner
    par (x - root).

    Les calculs sont effectués avec SymPy afin
    de conserver une représentation exacte.
    """

    if not coefficients:
        raise ValueError(
            "La liste des coefficients ne peut pas être vide."
        )

    # Conversion exacte des coefficients
    coefficients = [
        sp.Rational(str(coefficient))
        if isinstance(coefficient, float)
        else sp.sympify(coefficient)
        for coefficient in coefficients
    ]

    # Conversion exacte de la racine
    if isinstance(root, float):
        root = sp.Rational(str(root))
    else:
        root = sp.sympify(root)

    quotient = [coefficients[0]]

    for coefficient in coefficients[1:-1]:
        value = sp.cancel(
            coefficient
            + quotient[-1] * root
        )

        quotient.append(value)

    remainder = sp.cancel(
        coefficients[-1]
        + quotient[-1] * root
    )

    return {
        "quotient": quotient,
        "remainder": remainder,
        "is_root": remainder == 0,
    }

def find_rational_root_candidates(
    coefficients: Sequence[int],
) -> list[float]:
    """
    Détermine les candidats aux racines rationnelles
    à partir du théorème des racines rationnelles.

    Pour un polynôme :

        a_n x^n + ... + a_1 x + a_0

    les racines rationnelles possibles sont de la forme :

        ± p / q

    avec :
        p diviseur du terme constant
        q diviseur du coefficient dominant.
    """

    if not coefficients:
        raise ValueError(
            "La liste des coefficients ne peut pas être vide."
        )

    sympy_coefficients = [
        sp.sympify(coefficient)
        for coefficient in coefficients
    ]

    # Convertir les valeurs comme 1.0, -5.0, 6.0
    # en entiers exacts 1, -5, 6.
    normalized_coefficients = []

    for coefficient in sympy_coefficients:
        if coefficient.is_Integer:
            normalized_coefficients.append(
                sp.Integer(coefficient)
            )

        elif coefficient.is_Float and coefficient.is_integer:
            normalized_coefficients.append(
                sp.Integer(int(coefficient))
            )

        else:
            raise ValueError(
                "La recherche des racines rationnelles "
                "nécessite des coefficients entiers."
            )

    leading = normalized_coefficients[0]
    constant = normalized_coefficients[-1]

    if leading == 0:
        raise ValueError(
            "Le coefficient dominant doit être différent de zéro."
        )

    # Si le terme constant est nul,
    # alors 0 est une racine.
    if constant == 0:
        return [0.0]

    numerator_divisors = sp.divisors(
        abs(constant)
    )

    denominator_divisors = sp.divisors(
        abs(leading)
    )

    candidates: set[float] = set()

    for numerator in numerator_divisors:
        for denominator in denominator_divisors:

            candidate = float(
                sp.Rational(
                    numerator,
                    denominator,
                )
            )

            candidates.add(candidate)
            candidates.add(-candidate)

    return sorted(candidates)