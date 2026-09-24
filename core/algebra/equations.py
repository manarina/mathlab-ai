from __future__ import annotations

from math import isclose, sqrt


def solve_linear_equation(a: float, b: float) -> dict:
    """
    Résout l'équation du premier degré :

        ax + b = 0

    Retourne un dictionnaire contenant :
    - le type de solution
    - les solutions
    - un message explicatif
    """

    if isclose(a, 0.0):
        if isclose(b, 0.0):
            return {
                "type": "infinite",
                "solutions": None,
                "message": "Une infinité de solutions.",
            }

        return {
            "type": "none",
            "solutions": [],
            "message": "Aucune solution.",
        }

    x = -b / a

    return {
        "type": "unique",
        "solutions": [x],
        "message": "Une solution unique.",
    }


def calculate_discriminant(
    a: float,
    b: float,
    c: float,
) -> float:
    """
    Calcule le discriminant d'une équation :

        ax² + bx + c = 0

    Δ = b² - 4ac
    """

    return b**2 - 4 * a * c


def solve_quadratic_equation(
    a: float,
    b: float,
    c: float,
) -> dict:
    """
    Résout l'équation :

        ax² + bx + c = 0

    Si a = 0, l'équation est traitée comme
    une équation du premier degré.
    """

    if isclose(a, 0.0):
        result = solve_linear_equation(b, c)

        return {
            "degree": 1,
            **result,
        }

    discriminant = calculate_discriminant(a, b, c)

    if discriminant > 0:
        sqrt_delta = sqrt(discriminant)

        x1 = (-b - sqrt_delta) / (2 * a)
        x2 = (-b + sqrt_delta) / (2 * a)

        return {
            "degree": 2,
            "type": "two_real_solutions",
            "discriminant": discriminant,
            "solutions": sorted([x1, x2]),
            "message": "Deux solutions réelles distinctes.",
        }

    if isclose(discriminant, 0.0):
        x = -b / (2 * a)

        return {
            "degree": 2,
            "type": "one_real_solution",
            "discriminant": 0.0,
            "solutions": [x],
            "message": "Une solution réelle double.",
        }

    return {
        "degree": 2,
        "type": "no_real_solution",
        "discriminant": discriminant,
        "solutions": [],
        "message": "Aucune solution réelle.",
    }