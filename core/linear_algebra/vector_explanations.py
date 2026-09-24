from __future__ import annotations

from .vectors import Vector


def explain_vector(vector: Vector) -> str:
    """
    Explique la représentation d'un vecteur.
    """
    components = ", ".join(str(value) for value in vector.values)

    return (
        f"Le vecteur v = ({components}) est un vecteur "
        f"de dimension {vector.dimension}."
    )


def explain_addition(
    first: Vector,
    second: Vector,
    result: Vector,
) -> str:
    """
    Explique étape par étape l'addition de deux vecteurs.
    """
    if first.dimension != second.dimension:
        raise ValueError(
            "Les vecteurs doivent avoir la même dimension."
        )

    lines = [
        "### Addition de vecteurs",
        "",
        (
            f"Soient u = {first.values} et "
            f"v = {second.values}."
        ),
        "",
        "On additionne les composantes correspondantes :",
        "",
    ]

    for index, (a, b, r) in enumerate(
        zip(first.values, second.values, result.values),
        start=1,
    ):
        lines.append(
            f"- Composante {index} : {a} + {b} = {r}"
        )

    lines.extend(
        [
            "",
            f"Résultat : u + v = {result.values}",
        ]
    )

    return "\n".join(lines)


def explain_subtraction(
    first: Vector,
    second: Vector,
    result: Vector,
) -> str:
    """
    Explique étape par étape la soustraction de deux vecteurs.
    """
    if first.dimension != second.dimension:
        raise ValueError(
            "Les vecteurs doivent avoir la même dimension."
        )

    lines = [
        "### Soustraction de vecteurs",
        "",
        (
            f"Soient u = {first.values} et "
            f"v = {second.values}."
        ),
        "",
        "On soustrait les composantes correspondantes :",
        "",
    ]

    for index, (a, b, r) in enumerate(
        zip(first.values, second.values, result.values),
        start=1,
    ):
        lines.append(
            f"- Composante {index} : {a} - {b} = {r}"
        )

    lines.extend(
        [
            "",
            f"Résultat : u - v = {result.values}",
        ]
    )

    return "\n".join(lines)


def explain_scalar_multiplication(
    vector: Vector,
    scalar: int | float,
    result: Vector,
) -> str:
    """
    Explique la multiplication d'un vecteur par un scalaire.
    """
    lines = [
        "### Multiplication par un scalaire",
        "",
        f"Soit u = {vector.values}.",
        f"On multiplie u par le scalaire {scalar}.",
        "",
        "Chaque composante est multipliée par le scalaire :",
        "",
    ]

    for index, (value, r) in enumerate(
        zip(vector.values, result.values),
        start=1,
    ):
        lines.append(
            f"- Composante {index} : "
            f"{scalar} × {value} = {r}"
        )

    lines.extend(
        [
            "",
            f"Résultat : {scalar}u = {result.values}",
        ]
    )

    return "\n".join(lines)


def explain_dot_product(
    first: Vector,
    second: Vector,
    result: int | float,
) -> str:
    """
    Explique étape par étape le produit scalaire.
    """
    if first.dimension != second.dimension:
        raise ValueError(
            "Les vecteurs doivent avoir la même dimension."
        )

    products = [
        a * b
        for a, b in zip(first.values, second.values)
    ]

    expression = " + ".join(
        f"({a} × {b})"
        for a, b in zip(first.values, second.values)
    )

    product_values = " + ".join(
        str(value)
        for value in products
    )

    return "\n".join(
        [
            "### Produit scalaire",
            "",
            (
                f"Soient u = {first.values} et "
                f"v = {second.values}."
            ),
            "",
            "Formule :",
            "",
            "u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ",
            "",
            "Application :",
            "",
            f"u · v = {expression}",
            "",
            f"u · v = {product_values}",
            "",
            f"u · v = {result}",
        ]
    )


def explain_norm(
    vector: Vector,
    result: float,
) -> str:
    """
    Explique le calcul de la norme euclidienne.
    """
    squared_terms = " + ".join(
        f"{value}²"
        for value in vector.values
    )

    numeric_terms = " + ".join(
        str(value**2)
        for value in vector.values
    )

    return "\n".join(
        [
            "### Norme du vecteur",
            "",
            f"Soit u = {vector.values}.",
            "",
            "Formule :",
            "",
            "‖u‖ = √(u₁² + u₂² + ... + uₙ²)",
            "",
            "Application :",
            "",
            f"‖u‖ = √({squared_terms})",
            "",
            f"‖u‖ = √({numeric_terms})",
            "",
            f"‖u‖ ≈ {result:.6f}",
        ]
    )


def explain_distance(
    first: Vector,
    second: Vector,
    result: float,
) -> str:
    """
    Explique le calcul de la distance entre deux vecteurs.
    """
    if first.dimension != second.dimension:
        raise ValueError(
            "Les vecteurs doivent avoir la même dimension."
        )

    differences = [
        a - b
        for a, b in zip(first.values, second.values)
    ]

    squared_terms = " + ".join(
        f"({difference})²"
        for difference in differences
    )

    return "\n".join(
        [
            "### Distance entre deux vecteurs",
            "",
            (
                f"Soient u = {first.values} et "
                f"v = {second.values}."
            ),
            "",
            "Formule :",
            "",
            "d(u, v) = ‖u - v‖",
            "",
            "On calcule d'abord u - v :",
            "",
            f"u - v = {tuple(differences)}",
            "",
            "Puis :",
            "",
            f"d(u, v) = √({squared_terms})",
            "",
            f"d(u, v) ≈ {result:.6f}",
        ]
    )