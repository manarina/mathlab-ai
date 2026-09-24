from __future__ import annotations

from .matrices import Matrix


def _format_matrix(matrix: Matrix) -> str:
    """
    Formate une matrice sous une forme lisible.
    """
    lines = ["["]

    for row in matrix.values:
        lines.append(
            "  ["
            + ", ".join(str(value) for value in row)
            + "]"
        )

    lines.append("]")

    return "\n".join(lines)


def explain_matrix(matrix: Matrix) -> str:
    """
    Explique la structure et les dimensions d'une matrice.
    """
    matrix_text = _format_matrix(matrix)

    return (
        "### Matrice\n\n"
        "La matrice est :\n\n"
        f"{matrix_text}\n\n"
        f"Elle possède {matrix.rows} ligne(s) "
        f"et {matrix.columns} colonne(s).\n\n"
        f"Sa dimension est donc {matrix.rows} × {matrix.columns}."
    )


def explain_addition(
    first: Matrix,
    second: Matrix,
    result: Matrix,
) -> str:
    """
    Explique l'addition de deux matrices.
    """
    if first.shape != second.shape:
        raise ValueError(
            "Les matrices doivent avoir les mêmes dimensions."
        )

    lines = [
        "### Addition de matrices",
        "",
        f"Soient A = {first.values} et B = {second.values}.",
        "",
        "Pour additionner deux matrices, "
        "on additionne les éléments correspondants.",
        "",
        "Calcul élément par élément :",
        "",
    ]

    for row_index, (row_a, row_b, row_r) in enumerate(
        zip(first.values, second.values, result.values),
        start=1,
    ):
        for column_index, (a, b, r) in enumerate(
            zip(row_a, row_b, row_r),
            start=1,
        ):
            lines.append(
                f"- Élément ({row_index},{column_index}) : "
                f"{a} + {b} = {r}"
            )

    lines.extend(
        [
            "",
            f"Résultat : A + B = {result.values}",
        ]
    )

    return "\n".join(lines)


def explain_subtraction(
    first: Matrix,
    second: Matrix,
    result: Matrix,
) -> str:
    """
    Explique la soustraction de deux matrices.
    """
    if first.shape != second.shape:
        raise ValueError(
            "Les matrices doivent avoir les mêmes dimensions."
        )

    lines = [
        "### Soustraction de matrices",
        "",
        f"Soient A = {first.values} et B = {second.values}.",
        "",
        "On soustrait les éléments correspondants.",
        "",
        "Calcul élément par élément :",
        "",
    ]

    for row_index, (row_a, row_b, row_r) in enumerate(
        zip(first.values, second.values, result.values),
        start=1,
    ):
        for column_index, (a, b, r) in enumerate(
            zip(row_a, row_b, row_r),
            start=1,
        ):
            lines.append(
                f"- Élément ({row_index},{column_index}) : "
                f"{a} - {b} = {r}"
            )

    lines.extend(
        [
            "",
            f"Résultat : A - B = {result.values}",
        ]
    )

    return "\n".join(lines)


def explain_scalar_multiplication(
    matrix: Matrix,
    scalar: int | float,
    result: Matrix,
) -> str:
    """
    Explique la multiplication d'une matrice par un scalaire.
    """
    lines = [
        "### Multiplication par un scalaire",
        "",
        f"Soit A = {matrix.values}.",
        "",
        f"On multiplie chaque élément de A par {scalar}.",
        "",
        "Calcul élément par élément :",
        "",
    ]

    for row_index, (row_a, row_r) in enumerate(
        zip(matrix.values, result.values),
        start=1,
    ):
        for column_index, (a, r) in enumerate(
            zip(row_a, row_r),
            start=1,
        ):
            lines.append(
                f"- Élément ({row_index},{column_index}) : "
                f"{scalar} × {a} = {r}"
            )

    lines.extend(
        [
            "",
            f"Résultat : {scalar}A = {result.values}",
        ]
    )

    return "\n".join(lines)


def explain_multiplication(
    first: Matrix,
    second: Matrix,
    result: Matrix,
) -> str:
    """
    Explique la multiplication matricielle.
    """
    if first.columns != second.rows:
        raise ValueError(
            "Le nombre de colonnes de la première matrice "
            "doit être égal au nombre de lignes de la deuxième."
        )

    lines = [
        "### Multiplication matricielle",
        "",
        f"Soit A = {first.values}.",
        "",
        f"Soit B = {second.values}.",
        "",
        (
            f"A est de dimension {first.rows} × {first.columns} "
            f"et B est de dimension {second.rows} × {second.columns}."
        ),
        "",
        (
            "La multiplication est possible car le nombre de colonnes "
            "de A est égal au nombre de lignes de B."
        ),
        "",
        (
            f"Le résultat sera donc une matrice "
            f"{first.rows} × {second.columns}."
        ),
        "",
        "Chaque élément est obtenu en faisant le produit scalaire "
        "d'une ligne de A avec une colonne de B.",
        "",
    ]

    for row in range(first.rows):
        for column in range(second.columns):
            terms = [
                (
                    f"{first.values[row][index]} × "
                    f"{second.values[index][column]}"
                )
                for index in range(first.columns)
            ]

            expression = " + ".join(terms)

            values = [
                first.values[row][index]
                * second.values[index][column]
                for index in range(first.columns)
            ]

            numeric_expression = " + ".join(
                str(value)
                for value in values
            )

            result_value = result.values[row][column]

            lines.extend(
                [
                    f"Élément ({row + 1},{column + 1}) :",
                    f"= {expression}",
                    f"= {numeric_expression}",
                    f"= {result_value}",
                    "",
                ]
            )

    lines.append(
        f"Résultat : A × B = {result.values}"
    )

    return "\n".join(lines)


def explain_transpose(
    matrix: Matrix,
    result: Matrix,
) -> str:
    """
    Explique la transposée d'une matrice.
    """
    lines = [
        "### Transposée d'une matrice",
        "",
        f"Soit A = {matrix.values}.",
        "",
        "La transposée, notée Aᵀ, échange les lignes "
        "et les colonnes.",
        "",
        (
            f"A possède {matrix.rows} ligne(s) et "
            f"{matrix.columns} colonne(s)."
        ),
        "",
        (
            f"Aᵀ possède donc {result.rows} ligne(s) et "
            f"{result.columns} colonne(s)."
        ),
        "",
        "Les éléments sont réorganisés ainsi :",
        "",
    ]

    for row in range(matrix.rows):
        for column in range(matrix.columns):
            lines.append(
                f"- A({row + 1},{column + 1}) = "
                f"{matrix.values[row][column]} "
                f"devient Aᵀ({column + 1},{row + 1}) = "
                f"{matrix.values[row][column]}"
            )

    lines.extend(
        [
            "",
            f"Résultat : Aᵀ = {result.values}",
        ]
    )

    return "\n".join(lines)


def explain_determinant(
    matrix: Matrix,
    result: int | float,
) -> str:
    """
    Explique le calcul du déterminant.
    """
    if not matrix.is_square:
        raise ValueError(
            "Le déterminant est défini uniquement "
            "pour une matrice carrée."
        )

    size = matrix.rows

    lines = [
        "### Déterminant d'une matrice",
        "",
        f"Soit A = {matrix.values}.",
        "",
        f"A est une matrice carrée de dimension {size} × {size}.",
        "",
    ]

    if size == 1:
        value = matrix.values[0][0]

        lines.extend(
            [
                "Pour une matrice 1 × 1 :",
                "",
                "det(A) = a₁₁",
                "",
                f"det(A) = {value}",
            ]
        )

    elif size == 2:
        a = matrix.values[0][0]
        b = matrix.values[0][1]
        c = matrix.values[1][0]
        d = matrix.values[1][1]

        lines.extend(
            [
                "Pour une matrice 2 × 2 :",
                "",
                "A = [[a, b], [c, d]]",
                "",
                "Formule :",
                "",
                "det(A) = ad - bc",
                "",
                "Application :",
                "",
                f"det(A) = ({a} × {d}) - ({b} × {c})",
                "",
                f"det(A) = {(a * d)} - {(b * c)}",
                "",
                f"det(A) = {result}",
            ]
        )

    else:
        lines.extend(
            [
                (
                    "Pour une matrice de dimension supérieure à 2 × 2, "
                    "le calcul utilise une élimination de type "
                    "Gauss afin de transformer progressivement "
                    "la matrice."
                ),
                "",
                (
                    "Les échanges de lignes changent le signe du "
                    "déterminant, tandis que les opérations "
                    "d'élimination sont suivies pour conserver "
                    "la valeur correcte du déterminant."
                ),
                "",
                f"det(A) = {result}",
            ]
        )

    return "\n".join(lines)


def explain_inverse(
    matrix: Matrix,
    result: Matrix,
) -> str:
    """
    Explique l'inversion d'une matrice par la méthode de Gauss-Jordan.
    """
    if not matrix.is_square:
        raise ValueError(
            "Seule une matrice carrée peut avoir une inverse."
        )

    lines = [
        "### Inverse d'une matrice",
        "",
        f"Soit A = {matrix.values}.",
        "",
        (
            "Pour calculer l'inverse, on utilise la méthode "
            "de Gauss-Jordan."
        ),
        "",
        "On construit la matrice augmentée :",
        "",
        "[ A | I ]",
        "",
        "où I est la matrice identité de même dimension que A.",
        "",
    ]

    identity = tuple(
        tuple(
            1 if row == column else 0
            for column in range(matrix.rows)
        )
        for row in range(matrix.rows)
    )

    lines.append(
        f"I = {identity}"
    )

    lines.extend(
        [
            "",
            "On applique ensuite des opérations élémentaires "
            "sur les lignes afin d'obtenir :",
            "",
            "[ I | A⁻¹ ]",
            "",
            (
                "La partie droite de la matrice augmentée "
                "donne alors l'inverse."
            ),
            "",
            f"A⁻¹ = {result.values}",
            "",
            (
                "Condition : une matrice carrée est inversible "
                "si son déterminant est différent de zéro."
            ),
        ]
    )

    return "\n".join(lines)