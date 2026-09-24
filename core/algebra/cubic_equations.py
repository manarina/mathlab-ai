from __future__ import annotations

import sympy as sp

from core.algebra.horner import (
    find_rational_root_candidates,
    horner_division,
)


# ============================================================
# OUTILS INTERNES
# ============================================================


def _to_exact_number(
    value: sp.Expr,
) -> sp.Expr:
    """
    Convertit une valeur numérique en représentation exacte
    SymPy lorsque cela est possible.

    Exemples :

        1        -> 1
        -5       -> -5
        6.0      -> 6
        0.5      -> 1/2
        1.25     -> 5/4
    """

    value = sp.sympify(value)

    if value.is_Integer:
        return sp.Integer(value)

    if value.is_Rational:
        return sp.Rational(
            value.p,
            value.q,
        )

    if value.is_Float and value.is_integer:
        return sp.Integer(
            int(value)
        )

    if value.is_Float:
        return sp.Rational(
            str(value)
        )

    return value


def _normalize_integer_coefficients(
    coefficients: list[sp.Expr],
) -> list[sp.Integer]:
    """
    Convertit une liste de coefficients numériques
    en entiers SymPy exacts.

    La méthode des racines rationnelles nécessite
    des coefficients entiers.
    """

    normalized: list[sp.Integer] = []

    for coefficient in coefficients:

        coefficient = _to_exact_number(
            coefficient
        )

        if coefficient.is_Integer:
            normalized.append(
                sp.Integer(coefficient)
            )
            continue

        if (
            coefficient.is_Rational
            and coefficient.q == 1
        ):
            normalized.append(
                sp.Integer(
                    coefficient.p
                )
            )
            continue

        raise ValueError(
            "La méthode Horner avec recherche "
            "de racines rationnelles nécessite "
            "des coefficients entiers."
        )

    return normalized


# ============================================================
# PARSING
# ============================================================


def parse_cubic_equation(
    expression: str,
) -> sp.Poly:
    """
    Transforme une expression en polynôme cubique.

    Exemple :

        x**3 - 6*x**2 + 11*x - 6
    """

    x = sp.symbols("x")

    try:
        expr = sp.sympify(expression)
    except (
        sp.SympifyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Expression cubique invalide."
        ) from error

    polynomial = sp.Poly(
        expr,
        x,
    )

    if polynomial.degree() != 3:
        raise ValueError(
            "L'expression doit être un polynôme "
            "de degré 3."
        )

    return polynomial


# ============================================================
# DISCRIMINANT CUBIQUE
# ============================================================


def get_cubic_discriminant(
    a: sp.Expr,
    b: sp.Expr,
    c: sp.Expr,
    d: sp.Expr,
) -> sp.Expr:
    """
    Calcule le discriminant du polynôme cubique :

        ax³ + bx² + cx + d
    """

    a = sp.sympify(a)
    b = sp.sympify(b)
    c = sp.sympify(c)
    d = sp.sympify(d)

    return sp.expand(
        b**2 * c**2
        - 4 * a * c**3
        - 4 * b**3 * d
        - 27 * a**2 * d**2
        + 18 * a * b * c * d
    )


# ============================================================
# PARAMÈTRES DE CARDANO
# ============================================================


def get_cardano_parameters(
    a: sp.Expr,
    b: sp.Expr,
    c: sp.Expr,
    d: sp.Expr,
) -> dict:
    """
    Calcule les paramètres A, B, C, p, q et ΔC
    utilisés par la méthode de Cardano.

    Équation :

        ax³ + bx² + cx + d = 0

    Normalisation :

        x³ + Ax² + Bx + C = 0

    Transformation :

        x = y - A/3

    donnant :

        y³ + py + q = 0
    """

    a = _to_exact_number(a)
    b = _to_exact_number(b)
    c = _to_exact_number(c)
    d = _to_exact_number(d)

    if a == 0:
        raise ValueError(
            "a doit être différent de zéro."
        )

    A = sp.cancel(
        b / a
    )

    B = sp.cancel(
        c / a
    )

    C = sp.cancel(
        d / a
    )

    p = sp.cancel(
        B - A**2 / sp.Integer(3)
    )

    q = sp.cancel(
        sp.Rational(2, 27) * A**3
        - A * B / sp.Integer(3)
        + C
    )

    delta = sp.cancel(
        (q / sp.Integer(2))**2
        + (p / sp.Integer(3))**3
    )

    return {
        "A": A,
        "B": B,
        "C": C,
        "p": p,
        "q": q,
        "delta": delta,
    }


# ============================================================
# NATURE DES RACINES SELON CARDANO
# ============================================================


def get_cardano_root_nature(
    delta: sp.Expr,
) -> str:
    """
    Détermine la nature des racines selon
    le discriminant de Cardano.

    ΔC > 0 :
        une racine réelle et deux complexes

    ΔC = 0 :
        racines multiples

    ΔC < 0 :
        trois racines réelles distinctes
    """

    delta = sp.simplify(delta)

    if delta > 0:
        return (
            "one_real_root_and_two_complex_roots"
        )

    if delta == 0:
        return "multiple_real_roots"

    return "three_distinct_real_roots"


# ============================================================
# RACINES EXACTES
# ============================================================


def get_cubic_roots(
    polynomial: sp.Poly,
) -> list[sp.Expr]:
    """
    Retourne les racines exactes avec SymPy.
    """

    if polynomial.degree() != 3:
        raise ValueError(
            "Le polynôme doit être de degré 3."
        )

    x = polynomial.gens[0]

    return sp.solve(
        polynomial.as_expr(),
        x,
    )


# ============================================================
# FACTORISATION
# ============================================================


def factor_cubic(
    polynomial: sp.Poly,
) -> sp.Expr:
    """
    Factorise le polynôme cubique.
    """

    if polynomial.degree() != 3:
        raise ValueError(
            "Le polynôme doit être de degré 3."
        )

    return sp.factor(
        polynomial.as_expr()
    )


# ============================================================
# SOLVEUR CUBIQUE AVEC HORNER
# ============================================================


def solve_cubic_with_horner(
    polynomial: sp.Poly,
) -> dict:
    """
    Résout un polynôme cubique avec la méthode de Horner.

    Étapes :

    1. récupération des coefficients ;
    2. recherche des candidats aux racines rationnelles ;
    3. test des candidats avec Horner ;
    4. division synthétique lorsqu'une racine est trouvée ;
    5. répétition jusqu'au polynôme constant ;
    6. résolution éventuelle du polynôme restant.

    Exemple :

        x³ - 6x² + 11x - 6

    donne :

        (x - 1)(x - 2)(x - 3)

    donc :

        x = 1, 2, 3
    """

    if polynomial.degree() != 3:
        raise ValueError(
            "Le polynôme doit être de degré 3."
        )

    x = polynomial.gens[0]

    # --------------------------------------------------------
    # COEFFICIENTS EXACTS
    # --------------------------------------------------------

    coefficients = [
        _to_exact_number(coefficient)
        for coefficient in polynomial.all_coeffs()
    ]

    current_coefficients = (
        _normalize_integer_coefficients(
            coefficients
        )
    )

    tested_candidates = []
    rational_roots = []

    # --------------------------------------------------------
    # RECHERCHE DES RACINES RATIONNELLES
    # --------------------------------------------------------

    while len(current_coefficients) > 1:

        candidates = (
            find_rational_root_candidates(
                current_coefficients
            )
        )

        found = False

        for candidate in candidates:

            exact_candidate = sp.Rational(
                str(candidate)
            )

            division = horner_division(
                current_coefficients,
                exact_candidate,
            )

            remainder = _to_exact_number(
                division["remainder"]
            )

            tested_candidates.append(
                {
                    "candidate": exact_candidate,
                    "remainder": remainder,
                }
            )

            # ------------------------------------------------
            # RACINE TROUVÉE
            # ------------------------------------------------

            if remainder == 0:

                # Éviter les doublons
                if exact_candidate not in rational_roots:
                    rational_roots.append(
                        exact_candidate
                    )

                current_coefficients = [
                    _to_exact_number(value)
                    for value in division["quotient"]
                ]

                # Normalisation des coefficients
                normalized = []

                for value in current_coefficients:

                    value = _to_exact_number(
                        value
                    )

                    if (
                        value.is_Rational
                        and value.q == 1
                    ):
                        normalized.append(
                            sp.Integer(value)
                        )
                    else:
                        normalized.append(value)

                current_coefficients = normalized

                found = True
                break

        # ----------------------------------------------------
        # AUCUNE RACINE RATIONNELLE
        # ----------------------------------------------------

        if not found:
            break

    # --------------------------------------------------------
    # POLYNÔME RESTANT
    # --------------------------------------------------------

    remaining_degree = (
        len(current_coefficients) - 1
    )

    remaining_expression = sp.Integer(0)

    for index, coefficient in enumerate(
        current_coefficients
    ):
        remaining_expression += (
            coefficient
            * x ** (
                remaining_degree - index
            )
        )

    remaining_expression = sp.expand(
        remaining_expression
    )

    remaining_polynomial = sp.Poly(
        remaining_expression,
        x,
    )

    # --------------------------------------------------------
    # RACINES DU POLYNÔME RESTANT
    # --------------------------------------------------------

    remaining_roots = sp.solve(
        remaining_polynomial.as_expr(),
        x,
    )

    # --------------------------------------------------------
    # AJOUT DES RACINES RATIONNELLES RESTANTES
    # --------------------------------------------------------

    for root in remaining_roots:

        root = _to_exact_number(root)

        if (
            root.is_Rational
            and root not in rational_roots
        ):
            rational_roots.append(root)

    # --------------------------------------------------------
    # RACINES FINALES
    # --------------------------------------------------------

    roots = (
        rational_roots
        + [
            root
            for root in remaining_roots
            if root not in rational_roots
        ]
    )

    return {
        "method": "Horner",
        "roots": roots,
        "rational_roots": rational_roots,
        "tested_candidates": tested_candidates,
        "remaining_polynomial": remaining_polynomial,
        "factorized": sp.factor(
            polynomial.as_expr()
        ),
        "available": True,
    }


# ============================================================
# SOLVEUR COMPLET D'UNE ÉQUATION CUBIQUE
# ============================================================


def solve_cubic_equation(
    a: sp.Expr,
    b: sp.Expr,
    c: sp.Expr,
    d: sp.Expr,
) -> dict:
    """
    Analyse et résout une équation cubique :

        ax³ + bx² + cx + d = 0

    Le résultat contient :

        - degré
        - nature des racines
        - discriminant cubique
        - racines exactes
        - factorisation
        - paramètres de Cardano
        - résultat de Horner
    """

    a = _to_exact_number(a)
    b = _to_exact_number(b)
    c = _to_exact_number(c)
    d = _to_exact_number(d)

    if a == 0:
        raise ValueError(
            "a doit être différent de zéro "
            "pour une équation du troisième degré."
        )

    x = sp.symbols("x")

    polynomial = sp.Poly(
        a * x**3
        + b * x**2
        + c * x
        + d,
        x,
    )

    # --------------------------------------------------------
    # Discriminant
    # --------------------------------------------------------

    discriminant = get_cubic_discriminant(
        a,
        b,
        c,
        d,
    )

    # --------------------------------------------------------
    # Cardano
    # --------------------------------------------------------

    cardano = get_cardano_parameters(
        a,
        b,
        c,
        d,
    )

    # --------------------------------------------------------
    # Racines exactes
    # --------------------------------------------------------

    roots = get_cubic_roots(
        polynomial
    )

    # --------------------------------------------------------
    # Factorisation
    # --------------------------------------------------------

    factorized = factor_cubic(
        polynomial
    )

    # --------------------------------------------------------
    # Horner
    # --------------------------------------------------------

    try:
        horner_result = (
            solve_cubic_with_horner(
                polynomial
            )
        )

    except ValueError:

        horner_result = {
            "method": "Horner",
            "roots": [],
            "rational_roots": [],
            "tested_candidates": [],
            "remaining_polynomial": polynomial,
            "factorized": factorized,
            "available": False,
        }

    # --------------------------------------------------------
    # Nature des racines
    # --------------------------------------------------------

    if discriminant > 0:
        nature = (
            "three_distinct_real_roots"
        )

    elif discriminant == 0:
        nature = "multiple_real_roots"

    else:
        nature = (
            "one_real_root_and_two_complex_roots"
        )

    # --------------------------------------------------------
    # Résultat
    # --------------------------------------------------------

    return {
        "degree": 3,
        "type": nature,
        "discriminant": discriminant,
        "roots": roots,
        "factorized": factorized,
        "polynomial": polynomial,
        "cardano": cardano,
        "horner": horner_result,
        "message": (
            "Équation du troisième degré "
            "analysée et résolue."
        ),
    }