
"""
Méthodes d'interpolation numérique.

Ce module fournit les méthodes suivantes :

- Interpolation de Lagrange
- Interpolation de Newton
- Différences divisées
- Évaluation d'un polynôme interpolateur

Les polynômes sont représentés sous forme d'expressions SymPy.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import sympy as sp


# Variable symbolique utilisée par les polynômes interpolateurs.
x = sp.Symbol("x")


def _validate_interpolation_data(
    x_values: Sequence[Real],
    y_values: Sequence[Real],
) -> None:
    """
    Vérifie que les données d'interpolation sont valides.

    Conditions :
    - x_values et y_values doivent être des séquences.
    - Les deux séquences doivent avoir la même longueur.
    - Il faut au moins deux points.
    - Les valeurs doivent être numériques.
    - Les abscisses doivent être distinctes.

    Raises
    ------
    TypeError
        Si les données ne sont pas des séquences valides ou contiennent
        des valeurs non numériques.
    ValueError
        Si les dimensions sont différentes, s'il y a moins de deux points
        ou si deux abscisses sont identiques.
    """

    if isinstance(x_values, (str, bytes)) or isinstance(
        y_values, (str, bytes)
    ):
        raise TypeError(
            "Les valeurs x et y doivent être des séquences numériques."
        )

    if not isinstance(x_values, Sequence) or not isinstance(
        y_values, Sequence
    ):
        raise TypeError(
            "Les valeurs x et y doivent être des séquences numériques."
        )

    if len(x_values) != len(y_values):
        raise ValueError(
            "Les séquences x et y doivent avoir la même longueur."
        )

    if len(x_values) < 2:
        raise ValueError(
            "L'interpolation nécessite au moins deux points."
        )

    for value in x_values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "Toutes les abscisses x doivent être numériques."
            )

    for value in y_values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "Toutes les ordonnées y doivent être numériques."
            )

    if len(set(x_values)) != len(x_values):
        raise ValueError(
            "Les abscisses x doivent être distinctes."
        )


def lagrange_interpolation(
    x_values: Sequence[Real],
    y_values: Sequence[Real],
) -> sp.Expr:
    """
    Construit le polynôme interpolateur de Lagrange.

    Pour les points :

        (x_0, y_0), ..., (x_n, y_n)

    le polynôme est défini par :

        P(x) = Σ y_i L_i(x)

    avec :

        L_i(x) = Π_{j != i} (x - x_j) / (x_i - x_j)

    Parameters
    ----------
    x_values:
        Abscisses des points d'interpolation.
    y_values:
        Ordonnées des points d'interpolation.

    Returns
    -------
    sympy.Expr
        Polynôme interpolateur développé et simplifié.

    Examples
    --------
    >>> lagrange_interpolation([0, 1, 2], [1, 3, 7])
    x**2 + x + 1
    """

    _validate_interpolation_data(x_values, y_values)

    polynomial = sp.Integer(0)

    for i, (xi, yi) in enumerate(zip(x_values, y_values)):
        basis_polynomial = sp.Integer(1)

        for j, xj in enumerate(x_values):
            if i == j:
                continue

            basis_polynomial *= (x - sp.sympify(xj)) / (
                sp.sympify(xi) - sp.sympify(xj)
            )

        polynomial += sp.sympify(yi) * basis_polynomial

    return sp.expand(sp.simplify(polynomial))


def divided_differences(
    x_values: Sequence[Real],
    y_values: Sequence[Real],
) -> list[sp.Expr]:
    """
    Calcule les coefficients des différences divisées de Newton.

    Pour les points :

        (x_0, y_0), ..., (x_n, y_n)

    la fonction retourne :

        [f[x_0], f[x_0,x_1], ..., f[x_0,...,x_n]]

    Ces coefficients sont utilisés pour construire le polynôme
    interpolateur de Newton.

    Parameters
    ----------
    x_values:
        Abscisses des points d'interpolation.
    y_values:
        Ordonnées des points d'interpolation.

    Returns
    -------
    list[sympy.Expr]
        Première colonne du tableau des différences divisées.

    Examples
    --------
    >>> divided_differences([0, 1, 2], [1, 3, 7])
    [1, 2, 1]
    """

    _validate_interpolation_data(x_values, y_values)

    x_values_sympy = [sp.sympify(value) for value in x_values]
    coefficients = [sp.sympify(value) for value in y_values]

    # Construction du tableau des différences divisées.
    #
    # À chaque étape, coefficients[i] contient :
    #
    # f[x_i, ..., x_{i+j}]
    #
    # On travaille directement sur une copie de la colonne courante.
    for order in range(1, len(x_values_sympy)):
        for i in range(len(x_values_sympy) - 1, order - 1, -1):
            numerator = coefficients[i] - coefficients[i - 1]
            denominator = (
                x_values_sympy[i]
                - x_values_sympy[i - order]
            )

            coefficients[i] = sp.simplify(
                numerator / denominator
            )

    # La représentation ci-dessus stocke les coefficients dans
    # les dernières positions. On reconstruit explicitement la
    # première colonne du tableau des différences divisées.
    table = [
        [sp.sympify(value) for value in y_values]
    ]

    for order in range(1, len(x_values_sympy)):
        previous_column = table[-1]
        current_column = []

        for i in range(len(previous_column) - 1):
            numerator = previous_column[i + 1] - previous_column[i]
            denominator = (
                x_values_sympy[i + order]
                - x_values_sympy[i]
            )

            current_column.append(
                sp.simplify(numerator / denominator)
            )

        table.append(current_column)

    return [
        sp.simplify(table[order][0])
        for order in range(len(table))
    ]


def newton_interpolation(
    x_values: Sequence[Real],
    y_values: Sequence[Real],
) -> sp.Expr:
    """
    Construit le polynôme interpolateur de Newton.

    Le polynôme est construit sous la forme :

        P(x) =
            c_0
            + c_1(x-x_0)
            + c_2(x-x_0)(x-x_1)
            + ...

    où les coefficients c_i sont les différences divisées.

    Parameters
    ----------
    x_values:
        Abscisses des points d'interpolation.
    y_values:
        Ordonnées des points d'interpolation.

    Returns
    -------
    sympy.Expr
        Polynôme interpolateur développé et simplifié.

    Examples
    --------
    >>> newton_interpolation([0, 1, 2], [1, 3, 7])
    x**2 + x + 1
    """

    _validate_interpolation_data(x_values, y_values)

    coefficients = divided_differences(
        x_values,
        y_values,
    )

    x_values_sympy = [
        sp.sympify(value)
        for value in x_values
    ]

    polynomial = coefficients[0]
    product_term = sp.Integer(1)

    for i in range(1, len(coefficients)):
        product_term *= x - x_values_sympy[i - 1]
        polynomial += coefficients[i] * product_term

    return sp.expand(sp.simplify(polynomial))


def evaluate_interpolating_polynomial(
    polynomial: sp.Expr,
    value: Real,
) -> sp.Expr:
    """
    Évalue un polynôme interpolateur en un point donné.

    Parameters
    ----------
    polynomial:
        Expression SymPy représentant le polynôme.
    value:
        Valeur de x où le polynôme doit être évalué.

    Returns
    -------
    sympy.Expr
        Valeur exacte lorsque cela est possible.

    Raises
    ------
    TypeError
        Si polynomial n'est pas une expression SymPy ou si value
        n'est pas numérique.

    Examples
    --------
    >>> polynomial = lagrange_interpolation([0, 1, 2], [1, 3, 7])
    >>> evaluate_interpolating_polynomial(polynomial, 1.5)
    19/4
    """

    if not isinstance(polynomial, sp.Expr):
        raise TypeError(
            "Le polynôme doit être une expression SymPy."
        )

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "La valeur d'évaluation doit être numérique."
        )

    result = polynomial.subs(x, sp.sympify(value))

    return sp.simplify(result)

