"""
MathLab AI - Moteur mathématique de l'assistant.

Version V1 :

- équations
- dérivées
- intégrales
- limites
- simplification d'expressions
- calcul d'expressions

Le moteur utilise SymPy et fonctionne localement,
sans API externe ni modèle de langage.
"""

from __future__ import annotations

import re
from typing import Any

import sympy as sp


# ============================================================
# CONSTANTES
# ============================================================

SUPPORTED_OPERATIONS = (
    "equation",
    "derivative",
    "integral",
    "limit",
    "simplify",
    "calculate",
)


# ============================================================
# OUTILS DE VALIDATION
# ============================================================


def _validate_question(question: str) -> str:
    """
    Valide et normalise la question utilisateur.
    """

    if not isinstance(question, str):
        raise TypeError(
            "La question doit être une chaîne de caractères."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "La question ne peut pas être vide."
        )

    return question


# ============================================================
# NORMALISATION
# ============================================================


def normalize_expression(expression: str) -> str:
    """
    Normalise une expression mathématique simple.

    Exemples :
        x^2   -> x**2
        2x    -> 2*x
        π     -> pi
        ∞     -> oo
    """

    if not isinstance(expression, str):
        raise TypeError(
            "L'expression doit être une chaîne de caractères."
        )

    expression = expression.strip()

    if not expression:
        raise ValueError(
            "L'expression ne peut pas être vide."
        )

    # --------------------------------------------------------
    # Symboles mathématiques
    # --------------------------------------------------------

    expression = expression.replace("^", "**")
    expression = expression.replace("π", "pi")
    expression = expression.replace("∞", "oo")

    # --------------------------------------------------------
    # Multiplication implicite
    # --------------------------------------------------------

    # 2x -> 2*x
    expression = re.sub(
        r"(\d)([a-zA-Z])",
        r"\1*\2",
        expression,
    )

    # 2(x+1) -> 2*(x+1)
    expression = re.sub(
        r"(\d)\s*\(",
        r"\1*(",
        expression,
    )

    # )x -> )*x
    expression = re.sub(
        r"\)\s*([a-zA-Z])",
        r")*\1",
        expression,
    )

    # )( -> )*(
    expression = re.sub(
        r"\)\s*\(",
        r")*(",
        expression,
    )

    return expression


# ============================================================
# PARSING SYMPY
# ============================================================


def parse_expression(expression: str) -> sp.Expr:
    """
    Transforme une expression textuelle en expression SymPy.

    Cette fonction doit recevoir uniquement une expression
    mathématique, et non une phrase naturelle.
    """

    normalized = normalize_expression(expression)

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")

    local_dict = {
        "x": x,
        "y": y,
        "z": z,
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "oo": sp.oo,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
    }

    try:
        result = sp.sympify(
            normalized,
            locals=local_dict,
        )

    except (
        sp.SympifyError,
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"Expression mathématique invalide : {expression}"
        ) from exc

    return result


# ============================================================
# DÉTECTION DU TYPE DE PROBLÈME
# ============================================================


def detect_operation(question: str) -> str:
    """
    Détecte le type d'opération demandé.

    Retourne :

        equation
        derivative
        integral
        limit
        simplify
        calculate
    """

    question = _validate_question(question)

    text = question.lower()

    # --------------------------------------------------------
    # ÉQUATION
    # --------------------------------------------------------

    equation_keywords = (
        "résous",
        "résoudre",
        "résolution",
        "solution de",
        "solutions de",
        "solve",
        "équation",
        "equation",
    )

    if (
        any(
            keyword in text
            for keyword in equation_keywords
        )
        or "=" in text
    ):
        return "equation"

    # --------------------------------------------------------
    # DÉRIVÉE
    # --------------------------------------------------------

    derivative_keywords = (
        "dérivée",
        "derivee",
        "dérive",
        "deriver",
        "dériver",
        "différentiel",
        "differentielle",
        "différencier",
        "derivative",
        "differentiate",
    )

    if any(
        keyword in text
        for keyword in derivative_keywords
    ):
        return "derivative"

    # --------------------------------------------------------
    # INTÉGRALE
    # --------------------------------------------------------

    integral_keywords = (
        "intégrale",
        "integrale",
        "intégrer",
        "integrer",
        "intégration",
        "integration",
        "primitive",
        "integral",
    )

    if any(
        keyword in text
        for keyword in integral_keywords
    ):
        return "integral"

    # --------------------------------------------------------
    # LIMITE
    # --------------------------------------------------------

    limit_keywords = (
        "limite",
        "lim",
        "tend vers",
        "tendant vers",
        "limit",
    )

    if any(
        keyword in text
        for keyword in limit_keywords
    ):
        return "limit"

    # --------------------------------------------------------
    # SIMPLIFICATION
    # --------------------------------------------------------

    simplify_keywords = (
        "simplifie",
        "simplifier",
        "simplification",
        "réduis",
        "reduis",
        "réduire",
        "simplify",
    )

    if any(
        keyword in text
        for keyword in simplify_keywords
    ):
        return "simplify"

    # --------------------------------------------------------
    # CALCUL
    # --------------------------------------------------------

    calculate_keywords = (
        "calcule",
        "calculer",
        "calcul",
        "évalue",
        "evaluer",
        "évaluer",
        "value",
        "calculate",
    )

    if any(
        keyword in text
        for keyword in calculate_keywords
    ):
        return "calculate"

    # Par défaut, on considère la demande comme
    # une expression à simplifier.

    return "simplify"


# ============================================================
# EXTRACTION GÉNÉRALE DE L'EXPRESSION
# ============================================================


def extract_expression(question: str) -> str:
    """
    Extrait une expression mathématique d'une question naturelle.

    Cette fonction reste volontairement générale.

    Exemples :

        "Calcule 2 + 3"
            -> "2 + 3"

        "Simplifie (x + 1)^2"
            -> "(x + 1)^2"

        "Résous 2*x + 5 = 15"
            -> "2*x + 5 = 15"
    """

    question = _validate_question(question)

    text = question.strip()

    # --------------------------------------------------------
    # Expressions du type f(x) = ...
    # --------------------------------------------------------

    match = re.search(
        r"(?:f\s*\(\s*x\s*\)|y)\s*=\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip(
            " ?!."
        )

    # --------------------------------------------------------
    # Suppression des formulations générales
    # --------------------------------------------------------

    patterns = [
        r"^(?:résous|résoudre|solve)\s+",
        r"^(?:calcule|calculer|calculate)\s+",
        r"^(?:simplifie|simplifier|simplify)\s+",
        r"^(?:évalue|evaluer|évaluer)\s+",
        r"^(?:expression)\s*[:\-]\s*",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    text = text.strip(" ?!.")

    return text


# ============================================================
# EXTRACTION D'UNE EXPRESSION DE DÉRIVÉE
# ============================================================


def extract_derivative_expression(
    question: str,
) -> str:
    """
    Extrait uniquement l'expression à dériver.

    Exemples :

        "Calcule la dérivée de x^2"
            -> "x^2"

        "Dérivée de sin(x)"
            -> "sin(x)"

        "Quelle est la dérivée de 3*x + 2 ?"
            -> "3*x + 2"
    """

    question = _validate_question(question)

    text = question.strip()

    patterns = [
        r"^(?:calcule\s+)?la\s+dérivée\s+de\s+(.+)$",
        r"^(?:calcule\s+)?la\s+derivee\s+de\s+(.+)$",
        r"^(?:calcule\s+)?dérivée\s+de\s+(.+)$",
        r"^(?:calcule\s+)?derivee\s+de\s+(.+)$",
        r"^(?:calcule\s+)?dérive\s+(.+)$",
        r"^(?:calcule\s+)?deriver\s+(.+)$",
        r"^(?:calcule\s+)?dériver\s+(.+)$",
        r"^(?:calculate\s+)?the\s+derivative\s+of\s+(.+)$",
        r"^(?:calculate\s+)?derivative\s+of\s+(.+)$",
        r"^derivative\s+of\s+(.+)$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            expression = match.group(1)

            expression = re.sub(
                r"\s+(?:par\s+rapport\s+à|selon)\s+x\s*$",
                "",
                expression,
                flags=re.IGNORECASE,
            )

            return expression.strip(
                " ?!."
            )

    # Fallback
    return extract_expression(question)


# ============================================================
# EXTRACTION D'UNE EXPRESSION D'INTÉGRALE
# ============================================================


def extract_integral_expression(
    question: str,
) -> str:
    """
    Extrait uniquement l'expression à intégrer.

    Exemples :

        "Calcule l'intégrale de x^2"
            -> "x^2"

        "L'intégrale de x^2 de 0 à 2"
            -> "x^2"

        "Intègre sin(x)"
            -> "sin(x)"
    """

    question = _validate_question(question)

    text = question.strip()

    patterns = [
        r"^(?:calcule\s+)?l['’]intégrale\s+de\s+(.+)$",
        r"^(?:calcule\s+)?l['’]integrale\s+de\s+(.+)$",
        r"^(?:calcule\s+)?intégrale\s+de\s+(.+)$",
        r"^(?:calcule\s+)?integrale\s+de\s+(.+)$",
        r"^(?:calcule\s+)?intègre\s+(.+)$",
        r"^(?:calcule\s+)?integre\s+(.+)$",
        r"^(?:calcule\s+)?intégrer\s+(.+)$",
        r"^(?:calcule\s+)?integrer\s+(.+)$",
        r"^(?:calculate\s+)?the\s+integral\s+of\s+(.+)$",
        r"^(?:calculate\s+)?integral\s+of\s+(.+)$",
        r"^integral\s+of\s+(.+)$",
    ]

    expression = None

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            expression = match.group(1)
            break

    if expression is None:
        expression = extract_expression(question)

    # --------------------------------------------------------
    # Retirer les bornes
    # --------------------------------------------------------

    expression = re.sub(
        r"\s+de\s+"
        r"[+\-]?\d+(?:\.\d+)?"
        r"\s+à\s+"
        r"[+\-]?\d+(?:\.\d+)?"
        r"\s*$",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    expression = re.sub(
        r"\s+entre\s+"
        r"[+\-]?\d+(?:\.\d+)?"
        r"\s+et\s+"
        r"[+\-]?\d+(?:\.\d+)?"
        r"\s*$",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    # Cas de bornes x=0 à x=2
    expression = re.sub(
        r"\s+de\s+x\s*=\s*"
        r"[+\-]?\d+(?:\.\d+)?"
        r"\s+à\s+x\s*=\s*"
        r"[+\-]?\d+(?:\.\d+)?"
        r"\s*$",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    return expression.strip(
        " ?!."
    )


# ============================================================
# EXTRACTION D'UNE LIMITE
# ============================================================


def extract_limit_request(
    question: str,
) -> tuple[str, str]:
    """
    Extrait :

        - l'expression de la limite
        - le point vers lequel x tend

    Exemples :

        "Calcule la limite de sin(x)/x quand x tend vers 0"

        retourne :

        ("sin(x)/x", "0")
    """

    question = _validate_question(question)

    text = question.strip()

    # --------------------------------------------------------
    # Trouver le point
    # --------------------------------------------------------

    point = extract_limit_point(text)

    # --------------------------------------------------------
    # Extraire l'expression avant "quand x..."
    # --------------------------------------------------------

    patterns = [
        r"^(?:calcule\s+)?la\s+limite\s+de\s+(.+?)"
        r"\s+quand\s+x\s+tend\s+vers\s+.+$",

        r"^(?:calcule\s+)?la\s+limite\s+de\s+(.+?)"
        r"\s+lorsque\s+x\s+tend\s+vers\s+.+$",

        r"^(?:calcule\s+)?limite\s+de\s+(.+?)"
        r"\s+quand\s+x\s+tend\s+vers\s+.+$",

        r"^(?:calcule\s+)?limite\s+de\s+(.+?)"
        r"\s+lorsque\s+x\s+tend\s+vers\s+.+$",

        r"^(?:calculate\s+)?the\s+limit\s+of\s+(.+?)"
        r"\s+as\s+x\s+tends\s+to\s+.+$",

        r"^limit\s+of\s+(.+?)"
        r"\s+as\s+x\s+tends\s+to\s+.+$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            expression = match.group(1)

            return (
                expression.strip(" ?!."),
                point,
            )

    # --------------------------------------------------------
    # Cas x -> 0
    # --------------------------------------------------------

    for marker in (
        "→",
        "->",
    ):

        if marker in text:

            before, after = text.split(
                marker,
                1,
            )

            expression = before

            expression = re.sub(
                r"^(?:calcule\s+)?"
                r"(?:la\s+)?limite\s+"
                r"(?:de\s+)?",
                "",
                expression,
                flags=re.IGNORECASE,
            )

            return (
                expression.strip(" ?!."),
                point,
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    expression = re.sub(
        r"^(?:calcule\s+)?"
        r"(?:la\s+)?limite\s+"
        r"(?:de\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    expression = re.sub(
        r"\s+quand\s+x\s+tend\s+vers\s+.+$",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    expression = re.sub(
        r"\s+lorsque\s+x\s+tend\s+vers\s+.+$",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    return (
        expression.strip(" ?!."),
        point,
    )


# ============================================================
# EXTRACTION DU POINT D'UNE LIMITE
# ============================================================


def extract_limit_point(
    question: str,
) -> str:
    """
    Extrait le point vers lequel x tend.

    Exemples :

        x -> 0
        x tend vers 0
        x tend vers +inf
        quand x tend vers 2
    """

    question = _validate_question(question)

    patterns = [
        r"x\s*(?:->|→)\s*"
        r"([+\-]?(?:\d+(?:\.\d+)?|inf|infinity|∞))",

        r"x\s+tend\s+vers\s+"
        r"([+\-]?(?:\d+(?:\.\d+)?|inf|infinity|∞))",

        r"x\s+tendant\s+vers\s+"
        r"([+\-]?(?:\d+(?:\.\d+)?|inf|infinity|∞))",

        r"x\s+tends\s+to\s+"
        r"([+\-]?(?:\d+(?:\.\d+)?|inf|infinity|∞))",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    # Par défaut : x -> 0
    return "0"


# ============================================================
# EXTRACTION DES BORNES D'INTÉGRALE
# ============================================================


def extract_integral_bounds(
    question: str,
) -> tuple[str | None, str | None]:
    """
    Extrait les bornes d'une intégrale.

    Reconnaît notamment :

        de 0 à 2
        entre 0 et 2
        de x=0 à x=2
    """

    question = _validate_question(question)

    patterns = [
        # de x=0 à x=2
        r"de\s+x\s*=\s*"
        r"([+\-]?\d+(?:\.\d+)?)"
        r"\s+à\s+x\s*=\s*"
        r"([+\-]?\d+(?:\.\d+)?)",

        # de 0 à 2
        r"de\s+"
        r"([+\-]?\d+(?:\.\d+)?)"
        r"\s+à\s+"
        r"([+\-]?\d+(?:\.\d+)?)",

        # entre 0 et 2
        r"entre\s+"
        r"([+\-]?\d+(?:\.\d+)?)"
        r"\s+et\s+"
        r"([+\-]?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )

        if match:
            return (
                match.group(1),
                match.group(2),
            )

    return None, None


# ============================================================
# ÉQUATIONS
# ============================================================


def solve_equation(
    question: str,
) -> dict[str, Any]:
    """
    Résout une équation par rapport à x.
    """

    left_text, right_text = extract_equation(
        question
    )

    left = parse_expression(
        left_text
    )

    right = parse_expression(
        right_text
    )

    equation = sp.Eq(
        left,
        right,
    )

    symbols = sorted(
        equation.free_symbols,
        key=lambda symbol: symbol.name,
    )

    if not symbols:

        simplified = sp.simplify(
            left - right
        )

        if simplified == 0:
            solutions = [
                "Vraie pour tout x."
            ]
        else:
            solutions = [
                "Aucune solution."
            ]

        return {
            "operation": "equation",
            "expression": equation,
            "steps": [
                f"{sp.sstr(left)} = "
                f"{sp.sstr(right)}",
            ],
            "result": solutions,
            "explanation": (
                "L'équation ne contient pas de variable."
            ),
        }

    variable = (
        sp.Symbol("x")
        if sp.Symbol("x") in symbols
        else symbols[0]
    )

    solutions = sp.solve(
        equation,
        variable,
    )

    steps = [
        (
            f"Équation initiale : "
            f"{sp.sstr(left)} = "
            f"{sp.sstr(right)}"
        ),
        (
            f"On regroupe les termes pour résoudre "
            f"par rapport à {variable}."
        ),
        (
            f"Solutions : "
            f"{sp.sstr(solutions)}"
        ),
    ]

    return {
        "operation": "equation",
        "expression": equation,
        "variable": variable,
        "steps": steps,
        "result": solutions,
        "explanation": (
            f"L'équation a été résolue par rapport à "
            f"{variable}."
        ),
    }


# ============================================================
# EXTRACTION D'UNE ÉQUATION
# ============================================================


def extract_equation(
    question: str,
) -> tuple[str, str]:
    """
    Extrait les deux membres d'une équation.

    Exemple :

        "Résous 2x + 5 = 15"

    retourne :

        ("2x + 5", "15")
    """

    question = _validate_question(
        question
    )

    if "=" not in question:
        raise ValueError(
            "Aucune équation contenant '=' n'a été trouvée."
        )

    equation = question

    equation = re.sub(
        r"^(?:résous|résoudre|solve|"
        r"équation|equation)\s+",
        "",
        equation,
        flags=re.IGNORECASE,
    )

    if equation.count("=") != 1:
        raise ValueError(
            "L'équation doit contenir exactement un signe '='."
        )

    left, right = equation.split(
        "=",
        1,
    )

    left = left.strip(
        " ?!."
    )

    right = right.strip(
        " ?!."
    )

    if not left or not right:
        raise ValueError(
            "Les deux membres de l'équation sont nécessaires."
        )

    return left, right


# ============================================================
# DÉRIVÉES
# ============================================================


def calculate_derivative(
    question: str,
) -> dict[str, Any]:
    """
    Calcule la dérivée première par rapport à x.
    """

    expression_text = extract_derivative_expression(
        question
    )

    expression = parse_expression(
        expression_text
    )

    x = sp.Symbol("x")

    derivative = sp.diff(
        expression,
        x,
    )

    return {
        "operation": "derivative",
        "expression": expression,
        "variable": x,
        "steps": [
            (
                f"Fonction : f(x) = "
                f"{sp.sstr(expression)}"
            ),
            "On dérive terme par terme.",
            (
                f"f'(x) = "
                f"{sp.sstr(derivative)}"
            ),
        ],
        "result": derivative,
        "explanation": (
            "La dérivée représente le taux de variation "
            "de la fonction par rapport à x."
        ),
    }


# ============================================================
# INTÉGRALES
# ============================================================


def calculate_integral(
    question: str,
) -> dict[str, Any]:
    """
    Calcule une primitive ou une intégrale définie.
    """

    expression_text = extract_integral_expression(
        question
    )

    expression = parse_expression(
        expression_text
    )

    x = sp.Symbol("x")

    lower_text, upper_text = (
        extract_integral_bounds(
            question
        )
    )

    # --------------------------------------------------------
    # INTÉGRALE DÉFINIE
    # --------------------------------------------------------

    if (
        lower_text is not None
        and upper_text is not None
    ):

        lower = parse_expression(
            lower_text
        )

        upper = parse_expression(
            upper_text
        )

        result = sp.integrate(
            expression,
            (x, lower, upper),
        )

        return {
            "operation": "integral",
            "expression": expression,
            "variable": x,
            "lower_bound": lower,
            "upper_bound": upper,
            "steps": [
                (
                    f"Intégrale : "
                    f"∫ {sp.sstr(expression)} dx"
                ),
                (
                    f"Bornes : "
                    f"{sp.sstr(lower)} à "
                    f"{sp.sstr(upper)}"
                ),
                (
                    f"Résultat : "
                    f"{sp.sstr(result)}"
                ),
            ],
            "result": result,
            "explanation": (
                "Il s'agit d'une intégrale définie "
                "calculée entre les bornes indiquées."
            ),
        }

    # --------------------------------------------------------
    # INTÉGRALE INDÉFINIE
    # --------------------------------------------------------

    result = sp.integrate(
        expression,
        x,
    )

    return {
        "operation": "integral",
        "expression": expression,
        "variable": x,
        "steps": [
            (
                f"Fonction à intégrer : "
                f"{sp.sstr(expression)}"
            ),
            "On cherche une primitive par rapport à x.",
            (
                f"Primitive : "
                f"{sp.sstr(result)} + C"
            ),
        ],
        "result": result,
        "explanation": (
            "Une primitive F vérifie F'(x) = f(x). "
            "La constante C représente toutes les primitives."
        ),
    }


# ============================================================
# LIMITES
# ============================================================


def calculate_limit(
    question: str,
) -> dict[str, Any]:
    """
    Calcule la limite d'une expression lorsque x tend
    vers un point donné.
    """

    expression_text, point_text = (
        extract_limit_request(
            question
        )
    )

    # --------------------------------------------------------
    # Point de la limite
    # --------------------------------------------------------

    point_lower = point_text.lower()

    if point_lower in {
        "inf",
        "+inf",
        "infinity",
        "+infinity",
        "∞",
        "+∞",
    }:

        point = sp.oo

    elif point_lower in {
        "-inf",
        "-infinity",
        "-∞",
    }:

        point = -sp.oo

    else:

        point = parse_expression(
            point_text
        )

    # --------------------------------------------------------
    # Expression
    # --------------------------------------------------------

    expression = parse_expression(
        expression_text
    )

    x = sp.Symbol("x")

    result = sp.limit(
        expression,
        x,
        point,
    )

    return {
        "operation": "limit",
        "expression": expression,
        "variable": x,
        "point": point,
        "steps": [
            (
                f"Expression : "
                f"{sp.sstr(expression)}"
            ),
            (
                f"On cherche la limite lorsque "
                f"x → {sp.sstr(point)}."
            ),
            (
                f"Limite = "
                f"{sp.sstr(result)}"
            ),
        ],
        "result": result,
        "explanation": (
            "La limite décrit le comportement de la fonction "
            "lorsque x se rapproche du point indiqué."
        ),
    }


# ============================================================
# SIMPLIFICATION
# ============================================================


def simplify_expression(
    question: str,
) -> dict[str, Any]:
    """
    Simplifie une expression mathématique.
    """

    expression_text = extract_expression(
        question
    )

    expression = parse_expression(
        expression_text
    )

    result = sp.simplify(
        expression
    )

    return {
        "operation": "simplify",
        "expression": expression,
        "steps": [
            (
                f"Expression initiale : "
                f"{sp.sstr(expression)}"
            ),
            (
                "On applique les règles de "
                "simplification algébrique."
            ),
            (
                f"Expression simplifiée : "
                f"{sp.sstr(result)}"
            ),
        ],
        "result": result,
        "explanation": (
            "L'expression a été transformée en une forme "
            "mathématiquement équivalente et plus simple."
        ),
    }


# ============================================================
# CALCUL
# ============================================================


def calculate_expression(
    question: str,
) -> dict[str, Any]:
    """
    Évalue une expression mathématique.
    """

    expression_text = extract_expression(
        question
    )

    expression = parse_expression(
        expression_text
    )

    result = sp.simplify(
        expression
    )

    return {
        "operation": "calculate",
        "expression": expression,
        "steps": [
            (
                f"Expression : "
                f"{sp.sstr(expression)}"
            ),
            (
                f"Résultat : "
                f"{sp.sstr(result)}"
            ),
        ],
        "result": result,
        "explanation": (
            "L'expression a été évaluée symboliquement "
            "avec SymPy."
        ),
    }


# ============================================================
# MOTEUR PRINCIPAL
# ============================================================


def math_assistant(
    question: str,
) -> dict[str, Any]:
    """
    Analyse une question mathématique et retourne
    une réponse structurée.

    Exemple :

        result = math_assistant(
            "Résous 2*x + 5 = 15"
        )
    """

    question = _validate_question(
        question
    )

    operation = detect_operation(
        question
    )

    if operation == "equation":

        result = solve_equation(
            question
        )

    elif operation == "derivative":

        result = calculate_derivative(
            question
        )

    elif operation == "integral":

        result = calculate_integral(
            question
        )

    elif operation == "limit":

        result = calculate_limit(
            question
        )

    elif operation == "simplify":

        result = simplify_expression(
            question
        )

    elif operation == "calculate":

        result = calculate_expression(
            question
        )

    else:

        raise ValueError(
            f"Opération non supportée : {operation}"
        )

    result["question"] = question

    return result


# ============================================================
# FORMATAGE TEXTE
# ============================================================


def format_result(
    result: dict[str, Any],
) -> str:
    """
    Transforme le résultat structuré en texte lisible.
    """

    operation = result.get(
        "operation",
        "unknown",
    )

    labels = {
        "equation": "Équation",
        "derivative": "Dérivée",
        "integral": "Intégrale",
        "limit": "Limite",
        "simplify": "Simplification",
        "calculate": "Calcul",
    }

    label = labels.get(
        operation,
        operation,
    )

    lines = [
        f"Type : {label}",
        "",
        "Démarche :",
    ]

    for step in result.get(
        "steps",
        [],
    ):

        lines.append(
            f"- {step}"
        )

    lines.extend(
        [
            "",
            f"Résultat : "
            f"{sp.sstr(result['result'])}",
            "",
            "Explication :",
            result.get(
                "explanation",
                "",
            ),
        ]
    )

    return "\n".join(
        lines
    )