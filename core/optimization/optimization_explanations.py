
"""
Explications pédagogiques des algorithmes d'optimisation à une variable.

Ce module contient uniquement le contenu pédagogique associé
aux méthodes numériques du module d'optimisation.

Les algorithmes eux-mêmes sont implémentés dans :
    core.optimization.one_dimensional
"""


# ============================================================
# INTRODUCTION GÉNÉRALE
# ============================================================
def explain_optimization() -> str:
    """
    Explique le principe général de l'optimisation numérique.

    Returns
    -------
    str
        Explication pédagogique de l'optimisation.
    """
    return (
        "L'optimisation consiste à rechercher la meilleure solution possible "
        "pour une fonction objectif. Selon le problème, on peut rechercher "
        "un minimum ou un maximum de cette fonction sur un domaine donné. "
        "En optimisation numérique, les méthodes numériques permettent "
        "d'obtenir une approximation de la solution lorsque celle-ci ne peut "
        "pas être déterminée facilement de manière analytique."
    )


# ============================================================
# GRID SEARCH
# ============================================================


def explain_grid_search() -> str:
    """
    Explique la méthode de recherche par balayage.

    Returns
    -------
    str
        Explication pédagogique du Grid Search.
    """
    return (
        "La recherche par balayage consiste à diviser l'intervalle "
        "d'étude en plusieurs points régulièrement espacés. "
        "La fonction est évaluée sur chacun de ces points, puis le "
        "point donnant la plus petite valeur est sélectionné pour "
        "une recherche de minimum. "
        "Cette méthode est simple et robuste, mais sa précision dépend "
        "du nombre de points utilisés."
    )


def explain_grid_search_formula() -> str:
    """
    Retourne la formule conceptuelle du Grid Search.

    Returns
    -------
    str
        Formule mathématique sous forme textuelle.
    """
    return (
        "On construit des points x₀, x₁, ..., xₙ dans l'intervalle "
        "[a, b], puis on recherche : "
        "x_min = argmin f(xᵢ)."
    )


def explain_grid_search_advantages() -> str:
    """
    Présente les principaux avantages du Grid Search.

    Returns
    -------
    str
        Liste textuelle des avantages.
    """
    return (
        "Les principaux avantages du balayage sont sa simplicité, "
        "sa facilité d'implémentation et son absence de dérivées. "
        "Il permet également d'explorer visuellement le comportement "
        "d'une fonction sur un intervalle."
    )


def explain_grid_search_limitations() -> str:
    """
    Présente les principales limites du Grid Search.

    Returns
    -------
    str
        Liste textuelle des limites.
    """
    return (
        "La principale limite du balayage est que sa précision dépend "
        "du nombre de points. Une grille trop grossière peut manquer "
        "la position exacte du minimum. Une grille très fine augmente "
        "en revanche le nombre d'évaluations de la fonction."
    )


# ============================================================
# SECTION DORÉE
# ============================================================


def explain_golden_section() -> str:
    """
    Explique la méthode de la section dorée.

    Returns
    -------
    str
        Explication pédagogique de la section dorée.
    """
    return (
        "La méthode de la section dorée recherche le minimum d'une "
        "fonction unimodale sur un intervalle. "
        "À chaque étape, l'intervalle est réduit en utilisant deux "
        "points internes construits à partir du nombre d'or. "
        "Une partie de l'intervalle contenant le minimum est conservée, "
        "tandis que l'autre partie est éliminée."
    )


def explain_golden_section_formula() -> str:
    """
    Retourne les relations mathématiques principales de la section dorée.

    Returns
    -------
    str
        Formule mathématique sous forme textuelle.
    """
    return (
        "Le nombre d'or est φ = (1 + √5) / 2. "
        "Pour un intervalle [a, b], les points internes sont calculés "
        "à partir de φ afin de réduire progressivement la longueur "
        "de l'intervalle."
    )


def explain_golden_section_advantages() -> str:
    """
    Présente les avantages de la section dorée.

    Returns
    -------
    str
        Liste textuelle des avantages.
    """
    return (
        "La section dorée ne nécessite pas de dérivée et utilise "
        "efficacement les évaluations précédentes de la fonction. "
        "Elle offre une convergence régulière pour les fonctions "
        "unimodales."
    )


def explain_golden_section_limitations() -> str:
    """
    Présente les limites de la section dorée.

    Returns
    -------
    str
        Liste textuelle des limites.
    """
    return (
        "La section dorée suppose que la fonction possède un comportement "
        "unimodal sur l'intervalle étudié. "
        "Elle peut donc ne pas identifier le minimum global d'une fonction "
        "présentant plusieurs minima locaux sur cet intervalle."
    )


# ============================================================
# NEWTON
# ============================================================


def explain_newton_minimum() -> str:
    """
    Explique la méthode de Newton pour l'optimisation.

    Returns
    -------
    str
        Explication pédagogique de la méthode de Newton.
    """
    return (
        "La méthode de Newton recherche un point stationnaire d'une "
        "fonction en utilisant sa première et sa deuxième dérivée. "
        "Elle applique successivement une approximation locale de la "
        "fonction afin de rapprocher le point courant d'un minimum "
        "ou d'un autre point stationnaire."
    )


def explain_newton_formula() -> str:
    """
    Retourne la formule de l'itération de Newton.

    Returns
    -------
    str
        Formule mathématique de Newton.
    """
    return (
        "L'itération de Newton est : "
        "xₙ₊₁ = xₙ − f'(xₙ) / f''(xₙ)."
    )


def explain_newton_convergence() -> str:
    """
    Explique le principe de convergence de Newton.

    Returns
    -------
    str
        Explication de la convergence.
    """
    return (
        "La méthode de Newton est une méthode locale. "
        "Son comportement dépend notamment du point initial et de "
        "la forme de la fonction. "
        "Lorsque les conditions sont favorables, elle peut converger "
        "très rapidement vers un point stationnaire."
    )


def explain_newton_minimum_condition() -> str:
    """
    Explique la condition permettant d'identifier un minimum local.

    Returns
    -------
    str
        Explication de la condition sur la dérivée seconde.
    """
    return (
        "Après avoir obtenu un point stationnaire x*, la dérivée seconde "
        "permet de déterminer la nature locale du point. "
        "Si f''(x*) > 0, le point est un minimum local. "
        "Si f''(x*) < 0, le point est un maximum local."
    )


def explain_newton_advantages() -> str:
    """
    Présente les avantages de Newton.

    Returns
    -------
    str
        Liste textuelle des avantages.
    """
    return (
        "La méthode de Newton peut converger très rapidement lorsqu'elle "
        "est utilisée dans de bonnes conditions. "
        "Elle permet également d'obtenir une grande précision avec "
        "relativement peu d'itérations."
    )


def explain_newton_limitations() -> str:
    """
    Présente les limites de Newton.

    Returns
    -------
    str
        Liste textuelle des limites.
    """
    return (
        "La méthode de Newton nécessite les première et deuxième dérivées. "
        "Elle dépend également fortement du point initial et peut ne pas "
        "converger vers le minimum recherché. "
        "La méthode rencontre aussi des difficultés lorsque la dérivée "
        "seconde est nulle ou très proche de zéro."
    )


# ============================================================
# COMPARAISON DES MÉTHODES
# ============================================================


def explain_method_comparison() -> str:
    """
    Explique la différence générale entre les trois méthodes.

    Returns
    -------
    str
        Comparaison pédagogique.
    """
    return (
        "Le balayage est la méthode la plus simple et ne nécessite "
        "aucune dérivée, mais sa précision dépend fortement de la grille. "
        "La section dorée améliore la recherche en réduisant "
        "progressivement l'intervalle et ne nécessite pas non plus "
        "de dérivées. "
        "Newton utilise les dérivées et peut être beaucoup plus rapide, "
        "mais il est plus sensible au choix du point initial et aux "
        "propriétés de la fonction."
    )


# ============================================================
# CONSEILS GÉNÉRAUX
# ============================================================


def explain_optimization_workflow() -> str:
    """
    Explique une démarche générale pour résoudre un problème
    d'optimisation numérique à une variable.

    Returns
    -------
    str
        Démarche en plusieurs étapes.
    """
    return (
        "Une démarche classique consiste à : "
        "1) définir la fonction objectif ; "
        "2) préciser l'intervalle ou le point initial ; "
        "3) choisir une méthode numérique adaptée ; "
        "4) exécuter l'algorithme avec une tolérance appropriée ; "
        "5) vérifier la solution obtenue ; "
        "6) comparer éventuellement plusieurs méthodes."
    )

