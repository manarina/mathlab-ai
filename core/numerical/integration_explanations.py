
"""
Explications pédagogiques des méthodes d'intégration numérique.

Ce module contient uniquement les explications destinées à
l'interface utilisateur de MathLab AI.

Les calculs numériques sont réalisés dans integration.py.
"""

from __future__ import annotations


def explain_rectangle_rule() -> str:
    """
    Explique la méthode des rectangles au point milieu.
    """
    return r"""
### 📐 Méthode des rectangles

La méthode des rectangles consiste à remplacer l'aire située
sous la courbe par une somme d'aires de rectangles.

Pour chaque sous-intervalle, on utilise la valeur de la fonction
au **milieu** de l'intervalle.

Si :

\[
h = \frac{b-a}{n}
\]

alors l'approximation est :

\[
\int_a^b f(x)\,dx
\approx
h\sum_{i=0}^{n-1} f(x_i^*)
\]

où \(x_i^*\) est le milieu du \(i\)-ème sous-intervalle.

**Idée importante :**

Plus \(n\) augmente, plus les rectangles deviennent fins et
l'approximation tend généralement vers la valeur exacte de
l'intégrale.
"""


def explain_trapezoidal_rule() -> str:
    """
    Explique la méthode des trapèzes.
    """
    return r"""
### 🔺 Méthode des trapèzes

La méthode des trapèzes approxime la courbe par des segments
de droite.

Au lieu de considérer des rectangles, chaque sous-intervalle
forme un trapèze.

On définit :

\[
h = \frac{b-a}{n}
\]

et l'approximation est :

\[
\int_a^b f(x)\,dx
\approx
\frac{h}{2}
\left[
f(a)+f(b)
+
2\sum_{i=1}^{n-1}f(x_i)
\right]
\]

**Idée importante :**

La méthode des trapèzes utilise davantage d'information sur
la forme de la courbe que la méthode des rectangles.
"""


def explain_simpson_one_third() -> str:
    """
    Explique la méthode de Simpson 1/3.
    """
    return r"""
### 🧮 Méthode de Simpson 1/3

La méthode de Simpson 1/3 approxime la fonction par des
polynômes de degré 2 sur de petits intervalles.

On utilise :

\[
h = \frac{b-a}{n}
\]

avec **\(n\) obligatoirement pair**.

La formule est :

\[
\int_a^b f(x)\,dx
\approx
\frac{h}{3}
\left[
f(x_0)+f(x_n)
+
4\sum_{\text{indices impairs}}f(x_i)
+
2\sum_{\text{indices pairs}}f(x_i)
\right]
\]

Les coefficients suivent donc le schéma :

\[
1,\;4,\;2,\;4,\;2,\ldots,\;4,\;1
\]

**Idée importante :**

Simpson utilise une approximation quadratique de la fonction,
ce qui permet souvent d'obtenir une meilleure précision que
les rectangles et les trapèzes pour un même nombre de
subdivisions.
"""


def explain_integration_error(
    exact_value: float,
    approximate_value: float,
) -> str:
    """
    Explique l'erreur absolue entre une valeur exacte
    et une approximation numérique.
    """
    error = abs(exact_value - approximate_value)

    return f"""
### 📊 Erreur d'approximation

La valeur exacte de l'intégrale est :

\[
I = {exact_value:.10g}
\]

La valeur obtenue numériquement est :

\[
I_n = {approximate_value:.10g}
\]

L'erreur absolue est :

\[
E = |I-I_n|
\]

Dans notre cas :

\[
E = {error:.10g}
\]

Plus cette valeur est proche de zéro, plus l'approximation
est proche de la valeur exacte.
"""


def explain_method_comparison() -> str:
    """
    Explique la comparaison des méthodes d'intégration.
    """
    return r"""
### ⚖️ Comparaison des méthodes

Les trois méthodes utilisent des approximations différentes :

| Méthode | Approximation |
|---|---|
| Rectangles | rectangles |
| Trapèzes | segments / trapèzes |
| Simpson 1/3 | polynômes de degré 2 |

Pour comparer les méthodes, on peut observer :

- la valeur approchée ;
- l'erreur absolue ;
- l'évolution de l'erreur lorsque \(n\) augmente.

Une méthode donnant une erreur plus faible pour un même
nombre de subdivisions fournit une approximation plus proche
de la valeur exacte pour cet exemple.
"""


def explain_number_of_subdivisions() -> str:
    """
    Explique l'influence du nombre de subdivisions.
    """
    return r"""
### 🔢 Influence du nombre de subdivisions

Le nombre \(n\) représente le nombre de sous-intervalles
utilisés pour approximer l'intégrale.

Lorsque \(n\) augmente :

\[
h = \frac{b-a}{n}
\]

diminue.

Les sous-intervalles deviennent donc plus petits.

En général, cela améliore l'approximation et réduit l'erreur,
mais augmente également le nombre de calculs nécessaires.

Il faut donc trouver un compromis entre :

**précision** ↔ **coût de calcul**
"""


def explain_exact_vs_numerical() -> str:
    """
    Explique la différence entre intégration exacte
    et intégration numérique.
    """
    return r"""
### 🎯 Intégration exacte vs intégration numérique

**Intégration exacte**

On cherche une expression mathématique exacte de :

\[
\int_a^b f(x)\,dx
\]

Par exemple :

\[
\int_0^1 x^2\,dx
=
\frac{1}{3}
\]

**Intégration numérique**

On calcule une approximation à partir d'un nombre fini
de points.

Par exemple :

\[
\int_a^b f(x)\,dx
\approx I_n
\]

L'intégration numérique est particulièrement utile lorsque
la primitive de \(f\) est difficile ou impossible à obtenir
facilement sous forme analytique.
"""


def explain_simpson_requirement() -> str:
    """
    Explique pourquoi Simpson 1/3 nécessite un nombre pair
    de subdivisions.
    """
    return r"""
### ⚠️ Pourquoi \(n\) doit-il être pair ?

La méthode de Simpson 1/3 travaille sur des groupes de
**deux sous-intervalles**.

Chaque groupe utilise trois points :

\[
x_i,\quad x_{i+1},\quad x_{i+2}
\]

Il faut donc pouvoir regrouper les subdivisions deux par deux.

C'est pourquoi :

\[
\boxed{n\text{ doit être pair}}
\]

Par exemple :

- \(n=2\) ✅
- \(n=4\) ✅
- \(n=6\) ✅
- \(n=3\) ❌
- \(n=5\) ❌
"""

