
"""
Algorithmes de graphes pour MathLab AI.

Ce module contient plusieurs algorithmes classiques de théorie
des graphes :

- BFS (Breadth-First Search)
- DFS (Depth-First Search)
- Dijkstra
- Bellman-Ford
- Floyd-Warshall
- Kruskal
- Prim

Représentation des graphes
--------------------------

Graphes non pondérés :

    graph = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A", "D"],
        "D": ["B", "C"],
    }

Graphes pondérés :

    graph = {
        "A": {"B": 4, "C": 2},
        "B": {"A": 4, "D": 5},
        "C": {"A": 2, "D": 1},
        "D": {"B": 5, "C": 1},
    }

Les fonctions ne modifient pas le graphe fourni.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from numbers import Real


# ============================================================
# VALIDATION
# ============================================================


def _validate_graph(graph) -> None:
    """
    Vérifie qu'un graphe est représenté par un dictionnaire.

    Raises
    ------
    TypeError
        Si graph n'est pas un dictionnaire.
    """
    if not isinstance(graph, dict):
        raise TypeError("Le graphe doit être représenté par un dictionnaire.")


def _validate_unweighted_graph(graph) -> None:
    """
    Vérifie la structure d'un graphe non pondéré.
    """
    _validate_graph(graph)

    for node, neighbors in graph.items():
        if not isinstance(neighbors, (list, tuple, set)):
            raise TypeError(
                "Les voisins doivent être représentés par une liste, "
                "un tuple ou un ensemble."
            )

        for neighbor in neighbors:
            if neighbor not in graph:
                raise ValueError(
                    f"Le sommet voisin {neighbor!r} n'existe pas dans le graphe."
                )


def _validate_weighted_graph(graph) -> None:
    """
    Vérifie la structure d'un graphe pondéré.
    """
    _validate_graph(graph)

    for node, neighbors in graph.items():
        if not isinstance(neighbors, dict):
            raise TypeError(
                "Un graphe pondéré doit utiliser un dictionnaire "
                "de voisins et de poids."
            )

        for neighbor, weight in neighbors.items():
            if neighbor not in graph:
                raise ValueError(
                    f"Le sommet voisin {neighbor!r} n'existe pas dans le graphe."
                )

            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError(
                    "Les poids des arêtes doivent être numériques."
                )

            if not math.isfinite(float(weight)):
                raise ValueError(
                    "Les poids des arêtes doivent être finis."
                )


def _validate_start_node(graph, start) -> None:
    """
    Vérifie qu'un sommet de départ existe dans le graphe.
    """
    if start not in graph:
        raise ValueError(
            f"Le sommet de départ {start!r} n'existe pas dans le graphe."
        )


def _validate_edge_weights_non_negative(graph) -> None:
    """
    Vérifie que tous les poids sont positifs ou nuls.

    Nécessaire pour Dijkstra.
    """
    for neighbors in graph.values():
        for weight in neighbors.values():
            if weight < 0:
                raise ValueError(
                    "Dijkstra ne peut pas être utilisé avec des poids négatifs."
                )


# ============================================================
# 1. BFS
# ============================================================


def bfs(graph: dict, start) -> list:
    """
    Parcours en largeur d'un graphe.

    BFS explore d'abord les sommets les plus proches du sommet
    de départ.

    Complexité temporelle :
        O(V + E)

    Complexité spatiale :
        O(V)

    Parameters
    ----------
    graph : dict
        Graphe non pondéré.
    start
        Sommet de départ.

    Returns
    -------
    list
        Ordre de visite des sommets.
    """
    _validate_unweighted_graph(graph)
    _validate_start_node(graph, start)

    visited = set()
    queue = deque([start])
    visited.add(start)

    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# ============================================================
# 2. DFS
# ============================================================


def dfs(graph: dict, start) -> list:
    """
    Parcours en profondeur d'un graphe.

    DFS explore un chemin aussi profondément que possible avant
    de revenir en arrière.

    Cette implémentation utilise une pile explicite afin d'éviter
    les limites de récursion de Python.

    Complexité temporelle :
        O(V + E)

    Complexité spatiale :
        O(V)

    Returns
    -------
    list
        Ordre de visite des sommets.
    """
    _validate_unweighted_graph(graph)
    _validate_start_node(graph, start)

    visited = set()
    stack = [start]

    order = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        # Insertion inverse pour conserver l'ordre naturel
        # des voisins lors de l'exploration.
        for neighbor in reversed(list(graph[node])):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


# ============================================================
# 3. DIJKSTRA
# ============================================================


def dijkstra(graph: dict, start) -> dict:
    """
    Calcule les plus courtes distances depuis un sommet donné.

    Dijkstra fonctionne avec des poids d'arêtes non négatifs.

    Complexité temporelle :
        O((V + E) log V)

    Complexité spatiale :
        O(V)

    Parameters
    ----------
    graph : dict
        Graphe pondéré.
    start
        Sommet de départ.

    Returns
    -------
    dict
        Distance minimale depuis start vers chaque sommet.

    Notes
    -----
    Les sommets non accessibles conservent la valeur math.inf.
    """
    _validate_weighted_graph(graph)
    _validate_start_node(graph, start)
    _validate_edge_weights_non_negative(graph)

    distances = {
        node: math.inf
        for node in graph
    }

    distances[start] = 0.0

    priority_queue = [(0.0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + float(weight)

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(
                    priority_queue,
                    (distance, neighbor),
                )

    return distances


# ============================================================
# 4. BELLMAN-FORD
# ============================================================


def bellman_ford(graph: dict, start) -> dict:
    """
    Calcule les plus courtes distances avec Bellman-Ford.

    Contrairement à Dijkstra, Bellman-Ford peut gérer les poids
    négatifs, mais pas les cycles de poids négatif accessibles
    depuis le sommet de départ.

    Complexité temporelle :
        O(VE)

    Complexité spatiale :
        O(V)

    Raises
    ------
    ValueError
        Si un cycle négatif accessible depuis start est détecté.
    """
    _validate_weighted_graph(graph)
    _validate_start_node(graph, start)

    distances = {
        node: math.inf
        for node in graph
    }

    distances[start] = 0.0

    edges = []

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            edges.append(
                (node, neighbor, float(weight))
            )

    # Relaxation V - 1 fois.
    for _ in range(max(0, len(graph) - 1)):
        updated = False

        for source, target, weight in edges:
            if distances[source] == math.inf:
                continue

            new_distance = distances[source] + weight

            if new_distance < distances[target]:
                distances[target] = new_distance
                updated = True

        if not updated:
            break

    # Détection d'un cycle négatif accessible.
    for source, target, weight in edges:
        if distances[source] == math.inf:
            continue

        if distances[source] + weight < distances[target]:
            raise ValueError(
                "Le graphe contient un cycle de poids négatif "
                "accessible depuis le sommet de départ."
            )

    return distances


# ============================================================
# 5. FLOYD-WARSHALL
# ============================================================


def floyd_warshall(graph: dict) -> dict:
    """
    Calcule les plus courtes distances entre toutes les paires
    de sommets avec l'algorithme de Floyd-Warshall.

    Le graphe peut contenir des poids négatifs, mais ne doit pas
    contenir de cycle négatif.

    Complexité temporelle :
        O(V³)

    Complexité spatiale :
        O(V²)

    Parameters
    ----------
    graph : dict
        Graphe pondéré.

    Returns
    -------
    dict
        Dictionnaire imbriqué contenant les distances minimales.
    """
    _validate_weighted_graph(graph)

    nodes = list(graph.keys())

    distances = {
        source: {
            target: math.inf
            for target in nodes
        }
        for source in nodes
    }

    for node in nodes:
        distances[node][node] = 0.0

    for source, neighbors in graph.items():
        for target, weight in neighbors.items():
            distances[source][target] = min(
                distances[source][target],
                float(weight),
            )

    for intermediate in nodes:
        for source in nodes:
            if distances[source][intermediate] == math.inf:
                continue

            for target in nodes:
                new_distance = (
                    distances[source][intermediate]
                    + distances[intermediate][target]
                )

                if new_distance < distances[source][target]:
                    distances[source][target] = new_distance

    for node in nodes:
        if distances[node][node] < 0:
            raise ValueError(
                "Le graphe contient un cycle de poids négatif."
            )

    return distances


# ============================================================
# UTILITAIRE UNION-FIND
# ============================================================


class _UnionFind:
    """
    Structure Union-Find utilisée par Kruskal.
    """

    def __init__(self, elements):
        self.parent = {
            element: element
            for element in elements
        }

        self.rank = {
            element: 0
            for element in elements
        }

    def find(self, element):
        """Trouve le représentant d'un ensemble."""
        if self.parent[element] != element:
            self.parent[element] = self.find(
                self.parent[element]
            )

        return self.parent[element]

    def union(self, first, second) -> bool:
        """
        Fusionne deux ensembles.

        Returns
        -------
        bool
            True si une fusion a été effectuée.
        """
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return False

        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first

        self.parent[root_second] = root_first

        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1

        return True


# ============================================================
# 6. KRUSKAL
# ============================================================


def kruskal(graph: dict) -> tuple[list[tuple], Real]:
    """
    Calcule un arbre couvrant de poids minimal avec Kruskal.

    Le graphe doit être non orienté et pondéré.

    Complexité temporelle :
        O(E log E)

    Complexité spatiale :
        O(V + E)

    Returns
    -------
    tuple
        (edges, total_weight)

        edges :
            Liste des arêtes sélectionnées sous la forme
            (source, target, weight).

        total_weight :
            Poids total de l'arbre couvrant minimal.

    Raises
    ------
    ValueError
        Si le graphe n'est pas connexe.
    """
    _validate_weighted_graph(graph)

    edges = []

    for source, neighbors in graph.items():
        for target, weight in neighbors.items():
            # Pour un graphe non orienté, on évite de compter
            # deux fois la même arête.
            if source == target:
                continue

            if (target, source, float(weight)) in edges:
                continue

            edges.append(
                (source, target, float(weight))
            )

    edges.sort(key=lambda edge: edge[2])

    union_find = _UnionFind(graph.keys())

    selected_edges = []
    total_weight = 0.0

    for source, target, weight in edges:
        if union_find.union(source, target):
            selected_edges.append(
                (source, target, weight)
            )
            total_weight += weight

            if len(selected_edges) == len(graph) - 1:
                break

    if len(selected_edges) != max(0, len(graph) - 1):
        raise ValueError(
            "Le graphe doit être connexe pour construire "
            "un arbre couvrant minimal."
        )

    return selected_edges, total_weight


# ============================================================
# 7. PRIM
# ============================================================


def prim(graph: dict, start=None) -> tuple[list[tuple], Real]:
    """
    Calcule un arbre couvrant de poids minimal avec Prim.

    Le graphe doit être non orienté et pondéré.

    Complexité temporelle :
        O(E log V)

    Complexité spatiale :
        O(V + E)

    Parameters
    ----------
    graph : dict
        Graphe pondéré.
    start
        Sommet de départ. Si None, le premier sommet du graphe
        est utilisé.

    Returns
    -------
    tuple
        (edges, total_weight)

    Raises
    ------
    ValueError
        Si le graphe est vide ou non connexe.
    """
    _validate_weighted_graph(graph)

    if not graph:
        raise ValueError(
            "Le graphe ne peut pas être vide."
        )

    if start is None:
        start = next(iter(graph))

    _validate_start_node(graph, start)

    visited = {start}
    priority_queue = []

    for neighbor, weight in graph[start].items():
        heapq.heappush(
            priority_queue,
            (float(weight), start, neighbor),
        )

    selected_edges = []
    total_weight = 0.0

    while priority_queue and len(visited) < len(graph):
        weight, source, target = heapq.heappop(priority_queue)

        if target in visited:
            continue

        visited.add(target)

        selected_edges.append(
            (source, target, weight)
        )

        total_weight += weight

        for neighbor, edge_weight in graph[target].items():
            if neighbor not in visited:
                heapq.heappush(
                    priority_queue,
                    (
                        float(edge_weight),
                        target,
                        neighbor,
                    ),
                )

    if len(visited) != len(graph):
        raise ValueError(
            "Le graphe doit être connexe pour construire "
            "un arbre couvrant minimal."
        )

    return selected_edges, total_weight

