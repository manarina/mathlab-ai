
"""
Tests des algorithmes de graphes de MathLab AI.

Algorithmes testés :
- BFS
- DFS
- Dijkstra
- Bellman-Ford
- Floyd-Warshall
- Kruskal
- Prim
"""

import math

import pytest

from core.algorithms.graph_algorithms import (
    bellman_ford,
    bfs,
    dfs,
    dijkstra,
    floyd_warshall,
    kruskal,
    prim,
)


# ============================================================
# Graphes de test
# ============================================================

UNWEIGHTED_GRAPH = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}


WEIGHTED_GRAPH = {
    "A": {"B": 4, "C": 2},
    "B": {"A": 4, "C": 1, "D": 5},
    "C": {"A": 2, "B": 1, "D": 8, "E": 10},
    "D": {"B": 5, "C": 8, "E": 2},
    "E": {"C": 10, "D": 2},
}


UNDIRECTED_MST_GRAPH = {
    "A": {"B": 4, "C": 2},
    "B": {"A": 4, "C": 1, "D": 5},
    "C": {"A": 2, "B": 1, "D": 8, "E": 10},
    "D": {"B": 5, "C": 8, "E": 2},
    "E": {"C": 10, "D": 2},
}


DISCONNECTED_WEIGHTED_GRAPH = {
    "A": {"B": 1},
    "B": {"A": 1},
    "C": {"D": 2},
    "D": {"C": 2},
}


# ============================================================
# BFS
# ============================================================


class TestBFS:
    """Tests de Breadth-First Search."""

    def test_bfs_returns_expected_order(self):
        result = bfs(UNWEIGHTED_GRAPH, "A")

        assert result == ["A", "B", "C", "D", "E", "F"]

    def test_bfs_from_another_start(self):
        result = bfs(UNWEIGHTED_GRAPH, "C")

        assert result[0] == "C"
        assert set(result) == set(UNWEIGHTED_GRAPH)

    def test_bfs_single_node(self):
        graph = {"A": []}

        assert bfs(graph, "A") == ["A"]

    def test_bfs_empty_graph_with_start_fails(self):
        with pytest.raises(ValueError):
            bfs({}, "A")

    def test_bfs_missing_start_fails(self):
        with pytest.raises(ValueError):
            bfs(UNWEIGHTED_GRAPH, "Z")

    def test_bfs_invalid_graph_type_fails(self):
        with pytest.raises(TypeError):
            bfs([], "A")

    def test_bfs_neighbor_missing_from_graph_fails(self):
        graph = {
            "A": ["B"],
        }

        with pytest.raises(ValueError):
            bfs(graph, "A")

    def test_bfs_visits_each_vertex_once(self):
        graph = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }

        result = bfs(graph, "A")

        assert len(result) == 3
        assert len(set(result)) == 3

    def test_bfs_handles_cycle(self):
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],
        }

        result = bfs(graph, "A")

        assert result == ["A", "B", "C"]


# ============================================================
# DFS
# ============================================================


class TestDFS:
    """Tests de Depth-First Search."""

    def test_dfs_returns_expected_order(self):
        result = dfs(UNWEIGHTED_GRAPH, "A")

        assert result == ["A", "B", "D", "E", "F", "C"]

    def test_dfs_starts_with_requested_vertex(self):
        result = dfs(UNWEIGHTED_GRAPH, "C")

        assert result[0] == "C"

    def test_dfs_visits_all_vertices(self):
        result = dfs(UNWEIGHTED_GRAPH, "A")

        assert set(result) == set(UNWEIGHTED_GRAPH)
        assert len(result) == len(UNWEIGHTED_GRAPH)

    def test_dfs_single_node(self):
        graph = {"A": []}

        assert dfs(graph, "A") == ["A"]

    def test_dfs_empty_graph_with_start_fails(self):
        with pytest.raises(ValueError):
            dfs({}, "A")

    def test_dfs_missing_start_fails(self):
        with pytest.raises(ValueError):
            dfs(UNWEIGHTED_GRAPH, "Z")

    def test_dfs_invalid_graph_type_fails(self):
        with pytest.raises(TypeError):
            dfs([], "A")

    def test_dfs_neighbor_missing_from_graph_fails(self):
        graph = {
            "A": ["B"],
        }

        with pytest.raises(ValueError):
            dfs(graph, "A")

    def test_dfs_handles_cycle(self):
        graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],
        }

        result = dfs(graph, "A")

        assert result == ["A", "B", "C"]

    def test_dfs_does_not_visit_vertex_twice(self):
        graph = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }

        result = dfs(graph, "A")

        assert len(result) == len(set(result))
        assert set(result) == {"A", "B", "C", "D"}


# ============================================================
# Dijkstra
# ============================================================


class TestDijkstra:
    """Tests de l'algorithme de Dijkstra."""

    def test_dijkstra_returns_expected_distances(self):
        result = dijkstra(WEIGHTED_GRAPH, "A")

        assert result["A"] == 0
        assert result["B"] == 3
        assert result["C"] == 2
        assert result["D"] == 8
        assert result["E"] == 10

    def test_dijkstra_start_distance_is_zero(self):
        result = dijkstra(WEIGHTED_GRAPH, "A")

        assert result["A"] == 0

    def test_dijkstra_finds_shortest_path_through_intermediate_vertex(self):
        result = dijkstra(WEIGHTED_GRAPH, "A")

        # A -> C -> B = 2 + 1 = 3
        assert result["B"] == 3

    def test_dijkstra_handles_zero_weight_edges(self):
        graph = {
            "A": {"B": 0},
            "B": {"C": 2},
            "C": {},
        }

        result = dijkstra(graph, "A")

        assert result["A"] == 0
        assert result["B"] == 0
        assert result["C"] == 2

    def test_dijkstra_handles_unreachable_vertex(self):
        graph = {
            "A": {"B": 1},
            "B": {},
            "C": {},
        }

        result = dijkstra(graph, "A")

        assert result["A"] == 0
        assert result["B"] == 1
        assert math.isinf(result["C"])

    def test_dijkstra_rejects_negative_weight(self):
        graph = {
            "A": {"B": -1},
            "B": {},
        }

        with pytest.raises(ValueError):
            dijkstra(graph, "A")

    def test_dijkstra_rejects_negative_weight_anywhere(self):
        graph = {
            "A": {"B": 1},
            "B": {"C": -5},
            "C": {},
        }

        with pytest.raises(ValueError):
            dijkstra(graph, "A")

    def test_dijkstra_missing_start_fails(self):
        with pytest.raises(ValueError):
            dijkstra(WEIGHTED_GRAPH, "Z")

    def test_dijkstra_invalid_graph_type_fails(self):
        with pytest.raises(TypeError):
            dijkstra([], "A")

    def test_dijkstra_invalid_neighbor_structure_fails(self):
        graph = {
            "A": ["B"],
            "B": {},
        }

        with pytest.raises(TypeError):
            dijkstra(graph, "A")

    def test_dijkstra_missing_neighbor_fails(self):
        graph = {
            "A": {"B": 1},
        }

        with pytest.raises(ValueError):
            dijkstra(graph, "A")

    def test_dijkstra_rejects_non_numeric_weight(self):
        graph = {
            "A": {"B": "1"},
            "B": {},
        }

        with pytest.raises(TypeError):
            dijkstra(graph, "A")

    def test_dijkstra_rejects_nan_weight(self):
        graph = {
            "A": {"B": math.nan},
            "B": {},
        }

        with pytest.raises(ValueError):
            dijkstra(graph, "A")

    def test_dijkstra_rejects_infinite_weight(self):
        graph = {
            "A": {"B": math.inf},
            "B": {},
        }

        with pytest.raises(ValueError):
            dijkstra(graph, "A")


# ============================================================
# Bellman-Ford
# ============================================================


class TestBellmanFord:
    """Tests de l'algorithme de Bellman-Ford."""

    def test_bellman_ford_returns_expected_distances(self):
        graph = {
            "A": {"B": 4, "C": 5},
            "B": {"C": -2, "D": 4},
            "C": {"D": 3},
            "D": {},
        }

        result = bellman_ford(graph, "A")

        assert result["A"] == 0
        assert result["B"] == 4
        assert result["C"] == 2
        assert result["D"] == 5

    def test_bellman_ford_supports_negative_edges(self):
        graph = {
            "A": {"B": 2},
            "B": {"C": -5},
            "C": {},
        }

        result = bellman_ford(graph, "A")

        assert result["A"] == 0
        assert result["B"] == 2
        assert result["C"] == -3

    def test_bellman_ford_handles_unreachable_vertex(self):
        graph = {
            "A": {"B": 1},
            "B": {},
            "C": {},
        }

        result = bellman_ford(graph, "A")

        assert result["A"] == 0
        assert result["B"] == 1
        assert math.isinf(result["C"])

    def test_bellman_ford_detects_negative_cycle(self): 
        graph = { 
            "A": {"B": 1}, 
            "B": {"C": -2}, 
            "C": {"A": 0}, 
            } 
        with pytest.raises(ValueError): 
            bellman_ford(graph, "A")

    def test_bellman_ford_detects_negative_cycle_reachable_from_start(self): 
        graph = { 
            "A": {"B": 2}, 
            "B": {"C": -3}, 
            "C": {"B": 1}, 
            } 
        with pytest.raises(ValueError): 
            bellman_ford(graph, "A")

    def test_bellman_ford_ignores_unreachable_negative_cycle(self):
        graph = {
            "A": {"B": 1},
            "B": {},
            "C": {"D": -2},
            "D": {"C": 1},
        }

        result = bellman_ford(graph, "A")

        assert result["A"] == 0
        assert result["B"] == 1
        assert math.isinf(result["C"])
        assert math.isinf(result["D"])

    def test_bellman_ford_start_distance_is_zero(self):
        result = bellman_ford(WEIGHTED_GRAPH, "A")

        assert result["A"] == 0

    def test_bellman_ford_missing_start_fails(self):
        with pytest.raises(ValueError):
            bellman_ford(WEIGHTED_GRAPH, "Z")

    def test_bellman_ford_invalid_graph_type_fails(self):
        with pytest.raises(TypeError):
            bellman_ford([], "A")

    def test_bellman_ford_missing_neighbor_fails(self):
        graph = {
            "A": {"B": 1},
        }

        with pytest.raises(ValueError):
            bellman_ford(graph, "A")

    def test_bellman_ford_rejects_non_numeric_weight(self):
        graph = {
            "A": {"B": "invalid"},
            "B": {},
        }

        with pytest.raises(TypeError):
            bellman_ford(graph, "A")

    def test_bellman_ford_rejects_nan_weight(self):
        graph = {
            "A": {"B": math.nan},
            "B": {},
        }

        with pytest.raises(ValueError):
            bellman_ford(graph, "A")

    def test_bellman_ford_rejects_infinite_weight(self):
        graph = {
            "A": {"B": math.inf},
            "B": {},
        }

        with pytest.raises(ValueError):
            bellman_ford(graph, "A")


# ============================================================
# Floyd-Warshall
# ============================================================


class TestFloydWarshall:
    """Tests de l'algorithme de Floyd-Warshall."""

    def test_floyd_warshall_returns_all_pairs_distances(self):
        graph = {
            "A": {"B": 3, "C": 10},
            "B": {"C": 2},
            "C": {},
        }

        result = floyd_warshall(graph)

        assert result["A"]["A"] == 0
        assert result["A"]["B"] == 3
        assert result["A"]["C"] == 5

    def test_floyd_warshall_diagonal_is_zero(self):
        result = floyd_warshall(WEIGHTED_GRAPH)

        for node in WEIGHTED_GRAPH:
            assert result[node][node] == 0

    def test_floyd_warshall_finds_shortest_indirect_path(self):
        graph = {
            "A": {"B": 10, "C": 2},
            "B": {},
            "C": {"B": 3},
        }

        result = floyd_warshall(graph)

        assert result["A"]["B"] == 5

    def test_floyd_warshall_handles_unreachable_vertices(self):
        graph = {
            "A": {"B": 1},
            "B": {},
            "C": {},
        }

        result = floyd_warshall(graph)

        assert result["A"]["B"] == 1
        assert math.isinf(result["A"]["C"])
        assert math.isinf(result["C"]["A"])

    def test_floyd_warshall_detects_negative_cycle(self):
        graph = {
            "A": {"B": 1},
            "B": {"C": -2},
            "C": {"A": 0},
        }

        with pytest.raises(ValueError):
            floyd_warshall(graph)

    def test_floyd_warshall_empty_graph(self):
        assert floyd_warshall({}) == {}

    def test_floyd_warshall_single_vertex(self):
        graph = {"A": {}}

        result = floyd_warshall(graph)

        assert result == {"A": {"A": 0}}

    def test_floyd_warshall_missing_neighbor_fails(self):
        graph = {
            "A": {"B": 1},
        }

        with pytest.raises(ValueError):
            floyd_warshall(graph)

    def test_floyd_warshall_invalid_graph_type_fails(self):
        with pytest.raises(TypeError):
            floyd_warshall([])

    def test_floyd_warshall_invalid_neighbor_structure_fails(self):
        graph = {
            "A": ["B"],
            "B": {},
        }

        with pytest.raises(TypeError):
            floyd_warshall(graph)

    def test_floyd_warshall_rejects_non_numeric_weight(self):
        graph = {
            "A": {"B": "invalid"},
            "B": {},
        }

        with pytest.raises(TypeError):
            floyd_warshall(graph)

    def test_floyd_warshall_rejects_nan_weight(self):
        graph = {
            "A": {"B": math.nan},
            "B": {},
        }

        with pytest.raises(ValueError):
            floyd_warshall(graph)

    def test_floyd_warshall_rejects_infinite_weight(self):
        graph = {
            "A": {"B": math.inf},
            "B": {},
        }

        with pytest.raises(ValueError):
            floyd_warshall(graph)


# ============================================================
# Kruskal
# ============================================================


class TestKruskal:
    """Tests de l'algorithme de Kruskal."""

    def test_kruskal_returns_mst(self):
        edges, total_weight = kruskal(UNDIRECTED_MST_GRAPH)

        assert len(edges) == len(UNDIRECTED_MST_GRAPH) - 1
        assert total_weight == 10

    def test_kruskal_mst_contains_expected_weight(self):
        edges, total_weight = kruskal(UNDIRECTED_MST_GRAPH)

        assert total_weight == 1 + 2 + 2 + 5

    def test_kruskal_returns_edge_tuples(self):
        edges, _ = kruskal(UNDIRECTED_MST_GRAPH)

        for edge in edges:
            assert len(edge) == 3

    def test_kruskal_does_not_create_cycle(self):
        edges, _ = kruskal(UNDIRECTED_MST_GRAPH)

        assert len(edges) == len(UNDIRECTED_MST_GRAPH) - 1

    def test_kruskal_single_vertex(self):
        graph = {"A": {}}

        edges, total_weight = kruskal(graph)

        assert edges == []
        assert total_weight == 0.0

    def test_kruskal_empty_graph(self):
        edges, total_weight = kruskal({})

        assert edges == []
        assert total_weight == 0.0

    def test_kruskal_rejects_disconnected_graph(self):
        with pytest.raises(ValueError):
            kruskal(DISCONNECTED_WEIGHTED_GRAPH)

    def test_kruskal_missing_neighbor_fails(self):
        graph = {
            "A": {"B": 1},
        }

        with pytest.raises(ValueError):
            kruskal(graph)

    def test_kruskal_invalid_graph_type_fails(self):
        with pytest.raises(TypeError):
            kruskal([])

    def test_kruskal_rejects_non_numeric_weight(self):
        graph = {
            "A": {"B": "invalid"},
            "B": {"A": "invalid"},
        }

        with pytest.raises(TypeError):
            kruskal(graph)

    def test_kruskal_supports_negative_weight(self):
        graph = {
             "A": {"B": -1},
             "B": {"A": -1},
        }

        edges, total_weight = kruskal(graph)

        assert len(edges) == 1
        assert total_weight == -1

    def test_kruskal_rejects_nan_weight(self):
        graph = {
            "A": {"B": math.nan},
            "B": {"A": math.nan},
        }

        with pytest.raises(ValueError):
            kruskal(graph)

    def test_kruskal_rejects_infinite_weight(self):
        graph = {
            "A": {"B": math.inf},
            "B": {"A": math.inf},
        }

        with pytest.raises(ValueError):
            kruskal(graph)

    def test_kruskal_skips_self_loop(self):
        graph = {
            "A": {"A": 0, "B": 1},
            "B": {"A": 1},
        }

        edges, total_weight = kruskal(graph)

        assert len(edges) == 1
        assert total_weight == 1


# ============================================================
# Prim
# ============================================================


class TestPrim:
    """Tests de l'algorithme de Prim."""

    def test_prim_returns_mst(self):
        edges, total_weight = prim(UNDIRECTED_MST_GRAPH, "A")

        assert len(edges) == len(UNDIRECTED_MST_GRAPH) - 1
        assert total_weight == 10

    def test_prim_default_start_vertex(self):
        edges, total_weight = prim(UNDIRECTED_MST_GRAPH)

        assert len(edges) == len(UNDIRECTED_MST_GRAPH) - 1
        assert total_weight == 10

    def test_prim_mst_weight_matches_kruskal(self):
        prim_edges, prim_weight = prim(UNDIRECTED_MST_GRAPH, "A")
        kruskal_edges, kruskal_weight = kruskal(UNDIRECTED_MST_GRAPH)

        assert len(prim_edges) == len(kruskal_edges)
        assert prim_weight == kruskal_weight

    def test_prim_returns_edge_tuples(self):
        edges, _ = prim(UNDIRECTED_MST_GRAPH, "A")

        for edge in edges:
            assert len(edge) == 3

    def test_prim_single_vertex(self):
        graph = {"A": {}}

        edges, total_weight = prim(graph, "A")

        assert edges == []
        assert total_weight == 0.0

    def test_prim_empty_graph_fails(self):
        with pytest.raises(ValueError):
            prim({})

    def test_prim_missing_start_fails(self):
        with pytest.raises(ValueError):
            prim(UNDIRECTED_MST_GRAPH, "Z")

    def test_prim_rejects_disconnected_graph(self):
        with pytest.raises(ValueError):
            prim(DISCONNECTED_WEIGHTED_GRAPH, "A")

    def test_prim_missing_neighbor_fails(self):
        graph = {
            "A": {"B": 1},
        }

        with pytest.raises(ValueError):
            prim(graph, "A")

    def test_prim_invalid_graph_type_fails(self):
        with pytest.raises(TypeError):
            prim([], "A")

    def test_prim_rejects_non_numeric_weight(self):
        graph = {
            "A": {"B": "invalid"},
            "B": {"A": "invalid"},
        }

        with pytest.raises(TypeError):
            prim(graph, "A")

    def test_prim_supports_negative_weight(self):
        graph = {
            "A": {"B": -1},
            "B": {"A": -1},
        }

        edges, total_weight = prim(graph, "A")

        assert len(edges) == 1
        assert total_weight == -1

    def test_prim_rejects_nan_weight(self):
        graph = {
            "A": {"B": math.nan},
            "B": {"A": math.nan},
        }

        with pytest.raises(ValueError):
            prim(graph, "A")

    def test_prim_rejects_infinite_weight(self):
        graph = {
            "A": {"B": math.inf},
            "B": {"A": math.inf},
        }

        with pytest.raises(ValueError):
            prim(graph, "A")


# ============================================================
# Tests de cohérence entre algorithmes
# ============================================================


class TestGraphAlgorithmsConsistency:
    """Tests de cohérence entre différents algorithmes."""

    def test_dijkstra_and_bellman_ford_agree_on_non_negative_graph(self):
        dijkstra_result = dijkstra(WEIGHTED_GRAPH, "A")
        bellman_result = bellman_ford(WEIGHTED_GRAPH, "A")

        assert dijkstra_result == bellman_result

    def test_dijkstra_and_floyd_warshall_agree(self):
        dijkstra_result = dijkstra(WEIGHTED_GRAPH, "A")
        floyd_result = floyd_warshall(WEIGHTED_GRAPH)

        for node in WEIGHTED_GRAPH:
            assert dijkstra_result[node] == floyd_result["A"][node]

    def test_bellman_ford_and_floyd_warshall_agree_with_negative_edges(self):
        graph = {
            "A": {"B": 4, "C": 5},
            "B": {"C": -2},
            "C": {"D": 3},
            "D": {},
        }

        bellman_result = bellman_ford(graph, "A")
        floyd_result = floyd_warshall(graph)

        for node in graph:
            assert bellman_result[node] == floyd_result["A"][node]

    def test_prim_and_kruskal_return_same_mst_weight(self):
        _, kruskal_weight = kruskal(UNDIRECTED_MST_GRAPH)
        _, prim_weight = prim(UNDIRECTED_MST_GRAPH, "A")

        assert kruskal_weight == prim_weight

    def test_bfs_and_dfs_visit_same_vertices(self):
        bfs_result = bfs(UNWEIGHTED_GRAPH, "A")
        dfs_result = dfs(UNWEIGHTED_GRAPH, "A")

        assert set(bfs_result) == set(dfs_result)
        assert len(bfs_result) == len(dfs_result)

    def test_bfs_and_dfs_start_at_same_vertex(self):
        assert bfs(UNWEIGHTED_GRAPH, "A")[0] == "A"
        assert dfs(UNWEIGHTED_GRAPH, "A")[0] == "A"


# ============================================================
# Paramétrisation des validations communes
# ============================================================


@pytest.mark.parametrize(
    "algorithm",
    [bfs, dfs],
)
def test_unweighted_algorithms_reject_invalid_start(algorithm):
    with pytest.raises(ValueError):
        algorithm(UNWEIGHTED_GRAPH, "UNKNOWN")


@pytest.mark.parametrize(
    "algorithm",
    [dijkstra, bellman_ford],
)
def test_single_source_algorithms_reject_invalid_start(algorithm):
    with pytest.raises(ValueError):
        algorithm(WEIGHTED_GRAPH, "UNKNOWN")


@pytest.mark.parametrize(
    "algorithm",
    [dijkstra, bellman_ford, floyd_warshall, kruskal, prim],
)
def test_weighted_algorithms_reject_invalid_weight_type(algorithm):
    graph = {
        "A": {"B": "not-a-number"},
        "B": {"A": "not-a-number"},
    }

    if algorithm is prim:
        with pytest.raises(TypeError):
            algorithm(graph, "A")
    elif algorithm in (dijkstra, bellman_ford):
        with pytest.raises(TypeError):
            algorithm(graph, "A")
    else:
        with pytest.raises(TypeError):
            algorithm(graph)

