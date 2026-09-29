
"""
Algorithm Complexity Utilities
==============================

Utilities for describing and comparing algorithmic time and space
complexities.

Supported complexity classes
----------------------------
- O(1)
- O(log n)
- O(n)
- O(n log n)
- O(n²)
- O(2ⁿ)
- O(n!)

The module is intentionally lightweight so it can be reused by
the Streamlit Algorithms Lab.
"""

from __future__ import annotations

from math import factorial


# ============================================================================
# COMPLEXITY DEFINITIONS
# ============================================================================


_COMPLEXITIES = {
    "O(1)": {
        "name": "Constant",
        "label": "O(1)",
        "description": (
            "The execution time does not depend on the input size."
        ),
        "time_order": 0,
        "space_order": 0,
    },
    "O(log n)": {
        "name": "Logarithmic",
        "label": "O(log n)",
        "description": (
            "The execution time grows logarithmically with the input size."
        ),
        "time_order": 1,
        "space_order": 1,
    },
    "O(n)": {
        "name": "Linear",
        "label": "O(n)",
        "description": (
            "The execution time grows proportionally to the input size."
        ),
        "time_order": 2,
        "space_order": 2,
    },
    "O(n log n)": {
        "name": "Linearithmic",
        "label": "O(n log n)",
        "description": (
            "The execution time grows slightly faster than linear."
        ),
        "time_order": 3,
        "space_order": 3,
    },
    "O(n²)": {
        "name": "Quadratic",
        "label": "O(n²)",
        "description": (
            "The execution time grows proportionally to the square "
            "of the input size."
        ),
        "time_order": 4,
        "space_order": 4,
    },
    "O(2ⁿ)": {
        "name": "Exponential",
        "label": "O(2ⁿ)",
        "description": (
            "The execution time doubles approximately for each additional "
            "input element."
        ),
        "time_order": 5,
        "space_order": 5,
    },
    "O(n!)": {
        "name": "Factorial",
        "label": "O(n!)",
        "description": (
            "The execution time grows factorially and becomes impractical "
            "very quickly."
        ),
        "time_order": 6,
        "space_order": 6,
    },
}


# ============================================================================
# VALIDATION
# ============================================================================


def _validate_complexity(complexity: str) -> None:
    """Validate a complexity notation."""
    if not isinstance(complexity, str):
        raise ValueError("complexity must be a string.")

    if complexity not in _COMPLEXITIES:
        raise ValueError(
            f"Unknown complexity: {complexity!r}. "
            f"Supported values are: {', '.join(_COMPLEXITIES)}."
        )


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer.")

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


# ============================================================================
# COMPLEXITY INFORMATION
# ============================================================================


def get_complexity_info(complexity: str) -> dict[str, object]:
    """
    Return information about a complexity class.

    Parameters
    ----------
    complexity:
        Complexity notation such as ``"O(n)"``.

    Returns
    -------
    dict
        Complexity metadata.

    Examples
    --------
    >>> get_complexity_info("O(n)")["name"]
    'Linear'
    """
    _validate_complexity(complexity)

    return dict(_COMPLEXITIES[complexity])


def list_complexities() -> list[str]:
    """
    Return all supported complexity classes in increasing order.

    Returns
    -------
    list[str]
        Supported complexity notations.

    Examples
    --------
    >>> list_complexities()[:3]
    ['O(1)', 'O(log n)', 'O(n)']
    """
    return list(_COMPLEXITIES.keys())


# ============================================================================
# COMPLEXITY COMPARISON
# ============================================================================


def compare_complexities(
    first: str,
    second: str,
) -> int:
    """
    Compare two asymptotic complexity classes.

    Parameters
    ----------
    first:
        First complexity.
    second:
        Second complexity.

    Returns
    -------
    int
        -1 if first is asymptotically smaller,
         0 if equivalent,
         1 if first is asymptotically larger.

    Examples
    --------
    >>> compare_complexities("O(n)", "O(n²)")
    -1

    >>> compare_complexities("O(n²)", "O(n)")
    1

    >>> compare_complexities("O(n)", "O(n)")
    0
    """
    _validate_complexity(first)
    _validate_complexity(second)

    first_order = _COMPLEXITIES[first]["time_order"]
    second_order = _COMPLEXITIES[second]["time_order"]

    if first_order < second_order:
        return -1

    if first_order > second_order:
        return 1

    return 0


# ============================================================================
# ALGORITHM COMPLEXITY DATABASE
# ============================================================================


_ALGORITHM_COMPLEXITIES = {
    # Searching
    "linear_search": {
        "best": "O(1)",
        "average": "O(n)",
        "worst": "O(n)",
        "space": "O(1)",
    },
    "binary_search": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
    },
    "jump_search": {
        "best": "O(1)",
        "average": "O(√n)",
        "worst": "O(√n)",
        "space": "O(1)",
    },
    "interpolation_search": {
        "best": "O(1)",
        "average": "O(log log n)",
        "worst": "O(n)",
        "space": "O(1)",
    },
    "exponential_search": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
    },

    # Sorting
    "bubble_sort": {
        "best": "O(n)",
        "average": "O(n²)",
        "worst": "O(n²)",
        "space": "O(1)",
    },
    "selection_sort": {
        "best": "O(n²)",
        "average": "O(n²)",
        "worst": "O(n²)",
        "space": "O(1)",
    },
    "insertion_sort": {
        "best": "O(n)",
        "average": "O(n²)",
        "worst": "O(n²)",
        "space": "O(1)",
    },
    "merge_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
    },
    "quick_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n²)",
        "space": "O(log n)",
    },
    "heap_sort": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(1)",
    },
    "counting_sort": {
        "best": "O(n + k)",
        "average": "O(n + k)",
        "worst": "O(n + k)",
        "space": "O(n + k)",
    },
    "radix_sort": {
        "best": "O(d(n + k))",
        "average": "O(d(n + k))",
        "worst": "O(d(n + k))",
        "space": "O(n + k)",
    },
    "bucket_sort": {
        "best": "O(n + k)",
        "average": "O(n + k)",
        "worst": "O(n²)",
        "space": "O(n + k)",
    },

    # Graphs
    "bfs": {
        "best": "O(V + E)",
        "average": "O(V + E)",
        "worst": "O(V + E)",
        "space": "O(V)",
    },
    "dfs": {
        "best": "O(V + E)",
        "average": "O(V + E)",
        "worst": "O(V + E)",
        "space": "O(V)",
    },
    "dijkstra": {
        "best": "O((V + E) log V)",
        "average": "O((V + E) log V)",
        "worst": "O((V + E) log V)",
        "space": "O(V)",
    },
    "bellman_ford": {
        "best": "O(VE)",
        "average": "O(VE)",
        "worst": "O(VE)",
        "space": "O(V)",
    },
    "floyd_warshall": {
        "best": "O(V³)",
        "average": "O(V³)",
        "worst": "O(V³)",
        "space": "O(V²)",
    },
    "kruskal": {
        "best": "O(E log E)",
        "average": "O(E log E)",
        "worst": "O(E log E)",
        "space": "O(V + E)",
    },
    "prim": {
        "best": "O(E log V)",
        "average": "O(E log V)",
        "worst": "O(E log V)",
        "space": "O(V + E)",
    },

    # Dynamic Programming
    "fibonacci": {
        "best": "O(n)",
        "average": "O(n)",
        "worst": "O(n)",
        "space": "O(1)",
    },
    "climbing_stairs": {
        "best": "O(n)",
        "average": "O(n)",
        "worst": "O(n)",
        "space": "O(1)",
    },
    "knapsack_01": {
        "best": "O(nC)",
        "average": "O(nC)",
        "worst": "O(nC)",
        "space": "O(C)",
    },
    "coin_change": {
        "best": "O(nA)",
        "average": "O(nA)",
        "worst": "O(nA)",
        "space": "O(A)",
    },
    "longest_common_subsequence": {
        "best": "O(nm)",
        "average": "O(nm)",
        "worst": "O(nm)",
        "space": "O(nm)",
    },
    "longest_increasing_subsequence": {
        "best": "O(n²)",
        "average": "O(n²)",
        "worst": "O(n²)",
        "space": "O(n)",
    },
    "matrix_chain_multiplication": {
        "best": "O(n³)",
        "average": "O(n³)",
        "worst": "O(n³)",
        "space": "O(n²)",
    },

    # Greedy
    "activity_selection": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
    },
    "fractional_knapsack": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
    },
    "greedy_coin_change": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
    },
    "huffman_coding": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
    },
    "job_sequencing": {
        "best": "O(n log n)",
        "average": "O(n log n)",
        "worst": "O(n log n)",
        "space": "O(n)",
    },

    # Backtracking
    "n_queens": {
        "best": "O(n!)",
        "average": "O(n!)",
        "worst": "O(n!)",
        "space": "O(n)",
    },
    "solve_sudoku": {
        "best": "O(1)",
        "average": "O(9^n)",
        "worst": "O(9^n)",
        "space": "O(n)",
    },
    "solve_maze": {
        "best": "O(V + E)",
        "average": "O(V + E)",
        "worst": "O(V + E)",
        "space": "O(V)",
    },
    "subsets": {
        "best": "O(2^n)",
        "average": "O(2^n)",
        "worst": "O(2^n)",
        "space": "O(2^n)",
    },
    "permutations": {
        "best": "O(n!)",
        "average": "O(n!)",
        "worst": "O(n!)",
        "space": "O(n!)",
    },
    "combination_sum": {
        "best": "O(2^n)",
        "average": "O(2^n)",
        "worst": "O(2^n)",
        "space": "O(2^n)",
    },

    # Trees
    "bst_search": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(n)",
        "space": "O(1)",
    },
    "bst_insert": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(n)",
        "space": "O(1)",
    },
    "bst_delete": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(n)",
        "space": "O(1)",
    },
    "tree_traversal": {
        "best": "O(n)",
        "average": "O(n)",
        "worst": "O(n)",
        "space": "O(n)",
    },

    # Mathematical
    "gcd": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
    },
    "extended_gcd": {
        "best": "O(1)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(log n)",
    },
    "generate_primes": {
        "best": "O(n log log n)",
        "average": "O(n log log n)",
        "worst": "O(n log log n)",
        "space": "O(n)",
    },
    "sieve_of_eratosthenes": {
        "best": "O(n log log n)",
        "average": "O(n log log n)",
        "worst": "O(n log log n)",
        "space": "O(n)",
    },
    "fast_power": {
        "best": "O(log n)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
    },
    "modular_power": {
        "best": "O(log n)",
        "average": "O(log n)",
        "worst": "O(log n)",
        "space": "O(1)",
    },
    "factorial": {
        "best": "O(n)",
        "average": "O(n)",
        "worst": "O(n)",
        "space": "O(1)",
    },

    # Strings
    "naive_string_search": {
        "best": "O(n)",
        "average": "O(nm)",
        "worst": "O(nm)",
        "space": "O(1)",
    },
    "kmp_search": {
        "best": "O(n + m)",
        "average": "O(n + m)",
        "worst": "O(n + m)",
        "space": "O(m)",
    },
    "rabin_karp_search": {
        "best": "O(n + m)",
        "average": "O(n + m)",
        "worst": "O(nm)",
        "space": "O(1)",
    },
    "z_algorithm_search": {
        "best": "O(n + m)",
        "average": "O(n + m)",
        "worst": "O(n + m)",
        "space": "O(n + m)",
    },
    "levenshtein_distance": {
        "best": "O(nm)",
        "average": "O(nm)",
        "worst": "O(nm)",
        "space": "O(min(n, m))",
    },
}


# ============================================================================
# ALGORITHM INFORMATION
# ============================================================================


def get_algorithm_complexity(name: str) -> dict[str, str]:
    """
    Return complexity information for an algorithm.

    Parameters
    ----------
    name:
        Algorithm identifier.

    Returns
    -------
    dict[str, str]
        Best, average, worst-case and space complexity.

    Examples
    --------
    >>> get_algorithm_complexity("binary_search")["worst"]
    'O(log n)'
    """
    if not isinstance(name, str):
        raise ValueError("name must be a string.")

    if name not in _ALGORITHM_COMPLEXITIES:
        raise ValueError(f"Unknown algorithm: {name!r}.")

    return dict(_ALGORITHM_COMPLEXITIES[name])


def list_algorithms() -> list[str]:
    """
    Return the names of all algorithms in the complexity database.
    """
    return list(_ALGORITHM_COMPLEXITIES.keys())


def get_time_complexity(name: str) -> dict[str, str]:
    """
    Return best, average and worst-case time complexity.
    """
    info = get_algorithm_complexity(name)

    return {
        "best": info["best"],
        "average": info["average"],
        "worst": info["worst"],
    }


def get_space_complexity(name: str) -> str:
    """
    Return the auxiliary space complexity of an algorithm.
    """
    return get_algorithm_complexity(name)["space"]


# ============================================================================
# GROWTH FUNCTIONS
# ============================================================================


def complexity_growth(complexity: str, n: int) -> float:
    """
    Compute a representative growth value for a complexity class.

    This function is intended for educational visualization.

    Parameters
    ----------
    complexity:
        Complexity notation.
    n:
        Positive input size.

    Returns
    -------
    float
        Representative growth value.

    Notes
    -----
    This is not an exact operation-count model. It only provides
    a normalized mathematical growth function.
    """
    _validate_complexity(complexity)
    _validate_positive_integer(n, "n")

    if complexity == "O(1)":
        return 1.0

    if complexity == "O(log n)":
        return max(0.0, __import__("math").log2(n))

    if complexity == "O(n)":
        return float(n)

    if complexity == "O(n log n)":
        return float(n) * __import__("math").log2(n)

    if complexity == "O(n²)":
        return float(n**2)

    if complexity == "O(2ⁿ)":
        return float(2**n)

    if complexity == "O(n!)":
        return float(factorial(n))

    raise ValueError(f"Unsupported complexity: {complexity}")


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "get_complexity_info",
    "list_complexities",
    "compare_complexities",
    "get_algorithm_complexity",
    "list_algorithms",
    "get_time_complexity",
    "get_space_complexity",
    "complexity_growth",
]

