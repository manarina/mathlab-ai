
"""
Tests for Algorithm Complexity Utilities
========================================

Tests unitaires pour :
- les classes de complexité ;
- la comparaison de complexités ;
- la base de complexité des algorithmes ;
- les fonctions de croissance ;
- les validations.
"""

import math

import pytest

from core.algorithms.complexity import (
    compare_complexities,
    complexity_growth,
    get_algorithm_complexity,
    get_complexity_info,
    get_space_complexity,
    get_time_complexity,
    list_algorithms,
    list_complexities,
)


# ============================================================================
# COMPLEXITY CLASSES
# ============================================================================


class TestComplexityClasses:
    """Tests for the fundamental complexity classes."""

    def test_list_complexities_returns_list(self):
        result = list_complexities()

        assert isinstance(result, list)

    def test_list_complexities_not_empty(self):
        result = list_complexities()

        assert result

    def test_expected_complexities_are_present(self):
        result = list_complexities()

        expected = [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2ⁿ)",
            "O(n!)",
        ]

        for complexity in expected:
            assert complexity in result

    def test_complexities_are_unique(self):
        result = list_complexities()

        assert len(result) == len(set(result))

    def test_complexities_are_in_expected_order(self):
        result = list_complexities()

        expected = [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2ⁿ)",
            "O(n!)",
        ]

        assert result == expected


# ============================================================================
# GET COMPLEXITY INFO
# ============================================================================


class TestGetComplexityInfo:
    """Tests for get_complexity_info."""

    @pytest.mark.parametrize(
        "complexity,name",
        [
            ("O(1)", "Constant"),
            ("O(log n)", "Logarithmic"),
            ("O(n)", "Linear"),
            ("O(n log n)", "Linearithmic"),
            ("O(n²)", "Quadratic"),
            ("O(2ⁿ)", "Exponential"),
            ("O(n!)", "Factorial"),
        ],
    )
    def test_returns_expected_name(self, complexity, name):
        result = get_complexity_info(complexity)

        assert result["name"] == name

    @pytest.mark.parametrize(
        "complexity",
        [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2ⁿ)",
            "O(n!)",
        ],
    )
    def test_returns_expected_keys(self, complexity):
        result = get_complexity_info(complexity)

        assert set(result.keys()) == {
            "name",
            "label",
            "description",
            "time_order",
            "space_order",
        }

    @pytest.mark.parametrize(
        "complexity",
        [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2ⁿ)",
            "O(n!)",
        ],
    )
    def test_label_matches_requested_complexity(self, complexity):
        result = get_complexity_info(complexity)

        assert result["label"] == complexity

    def test_description_is_non_empty(self):
        for complexity in list_complexities():
            result = get_complexity_info(complexity)

            assert isinstance(result["description"], str)
            assert result["description"]

    def test_time_order_is_integer(self):
        for complexity in list_complexities():
            result = get_complexity_info(complexity)

            assert isinstance(result["time_order"], int)

    def test_space_order_is_integer(self):
        for complexity in list_complexities():
            result = get_complexity_info(complexity)

            assert isinstance(result["space_order"], int)

    def test_orders_are_increasing(self):
        orders = [
            get_complexity_info(complexity)["time_order"]
            for complexity in list_complexities()
        ]

        assert orders == sorted(orders)


# ============================================================================
# VALIDATION OF COMPLEXITY
# ============================================================================


class TestComplexityValidation:
    """Tests for invalid complexity inputs."""

    @pytest.mark.parametrize(
        "value",
        [
            None,
            1,
            1.5,
            [],
            {},
            True,
            False,
        ],
    )
    def test_get_complexity_info_rejects_invalid_type(self, value):
        with pytest.raises(ValueError):
            get_complexity_info(value)

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "O(n^3)",
            "O(n² + n)",
            "linear",
            "constant",
            "O(log)",
            "unknown",
        ],
    )
    def test_get_complexity_info_rejects_unknown_complexity(self, value):
        with pytest.raises(ValueError):
            get_complexity_info(value)


# ============================================================================
# COMPARE COMPLEXITIES
# ============================================================================


class TestCompareComplexities:
    """Tests for complexity comparison."""

    def test_same_complexity(self):
        assert compare_complexities("O(n)", "O(n)") == 0

    def test_constant_less_than_logarithmic(self):
        assert compare_complexities("O(1)", "O(log n)") == -1

    def test_logarithmic_less_than_linear(self):
        assert compare_complexities("O(log n)", "O(n)") == -1

    def test_linear_less_than_linearithmic(self):
        assert compare_complexities("O(n)", "O(n log n)") == -1

    def test_linearithmic_less_than_quadratic(self):
        assert compare_complexities("O(n log n)", "O(n²)") == -1

    def test_quadratic_less_than_exponential(self):
        assert compare_complexities("O(n²)", "O(2ⁿ)") == -1

    def test_exponential_less_than_factorial(self):
        assert compare_complexities("O(2ⁿ)", "O(n!)") == -1

    def test_reverse_comparison(self):
        assert compare_complexities("O(n²)", "O(n)") == 1

    @pytest.mark.parametrize(
        "first,second",
        [
            ("O(1)", "O(log n)"),
            ("O(1)", "O(n)"),
            ("O(log n)", "O(n log n)"),
            ("O(n)", "O(n²)"),
            ("O(n log n)", "O(2ⁿ)"),
            ("O(n²)", "O(n!)"),
        ],
    )
    def test_transitive_order_examples(self, first, second):
        assert compare_complexities(first, second) == -1

    @pytest.mark.parametrize(
        "value",
        [
            None,
            1,
            1.5,
            [],
            {},
            True,
        ],
    )
    def test_rejects_invalid_first_argument(self, value):
        with pytest.raises(ValueError):
            compare_complexities(value, "O(n)")

    @pytest.mark.parametrize(
        "value",
        [
            None,
            1,
            1.5,
            [],
            {},
            True,
        ],
    )
    def test_rejects_invalid_second_argument(self, value):
        with pytest.raises(ValueError):
            compare_complexities("O(n)", value)

    def test_rejects_unknown_first_complexity(self):
        with pytest.raises(ValueError):
            compare_complexities("O(n³)", "O(n)")

    def test_rejects_unknown_second_complexity(self):
        with pytest.raises(ValueError):
            compare_complexities("O(n)", "O(n³)")


# ============================================================================
# ALGORITHM COMPLEXITY DATABASE
# ============================================================================


class TestAlgorithmComplexity:
    """Tests for the algorithm complexity database."""

    def test_list_algorithms_returns_list(self):
        result = list_algorithms()

        assert isinstance(result, list)

    def test_algorithm_list_is_not_empty(self):
        assert list_algorithms()

    def test_algorithms_are_unique(self):
        algorithms = list_algorithms()

        assert len(algorithms) == len(set(algorithms))

    @pytest.mark.parametrize(
        "algorithm",
        [
            "linear_search",
            "binary_search",
            "jump_search",
            "interpolation_search",
            "exponential_search",
            "bubble_sort",
            "selection_sort",
            "insertion_sort",
            "merge_sort",
            "quick_sort",
            "heap_sort",
            "counting_sort",
            "radix_sort",
            "bucket_sort",
            "bfs",
            "dfs",
            "dijkstra",
            "bellman_ford",
            "floyd_warshall",
            "kruskal",
            "prim",
            "fibonacci",
            "climbing_stairs",
            "knapsack_01",
            "coin_change",
            "longest_common_subsequence",
            "longest_increasing_subsequence",
            "matrix_chain_multiplication",
            "activity_selection",
            "fractional_knapsack",
            "greedy_coin_change",
            "huffman_coding",
            "job_sequencing",
            "n_queens",
            "solve_sudoku",
            "solve_maze",
            "subsets",
            "permutations",
            "combination_sum",
            "bst_search",
            "bst_insert",
            "bst_delete",
            "tree_traversal",
            "gcd",
            "extended_gcd",
            "generate_primes",
            "sieve_of_eratosthenes",
            "fast_power",
            "modular_power",
            "factorial",
            "naive_string_search",
            "kmp_search",
            "rabin_karp_search",
            "z_algorithm_search",
            "levenshtein_distance",
        ],
    )
    def test_expected_algorithm_is_present(self, algorithm):
        assert algorithm in list_algorithms()

    @pytest.mark.parametrize(
        "algorithm",
        [
            "linear_search",
            "binary_search",
            "merge_sort",
            "quick_sort",
            "bfs",
            "dijkstra",
            "fibonacci",
            "n_queens",
            "gcd",
            "kmp_search",
            "levenshtein_distance",
        ],
    )
    def test_algorithm_complexity_has_required_keys(self, algorithm):
        result = get_algorithm_complexity(algorithm)

        assert set(result.keys()) == {
            "best",
            "average",
            "worst",
            "space",
        }

    def test_linear_search_complexity(self):
        result = get_algorithm_complexity("linear_search")

        assert result == {
            "best": "O(1)",
            "average": "O(n)",
            "worst": "O(n)",
            "space": "O(1)",
        }

    def test_binary_search_complexity(self):
        result = get_algorithm_complexity("binary_search")

        assert result == {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(log n)",
            "space": "O(1)",
        }

    def test_merge_sort_complexity(self):
        result = get_algorithm_complexity("merge_sort")

        assert result == {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)",
            "space": "O(n)",
        }

    def test_quick_sort_complexity(self):
        result = get_algorithm_complexity("quick_sort")

        assert result == {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n²)",
            "space": "O(log n)",
        }

    def test_kmp_complexity(self):
        result = get_algorithm_complexity("kmp_search")

        assert result == {
            "best": "O(n + m)",
            "average": "O(n + m)",
            "worst": "O(n + m)",
            "space": "O(m)",
        }

    def test_levenshtein_complexity(self):
        result = get_algorithm_complexity("levenshtein_distance")

        assert result == {
            "best": "O(nm)",
            "average": "O(nm)",
            "worst": "O(nm)",
            "space": "O(min(n, m))",
        }

    @pytest.mark.parametrize(
        "value",
        [
            None,
            1,
            1.5,
            [],
            {},
            True,
        ],
    )
    def test_get_algorithm_complexity_rejects_invalid_name(self, value):
        with pytest.raises(ValueError):
            get_algorithm_complexity(value)

    def test_unknown_algorithm_raises_value_error(self):
        with pytest.raises(ValueError):
            get_algorithm_complexity("unknown_algorithm")


# ============================================================================
# TIME COMPLEXITY
# ============================================================================


class TestTimeComplexity:
    """Tests for get_time_complexity."""

    def test_returns_only_time_complexities(self):
        result = get_time_complexity("binary_search")

        assert result == {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(log n)",
        }

    @pytest.mark.parametrize(
        "algorithm",
        [
            "linear_search",
            "binary_search",
            "merge_sort",
            "quick_sort",
            "bfs",
            "fibonacci",
            "kmp_search",
            "levenshtein_distance",
        ],
    )
    def test_returns_dictionary(self, algorithm):
        result = get_time_complexity(algorithm)

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "best",
            "average",
            "worst",
        }

    def test_does_not_include_space(self):
        result = get_time_complexity("merge_sort")

        assert "space" not in result


# ============================================================================
# SPACE COMPLEXITY
# ============================================================================


class TestSpaceComplexity:
    """Tests for get_space_complexity."""

    @pytest.mark.parametrize(
        "algorithm,expected",
        [
            ("linear_search", "O(1)"),
            ("binary_search", "O(1)"),
            ("merge_sort", "O(n)"),
            ("quick_sort", "O(log n)"),
            ("bfs", "O(V)"),
            ("kmp_search", "O(m)"),
            ("levenshtein_distance", "O(min(n, m))"),
        ],
    )
    def test_returns_expected_space_complexity(
        self,
        algorithm,
        expected,
    ):
        assert get_space_complexity(algorithm) == expected

    def test_returns_string(self):
        result = get_space_complexity("binary_search")

        assert isinstance(result, str)
        assert result


# ============================================================================
# COMPLEXITY GROWTH — BASIC CASES
# ============================================================================


class TestComplexityGrowth:
    """Tests for complexity_growth."""

    def test_constant_growth(self):
        assert complexity_growth("O(1)", 1) == 1.0
        assert complexity_growth("O(1)", 10) == 1.0
        assert complexity_growth("O(1)", 1000) == 1.0

    def test_logarithmic_growth(self):
        assert complexity_growth("O(log n)", 1) == pytest.approx(0.0)
        assert complexity_growth("O(log n)", 2) == pytest.approx(1.0)
        assert complexity_growth("O(log n)", 8) == pytest.approx(3.0)

    def test_linear_growth(self):
        assert complexity_growth("O(n)", 1) == 1.0
        assert complexity_growth("O(n)", 10) == 10.0
        assert complexity_growth("O(n)", 100) == 100.0

    def test_linearithmic_growth(self):
        assert complexity_growth("O(n log n)", 1) == pytest.approx(0.0)
        assert complexity_growth("O(n log n)", 2) == pytest.approx(2.0)
        assert complexity_growth("O(n log n)", 8) == pytest.approx(24.0)

    def test_quadratic_growth(self):
        assert complexity_growth("O(n²)", 1) == 1.0
        assert complexity_growth("O(n²)", 10) == 100.0
        assert complexity_growth("O(n²)", 20) == 400.0

    def test_exponential_growth(self):
        assert complexity_growth("O(2ⁿ)", 1) == 2.0
        assert complexity_growth("O(2ⁿ)", 5) == 32.0
        assert complexity_growth("O(2ⁿ)", 10) == 1024.0

    def test_factorial_growth(self):
        assert complexity_growth("O(n!)", 1) == 1.0
        assert complexity_growth("O(n!)", 5) == 120.0
        assert complexity_growth("O(n!)", 7) == 5040.0

    @pytest.mark.parametrize(
        "complexity",
        [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2ⁿ)",
            "O(n!)",
        ],
    )
    def test_growth_is_non_negative(self, complexity):
        result = complexity_growth(complexity, 10)

        assert result >= 0.0

    @pytest.mark.parametrize(
        "complexity",
        [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2ⁿ)",
            "O(n!)",
        ],
    )
    def test_growth_returns_float(self, complexity):
        result = complexity_growth(complexity, 10)

        assert isinstance(result, float)


# ============================================================================
# COMPLEXITY GROWTH — MONOTONICITY
# ============================================================================


class TestComplexityGrowthProperties:
    """Property-style tests for growth functions."""

    @pytest.mark.parametrize(
        "complexity",
        [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2ⁿ)",
            "O(n!)",
        ],
    )
    def test_growth_is_non_decreasing(self, complexity):
        values = [
            complexity_growth(complexity, n)
            for n in [1, 2, 3, 4, 5]
        ]

        assert values == sorted(values)

    def test_linear_growth_doubles_when_n_doubles(self):
        first = complexity_growth("O(n)", 10)
        second = complexity_growth("O(n)", 20)

        assert second == pytest.approx(2 * first)

    def test_quadratic_growth_is_four_times_larger_when_n_doubles(self):
        first = complexity_growth("O(n²)", 10)
        second = complexity_growth("O(n²)", 20)

        assert second == pytest.approx(4 * first)

    def test_exponential_growth_doubles_when_n_increases_by_one(self):
        first = complexity_growth("O(2ⁿ)", 10)
        second = complexity_growth("O(2ⁿ)", 11)

        assert second == pytest.approx(2 * first)

    def test_factorial_growth_ratio(self):
        first = complexity_growth("O(n!)", 5)
        second = complexity_growth("O(n!)", 6)

        assert second == pytest.approx(6 * first)


# ============================================================================
# COMPLEXITY GROWTH VALIDATION
# ============================================================================


class TestComplexityGrowthValidation:
    """Tests for invalid complexity_growth inputs."""

    @pytest.mark.parametrize(
        "value",
        [
            None,
            1,
            1.5,
            [],
            {},
            True,
        ],
    )
    def test_invalid_complexity(self, value):
        with pytest.raises(ValueError):
            complexity_growth(value, 10)

    @pytest.mark.parametrize(
        "n",
        [
            0,
            -1,
            -10,
            1.5,
            2.5,
            None,
            [],
            {},
            True,
            False,
        ],
    )
    def test_invalid_n(self, n):
        with pytest.raises(ValueError):
            complexity_growth("O(n)", n)

    def test_unknown_complexity(self):
        with pytest.raises(ValueError):
            complexity_growth("O(n³)", 10)


# ============================================================================
# LARGE INPUT VALUES
# ============================================================================


class TestComplexityGrowthLargeInputs:
    """Tests for reasonably large input sizes."""

    def test_large_linear_value(self):
        result = complexity_growth("O(n)", 100_000)

        assert result == 100_000.0

    def test_large_logarithmic_value(self):
        result = complexity_growth("O(log n)", 1024)

        assert result == pytest.approx(10.0)

    def test_large_quadratic_value(self):
        result = complexity_growth("O(n²)", 1_000)

        assert result == 1_000_000.0

    def test_large_exponential_value(self):
        result = complexity_growth("O(2ⁿ)", 20)

        assert result == 1_048_576.0


# ============================================================================
# CONSISTENCY OF ALGORITHM DATABASE
# ============================================================================


class TestAlgorithmDatabaseConsistency:
    """Cross-check the structure of all algorithm entries."""

    def test_every_algorithm_has_all_complexity_fields(self):
        for algorithm in list_algorithms():
            result = get_algorithm_complexity(algorithm)

            assert set(result.keys()) == {
                "best",
                "average",
                "worst",
                "space",
            }

    def test_every_algorithm_has_string_complexities(self):
        for algorithm in list_algorithms():
            result = get_algorithm_complexity(algorithm)

            for key in ["best", "average", "worst", "space"]:
                assert isinstance(result[key], str)
                assert result[key]

    def test_time_complexity_matches_algorithm_database(self):
        for algorithm in list_algorithms():
            full = get_algorithm_complexity(algorithm)
            time = get_time_complexity(algorithm)

            assert time["best"] == full["best"]
            assert time["average"] == full["average"]
            assert time["worst"] == full["worst"]

    def test_space_complexity_matches_algorithm_database(self):
        for algorithm in list_algorithms():
            full = get_algorithm_complexity(algorithm)

            assert get_space_complexity(algorithm) == full["space"]

    def test_list_algorithms_contains_no_empty_names(self):
        for algorithm in list_algorithms():
            assert algorithm
            assert isinstance(algorithm, str)


# ============================================================================
# SPECIFIC ALGORITHM FAMILIES
# ============================================================================


class TestSearchingComplexities:
    """Consistency checks for searching algorithms."""

    def test_linear_search_is_linear_in_average_and_worst(self):
        result = get_algorithm_complexity("linear_search")

        assert result["average"] == "O(n)"
        assert result["worst"] == "O(n)"

    def test_binary_search_is_logarithmic(self):
        result = get_algorithm_complexity("binary_search")

        assert result["average"] == "O(log n)"
        assert result["worst"] == "O(log n)"

    def test_interpolation_search_has_linear_worst_case(self):
        result = get_algorithm_complexity("interpolation_search")

        assert result["worst"] == "O(n)"


class TestSortingComplexities:
    """Consistency checks for sorting algorithms."""

    def test_bubble_sort_quadratic_average(self):
        result = get_algorithm_complexity("bubble_sort")

        assert result["average"] == "O(n²)"

    def test_selection_sort_quadratic(self):
        result = get_algorithm_complexity("selection_sort")

        assert result["best"] == "O(n²)"
        assert result["average"] == "O(n²)"
        assert result["worst"] == "O(n²)"

    def test_insertion_sort_linear_best_case(self):
        result = get_algorithm_complexity("insertion_sort")

        assert result["best"] == "O(n)"

    def test_merge_sort_n_log_n(self):
        result = get_algorithm_complexity("merge_sort")

        assert result["worst"] == "O(n log n)"

    def test_quick_sort_has_quadratic_worst_case(self):
        result = get_algorithm_complexity("quick_sort")

        assert result["worst"] == "O(n²)"

    def test_heap_sort_has_n_log_n_worst_case(self):
        result = get_algorithm_complexity("heap_sort")

        assert result["worst"] == "O(n log n)"


class TestGraphComplexities:
    """Consistency checks for graph algorithms."""

    def test_bfs_complexity(self):
        result = get_algorithm_complexity("bfs")

        assert result["worst"] == "O(V + E)"

    def test_dfs_complexity(self):
        result = get_algorithm_complexity("dfs")

        assert result["worst"] == "O(V + E)"

    def test_dijkstra_complexity(self):
        result = get_algorithm_complexity("dijkstra")

        assert result["worst"] == "O((V + E) log V)"

    def test_bellman_ford_complexity(self):
        result = get_algorithm_complexity("bellman_ford")

        assert result["worst"] == "O(VE)"

    def test_floyd_warshall_complexity(self):
        result = get_algorithm_complexity("floyd_warshall")

        assert result["worst"] == "O(V³)"


class TestStringComplexities:
    """Consistency checks for string algorithms."""

    def test_naive_search_complexity(self):
        result = get_algorithm_complexity("naive_string_search")

        assert result["average"] == "O(nm)"
        assert result["worst"] == "O(nm)"

    def test_kmp_complexity(self):
        result = get_algorithm_complexity("kmp_search")

        assert result["worst"] == "O(n + m)"

    def test_rabin_karp_worst_case(self):
        result = get_algorithm_complexity("rabin_karp_search")

        assert result["worst"] == "O(nm)"

    def test_z_algorithm_complexity(self):
        result = get_algorithm_complexity("z_algorithm_search")

        assert result["worst"] == "O(n + m)"

    def test_levenshtein_complexity(self):
        result = get_algorithm_complexity("levenshtein_distance")

        assert result["worst"] == "O(nm)"
        assert result["space"] == "O(min(n, m))"


# ============================================================================
# MATHEMATICAL COMPLEXITIES
# ============================================================================


class TestMathematicalComplexities:
    """Consistency checks for mathematical algorithms."""

    def test_gcd_complexity(self):
        result = get_algorithm_complexity("gcd")

        assert result["average"] == "O(log n)"
        assert result["worst"] == "O(log n)"

    def test_fast_power_complexity(self):
        result = get_algorithm_complexity("fast_power")

        assert result["worst"] == "O(log n)"

    def test_modular_power_complexity(self):
        result = get_algorithm_complexity("modular_power")

        assert result["worst"] == "O(log n)"

    def test_factorial_complexity(self):
        result = get_algorithm_complexity("factorial")

        assert result["worst"] == "O(n)"


# ============================================================================
# SPECIAL NUMERICAL CHECKS
# ============================================================================


class TestNumericalProperties:
    """Additional numerical consistency checks."""

    def test_logarithmic_growth_matches_math_log2(self):
        for n in [1, 2, 4, 8, 16, 32]:
            assert complexity_growth(
                "O(log n)",
                n,
            ) == pytest.approx(math.log2(n))

    def test_linearithmic_growth_matches_definition(self):
        for n in [1, 2, 4, 8, 16]:
            expected = n * math.log2(n)

            assert complexity_growth(
                "O(n log n)",
                n,
            ) == pytest.approx(expected)

    def test_constant_growth_is_independent_of_n(self):
        values = [
            complexity_growth("O(1)", n)
            for n in [1, 10, 100, 1_000]
        ]

        assert values == [1.0, 1.0, 1.0, 1.0]

